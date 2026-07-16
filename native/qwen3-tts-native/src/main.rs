mod cuda;

use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde_json::{Value, json};

const EXPECTED_FILE_BYTES: u64 = 3_833_402_552;
const EXPECTED_TENSOR_COUNT: usize = 404;
const EXPECTED_PARAMETER_COUNT: u64 = 1_916_676_352;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut arguments = env::args_os();
    let program = arguments
        .next()
        .unwrap_or_else(|| OsString::from("qwen3-tts-native"));
    let command = arguments.next().with_context(|| usage(&program))?;
    match command.to_str() {
        Some("probe-cuda") => return cuda::run_probe(arguments),
        Some("benchmark-argmax") => return cuda::run_argmax_benchmark(arguments),
        Some("benchmark-gemv") => return cuda::run_gemv_benchmark(arguments),
        _ => {}
    }

    let model_path = arguments.next().with_context(|| usage(&program))?;

    let mut output_path = None;
    let mut include_all_tensors = false;
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--all-tensors") => include_all_tensors = true,
            Some("--output") => {
                output_path = Some(PathBuf::from(
                    arguments
                        .next()
                        .context("--output requires a destination path")?,
                ));
            }
            _ => bail!("unknown argument {:?}\n{}", argument, usage(&program)),
        }
    }

    let model_path = PathBuf::from(model_path);
    let report = inspect_checkpoint(&model_path, include_all_tensors)?;

    match command.to_str() {
        Some("inspect") => {}
        Some("validate-voice-design") => validate_voice_design(&report)?,
        _ => bail!("unknown command {:?}\n{}", command, usage(&program)),
    }

    let encoded = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output_path {
        fs::write(&path, encoded)
            .with_context(|| format!("failed to write report to {}", path.display()))?;
    } else {
        println!("{encoded}");
    }

    Ok(())
}

fn usage(program: &OsString) -> String {
    format!(
        "usage: {} <inspect|validate-voice-design> <model.safetensors> [options]\n       {} <probe-cuda|benchmark-argmax|benchmark-gemv> <libqwen3_tts_cuda.so> [options]",
        Path::new(program).display(),
        Path::new(program).display()
    )
}

fn inspect_checkpoint(path: &Path, include_all_tensors: bool) -> Result<Value> {
    let file = File::open(path)
        .with_context(|| format!("failed to open checkpoint {}", path.display()))?;
    let file_bytes = file
        .metadata()
        .with_context(|| format!("failed to stat checkpoint {}", path.display()))?
        .len();

    // SAFETY: The mapping is read-only and this process never mutates or
    // truncates the immutable checkpoint backing it.
    let mapping = unsafe { Mmap::map(&file) }
        .with_context(|| format!("failed to memory-map checkpoint {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&mapping[..])
        .with_context(|| format!("invalid Safetensors checkpoint {}", path.display()))?;

    let mut dtype_counts = BTreeMap::<String, u64>::new();
    let mut component_bytes = BTreeMap::<String, u64>::new();
    let mut component_parameters = BTreeMap::<String, u64>::new();
    let mut total_parameters = 0_u64;
    let mut payload_bytes = 0_u64;
    let mut inventory = Vec::<Value>::new();
    let mut largest = Vec::<(u64, Value)>::new();

    for name in tensors.names() {
        let tensor = tensors
            .tensor(name)
            .with_context(|| format!("failed to read tensor metadata for {name}"))?;
        let parameters = tensor
            .shape()
            .iter()
            .try_fold(1_u64, |product, dimension| {
                product
                    .checked_mul(*dimension as u64)
                    .with_context(|| format!("shape overflow while counting {name}"))
            })?;
        let bytes = tensor.data().len() as u64;
        let dtype = format!("{:?}", tensor.dtype());
        let component = component_for(name);

        total_parameters = total_parameters
            .checked_add(parameters)
            .context("total parameter count overflowed")?;
        payload_bytes = payload_bytes
            .checked_add(bytes)
            .context("total payload byte count overflowed")?;
        *dtype_counts.entry(dtype.clone()).or_default() += 1;
        *component_bytes.entry(component.clone()).or_default() += bytes;
        *component_parameters.entry(component).or_default() += parameters;

        let entry = json!({
            "name": name,
            "dtype": dtype,
            "shape": tensor.shape(),
            "parameters": parameters,
            "bytes": bytes,
        });
        if include_all_tensors {
            inventory.push(entry.clone());
        }
        largest.push((bytes, entry));
    }

    largest.sort_unstable_by_key(|tensor| std::cmp::Reverse(tensor.0));
    let largest = largest
        .into_iter()
        .take(20)
        .map(|(_, tensor)| tensor)
        .collect::<Vec<_>>();

    Ok(json!({
        "schema_version": 1,
        "checkpoint": path.display().to_string(),
        "file_bytes": file_bytes,
        "payload_bytes": payload_bytes,
        "header_bytes": file_bytes.checked_sub(payload_bytes),
        "tensor_count": tensors.len(),
        "parameter_count": total_parameters,
        "dtype_counts": dtype_counts,
        "component_bytes": component_bytes,
        "component_parameters": component_parameters,
        "largest_tensors": largest,
        "tensors": inventory,
    }))
}

fn component_for(name: &str) -> String {
    if name.contains("code_predictor") {
        return "code_predictor".to_owned();
    }
    if name.contains("talker") {
        return "talker".to_owned();
    }

    name.split('.').take(2).collect::<Vec<_>>().join(".")
}

fn validate_voice_design(report: &Value) -> Result<()> {
    expect_u64(report, "file_bytes", EXPECTED_FILE_BYTES)?;
    expect_u64(report, "tensor_count", EXPECTED_TENSOR_COUNT as u64)?;
    expect_u64(report, "parameter_count", EXPECTED_PARAMETER_COUNT)?;

    let dtype_counts = report["dtype_counts"]
        .as_object()
        .context("dtype_counts is not a JSON object")?;
    if dtype_counts.len() != 1 || dtype_counts.get("BF16").and_then(Value::as_u64) != Some(404) {
        bail!("expected exactly 404 BF16 tensors, found {dtype_counts:?}");
    }

    Ok(())
}

fn expect_u64(report: &Value, field: &str, expected: u64) -> Result<()> {
    let actual = report[field]
        .as_u64()
        .with_context(|| format!("{field} is not an unsigned integer"))?;
    if actual != expected {
        bail!("{field}: expected {expected}, found {actual}");
    }
    Ok(())
}
