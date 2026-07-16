use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use qwen3_tts_native::device_weights::DeviceWeightBuffer;
use qwen3_tts_native::loader::{NativeArtifact, TensorRef, VerificationMode};
use serde_json::json;

const READBACK_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Scope {
    All,
    Voice,
    Decoder,
}

impl Scope {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "all" => Ok(Self::All),
            "voice" => Ok(Self::Voice),
            "decoder" => Ok(Self::Decoder),
            _ => bail!("scope must be all, voice, or decoder"),
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Voice => "voice",
            Self::Decoder => "decoder",
        }
    }

    const fn includes_voice(self) -> bool {
        matches!(self, Self::All | Self::Voice)
    }

    const fn includes_decoder(self) -> bool {
        matches!(self, Self::All | Self::Decoder)
    }
}

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
        .unwrap_or_else(|| OsString::from("upload_artifact"));
    let artifact_path = PathBuf::from(arguments.next().with_context(|| usage(&program))?);
    let library_path = PathBuf::from(arguments.next().with_context(|| usage(&program))?);
    let mut device = 0_i32;
    let mut staging_mib = 8_usize;
    let mut scope = Scope::All;
    let mut report_path = None;

    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--device") => {
                device = parse_next::<i32>(&mut arguments, "--device")?;
                if device < 0 {
                    bail!("--device must be non-negative");
                }
            }
            Some("--staging-mib") => {
                staging_mib = parse_next::<usize>(&mut arguments, "--staging-mib")?;
                if !(1..=256).contains(&staging_mib) {
                    bail!("--staging-mib must be between 1 and 256");
                }
            }
            Some("--scope") => {
                let value = arguments.next().context("--scope requires a value")?;
                scope = Scope::parse(value.to_str().context("--scope must be UTF-8")?)?;
            }
            Some("--report") => {
                report_path = Some(PathBuf::from(
                    arguments.next().context("--report requires a path")?,
                ));
            }
            _ => bail!("unknown argument {argument:?}\n{}", usage(&program)),
        }
    }

    let started = Instant::now();
    let artifact = NativeArtifact::open(&artifact_path, VerificationMode::ContractsOnly)?;
    let open_milliseconds = started.elapsed().as_secs_f64() * 1_000.0;
    let artifact_root = artifact.root().display().to_string();
    let revision = artifact.revision().to_owned();
    let model_memory = artifact.model_memory_metrics();
    let staging_bytes = staging_mib
        .checked_mul(1024 * 1024)
        .context("staging bytes overflow usize")?;
    let names = selected_names(&artifact, scope);
    let allocation_bytes = selected_bytes(&artifact, &names)?;
    let mut buffer = DeviceWeightBuffer::create(
        &library_path,
        device,
        allocation_bytes,
        staging_bytes as u64,
    )?;
    let allocation_metrics = buffer.allocation_metrics();
    let _device_pointer = buffer.device_pointer();

    let upload_started = Instant::now();
    let mut offset = 0_u64;
    let mut first_probe = None;
    let mut final_probe = None;
    for (is_decoder, name) in &names {
        let tensor = if *is_decoder {
            artifact.decoder_tensor(name)?
        } else {
            artifact.voice_tensor(name)?
        };
        let tensor_bytes = u64::try_from(tensor.bytes.len()).context("tensor bytes exceed u64")?;
        let sample_bytes = tensor.bytes.len().min(READBACK_BYTES);
        if first_probe.is_none() {
            first_probe = Some((offset, tensor.bytes[..sample_bytes].to_vec()));
        }
        final_probe = Some((
            offset + tensor_bytes - sample_bytes as u64,
            tensor.bytes[tensor.bytes.len() - sample_bytes..].to_vec(),
        ));
        upload_tensor(&mut buffer, offset, tensor, staging_bytes)?;
        offset = offset
            .checked_add(tensor_bytes)
            .context("device arena offset overflowed")?;
    }
    if offset != allocation_bytes {
        bail!("uploaded byte total {offset} differs from planned {allocation_bytes}");
    }
    let metrics = buffer.finish()?;
    let finished_state = buffer.state_info();
    if !finished_state.finished {
        bail!("device buffer did not enter the finished state");
    }
    let upload_wall_milliseconds = upload_started.elapsed().as_secs_f64() * 1_000.0;

    let (first_offset, first_expected) = first_probe.context("no first readback probe")?;
    let (final_offset, final_expected) = final_probe.context("no final readback probe")?;
    // The upload ABI synchronizes each staged copy. Release every source
    // SafeTensors mapping before probing the independent device allocation.
    drop(artifact);
    let first_actual = buffer.readback(first_offset, first_expected.len())?;
    let final_actual = buffer.readback(final_offset, final_expected.len())?;
    if first_actual != first_expected || final_actual != final_expected {
        bail!("device readback did not match the first and final uploaded bytes");
    }

    let gib_per_second = if metrics.upload_microseconds == 0.0 {
        0.0
    } else {
        metrics.uploaded_bytes as f64
            / (1024.0 * 1024.0 * 1024.0)
            / (metrics.upload_microseconds as f64 / 1_000_000.0)
    };
    let report = json!({
        "schema_version": 1,
        "operation": "upload-native-qwen3-tts-artifact",
        "artifact": artifact_root,
        "revision": revision,
        "scope": scope.label(),
        "device_index": metrics.device_index,
        "tensor_count": names.len(),
        "artifact_open_milliseconds": open_milliseconds,
        "allocation_bytes": metrics.allocation_bytes,
        "pinned_staging_bytes": metrics.pinned_staging_bytes,
        "uploaded_bytes": metrics.uploaded_bytes,
        "upload_calls": metrics.upload_calls,
        "allocation_microseconds": metrics.allocation_microseconds,
        "native_upload_microseconds": metrics.upload_microseconds,
        "upload_wall_milliseconds": upload_wall_milliseconds,
        "effective_gib_per_second": gib_per_second,
        "free_before_bytes": metrics.free_before_bytes,
        "free_after_allocation_bytes": metrics.free_after_allocation_bytes,
        "reported_free_memory_delta_bytes": metrics
            .free_before_bytes
            .saturating_sub(metrics.free_after_allocation_bytes),
        "device_pointer_exposed": true,
        "source_mappings_released_before_readback": true,
        "readback_probe_bytes": first_expected.len() + final_expected.len(),
        "readback_exact": true,
        "memory_bytes": {
            "host_mapped_weight_files_before_release": model_memory.total_mapped_file_bytes,
            "host_tensor_payload_views_before_release": model_memory.total_tensor_payload_bytes,
            "host_committed_weight_copy": model_memory.host_committed_weight_copy_bytes,
            "runtime_dtype_conversion": model_memory.runtime_dtype_conversion_bytes,
            "pinned_staging": metrics.pinned_staging_bytes,
            "device_allocation": metrics.allocation_bytes,
        },
        "device_state": {
            "finished": finished_state.finished,
            "allocation_bytes": finished_state.allocation.allocation_bytes,
            "uploaded_bytes": finished_state.allocation.uploaded_bytes,
            "upload_calls": finished_state.allocation.upload_calls,
        },
        "allocation_initial_metrics": {
            "allocation_bytes": allocation_metrics.allocation_bytes,
            "pinned_staging_bytes": allocation_metrics.pinned_staging_bytes,
        },
    });
    let encoded = serde_json::to_string_pretty(&report)?;
    if let Some(path) = report_path {
        fs::write(&path, format!("{encoded}\n"))
            .with_context(|| format!("failed to write {}", path.display()))?;
    } else {
        println!("{encoded}");
    }
    Ok(())
}

fn selected_names(artifact: &NativeArtifact, scope: Scope) -> Vec<(bool, String)> {
    let mut names = Vec::new();
    if scope.includes_voice() {
        names.extend(
            artifact
                .voice_tensor_names()
                .map(|name| (false, name.to_owned())),
        );
    }
    if scope.includes_decoder() {
        names.extend(
            artifact
                .decoder_tensor_names()
                .map(|name| (true, name.to_owned())),
        );
    }
    names
}

fn selected_bytes(artifact: &NativeArtifact, names: &[(bool, String)]) -> Result<u64> {
    names.iter().try_fold(0_u64, |total, (is_decoder, name)| {
        let tensor = if *is_decoder {
            artifact.decoder_tensor(name)?
        } else {
            artifact.voice_tensor(name)?
        };
        total
            .checked_add(tensor.bytes.len() as u64)
            .context("selected tensor byte total overflowed")
    })
}

fn upload_tensor(
    buffer: &mut DeviceWeightBuffer,
    arena_offset: u64,
    tensor: TensorRef<'_>,
    staging_bytes: usize,
) -> Result<()> {
    for chunk in tensor.chunks(staging_bytes)? {
        let chunk_offset = u64::try_from(chunk.offset_bytes).context("chunk offset exceeds u64")?;
        buffer.upload(
            arena_offset
                .checked_add(chunk_offset)
                .context("device upload offset overflowed")?,
            chunk.bytes,
        )?;
    }
    Ok(())
}

fn parse_next<T>(arguments: &mut impl Iterator<Item = OsString>, option: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    arguments
        .next()
        .with_context(|| format!("{option} requires a value"))?
        .to_str()
        .with_context(|| format!("{option} must be UTF-8"))?
        .parse()
        .with_context(|| format!("{option} has an invalid value"))
}

fn usage(program: &OsString) -> String {
    format!(
        "usage: {} <artifact-directory> <libqwen3_tts_cuda.so> [--scope all|voice|decoder] [--device N] [--staging-mib 1..256] [--report path]",
        Path::new(program).display()
    )
}
