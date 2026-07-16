use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use qwen3_tts_native::loader::{NativeArtifact, VerificationMode};
use qwen3_tts_native::sha256::to_hex;
use serde_json::json;

const TALKER_PROBE: &str = "talker.model.layers.0.self_attn.q_proj.weight";
const PREDICTOR_PROBE: &str = "talker.code_predictor.model.layers.0.self_attn.q_proj.weight";
const DECODER_PROBE: &str = "decoder.pre_transformer.layers.0.self_attn.q_proj.weight";

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
        .unwrap_or_else(|| OsString::from("load_artifact"));
    let artifact_path = PathBuf::from(arguments.next().with_context(|| usage(&program))?);
    let mut verification = VerificationMode::Full;
    let mut report_path = None;
    let mut staging_mib = 8_usize;

    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--verification") => {
                let value = arguments
                    .next()
                    .context("--verification requires full or contracts")?;
                verification = VerificationMode::parse(
                    value
                        .to_str()
                        .context("--verification must be valid UTF-8")?,
                )?;
            }
            Some("--staging-mib") => {
                let value = arguments
                    .next()
                    .context("--staging-mib requires an integer")?;
                staging_mib = value
                    .to_str()
                    .context("--staging-mib must be valid UTF-8")?
                    .parse()
                    .context("--staging-mib must be an integer")?;
                if !(1..=256).contains(&staging_mib) {
                    bail!("--staging-mib must be between 1 and 256");
                }
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
    let artifact = NativeArtifact::open(&artifact_path, verification)?;
    let open_milliseconds = started.elapsed().as_millis() as u64;
    let staging_bytes = staging_mib
        .checked_mul(1024 * 1024)
        .context("staging size overflows usize")?;

    let talker = artifact.talker_tensor(TALKER_PROBE)?;
    let predictor = artifact.predictor_tensor(PREDICTOR_PROBE)?;
    let decoder = artifact.decoder_tensor(DECODER_PROBE)?;
    let probes = [&talker, &predictor, &decoder]
        .into_iter()
        .map(|tensor| {
            let chunks = tensor.chunks(staging_bytes)?;
            Ok(json!({
                "name": tensor.name,
                "dtype": format!("{:?}", tensor.dtype),
                "shape": tensor.shape,
                "bytes": tensor.bytes.len(),
                "sha256": to_hex(tensor.sha256),
                "arena_offset_bytes": tensor.arena_offset_bytes,
                "staging_chunks": chunks.len(),
            }))
        })
        .collect::<Result<Vec<_>>>()?;

    let memory = artifact.model_memory_metrics();
    let report = json!({
        "schema_version": 1,
        "operation": "load-native-qwen3-tts-artifact",
        "artifact": artifact.root().display().to_string(),
        "revision": artifact.revision(),
        "verification": artifact.verification().label(),
        "decoder_dtype": artifact.decoder_dtype().label(),
        "open_milliseconds": open_milliseconds,
        "staging_buffer_bytes": staging_bytes,
        "voice_tensor_count": artifact.voice_tensor_names().count(),
        "decoder_tensor_count": artifact.decoder_tensor_names().count(),
        "model_memory": {
            "voice_mapped_file_bytes": memory.voice_mapped_file_bytes,
            "voice_tensor_payload_bytes": memory.voice_tensor_payload_bytes,
            "decoder_mapped_file_bytes": memory.decoder_mapped_file_bytes,
            "decoder_tensor_payload_bytes": memory.decoder_tensor_payload_bytes,
            "total_mapped_file_bytes": memory.total_mapped_file_bytes,
            "total_tensor_payload_bytes": memory.total_tensor_payload_bytes,
            "host_committed_weight_copy_bytes": memory.host_committed_weight_copy_bytes,
            "runtime_dtype_conversion_bytes": memory.runtime_dtype_conversion_bytes,
        },
        "zero_copy_probe_tensors": probes,
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

fn usage(program: &OsString) -> String {
    format!(
        "usage: {} <artifact-directory> [--verification full|contracts] [--staging-mib 1..256] [--report path]",
        Path::new(program).display()
    )
}
