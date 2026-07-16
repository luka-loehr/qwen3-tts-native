use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde_json::{Map, Value, json};

const VOICE_DESIGN_INVENTORY: &str = include_str!(
    "../../../../benchmarks/results/voice-design-model-inventory.json"
);
const SPEECH_TOKENIZER_INVENTORY: &str = include_str!(
    "../../../../benchmarks/results/speech-tokenizer-model-inventory.json"
);

struct Arguments {
    artifact: PathBuf,
    allow_symlinks: bool,
    output: Option<PathBuf>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let arguments = parse_arguments()?;
    let artifact = arguments
        .artifact
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", arguments.artifact.display()))?;

    let config_path = artifact.join("config.json");
    let generation_config_path = artifact.join("generation_config.json");
    let model_path = artifact.join("model.safetensors");
    let tokenizer_dir = artifact.join("speech_tokenizer");
    let tokenizer_config_path = tokenizer_dir.join("config.json");
    let tokenizer_model_path = tokenizer_dir.join("model.safetensors");

    let material = [
        &config_path,
        &generation_config_path,
        &model_path,
        &tokenizer_config_path,
        &tokenizer_model_path,
    ]
    .into_iter()
    .map(|path| inspect_material_file(path, arguments.allow_symlinks))
    .collect::<Result<Vec<_>>>()?;

    let config = load_json(&config_path)?;
    let generation_config = load_json(&generation_config_path)?;
    let tokenizer_config = load_json(&tokenizer_config_path)?;
    let language_ids = validate_voice_design_config(&config)?;
    validate_generation_config(&generation_config)?;
    validate_tokenizer_config(&tokenizer_config)?;

    let voice_design = validate_tensor_contract(
        &model_path,
        VOICE_DESIGN_INVENTORY,
        "voice-design-1.7b",
    )?;
    let speech_tokenizer = validate_tensor_contract(
        &tokenizer_model_path,
        SPEECH_TOKENIZER_INVENTORY,
        "speech-tokenizer",
    )?;

    let symlink_count = material
        .iter()
        .filter(|entry| entry["symlink"].as_bool() == Some(true))
        .count();
    let report = json!({
        "schema_version": 1,
        "operation": "validate-qwen3-tts-1.7b-artifact",
        "artifact": artifact.display().to_string(),
        "valid": true,
        "production_material": symlink_count == 0,
        "allow_symlinks": arguments.allow_symlinks,
        "symlink_count": symlink_count,
        "material": material,
        "voice_design": voice_design,
        "speech_tokenizer": speech_tokenizer,
        "language_ids": language_ids,
        "explicit_turkish_language_id": false,
    });

    let encoded = serde_json::to_string_pretty(&report)?;
    if let Some(path) = arguments.output {
        fs::write(&path, encoded)
            .with_context(|| format!("failed to write {}", path.display()))?;
    } else {
        println!("{encoded}");
    }
    Ok(())
}

fn parse_arguments() -> Result<Arguments> {
    let mut arguments = env::args_os();
    let program = arguments
        .next()
        .unwrap_or_else(|| OsString::from("validate_artifact"));
    let artifact = arguments.next().with_context(|| usage(&program))?;
    let mut allow_symlinks = false;
    let mut output = None;

    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--allow-symlinks") => allow_symlinks = true,
            Some("--output") => {
                output = Some(PathBuf::from(
                    arguments.next().context("--output requires a destination")?,
                ));
            }
            _ => bail!("unknown argument {argument:?}\n{}", usage(&program)),
        }
    }

    Ok(Arguments {
        artifact: PathBuf::from(artifact),
        allow_symlinks,
        output,
    })
}

fn usage(program: &OsString) -> String {
    format!(
        "usage: {} <artifact-directory> [--allow-symlinks] [--output report.json]",
        Path::new(program).display()
    )
}

fn inspect_material_file(path: &Path, allow_symlinks: bool) -> Result<Value> {
    let link_metadata = fs::symlink_metadata(path)
        .with_context(|| format!("required artifact file is missing: {}", path.display()))?;
    let symlink = link_metadata.file_type().is_symlink();
    if symlink && !allow_symlinks {
        bail!(
            "{} is a symlink; production artifacts must be flat regular files",
            path.display()
        );
    }
    let metadata = fs::metadata(path)
        .with_context(|| format!("failed to stat {}", path.display()))?;
    if !metadata.is_file() {
        bail!("{} is not a regular file", path.display());
    }
    let resolved = path
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", path.display()))?;

    Ok(json!({
        "path": path.display().to_string(),
        "resolved_path": resolved.display().to_string(),
        "bytes": metadata.len(),
        "symlink": symlink,
    }))
}

fn load_json(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("invalid JSON in {}", path.display()))
}

fn validate_voice_design_config(config: &Value) -> Result<Value> {
    expect_string(config, "/model_type", "qwen3_tts")?;
    expect_string(config, "/tokenizer_type", "qwen3_tts_tokenizer_12hz")?;
    expect_string(config, "/tts_model_size", "1b7")?;
    expect_string(config, "/tts_model_type", "voice_design")?;
    expect_u64(config, "/assistant_token_id", 77_091)?;
    expect_u64(config, "/im_end_token_id", 151_645)?;
    expect_u64(config, "/im_start_token_id", 151_644)?;
    expect_u64(config, "/tts_bos_token_id", 151_672)?;
    expect_u64(config, "/tts_eos_token_id", 151_673)?;
    expect_u64(config, "/tts_pad_token_id", 151_671)?;

    for (pointer, expected) in [
        ("/talker_config/hidden_size", 2_048),
        ("/talker_config/intermediate_size", 6_144),
        ("/talker_config/num_hidden_layers", 28),
        ("/talker_config/num_attention_heads", 16),
        ("/talker_config/num_key_value_heads", 8),
        ("/talker_config/head_dim", 128),
        ("/talker_config/vocab_size", 3_072),
        ("/talker_config/text_vocab_size", 151_936),
        ("/talker_config/num_code_groups", 16),
        ("/talker_config/code_predictor_config/hidden_size", 1_024),
        ("/talker_config/code_predictor_config/intermediate_size", 3_072),
        ("/talker_config/code_predictor_config/num_hidden_layers", 5),
        ("/talker_config/code_predictor_config/num_attention_heads", 16),
        ("/talker_config/code_predictor_config/num_key_value_heads", 8),
        ("/talker_config/code_predictor_config/vocab_size", 2_048),
        ("/talker_config/code_predictor_config/num_code_groups", 16),
    ] {
        expect_u64(config, pointer, expected)?;
    }

    expect_value(
        config,
        "/talker_config/rope_scaling/mrope_section",
        &json!([24, 20, 20]),
    )?;
    expect_value(
        config,
        "/talker_config/codec_language_id",
        &json!({
            "chinese": 2055,
            "english": 2050,
            "french": 2061,
            "german": 2053,
            "italian": 2070,
            "japanese": 2058,
            "korean": 2064,
            "portuguese": 2071,
            "russian": 2069,
            "spanish": 2054,
        }),
    )?;

    Ok(config["talker_config"]["codec_language_id"].clone())
}

fn validate_generation_config(config: &Value) -> Result<()> {
    let expected = json!({
        "do_sample": true,
        "max_new_tokens": 8192,
        "repetition_penalty": 1.05,
        "subtalker_dosample": true,
        "subtalker_temperature": 0.9,
        "subtalker_top_k": 50,
        "subtalker_top_p": 1.0,
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 1.0,
    });
    if config != &expected {
        bail!("generation_config.json does not match the pinned VoiceDesign defaults");
    }
    Ok(())
}

fn validate_tokenizer_config(config: &Value) -> Result<()> {
    expect_string(config, "/model_type", "qwen3_tts_tokenizer_12hz")?;
    expect_u64(config, "/input_sample_rate", 24_000)?;
    expect_u64(config, "/output_sample_rate", 24_000)?;
    expect_u64(config, "/decode_upsample_rate", 1_920)?;
    expect_u64(config, "/encode_downsample_rate", 1_920)?;

    for (pointer, expected) in [
        ("/decoder_config/latent_dim", 1_024),
        ("/decoder_config/codebook_dim", 512),
        ("/decoder_config/codebook_size", 2_048),
        ("/decoder_config/decoder_dim", 1_536),
        ("/decoder_config/hidden_size", 512),
        ("/decoder_config/intermediate_size", 1_024),
        ("/decoder_config/head_dim", 64),
        ("/decoder_config/num_attention_heads", 16),
        ("/decoder_config/num_key_value_heads", 16),
        ("/decoder_config/num_hidden_layers", 8),
        ("/decoder_config/num_quantizers", 16),
        ("/decoder_config/sliding_window", 72),
    ] {
        expect_u64(config, pointer, expected)?;
    }
    expect_value(
        config,
        "/decoder_config/upsample_rates",
        &json!([8, 5, 4, 3]),
    )?;
    expect_value(
        config,
        "/decoder_config/upsampling_ratios",
        &json!([2, 2]),
    )?;
    Ok(())
}

fn validate_tensor_contract(path: &Path, inventory_json: &str, label: &str) -> Result<Value> {
    let expected: Value = serde_json::from_str(inventory_json)
        .with_context(|| format!("embedded {label} inventory is invalid"))?;
    let expected_file_bytes = expected["file_bytes"]
        .as_u64()
        .with_context(|| format!("{label} inventory has no file_bytes"))?;
    let expected_payload_bytes = expected["payload_bytes"]
        .as_u64()
        .with_context(|| format!("{label} inventory has no payload_bytes"))?;
    let expected_parameters = expected["parameter_count"]
        .as_u64()
        .with_context(|| format!("{label} inventory has no parameter_count"))?;
    let expected_tensors = expected["tensors"]
        .as_array()
        .with_context(|| format!("{label} inventory has no tensor list"))?;

    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let file_bytes = file
        .metadata()
        .with_context(|| format!("failed to stat {}", path.display()))?
        .len();
    if file_bytes != expected_file_bytes {
        bail!(
            "{label} file size mismatch: expected {expected_file_bytes}, found {file_bytes}"
        );
    }

    // SAFETY: The immutable model file is mapped read-only and remains open for
    // the complete SafeTensors validation lifetime.
    let mapping = unsafe { Mmap::map(&file) }
        .with_context(|| format!("failed to memory-map {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&mapping[..])
        .with_context(|| format!("invalid SafeTensors file {}", path.display()))?;
    if tensors.len() != expected_tensors.len() {
        bail!(
            "{label} tensor count mismatch: expected {}, found {}",
            expected_tensors.len(),
            tensors.len()
        );
    }

    let mut expected_names = BTreeSet::new();
    let mut payload_bytes = 0_u64;
    let mut parameter_count = 0_u64;
    let mut decoder_tensor_count = 0_u64;
    let mut decoder_payload_bytes = 0_u64;
    let mut decoder_parameter_count = 0_u64;
    let mut dtype_counts = BTreeMap::<String, u64>::new();

    for entry in expected_tensors {
        let name = entry["name"]
            .as_str()
            .with_context(|| format!("{label} inventory contains a tensor without a name"))?;
        if !expected_names.insert(name.to_owned()) {
            bail!("{label} inventory contains duplicate tensor {name}");
        }
        let expected_dtype = entry["dtype"]
            .as_str()
            .with_context(|| format!("{label} tensor {name} has no dtype"))?;
        let expected_shape = parse_shape(&entry["shape"], label, name)?;
        let expected_bytes = entry["bytes"]
            .as_u64()
            .with_context(|| format!("{label} tensor {name} has no byte count"))?;
        let expected_tensor_parameters = entry["parameters"]
            .as_u64()
            .with_context(|| format!("{label} tensor {name} has no parameter count"))?;

        let tensor = tensors
            .tensor(name)
            .with_context(|| format!("{label} is missing tensor {name}"))?;
        let actual_dtype = format!("{:?}", tensor.dtype());
        if actual_dtype != expected_dtype {
            bail!(
                "{label} tensor {name} dtype mismatch: expected {expected_dtype}, found {actual_dtype}"
            );
        }
        if tensor.shape() != expected_shape.as_slice() {
            bail!(
                "{label} tensor {name} shape mismatch: expected {:?}, found {:?}",
                expected_shape,
                tensor.shape()
            );
        }
        let actual_bytes = tensor.data().len() as u64;
        if actual_bytes != expected_bytes {
            bail!(
                "{label} tensor {name} byte mismatch: expected {expected_bytes}, found {actual_bytes}"
            );
        }

        payload_bytes = payload_bytes
            .checked_add(actual_bytes)
            .context("payload byte count overflow")?;
        parameter_count = parameter_count
            .checked_add(expected_tensor_parameters)
            .context("parameter count overflow")?;
        *dtype_counts.entry(actual_dtype).or_default() += 1;
        if name.starts_with("decoder.") {
            decoder_tensor_count += 1;
            decoder_payload_bytes += actual_bytes;
            decoder_parameter_count += expected_tensor_parameters;
        }
    }

    let actual_names = tensors
        .names()
        .into_iter()
        .map(ToOwned::to_owned)
        .collect::<BTreeSet<_>>();
    if actual_names != expected_names {
        bail!("{label} tensor-name set differs from the pinned inventory");
    }
    if payload_bytes != expected_payload_bytes {
        bail!(
            "{label} payload mismatch: expected {expected_payload_bytes}, found {payload_bytes}"
        );
    }
    if parameter_count != expected_parameters {
        bail!(
            "{label} parameter mismatch: expected {expected_parameters}, found {parameter_count}"
        );
    }

    Ok(json!({
        "contract": label,
        "file_bytes": file_bytes,
        "payload_bytes": payload_bytes,
        "tensor_count": tensors.len(),
        "parameter_count": parameter_count,
        "dtype_counts": dtype_counts,
        "decoder_only": {
            "tensor_count": decoder_tensor_count,
            "f32_payload_bytes": decoder_payload_bytes,
            "parameter_count": decoder_parameter_count,
            "projected_bf16_payload_bytes": decoder_parameter_count * 2,
        },
    }))
}

fn parse_shape(value: &Value, label: &str, name: &str) -> Result<Vec<usize>> {
    value
        .as_array()
        .with_context(|| format!("{label} tensor {name} shape is not an array"))?
        .iter()
        .map(|dimension| {
            let dimension = dimension
                .as_u64()
                .with_context(|| format!("{label} tensor {name} has an invalid dimension"))?;
            usize::try_from(dimension)
                .with_context(|| format!("{label} tensor {name} dimension is too large"))
        })
        .collect()
}

fn expect_string(root: &Value, pointer: &str, expected: &str) -> Result<()> {
    let actual = root
        .pointer(pointer)
        .and_then(Value::as_str)
        .with_context(|| format!("{pointer} is missing or is not a string"))?;
    if actual != expected {
        bail!("{pointer}: expected {expected:?}, found {actual:?}");
    }
    Ok(())
}

fn expect_u64(root: &Value, pointer: &str, expected: u64) -> Result<()> {
    let actual = root
        .pointer(pointer)
        .and_then(Value::as_u64)
        .with_context(|| format!("{pointer} is missing or is not an unsigned integer"))?;
    if actual != expected {
        bail!("{pointer}: expected {expected}, found {actual}");
    }
    Ok(())
}

fn expect_value(root: &Value, pointer: &str, expected: &Value) -> Result<()> {
    let actual = root
        .pointer(pointer)
        .with_context(|| format!("{pointer} is missing"))?;
    if actual != expected {
        bail!("{pointer}: expected {expected}, found {actual}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_inventories_are_complete_and_unique() {
        for (label, encoded, expected_count) in [
            ("voice", VOICE_DESIGN_INVENTORY, 404),
            ("tokenizer", SPEECH_TOKENIZER_INVENTORY, 496),
        ] {
            let inventory: Value = serde_json::from_str(encoded).unwrap();
            let tensors = inventory["tensors"].as_array().unwrap();
            assert_eq!(tensors.len(), expected_count, "{label}");
            let names = tensors
                .iter()
                .map(|tensor| tensor["name"].as_str().unwrap())
                .collect::<BTreeSet<_>>();
            assert_eq!(names.len(), expected_count, "{label}");
        }
    }

    #[test]
    fn pinned_language_map_does_not_claim_turkish_support() {
        let mut root = Map::new();
        root.insert(
            "language_ids".to_owned(),
            json!({"german": 2053, "english": 2050, "french": 2061, "italian": 2070}),
        );
        assert!(root["language_ids"].get("turkish").is_none());
    }
}
