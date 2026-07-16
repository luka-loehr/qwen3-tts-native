use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use safetensors::tensor::{View, serialize_to_file};
use safetensors::{Dtype, SafeTensors};
use serde_json::{Map, Value, json};

use crate::contract::{
    CodecWeightDtype, TensorContract, speech_decoder_contract, speech_tokenizer_contract,
    validate_tensors, voice_design_contract,
};
use crate::sha256::{digest_reader, to_hex};

const MODEL_ID: &str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign";
const VOICE_PATH: &str = "model.safetensors";
const CODEC_PATH: &str = "speech_tokenizer/model.safetensors";
const DEFAULT_COPY_BUFFER_BYTES: usize = 8 * 1024 * 1024;

const MATERIAL_FILES: &[(&str, &str)] = &[
    ("config.json", "voice-design-config"),
    ("generation_config.json", "generation-config"),
    ("tokenizer_config.json", "text-tokenizer-config"),
    ("vocab.json", "text-tokenizer-vocabulary"),
    ("merges.txt", "text-tokenizer-merges"),
    ("preprocessor_config.json", "preprocessor-config"),
    ("speech_tokenizer/config.json", "speech-decoder-config"),
    (
        "speech_tokenizer/configuration.json",
        "speech-tokenizer-configuration",
    ),
    (
        "speech_tokenizer/preprocessor_config.json",
        "speech-preprocessor-config",
    ),
];

#[derive(Clone, Debug)]
pub struct PackOptions {
    pub source_snapshot: PathBuf,
    pub output: PathBuf,
    pub codec_dtype: CodecWeightDtype,
    pub copy_buffer_bytes: usize,
}

impl PackOptions {
    pub fn new(source_snapshot: PathBuf, output: PathBuf) -> Self {
        Self {
            source_snapshot,
            output,
            codec_dtype: CodecWeightDtype::Bf16,
            copy_buffer_bytes: DEFAULT_COPY_BUFFER_BYTES,
        }
    }
}

#[derive(Clone, Debug)]
struct FileRecord {
    role: String,
    bytes: u64,
    sha256: String,
}

impl FileRecord {
    fn as_json(&self) -> Value {
        json!({
            "role": self.role,
            "bytes": self.bytes,
            "sha256": self.sha256,
        })
    }
}

struct StagingDirectory {
    path: PathBuf,
    committed: bool,
}

impl StagingDirectory {
    fn create(output: &Path) -> Result<Self> {
        if output.exists() {
            bail!("output already exists: {}", output.display());
        }
        let parent = output
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
        let output_name = output
            .file_name()
            .context("artifact output must have a final path component")?
            .to_string_lossy();
        let path = parent.join(format!(".{output_name}.staging-{}", std::process::id()));
        fs::create_dir(&path)
            .with_context(|| format!("failed to create staging directory {}", path.display()))?;
        Ok(Self {
            path,
            committed: false,
        })
    }

    fn commit(mut self, output: &Path) -> Result<()> {
        sync_directory(&self.path)?;
        fs::rename(&self.path, output).with_context(|| {
            format!(
                "failed to atomically publish {} as {}",
                self.path.display(),
                output.display()
            )
        })?;
        if let Some(parent) = output.parent() {
            sync_directory(parent)?;
        }
        self.committed = true;
        Ok(())
    }
}

impl Drop for StagingDirectory {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

enum PackedTensor<'data> {
    Borrowed {
        shape: Vec<usize>,
        data: &'data [u8],
    },
    F32ToBf16 {
        shape: Vec<usize>,
        data: &'data [u8],
    },
}

impl View for PackedTensor<'_> {
    fn dtype(&self) -> Dtype {
        match self {
            Self::Borrowed { .. } => Dtype::F32,
            Self::F32ToBf16 { .. } => Dtype::BF16,
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            Self::Borrowed { shape, .. } | Self::F32ToBf16 { shape, .. } => shape,
        }
    }

    fn data(&self) -> Cow<'_, [u8]> {
        match self {
            Self::Borrowed { data, .. } => Cow::Borrowed(data),
            Self::F32ToBf16 { data, .. } => Cow::Owned(f32_bytes_to_bf16(data)),
        }
    }

    fn data_len(&self) -> usize {
        match self {
            Self::Borrowed { data, .. } => data.len(),
            Self::F32ToBf16 { data, .. } => data.len() / 2,
        }
    }
}

pub fn pack_artifact(options: &PackOptions) -> Result<Value> {
    if options.copy_buffer_bytes == 0 {
        bail!("copy buffer size must be greater than zero");
    }
    let started = Instant::now();
    let source = options
        .source_snapshot
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", options.source_snapshot.display()))?;
    if !source.is_dir() {
        bail!("source snapshot is not a directory: {}", source.display());
    }
    let revision = snapshot_revision(&source)?;

    let voice_source = source.join(VOICE_PATH);
    let codec_source = source.join(CODEC_PATH);
    let voice_file = File::open(&voice_source)
        .with_context(|| format!("failed to open {}", voice_source.display()))?;
    let codec_file = File::open(&codec_source)
        .with_context(|| format!("failed to open {}", codec_source.display()))?;
    ensure_expected_file_size(&voice_file, &voice_design_contract()?, &voice_source)?;
    ensure_expected_file_size(&codec_file, &speech_tokenizer_contract()?, &codec_source)?;

    // SAFETY: Both source files are immutable Hugging Face blob snapshots,
    // mapped read-only, and kept open until packing and validation complete.
    let voice_mapping = unsafe { Mmap::map(&voice_file) }
        .with_context(|| format!("failed to map {}", voice_source.display()))?;
    // SAFETY: Same immutable read-only lifetime guarantee as above.
    let codec_mapping = unsafe { Mmap::map(&codec_file) }
        .with_context(|| format!("failed to map {}", codec_source.display()))?;
    let voice_tensors = SafeTensors::deserialize(&voice_mapping)
        .with_context(|| format!("invalid SafeTensors file {}", voice_source.display()))?;
    let codec_tensors = SafeTensors::deserialize(&codec_mapping)
        .with_context(|| format!("invalid SafeTensors file {}", codec_source.display()))?;
    let voice_contract = voice_design_contract()?;
    let tokenizer_contract = speech_tokenizer_contract()?;
    validate_tensors(&voice_tensors, &voice_contract)?;
    validate_tensors(&codec_tensors, &tokenizer_contract)?;

    let decoder_contract = speech_decoder_contract(options.codec_dtype)?;
    if options.codec_dtype == CodecWeightDtype::Bf16 {
        validate_decoder_finite(&codec_tensors, &decoder_contract)?;
    }

    let validation_milliseconds = started.elapsed().as_millis() as u64;
    let staging = StagingDirectory::create(&options.output)?;
    let mut records = BTreeMap::<String, FileRecord>::new();

    for (relative, role) in MATERIAL_FILES {
        let record = copy_regular_file(
            &source.join(relative),
            &staging.path.join(relative),
            role,
            options.copy_buffer_bytes,
        )?;
        records.insert((*relative).to_owned(), record);
    }

    let voice_record = copy_regular_file(
        &voice_source,
        &staging.path.join(VOICE_PATH),
        "voice-design-weights",
        options.copy_buffer_bytes,
    )?;
    records.insert(VOICE_PATH.to_owned(), voice_record);
    let copy_milliseconds = started.elapsed().as_millis() as u64 - validation_milliseconds;

    let packed_codec = staging.path.join(CODEC_PATH);
    if let Some(parent) = packed_codec.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    write_decoder_checkpoint(
        &codec_tensors,
        &decoder_contract,
        options.codec_dtype,
        &packed_codec,
    )?;
    let packed_codec_file = File::open(&packed_codec)
        .with_context(|| format!("failed to open {}", packed_codec.display()))?;
    // SAFETY: The newly written checkpoint is immutable during this read-only
    // mapping and remains open for the complete validation lifetime.
    let packed_codec_mapping = unsafe { Mmap::map(&packed_codec_file) }
        .with_context(|| format!("failed to map {}", packed_codec.display()))?;
    let packed_codec_tensors = SafeTensors::deserialize(&packed_codec_mapping)
        .with_context(|| format!("invalid packed checkpoint {}", packed_codec.display()))?;
    validate_tensors(&packed_codec_tensors, &decoder_contract)?;

    let codec_record = hash_regular_file(
        &packed_codec,
        "speech-decoder-weights",
        options.copy_buffer_bytes,
    )?;
    records.insert(CODEC_PATH.to_owned(), codec_record);
    let decoder_milliseconds =
        started.elapsed().as_millis() as u64 - validation_milliseconds - copy_milliseconds;

    let source_codec_sha256 = hash_regular_file(
        &codec_source,
        "source-speech-tokenizer",
        options.copy_buffer_bytes,
    )?
    .sha256;
    let manifest = build_manifest(
        &revision,
        options.codec_dtype,
        &voice_contract,
        &tokenizer_contract,
        &decoder_contract,
        &source_codec_sha256,
        &records,
    );
    write_json_file(&staging.path.join("manifest.json"), &manifest)?;
    assert_flat_regular_tree(&staging.path)?;
    staging.commit(&options.output)?;

    let total_milliseconds = started.elapsed().as_millis() as u64;
    Ok(json!({
        "schema_version": 1,
        "operation": "pack-native-qwen3-tts-artifact",
        "output": options.output.display().to_string(),
        "codec_dtype": options.codec_dtype.label(),
        "regular_files_only": true,
        "voice_design_bytes": records[VOICE_PATH].bytes,
        "speech_decoder_bytes": records[CODEC_PATH].bytes,
        "speech_decoder_payload_bytes": decoder_contract.payload_bytes,
        "timing_milliseconds": {
            "source_contract_validation": validation_milliseconds,
            "material_and_voice_copy": copy_milliseconds,
            "decoder_write_and_validation": decoder_milliseconds,
            "total": total_milliseconds,
        },
        "copy_buffer_bytes": options.copy_buffer_bytes,
    }))
}

fn write_decoder_checkpoint(
    source: &SafeTensors<'_>,
    contract: &TensorContract,
    dtype: CodecWeightDtype,
    destination: &Path,
) -> Result<()> {
    let tensors = contract
        .tensors
        .iter()
        .map(|(name, spec)| {
            let tensor = source
                .tensor(name)
                .with_context(|| format!("source speech tokenizer is missing {name}"))?;
            if tensor.dtype() != Dtype::F32 {
                bail!("source tensor {name} is not F32");
            }
            let packed = match dtype {
                CodecWeightDtype::F32 => PackedTensor::Borrowed {
                    shape: spec.shape.clone(),
                    data: tensor.data(),
                },
                CodecWeightDtype::Bf16 => PackedTensor::F32ToBf16 {
                    shape: spec.shape.clone(),
                    data: tensor.data(),
                },
            };
            Ok((name.as_str(), packed))
        })
        .collect::<Result<Vec<_>>>()?;

    serialize_to_file(tensors, None, destination)
        .with_context(|| format!("failed to serialize {}", destination.display()))?;
    File::open(destination)
        .with_context(|| format!("failed to reopen {}", destination.display()))?
        .sync_all()
        .with_context(|| format!("failed to sync {}", destination.display()))
}

fn validate_decoder_finite(
    source: &SafeTensors<'_>,
    decoder_contract: &TensorContract,
) -> Result<()> {
    for name in decoder_contract.tensors.keys() {
        let tensor = source
            .tensor(name)
            .with_context(|| format!("source speech tokenizer is missing {name}"))?;
        let mut values = tensor.data().chunks_exact(4);
        for (index, bytes) in values.by_ref().enumerate() {
            let value = f32::from_bits(u32::from_le_bytes(
                bytes.try_into().expect("four-byte chunk"),
            ));
            if !value.is_finite() {
                bail!("source decoder tensor {name} contains non-finite value at index {index}");
            }
        }
        if !values.remainder().is_empty() {
            bail!("source decoder tensor {name} has a truncated F32 payload");
        }
    }
    Ok(())
}

fn f32_bytes_to_bf16(source: &[u8]) -> Vec<u8> {
    debug_assert_eq!(source.len() % 4, 0);
    let mut output = Vec::with_capacity(source.len() / 2);
    for bytes in source.chunks_exact(4) {
        let bits = u32::from_le_bytes(bytes.try_into().expect("four-byte chunk"));
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
        output.extend_from_slice(&((rounded >> 16) as u16).to_le_bytes());
    }
    output
}

fn copy_regular_file(
    source: &Path,
    destination: &Path,
    role: &str,
    buffer_bytes: usize,
) -> Result<FileRecord> {
    let metadata =
        fs::metadata(source).with_context(|| format!("failed to stat {}", source.display()))?;
    if !metadata.is_file() {
        bail!(
            "source material is not a regular file: {}",
            source.display()
        );
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let input =
        File::open(source).with_context(|| format!("failed to open {}", source.display()))?;
    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .with_context(|| format!("failed to create {}", destination.display()))?;
    let mut reader = BufReader::with_capacity(buffer_bytes, input);
    let mut writer = BufWriter::with_capacity(buffer_bytes, output);
    let mut hasher = crate::sha256::Sha256::new();
    let mut buffer = vec![0_u8; buffer_bytes];
    let mut bytes = 0_u64;

    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("failed to read {}", source.display()))?;
        if read == 0 {
            break;
        }
        writer
            .write_all(&buffer[..read])
            .with_context(|| format!("failed to write {}", destination.display()))?;
        hasher.update(&buffer[..read]);
        bytes = bytes
            .checked_add(read as u64)
            .context("copied byte count overflowed")?;
    }
    writer
        .flush()
        .with_context(|| format!("failed to flush {}", destination.display()))?;
    writer
        .get_ref()
        .sync_all()
        .with_context(|| format!("failed to sync {}", destination.display()))?;
    if bytes != metadata.len() {
        bail!(
            "source changed while copying {}: expected {} bytes, read {bytes}",
            source.display(),
            metadata.len()
        );
    }
    Ok(FileRecord {
        role: role.to_owned(),
        bytes,
        sha256: to_hex(&hasher.finalize()),
    })
}

fn hash_regular_file(path: &Path, role: &str, buffer_bytes: usize) -> Result<FileRecord> {
    let metadata =
        fs::metadata(path).with_context(|| format!("failed to stat {}", path.display()))?;
    if !metadata.is_file() {
        bail!("material is not a regular file: {}", path.display());
    }
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::with_capacity(buffer_bytes, file);
    let (digest, bytes) = digest_reader(&mut reader, buffer_bytes)
        .with_context(|| format!("failed to hash {}", path.display()))?;
    if bytes != metadata.len() {
        bail!(
            "file changed while hashing {}: expected {} bytes, read {bytes}",
            path.display(),
            metadata.len()
        );
    }
    Ok(FileRecord {
        role: role.to_owned(),
        bytes,
        sha256: to_hex(&digest),
    })
}

fn ensure_expected_file_size(file: &File, contract: &TensorContract, path: &Path) -> Result<()> {
    let actual = file
        .metadata()
        .with_context(|| format!("failed to stat {}", path.display()))?
        .len();
    if let Some(expected) = contract.exact_file_bytes
        && actual != expected
    {
        bail!(
            "{} file size mismatch: expected {expected}, found {actual}",
            contract.label
        );
    }
    Ok(())
}

fn snapshot_revision(source: &Path) -> Result<String> {
    let revision = source
        .file_name()
        .and_then(|name| name.to_str())
        .context("source snapshot path has no UTF-8 revision component")?;
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!(
            "source snapshot revision must be a 40-character hexadecimal commit, found {revision:?}"
        );
    }
    Ok(revision.to_ascii_lowercase())
}

fn build_manifest(
    revision: &str,
    codec_dtype: CodecWeightDtype,
    voice: &TensorContract,
    source_tokenizer: &TensorContract,
    decoder: &TensorContract,
    source_codec_sha256: &str,
    records: &BTreeMap<String, FileRecord>,
) -> Value {
    let files = records
        .iter()
        .map(|(path, record)| (path.clone(), record.as_json()))
        .collect::<Map<String, Value>>();
    json!({
        "schema_version": 1,
        "artifact": "qwen3-tts-native-1.7b-voice-design",
        "model": {
            "id": MODEL_ID,
            "revision": revision,
        },
        "production_material": {
            "regular_files_only": true,
            "symlink_count": 0,
            "runtime_dependencies": ["rust", "cuda"],
        },
        "files": files,
        "source_contract": {
            "voice_design": {
                "dtype": "BF16",
                "tensor_count": voice.tensor_count,
                "parameter_count": voice.parameter_count,
                "payload_bytes": voice.payload_bytes,
            },
            "speech_tokenizer": {
                "dtype": "F32",
                "sha256": source_codec_sha256,
                "tensor_count": source_tokenizer.tensor_count,
                "parameter_count": source_tokenizer.parameter_count,
                "payload_bytes": source_tokenizer.payload_bytes,
            },
        },
        "weights": {
            "voice_design": {
                "path": VOICE_PATH,
                "dtype": "BF16",
                "tensor_count": voice.tensor_count,
                "parameter_count": voice.parameter_count,
                "payload_bytes": voice.payload_bytes,
            },
            "speech_decoder": {
                "path": CODEC_PATH,
                "source_dtype": "F32",
                "dtype": codec_dtype.label(),
                "tensor_count": decoder.tensor_count,
                "parameter_count": decoder.parameter_count,
                "payload_bytes": decoder.payload_bytes,
                "encoder_included": false,
                "tensor_name_prefix": "decoder.",
                "conversion": if codec_dtype == CodecWeightDtype::Bf16 {
                    "round-to-nearest-even"
                } else {
                    "none"
                },
            },
        },
    })
}

fn write_json_file(path: &Path, value: &Value) -> Result<()> {
    let mut encoded = serde_json::to_vec_pretty(value)?;
    encoded.push(b'\n');
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    file.write_all(&encoded)
        .with_context(|| format!("failed to write {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("failed to sync {}", path.display()))
}

fn assert_flat_regular_tree(root: &Path) -> Result<()> {
    let mut pending = vec![root.to_owned()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)
            .with_context(|| format!("failed to list {}", directory.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)
                .with_context(|| format!("failed to inspect {}", path.display()))?;
            if metadata.file_type().is_symlink() {
                bail!("packed artifact contains a symlink: {}", path.display());
            }
            if metadata.is_dir() {
                pending.push(path);
            } else if !metadata.is_file() {
                bail!(
                    "packed artifact contains non-regular material: {}",
                    path.display()
                );
            }
        }
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .with_context(|| format!("failed to open directory {}", path.display()))?
        .sync_all()
        .with_context(|| format!("failed to sync directory {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_conversion_uses_round_to_nearest_even() {
        let values = [
            0x3f80_0000_u32,
            0x3f80_7fff,
            0x3f80_8000,
            0x3f81_8000,
            0xbf80_8000,
        ];
        let source = values
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let converted = f32_bytes_to_bf16(&source);
        let actual = converted
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(actual, [0x3f80, 0x3f80, 0x3f80, 0x3f82, 0xbf80]);
    }

    #[test]
    fn snapshot_revision_is_strict() {
        assert_eq!(
            snapshot_revision(Path::new(
                "/models/snapshots/5ecdb67327fd37bb2e042aab12ff7391903235d3"
            ))
            .unwrap(),
            "5ecdb67327fd37bb2e042aab12ff7391903235d3"
        );
        assert!(snapshot_revision(Path::new("/models/snapshots/main")).is_err());
    }

    #[test]
    fn zero_copy_f32_and_bounded_bf16_views_report_exact_lengths() {
        let source = [0_u8; 16];
        let f32 = PackedTensor::Borrowed {
            shape: vec![4],
            data: &source,
        };
        let bf16 = PackedTensor::F32ToBf16 {
            shape: vec![4],
            data: &source,
        };
        assert!(matches!(f32.data(), Cow::Borrowed(_)));
        assert_eq!(f32.data_len(), 16);
        assert_eq!(bf16.data_len(), 8);
        assert_eq!(bf16.data().len(), 8);
    }
}
