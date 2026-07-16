use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;

use crate::contract::{
    CodecWeightDtype, TensorContract, speech_decoder_contract, validate_tensors,
    voice_design_contract,
};
use crate::sha256::{digest_reader, to_hex};

const MODEL_ID: &str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign";
const ARTIFACT_KIND: &str = "qwen3-tts-native-1.7b-voice-design";
const VOICE_PATH: &str = "model.safetensors";
const CODEC_PATH: &str = "speech_tokenizer/model.safetensors";
const MANIFEST_MAX_BYTES: u64 = 1024 * 1024;
const HASH_BUFFER_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerificationMode {
    Full,
    ContractsOnly,
}

impl VerificationMode {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "full" => Ok(Self::Full),
            "contracts" => Ok(Self::ContractsOnly),
            _ => bail!("unknown verification mode {value:?}; expected full or contracts"),
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::ContractsOnly => "contracts",
        }
    }
}

#[derive(Clone, Debug)]
struct ManifestFile {
    bytes: u64,
    sha256: String,
}

struct MappedCheckpoint {
    mapping: Mmap,
    contract: TensorContract,
}

impl MappedCheckpoint {
    fn open(path: &Path, contract: TensorContract) -> Result<Self> {
        ensure_regular_material(path)?;
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        if let Some(expected) = contract.exact_file_bytes {
            let actual = file
                .metadata()
                .with_context(|| format!("failed to stat {}", path.display()))?
                .len();
            if actual != expected {
                bail!(
                    "{} file size mismatch: expected {expected}, found {actual}",
                    contract.label
                );
            }
        }
        // SAFETY: The artifact is immutable production material, mapped
        // read-only. The mapping owns its OS lifetime after File is dropped.
        let mapping = unsafe { Mmap::map(&file) }
            .with_context(|| format!("failed to map {}", path.display()))?;
        let tensors = SafeTensors::deserialize(&mapping)
            .with_context(|| format!("invalid SafeTensors checkpoint {}", path.display()))?;
        validate_tensors(&tensors, &contract)?;
        Ok(Self { mapping, contract })
    }

    fn tensor<'a>(&'a self, name: &str) -> Result<TensorRef<'a>> {
        let spec = self
            .contract
            .tensors
            .get(name)
            .with_context(|| format!("{} has no tensor {name}", self.contract.label))?;
        let tensors = SafeTensors::deserialize(&self.mapping)
            .with_context(|| format!("{} metadata became invalid", self.contract.label))?;
        let tensor = tensors
            .tensor(name)
            .with_context(|| format!("{} is missing tensor {name}", self.contract.label))?;
        Ok(TensorRef {
            name: spec.name.as_str(),
            dtype: spec.dtype,
            shape: spec.shape.as_slice(),
            bytes: tensor.data(),
        })
    }

    fn names(&self) -> impl Iterator<Item = &str> {
        self.contract.tensors.keys().map(String::as_str)
    }
}

pub struct NativeArtifact {
    root: PathBuf,
    revision: String,
    verification: VerificationMode,
    voice: MappedCheckpoint,
    decoder: MappedCheckpoint,
    decoder_dtype: CodecWeightDtype,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelMemoryMetrics {
    pub voice_mapped_file_bytes: u64,
    pub voice_tensor_payload_bytes: u64,
    pub decoder_mapped_file_bytes: u64,
    pub decoder_tensor_payload_bytes: u64,
    pub total_mapped_file_bytes: u64,
    pub total_tensor_payload_bytes: u64,
}

#[derive(Clone, Copy)]
pub struct TensorRef<'data> {
    pub name: &'data str,
    pub dtype: Dtype,
    pub shape: &'data [usize],
    pub bytes: &'data [u8],
}

impl<'data> TensorRef<'data> {
    pub fn chunks(self, max_bytes: usize) -> Result<TensorChunks<'data>> {
        if max_bytes == 0 {
            bail!("tensor staging chunk size must be greater than zero");
        }
        let alignment = match self.dtype {
            Dtype::BF16 => 2,
            Dtype::F32 => 4,
            dtype => bail!("unsupported staging dtype {dtype:?}"),
        };
        let chunk_bytes = max_bytes - (max_bytes % alignment);
        if chunk_bytes == 0 {
            bail!(
                "tensor staging chunk size {max_bytes} is smaller than {alignment}-byte element alignment"
            );
        }
        Ok(TensorChunks {
            data: self.bytes,
            chunk_bytes,
            offset: 0,
        })
    }
}

pub struct TensorChunk<'data> {
    pub offset_bytes: usize,
    pub bytes: &'data [u8],
}

pub struct TensorChunks<'data> {
    data: &'data [u8],
    chunk_bytes: usize,
    offset: usize,
}

impl<'data> Iterator for TensorChunks<'data> {
    type Item = TensorChunk<'data>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset == self.data.len() {
            return None;
        }
        let end = self
            .offset
            .saturating_add(self.chunk_bytes)
            .min(self.data.len());
        let chunk = TensorChunk {
            offset_bytes: self.offset,
            bytes: &self.data[self.offset..end],
        };
        self.offset = end;
        Some(chunk)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.data.len() - self.offset;
        let chunks = remaining.div_ceil(self.chunk_bytes);
        (chunks, Some(chunks))
    }
}

impl ExactSizeIterator for TensorChunks<'_> {}

impl NativeArtifact {
    pub fn open(root: impl AsRef<Path>, verification: VerificationMode) -> Result<Self> {
        let root = root.as_ref();
        let root_metadata = fs::symlink_metadata(root)
            .with_context(|| format!("failed to inspect artifact root {}", root.display()))?;
        if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
            bail!(
                "artifact root must be a real directory, not a symlink: {}",
                root.display()
            );
        }
        let root = root
            .canonicalize()
            .with_context(|| format!("failed to resolve {}", root.display()))?;
        let manifest = read_manifest(&root.join("manifest.json"))?;
        validate_manifest_identity(&manifest)?;

        let files = parse_manifest_files(&manifest)?;
        verify_manifest_files(&root, &files, verification)?;

        let decoder_dtype = CodecWeightDtype::parse(
            manifest
                .pointer("/weights/speech_decoder/dtype")
                .and_then(Value::as_str)
                .context("manifest speech-decoder dtype is missing")?
                .to_ascii_lowercase()
                .as_str(),
        )?;
        expect_manifest_string(&manifest, "/weights/voice_design/path", VOICE_PATH)?;
        expect_manifest_string(&manifest, "/weights/voice_design/dtype", "BF16")?;
        expect_manifest_string(&manifest, "/weights/speech_decoder/path", CODEC_PATH)?;
        expect_manifest_bool(&manifest, "/weights/speech_decoder/encoder_included", false)?;

        let voice_contract = voice_design_contract()?;
        let decoder_contract = speech_decoder_contract(decoder_dtype)?;
        validate_manifest_contract(&manifest, "/weights/voice_design", &voice_contract)?;
        validate_manifest_contract(&manifest, "/weights/speech_decoder", &decoder_contract)?;

        let voice = MappedCheckpoint::open(&root.join(VOICE_PATH), voice_contract)?;
        let decoder = MappedCheckpoint::open(&root.join(CODEC_PATH), decoder_contract)?;
        let revision = manifest
            .pointer("/model/revision")
            .and_then(Value::as_str)
            .context("manifest model revision is missing")?
            .to_owned();

        Ok(Self {
            root,
            revision,
            verification,
            voice,
            decoder,
            decoder_dtype,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn revision(&self) -> &str {
        &self.revision
    }

    pub const fn verification(&self) -> VerificationMode {
        self.verification
    }

    pub const fn decoder_dtype(&self) -> CodecWeightDtype {
        self.decoder_dtype
    }

    pub fn model_memory_metrics(&self) -> ModelMemoryMetrics {
        let voice_mapped_file_bytes = self.voice.mapping.len() as u64;
        let voice_tensor_payload_bytes = self.voice.contract.payload_bytes;
        let decoder_mapped_file_bytes = self.decoder.mapping.len() as u64;
        let decoder_tensor_payload_bytes = self.decoder.contract.payload_bytes;
        ModelMemoryMetrics {
            voice_mapped_file_bytes,
            voice_tensor_payload_bytes,
            decoder_mapped_file_bytes,
            decoder_tensor_payload_bytes,
            total_mapped_file_bytes: voice_mapped_file_bytes
                .checked_add(decoder_mapped_file_bytes)
                .expect("validated model file byte total fits u64"),
            total_tensor_payload_bytes: voice_tensor_payload_bytes
                .checked_add(decoder_tensor_payload_bytes)
                .expect("validated model payload byte total fits u64"),
        }
    }

    pub fn voice_tensor(&self, name: &str) -> Result<TensorRef<'_>> {
        self.voice.tensor(name)
    }

    pub fn talker_tensor(&self, name: &str) -> Result<TensorRef<'_>> {
        if !name.starts_with("talker.") || name.starts_with("talker.code_predictor.") {
            bail!("tensor {name} is not in the talker namespace");
        }
        self.voice.tensor(name)
    }

    pub fn predictor_tensor(&self, name: &str) -> Result<TensorRef<'_>> {
        if !name.starts_with("talker.code_predictor.") {
            bail!("tensor {name} is not in the code-predictor namespace");
        }
        self.voice.tensor(name)
    }

    pub fn decoder_tensor(&self, name: &str) -> Result<TensorRef<'_>> {
        if !name.starts_with("decoder.") {
            bail!("tensor {name} is not in the speech-decoder namespace");
        }
        self.decoder.tensor(name)
    }

    pub fn voice_tensor_names(&self) -> impl Iterator<Item = &str> {
        self.voice.names()
    }

    pub fn decoder_tensor_names(&self) -> impl Iterator<Item = &str> {
        self.decoder.names()
    }
}

fn read_manifest(path: &Path) -> Result<Value> {
    ensure_regular_material(path)?;
    let metadata =
        fs::metadata(path).with_context(|| format!("failed to stat {}", path.display()))?;
    if metadata.len() > MANIFEST_MAX_BYTES {
        bail!(
            "manifest exceeds {MANIFEST_MAX_BYTES} byte limit: {}",
            path.display()
        );
    }
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("invalid JSON in {}", path.display()))
}

fn validate_manifest_identity(manifest: &Value) -> Result<()> {
    expect_manifest_u64(manifest, "/schema_version", 1)?;
    expect_manifest_string(manifest, "/artifact", ARTIFACT_KIND)?;
    expect_manifest_string(manifest, "/model/id", MODEL_ID)?;
    expect_manifest_bool(manifest, "/production_material/regular_files_only", true)?;
    expect_manifest_u64(manifest, "/production_material/symlink_count", 0)?;
    let revision = manifest
        .pointer("/model/revision")
        .and_then(Value::as_str)
        .context("manifest model revision is missing")?;
    if revision.len() != 40
        || !revision
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        bail!("manifest model revision is not a lowercase 40-character commit hash");
    }
    Ok(())
}

fn parse_manifest_files(manifest: &Value) -> Result<BTreeMap<PathBuf, ManifestFile>> {
    let entries = manifest["files"]
        .as_object()
        .context("manifest files is not an object")?;
    let mut files = BTreeMap::new();
    for (encoded_path, entry) in entries {
        let relative = safe_relative_path(encoded_path)?;
        let bytes = entry["bytes"]
            .as_u64()
            .with_context(|| format!("manifest file {encoded_path} has no byte count"))?;
        let sha256 = entry["sha256"]
            .as_str()
            .with_context(|| format!("manifest file {encoded_path} has no SHA-256"))?;
        if sha256.len() != 64
            || !sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            bail!("manifest file {encoded_path} has an invalid lowercase SHA-256");
        }
        if files
            .insert(
                relative,
                ManifestFile {
                    bytes,
                    sha256: sha256.to_owned(),
                },
            )
            .is_some()
        {
            bail!("manifest contains duplicate normalized path {encoded_path}");
        }
    }
    for required in [VOICE_PATH, CODEC_PATH] {
        if !files.contains_key(Path::new(required)) {
            bail!("manifest does not include required file {required}");
        }
    }
    Ok(files)
}

fn verify_manifest_files(
    root: &Path,
    files: &BTreeMap<PathBuf, ManifestFile>,
    verification: VerificationMode,
) -> Result<()> {
    for (relative, expected) in files {
        let path = root.join(relative);
        ensure_regular_material(&path)?;
        let actual_bytes = fs::metadata(&path)
            .with_context(|| format!("failed to stat {}", path.display()))?
            .len();
        if actual_bytes != expected.bytes {
            bail!(
                "{} size mismatch: manifest {}, actual {actual_bytes}",
                relative.display(),
                expected.bytes
            );
        }
        if verification == VerificationMode::Full {
            let file =
                File::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
            let mut reader = BufReader::with_capacity(HASH_BUFFER_BYTES, file);
            let (digest, hashed_bytes) = digest_reader(&mut reader, HASH_BUFFER_BYTES)
                .with_context(|| format!("failed to hash {}", path.display()))?;
            if hashed_bytes != actual_bytes {
                bail!("{} changed while being hashed", relative.display());
            }
            let actual_sha256 = to_hex(&digest);
            if actual_sha256 != expected.sha256 {
                bail!(
                    "{} SHA-256 mismatch: manifest {}, actual {actual_sha256}",
                    relative.display(),
                    expected.sha256
                );
            }
        }
    }
    Ok(())
}

fn validate_manifest_contract(
    manifest: &Value,
    pointer: &str,
    contract: &TensorContract,
) -> Result<()> {
    expect_manifest_u64(
        manifest,
        &format!("{pointer}/tensor_count"),
        contract.tensor_count as u64,
    )?;
    expect_manifest_u64(
        manifest,
        &format!("{pointer}/parameter_count"),
        contract.parameter_count,
    )?;
    expect_manifest_u64(
        manifest,
        &format!("{pointer}/payload_bytes"),
        contract.payload_bytes,
    )
}

fn safe_relative_path(encoded: &str) -> Result<PathBuf> {
    let path = Path::new(encoded);
    if path.as_os_str().is_empty() || path.is_absolute() {
        bail!("manifest path must be a non-empty relative path: {encoded:?}");
    }
    for component in path.components() {
        if !matches!(component, Component::Normal(_)) {
            bail!("manifest path contains an unsafe component: {encoded:?}");
        }
    }
    Ok(path.to_owned())
}

fn ensure_regular_material(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("required artifact file is missing: {}", path.display()))?;
    if metadata.file_type().is_symlink() {
        bail!(
            "artifact material must not be a symlink: {}",
            path.display()
        );
    }
    if !metadata.is_file() {
        bail!(
            "artifact material is not a regular file: {}",
            path.display()
        );
    }
    Ok(())
}

fn expect_manifest_string(root: &Value, pointer: &str, expected: &str) -> Result<()> {
    let actual = root
        .pointer(pointer)
        .and_then(Value::as_str)
        .with_context(|| format!("manifest field {pointer} is missing or not a string"))?;
    if actual != expected {
        bail!("manifest field {pointer}: expected {expected:?}, found {actual:?}");
    }
    Ok(())
}

fn expect_manifest_bool(root: &Value, pointer: &str, expected: bool) -> Result<()> {
    let actual = root
        .pointer(pointer)
        .and_then(Value::as_bool)
        .with_context(|| format!("manifest field {pointer} is missing or not a boolean"))?;
    if actual != expected {
        bail!("manifest field {pointer}: expected {expected}, found {actual}");
    }
    Ok(())
}

fn expect_manifest_u64(root: &Value, pointer: &str, expected: u64) -> Result<()> {
    let actual = root
        .pointer(pointer)
        .and_then(Value::as_u64)
        .with_context(|| {
            format!("manifest field {pointer} is missing or not an unsigned integer")
        })?;
    if actual != expected {
        bail!("manifest field {pointer}: expected {expected}, found {actual}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_material_paths_are_strict() {
        for accepted in ["model.safetensors", "speech_tokenizer/model.safetensors"] {
            assert_eq!(safe_relative_path(accepted).unwrap(), Path::new(accepted));
        }
        for rejected in [
            "",
            "/absolute",
            "../escape",
            "speech_tokenizer/../model.safetensors",
            "./model.safetensors",
        ] {
            assert!(safe_relative_path(rejected).is_err(), "{rejected}");
        }
    }

    #[test]
    fn tensor_chunks_preserve_alignment_and_offsets() {
        let bytes = [0_u8; 22];
        let tensor = TensorRef {
            name: "test",
            dtype: Dtype::BF16,
            shape: &[11],
            bytes: &bytes,
        };
        let chunks = tensor
            .chunks(9)
            .unwrap()
            .map(|chunk| (chunk.offset_bytes, chunk.bytes.len()))
            .collect::<Vec<_>>();
        assert_eq!(chunks, [(0, 8), (8, 8), (16, 6)]);
        assert!(tensor.chunks(1).is_err());
        assert!(tensor.chunks(0).is_err());
    }

    #[test]
    fn verification_mode_parser_is_strict() {
        assert_eq!(
            VerificationMode::parse("full").unwrap(),
            VerificationMode::Full
        );
        assert_eq!(
            VerificationMode::parse("contracts").unwrap(),
            VerificationMode::ContractsOnly
        );
        assert!(VerificationMode::parse("none").is_err());
    }
}
