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
    descriptors: BTreeMap<String, TensorDescriptor>,
}

impl MappedCheckpoint {
    fn open(
        path: &Path,
        contract: TensorContract,
        descriptors: BTreeMap<String, TensorDescriptor>,
    ) -> Result<Self> {
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
        Ok(Self {
            mapping,
            contract,
            descriptors,
        })
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
        let descriptor = self
            .descriptors
            .get(name)
            .with_context(|| format!("{} has no descriptor for {name}", self.contract.label))?;
        Ok(TensorRef {
            name: spec.name.as_str(),
            dtype: spec.dtype,
            shape: spec.shape.as_slice(),
            sha256: &descriptor.sha256,
            arena_offset_bytes: descriptor.arena_offset_bytes,
            bytes: tensor.data(),
        })
    }

    fn names(&self) -> impl Iterator<Item = &str> {
        self.contract.tensors.keys().map(String::as_str)
    }

    fn descriptor(&self, name: &str) -> Result<&TensorDescriptor> {
        self.descriptors
            .get(name)
            .with_context(|| format!("{} has no tensor descriptor {name}", self.contract.label))
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
    /// Weight bytes copied into ordinary host heap allocations by the loader.
    /// This remains zero because tensor payloads are borrowed from read-only mappings.
    pub host_committed_weight_copy_bytes: u64,
    /// Runtime conversion scratch. Conversion is intentionally offline-only.
    pub runtime_dtype_conversion_bytes: u64,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorDtypeCode {
    Bf16 = 1,
    F32 = 2,
}

impl TensorDtypeCode {
    fn from_safetensors(dtype: Dtype) -> Result<Self> {
        match dtype {
            Dtype::BF16 => Ok(Self::Bf16),
            Dtype::F32 => Ok(Self::F32),
            other => bail!("unsupported native tensor dtype {other:?}"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorDescriptor {
    pub name: String,
    pub sha256: [u8; 32],
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub arena_offset_bytes: u64,
    pub bytes: u64,
}

/// Borrowed C-compatible tensor metadata.
///
/// `name_ptr` and `shape_ptr` remain valid only while the originating
/// [`TensorDescriptor`] is alive and is not mutated. The view never owns or
/// releases either pointer.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TensorDescriptorView {
    pub name_ptr: *const u8,
    pub name_len: usize,
    pub sha256: [u8; 32],
    pub dtype: TensorDtypeCode,
    pub rank: usize,
    pub shape_ptr: *const u64,
    pub arena_offset_bytes: u64,
    pub bytes: u64,
}

impl TensorDescriptor {
    pub fn as_c_view(&self) -> Result<TensorDescriptorView> {
        Ok(TensorDescriptorView {
            name_ptr: self.name.as_ptr(),
            name_len: self.name.len(),
            sha256: self.sha256,
            dtype: TensorDtypeCode::from_safetensors(self.dtype)?,
            rank: self.shape.len(),
            shape_ptr: self.shape.as_ptr(),
            arena_offset_bytes: self.arena_offset_bytes,
            bytes: self.bytes,
        })
    }
}

#[derive(Clone, Copy)]
pub struct TensorRef<'data> {
    pub name: &'data str,
    pub dtype: Dtype,
    pub shape: &'data [usize],
    pub sha256: &'data [u8; 32],
    pub arena_offset_bytes: u64,
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
        ensure_regular_tree(&root)?;
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
        let voice_descriptors =
            validate_manifest_tensor_index(&manifest, "/weights/voice_design", &voice_contract)?;
        let decoder_descriptors = validate_manifest_tensor_index(
            &manifest,
            "/weights/speech_decoder",
            &decoder_contract,
        )?;

        let voice =
            MappedCheckpoint::open(&root.join(VOICE_PATH), voice_contract, voice_descriptors)?;
        let decoder = MappedCheckpoint::open(
            &root.join(CODEC_PATH),
            decoder_contract,
            decoder_descriptors,
        )?;
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
            host_committed_weight_copy_bytes: 0,
            runtime_dtype_conversion_bytes: 0,
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

    pub fn voice_tensor_descriptor(&self, name: &str) -> Result<&TensorDescriptor> {
        self.voice.descriptor(name)
    }

    pub fn decoder_tensor_descriptor(&self, name: &str) -> Result<&TensorDescriptor> {
        self.decoder.descriptor(name)
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

fn validate_manifest_tensor_index(
    manifest: &Value,
    pointer: &str,
    contract: &TensorContract,
) -> Result<BTreeMap<String, TensorDescriptor>> {
    let index_pointer = format!("{pointer}/tensors");
    let entries = manifest
        .pointer(&index_pointer)
        .and_then(Value::as_array)
        .with_context(|| format!("manifest field {index_pointer} is missing or not an array"))?;
    if entries.len() != contract.tensor_count {
        bail!(
            "manifest field {index_pointer}: expected {} tensors, found {}",
            contract.tensor_count,
            entries.len()
        );
    }

    let mut descriptors = BTreeMap::new();
    let mut expected_offset = 0_u64;
    for (position, (entry, (expected_name, spec))) in
        entries.iter().zip(&contract.tensors).enumerate()
    {
        let context = format!("manifest tensor {index_pointer}[{position}]");
        let name = entry["name"]
            .as_str()
            .with_context(|| format!("{context} has no name"))?;
        if name != expected_name {
            bail!("{context} name mismatch: expected {expected_name:?}, found {name:?}");
        }
        let expected_component = tensor_component(expected_name);
        let component = entry["component"]
            .as_str()
            .with_context(|| format!("{context} has no component"))?;
        if component != expected_component {
            bail!(
                "{context} component mismatch: expected {expected_component:?}, found {component:?}"
            );
        }
        let expected_dtype = format!("{:?}", spec.dtype);
        let dtype = entry["dtype"]
            .as_str()
            .with_context(|| format!("{context} has no dtype"))?;
        if dtype != expected_dtype {
            bail!("{context} dtype mismatch: expected {expected_dtype:?}, found {dtype:?}");
        }
        let shape = entry["shape"]
            .as_array()
            .with_context(|| format!("{context} shape is not an array"))?
            .iter()
            .enumerate()
            .map(|(dimension, value)| {
                value
                    .as_u64()
                    .with_context(|| format!("{context} shape[{dimension}] is not a u64"))
            })
            .collect::<Result<Vec<_>>>()?;
        let expected_shape = spec
            .shape
            .iter()
            .map(|&dimension| u64::try_from(dimension).context("tensor dimension exceeds u64"))
            .collect::<Result<Vec<_>>>()?;
        if shape != expected_shape {
            bail!(
                "{context} shape mismatch: expected {:?}, found {shape:?}",
                spec.shape
            );
        }
        expect_tensor_u64(entry, "parameters", spec.parameters, &context)?;
        expect_tensor_u64(entry, "arena_offset_bytes", expected_offset, &context)?;
        expect_tensor_u64(entry, "bytes", spec.bytes, &context)?;
        let sha256 = parse_sha256(
            entry["sha256"]
                .as_str()
                .with_context(|| format!("{context} has no SHA-256"))?,
        )
        .with_context(|| format!("{context} has an invalid SHA-256"))?;

        let descriptor = TensorDescriptor {
            name: name.to_owned(),
            sha256,
            dtype: spec.dtype,
            shape,
            arena_offset_bytes: expected_offset,
            bytes: spec.bytes,
        };
        if descriptors.insert(name.to_owned(), descriptor).is_some() {
            bail!("{context} duplicates tensor {name}");
        }
        expected_offset = expected_offset
            .checked_add(spec.bytes)
            .context("tensor arena byte offset overflowed")?;
    }
    if expected_offset != contract.payload_bytes {
        bail!(
            "manifest tensor index covers {expected_offset} bytes, expected {}",
            contract.payload_bytes
        );
    }
    Ok(descriptors)
}

fn expect_tensor_u64(entry: &Value, field: &str, expected: u64, context: &str) -> Result<()> {
    let actual = entry[field]
        .as_u64()
        .with_context(|| format!("{context} field {field} is not a u64"))?;
    if actual != expected {
        bail!("{context} field {field}: expected {expected}, found {actual}");
    }
    Ok(())
}

fn parse_sha256(encoded: &str) -> Result<[u8; 32]> {
    if encoded.len() != 64
        || !encoded
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        bail!("SHA-256 must be exactly 64 lowercase hexadecimal characters");
    }
    let mut digest = [0_u8; 32];
    for (destination, pair) in digest.iter_mut().zip(encoded.as_bytes().chunks_exact(2)) {
        let pair = std::str::from_utf8(pair).expect("validated ASCII hexadecimal");
        *destination = u8::from_str_radix(pair, 16).expect("validated hexadecimal byte");
    }
    Ok(digest)
}

fn tensor_component(name: &str) -> &'static str {
    if name.starts_with("talker.code_predictor.") {
        "code-predictor"
    } else if name.starts_with("talker.") {
        "talker"
    } else {
        "speech-decoder"
    }
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

fn ensure_regular_tree(root: &Path) -> Result<()> {
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
                bail!("artifact tree contains a symlink: {}", path.display());
            }
            if metadata.is_dir() {
                pending.push(path);
            } else if !metadata.is_file() {
                bail!(
                    "artifact tree contains non-regular material: {}",
                    path.display()
                );
            }
        }
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
    use crate::contract::TensorSpec;
    use serde_json::json;

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
        let sha256 = [7_u8; 32];
        let tensor = TensorRef {
            name: "test",
            dtype: Dtype::BF16,
            shape: &[11],
            sha256: &sha256,
            arena_offset_bytes: 42,
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
    fn tensor_index_is_ordered_and_exact() {
        let name = "decoder.test".to_owned();
        let spec = TensorSpec {
            name: name.clone(),
            dtype: Dtype::BF16,
            shape: vec![2, 3],
            parameters: 6,
            bytes: 12,
        };
        let contract = TensorContract {
            label: "test".to_owned(),
            tensors: BTreeMap::from([(name.clone(), spec)]),
            tensor_count: 1,
            parameter_count: 6,
            payload_bytes: 12,
            exact_file_bytes: None,
        };
        let mut manifest = json!({
            "weights": {
                "test": {
                    "tensors": [{
                        "name": name,
                        "component": "speech-decoder",
                        "dtype": "BF16",
                        "shape": [2, 3],
                        "parameters": 6,
                        "arena_offset_bytes": 0,
                        "bytes": 12,
                        "sha256": "42".repeat(32),
                    }]
                }
            }
        });
        let descriptors =
            validate_manifest_tensor_index(&manifest, "/weights/test", &contract).unwrap();
        let descriptor = &descriptors["decoder.test"];
        assert_eq!(descriptor.sha256, [0x42; 32]);
        assert_eq!(descriptor.shape, [2, 3]);
        assert_eq!(descriptor.as_c_view().unwrap().dtype, TensorDtypeCode::Bf16);

        manifest["weights"]["test"]["tensors"][0]["arena_offset_bytes"] = json!(4);
        let error =
            validate_manifest_tensor_index(&manifest, "/weights/test", &contract).unwrap_err();
        assert!(error.to_string().contains("arena_offset_bytes"));
    }

    #[test]
    fn sha256_parser_rejects_noncanonical_text() {
        assert_eq!(parse_sha256(&"ab".repeat(32)).unwrap(), [0xab; 32]);
        assert!(parse_sha256(&"AB".repeat(32)).is_err());
        assert!(parse_sha256("abc").is_err());
        assert!(parse_sha256(&"xz".repeat(32)).is_err());
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

    #[test]
    fn full_verification_rejects_hash_tampering() {
        let root = TestDirectory::new("hash-tamper");
        let relative = PathBuf::from("tiny.bin");
        fs::write(root.path.join(&relative), b"verified bytes").unwrap();
        let mut files = BTreeMap::new();
        files.insert(
            relative,
            ManifestFile {
                bytes: 14,
                sha256: "00".repeat(32),
            },
        );
        let error = verify_manifest_files(&root.path, &files, VerificationMode::Full).unwrap_err();
        assert!(error.to_string().contains("SHA-256 mismatch"));
    }

    #[cfg(unix)]
    #[test]
    fn regular_material_check_rejects_symlinks() {
        use std::os::unix::fs::symlink;

        let root = TestDirectory::new("symlink");
        let target = root.path.join("target");
        let link = root.path.join("link");
        fs::write(&target, b"material").unwrap();
        symlink(&target, &link).unwrap();
        let error = ensure_regular_material(&link).unwrap_err();
        assert!(error.to_string().contains("must not be a symlink"));
    }

    #[cfg(unix)]
    #[test]
    fn artifact_tree_rejects_unlisted_symlinks() {
        use std::os::unix::fs::symlink;

        let root = TestDirectory::new("tree-symlink");
        let target = root.path.join("target");
        let link = root.path.join("unlisted-link");
        fs::write(&target, b"material").unwrap();
        symlink(&target, &link).unwrap();
        let error = ensure_regular_tree(&root.path).unwrap_err();
        assert!(error.to_string().contains("tree contains a symlink"));
    }

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn new(label: &str) -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let sequence = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "qwen3-tts-native-loader-{label}-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}
