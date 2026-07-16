use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::Path;
use std::ptr;

use anyhow::{Context, Result, bail, ensure};
use libloading::{Library, Symbol};
use safetensors::Dtype;
use serde::Serialize;

use crate::config::ModelConfig;
use crate::prompt::{TextMode, TextSource, VoiceDesignPrompt};
use crate::tokenizer::Qwen2Tokenizer;
use crate::weights::{SafeTensorProvider, WeightProvider};

const ERROR_CAPACITY: usize = 1_024;
const CODEBOOKS: usize = 16;

type TalkerHandle = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct NativeSamplingConfig {
    do_sample: c_int,
    top_k: c_int,
    top_p: f32,
    temperature: f32,
    repetition_penalty: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct NativePrefillResult {
    first_semantic_token: u16,
    reserved: u16,
    prompt_tokens: u32,
    talker_gpu_milliseconds: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct NativeFrameResult {
    codes: [u16; CODEBOOKS],
    next_semantic_token: u16,
    reserved: u16,
    talker_position: u32,
    predictor_gpu_milliseconds: f32,
    talker_gpu_milliseconds: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct NativeMemory {
    weight_bytes: u64,
    talker_kv_bytes: u64,
    predictor_kv_bytes: u64,
    workspace_bytes: u64,
    max_sequence_length: u32,
    tensor_count: u32,
}

type Create =
    unsafe extern "C" fn(c_int, c_int, u64, *mut TalkerHandle, *mut c_char, usize) -> c_int;
type Destroy = unsafe extern "C" fn(TalkerHandle);
type UploadTensor = unsafe extern "C" fn(
    TalkerHandle,
    *const c_char,
    *const c_void,
    u64,
    c_int,
    *const u64,
    *mut c_char,
    usize,
) -> c_int;
type FinalizeWeights =
    unsafe extern "C" fn(TalkerHandle, *mut NativeMemory, *mut c_char, usize) -> c_int;
type Reset = unsafe extern "C" fn(TalkerHandle, u64, *mut c_char, usize) -> c_int;
type Prefill = unsafe extern "C" fn(
    TalkerHandle,
    *const c_int,
    *const c_int,
    c_int,
    NativeSamplingConfig,
    *mut NativePrefillResult,
    *mut c_char,
    usize,
) -> c_int;
type NextFrame = unsafe extern "C" fn(
    TalkerHandle,
    u16,
    c_int,
    NativeSamplingConfig,
    NativeSamplingConfig,
    *mut NativeFrameResult,
    *mut c_char,
    usize,
) -> c_int;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct SamplingConfig {
    pub do_sample: bool,
    pub top_k: u32,
    pub top_p: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
}

impl SamplingConfig {
    pub const fn official_talker_defaults() -> Self {
        Self {
            do_sample: true,
            top_k: 50,
            top_p: 1.0,
            temperature: 0.9,
            repetition_penalty: 1.05,
        }
    }

    pub const fn official_predictor_defaults() -> Self {
        Self {
            do_sample: true,
            top_k: 50,
            top_p: 1.0,
            temperature: 0.9,
            repetition_penalty: 1.0,
        }
    }

    pub const fn greedy() -> Self {
        Self {
            do_sample: false,
            top_k: 0,
            top_p: 1.0,
            temperature: 1.0,
            repetition_penalty: 1.0,
        }
    }

    fn native(self) -> Result<NativeSamplingConfig> {
        ensure!(self.top_k <= i32::MAX as u32, "top_k is too large");
        ensure!(
            self.top_p > 0.0 && self.top_p <= 1.0,
            "top_p must be in (0, 1]"
        );
        ensure!(self.temperature > 0.0, "temperature must be positive");
        ensure!(
            self.repetition_penalty > 0.0,
            "repetition penalty must be positive"
        );
        Ok(NativeSamplingConfig {
            do_sample: if self.do_sample { 1 } else { 0 },
            top_k: self.top_k as i32,
            top_p: self.top_p,
            temperature: self.temperature,
            repetition_penalty: self.repetition_penalty,
        })
    }
}

#[derive(Clone, Debug)]
pub struct VoiceDesignRequest {
    pub text: String,
    pub instruction: String,
    pub language: String,
    pub text_mode: TextMode,
    pub max_frames: usize,
    pub random_seed: u64,
    pub talker_sampling: SamplingConfig,
    pub predictor_sampling: SamplingConfig,
}

impl VoiceDesignRequest {
    pub fn new(
        text: impl Into<String>,
        instruction: impl Into<String>,
        language: impl Into<String>,
    ) -> Self {
        Self {
            text: text.into(),
            instruction: instruction.into(),
            language: language.into(),
            text_mode: TextMode::Streaming,
            max_frames: 512,
            random_seed: 0,
            talker_sampling: SamplingConfig::official_talker_defaults(),
            predictor_sampling: SamplingConfig::official_predictor_defaults(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GenerationSettings {
    pub max_frames: usize,
    pub random_seed: u64,
    pub talker_sampling: SamplingConfig,
    pub predictor_sampling: SamplingConfig,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct FrameTiming {
    pub talker_position: u32,
    pub predictor_gpu_milliseconds: f32,
    pub talker_gpu_milliseconds: f32,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct MemoryUsage {
    pub weight_bytes: u64,
    pub talker_kv_bytes: u64,
    pub predictor_kv_bytes: u64,
    pub workspace_bytes: u64,
    pub max_sequence_length: u32,
    pub tensor_count: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct GenerationOutput {
    /// Frame-major contiguous codec tokens. Each consecutive 16 values form
    /// codebooks zero through fifteen for one 12.5 Hz codec frame.
    pub codec_codes: Vec<u16>,
    pub frame_count: usize,
    pub ended_by_eos: bool,
    pub first_semantic_token: u16,
    pub final_semantic_token: u16,
    pub prompt_tokens: u32,
    pub prefill_talker_gpu_milliseconds: f32,
    pub frame_timings: Vec<FrameTiming>,
    pub memory: MemoryUsage,
}

pub struct NativeTalker {
    library: Library,
    handle: TalkerHandle,
    config: ModelConfig,
    tokenizer: Qwen2Tokenizer,
    memory: MemoryUsage,
}

impl NativeTalker {
    pub fn load(
        library_path: &Path,
        model_directory: &Path,
        device_index: i32,
        max_sequence_length: usize,
        random_seed: u64,
    ) -> Result<Self> {
        ensure!(
            max_sequence_length <= i32::MAX as usize,
            "max sequence length is too large"
        );
        let config = ModelConfig::load(&model_directory.join("config.json"))?;
        let tokenizer = Qwen2Tokenizer::load(model_directory)?;
        let provider = SafeTensorProvider::open(&model_directory.join("model.safetensors"))?;
        let library = unsafe { Library::new(library_path) }
            .with_context(|| format!("failed to load {}", library_path.display()))?;

        let mut error = [0 as c_char; ERROR_CAPACITY];
        let mut handle = ptr::null_mut();
        let create: Symbol<'_, Create> = unsafe { library.get(b"qwen3_tts_talker_create\0") }
            .context("missing qwen3_tts_talker_create symbol")?;
        let status = unsafe {
            create(
                device_index,
                max_sequence_length as i32,
                random_seed,
                &mut handle,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ensure_native_success(status, &error)?;
        let mut engine = Self {
            library,
            handle,
            config,
            tokenizer,
            memory: MemoryUsage {
                weight_bytes: 0,
                talker_kv_bytes: 0,
                predictor_kv_bytes: 0,
                workspace_bytes: 0,
                max_sequence_length: max_sequence_length as u32,
                tensor_count: 0,
            },
        };
        engine.upload_weights(&provider)?;
        engine.finalize_weights()?;
        Ok(engine)
    }

    pub fn memory_usage(&self) -> MemoryUsage {
        self.memory
    }

    pub fn generate(&mut self, request: VoiceDesignRequest) -> Result<GenerationOutput> {
        let prompt = VoiceDesignPrompt::tokenize(
            &self.tokenizer,
            &self.config,
            &request.text,
            &request.instruction,
            &request.language,
            request.text_mode,
        )?;
        self.generate_prompt(
            &prompt,
            GenerationSettings {
                max_frames: request.max_frames,
                random_seed: request.random_seed,
                talker_sampling: request.talker_sampling,
                predictor_sampling: request.predictor_sampling,
            },
        )
    }

    pub fn generate_prompt(
        &mut self,
        prompt: &VoiceDesignPrompt,
        settings: GenerationSettings,
    ) -> Result<GenerationOutput> {
        ensure!(settings.max_frames > 0, "max_frames must be positive");
        self.reset(settings.random_seed)?;
        let (text_ids, codec_ids) = self.native_prompt_steps(prompt)?;
        ensure!(text_ids.len() <= i32::MAX as usize, "prompt is too long");

        let prefill = self.prefill(&text_ids, &codec_ids, settings.talker_sampling.native()?)?;
        let mut current_semantic = prefill.first_semantic_token;
        let mut codec_codes = Vec::with_capacity(settings.max_frames * CODEBOOKS);
        let mut frame_timings = Vec::with_capacity(settings.max_frames);

        for frame_index in 0..settings.max_frames {
            if current_semantic == self.config.talker_config.codec_eos_token_id as u16 {
                break;
            }
            let text = prompt
                .trailing_text
                .get(frame_index)
                .copied()
                .unwrap_or(TextSource::TtsPad);
            let text_id = self.text_source_id(text)?;
            let frame = self.next_frame(
                current_semantic,
                text_id,
                settings.talker_sampling.native()?,
                settings.predictor_sampling.native()?,
            )?;
            codec_codes.extend_from_slice(&frame.codes);
            frame_timings.push(FrameTiming {
                talker_position: frame.talker_position,
                predictor_gpu_milliseconds: frame.predictor_gpu_milliseconds,
                talker_gpu_milliseconds: frame.talker_gpu_milliseconds,
            });
            current_semantic = frame.next_semantic_token;
        }

        let ended_by_eos = current_semantic == self.config.talker_config.codec_eos_token_id as u16;
        Ok(GenerationOutput {
            frame_count: codec_codes.len() / CODEBOOKS,
            codec_codes,
            ended_by_eos,
            first_semantic_token: prefill.first_semantic_token,
            final_semantic_token: current_semantic,
            prompt_tokens: prefill.prompt_tokens,
            prefill_talker_gpu_milliseconds: prefill.talker_gpu_milliseconds,
            frame_timings,
            memory: self.memory,
        })
    }

    fn upload_weights(&mut self, provider: &impl WeightProvider) -> Result<()> {
        let upload: Symbol<'_, UploadTensor> =
            unsafe { self.library.get(b"qwen3_tts_talker_upload_tensor\0") }
                .context("missing qwen3_tts_talker_upload_tensor symbol")?;
        let mut names = provider.tensor_names()?;
        names.sort_unstable();
        for name in names {
            let tensor = provider.tensor(&name)?;
            ensure!(tensor.dtype() == Dtype::BF16, "{name} is not BF16");
            ensure!(
                tensor.shape().len() <= 4,
                "{name} has more than four dimensions"
            );
            let name = CString::new(name).context("tensor name contains NUL")?;
            let shape = tensor
                .shape()
                .iter()
                .map(|dimension| *dimension as u64)
                .collect::<Vec<_>>();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                upload(
                    self.handle,
                    name.as_ptr(),
                    tensor.data().as_ptr().cast::<c_void>(),
                    tensor.data().len() as u64,
                    shape.len() as i32,
                    shape.as_ptr(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            ensure_native_success(status, &error)?;
        }
        Ok(())
    }

    fn finalize_weights(&mut self) -> Result<()> {
        let finalize: Symbol<'_, FinalizeWeights> =
            unsafe { self.library.get(b"qwen3_tts_talker_finalize_weights\0") }
                .context("missing qwen3_tts_talker_finalize_weights symbol")?;
        let mut output = NativeMemory::default();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let status = unsafe { finalize(self.handle, &mut output, error.as_mut_ptr(), error.len()) };
        ensure_native_success(status, &error)?;
        self.memory = MemoryUsage {
            weight_bytes: output.weight_bytes,
            talker_kv_bytes: output.talker_kv_bytes,
            predictor_kv_bytes: output.predictor_kv_bytes,
            workspace_bytes: output.workspace_bytes,
            max_sequence_length: output.max_sequence_length,
            tensor_count: output.tensor_count,
        };
        Ok(())
    }

    fn reset(&mut self, random_seed: u64) -> Result<()> {
        let reset: Symbol<'_, Reset> = unsafe { self.library.get(b"qwen3_tts_talker_reset\0") }
            .context("missing qwen3_tts_talker_reset symbol")?;
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let status = unsafe { reset(self.handle, random_seed, error.as_mut_ptr(), error.len()) };
        ensure_native_success(status, &error)
    }

    fn prefill(
        &mut self,
        text_ids: &[i32],
        codec_ids: &[i32],
        sampling: NativeSamplingConfig,
    ) -> Result<NativePrefillResult> {
        let prefill: Symbol<'_, Prefill> =
            unsafe { self.library.get(b"qwen3_tts_talker_prefill\0") }
                .context("missing qwen3_tts_talker_prefill symbol")?;
        let mut output = NativePrefillResult::default();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let status = unsafe {
            prefill(
                self.handle,
                text_ids.as_ptr(),
                codec_ids.as_ptr(),
                text_ids.len() as i32,
                sampling,
                &mut output,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ensure_native_success(status, &error)?;
        Ok(output)
    }

    fn next_frame(
        &mut self,
        semantic_token: u16,
        text_token: i32,
        talker_sampling: NativeSamplingConfig,
        predictor_sampling: NativeSamplingConfig,
    ) -> Result<NativeFrameResult> {
        let next: Symbol<'_, NextFrame> =
            unsafe { self.library.get(b"qwen3_tts_talker_next_frame\0") }
                .context("missing qwen3_tts_talker_next_frame symbol")?;
        let mut output = NativeFrameResult::default();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let status = unsafe {
            next(
                self.handle,
                semantic_token,
                text_token,
                talker_sampling,
                predictor_sampling,
                &mut output,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ensure_native_success(status, &error)?;
        Ok(output)
    }

    fn native_prompt_steps(&self, prompt: &VoiceDesignPrompt) -> Result<(Vec<i32>, Vec<i32>)> {
        let mut text = Vec::with_capacity(prompt.prefill.len());
        let mut codec = Vec::with_capacity(prompt.prefill.len());
        for step in &prompt.prefill {
            text.push(match step.text {
                Some(source) => self.text_source_id(source)?,
                None => -1,
            });
            codec.push(match step.codec {
                Some(token) => i32::try_from(token).context("codec token exceeds i32")?,
                None => -1,
            });
        }
        Ok((text, codec))
    }

    fn text_source_id(&self, source: TextSource) -> Result<i32> {
        let token = match source {
            TextSource::Token(token) => token,
            TextSource::TtsBos => self.config.tts_bos_token_id,
            TextSource::TtsEos => self.config.tts_eos_token_id,
            TextSource::TtsPad => self.config.tts_pad_token_id,
        };
        i32::try_from(token).context("text token exceeds i32")
    }
}

impl Drop for NativeTalker {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Ok(destroy) = unsafe { self.library.get::<Destroy>(b"qwen3_tts_talker_destroy\0") } {
            unsafe { destroy(self.handle) };
        }
        self.handle = ptr::null_mut();
    }
}

fn ensure_native_success(status: i32, error: &[c_char]) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
    bail!("native talker failed with status {status}: {message}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_layouts_match_the_c_abi() {
        assert_eq!(std::mem::size_of::<NativeSamplingConfig>(), 20);
        assert_eq!(std::mem::size_of::<NativePrefillResult>(), 12);
        assert_eq!(std::mem::size_of::<NativeFrameResult>(), 48);
        assert_eq!(std::mem::size_of::<NativeMemory>(), 40);
    }

    #[test]
    fn greedy_sampling_is_deterministic() {
        let sampling = SamplingConfig::greedy();
        assert!(!sampling.do_sample);
        assert_eq!(sampling.top_p, 1.0);
    }
}
