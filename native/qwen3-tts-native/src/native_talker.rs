use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::Path;
use std::ptr;
use std::sync::Arc;
use std::time::Instant;

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

type ModelHandle = *mut c_void;
type SessionHandle = *mut c_void;

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
struct NativeFrameInfo {
    talker_position: u32,
    ended_by_eos: u32,
    predictor_gpu_milliseconds: f32,
    talker_gpu_milliseconds: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct NativeModelMemory {
    shared_weight_bytes: u64,
    tensor_count: u32,
    device_index: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct NativeSessionMemory {
    talker_kv_bytes: u64,
    predictor_kv_bytes: u64,
    workspace_bytes: u64,
    max_sequence_length: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct NativeStateInfo {
    abi_version: u32,
    phase: u32,
    talker_position: u32,
    semantic_history_count: u32,
    frames_generated: u64,
    device_sample_count: u64,
    host_sync_count: u64,
}

type ModelCreate = unsafe extern "C" fn(c_int, *mut ModelHandle, *mut c_char, usize) -> c_int;
type ModelDestroy = unsafe extern "C" fn(ModelHandle);
type ModelUploadTensor = unsafe extern "C" fn(
    ModelHandle,
    *const c_char,
    *const c_void,
    u64,
    c_int,
    *const u64,
    *mut c_char,
    usize,
) -> c_int;
type ModelFinalize =
    unsafe extern "C" fn(ModelHandle, *mut NativeModelMemory, *mut c_char, usize) -> c_int;
type SessionCreate = unsafe extern "C" fn(
    ModelHandle,
    c_int,
    u64,
    *mut SessionHandle,
    *mut NativeSessionMemory,
    *mut c_char,
    usize,
) -> c_int;
type SessionDestroy = unsafe extern "C" fn(SessionHandle);
type SessionPrefill = unsafe extern "C" fn(
    SessionHandle,
    *const c_int,
    *const c_int,
    c_int,
    NativeSamplingConfig,
    *mut NativePrefillResult,
    *mut c_char,
    usize,
) -> c_int;
type SessionNextFrame = unsafe extern "C" fn(
    SessionHandle,
    u16,
    c_int,
    NativeSamplingConfig,
    NativeSamplingConfig,
    *mut u16,
    usize,
    *mut u16,
    *mut NativeFrameInfo,
    *mut c_char,
    usize,
) -> c_int;
type SessionStateInfo =
    unsafe extern "C" fn(SessionHandle, *mut NativeStateInfo, *mut c_char, usize) -> c_int;

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
    pub max_sequence_length: usize,
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
            max_sequence_length: 1_024,
            random_seed: 0,
            talker_sampling: SamplingConfig::official_talker_defaults(),
            predictor_sampling: SamplingConfig::official_predictor_defaults(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GenerationSettings {
    pub max_frames: usize,
    pub max_sequence_length: usize,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionEndReason {
    CodecEos,
    MaxFrames,
    Cancelled,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct StreamingCodecFrame {
    pub codes: [u16; CODEBOOKS],
    pub frame_index: usize,
    pub next_semantic_token: u16,
    pub talker_position: u32,
    pub predictor_gpu_milliseconds: f32,
    pub talker_gpu_milliseconds: f32,
    pub ended_by_eos: bool,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct SessionPrefillInfo {
    pub first_semantic_token: u16,
    pub prompt_tokens: u32,
    pub talker_gpu_milliseconds: f32,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct SessionStartTiming {
    pub tokenize_wall_milliseconds: f64,
    pub prompt_plan_wall_milliseconds: f64,
    pub session_create_wall_milliseconds: f64,
    pub prefill_wall_milliseconds: f64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct SharedModelMemory {
    pub weight_bytes: u64,
    pub tensor_count: u32,
    pub device_index: i32,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct SessionMemory {
    pub talker_kv_bytes: u64,
    pub predictor_kv_bytes: u64,
    pub workspace_bytes: u64,
    pub max_sequence_length: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TalkerPhase {
    Created,
    Ready,
    Prefilled,
    Ended,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct SessionRuntimeState {
    pub abi_version: u32,
    pub phase: TalkerPhase,
    pub talker_position: u32,
    pub semantic_history_count: u32,
    pub frames_generated: u64,
    pub device_sample_count: u64,
    pub host_sync_count: u64,
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

pub struct NativeTalkerModel {
    library: Library,
    handle: ModelHandle,
    config: ModelConfig,
    tokenizer: Qwen2Tokenizer,
    memory: SharedModelMemory,
}

pub type NativeTalker = NativeTalkerModel;

pub struct NativeTalkerSession {
    model: Arc<NativeTalkerModel>,
    handle: SessionHandle,
    memory: SessionMemory,
    trailing_text: Vec<TextSource>,
    current_semantic: u16,
    codec_eos_token: u16,
    frames_emitted: usize,
    max_frames: usize,
    talker_sampling: NativeSamplingConfig,
    predictor_sampling: NativeSamplingConfig,
    prefill: SessionPrefillInfo,
    start_timing: SessionStartTiming,
    end_reason: Option<SessionEndReason>,
}

// SAFETY: after finalization the model handle exposes immutable device weights only.
// The native library is process-global and each session creates its own CUDA stream,
// cuBLAS handle, caches, workspaces, RNG, and counters.
unsafe impl Send for NativeTalkerModel {}
unsafe impl Sync for NativeTalkerModel {}

// SAFETY: a session handle is uniquely owned by this value and is never aliased.
// Moving it between threads is supported; concurrent access requires `&mut self`
// and the type intentionally does not implement Sync.
unsafe impl Send for NativeTalkerSession {}

impl NativeTalkerModel {
    pub fn load(
        library_path: &Path,
        model_directory: &Path,
        device_index: i32,
    ) -> Result<Arc<Self>> {
        let config = ModelConfig::load(&model_directory.join("config.json"))?;
        let tokenizer = Qwen2Tokenizer::load(model_directory)?;
        let provider = SafeTensorProvider::open(&model_directory.join("model.safetensors"))?;
        let library = unsafe { Library::new(library_path) }
            .with_context(|| format!("failed to load {}", library_path.display()))?;

        let mut error = [0 as c_char; ERROR_CAPACITY];
        let mut handle = ptr::null_mut();
        let create: Symbol<'_, ModelCreate> = unsafe { library.get(b"qwen3_tts_model_create\0") }
            .context("missing qwen3_tts_model_create symbol")?;
        let status = unsafe { create(device_index, &mut handle, error.as_mut_ptr(), error.len()) };
        ensure_native_success(status, &error)?;
        let mut model = Self {
            library,
            handle,
            config,
            tokenizer,
            memory: SharedModelMemory {
                weight_bytes: 0,
                tensor_count: 0,
                device_index,
            },
        };
        model.upload_weights(&provider)?;
        model.finalize_weights()?;
        Ok(Arc::new(model))
    }

    pub fn shared_memory(&self) -> SharedModelMemory {
        self.memory
    }

    pub fn generate(self: &Arc<Self>, request: VoiceDesignRequest) -> Result<GenerationOutput> {
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
                max_sequence_length: request.max_sequence_length,
                random_seed: request.random_seed,
                talker_sampling: request.talker_sampling,
                predictor_sampling: request.predictor_sampling,
            },
        )
    }

    pub fn start(self: &Arc<Self>, request: VoiceDesignRequest) -> Result<NativeTalkerSession> {
        let tokenize_started = Instant::now();
        let prompt = VoiceDesignPrompt::tokenize(
            &self.tokenizer,
            &self.config,
            &request.text,
            &request.instruction,
            &request.language,
            request.text_mode,
        )?;
        let tokenize_wall_milliseconds = tokenize_started.elapsed().as_secs_f64() * 1_000.0;
        let mut session = self.start_prompt(
            &prompt,
            GenerationSettings {
                max_frames: request.max_frames,
                max_sequence_length: request.max_sequence_length,
                random_seed: request.random_seed,
                talker_sampling: request.talker_sampling,
                predictor_sampling: request.predictor_sampling,
            },
        )?;
        session.start_timing.tokenize_wall_milliseconds = tokenize_wall_milliseconds;
        Ok(session)
    }

    pub fn start_prompt(
        self: &Arc<Self>,
        prompt: &VoiceDesignPrompt,
        settings: GenerationSettings,
    ) -> Result<NativeTalkerSession> {
        ensure!(settings.max_frames > 0, "max_frames must be positive");
        ensure!(
            settings.max_sequence_length <= i32::MAX as usize,
            "max sequence length is too large"
        );
        let prompt_plan_started = Instant::now();
        let (text_ids, codec_ids) = self.native_prompt_steps(prompt)?;
        let prompt_plan_wall_milliseconds = prompt_plan_started.elapsed().as_secs_f64() * 1_000.0;
        ensure!(text_ids.len() <= i32::MAX as usize, "prompt is too long");

        let create_started = Instant::now();
        let create: Symbol<'_, SessionCreate> =
            unsafe { self.library.get(b"qwen3_tts_session_create\0") }
                .context("missing qwen3_tts_session_create symbol")?;
        let mut handle = ptr::null_mut();
        let mut native_memory = NativeSessionMemory::default();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let status = unsafe {
            create(
                self.handle,
                settings.max_sequence_length as i32,
                settings.random_seed,
                &mut handle,
                &mut native_memory,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ensure_native_success(status, &error)?;
        let session_create_wall_milliseconds = create_started.elapsed().as_secs_f64() * 1_000.0;

        let codec_eos_token = self.config.talker_config.codec_eos_token_id as u16;
        let mut session = NativeTalkerSession {
            model: Arc::clone(self),
            handle,
            memory: SessionMemory {
                talker_kv_bytes: native_memory.talker_kv_bytes,
                predictor_kv_bytes: native_memory.predictor_kv_bytes,
                workspace_bytes: native_memory.workspace_bytes,
                max_sequence_length: native_memory.max_sequence_length,
            },
            trailing_text: prompt.trailing_text.clone(),
            current_semantic: 0,
            codec_eos_token,
            frames_emitted: 0,
            max_frames: settings.max_frames,
            talker_sampling: settings.talker_sampling.native()?,
            predictor_sampling: settings.predictor_sampling.native()?,
            prefill: SessionPrefillInfo {
                first_semantic_token: 0,
                prompt_tokens: 0,
                talker_gpu_milliseconds: 0.0,
            },
            start_timing: SessionStartTiming {
                tokenize_wall_milliseconds: 0.0,
                prompt_plan_wall_milliseconds,
                session_create_wall_milliseconds,
                prefill_wall_milliseconds: 0.0,
            },
            end_reason: None,
        };
        let prefill_started = Instant::now();
        let prefill =
            session.prefill_native(&text_ids, &codec_ids, settings.talker_sampling.native()?)?;
        session.start_timing.prefill_wall_milliseconds =
            prefill_started.elapsed().as_secs_f64() * 1_000.0;
        session.current_semantic = prefill.first_semantic_token;
        session.prefill = SessionPrefillInfo {
            first_semantic_token: prefill.first_semantic_token,
            prompt_tokens: prefill.prompt_tokens,
            talker_gpu_milliseconds: prefill.talker_gpu_milliseconds,
        };
        session.end_reason =
            (prefill.first_semantic_token == codec_eos_token).then_some(SessionEndReason::CodecEos);
        Ok(session)
    }

    pub fn generate_prompt(
        self: &Arc<Self>,
        prompt: &VoiceDesignPrompt,
        settings: GenerationSettings,
    ) -> Result<GenerationOutput> {
        let mut session = self.start_prompt(prompt, settings)?;
        let memory = MemoryUsage {
            weight_bytes: self.memory.weight_bytes,
            talker_kv_bytes: session.memory.talker_kv_bytes,
            predictor_kv_bytes: session.memory.predictor_kv_bytes,
            workspace_bytes: session.memory.workspace_bytes,
            max_sequence_length: session.memory.max_sequence_length,
            tensor_count: self.memory.tensor_count,
        };
        let prefill = session.prefill_result();
        let mut codec_codes = Vec::with_capacity(settings.max_frames * CODEBOOKS);
        let mut frame_timings = Vec::with_capacity(settings.max_frames);
        while let Some(frame) = session.next_frame()? {
            codec_codes.extend_from_slice(&frame.codes);
            frame_timings.push(FrameTiming {
                talker_position: frame.talker_position,
                predictor_gpu_milliseconds: frame.predictor_gpu_milliseconds,
                talker_gpu_milliseconds: frame.talker_gpu_milliseconds,
            });
        }
        let ended_by_eos = session.end_reason() == Some(SessionEndReason::CodecEos);
        let final_semantic_token = session.current_semantic_token();
        Ok(GenerationOutput {
            frame_count: codec_codes.len() / CODEBOOKS,
            codec_codes,
            ended_by_eos,
            first_semantic_token: prefill.first_semantic_token,
            final_semantic_token,
            prompt_tokens: prefill.prompt_tokens,
            prefill_talker_gpu_milliseconds: prefill.talker_gpu_milliseconds,
            frame_timings,
            memory,
        })
    }

    fn upload_weights(&mut self, provider: &impl WeightProvider) -> Result<()> {
        let upload: Symbol<'_, ModelUploadTensor> =
            unsafe { self.library.get(b"qwen3_tts_model_upload_tensor\0") }
                .context("missing qwen3_tts_model_upload_tensor symbol")?;
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
        let finalize: Symbol<'_, ModelFinalize> =
            unsafe { self.library.get(b"qwen3_tts_model_finalize\0") }
                .context("missing qwen3_tts_model_finalize symbol")?;
        let mut output = NativeModelMemory::default();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let status = unsafe { finalize(self.handle, &mut output, error.as_mut_ptr(), error.len()) };
        ensure_native_success(status, &error)?;
        self.memory = SharedModelMemory {
            weight_bytes: output.shared_weight_bytes,
            tensor_count: output.tensor_count,
            device_index: output.device_index,
        };
        Ok(())
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

impl NativeTalkerSession {
    pub fn prefill_result(&self) -> SessionPrefillInfo {
        self.prefill
    }

    pub fn start_timing(&self) -> SessionStartTiming {
        self.start_timing
    }

    pub fn memory_usage(&self) -> SessionMemory {
        self.memory
    }

    pub fn current_semantic_token(&self) -> u16 {
        self.current_semantic
    }

    pub fn frames_emitted(&self) -> usize {
        self.frames_emitted
    }

    pub fn end_reason(&self) -> Option<SessionEndReason> {
        self.end_reason
    }

    pub fn is_ended(&self) -> bool {
        self.end_reason.is_some()
    }

    pub fn cancel(&mut self) {
        if self.end_reason.is_none() {
            self.end_reason = Some(SessionEndReason::Cancelled);
        }
    }

    pub fn runtime_state(&self) -> Result<SessionRuntimeState> {
        let state_info: Symbol<'_, SessionStateInfo> =
            unsafe { self.model.library.get(b"qwen3_tts_session_state_info\0") }
                .context("missing qwen3_tts_session_state_info symbol")?;
        let mut native = NativeStateInfo::default();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let status =
            unsafe { state_info(self.handle, &mut native, error.as_mut_ptr(), error.len()) };
        ensure_native_success(status, &error)?;
        let phase = match native.phase {
            0 => TalkerPhase::Created,
            1 => TalkerPhase::Ready,
            2 => TalkerPhase::Prefilled,
            3 => TalkerPhase::Ended,
            value => bail!("native session returned unknown phase {value}"),
        };
        Ok(SessionRuntimeState {
            abi_version: native.abi_version,
            phase,
            talker_position: native.talker_position,
            semantic_history_count: native.semantic_history_count,
            frames_generated: native.frames_generated,
            device_sample_count: native.device_sample_count,
            host_sync_count: native.host_sync_count,
        })
    }

    pub fn next_frame(&mut self) -> Result<Option<StreamingCodecFrame>> {
        if self.end_reason.is_some() {
            return Ok(None);
        }
        if self.frames_emitted >= self.max_frames {
            self.end_reason = Some(SessionEndReason::MaxFrames);
            return Ok(None);
        }
        if self.current_semantic == self.codec_eos_token {
            self.end_reason = Some(SessionEndReason::CodecEos);
            return Ok(None);
        }

        let text = self
            .trailing_text
            .get(self.frames_emitted)
            .copied()
            .unwrap_or(TextSource::TtsPad);
        let text_id = self.model.text_source_id(text)?;
        let (codes, next_semantic_token, native) = self.next_native(
            self.current_semantic,
            text_id,
            self.talker_sampling,
            self.predictor_sampling,
        )?;
        let frame_index = self.frames_emitted;
        self.frames_emitted += 1;
        self.current_semantic = next_semantic_token;
        let ended_by_eos = self.current_semantic == self.codec_eos_token;
        ensure!(
            ended_by_eos == (native.ended_by_eos != 0),
            "native EOS flag disagrees with the semantic token"
        );
        if ended_by_eos {
            self.end_reason = Some(SessionEndReason::CodecEos);
        } else if self.frames_emitted >= self.max_frames {
            self.end_reason = Some(SessionEndReason::MaxFrames);
        }

        Ok(Some(StreamingCodecFrame {
            codes,
            frame_index,
            next_semantic_token,
            talker_position: native.talker_position,
            predictor_gpu_milliseconds: native.predictor_gpu_milliseconds,
            talker_gpu_milliseconds: native.talker_gpu_milliseconds,
            ended_by_eos,
        }))
    }

    fn prefill_native(
        &mut self,
        text_ids: &[i32],
        codec_ids: &[i32],
        sampling: NativeSamplingConfig,
    ) -> Result<NativePrefillResult> {
        let prefill: Symbol<'_, SessionPrefill> =
            unsafe { self.model.library.get(b"qwen3_tts_session_prefill\0") }
                .context("missing qwen3_tts_session_prefill symbol")?;
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

    fn next_native(
        &mut self,
        semantic_token: u16,
        text_token: i32,
        talker_sampling: NativeSamplingConfig,
        predictor_sampling: NativeSamplingConfig,
    ) -> Result<([u16; CODEBOOKS], u16, NativeFrameInfo)> {
        let next: Symbol<'_, SessionNextFrame> =
            unsafe { self.model.library.get(b"qwen3_tts_session_next_frame\0") }
                .context("missing qwen3_tts_session_next_frame symbol")?;
        let mut codes = [0_u16; CODEBOOKS];
        let mut next_semantic_token = 0_u16;
        let mut frame_info = NativeFrameInfo::default();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let status = unsafe {
            next(
                self.handle,
                semantic_token,
                text_token,
                talker_sampling,
                predictor_sampling,
                codes.as_mut_ptr(),
                codes.len(),
                &mut next_semantic_token,
                &mut frame_info,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ensure_native_success(status, &error)?;
        Ok((codes, next_semantic_token, frame_info))
    }
}

impl Drop for NativeTalkerSession {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Ok(destroy) = unsafe {
            self.model
                .library
                .get::<SessionDestroy>(b"qwen3_tts_session_destroy\0")
        } {
            unsafe { destroy(self.handle) };
        }
        self.handle = ptr::null_mut();
    }
}

impl Drop for NativeTalkerModel {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Ok(destroy) = unsafe {
            self.library
                .get::<ModelDestroy>(b"qwen3_tts_model_destroy\0")
        } {
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
        assert_eq!(std::mem::size_of::<NativeFrameInfo>(), 16);
        assert_eq!(std::mem::size_of::<NativeModelMemory>(), 16);
        assert_eq!(std::mem::size_of::<NativeSessionMemory>(), 32);
        assert_eq!(std::mem::size_of::<NativeStateInfo>(), 40);
    }

    #[test]
    fn owned_sessions_are_send_and_static() {
        fn assert_send_static<T: Send + 'static>() {}
        fn assert_send_sync_static<T: Send + Sync + 'static>() {}
        assert_send_static::<NativeTalkerSession>();
        assert_send_sync_static::<NativeTalkerModel>();
    }

    #[test]
    fn greedy_sampling_is_deterministic() {
        let sampling = SamplingConfig::greedy();
        assert!(!sampling.do_sample);
        assert_eq!(sampling.top_p, 1.0);
    }
}
