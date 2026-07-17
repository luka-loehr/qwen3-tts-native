use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fmt;
use std::marker::PhantomData;
use std::path::Path;
use std::ptr;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
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
pub const STATUS_INVALID_ARGUMENT: i32 = -1;
pub const STATUS_CUDA: i32 = -2;
pub const STATUS_STATE: i32 = -3;
pub const STATUS_ALLOCATION: i32 = -4;
pub const STATUS_MODEL: i32 = -5;
const CODEBOOKS: usize = 16;
const MIN_SESSION_CAPACITY: usize = 16;
const SESSION_CAPACITY_BLOCK: usize = 32;
const MAX_POOLED_SESSIONS: usize = 8;

#[derive(Debug)]
pub struct NativeTalkerStatusError {
    status: i32,
    message: String,
}

impl NativeTalkerStatusError {
    pub fn status(&self) -> i32 {
        self.status
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for NativeTalkerStatusError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "native talker failed with status {}: {}",
            self.status, self.message
        )
    }
}

impl std::error::Error for NativeTalkerStatusError {}

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
#[derive(Clone, Copy, Debug)]
struct NativeDeviceFrameViewV2 {
    struct_size: u32,
    code_count: u32,
    device_codes: *const u16,
    ready_event: *mut c_void,
    lease_id: u64,
    device_index: c_int,
    reserved: u32,
}

impl Default for NativeDeviceFrameViewV2 {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            code_count: 0,
            device_codes: ptr::null(),
            ready_event: ptr::null_mut(),
            lease_id: 0,
            device_index: -1,
            reserved: 0,
        }
    }
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
type SessionReset = unsafe extern "C" fn(SessionHandle, u64, *mut c_char, usize) -> c_int;
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
type TalkerAbiVersion = unsafe extern "C" fn() -> u32;
type SessionNextFrameBeginV2 = unsafe extern "C" fn(
    SessionHandle,
    c_int,
    NativeSamplingConfig,
    NativeSamplingConfig,
    *mut NativeDeviceFrameViewV2,
    *mut c_char,
    usize,
) -> c_int;
type SessionNextFrameFinishV2 = unsafe extern "C" fn(
    SessionHandle,
    u64,
    *mut c_void,
    *mut u16,
    *mut NativeFrameInfo,
    *mut c_char,
    usize,
) -> c_int;

#[derive(Clone, Copy)]
struct DeviceFrameApiV2 {
    begin: SessionNextFrameBeginV2,
    finish: SessionNextFrameFinishV2,
}

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
    /// Hard safety limit. The native session allocates only enough capacity for
    /// the prompt plus `max_frames`, rounded to a small reusable size class.
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
    /// Hard safety limit rather than an unconditional KV-cache allocation.
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
pub struct FinishedDeviceFrame {
    pub frame_index: usize,
    pub next_semantic_token: u16,
    pub talker_position: u32,
    pub predictor_gpu_milliseconds: f32,
    pub talker_gpu_milliseconds: f32,
    pub ended_by_eos: bool,
}

/// One in-flight Talker frame whose codec codes remain owned by the native
/// session. The frame must be finished before the session can be used again.
#[must_use = "an in-flight device frame must be handed to the codec and finished"]
pub struct PendingTalkerFrame<'a> {
    session: &'a mut NativeTalkerSession,
    view: NativeDeviceFrameViewV2,
    finished: bool,
    not_send_or_sync: PhantomData<Rc<()>>,
}

impl PendingTalkerFrame<'_> {
    pub fn code_count(&self) -> usize {
        self.view.code_count as usize
    }

    pub fn device_index(&self) -> i32 {
        self.view.device_index
    }

    pub fn lease_id(&self) -> u64 {
        self.view.lease_id
    }

    /// Returns borrowed CUDA values for a consumer that guarantees the
    /// producer-ready wait and records a completion event after its final read.
    ///
    /// # Safety
    ///
    /// The caller must not dereference, free, overwrite, record, or destroy
    /// either value. Both values are valid only while this pending frame lives.
    pub unsafe fn raw_device_parts(&self) -> (*const u16, *mut c_void) {
        (self.view.device_codes, self.view.ready_event)
    }

    pub fn finish_without_consumer(mut self) -> Result<FinishedDeviceFrame> {
        self.complete(ptr::null_mut())
    }

    /// Completes the frame after a CUDA consumer has recorded its final read.
    ///
    /// # Safety
    ///
    /// `consumer_done_event` must be a valid, already-recorded CUDA event on
    /// `device_index()`, and its producer stream must encompass every access to
    /// the borrowed frame-code pointer.
    pub unsafe fn finish_with_consumer_event(
        mut self,
        consumer_done_event: *mut c_void,
    ) -> Result<FinishedDeviceFrame> {
        ensure!(
            !consumer_done_event.is_null(),
            "consumer completion event must not be null"
        );
        self.complete(consumer_done_event)
    }

    fn complete(&mut self, consumer_done_event: *mut c_void) -> Result<FinishedDeviceFrame> {
        let (next_semantic_token, native) = self
            .session
            .finish_device_native_raw(self.view.lease_id, consumer_done_event)?;
        self.finished = true;
        let output = self
            .session
            .commit_finished_device_frame(next_semantic_token, native)?;
        Ok(output)
    }
}

impl Drop for PendingTalkerFrame<'_> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let _ = self
            .session
            .finish_device_native_raw(self.view.lease_id, ptr::null_mut());
        self.session.recyclable = false;
        self.session.end_reason = Some(SessionEndReason::Cancelled);
        self.finished = true;
    }
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
    pub session_acquire_wall_milliseconds: f64,
    pub session_create_wall_milliseconds: f64,
    pub session_reset_wall_milliseconds: f64,
    pub prefill_wall_milliseconds: f64,
    pub session_pool_hit: bool,
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
    talker_abi_version: u32,
    session_next_frame: SessionNextFrame,
    device_frame_api: Option<DeviceFrameApiV2>,
    config: ModelConfig,
    tokenizer: Qwen2Tokenizer,
    memory: SharedModelMemory,
    session_pool: Mutex<Vec<PooledSession>>,
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
    recyclable: bool,
}

struct PooledSession {
    handle: SessionHandle,
    memory: SessionMemory,
}

struct AcquiredSession {
    pooled: PooledSession,
    pool_hit: bool,
    create_wall_milliseconds: f64,
    reset_wall_milliseconds: f64,
}

// SAFETY: a pooled handle is uniquely owned by the pool while inactive. It is
// moved out before reset or inference and never accessed through two threads.
unsafe impl Send for PooledSession {}

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

        let talker_abi_version = {
            let version: Symbol<'_, TalkerAbiVersion> =
                unsafe { library.get(b"qwen3_tts_talker_abi_version\0") }
                    .context("missing qwen3_tts_talker_abi_version symbol")?;
            unsafe { version() }
        };
        ensure!(
            talker_abi_version >= 1,
            "unsupported talker ABI version {talker_abi_version}"
        );
        let session_next_frame = {
            let next: Symbol<'_, SessionNextFrame> =
                unsafe { library.get(b"qwen3_tts_session_next_frame\0") }
                    .context("missing qwen3_tts_session_next_frame symbol")?;
            *next
        };
        let device_frame_api = if talker_abi_version >= 2 {
            let begin: Symbol<'_, SessionNextFrameBeginV2> =
                unsafe { library.get(b"qwen3_tts_session_next_frame_begin_v2\0") }
                    .context("talker ABI v2 is missing qwen3_tts_session_next_frame_begin_v2")?;
            let finish: Symbol<'_, SessionNextFrameFinishV2> =
                unsafe { library.get(b"qwen3_tts_session_next_frame_finish_v2\0") }
                    .context("talker ABI v2 is missing qwen3_tts_session_next_frame_finish_v2")?;
            Some(DeviceFrameApiV2 {
                begin: *begin,
                finish: *finish,
            })
        } else {
            None
        };

        let mut error = [0 as c_char; ERROR_CAPACITY];
        let mut handle = ptr::null_mut();
        let create: Symbol<'_, ModelCreate> = unsafe { library.get(b"qwen3_tts_model_create\0") }
            .context("missing qwen3_tts_model_create symbol")?;
        let status = unsafe { create(device_index, &mut handle, error.as_mut_ptr(), error.len()) };
        ensure_native_success(status, &error)?;
        let mut model = Self {
            library,
            handle,
            talker_abi_version,
            session_next_frame,
            device_frame_api,
            config,
            tokenizer,
            memory: SharedModelMemory {
                weight_bytes: 0,
                tensor_count: 0,
                device_index,
            },
            session_pool: Mutex::new(Vec::new()),
        };
        model.upload_weights(&provider)?;
        model.finalize_weights()?;
        Ok(Arc::new(model))
    }

    pub fn shared_memory(&self) -> SharedModelMemory {
        self.memory
    }

    pub fn talker_abi_version(&self) -> u32 {
        self.talker_abi_version
    }

    pub fn supports_device_frames(&self) -> bool {
        self.device_frame_api.is_some()
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
        let capacity = session_capacity(
            text_ids.len(),
            settings.max_frames,
            settings.max_sequence_length,
        )?;
        let talker_sampling = settings.talker_sampling.native()?;
        let predictor_sampling = settings.predictor_sampling.native()?;
        let acquire_started = Instant::now();
        let acquired = self.acquire_session(capacity, settings.random_seed)?;
        let session_acquire_wall_milliseconds = acquire_started.elapsed().as_secs_f64() * 1_000.0;

        let codec_eos_token = self.config.talker_config.codec_eos_token_id as u16;
        let mut session = NativeTalkerSession {
            model: Arc::clone(self),
            handle: acquired.pooled.handle,
            memory: acquired.pooled.memory,
            trailing_text: prompt.trailing_text.clone(),
            current_semantic: 0,
            codec_eos_token,
            frames_emitted: 0,
            max_frames: settings.max_frames,
            talker_sampling,
            predictor_sampling,
            prefill: SessionPrefillInfo {
                first_semantic_token: 0,
                prompt_tokens: 0,
                talker_gpu_milliseconds: 0.0,
            },
            start_timing: SessionStartTiming {
                tokenize_wall_milliseconds: 0.0,
                prompt_plan_wall_milliseconds,
                session_acquire_wall_milliseconds,
                session_create_wall_milliseconds: acquired.create_wall_milliseconds,
                session_reset_wall_milliseconds: acquired.reset_wall_milliseconds,
                prefill_wall_milliseconds: 0.0,
                session_pool_hit: acquired.pool_hit,
            },
            end_reason: None,
            recyclable: false,
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
        session.recyclable = true;
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

    fn acquire_session(&self, capacity: usize, random_seed: u64) -> Result<AcquiredSession> {
        let pooled = {
            let mut pool = self
                .session_pool
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            smallest_adequate_session_index(&pool, capacity).map(|index| pool.swap_remove(index))
        };

        if let Some(pooled) = pooled {
            let reset_started = Instant::now();
            let reset: Symbol<'_, SessionReset> =
                match unsafe { self.library.get(b"qwen3_tts_session_reset\0") } {
                    Ok(reset) => reset,
                    Err(error) => {
                        self.destroy_session(pooled.handle);
                        return Err(error).context("missing qwen3_tts_session_reset symbol");
                    }
                };
            let mut native_error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                reset(
                    pooled.handle,
                    random_seed,
                    native_error.as_mut_ptr(),
                    native_error.len(),
                )
            };
            if let Err(error) = ensure_native_success(status, &native_error) {
                self.destroy_session(pooled.handle);
                return Err(error).context("failed to reset a pooled talker session");
            }
            return Ok(AcquiredSession {
                pooled,
                pool_hit: true,
                create_wall_milliseconds: 0.0,
                reset_wall_milliseconds: reset_started.elapsed().as_secs_f64() * 1_000.0,
            });
        }

        let create_started = Instant::now();
        let create: Symbol<'_, SessionCreate> =
            unsafe { self.library.get(b"qwen3_tts_session_create\0") }
                .context("missing qwen3_tts_session_create symbol")?;
        let mut handle = ptr::null_mut();
        let mut native_memory = NativeSessionMemory::default();
        let mut native_error = [0 as c_char; ERROR_CAPACITY];
        let status = unsafe {
            create(
                self.handle,
                capacity as i32,
                random_seed,
                &mut handle,
                &mut native_memory,
                native_error.as_mut_ptr(),
                native_error.len(),
            )
        };
        ensure_native_success(status, &native_error)?;
        ensure!(
            !handle.is_null(),
            "native session creation returned a null handle"
        );
        Ok(AcquiredSession {
            pooled: PooledSession {
                handle,
                memory: SessionMemory {
                    talker_kv_bytes: native_memory.talker_kv_bytes,
                    predictor_kv_bytes: native_memory.predictor_kv_bytes,
                    workspace_bytes: native_memory.workspace_bytes,
                    max_sequence_length: native_memory.max_sequence_length,
                },
            },
            pool_hit: false,
            create_wall_milliseconds: create_started.elapsed().as_secs_f64() * 1_000.0,
            reset_wall_milliseconds: 0.0,
        })
    }

    fn recycle_session(&self, pooled: PooledSession) {
        let mut pool = self
            .session_pool
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if pool.len() < MAX_POOLED_SESSIONS {
            pool.push(pooled);
            return;
        }
        drop(pool);
        self.destroy_session(pooled.handle);
    }

    fn destroy_session(&self, handle: SessionHandle) {
        if handle.is_null() {
            return;
        }
        if let Ok(destroy) = unsafe {
            self.library
                .get::<SessionDestroy>(b"qwen3_tts_session_destroy\0")
        } {
            unsafe { destroy(handle) };
        }
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

fn smallest_adequate_session_index(
    pool: &[PooledSession],
    required_capacity: usize,
) -> Option<usize> {
    pool.iter()
        .enumerate()
        .filter(|(_, entry)| entry.memory.max_sequence_length as usize >= required_capacity)
        .min_by_key(|(_, entry)| entry.memory.max_sequence_length)
        .map(|(index, _)| index)
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
        let native_result = self.next_native(
            self.current_semantic,
            text_id,
            self.talker_sampling,
            self.predictor_sampling,
        );
        let (codes, next_semantic_token, native) = match native_result {
            Ok(frame) => frame,
            Err(error) => {
                self.recyclable = false;
                return Err(error);
            }
        };
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

    pub fn supports_device_frames(&self) -> bool {
        self.model.supports_device_frames()
    }

    pub fn begin_device_frame(&mut self) -> Result<Option<PendingTalkerFrame<'_>>> {
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

        let api = self
            .model
            .device_frame_api
            .context("native talker does not support device-frame leases")?;
        let text = self
            .trailing_text
            .get(self.frames_emitted)
            .copied()
            .unwrap_or(TextSource::TtsPad);
        let text_id = self.model.text_source_id(text)?;
        let mut view = NativeDeviceFrameViewV2::default();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let status = unsafe {
            (api.begin)(
                self.handle,
                text_id,
                self.talker_sampling,
                self.predictor_sampling,
                &mut view,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if let Err(error) = ensure_native_success(status, &error) {
            self.recyclable = false;
            return Err(error).context("failed to begin a native device frame");
        }
        let validation = (|| -> Result<()> {
            ensure!(
                view.struct_size as usize == std::mem::size_of::<NativeDeviceFrameViewV2>(),
                "native device-frame view has an invalid size"
            );
            ensure!(
                view.code_count as usize == CODEBOOKS,
                "native device-frame view has an invalid code count"
            );
            ensure!(
                !view.device_codes.is_null() && !view.ready_event.is_null(),
                "native device-frame view returned a null CUDA value"
            );
            ensure!(
                view.lease_id != 0,
                "native device-frame view returned lease zero"
            );
            ensure!(
                view.device_index == self.model.memory.device_index,
                "native device-frame view returned the wrong CUDA device"
            );
            ensure!(
                view.reserved == 0,
                "native device-frame view returned a nonzero reserved field"
            );
            Ok(())
        })();
        if let Err(error) = validation {
            self.recyclable = false;
            return Err(error);
        }
        Ok(Some(PendingTalkerFrame {
            session: self,
            view,
            finished: false,
            not_send_or_sync: PhantomData,
        }))
    }

    fn finish_device_native_raw(
        &mut self,
        lease_id: u64,
        consumer_done_event: *mut c_void,
    ) -> Result<(u16, NativeFrameInfo)> {
        let api = self
            .model
            .device_frame_api
            .context("native talker does not support device-frame leases")?;
        let mut next_semantic_token = 0_u16;
        let mut frame_info = NativeFrameInfo::default();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let status = unsafe {
            (api.finish)(
                self.handle,
                lease_id,
                consumer_done_event,
                &mut next_semantic_token,
                &mut frame_info,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if let Err(error) = ensure_native_success(status, &error) {
            self.recyclable = false;
            return Err(error).context("failed to finish a native device frame");
        }
        Ok((next_semantic_token, frame_info))
    }

    fn commit_finished_device_frame(
        &mut self,
        next_semantic_token: u16,
        native: NativeFrameInfo,
    ) -> Result<FinishedDeviceFrame> {
        let ended_by_eos = next_semantic_token == self.codec_eos_token;
        if ended_by_eos != (native.ended_by_eos != 0) {
            self.recyclable = false;
            bail!("native EOS flag disagrees with the semantic token");
        }
        let frame_index = self.frames_emitted;
        self.frames_emitted += 1;
        self.current_semantic = next_semantic_token;
        if ended_by_eos {
            self.end_reason = Some(SessionEndReason::CodecEos);
        } else if self.frames_emitted >= self.max_frames {
            self.end_reason = Some(SessionEndReason::MaxFrames);
        }
        Ok(FinishedDeviceFrame {
            frame_index,
            next_semantic_token,
            talker_position: native.talker_position,
            predictor_gpu_milliseconds: native.predictor_gpu_milliseconds,
            talker_gpu_milliseconds: native.talker_gpu_milliseconds,
            ended_by_eos,
        })
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
        let mut codes = [0_u16; CODEBOOKS];
        let mut next_semantic_token = 0_u16;
        let mut frame_info = NativeFrameInfo::default();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let status = unsafe {
            (self.model.session_next_frame)(
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
        let pooled = PooledSession {
            handle: std::mem::replace(&mut self.handle, ptr::null_mut()),
            memory: self.memory,
        };
        if self.recyclable {
            self.model.recycle_session(pooled);
        } else {
            self.model.destroy_session(pooled.handle);
        }
    }
}

impl Drop for NativeTalkerModel {
    fn drop(&mut self) {
        let pool = self
            .session_pool
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let pooled_handles: Vec<_> = pool.drain(..).map(|entry| entry.handle).collect();
        for handle in pooled_handles {
            self.destroy_session(handle);
        }
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

fn session_capacity(
    prompt_tokens: usize,
    max_frames: usize,
    max_sequence_length: usize,
) -> Result<usize> {
    ensure!(
        max_sequence_length >= MIN_SESSION_CAPACITY,
        "max_sequence_length must be at least {MIN_SESSION_CAPACITY}"
    );
    let required = prompt_tokens
        .checked_add(max_frames)
        .context("prompt and frame count overflow the session capacity")?;
    ensure!(
        required <= max_sequence_length,
        "request needs {required} talker positions ({prompt_tokens} prompt tokens + \
         {max_frames} codec frames), exceeding max_sequence_length={max_sequence_length}"
    );
    let rounded = required
        .max(MIN_SESSION_CAPACITY)
        .checked_add(SESSION_CAPACITY_BLOCK - 1)
        .context("session capacity overflow")?
        / SESSION_CAPACITY_BLOCK
        * SESSION_CAPACITY_BLOCK;
    Ok(rounded.min(max_sequence_length))
}

fn ensure_native_success(status: i32, error: &[c_char]) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let message = unsafe { CStr::from_ptr(error.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    Err(NativeTalkerStatusError { status, message }.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pooled_session(capacity: u32) -> PooledSession {
        PooledSession {
            handle: ptr::null_mut(),
            memory: SessionMemory {
                talker_kv_bytes: 0,
                predictor_kv_bytes: 0,
                workspace_bytes: 0,
                max_sequence_length: capacity,
            },
        }
    }

    #[test]
    fn ffi_layouts_match_the_c_abi() {
        assert_eq!(std::mem::size_of::<NativeSamplingConfig>(), 20);
        assert_eq!(std::mem::size_of::<NativePrefillResult>(), 12);
        assert_eq!(std::mem::size_of::<NativeFrameInfo>(), 16);
        assert_eq!(std::mem::size_of::<NativeModelMemory>(), 16);
        assert_eq!(std::mem::size_of::<NativeSessionMemory>(), 32);
        assert_eq!(std::mem::size_of::<NativeStateInfo>(), 40);
        assert_eq!(std::mem::size_of::<NativeDeviceFrameViewV2>(), 40);
        assert_eq!(std::mem::align_of::<NativeDeviceFrameViewV2>(), 8);
    }

    #[test]
    fn native_status_errors_remain_typed_through_anyhow() {
        let mut message = [0 as c_char; ERROR_CAPACITY];
        for (destination, source) in message.iter_mut().zip(b"CUDA launch failed\0") {
            *destination = *source as c_char;
        }
        let error = ensure_native_success(STATUS_CUDA, &message).unwrap_err();
        let native = error
            .downcast_ref::<NativeTalkerStatusError>()
            .expect("native status error must remain downcastable");
        assert_eq!(native.status(), STATUS_CUDA);
        assert_eq!(native.message(), "CUDA launch failed");
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

    #[test]
    fn session_capacity_tracks_prompt_and_requested_frames() {
        assert_eq!(session_capacity(30, 2, 1_024).unwrap(), 32);
        assert_eq!(session_capacity(32, 40, 1_024).unwrap(), 96);
        assert_eq!(session_capacity(30, 256, 1_024).unwrap(), 288);
    }

    #[test]
    fn session_capacity_respects_a_non_aligned_hard_limit() {
        assert_eq!(session_capacity(16, 1, 17).unwrap(), 17);
    }

    #[test]
    fn session_capacity_rejects_truncation() {
        let error = session_capacity(32, 40, 64).unwrap_err().to_string();
        assert!(error.contains("request needs 72 talker positions"));
    }

    #[test]
    fn session_capacity_rejects_integer_overflow() {
        assert!(session_capacity(usize::MAX, 1, usize::MAX).is_err());
    }

    #[test]
    fn pooled_session_reuse_selects_the_smallest_adequate_capacity() {
        let pool = vec![
            pooled_session(64),
            pooled_session(256),
            pooled_session(128),
            pooled_session(192),
        ];
        assert_eq!(smallest_adequate_session_index(&pool, 100), Some(2));
        assert_eq!(smallest_adequate_session_index(&pool, 129), Some(3));
        assert_eq!(smallest_adequate_session_index(&pool, 256), Some(1));
    }

    #[test]
    fn pooled_session_reuse_rejects_only_undersized_sessions() {
        let pool = vec![pooled_session(32), pooled_session(64)];
        assert_eq!(smallest_adequate_session_index(&pool, 65), None);
        assert_eq!(smallest_adequate_session_index(&[], 1), None);
    }
}
