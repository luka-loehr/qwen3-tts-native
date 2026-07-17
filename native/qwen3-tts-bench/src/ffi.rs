use std::ffi::{c_char, c_void};
use std::path::Path;
use std::ptr::NonNull;

use anyhow::{Context, Result, bail};
use libloading::Library;

pub const STATUS_OK: i32 = 0;
pub const STATUS_WOULD_BLOCK: i32 = 1;
pub const STATUS_END_OF_STREAM: i32 = 2;
pub const SAMPLE_RATE: u32 = 24_000;
pub const SAMPLES_PER_FRAME: u32 = 1_920;
const FINISH_REASON_NONE: u32 = 0;
const FINISH_REASON_CODEC_EOS: u32 = 1;
const FINISH_REASON_MAX_CODEC_FRAMES: u32 = 2;
const ERROR_CAPACITY: usize = 1_024;
const PCM_SENTINEL: i16 = i16::MIN;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    None,
    CodecEos,
    MaxCodecFrames,
}

impl FinishReason {
    fn from_raw(value: u32) -> Result<Self> {
        match value {
            FINISH_REASON_NONE => Ok(Self::None),
            FINISH_REASON_CODEC_EOS => Ok(Self::CodecEos),
            FINISH_REASON_MAX_CODEC_FRAMES => Ok(Self::MaxCodecFrames),
            unknown => bail!("request_finish_reason returned unknown value {unknown}"),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::CodecEos => "codec_eos",
            Self::MaxCodecFrames => "max_codec_frames",
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct EngineConfig {
    pub struct_size: u32,
    pub device_index: i32,
    pub max_concurrent_requests: u32,
    pub packet_frames: u32,
    pub pcm_ring_slots: u32,
    pub max_text_bytes: u32,
    pub max_instruct_bytes: u32,
    pub flags: u32,
    pub reserved: [u64; 8],
}

impl EngineConfig {
    pub fn new(max_concurrent_requests: u32, packet_frames: u32, pcm_ring_slots: u32) -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            device_index: 0,
            max_concurrent_requests,
            packet_frames,
            pcm_ring_slots,
            max_text_bytes: 64 * 1_024,
            max_instruct_bytes: 16 * 1_024,
            flags: 0,
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GenerationConfig {
    pub struct_size: u32,
    pub max_codec_frames: u32,
    pub seed: u64,
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub top_k: u32,
    pub do_sample: u32,
    pub predictor_temperature: f32,
    pub predictor_top_p: f32,
    pub predictor_top_k: u32,
    pub predictor_do_sample: u32,
    pub reserved: [u64; 8],
}

impl GenerationConfig {
    pub fn official_defaults(seed: u64, max_codec_frames: u32) -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            max_codec_frames,
            seed,
            temperature: 0.9,
            top_p: 1.0,
            repetition_penalty: 1.05,
            top_k: 50,
            do_sample: 1,
            predictor_temperature: 0.9,
            predictor_top_p: 1.0,
            predictor_top_k: 50,
            predictor_do_sample: 1,
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
struct RequestInput {
    struct_size: u32,
    language: u32,
    text_utf8: *const u8,
    text_bytes: usize,
    instruct_utf8: *const u8,
    instruct_bytes: usize,
    generation: GenerationConfig,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AudioPacket {
    pub request_id: u64,
    pub sequence: u64,
    pub first_codec_frame: u64,
    pub first_sample: u64,
    pub codec_frames: u32,
    pub sample_count: u32,
    pub sample_rate: u32,
    pub channels: u32,
    pub is_final: u32,
    pub reserved: u32,
    pub talker_gpu_microseconds: f32,
    pub codec_gpu_microseconds: f32,
    pub end_to_end_microseconds: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RequestMetrics {
    pub queue_microseconds: u64,
    pub prefill_microseconds: u64,
    pub first_codec_frame_microseconds: u64,
    pub first_audio_microseconds: u64,
    pub wall_microseconds: u64,
    pub generated_codec_frames: u64,
    pub emitted_samples: u64,
    pub emitted_packets: u64,
    pub talker_gpu_microseconds: f64,
    pub codec_gpu_microseconds: f64,
    pub peak_request_device_bytes: u64,
    pub peak_request_host_bytes: u64,
}

type AbiVersion = unsafe extern "C" fn() -> u32;
type EngineCreate = unsafe extern "C" fn(
    *const u8,
    usize,
    *const EngineConfig,
    *mut *mut c_void,
    *mut c_char,
    usize,
) -> i32;
type EngineDestroy = unsafe extern "C" fn(*mut c_void, *mut c_char, usize) -> i32;
type RequestStart = unsafe extern "C" fn(
    *mut c_void,
    *const RequestInput,
    *mut *mut c_void,
    *mut c_char,
    usize,
) -> i32;
type RequestPoll = unsafe extern "C" fn(
    *mut c_void,
    u32,
    *mut i16,
    usize,
    *mut AudioPacket,
    *mut c_char,
    usize,
) -> i32;
type RequestCancel = unsafe extern "C" fn(*mut c_void, *mut c_char, usize) -> i32;
type RequestMetricsFn =
    unsafe extern "C" fn(*const c_void, *mut RequestMetrics, *mut c_char, usize) -> i32;
type RequestFinishReason = unsafe extern "C" fn(*const c_void, *mut u32, *mut c_char, usize) -> i32;
type RequestDestroy = unsafe extern "C" fn(*mut c_void, *mut c_char, usize) -> i32;

pub struct Api {
    _library: Library,
    abi_version: AbiVersion,
    engine_create: EngineCreate,
    engine_destroy: EngineDestroy,
    request_start: RequestStart,
    request_poll: RequestPoll,
    request_cancel: RequestCancel,
    request_metrics: RequestMetricsFn,
    request_finish_reason: RequestFinishReason,
    request_destroy: RequestDestroy,
}

pub enum StartResult<'api> {
    Started(Request<'api>),
    WouldBlock,
}

pub enum PollResult {
    Packet {
        descriptor: AudioPacket,
        tail_untouched: bool,
    },
    WouldBlock,
    EndOfStream,
}

impl Api {
    pub fn load(path: &Path) -> Result<Self> {
        // SAFETY: The qualification harness keeps the library loaded for the
        // entire lifetime of every copied function pointer and opaque handle.
        let library = unsafe { Library::new(path) }
            .with_context(|| format!("failed to load runtime library {}", path.display()))?;
        // SAFETY: Symbol signatures are the versioned public ABI contract.
        let abi_version =
            unsafe { *library.get::<AbiVersion>(b"qwen3_tts_runtime_abi_version_v1\0")? };
        // SAFETY: See the versioned C header shipped with qwen3-tts-runtime.
        let engine_create =
            unsafe { *library.get::<EngineCreate>(b"qwen3_tts_engine_create_v1\0")? };
        // SAFETY: See the versioned C header shipped with qwen3-tts-runtime.
        let engine_destroy =
            unsafe { *library.get::<EngineDestroy>(b"qwen3_tts_engine_destroy_v1\0")? };
        // SAFETY: See the versioned C header shipped with qwen3-tts-runtime.
        let request_start =
            unsafe { *library.get::<RequestStart>(b"qwen3_tts_request_start_v1\0")? };
        // SAFETY: See the versioned C header shipped with qwen3-tts-runtime.
        let request_poll = unsafe { *library.get::<RequestPoll>(b"qwen3_tts_request_poll_v1\0")? };
        // SAFETY: See the versioned C header shipped with qwen3-tts-runtime.
        let request_cancel =
            unsafe { *library.get::<RequestCancel>(b"qwen3_tts_request_cancel_v1\0")? };
        // SAFETY: See the versioned C header shipped with qwen3-tts-runtime.
        let request_metrics =
            unsafe { *library.get::<RequestMetricsFn>(b"qwen3_tts_request_metrics_v1\0")? };
        // SAFETY: See the versioned C header shipped with qwen3-tts-runtime.
        let request_finish_reason = unsafe {
            *library.get::<RequestFinishReason>(b"qwen3_tts_request_finish_reason_v1\0")?
        };
        // SAFETY: See the versioned C header shipped with qwen3-tts-runtime.
        let request_destroy =
            unsafe { *library.get::<RequestDestroy>(b"qwen3_tts_request_destroy_v1\0")? };

        Ok(Self {
            _library: library,
            abi_version,
            engine_create,
            engine_destroy,
            request_start,
            request_poll,
            request_cancel,
            request_metrics,
            request_finish_reason,
            request_destroy,
        })
    }

    pub fn abi_version(&self) -> u32 {
        // SAFETY: The function has no arguments and the library remains loaded.
        unsafe { (self.abi_version)() }
    }

    pub fn create_engine<'api>(
        &'api self,
        model_root: &Path,
        config: &EngineConfig,
    ) -> Result<Engine<'api>> {
        let model_root = model_root
            .to_str()
            .context("model root is not valid UTF-8")?
            .as_bytes();
        let mut raw = std::ptr::null_mut();
        let mut error = ErrorBuffer::new();
        // SAFETY: All input pointers remain valid for the duration of the call;
        // the runtime contract copies the model-root bytes.
        let status = unsafe {
            (self.engine_create)(
                model_root.as_ptr(),
                model_root.len(),
                config,
                &mut raw,
                error.as_mut_ptr(),
                error.capacity(),
            )
        };
        ensure_status(status, STATUS_OK, &error, "engine_create")?;
        let raw = NonNull::new(raw).context("engine_create returned a null engine")?;
        Ok(Engine {
            api: self,
            raw: Some(raw),
        })
    }

    pub fn start_request<'api>(
        &'api self,
        engine: &Engine<'api>,
        language: u32,
        text: &str,
        instruct: &str,
        generation: GenerationConfig,
    ) -> Result<StartResult<'api>> {
        let input = RequestInput {
            struct_size: size_of::<RequestInput>() as u32,
            language,
            text_utf8: text.as_ptr(),
            text_bytes: text.len(),
            instruct_utf8: instruct.as_ptr(),
            instruct_bytes: instruct.len(),
            generation,
        };
        let mut raw = std::ptr::null_mut();
        let mut error = ErrorBuffer::new();
        // SAFETY: The engine is live, input pointers are valid for this call,
        // and the runtime copies all request-owned UTF-8 before returning.
        let status = unsafe {
            (self.request_start)(
                engine.raw()?.as_ptr(),
                &input,
                &mut raw,
                error.as_mut_ptr(),
                error.capacity(),
            )
        };
        if status == STATUS_WOULD_BLOCK {
            return Ok(StartResult::WouldBlock);
        }
        ensure_status(status, STATUS_OK, &error, "request_start")?;
        let raw = NonNull::new(raw).context("request_start returned a null request")?;
        Ok(StartResult::Started(Request {
            api: self,
            raw: Some(raw),
        }))
    }
}

pub struct Engine<'api> {
    api: &'api Api,
    raw: Option<NonNull<c_void>>,
}

impl Engine<'_> {
    fn raw(&self) -> Result<NonNull<c_void>> {
        self.raw.context("engine has already been destroyed")
    }

    pub fn destroy(&mut self) -> Result<()> {
        let Some(raw) = self.raw.take() else {
            return Ok(());
        };
        let mut error = ErrorBuffer::new();
        // SAFETY: The handle came from this API and is destroyed at most once.
        let status = unsafe {
            (self.api.engine_destroy)(raw.as_ptr(), error.as_mut_ptr(), error.capacity())
        };
        ensure_status(status, STATUS_OK, &error, "engine_destroy")
    }
}

impl Drop for Engine<'_> {
    fn drop(&mut self) {
        let _ = self.destroy();
    }
}

pub struct Request<'api> {
    api: &'api Api,
    raw: Option<NonNull<c_void>>,
}

impl Request<'_> {
    fn raw(&self) -> Result<NonNull<c_void>> {
        self.raw.context("request has already been destroyed")
    }

    pub fn poll(&mut self, timeout_milliseconds: u32, pcm: &mut [i16]) -> Result<PollResult> {
        pcm.fill(PCM_SENTINEL);
        let mut descriptor = AudioPacket::default();
        let mut error = ErrorBuffer::new();
        // SAFETY: The request is live and all output buffers are writable for
        // their declared capacities.
        let status = unsafe {
            (self.api.request_poll)(
                self.raw()?.as_ptr(),
                timeout_milliseconds,
                pcm.as_mut_ptr(),
                pcm.len(),
                &mut descriptor,
                error.as_mut_ptr(),
                error.capacity(),
            )
        };
        match status {
            STATUS_OK => {
                let samples = descriptor.sample_count as usize;
                if samples > pcm.len() {
                    bail!(
                        "request_poll reported {samples} samples for a {}-sample caller buffer",
                        pcm.len()
                    );
                }
                let tail_untouched = pcm[samples..].iter().all(|sample| *sample == PCM_SENTINEL);
                Ok(PollResult::Packet {
                    descriptor,
                    tail_untouched,
                })
            }
            STATUS_WOULD_BLOCK => Ok(PollResult::WouldBlock),
            STATUS_END_OF_STREAM => Ok(PollResult::EndOfStream),
            _ => {
                ensure_status(status, STATUS_OK, &error, "request_poll")?;
                unreachable!()
            }
        }
    }

    pub fn metrics(&self) -> Result<RequestMetrics> {
        let mut output = RequestMetrics::default();
        let mut error = ErrorBuffer::new();
        // SAFETY: The request remains live and output points to a correctly
        // sized writable metrics structure.
        let status = unsafe {
            (self.api.request_metrics)(
                self.raw()?.as_ptr(),
                &mut output,
                error.as_mut_ptr(),
                error.capacity(),
            )
        };
        ensure_status(status, STATUS_OK, &error, "request_metrics")?;
        Ok(output)
    }

    pub fn finish_reason(&self) -> Result<FinishReason> {
        let mut output = FINISH_REASON_NONE;
        let mut error = ErrorBuffer::new();
        // SAFETY: The request remains live and output points to a correctly
        // aligned writable u32.
        let status = unsafe {
            (self.api.request_finish_reason)(
                self.raw()?.as_ptr(),
                &mut output,
                error.as_mut_ptr(),
                error.capacity(),
            )
        };
        ensure_status(status, STATUS_OK, &error, "request_finish_reason")?;
        FinishReason::from_raw(output)
    }

    pub fn cancel(&mut self) -> Result<()> {
        let mut error = ErrorBuffer::new();
        // SAFETY: The request is live and the call does not retain pointers.
        let status = unsafe {
            (self.api.request_cancel)(self.raw()?.as_ptr(), error.as_mut_ptr(), error.capacity())
        };
        ensure_status(status, STATUS_OK, &error, "request_cancel")
    }

    pub fn destroy(&mut self) -> Result<()> {
        let Some(raw) = self.raw.take() else {
            return Ok(());
        };
        let mut error = ErrorBuffer::new();
        // SAFETY: The handle came from this API and is destroyed at most once.
        let status = unsafe {
            (self.api.request_destroy)(raw.as_ptr(), error.as_mut_ptr(), error.capacity())
        };
        ensure_status(status, STATUS_OK, &error, "request_destroy")
    }
}

impl Drop for Request<'_> {
    fn drop(&mut self) {
        if self.raw.is_some() {
            let _ = self.cancel();
            let _ = self.destroy();
        }
    }
}

struct ErrorBuffer {
    bytes: [c_char; ERROR_CAPACITY],
}

impl ErrorBuffer {
    fn new() -> Self {
        Self {
            bytes: [0; ERROR_CAPACITY],
        }
    }

    fn as_mut_ptr(&mut self) -> *mut c_char {
        self.bytes.as_mut_ptr()
    }

    const fn capacity(&self) -> usize {
        ERROR_CAPACITY
    }

    fn message(&self) -> String {
        let end = self
            .bytes
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(self.bytes.len());
        let bytes = self.bytes[..end]
            .iter()
            .map(|byte| byte.to_ne_bytes()[0])
            .collect::<Vec<_>>();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

fn ensure_status(status: i32, expected: i32, error: &ErrorBuffer, operation: &str) -> Result<()> {
    if status == expected {
        return Ok(());
    }
    let message = error.message();
    if message.is_empty() {
        bail!("{operation} failed with runtime status {status}");
    }
    bail!("{operation} failed with runtime status {status}: {message}")
}

#[cfg(test)]
mod tests {
    use super::FinishReason;

    #[test]
    fn finish_reason_mapping_rejects_unknown_abi_values() {
        assert_eq!(FinishReason::from_raw(0).unwrap(), FinishReason::None);
        assert_eq!(FinishReason::from_raw(1).unwrap(), FinishReason::CodecEos);
        assert_eq!(
            FinishReason::from_raw(2).unwrap(),
            FinishReason::MaxCodecFrames
        );
        assert!(FinishReason::from_raw(3).is_err());
    }
}
