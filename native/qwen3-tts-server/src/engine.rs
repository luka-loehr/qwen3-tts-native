use std::mem::size_of_val;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use qwen3_tts_runtime::{
    BackendError, EngineConfig, FinishReason, GenerationConfig, Language, NativeBackend, PollError,
    PollOutcome, RequestHandle, RequestInput, RequestMetrics, RuntimeStatus, Scheduler,
    SchedulerError,
};
use serde::Serialize;
use thiserror::Error;

#[derive(Clone, Debug)]
pub struct EngineSynthesisRequest {
    pub input: RequestInput,
    pub generation: GenerationConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EnginePacket {
    pub sequence: u64,
    pub first_codec_frame: u64,
    pub first_sample: u64,
    pub codec_frames: u32,
    pub sample_count: u32,
    pub sample_rate: u32,
    pub channels: u32,
    pub is_final: bool,
    pub pcm_s16le: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct EngineMetrics {
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

impl From<RequestMetrics> for EngineMetrics {
    fn from(value: RequestMetrics) -> Self {
        Self {
            queue_microseconds: value.queue_microseconds,
            prefill_microseconds: value.prefill_microseconds,
            first_codec_frame_microseconds: value.first_codec_frame_microseconds,
            first_audio_microseconds: value.first_audio_microseconds,
            wall_microseconds: value.wall_microseconds,
            generated_codec_frames: value.generated_codec_frames,
            emitted_samples: value.emitted_samples,
            emitted_packets: value.emitted_packets,
            talker_gpu_microseconds: value.talker_gpu_microseconds,
            codec_gpu_microseconds: value.codec_gpu_microseconds,
            peak_request_device_bytes: value.peak_request_device_bytes,
            peak_request_host_bytes: value.peak_request_host_bytes,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineErrorKind {
    InvalidRequest,
    Capacity,
    Cancelled,
    BackendUnavailable,
    Internal,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("{message}")]
pub struct EngineError {
    pub kind: EngineErrorKind,
    pub message: String,
}

impl EngineError {
    pub fn new(kind: EngineErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

pub enum EnginePoll {
    Packet(EnginePacket),
    WouldBlock,
    EndOfStream(EngineFinishReason),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineFinishReason {
    Stop,
    Length,
}

impl EngineFinishReason {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

pub trait SpeechRequest: Send + 'static {
    /// Polls at most one progressive native audio event.
    ///
    /// # Errors
    ///
    /// Returns a typed native cancellation, availability, request, or invariant error.
    fn poll(&self, timeout: Duration) -> Result<EnginePoll, EngineError>;

    /// Requests idempotent native cancellation.
    ///
    /// # Errors
    ///
    /// Returns an error if the native scheduler cannot accept cancellation.
    fn cancel(&self) -> Result<(), EngineError>;
    fn wait_retired(&self, timeout: Duration) -> bool;
    fn metrics(&self) -> EngineMetrics;
}

pub trait SpeechEngine: Send + Sync + 'static {
    /// Starts a validated `VoiceDesign` request.
    ///
    /// # Errors
    ///
    /// Returns a typed error when capacity is exhausted or the runtime cannot start.
    fn start(&self, request: EngineSynthesisRequest)
    -> Result<Box<dyn SpeechRequest>, EngineError>;
    fn is_ready(&self) -> bool;
}

#[derive(Clone, Debug)]
pub struct NativeEngineConfig {
    pub talker_library: PathBuf,
    pub codec_library: PathBuf,
    pub model_root: PathBuf,
    pub runtime: EngineConfig,
    pub warmup_max_codec_frames: u32,
}

pub struct NativeRuntimeEngine {
    scheduler: Mutex<Scheduler<NativeBackend>>,
    healthy: Arc<AtomicBool>,
}

impl NativeRuntimeEngine {
    /// Loads and warms the pinned native `VoiceDesign` talker and neural codec once.
    ///
    /// # Errors
    ///
    /// Returns a backend error if model validation, library loading, CUDA
    /// initialization, or scheduler construction fails.
    pub fn load(config: &NativeEngineConfig) -> Result<Self, EngineError> {
        let backend = NativeBackend::load(
            Path::new(&config.talker_library),
            Path::new(&config.codec_library),
            Path::new(&config.model_root),
            config.runtime.device_index,
        )
        .map_err(map_backend_error)?;
        let scheduler = Scheduler::new(config.runtime, backend).map_err(map_scheduler_error)?;
        let engine = Self {
            scheduler: Mutex::new(scheduler),
            healthy: Arc::new(AtomicBool::new(true)),
        };
        engine.warm_up(config.warmup_max_codec_frames)?;
        Ok(engine)
    }

    fn warm_up(&self, max_codec_frames: u32) -> Result<(), EngineError> {
        let request = self.start(EngineSynthesisRequest {
            input: RequestInput {
                text: "Ready.".to_owned(),
                instruct: "A calm, neutral adult voice.".to_owned(),
                language: Language::English,
            },
            generation: GenerationConfig {
                max_codec_frames,
                seed: 0,
                ..GenerationConfig::default()
            },
        })?;

        let packet = match request.poll(Duration::from_secs(30))? {
            EnginePoll::Packet(packet) => packet,
            EnginePoll::WouldBlock => {
                return Err(EngineError::new(
                    EngineErrorKind::BackendUnavailable,
                    "native startup warm-up timed out before first audio",
                ));
            }
            EnginePoll::EndOfStream(_) => {
                return Err(EngineError::new(
                    EngineErrorKind::Internal,
                    "native startup warm-up ended without audio",
                ));
            }
        };
        if packet.codec_frames == 0
            || packet.codec_frames > max_codec_frames
            || packet.sample_count
                != packet
                    .codec_frames
                    .saturating_mul(qwen3_tts_runtime::SAMPLES_PER_CODEC_FRAME)
            || packet.pcm_s16le.len() != packet.sample_count as usize * size_of::<i16>()
        {
            return Err(EngineError::new(
                EngineErrorKind::Internal,
                "native startup warm-up produced an invalid audio packet",
            ));
        }

        if packet.is_final {
            match request.poll(Duration::from_secs(30))? {
                EnginePoll::EndOfStream(EngineFinishReason::Length | EngineFinishReason::Stop) => {}
                EnginePoll::Packet(_) | EnginePoll::WouldBlock => {
                    return Err(EngineError::new(
                        EngineErrorKind::Internal,
                        "native startup warm-up did not terminate after its final packet",
                    ));
                }
            }
        } else {
            request.cancel()?;
        }
        if !request.wait_retired(Duration::from_secs(5)) {
            return Err(EngineError::new(
                EngineErrorKind::BackendUnavailable,
                "native startup warm-up did not retire within five seconds",
            ));
        }
        let metrics = request.metrics();
        if metrics.generated_codec_frames < u64::from(packet.codec_frames)
            || metrics.emitted_packets == 0
            || metrics.emitted_samples < u64::from(packet.sample_count)
            || metrics.first_audio_microseconds == 0
        {
            return Err(EngineError::new(
                EngineErrorKind::Internal,
                "native startup warm-up metrics did not match delivered audio",
            ));
        }
        Ok(())
    }
}

impl SpeechEngine for NativeRuntimeEngine {
    fn start(
        &self,
        request: EngineSynthesisRequest,
    ) -> Result<Box<dyn SpeechRequest>, EngineError> {
        if !self.is_ready() {
            return Err(EngineError::new(
                EngineErrorKind::BackendUnavailable,
                "native runtime is not ready",
            ));
        }
        let handle = match lock_unpoisoned(&self.scheduler)
            .start(request.input, request.generation)
            .map_err(map_scheduler_error)
        {
            Ok(handle) => handle,
            Err(error) => {
                if error.kind == EngineErrorKind::BackendUnavailable {
                    self.healthy.store(false, Ordering::Release);
                }
                return Err(error);
            }
        };
        Ok(Box::new(NativeSpeechRequest {
            handle,
            healthy: Arc::clone(&self.healthy),
        }))
    }

    fn is_ready(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }
}

struct NativeSpeechRequest {
    handle: RequestHandle,
    healthy: Arc<AtomicBool>,
}

impl SpeechRequest for NativeSpeechRequest {
    fn poll(&self, timeout: Duration) -> Result<EnginePoll, EngineError> {
        match self.handle.poll(timeout) {
            Ok(PollOutcome::Packet(packet)) => {
                let descriptor = packet.descriptor;
                let pcm = packet.pcm();
                let mut pcm_s16le = vec![0u8; size_of_val(pcm)];
                for (bytes, sample) in pcm_s16le.chunks_exact_mut(2).zip(pcm) {
                    bytes.copy_from_slice(&sample.to_le_bytes());
                }
                Ok(EnginePoll::Packet(EnginePacket {
                    sequence: descriptor.sequence,
                    first_codec_frame: descriptor.first_codec_frame,
                    first_sample: descriptor.first_sample,
                    codec_frames: descriptor.codec_frames,
                    sample_count: descriptor.sample_count,
                    sample_rate: descriptor.sample_rate,
                    channels: descriptor.channels,
                    is_final: descriptor.is_final != 0,
                    pcm_s16le,
                }))
            }
            Ok(PollOutcome::WouldBlock) => Ok(EnginePoll::WouldBlock),
            Ok(PollOutcome::EndOfStream(reason)) => match reason {
                FinishReason::CodecEos => Ok(EnginePoll::EndOfStream(EngineFinishReason::Stop)),
                FinishReason::MaxCodecFrames => {
                    Ok(EnginePoll::EndOfStream(EngineFinishReason::Length))
                }
                FinishReason::None => Err(EngineError::new(
                    EngineErrorKind::Internal,
                    "native runtime reached end-of-stream without a finish reason",
                )),
            },
            Err(PollError::Cancelled) => Err(EngineError::new(
                EngineErrorKind::Cancelled,
                "native request was cancelled",
            )),
            Err(PollError::Failed(error)) => {
                let mapped = map_backend_error(error);
                if mapped.kind == EngineErrorKind::BackendUnavailable {
                    self.healthy.store(false, Ordering::Release);
                }
                Err(mapped)
            }
        }
    }

    fn cancel(&self) -> Result<(), EngineError> {
        self.handle.cancel().map_err(map_scheduler_error)
    }

    fn wait_retired(&self, timeout: Duration) -> bool {
        let retired = self.handle.wait_retired(timeout);
        if !retired {
            self.healthy.store(false, Ordering::Release);
        }
        retired
    }

    fn metrics(&self) -> EngineMetrics {
        self.handle.metrics().into()
    }
}

fn map_scheduler_error(error: SchedulerError) -> EngineError {
    let kind = match error {
        SchedulerError::Full => EngineErrorKind::Capacity,
        SchedulerError::InvalidConfiguration(_)
        | SchedulerError::InvalidGeneration(_)
        | SchedulerError::InvalidInput(_) => EngineErrorKind::InvalidRequest,
        SchedulerError::Closed | SchedulerError::Worker(_) => EngineErrorKind::BackendUnavailable,
    };
    EngineError::new(kind, error.to_string())
}

fn map_backend_error(error: BackendError) -> EngineError {
    let kind = match error.status() {
        RuntimeStatus::InvalidArgument
        | RuntimeStatus::InvalidUtf8
        | RuntimeStatus::UnsupportedLanguage => EngineErrorKind::InvalidRequest,
        RuntimeStatus::Cancelled => EngineErrorKind::Cancelled,
        RuntimeStatus::Model | RuntimeStatus::Allocation | RuntimeStatus::Cuda => {
            EngineErrorKind::BackendUnavailable
        }
        RuntimeStatus::Ok
        | RuntimeStatus::WouldBlock
        | RuntimeStatus::EndOfStream
        | RuntimeStatus::State
        | RuntimeStatus::Internal => EngineErrorKind::Internal,
    };
    EngineError::new(kind, error.message())
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
