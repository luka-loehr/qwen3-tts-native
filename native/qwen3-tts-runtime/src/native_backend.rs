use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread;
use std::time::Instant;

use anyhow::Error as AnyError;
use qwen3_tts_native::native_talker::{
    NativeTalkerModel, NativeTalkerSession, NativeTalkerStatusError,
    STATUS_ALLOCATION as TALKER_STATUS_ALLOCATION, STATUS_CUDA as TALKER_STATUS_CUDA,
    STATUS_INVALID_ARGUMENT as TALKER_STATUS_INVALID_ARGUMENT, STATUS_MODEL as TALKER_STATUS_MODEL,
    STATUS_STATE as TALKER_STATUS_STATE, SamplingConfig, VoiceDesignRequest,
};
use qwen3_tts_native_codec::{
    CODEBOOKS, DecoderWeights, MAX_PACKET_FRAMES as CODEC_MAX_PACKET_FRAMES, NativeCodecLibrary,
    NativeCodecModel, NativeCodecSession, SAMPLES_PER_FRAME, STATUS_MODEL as CODEC_STATUS_MODEL,
    STATUS_STATE as CODEC_STATUS_STATE,
};

use crate::{
    BackendError, BackendPacket, BackendRequest, BackendStarted, BackendStepInput,
    MAX_CODEC_FRAMES, RuntimeStatus, StreamingBackend,
};

const CODEC_STATUS_INVALID_ARGUMENT: i32 = -1;
const CODEC_STATUS_CUDA: i32 = -2;
const CODEC_STATUS_ALLOCATION: i32 = -4;
const TALKER_PREFETCH_FRAMES: usize = CODEC_MAX_PACKET_FRAMES;
const TALKER_PREFETCH_COMMANDS: usize = 2;

pub struct NativeBackend {
    talker: Arc<NativeTalkerModel>,
    codec: Arc<NativeCodecModel>,
}

pub struct NativeBackendSession {
    talker: TalkerProducer,
    codec: NativeCodecSession,
    first_packet: bool,
    delivered_codec_frames: u64,
    peak_request_device_bytes: u64,
    peak_request_host_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
struct GeneratedTalkerFrame {
    codes: [u16; CODEBOOKS],
    gpu_microseconds: f32,
    is_final: bool,
}

#[derive(Debug)]
enum TalkerProducerMessage {
    Frame(GeneratedTalkerFrame),
    Ended,
    Failed(BackendError),
}

#[derive(Debug)]
struct GeneratedTalkerPacket {
    frames: [[u16; CODEBOOKS]; CODEC_MAX_PACKET_FRAMES],
    frame_count: usize,
    gpu_microseconds: f32,
    is_final: bool,
}

struct TalkerProducer {
    receiver: Option<Receiver<TalkerProducerMessage>>,
    permits: Option<SyncSender<usize>>,
    cancelled: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl TalkerProducer {
    fn spawn(request_id: u64, talker: NativeTalkerSession) -> Result<Self, BackendError> {
        let (sender, receiver) = sync_channel(TALKER_PREFETCH_FRAMES);
        let (permits, permit_receiver) = sync_channel(TALKER_PREFETCH_COMMANDS);
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker_cancelled = Arc::clone(&cancelled);
        let worker = thread::Builder::new()
            .name(format!("qwen3-tts-talker-{request_id}"))
            .spawn(move || run_talker_producer(talker, sender, permit_receiver, &worker_cancelled))
            .map_err(|error| {
                BackendError::with_status(
                    RuntimeStatus::Allocation,
                    format!("failed to start talker producer thread: {error}"),
                )
            })?;
        Ok(Self {
            receiver: Some(receiver),
            permits: Some(permits),
            cancelled,
            worker: Some(worker),
        })
    }

    fn receive_packet(
        &self,
        requested_frames: usize,
    ) -> Result<GeneratedTalkerPacket, BackendError> {
        let receiver = self.receiver.as_ref().ok_or_else(|| {
            BackendError::with_status(RuntimeStatus::State, "talker producer is already closed")
        })?;
        receive_talker_packet(receiver, &self.cancelled, requested_frames)
    }

    fn prefetch(&self, frames: usize) -> Result<(), BackendError> {
        if frames == 0 || frames > CODEC_MAX_PACKET_FRAMES {
            return Err(BackendError::with_status(
                RuntimeStatus::InvalidArgument,
                "talker prefetch count is outside the packet limit",
            ));
        }
        let permits = self.permits.as_ref().ok_or_else(|| {
            BackendError::with_status(RuntimeStatus::State, "talker producer is already closed")
        })?;
        let _ = permits.send(frames);
        Ok(())
    }

    fn cancel_and_join(&mut self) -> Result<(), BackendError> {
        self.cancelled.store(true, Ordering::Release);
        self.receiver.take();
        self.permits.take();
        let Some(worker) = self.worker.take() else {
            return Ok(());
        };
        worker.join().map_err(|_| {
            BackendError::with_status(RuntimeStatus::Internal, "talker producer thread panicked")
        })
    }
}

impl Drop for TalkerProducer {
    fn drop(&mut self) {
        let _ = self.cancel_and_join();
    }
}

impl NativeBackend {
    pub fn load(
        talker_library: &Path,
        codec_library: &Path,
        model_root: &Path,
        device_index: i32,
    ) -> Result<Self, BackendError> {
        let talker =
            NativeTalkerModel::load(talker_library, model_root, device_index).map_err(|error| {
                map_talker_error("failed to load talker model", error, RuntimeStatus::Model)
            })?;
        let codec_api = Arc::new(NativeCodecLibrary::load(codec_library).map_err(|error| {
            BackendError::with_status(
                RuntimeStatus::Model,
                format!("failed to load codec library: {error}"),
            )
        })?);
        let decoder_weights = DecoderWeights::open(
            &model_root.join("speech_tokenizer/model.safetensors"),
        )
        .map_err(|error| {
            BackendError::with_status(
                RuntimeStatus::Model,
                format!("failed to open decoder weights: {error}"),
            )
        })?;
        let codec = codec_api
            .load_shared_model(device_index, &decoder_weights)
            .map_err(|message| {
                BackendError::with_status(
                    RuntimeStatus::Model,
                    format!("failed to load codec model: {message}"),
                )
            })?;
        Ok(Self { talker, codec })
    }

    fn start_session(
        talker: &Arc<NativeTalkerModel>,
        codec: &Arc<NativeCodecModel>,
        request: BackendRequest,
    ) -> Result<BackendStarted<NativeBackendSession>, BackendError> {
        let started_at = Instant::now();
        let request_id = request.id;
        let voice_request = voice_request(&request);
        let talker_session = talker.start(voice_request).map_err(|error| {
            map_talker_error(
                "failed to start talker session",
                error,
                RuntimeStatus::State,
            )
        })?;
        if talker_session.is_ended() {
            return Err(BackendError::with_status(
                RuntimeStatus::State,
                "talker ended during prefill without producing a codec frame",
            ));
        }
        let talker_memory = talker_session.memory_usage();
        let codec_session = codec.start_session().map_err(|message| {
            BackendError::with_status(
                RuntimeStatus::State,
                format!("failed to start codec session: {message}"),
            )
        })?;
        let codec_memory = codec_session.memory_info().map_err(|message| {
            BackendError::with_status(
                RuntimeStatus::State,
                format!("failed to inspect codec session memory: {message}"),
            )
        })?;
        let peak_request_device_bytes = checked_sum(
            &[
                talker_memory.talker_kv_bytes,
                talker_memory.predictor_kv_bytes,
                talker_memory.workspace_bytes,
                codec_memory.device_bytes,
            ],
            "request device memory accounting overflowed",
        )?;
        let talker_producer = TalkerProducer::spawn(request_id, talker_session)?;

        Ok(BackendStarted {
            session: NativeBackendSession {
                talker: talker_producer,
                codec: codec_session,
                first_packet: true,
                delivered_codec_frames: 0,
                peak_request_device_bytes,
                peak_request_host_bytes: codec_memory.host_pinned_bytes,
            },
            prefill_microseconds: duration_microseconds(started_at.elapsed()),
            peak_request_device_bytes,
            peak_request_host_bytes: codec_memory.host_pinned_bytes,
        })
    }

    fn step_session(
        session: &mut NativeBackendSession,
        packet_frames: u32,
        pcm: &mut [i16],
    ) -> Result<BackendPacket, BackendError> {
        let configured_packet_frames = usize::try_from(packet_frames).map_err(|_| {
            BackendError::with_status(
                RuntimeStatus::InvalidArgument,
                "packet frame count overflowed",
            )
        })?;
        let requested_frames = if session.first_packet {
            1
        } else {
            configured_packet_frames
        };
        if requested_frames == 0 || requested_frames > CODEC_MAX_PACKET_FRAMES {
            return Err(BackendError::with_status(
                RuntimeStatus::InvalidArgument,
                "packet frame count is outside the decoder limit",
            ));
        }
        let required_capacity =
            requested_frames
                .checked_mul(SAMPLES_PER_FRAME)
                .ok_or_else(|| {
                    BackendError::with_status(
                        RuntimeStatus::InvalidArgument,
                        "PCM capacity overflowed",
                    )
                })?;
        if pcm.len() < required_capacity {
            return Err(BackendError::with_status(
                RuntimeStatus::InvalidArgument,
                "caller PCM buffer is too small for the configured packet",
            ));
        }

        let talker_packet = session.talker.receive_packet(requested_frames)?;
        let sample_count = talker_packet.frame_count * SAMPLES_PER_FRAME;
        if !talker_packet.is_final {
            session.talker.prefetch(1)?;
        }
        let result = session
            .codec
            .process_into(
                &talker_packet.frames[..talker_packet.frame_count],
                talker_packet.is_final,
                &mut pcm[..sample_count],
            )
            .map_err(|(status, message)| {
                map_codec_error("codec packet generation failed", status, message)
            })?;
        if !talker_packet.is_final && configured_packet_frames > 1 {
            session.talker.prefetch(configured_packet_frames - 1)?;
        }
        let expected_first_sample = session
            .delivered_codec_frames
            .checked_mul(SAMPLES_PER_FRAME as u64)
            .ok_or_else(|| {
                BackendError::with_status(RuntimeStatus::State, "codec sample position overflowed")
            })?;
        if result.first_frame_position != session.delivered_codec_frames
            || result.first_sample_position != expected_first_sample
            || result.frame_count as usize != talker_packet.frame_count
            || result.sample_count as usize != sample_count
            || (result.is_final != 0) != talker_packet.is_final
        {
            return Err(BackendError::with_status(
                RuntimeStatus::State,
                "codec returned an inconsistent packet descriptor",
            ));
        }
        session.delivered_codec_frames = session
            .delivered_codec_frames
            .checked_add(talker_packet.frame_count as u64)
            .ok_or_else(|| {
                BackendError::with_status(RuntimeStatus::State, "codec frame position overflowed")
            })?;
        session.first_packet = false;

        Ok(BackendPacket {
            codec_frames: talker_packet.frame_count as u32,
            is_final: talker_packet.is_final,
            talker_gpu_microseconds: talker_packet.gpu_microseconds,
            codec_gpu_microseconds: result.gpu_microseconds,
            peak_request_device_bytes: session.peak_request_device_bytes,
            peak_request_host_bytes: session.peak_request_host_bytes,
        })
    }
}

impl StreamingBackend for NativeBackend {
    type Session = NativeBackendSession;

    fn start(
        &mut self,
        request: BackendRequest,
    ) -> Result<BackendStarted<Self::Session>, BackendError> {
        Self::start_session(&self.talker, &self.codec, request)
    }

    fn start_batch(
        &mut self,
        requests: Vec<BackendRequest>,
    ) -> Vec<Result<BackendStarted<Self::Session>, BackendError>> {
        let talker = Arc::clone(&self.talker);
        let codec = Arc::clone(&self.codec);
        thread::scope(|scope| {
            let handles = requests
                .into_iter()
                .map(|request| {
                    let talker = Arc::clone(&talker);
                    let codec = Arc::clone(&codec);
                    scope.spawn(move || Self::start_session(&talker, &codec, request))
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| {
                    handle.join().unwrap_or_else(|_| {
                        Err(BackendError::with_status(
                            RuntimeStatus::Internal,
                            "native backend prefill thread panicked",
                        ))
                    })
                })
                .collect()
        })
    }

    fn step(
        &mut self,
        session: &mut Self::Session,
        packet_frames: u32,
        pcm: &mut [i16],
    ) -> Result<BackendPacket, BackendError> {
        Self::step_session(session, packet_frames, pcm)
    }

    fn step_batch(
        &mut self,
        requests: &mut [BackendStepInput<'_, Self::Session>],
        packet_frames: u32,
    ) -> Vec<Result<BackendPacket, BackendError>> {
        thread::scope(|scope| {
            let handles = requests
                .iter_mut()
                .map(|request| {
                    scope.spawn(move || {
                        Self::step_session(request.session, packet_frames, request.pcm)
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| {
                    handle.join().unwrap_or_else(|_| {
                        Err(BackendError::with_status(
                            RuntimeStatus::Internal,
                            "native backend generation thread panicked",
                        ))
                    })
                })
                .collect()
        })
    }

    fn cancel(&mut self, session: &mut Self::Session) -> Result<(), BackendError> {
        let talker_result = session.talker.cancel_and_join();
        let codec_result = session.codec.cancel().map_err(|message| {
            BackendError::with_status(
                RuntimeStatus::State,
                format!("failed to cancel codec session: {message}"),
            )
        });
        match (talker_result, codec_result) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
            (Err(talker_error), Err(codec_error)) => Err(BackendError::with_status(
                talker_error.status(),
                format!("{talker_error}; {codec_error}"),
            )),
        }
    }
}

fn run_talker_producer(
    mut talker: NativeTalkerSession,
    sender: SyncSender<TalkerProducerMessage>,
    permits: Receiver<usize>,
    cancelled: &AtomicBool,
) {
    let mut permitted_frames = 1_usize;
    loop {
        for _ in 0..permitted_frames {
            if cancelled.load(Ordering::Acquire) {
                talker.cancel();
                return;
            }
            let frame = match talker.next_frame() {
                Ok(Some(frame)) => frame,
                Ok(None) => {
                    let _ = sender.send(TalkerProducerMessage::Ended);
                    return;
                }
                Err(error) => {
                    let error = map_talker_error(
                        "talker frame generation failed",
                        error,
                        RuntimeStatus::State,
                    );
                    let _ = sender.send(TalkerProducerMessage::Failed(error));
                    return;
                }
            };
            if cancelled.load(Ordering::Acquire) {
                talker.cancel();
                return;
            }
            let is_final = talker.is_ended();
            let message = TalkerProducerMessage::Frame(GeneratedTalkerFrame {
                codes: frame.codes,
                gpu_microseconds: (frame.talker_gpu_milliseconds
                    + frame.predictor_gpu_milliseconds)
                    * 1_000.0,
                is_final,
            });
            if sender.send(message).is_err() {
                talker.cancel();
                return;
            }
            if is_final {
                return;
            }
        }
        permitted_frames = match permits.recv() {
            Ok(frames) => frames,
            Err(_) => {
                talker.cancel();
                return;
            }
        };
        if permitted_frames == 0 || permitted_frames > CODEC_MAX_PACKET_FRAMES {
            let _ = sender.send(TalkerProducerMessage::Failed(BackendError::with_status(
                RuntimeStatus::Internal,
                "talker producer received an invalid prefetch count",
            )));
            return;
        }
    }
}

fn receive_talker_packet(
    receiver: &Receiver<TalkerProducerMessage>,
    cancelled: &AtomicBool,
    requested_frames: usize,
) -> Result<GeneratedTalkerPacket, BackendError> {
    let mut packet = GeneratedTalkerPacket {
        frames: [[0_u16; CODEBOOKS]; CODEC_MAX_PACKET_FRAMES],
        frame_count: 0,
        gpu_microseconds: 0.0,
        is_final: false,
    };
    while packet.frame_count < requested_frames {
        let message = receiver.recv().map_err(|_| {
            if cancelled.load(Ordering::Acquire) {
                BackendError::with_status(RuntimeStatus::Cancelled, "talker producer cancelled")
            } else {
                BackendError::with_status(
                    RuntimeStatus::Internal,
                    "talker producer disconnected before a final frame",
                )
            }
        })?;
        match message {
            TalkerProducerMessage::Frame(frame) => {
                packet.frames[packet.frame_count] = frame.codes;
                packet.frame_count += 1;
                packet.gpu_microseconds += frame.gpu_microseconds;
                if frame.is_final {
                    packet.is_final = true;
                    break;
                }
            }
            TalkerProducerMessage::Ended => {
                packet.is_final = true;
                break;
            }
            TalkerProducerMessage::Failed(error) => return Err(error),
        }
    }
    if packet.frame_count == 0 {
        return Err(BackendError::with_status(
            RuntimeStatus::State,
            "talker ended without a decodable codec frame",
        ));
    }
    Ok(packet)
}

fn voice_request(request: &BackendRequest) -> VoiceDesignRequest {
    let generation = request.generation;
    let mut voice = VoiceDesignRequest::new(
        request.input.text.clone(),
        request.input.instruct.clone(),
        request.input.language.as_official_name(),
    );
    voice.max_frames = generation.max_codec_frames as usize;
    voice.max_sequence_length = MAX_CODEC_FRAMES as usize;
    voice.random_seed = generation.seed;
    voice.talker_sampling = SamplingConfig {
        do_sample: generation.do_sample != 0,
        top_k: generation.top_k,
        top_p: generation.top_p,
        temperature: generation.temperature,
        repetition_penalty: generation.repetition_penalty,
    };
    voice.predictor_sampling = SamplingConfig {
        do_sample: generation.predictor_do_sample != 0,
        top_k: generation.predictor_top_k,
        top_p: generation.predictor_top_p,
        temperature: generation.predictor_temperature,
        repetition_penalty: 1.0,
    };
    voice
}

fn map_talker_error(
    context: &'static str,
    error: AnyError,
    fallback: RuntimeStatus,
) -> BackendError {
    let status = error
        .downcast_ref::<NativeTalkerStatusError>()
        .map(|native| match native.status() {
            TALKER_STATUS_INVALID_ARGUMENT => RuntimeStatus::InvalidArgument,
            TALKER_STATUS_CUDA => RuntimeStatus::Cuda,
            TALKER_STATUS_STATE => RuntimeStatus::State,
            TALKER_STATUS_ALLOCATION => RuntimeStatus::Allocation,
            TALKER_STATUS_MODEL => RuntimeStatus::Model,
            _ => RuntimeStatus::Internal,
        })
        .unwrap_or(fallback);
    BackendError::with_status(status, format!("{context}: {error:#}"))
}

fn map_codec_error(context: &'static str, status: i32, message: String) -> BackendError {
    let runtime_status = match status {
        CODEC_STATUS_INVALID_ARGUMENT => RuntimeStatus::InvalidArgument,
        CODEC_STATUS_CUDA => RuntimeStatus::Cuda,
        CODEC_STATUS_STATE => RuntimeStatus::State,
        CODEC_STATUS_ALLOCATION => RuntimeStatus::Allocation,
        CODEC_STATUS_MODEL => RuntimeStatus::Model,
        _ => RuntimeStatus::Internal,
    };
    BackendError::with_status(runtime_status, format!("{context}: {message}"))
}

fn checked_sum(values: &[u64], message: &'static str) -> Result<u64, BackendError> {
    values.iter().try_fold(0_u64, |total, value| {
        total
            .checked_add(*value)
            .ok_or_else(|| BackendError::with_status(RuntimeStatus::Internal, message))
    })
}

fn duration_microseconds(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GenerationConfig, Language, RequestInput};

    #[test]
    fn generation_contract_maps_without_changing_sampling() {
        let generation = GenerationConfig {
            max_codec_frames: 123,
            seed: 42,
            temperature: 0.7,
            top_p: 0.8,
            repetition_penalty: 1.1,
            top_k: 17,
            do_sample: 1,
            predictor_temperature: 0.6,
            predictor_top_p: 0.75,
            predictor_top_k: 19,
            predictor_do_sample: 0,
            ..GenerationConfig::default()
        };
        let request = BackendRequest {
            id: 7,
            input: RequestInput {
                text: "Hallo".to_owned(),
                instruct: "Calm male voice".to_owned(),
                language: Language::German,
            },
            generation,
        };
        let mapped = voice_request(&request);
        assert_eq!(mapped.max_frames, 123);
        assert_eq!(mapped.random_seed, 42);
        assert_eq!(mapped.language, "German");
        assert_eq!(mapped.talker_sampling.temperature, 0.7);
        assert_eq!(mapped.talker_sampling.top_k, 17);
        assert!(!mapped.predictor_sampling.do_sample);
        assert_eq!(mapped.predictor_sampling.top_p, 0.75);
    }

    #[test]
    fn codec_status_mapping_is_exact() {
        let cases = [
            (
                CODEC_STATUS_INVALID_ARGUMENT,
                RuntimeStatus::InvalidArgument,
            ),
            (CODEC_STATUS_CUDA, RuntimeStatus::Cuda),
            (CODEC_STATUS_STATE, RuntimeStatus::State),
            (CODEC_STATUS_ALLOCATION, RuntimeStatus::Allocation),
            (CODEC_STATUS_MODEL, RuntimeStatus::Model),
            (-99, RuntimeStatus::Internal),
        ];
        for (status, expected) in cases {
            assert_eq!(
                map_codec_error("codec", status, "failure".to_owned()).status(),
                expected
            );
        }
    }

    #[test]
    fn request_memory_accounting_rejects_overflow() {
        let error = checked_sum(&[u64::MAX, 1], "overflow").unwrap_err();
        assert_eq!(error.status(), RuntimeStatus::Internal);
    }

    #[test]
    fn io_error_is_not_misclassified_as_a_native_status() {
        let error = map_talker_error(
            "load",
            AnyError::new(std::io::Error::other("missing file")),
            RuntimeStatus::Model,
        );
        assert_eq!(error.status(), RuntimeStatus::Model);
    }

    #[test]
    fn producer_packet_preserves_frame_order_and_timing() {
        let (sender, receiver) = sync_channel(4);
        sender
            .send(TalkerProducerMessage::Frame(test_frame(11, 2.5, false)))
            .unwrap();
        sender
            .send(TalkerProducerMessage::Frame(test_frame(22, 3.5, true)))
            .unwrap();
        let cancelled = AtomicBool::new(false);
        let packet = receive_talker_packet(&receiver, &cancelled, 4).unwrap();
        assert_eq!(packet.frame_count, 2);
        assert_eq!(packet.frames[0], [11; CODEBOOKS]);
        assert_eq!(packet.frames[1], [22; CODEBOOKS]);
        assert_eq!(packet.gpu_microseconds, 6.0);
        assert!(packet.is_final);
    }

    #[test]
    fn producer_end_marks_a_partial_packet_final() {
        let (sender, receiver) = sync_channel(4);
        sender
            .send(TalkerProducerMessage::Frame(test_frame(7, 1.0, false)))
            .unwrap();
        sender.send(TalkerProducerMessage::Ended).unwrap();
        let cancelled = AtomicBool::new(false);
        let packet = receive_talker_packet(&receiver, &cancelled, 4).unwrap();
        assert_eq!(packet.frame_count, 1);
        assert!(packet.is_final);
    }

    #[test]
    fn producer_failure_preserves_the_runtime_status() {
        let (sender, receiver) = sync_channel(1);
        sender
            .send(TalkerProducerMessage::Failed(BackendError::with_status(
                RuntimeStatus::Cuda,
                "kernel failed",
            )))
            .unwrap();
        let cancelled = AtomicBool::new(false);
        let error = receive_talker_packet(&receiver, &cancelled, 1).unwrap_err();
        assert_eq!(error.status(), RuntimeStatus::Cuda);
        assert_eq!(error.message(), "kernel failed");
    }

    #[test]
    fn cancelled_producer_disconnect_is_not_reported_as_internal() {
        let (sender, receiver) = sync_channel(1);
        drop(sender);
        let cancelled = AtomicBool::new(true);
        let error = receive_talker_packet(&receiver, &cancelled, 1).unwrap_err();
        assert_eq!(error.status(), RuntimeStatus::Cancelled);
    }

    fn test_frame(code: u16, gpu_microseconds: f32, is_final: bool) -> GeneratedTalkerFrame {
        GeneratedTalkerFrame {
            codes: [code; CODEBOOKS],
            gpu_microseconds,
            is_final,
        }
    }
}
