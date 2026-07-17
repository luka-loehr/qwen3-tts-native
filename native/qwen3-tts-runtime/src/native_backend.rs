use std::path::Path;
use std::sync::Arc;
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

use crate::cuda_packet_stager::{CudaPacketStager, CudaRuntime};
use crate::{
    BackendError, BackendPacket, BackendRequest, BackendStarted, BackendStepInput,
    MAX_CODEC_FRAMES, RuntimeStatus, StreamingBackend,
};

const CODEC_STATUS_INVALID_ARGUMENT: i32 = -1;
const CODEC_STATUS_CUDA: i32 = -2;
const CODEC_STATUS_ALLOCATION: i32 = -4;

pub struct NativeBackend {
    talker: Arc<NativeTalkerModel>,
    codec: Arc<NativeCodecModel>,
    cuda_stager_runtime: Option<Arc<CudaRuntime>>,
    device_index: i32,
}

pub struct NativeBackendSession {
    talker: NativeTalkerSession,
    codec: NativeCodecSession,
    cuda_stager: Option<CudaPacketStager>,
    first_packet: bool,
    peak_request_device_bytes: u64,
    peak_request_host_bytes: u64,
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
        let cuda_stager_runtime = match (
            talker.supports_device_frames(),
            codec.supports_device_packets(),
        ) {
            (true, true) => Some(CudaRuntime::load().map_err(|message| {
                BackendError::with_status(
                    RuntimeStatus::Cuda,
                    format!("failed to load CUDA packet stager: {message}"),
                )
            })?),
            (false, false) => None,
            (talker_device_frames, codec_device_packets) => {
                return Err(BackendError::with_status(
                    RuntimeStatus::Model,
                    format!(
                        "native device handoff ABI mismatch: talker device frames={talker_device_frames}, codec device packets={codec_device_packets}"
                    ),
                ));
            }
        };
        Ok(Self {
            talker,
            codec,
            cuda_stager_runtime,
            device_index,
        })
    }

    fn start_session(
        talker: &Arc<NativeTalkerModel>,
        codec: &Arc<NativeCodecModel>,
        cuda_stager_runtime: Option<&Arc<CudaRuntime>>,
        device_index: i32,
        request: BackendRequest,
    ) -> Result<BackendStarted<NativeBackendSession>, BackendError> {
        let started_at = Instant::now();
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
        let cuda_stager = cuda_stager_runtime
            .map(|runtime| {
                CudaPacketStager::new(Arc::clone(runtime), device_index).map_err(|message| {
                    BackendError::with_status(
                        RuntimeStatus::Cuda,
                        format!("failed to create CUDA packet stager: {message}"),
                    )
                })
            })
            .transpose()?;
        let stager_device_bytes = cuda_stager
            .as_ref()
            .map_or(0, CudaPacketStager::device_bytes);
        let peak_request_device_bytes = checked_sum(
            &[
                talker_memory.talker_kv_bytes,
                talker_memory.predictor_kv_bytes,
                talker_memory.workspace_bytes,
                codec_memory.device_bytes,
                stager_device_bytes,
            ],
            "request device memory accounting overflowed",
        )?;

        Ok(BackendStarted {
            session: NativeBackendSession {
                talker: talker_session,
                codec: codec_session,
                cuda_stager,
                first_packet: true,
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
        let requested_frames = if session.first_packet {
            1
        } else {
            usize::try_from(packet_frames).map_err(|_| {
                BackendError::with_status(
                    RuntimeStatus::InvalidArgument,
                    "packet frame count overflowed",
                )
            })?
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

        let packet = if session.cuda_stager.is_some() {
            Self::step_device_session(session, requested_frames, pcm)?
        } else {
            Self::step_host_session(session, requested_frames, pcm)?
        };
        session.first_packet = false;
        Ok(packet)
    }

    fn step_host_session(
        session: &mut NativeBackendSession,
        requested_frames: usize,
        pcm: &mut [i16],
    ) -> Result<BackendPacket, BackendError> {
        let mut frames = [[0_u16; CODEBOOKS]; CODEC_MAX_PACKET_FRAMES];
        let mut frame_count = 0_usize;
        let mut talker_gpu_microseconds = 0.0_f32;
        while frame_count < requested_frames {
            let Some(frame) = session.talker.next_frame().map_err(|error| {
                map_talker_error(
                    "talker frame generation failed",
                    error,
                    RuntimeStatus::State,
                )
            })?
            else {
                break;
            };
            frames[frame_count] = frame.codes;
            frame_count += 1;
            talker_gpu_microseconds +=
                (frame.talker_gpu_milliseconds + frame.predictor_gpu_milliseconds) * 1_000.0;
            if session.talker.is_ended() {
                break;
            }
        }
        if frame_count == 0 {
            return Err(BackendError::with_status(
                RuntimeStatus::State,
                "talker ended without a decodable codec frame",
            ));
        }

        let is_final = session.talker.is_ended();
        let sample_count = frame_count * SAMPLES_PER_FRAME;
        let result = session
            .codec
            .process_into(&frames[..frame_count], is_final, &mut pcm[..sample_count])
            .map_err(|(status, message)| {
                map_codec_error("codec packet generation failed", status, message)
            })?;
        if result.frame_count as usize != frame_count
            || result.sample_count as usize != sample_count
            || (result.is_final != 0) != is_final
        {
            return Err(BackendError::with_status(
                RuntimeStatus::State,
                "codec returned an inconsistent packet descriptor",
            ));
        }

        Ok(BackendPacket {
            codec_frames: frame_count as u32,
            is_final,
            talker_gpu_microseconds,
            codec_gpu_microseconds: result.gpu_microseconds,
            peak_request_device_bytes: session.peak_request_device_bytes,
            peak_request_host_bytes: session.peak_request_host_bytes,
        })
    }

    fn step_device_session(
        session: &mut NativeBackendSession,
        requested_frames: usize,
        pcm: &mut [i16],
    ) -> Result<BackendPacket, BackendError> {
        let stager = session.cuda_stager.as_mut().ok_or_else(|| {
            BackendError::with_status(RuntimeStatus::Internal, "CUDA packet stager is missing")
        })?;
        if stager.staged_frames() != 0 {
            return Err(BackendError::with_status(
                RuntimeStatus::State,
                "CUDA packet stager contains an unconsumed packet",
            ));
        }

        let mut frame_count = 0_usize;
        let mut talker_gpu_microseconds = 0.0_f32;
        while frame_count < requested_frames {
            let Some(frame) = session.talker.begin_device_frame().map_err(|error| {
                map_talker_error(
                    "Talker device-frame generation failed",
                    error,
                    RuntimeStatus::State,
                )
            })?
            else {
                break;
            };
            let code_count = frame.code_count();
            let device_index = frame.device_index();
            let (device_codes, producer_ready_event) = unsafe { frame.raw_device_parts() };
            let copied_event = unsafe {
                stager.stage_frame(device_codes, code_count, producer_ready_event, device_index)
            }
            .map_err(|message| {
                BackendError::with_status(
                    RuntimeStatus::Cuda,
                    format!("failed to stage Talker device frame: {message}"),
                )
            })?;
            let finished =
                unsafe { frame.finish_with_consumer_event(copied_event) }.map_err(|error| {
                    map_talker_error(
                        "Talker device-frame completion failed",
                        error,
                        RuntimeStatus::State,
                    )
                })?;
            frame_count += 1;
            talker_gpu_microseconds +=
                (finished.talker_gpu_milliseconds + finished.predictor_gpu_milliseconds) * 1_000.0;
            if session.talker.is_ended() {
                break;
            }
        }
        if frame_count == 0 {
            return Err(BackendError::with_status(
                RuntimeStatus::State,
                "Talker ended without a decodable device codec frame",
            ));
        }

        let view = stager.packet_view().map_err(|message| {
            BackendError::with_status(
                RuntimeStatus::State,
                format!("failed to inspect staged CUDA packet: {message}"),
            )
        })?;
        if view.frame_count != frame_count {
            return Err(BackendError::with_status(
                RuntimeStatus::State,
                "CUDA packet stager returned an inconsistent frame count",
            ));
        }
        let packet = unsafe {
            session
                .codec
                .begin_device_packet(view.device_codes, view.frame_count, view.ready_event)
        }
        .map_err(|(status, message)| {
            map_codec_error("Codec device-packet begin failed", status, message)
        })?;
        let codes_consumed_event = packet.codes_consumed_event();
        unsafe { stager.release_after_consumer(codes_consumed_event) }.map_err(|message| {
            BackendError::with_status(
                RuntimeStatus::Cuda,
                format!("failed to release CUDA packet staging buffer: {message}"),
            )
        })?;

        let is_final = session.talker.is_ended();
        let sample_count = frame_count * SAMPLES_PER_FRAME;
        let result =
            packet
                .finish(is_final, &mut pcm[..sample_count])
                .map_err(|(status, message)| {
                    map_codec_error("Codec device-packet finish failed", status, message)
                })?;
        if result.frame_count as usize != frame_count
            || result.sample_count as usize != sample_count
            || (result.is_final != 0) != is_final
        {
            return Err(BackendError::with_status(
                RuntimeStatus::State,
                "Codec device path returned an inconsistent packet descriptor",
            ));
        }

        Ok(BackendPacket {
            codec_frames: frame_count as u32,
            is_final,
            talker_gpu_microseconds,
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
        Self::start_session(
            &self.talker,
            &self.codec,
            self.cuda_stager_runtime.as_ref(),
            self.device_index,
            request,
        )
    }

    fn start_batch(
        &mut self,
        requests: Vec<BackendRequest>,
    ) -> Vec<Result<BackendStarted<Self::Session>, BackendError>> {
        let talker = Arc::clone(&self.talker);
        let codec = Arc::clone(&self.codec);
        let cuda_stager_runtime = self.cuda_stager_runtime.clone();
        let device_index = self.device_index;
        thread::scope(|scope| {
            let handles = requests
                .into_iter()
                .map(|request| {
                    let talker = Arc::clone(&talker);
                    let codec = Arc::clone(&codec);
                    let cuda_stager_runtime = cuda_stager_runtime.clone();
                    scope.spawn(move || {
                        Self::start_session(
                            &talker,
                            &codec,
                            cuda_stager_runtime.as_ref(),
                            device_index,
                            request,
                        )
                    })
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
        session.talker.cancel();
        session.codec.cancel().map_err(|message| {
            BackendError::with_status(
                RuntimeStatus::State,
                format!("failed to cancel codec session: {message}"),
            )
        })
    }
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
}
