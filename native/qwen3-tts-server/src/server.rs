use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::convert::Infallible;
use std::mem::size_of;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};

use axum::body::Body;
use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::header::{CACHE_CONTROL, CONTENT_DISPOSITION, CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use bytes::Bytes;
use futures_core::Stream;
use qwen3_tts_runtime::{MAX_CODEC_FRAMES, SAMPLE_RATE, SAMPLES_PER_CODEC_FRAME};
use serde::Serialize;
use tokio::runtime::Handle;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tokio::time::timeout;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::api::{
    ACCEPTED_LANGUAGES, MODEL_ID, OpenAiSpeechRequestBody, OutputMode, PreparedSpeech,
    SpeechRequestBody,
};
use crate::config::ServerConfig;
use crate::engine::{
    EngineError, EngineErrorKind, EngineFinishReason, EnginePacket, EnginePoll, SpeechEngine,
    SpeechRequest,
};
use crate::error::ApiError;
use crate::metrics::ServiceMetrics;
use crate::{multipart, wav};

const X_REQUEST_ID: &str = "x-request-id";
const X_QWEN3_SEED: &str = "x-qwen3-seed";
const X_FINISH_REASON: &str = "x-finish-reason";
const X_ACCEL_BUFFERING: &str = "x-accel-buffering";
const BUFFERED_BODY_CHUNK_BYTES: usize = 64 * 1024;

#[derive(Clone)]
struct AppState {
    engine: Arc<dyn SpeechEngine>,
    config: ServerConfig,
    active: Arc<Mutex<HashMap<Uuid, ActiveRequest>>>,
    metrics: Arc<ServiceMetrics>,
    shutdown: ShutdownController,
    buffered_responses: Arc<Semaphore>,
}

/// Serializes request admission against process shutdown and fans cancellation
/// out to every request admitted before the shutdown boundary.
#[derive(Clone)]
pub struct ShutdownController {
    token: CancellationToken,
    admission: Arc<Mutex<()>>,
}

impl ShutdownController {
    #[must_use]
    pub fn new() -> Self {
        Self {
            token: CancellationToken::new(),
            admission: Arc::new(Mutex::new(())),
        }
    }

    /// Closes admission and atomically establishes the shutdown boundary.
    pub fn cancel(&self) {
        let _admission = lock_unpoisoned(&self.admission);
        self.token.cancel();
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }

    pub async fn cancelled(&self) {
        self.token.cancelled().await;
    }

    fn child_token(&self) -> CancellationToken {
        self.token.child_token()
    }

    fn lock_admission(&self) -> MutexGuard<'_, ()> {
        lock_unpoisoned(&self.admission)
    }
}

impl Default for ShutdownController {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
enum ActiveRequest {
    Running(CancellationToken),
    RetirementTimedOut,
}

/// Constructs the bounded HTTP surface around one long-lived native engine.
///
/// # Errors
///
/// Returns an error when HTTP limits exceed intrinsic native runtime limits.
pub fn build_router(engine: Arc<dyn SpeechEngine>, config: ServerConfig) -> Result<Router, String> {
    build_router_with_shutdown(engine, config, ShutdownController::new())
}

/// Constructs the HTTP surface with a process-wide cancellation source.
/// Cancelling `shutdown` propagates to every active native request.
///
/// # Errors
///
/// Returns an error when HTTP limits exceed intrinsic native runtime limits.
pub fn build_router_with_shutdown(
    engine: Arc<dyn SpeechEngine>,
    config: ServerConfig,
    shutdown: ShutdownController,
) -> Result<Router, String> {
    config.validate()?;
    let body_limit = config.max_body_bytes();
    let buffered_capacity = config.max_concurrent_requests as usize;
    let state = AppState {
        engine,
        config,
        active: Arc::new(Mutex::new(HashMap::new())),
        metrics: Arc::new(ServiceMetrics::default()),
        shutdown,
        buffered_responses: Arc::new(Semaphore::new(buffered_capacity)),
    };
    Ok(Router::new()
        .route("/health/live", get(live))
        .route("/health/ready", get(ready))
        .route("/v1/capabilities", get(capabilities))
        .route("/metrics", get(prometheus_metrics))
        .route("/v1/voice-design/speech", post(speech))
        .route("/v1/audio/speech", post(openai_speech))
        .route("/v1/requests/{request_id}", delete(cancel_request))
        .layer(DefaultBodyLimit::max(body_limit))
        .with_state(state))
}

async fn live() -> impl IntoResponse {
    Json(serde_json::json!({"status": "live"}))
}

async fn ready(State(state): State<AppState>) -> Response {
    let is_ready = state.engine.is_ready();
    let status = if is_ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(serde_json::json!({
            "status": if is_ready { "ready" } else { "not_ready" },
            "model": MODEL_ID,
            "model_kind": "voice_design",
            "sample_rate_hz": SAMPLE_RATE,
            "engine_loaded": is_ready,
        })),
    )
        .into_response()
}

#[derive(Serialize)]
struct Capabilities<'a> {
    model: &'a str,
    model_kind: &'a str,
    voice_clone: bool,
    sample_rate_hz: u32,
    channels: u32,
    encoding: &'a str,
    samples_per_codec_frame: u32,
    max_codec_frames: u32,
    max_concurrent_requests: u32,
    max_text_bytes: usize,
    max_voice_description_bytes: usize,
    max_duration_seconds: f64,
    languages: &'a [&'a str],
    streaming: &'a str,
}

async fn capabilities(State(state): State<AppState>) -> impl IntoResponse {
    Json(Capabilities {
        model: MODEL_ID,
        model_kind: "voice_design",
        voice_clone: false,
        sample_rate_hz: SAMPLE_RATE,
        channels: 1,
        encoding: "pcm_s16le",
        samples_per_codec_frame: SAMPLES_PER_CODEC_FRAME,
        max_codec_frames: MAX_CODEC_FRAMES,
        max_concurrent_requests: state.config.max_concurrent_requests,
        max_text_bytes: state.config.max_text_bytes,
        max_voice_description_bytes: state.config.max_voice_description_bytes,
        max_duration_seconds: state.config.max_duration_seconds,
        languages: &ACCEPTED_LANGUAGES,
        streaming: "multipart/mixed",
    })
}

async fn prometheus_metrics(State(state): State<AppState>) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")
        .body(Body::from(state.metrics.render_prometheus()))
        .expect("static metrics headers are valid")
}

async fn speech(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<SpeechRequestBody>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(payload) = payload.map_err(map_json_rejection)?;
    let prepared = payload
        .prepare(&state.config)
        .map_err(ApiError::validation)?;
    let request_id = requested_request_id(&headers)?;
    start_prepared(state, prepared, request_id).await
}

async fn openai_speech(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<OpenAiSpeechRequestBody>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(payload) = payload.map_err(map_json_rejection)?;
    let native = payload.into_native().map_err(ApiError::validation)?;
    let prepared = native
        .prepare(&state.config)
        .map_err(ApiError::validation)?;
    let request_id = requested_request_id(&headers)?;
    start_prepared(state, prepared, request_id).await
}

async fn start_prepared(
    state: AppState,
    prepared: PreparedSpeech,
    requested_request_id: Option<Uuid>,
) -> Result<Response, ApiError> {
    let PreparedSpeech {
        engine_request,
        output,
        seed,
    } = prepared;
    let request_id = requested_request_id.unwrap_or_else(Uuid::now_v7);
    let buffered_permit = if output == OutputMode::BufferedWav {
        Some(
            Arc::clone(&state.buffered_responses)
                .try_acquire_owned()
                .map_err(|_| {
                    state.metrics.request_rejected();
                    ApiError::engine(
                        EngineError::new(
                            EngineErrorKind::Capacity,
                            "buffered response egress capacity is exhausted",
                        ),
                        Some(request_id),
                    )
                })?,
        )
    } else {
        None
    };
    let (request, cancellation) = {
        let _admission = state.shutdown.lock_admission();
        if state.shutdown.is_cancelled() {
            return Err(ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_shutting_down",
                "Server is shutting down",
                "new synthesis requests are not accepted during shutdown",
            ));
        }
        let cancellation = state.shutdown.child_token();
        let mut active = lock_unpoisoned(&state.active);
        match active.entry(request_id) {
            Entry::Vacant(entry) => {
                entry.insert(ActiveRequest::Running(cancellation.clone()));
            }
            Entry::Occupied(_) => {
                return Err(ApiError::new(
                    StatusCode::CONFLICT,
                    "request_id_in_use",
                    "Request ID is already active",
                    "choose a different x-request-id UUID",
                )
                .with_request_id(request_id));
            }
        }
        drop(active);
        let request = match state.engine.start(engine_request) {
            Ok(request) => request,
            Err(error) => {
                lock_unpoisoned(&state.active).remove(&request_id);
                state.metrics.request_rejected();
                return Err(ApiError::engine(error, Some(request_id)));
            }
        };
        (request, cancellation)
    };
    let streaming = output == OutputMode::StreamPcm;
    state.metrics.request_started(streaming);
    match output {
        OutputMode::StreamPcm => Ok(streaming_response(
            state,
            request,
            request_id,
            seed,
            cancellation,
        )),
        OutputMode::BufferedWav => {
            buffered_response(
                state,
                request,
                request_id,
                seed,
                cancellation,
                buffered_permit.expect("buffered output reserves an egress permit"),
            )
            .await
        }
    }
}

fn requested_request_id(headers: &HeaderMap) -> Result<Option<Uuid>, ApiError> {
    let Some(value) = headers.get(X_REQUEST_ID) else {
        return Ok(None);
    };
    let value = value.to_str().map_err(|_| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_id",
            "Invalid request ID",
            "x-request-id must be an ASCII UUID",
        )
    })?;
    Uuid::parse_str(value).map(Some).map_err(|_| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_id",
            "Invalid request ID",
            "x-request-id must be a UUID",
        )
    })
}

fn streaming_response(
    state: AppState,
    request: Box<dyn SpeechRequest>,
    request_id: Uuid,
    seed: u64,
    cancellation: CancellationToken,
) -> Response {
    let boundary = multipart::boundary(request_id);
    let (sender, receiver) = mpsc::channel::<Result<Bytes, Infallible>>(1);
    sender
        .try_send(Ok(multipart::start_part(&boundary, request_id, seed)))
        .expect("a new one-slot stream channel accepts its start part");

    let worker_state = state.clone();
    let worker_boundary = boundary.clone();
    let worker_cancellation = cancellation.clone();
    let runtime = Handle::current();
    tokio::task::spawn_blocking(move || {
        run_streaming_worker(
            worker_state,
            request,
            request_id,
            worker_boundary,
            worker_cancellation,
            sender,
            runtime,
        );
    });

    let stream = CancelOnDropStream {
        inner: ReceiverStream::new(receiver),
        cancellation,
    };
    let content_type = format!("multipart/mixed; boundary={boundary}");
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, content_type)
        .header(CACHE_CONTROL, "no-store")
        .header(X_ACCEL_BUFFERING, "no")
        .header(X_REQUEST_ID, request_id.to_string())
        .header(X_QWEN3_SEED, seed.to_string())
        .body(Body::from_stream(stream))
        .expect("generated streaming response headers are valid")
}

fn run_streaming_worker(
    state: AppState,
    request: Box<dyn SpeechRequest>,
    request_id: Uuid,
    boundary: String,
    cancellation: CancellationToken,
    sender: mpsc::Sender<Result<Bytes, Infallible>>,
    runtime: Handle,
) {
    let guard = ActiveGuard::new(state.clone(), request_id);
    StreamingWorker {
        state,
        request,
        request_id,
        boundary,
        cancellation,
        sender,
        runtime,
        guard,
        sent_samples: 0,
        continuity: PacketContinuity::default(),
    }
    .run();
}

struct StreamingWorker {
    state: AppState,
    request: Box<dyn SpeechRequest>,
    request_id: Uuid,
    boundary: String,
    cancellation: CancellationToken,
    sender: mpsc::Sender<Result<Bytes, Infallible>>,
    runtime: Handle,
    guard: ActiveGuard,
    sent_samples: u64,
    continuity: PacketContinuity,
}

impl StreamingWorker {
    fn run(&mut self) {
        loop {
            if self.cancellation.is_cancelled() {
                self.handle_cancellation();
                return;
            }
            match self.request.poll(self.state.config.poll_timeout) {
                Ok(EnginePoll::WouldBlock) => {}
                Ok(EnginePoll::Packet(packet)) => {
                    if !self.handle_packet(packet) {
                        return;
                    }
                }
                Ok(EnginePoll::EndOfStream(finish_reason)) => {
                    self.handle_end_of_stream(finish_reason);
                    return;
                }
                Err(error) => {
                    self.handle_engine_error(error);
                    return;
                }
            }
        }
    }

    fn handle_cancellation(&mut self) {
        let _ = self.request.cancel();
        let error = ApiError::engine(
            EngineError::new(EngineErrorKind::Cancelled, "request was cancelled"),
            Some(self.request_id),
        );
        self.send_error(&error);
        self.retire(Completion::Cancelled);
    }

    fn handle_packet(&mut self, packet: EnginePacket) -> bool {
        if let Err(error) = self.continuity.accept(&packet) {
            self.fail_native_output(error);
            return false;
        }
        let sample_count = u64::from(packet.sample_count);
        let chunk = multipart::audio_part(&self.boundary, &packet);
        if self.send(chunk, true).is_err() {
            let _ = self.request.cancel();
            self.retire(Completion::Cancelled);
            return false;
        }
        self.sent_samples = self.sent_samples.saturating_add(sample_count);
        true
    }

    fn handle_end_of_stream(&mut self, finish_reason: EngineFinishReason) {
        if let Err(error) = self.continuity.finish() {
            self.fail_native_output(error);
            return;
        }
        if !self
            .request
            .wait_retired(self.state.config.retirement_timeout)
        {
            let error = retirement_timeout_error(self.request_id);
            self.send_error(&error);
            self.guard.retirement_timed_out();
            return;
        }
        let chunk = multipart::end_part(
            &self.boundary,
            self.request_id,
            finish_reason,
            self.request.metrics(),
        );
        if self.send(chunk, true).is_err() {
            self.guard.cancelled();
        } else {
            self.guard.completed(self.sent_samples);
        }
    }

    fn handle_engine_error(&mut self, error: EngineError) {
        let completion = if error.kind == EngineErrorKind::Cancelled {
            Completion::Cancelled
        } else {
            Completion::Failed
        };
        self.send_error(&ApiError::engine(error, Some(self.request_id)));
        self.retire(completion);
    }

    fn fail_native_output(&mut self, error: EngineError) {
        let _ = self.request.cancel();
        self.send_error(&ApiError::engine(error, Some(self.request_id)));
        self.retire(Completion::Failed);
    }

    fn retire(&mut self, completion: Completion) {
        mark_retirement(
            self.request.as_ref(),
            &mut self.guard,
            self.state.config.retirement_timeout,
            completion,
        );
    }

    fn send_error(&self, error: &ApiError) {
        let _ = self.send(
            multipart::error_part(&self.boundary, &error.stream_payload()),
            false,
        );
    }

    fn send(&self, chunk: Bytes, observe_cancellation: bool) -> Result<(), SendChunkError> {
        send_chunk(
            &self.runtime,
            &self.sender,
            chunk,
            &self.cancellation,
            self.state.config.slow_client_timeout,
            observe_cancellation,
        )
    }
}

async fn buffered_response(
    state: AppState,
    request: Box<dyn SpeechRequest>,
    request_id: Uuid,
    seed: u64,
    cancellation: CancellationToken,
    buffered_permit: OwnedSemaphorePermit,
) -> Result<Response, ApiError> {
    let worker_state = state.clone();
    let worker_cancellation = cancellation.clone();
    let cancel_on_drop = CancelOnFutureDrop::new(cancellation);
    let result = tokio::task::spawn_blocking(move || {
        run_buffered_worker(worker_state, request, request_id, worker_cancellation)
    })
    .await;
    cancel_on_drop.disarm();
    let output = result.map_err(|_| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "worker_panicked",
            "Audio worker failed",
            "the buffered audio worker terminated unexpectedly",
        )
        .with_request_id(request_id)
    })??;
    let length = output.wav.len();
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "audio/wav")
        .header(CONTENT_LENGTH, length.to_string())
        .header(CONTENT_DISPOSITION, "inline; filename=voice-design.wav")
        .header(CACHE_CONTROL, "no-store")
        .header(X_REQUEST_ID, request_id.to_string())
        .header(X_QWEN3_SEED, seed.to_string())
        .header(X_FINISH_REASON, output.finish_reason.as_str())
        .body(Body::from_stream(BufferedWavStream::new(
            Bytes::from(output.wav),
            buffered_permit,
        )))
        .expect("generated WAV response headers are valid"))
}

fn run_buffered_worker(
    state: AppState,
    request: Box<dyn SpeechRequest>,
    request_id: Uuid,
    cancellation: CancellationToken,
) -> Result<BufferedOutput, ApiError> {
    let mut guard = ActiveGuard::new(state.clone(), request_id);
    let mut pcm = Vec::new();
    let mut continuity = PacketContinuity::default();
    loop {
        if cancellation.is_cancelled() {
            let _ = request.cancel();
            mark_retirement(
                request.as_ref(),
                &mut guard,
                state.config.retirement_timeout,
                Completion::Cancelled,
            );
            return Err(ApiError::engine(
                EngineError::new(EngineErrorKind::Cancelled, "request was cancelled"),
                Some(request_id),
            ));
        }
        match request.poll(state.config.poll_timeout) {
            Ok(EnginePoll::WouldBlock) => {}
            Ok(EnginePoll::Packet(packet)) => {
                if let Err(error) = continuity.accept(&packet) {
                    let _ = request.cancel();
                    mark_retirement(
                        request.as_ref(),
                        &mut guard,
                        state.config.retirement_timeout,
                        Completion::Failed,
                    );
                    return Err(ApiError::engine(error, Some(request_id)));
                }
                let next_length = pcm.len().saturating_add(packet.pcm_s16le.len());
                if next_length > state.config.max_buffered_pcm_bytes() {
                    let _ = request.cancel();
                    mark_retirement(
                        request.as_ref(),
                        &mut guard,
                        state.config.retirement_timeout,
                        Completion::Failed,
                    );
                    return Err(ApiError::new(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "buffer_limit_exceeded",
                        "Buffered audio limit was exceeded",
                        "native audio exceeded the validated duration bound",
                    )
                    .with_request_id(request_id));
                }
                pcm.extend_from_slice(&packet.pcm_s16le);
            }
            Ok(EnginePoll::EndOfStream(finish_reason)) => {
                if let Err(error) = continuity.finish() {
                    let _ = request.cancel();
                    mark_retirement(
                        request.as_ref(),
                        &mut guard,
                        state.config.retirement_timeout,
                        Completion::Failed,
                    );
                    return Err(ApiError::engine(error, Some(request_id)));
                }
                if !request.wait_retired(state.config.retirement_timeout) {
                    guard.retirement_timed_out();
                    return Err(retirement_timeout_error(request_id));
                }
                let sample_count = u64::try_from(pcm.len() / size_of::<i16>()).unwrap_or(u64::MAX);
                let output = wav::encode_pcm16_mono(&pcm).map_err(|detail| {
                    ApiError::new(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "wav_encoding_failed",
                        "WAV encoding failed",
                        detail,
                    )
                    .with_request_id(request_id)
                })?;
                guard.completed(sample_count);
                return Ok(BufferedOutput {
                    wav: output,
                    finish_reason,
                });
            }
            Err(error) => {
                let cancelled = error.kind == EngineErrorKind::Cancelled;
                mark_retirement(
                    request.as_ref(),
                    &mut guard,
                    state.config.retirement_timeout,
                    if cancelled {
                        Completion::Cancelled
                    } else {
                        Completion::Failed
                    },
                );
                return Err(ApiError::engine(error, Some(request_id)));
            }
        }
    }
}

struct BufferedOutput {
    wav: Vec<u8>,
    finish_reason: EngineFinishReason,
}

fn validate_packet(packet: &EnginePacket) -> Result<(), EngineError> {
    let expected_bytes = usize::try_from(packet.sample_count)
        .unwrap_or(usize::MAX)
        .saturating_mul(size_of::<i16>());
    if packet.sample_rate != SAMPLE_RATE
        || packet.channels != 1
        || packet.codec_frames == 0
        || packet.codec_frames > 4
        || packet.sample_count != packet.codec_frames * SAMPLES_PER_CODEC_FRAME
        || packet.pcm_s16le.len() != expected_bytes
    {
        return Err(EngineError::new(
            EngineErrorKind::Internal,
            "native runtime emitted an invalid audio packet",
        ));
    }
    Ok(())
}

#[derive(Default)]
struct PacketContinuity {
    next_sequence: u64,
    next_codec_frame: u64,
    next_sample: u64,
    saw_packet: bool,
    saw_final: bool,
}

impl PacketContinuity {
    fn accept(&mut self, packet: &EnginePacket) -> Result<(), EngineError> {
        validate_packet(packet)?;
        if self.saw_final
            || packet.sequence != self.next_sequence
            || packet.first_codec_frame != self.next_codec_frame
            || packet.first_sample != self.next_sample
        {
            return Err(EngineError::new(
                EngineErrorKind::Internal,
                "native runtime emitted a discontinuous audio packet",
            ));
        }
        self.next_sequence = self.next_sequence.checked_add(1).ok_or_else(|| {
            EngineError::new(EngineErrorKind::Internal, "audio sequence overflowed")
        })?;
        self.next_codec_frame = self
            .next_codec_frame
            .checked_add(u64::from(packet.codec_frames))
            .ok_or_else(|| {
                EngineError::new(EngineErrorKind::Internal, "audio frame position overflowed")
            })?;
        self.next_sample = self
            .next_sample
            .checked_add(u64::from(packet.sample_count))
            .ok_or_else(|| {
                EngineError::new(
                    EngineErrorKind::Internal,
                    "audio sample position overflowed",
                )
            })?;
        self.saw_packet = true;
        self.saw_final = packet.is_final;
        Ok(())
    }

    fn finish(&self) -> Result<(), EngineError> {
        if !self.saw_packet || !self.saw_final {
            return Err(EngineError::new(
                EngineErrorKind::Internal,
                "native runtime reached end-of-stream without a final audio packet",
            ));
        }
        Ok(())
    }
}

fn mark_retirement(
    request: &dyn SpeechRequest,
    guard: &mut ActiveGuard,
    timeout: std::time::Duration,
    completion: Completion,
) {
    if request.wait_retired(timeout) {
        guard.completion = completion;
    } else {
        guard.retirement_timed_out();
    }
}

fn retirement_timeout_error(request_id: Uuid) -> ApiError {
    ApiError::new(
        StatusCode::SERVICE_UNAVAILABLE,
        "native_retirement_timeout",
        "Native request did not retire",
        "the native request exceeded its bounded retirement deadline; restart the unhealthy engine",
    )
    .with_request_id(request_id)
}

#[derive(Debug)]
enum SendChunkError {
    Cancelled,
    Disconnected,
    TimedOut,
}

fn send_chunk(
    runtime: &Handle,
    sender: &mpsc::Sender<Result<Bytes, Infallible>>,
    chunk: Bytes,
    cancellation: &CancellationToken,
    send_timeout: std::time::Duration,
    observe_cancellation: bool,
) -> Result<(), SendChunkError> {
    runtime.block_on(async {
        if observe_cancellation {
            tokio::select! {
                () = cancellation.cancelled() => Err(SendChunkError::Cancelled),
                result = timeout(send_timeout, sender.send(Ok(chunk))) => map_send_result(result),
            }
        } else {
            map_send_result(timeout(send_timeout, sender.send(Ok(chunk))).await)
        }
    })
}

fn map_send_result(
    result: Result<
        Result<(), mpsc::error::SendError<Result<Bytes, Infallible>>>,
        tokio::time::error::Elapsed,
    >,
) -> Result<(), SendChunkError> {
    match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(_)) => Err(SendChunkError::Disconnected),
        Err(_) => Err(SendChunkError::TimedOut),
    }
}

#[derive(Serialize)]
struct CancellationAccepted {
    request_id: Uuid,
    status: &'static str,
}

async fn cancel_request(
    State(state): State<AppState>,
    Path(raw_request_id): Path<String>,
) -> Result<Response, ApiError> {
    let request_id = Uuid::parse_str(&raw_request_id).map_err(|_| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_id",
            "Invalid request ID",
            "request_id must be a UUID",
        )
    })?;
    let active = lock_unpoisoned(&state.active).get(&request_id).cloned();
    match active {
        Some(ActiveRequest::Running(cancellation)) => cancellation.cancel(),
        Some(ActiveRequest::RetirementTimedOut) => {
            return Err(retirement_timeout_error(request_id));
        }
        None => {
            return Err(ApiError::new(
                StatusCode::NOT_FOUND,
                "request_not_found",
                "Request not found",
                "the request is not active or has already retired",
            )
            .with_request_id(request_id));
        }
    }
    Ok((
        StatusCode::ACCEPTED,
        Json(CancellationAccepted {
            request_id,
            status: "cancellation_requested",
        }),
    )
        .into_response())
}

fn map_json_rejection(rejection: JsonRejection) -> ApiError {
    ApiError::malformed_json(rejection.status())
}

enum Completion {
    Pending,
    Completed(u64),
    Failed,
    Cancelled,
    RetirementTimedOut,
}

struct ActiveGuard {
    state: AppState,
    request_id: Uuid,
    completion: Completion,
}

impl ActiveGuard {
    fn new(state: AppState, request_id: Uuid) -> Self {
        Self {
            state,
            request_id,
            completion: Completion::Pending,
        }
    }

    fn completed(&mut self, samples: u64) {
        self.completion = Completion::Completed(samples);
    }

    fn cancelled(&mut self) {
        self.completion = Completion::Cancelled;
    }

    fn retirement_timed_out(&mut self) {
        self.completion = Completion::RetirementTimedOut;
    }
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        if matches!(self.completion, Completion::RetirementTimedOut) {
            if let Some(entry) = lock_unpoisoned(&self.state.active).get_mut(&self.request_id) {
                *entry = ActiveRequest::RetirementTimedOut;
            }
        } else {
            lock_unpoisoned(&self.state.active).remove(&self.request_id);
        }
        match self.completion {
            Completion::Completed(samples) => self.state.metrics.request_completed(samples),
            Completion::Cancelled => self.state.metrics.request_cancelled(),
            Completion::Pending | Completion::Failed => self.state.metrics.request_failed(),
            Completion::RetirementTimedOut => {
                self.state.metrics.request_retirement_timed_out();
            }
        }
    }
}

struct CancelOnDropStream {
    inner: ReceiverStream<Result<Bytes, Infallible>>,
    cancellation: CancellationToken,
}

struct BufferedWavStream {
    wav: Bytes,
    offset: usize,
    permit: Option<OwnedSemaphorePermit>,
}

impl BufferedWavStream {
    fn new(wav: Bytes, permit: OwnedSemaphorePermit) -> Self {
        Self {
            wav,
            offset: 0,
            permit: Some(permit),
        }
    }
}

impl Stream for BufferedWavStream {
    type Item = Result<Bytes, Infallible>;

    fn poll_next(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.offset >= this.wav.len() {
            this.permit.take();
            return Poll::Ready(None);
        }
        let end = this
            .offset
            .saturating_add(BUFFERED_BODY_CHUNK_BYTES)
            .min(this.wav.len());
        let chunk = this.wav.slice(this.offset..end);
        this.offset = end;
        Poll::Ready(Some(Ok(chunk)))
    }
}

impl Stream for CancelOnDropStream {
    type Item = Result<Bytes, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(context)
    }
}

impl Drop for CancelOnDropStream {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

struct CancelOnFutureDrop {
    cancellation: CancellationToken,
    armed: std::sync::atomic::AtomicBool,
}

impl CancelOnFutureDrop {
    fn new(cancellation: CancellationToken) -> Self {
        Self {
            cancellation,
            armed: std::sync::atomic::AtomicBool::new(true),
        }
    }

    fn disarm(&self) {
        self.armed
            .store(false, std::sync::atomic::Ordering::Release);
    }
}

impl Drop for CancelOnFutureDrop {
    fn drop(&mut self) {
        if self.armed.load(std::sync::atomic::Ordering::Acquire) {
            self.cancellation.cancel();
        }
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    fn continuity_packet(
        sequence: u64,
        first_codec_frame: u64,
        first_sample: u64,
        is_final: bool,
    ) -> EnginePacket {
        EnginePacket {
            sequence,
            first_codec_frame,
            first_sample,
            codec_frames: 1,
            sample_count: SAMPLES_PER_CODEC_FRAME,
            sample_rate: SAMPLE_RATE,
            channels: 1,
            is_final,
            pcm_s16le: vec![0; SAMPLES_PER_CODEC_FRAME as usize * size_of::<i16>()],
        }
    }

    #[test]
    fn packet_continuity_accepts_one_exact_final_sequence() {
        let mut continuity = PacketContinuity::default();
        continuity
            .accept(&continuity_packet(0, 0, 0, false))
            .unwrap();
        continuity
            .accept(&continuity_packet(
                1,
                1,
                u64::from(SAMPLES_PER_CODEC_FRAME),
                true,
            ))
            .unwrap();
        continuity.finish().unwrap();
    }

    #[test]
    fn packet_continuity_rejects_gaps_and_packets_after_final() {
        let mut gap = PacketContinuity::default();
        gap.accept(&continuity_packet(0, 0, 0, false)).unwrap();
        assert!(
            gap.accept(&continuity_packet(
                2,
                1,
                u64::from(SAMPLES_PER_CODEC_FRAME),
                true,
            ))
            .is_err()
        );

        let mut after_final = PacketContinuity::default();
        after_final
            .accept(&continuity_packet(0, 0, 0, true))
            .unwrap();
        assert!(
            after_final
                .accept(&continuity_packet(
                    1,
                    1,
                    u64::from(SAMPLES_PER_CODEC_FRAME),
                    true,
                ))
                .is_err()
        );
    }

    #[test]
    fn packet_continuity_requires_audio_and_an_explicit_final_packet() {
        assert!(PacketContinuity::default().finish().is_err());
        let mut continuity = PacketContinuity::default();
        continuity
            .accept(&continuity_packet(0, 0, 0, false))
            .unwrap();
        assert!(continuity.finish().is_err());
    }

    #[test]
    fn shutdown_waits_for_the_atomic_admission_boundary() {
        let shutdown = ShutdownController::new();
        let admission = shutdown.lock_admission();
        let completed = Arc::new(AtomicBool::new(false));
        let thread_shutdown = shutdown.clone();
        let thread_completed = Arc::clone(&completed);
        let worker = std::thread::spawn(move || {
            thread_shutdown.cancel();
            thread_completed.store(true, Ordering::Release);
        });
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(!completed.load(Ordering::Acquire));
        assert!(!shutdown.is_cancelled());
        drop(admission);
        worker.join().unwrap();
        assert!(completed.load(Ordering::Acquire));
        assert!(shutdown.is_cancelled());
    }
}
