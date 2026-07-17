use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use qwen3_tts_runtime::{Language, SAMPLE_RATE, SAMPLES_PER_CODEC_FRAME};
use qwen3_tts_server::{
    EngineError, EngineErrorKind, EngineFinishReason, EngineMetrics, EnginePacket, EnginePoll,
    EngineSynthesisRequest, ServerConfig, ShutdownController, SpeechEngine, SpeechRequest,
    build_router, build_router_with_shutdown,
};
use serde_json::{Value, json};
use tower::ServiceExt;

#[derive(Clone)]
struct FakeEngine {
    state: Arc<FakeState>,
}

struct FakeState {
    ready: AtomicBool,
    starts: AtomicUsize,
    polls: AtomicUsize,
    cancelled: AtomicBool,
    completed: AtomicBool,
    retires: AtomicBool,
    second_packet_delay: Duration,
    captured_language: Mutex<Option<Language>>,
    finish_reason: Mutex<EngineFinishReason>,
}

impl FakeEngine {
    fn new(second_packet_delay: Duration) -> Self {
        Self {
            state: Arc::new(FakeState {
                ready: AtomicBool::new(true),
                starts: AtomicUsize::new(0),
                polls: AtomicUsize::new(0),
                cancelled: AtomicBool::new(false),
                completed: AtomicBool::new(false),
                retires: AtomicBool::new(true),
                second_packet_delay,
                captured_language: Mutex::new(None),
                finish_reason: Mutex::new(EngineFinishReason::Stop),
            }),
        }
    }
}

impl SpeechEngine for FakeEngine {
    fn start(
        &self,
        request: EngineSynthesisRequest,
    ) -> Result<Box<dyn SpeechRequest>, EngineError> {
        self.state.starts.fetch_add(1, Ordering::Relaxed);
        *lock_unpoisoned(&self.state.captured_language) = Some(request.input.language);
        Ok(Box::new(FakeRequest {
            state: Arc::clone(&self.state),
            next: Mutex::new(0),
        }))
    }

    fn is_ready(&self) -> bool {
        self.state.ready.load(Ordering::Acquire)
    }
}

struct FakeRequest {
    state: Arc<FakeState>,
    next: Mutex<usize>,
}

impl SpeechRequest for FakeRequest {
    fn poll(&self, _timeout: Duration) -> Result<EnginePoll, EngineError> {
        if self.state.cancelled.load(Ordering::Acquire) {
            return Err(EngineError::new(
                EngineErrorKind::Cancelled,
                "fake request cancelled",
            ));
        }
        let index = {
            let mut next = lock_unpoisoned(&self.next);
            let index = *next;
            *next += 1;
            index
        };
        self.state.polls.fetch_add(1, Ordering::Relaxed);
        match index {
            0 => Ok(EnginePoll::Packet(packet(0, false, 7))),
            1 => {
                thread::sleep(self.state.second_packet_delay);
                if self.state.cancelled.load(Ordering::Acquire) {
                    return Err(EngineError::new(
                        EngineErrorKind::Cancelled,
                        "fake request cancelled",
                    ));
                }
                Ok(EnginePoll::Packet(packet(1, true, -11)))
            }
            _ => {
                self.state.completed.store(true, Ordering::Release);
                Ok(EnginePoll::EndOfStream(*lock_unpoisoned(
                    &self.state.finish_reason,
                )))
            }
        }
    }

    fn cancel(&self) -> Result<(), EngineError> {
        self.state.cancelled.store(true, Ordering::Release);
        Ok(())
    }

    fn wait_retired(&self, _timeout: Duration) -> bool {
        self.state.retires.load(Ordering::Acquire)
    }

    fn metrics(&self) -> EngineMetrics {
        let packets = (*lock_unpoisoned(&self.next)).saturating_sub(1).min(2) as u64;
        EngineMetrics {
            emitted_packets: packets,
            emitted_samples: packets * u64::from(SAMPLES_PER_CODEC_FRAME),
            generated_codec_frames: packets,
            first_audio_microseconds: 10_000,
            ..EngineMetrics::default()
        }
    }
}

fn packet(sequence: u64, is_final: bool, sample: i16) -> EnginePacket {
    let sample_count = SAMPLES_PER_CODEC_FRAME;
    let mut pcm_s16le = Vec::with_capacity(sample_count as usize * 2);
    for _ in 0..sample_count {
        pcm_s16le.extend_from_slice(&sample.to_le_bytes());
    }
    EnginePacket {
        sequence,
        first_codec_frame: sequence,
        first_sample: sequence * u64::from(SAMPLES_PER_CODEC_FRAME),
        codec_frames: 1,
        sample_count,
        sample_rate: SAMPLE_RATE,
        channels: 1,
        is_final,
        pcm_s16le,
    }
}

fn speech_json() -> Value {
    json!({
        "text": "Guten Morgen",
        "voice_description": "A calm adult male voice with measured delivery",
        "language": "German",
        "seed": 42,
        "max_duration_seconds": 5.0
    })
}

fn json_request(uri: &str, payload: &Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap()
}

async fn response_json(response: axum::response::Response) -> Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn health_and_capabilities_expose_only_voice_design() {
    let engine = FakeEngine::new(Duration::ZERO);
    let app = build_router(Arc::new(engine), ServerConfig::default()).unwrap();

    let live = app
        .clone()
        .oneshot(Request::get("/health/live").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(live.status(), StatusCode::OK);

    let capabilities = app
        .oneshot(
            Request::get("/v1/capabilities")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = response_json(capabilities).await;
    assert_eq!(body["model_kind"], "voice_design");
    assert_eq!(body["voice_clone"], false);
    assert_eq!(body["sample_rate_hz"], 24_000);
    assert!(
        body["languages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "italian")
    );
}

#[tokio::test]
async fn not_ready_is_a_real_503_without_starting_a_request() {
    let engine = FakeEngine::new(Duration::ZERO);
    engine.state.ready.store(false, Ordering::Release);
    let app = build_router(Arc::new(engine), ServerConfig::default()).unwrap();
    let response = app
        .oneshot(Request::get("/health/ready").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn turkish_returns_structured_422_before_the_engine_is_called() {
    let engine = FakeEngine::new(Duration::ZERO);
    let state = Arc::clone(&engine.state);
    let app = build_router(Arc::new(engine), ServerConfig::default()).unwrap();
    let mut payload = speech_json();
    payload["language"] = json!("Turkish");
    let response = app
        .oneshot(json_request("/v1/voice-design/speech", &payload))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "application/problem+json"
    );
    let body = response_json(response).await;
    assert_eq!(body["code"], "unsupported_language");
    assert_eq!(
        body["type"],
        "urn:qwen3-tts-native:problem:unsupported_language"
    );
    assert_eq!(state.starts.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn voice_clone_fields_are_rejected_as_malformed_contract_input() {
    let engine = FakeEngine::new(Duration::ZERO);
    let app = build_router(Arc::new(engine), ServerConfig::default()).unwrap();
    let mut payload = speech_json();
    payload["reference_audio"] = json!("base64-is-deliberately-not-supported");
    let response = app
        .oneshot(json_request("/v1/voice-design/speech", &payload))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(response_json(response).await["code"], "malformed_json");
}

#[tokio::test]
async fn multipart_audio_is_delivered_before_generation_completes() {
    let engine = FakeEngine::new(Duration::from_millis(250));
    let state = Arc::clone(&engine.state);
    let app = build_router(Arc::new(engine), ServerConfig::default()).unwrap();
    let mut response = app
        .oneshot(json_request("/v1/voice-design/speech", &speech_json()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response.headers()[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("multipart/mixed; boundary=qwen3tts-")
    );
    assert_eq!(response.headers()["x-qwen3-seed"], "42");

    let start = response
        .body_mut()
        .frame()
        .await
        .unwrap()
        .unwrap()
        .into_data()
        .unwrap();
    let start_marker = b"\"type\":\"start\"";
    assert!(
        start
            .windows(start_marker.len())
            .any(|window| window == start_marker)
    );

    let audio = response
        .body_mut()
        .frame()
        .await
        .unwrap()
        .unwrap()
        .into_data()
        .unwrap();
    assert!(audio.windows(9).any(|window| window == b"audio/pcm"));
    assert!(!state.completed.load(Ordering::Acquire));

    let remainder = response.into_body().collect().await.unwrap().to_bytes();
    let end_marker = b"\"type\":\"end\"";
    assert!(
        remainder
            .windows(end_marker.len())
            .any(|window| window == end_marker)
    );
    assert!(
        remainder
            .windows(b"\"finish_reason\":\"stop\"".len())
            .any(|window| window == b"\"finish_reason\":\"stop\"")
    );
    assert!(state.completed.load(Ordering::Acquire));
}

#[tokio::test]
async fn unread_stream_applies_one_packet_backpressure_and_drop_cancels() {
    let engine = FakeEngine::new(Duration::from_secs(2));
    let state = Arc::clone(&engine.state);
    let app = build_router(Arc::new(engine), ServerConfig::default()).unwrap();
    let response = app
        .oneshot(json_request("/v1/voice-design/speech", &speech_json()))
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(state.polls.load(Ordering::Relaxed), 1);
    drop(response);
    tokio::time::timeout(Duration::from_secs(1), async {
        while !state.cancelled.load(Ordering::Acquire) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn buffered_mode_returns_a_final_length_correct_wav() {
    let engine = FakeEngine::new(Duration::ZERO);
    let state = Arc::clone(&engine.state);
    let app = build_router(Arc::new(engine), ServerConfig::default()).unwrap();
    let mut payload = speech_json();
    payload["stream"] = json!(false);
    let response = app
        .oneshot(json_request("/v1/voice-design/speech", &payload))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CONTENT_TYPE], "audio/wav");
    assert_eq!(response.headers()["x-finish-reason"], "stop");
    let declared = response.headers()[header::CONTENT_LENGTH]
        .to_str()
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(declared, bytes.len());
    assert_eq!(&bytes[..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    assert_eq!(
        u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
        24_000
    );
    assert_eq!(
        u32::from_le_bytes(bytes[40..44].try_into().unwrap()) as usize,
        2 * SAMPLES_PER_CODEC_FRAME as usize * 2
    );
    assert_eq!(
        *lock_unpoisoned(&state.captured_language),
        Some(Language::German)
    );
}

#[tokio::test]
async fn length_finish_reason_is_preserved_in_streaming_and_wav_outputs() {
    let streaming_engine = FakeEngine::new(Duration::ZERO);
    *lock_unpoisoned(&streaming_engine.state.finish_reason) = EngineFinishReason::Length;
    let streaming_app = build_router(Arc::new(streaming_engine), ServerConfig::default()).unwrap();
    let streaming = streaming_app
        .oneshot(json_request("/v1/voice-design/speech", &speech_json()))
        .await
        .unwrap()
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
    assert!(
        streaming
            .windows(b"\"finish_reason\":\"length\"".len())
            .any(|window| window == b"\"finish_reason\":\"length\"")
    );

    let buffered_engine = FakeEngine::new(Duration::ZERO);
    *lock_unpoisoned(&buffered_engine.state.finish_reason) = EngineFinishReason::Length;
    let buffered_app = build_router(Arc::new(buffered_engine), ServerConfig::default()).unwrap();
    let mut payload = speech_json();
    payload["stream"] = json!(false);
    let buffered = buffered_app
        .oneshot(json_request("/v1/voice-design/speech", &payload))
        .await
        .unwrap();
    assert_eq!(buffered.status(), StatusCode::OK);
    assert_eq!(buffered.headers()["x-finish-reason"], "length");
}

#[tokio::test]
async fn delete_signals_the_active_native_request() {
    let engine = FakeEngine::new(Duration::from_secs(2));
    let state = Arc::clone(&engine.state);
    let app = build_router(Arc::new(engine), ServerConfig::default()).unwrap();
    let response = app
        .clone()
        .oneshot(json_request("/v1/voice-design/speech", &speech_json()))
        .await
        .unwrap();
    let request_id = response.headers()["x-request-id"].to_str().unwrap();
    let cancellation = app
        .oneshot(
            Request::delete(format!("/v1/requests/{request_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(cancellation.status(), StatusCode::ACCEPTED);
    tokio::time::timeout(Duration::from_secs(1), async {
        while !state.cancelled.load(Ordering::Acquire) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    drop(response);
}

#[tokio::test]
async fn client_chosen_id_cancels_buffered_request_before_response_headers() {
    let engine = FakeEngine::new(Duration::from_millis(250));
    let state = Arc::clone(&engine.state);
    let app = build_router(Arc::new(engine), ServerConfig::default()).unwrap();
    let request_id = "0198f65d-a679-7411-8f7c-151dbf0486be";
    let mut payload = speech_json();
    payload["stream"] = json!(false);
    let request = Request::builder()
        .method("POST")
        .uri("/v1/voice-design/speech")
        .header(header::CONTENT_TYPE, "application/json")
        .header("x-request-id", request_id)
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();
    let request_app = app.clone();
    let response_task = tokio::spawn(async move { request_app.oneshot(request).await.unwrap() });

    tokio::time::timeout(Duration::from_secs(1), async {
        while state.polls.load(Ordering::Acquire) == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    let cancellation = app
        .oneshot(
            Request::delete(format!("/v1/requests/{request_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(cancellation.status(), StatusCode::ACCEPTED);

    let response = response_task.await.unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
    assert_eq!(response.headers()["x-request-id"], request_id);
    assert_eq!(response_json(response).await["code"], "request_cancelled");
}

#[tokio::test]
async fn duplicate_client_request_id_never_starts_a_second_native_request() {
    let engine = FakeEngine::new(Duration::from_secs(2));
    let state = Arc::clone(&engine.state);
    let app = build_router(Arc::new(engine), ServerConfig::default()).unwrap();
    let request_id = "0198f65d-a679-7411-8f7c-151dbf0486bf";
    let make_request = || {
        Request::builder()
            .method("POST")
            .uri("/v1/voice-design/speech")
            .header(header::CONTENT_TYPE, "application/json")
            .header("x-request-id", request_id)
            .body(Body::from(serde_json::to_vec(&speech_json()).unwrap()))
            .unwrap()
    };
    let first = app.clone().oneshot(make_request()).await.unwrap();
    assert_eq!(first.status(), StatusCode::OK);

    let duplicate = app.oneshot(make_request()).await.unwrap();
    assert_eq!(duplicate.status(), StatusCode::CONFLICT);
    assert_eq!(response_json(duplicate).await["code"], "request_id_in_use");
    assert_eq!(state.starts.load(Ordering::Acquire), 1);
    drop(first);
}

#[tokio::test]
async fn global_shutdown_cancels_active_work_and_closes_admission() {
    let engine = FakeEngine::new(Duration::from_secs(2));
    let state = Arc::clone(&engine.state);
    let shutdown = ShutdownController::new();
    let app =
        build_router_with_shutdown(Arc::new(engine), ServerConfig::default(), shutdown.clone())
            .unwrap();
    let active_response = app
        .clone()
        .oneshot(json_request("/v1/voice-design/speech", &speech_json()))
        .await
        .unwrap();

    shutdown.cancel();
    let rejected = app
        .oneshot(json_request("/v1/voice-design/speech", &speech_json()))
        .await
        .unwrap();
    assert_eq!(rejected.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        response_json(rejected).await["code"],
        "server_shutting_down"
    );
    tokio::time::timeout(Duration::from_secs(1), async {
        while !state.cancelled.load(Ordering::Acquire) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    drop(active_response);
}

#[tokio::test]
async fn retirement_timeout_becomes_an_explicit_stuck_request_tombstone() {
    let engine = FakeEngine::new(Duration::ZERO);
    engine.state.retires.store(false, Ordering::Release);
    let config = ServerConfig {
        slow_client_timeout: Duration::from_millis(2),
        retirement_timeout: Duration::from_millis(3),
        shutdown_timeout: Duration::from_millis(10),
        ..ServerConfig::default()
    };
    let app = build_router(Arc::new(engine), config).unwrap();
    let response = app
        .clone()
        .oneshot(json_request("/v1/voice-design/speech", &speech_json()))
        .await
        .unwrap();
    let request_id = response.headers()["x-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert!(
        body.windows(b"native_retirement_timeout".len())
            .any(|window| window == b"native_retirement_timeout")
    );

    let retry = app
        .clone()
        .oneshot(
            Request::delete(format!("/v1/requests/{request_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(retry.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        response_json(retry).await["code"],
        "native_retirement_timeout"
    );

    let metrics = app
        .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
        .await
        .unwrap()
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
    let metrics = std::str::from_utf8(&metrics).unwrap();
    assert!(metrics.contains("qwen3_tts_active_requests 1"));
    assert!(metrics.contains("qwen3_tts_retirement_timeouts_total 1"));
}

#[tokio::test]
async fn buffered_egress_capacity_is_held_until_the_wav_body_is_dropped() {
    let engine = FakeEngine::new(Duration::ZERO);
    let state = Arc::clone(&engine.state);
    let config = ServerConfig {
        max_concurrent_requests: 1,
        ..ServerConfig::default()
    };
    let app = build_router(Arc::new(engine), config).unwrap();
    let mut payload = speech_json();
    payload["stream"] = json!(false);
    let first = app
        .clone()
        .oneshot(json_request("/v1/voice-design/speech", &payload))
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::OK);

    let rejected = app
        .clone()
        .oneshot(json_request("/v1/voice-design/speech", &payload))
        .await
        .unwrap();
    assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(state.starts.load(Ordering::Acquire), 1);
    drop(first);

    let accepted = app
        .oneshot(json_request("/v1/voice-design/speech", &payload))
        .await
        .unwrap();
    assert_eq!(accepted.status(), StatusCode::OK);
}

#[tokio::test]
async fn metrics_are_prometheus_text_without_prompt_labels() {
    let engine = FakeEngine::new(Duration::ZERO);
    let app = build_router(Arc::new(engine), ServerConfig::default()).unwrap();
    let response = app
        .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let body = std::str::from_utf8(&body).unwrap();
    assert!(body.contains("qwen3_tts_http_requests_total"));
    assert!(!body.contains("voice_description"));
    assert!(!body.contains("Guten Morgen"));
}

#[tokio::test]
async fn conservative_openai_alias_accepts_only_the_fixed_voice_design_model() {
    let engine = FakeEngine::new(Duration::ZERO);
    let app = build_router(Arc::new(engine), ServerConfig::default()).unwrap();
    let response = app
        .clone()
        .oneshot(json_request(
            "/v1/audio/speech",
            &json!({
                "model": "base-model-is-not-supported",
                "input": "Hello",
                "voice": "A calm voice",
                "response_format": "wav"
            }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(response_json(response).await["code"], "unsupported_model");

    let accepted = app
        .oneshot(json_request(
            "/v1/audio/speech",
            &json!({
                "model": "qwen3-tts-1.7b-voice-design",
                "input": "Hello",
                "voice": "A calm voice",
                "response_format": "wav"
            }),
        ))
        .await
        .unwrap();
    assert_eq!(accepted.status(), StatusCode::OK);
    assert_eq!(accepted.headers()[header::CONTENT_TYPE], "audio/wav");
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
