#![allow(clippy::needless_pass_by_value)]

mod api;
mod config;
mod engine;
mod error;
mod metrics;
mod multipart;
mod server;
mod wav;

pub use api::{
    ACCEPTED_LANGUAGES, MODEL_ID, OpenAiResponseFormat, OpenAiSpeechRequestBody, OutputFormat,
    SamplingOptions, SamplingStrategy, SpeechRequestBody,
};
pub use config::{
    DEFAULT_MAX_DURATION_SECONDS, DEFAULT_MAX_TEXT_BYTES, DEFAULT_MAX_VOICE_DESCRIPTION_BYTES,
    INTRINSIC_MAX_DURATION_SECONDS, ServerConfig,
};
pub use engine::{
    EngineError, EngineErrorKind, EngineFinishReason, EngineMetrics, EnginePacket, EnginePoll,
    EngineSynthesisRequest, NativeEngineConfig, NativeRuntimeEngine, SpeechEngine, SpeechRequest,
};
pub use server::{ShutdownController, build_router, build_router_with_shutdown};
