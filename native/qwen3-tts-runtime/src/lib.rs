//! Native Qwen3-TTS request lifecycle and public ABI contract.
//!
//! This crate deliberately contains no placeholder neural inference. It owns
//! the invariants shared by the real talker, predictor, and codec backends.

mod request;
mod types;

pub use request::{PacketQueue, PacketQueueError, RequestRecord, TransitionError};
pub use types::{
    AudioPacketDescriptor, EngineConfig, GenerationConfig, Language, RequestInput,
    RequestInputError, RequestMetrics, RequestPhase, RuntimeStatus,
};

pub const ABI_VERSION_V1: u32 = 1;
pub const CODEBOOKS: u32 = 16;
pub const CODEC_FRAMES_PER_SECOND_NUMERATOR: u32 = 25;
pub const CODEC_FRAMES_PER_SECOND_DENOMINATOR: u32 = 2;
pub const SAMPLE_RATE: u32 = 24_000;
pub const SAMPLES_PER_CODEC_FRAME: u32 = 1_920;

#[unsafe(no_mangle)]
pub extern "C" fn qwen3_tts_runtime_abi_version_v1() -> u32 {
    ABI_VERSION_V1
}
