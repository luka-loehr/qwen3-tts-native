//! Reusable Rust host API for the native Qwen3-TTS speech-tokenizer decoder.
//!
//! The library parses decoder-only F32 or BF16 safetensors, owns the C weight
//! provider callback, dynamically loads the versioned native CUDA ABI, and
//! exposes safe packet-oriented Rust methods. The CUDA library path remains an
//! explicit deployment choice.
//!
//! ```no_run
//! use qwen3_tts_native_codec::{
//!     CODEBOOKS, DecoderWeights, NativeCodecLibrary,
//! };
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let native = NativeCodecLibrary::load(Path::new("libqwen3_tts_codec_cuda.so"))?;
//! let weights = DecoderWeights::open(Path::new("speech_tokenizer/model.safetensors"))?;
//! let mut stream = native.create_codec(0).map_err(std::io::Error::other)?;
//! stream.load_model(&weights).map_err(std::io::Error::other)?;
//! stream.warmup().map_err(std::io::Error::other)?;
//!
//! let frames = [[0_u16; CODEBOOKS]];
//! let (pcm, packet) = stream
//!     .process(&frames, true)
//!     .map_err(|(status, message)| std::io::Error::other(format!("{status}: {message}")))?;
//! assert_eq!(pcm.len(), packet.sample_count as usize);
//! # Ok(())
//! # }
//! ```

pub mod ffi;
pub mod model;

pub use ffi::{
    Api as NativeCodecLibrary, BatchOutput, CODEBOOKS, Codec as NativeCodec, MAX_BATCH_STREAMS,
    MAX_PACKET_FRAMES, MAX_PACKET_SAMPLES, ModelInfo, PacketResult, SAMPLES_PER_FRAME,
    STATUS_MODEL, STATUS_STATE, StateInfo,
};
pub use model::{
    DecoderWeightProvider, DecoderWeightTensor, SafetensorsFile as DecoderWeights,
    TensorDType as DecoderTensorDType, TensorEntry as DecoderTensorEntry,
};
