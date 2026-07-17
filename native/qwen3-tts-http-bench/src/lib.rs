//! Reproducible, prompt-safe HTTP benchmark client for the native Qwen3-TTS server.

mod error;
mod http;
mod model;
mod multipart;
mod phase_events;
mod report;
mod runner;
mod url;
mod wav;

pub use error::{BenchError, Result};
pub use model::{BackendProfile, BenchmarkConfig, Concurrency, WorkloadEntry};
pub use runner::run_benchmark;
