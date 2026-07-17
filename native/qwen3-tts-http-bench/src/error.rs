use std::io;

/// Result type used throughout the benchmark client.
pub type Result<T> = std::result::Result<T, BenchError>;

/// A benchmark configuration, transport, protocol, or validation failure.
#[derive(Debug, thiserror::Error)]
pub enum BenchError {
    #[error("configuration error: {0}")]
    Configuration(String),
    #[error("workload error: {0}")]
    Workload(String),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("HTTP protocol error: {0}")]
    Http(String),
    #[error("multipart protocol error: {0}")]
    Multipart(String),
    #[error("WAV validation error: {0}")]
    Wav(String),
    #[error("response validation error: {0}")]
    Validation(String),
    #[error("request timed out after {0} seconds")]
    Timeout(u64),
    #[error("timing evidence error: {0}")]
    Timing(String),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("task failed: {0}")]
    Task(String),
}
