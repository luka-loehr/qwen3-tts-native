use serde::Serialize;

use crate::ffi::RequestMetrics;

#[derive(Debug, Serialize)]
pub struct QualificationReport {
    pub schema_version: u32,
    pub qualifying_run: bool,
    pub runtime_abi_version: u32,
    pub model_root: String,
    pub library: String,
    pub packet_frames: u32,
    pub samples_per_packet_capacity: usize,
    pub corpus_entries: usize,
    pub warmup_requests: usize,
    pub requests_per_concurrency: usize,
    pub preflight: PreflightReport,
    pub scenarios: Vec<ScenarioReport>,
    pub gates: QualificationGates,
}

#[derive(Debug, Serialize)]
pub struct PreflightReport {
    pub configured_capacity: usize,
    pub capacity_filled: bool,
    pub overflow_returned_would_block: bool,
    pub all_requests_cancelled_and_destroyed: bool,
    pub capacity_recovered_after_cancellation: bool,
    pub passed: bool,
}

#[derive(Debug, Serialize)]
pub struct QualificationGates {
    pub all_requests_completed: bool,
    pub at_least_200_requests_per_scenario: bool,
    pub progressive_streaming_observed: bool,
    pub packet_positions_contiguous: bool,
    pub exact_pcm_copy_bounds: bool,
    pub backpressure_and_cancellation: bool,
    pub rtf_below_one_all_scenarios: bool,
    pub first_audio_p95_below_200_ms_all_scenarios: bool,
    pub passed: bool,
}

#[derive(Debug, Serialize)]
pub struct ScenarioReport {
    pub concurrency: usize,
    pub requested: usize,
    pub completed: usize,
    pub failed: usize,
    pub wall_seconds: f64,
    pub synthesized_audio_seconds: f64,
    pub aggregate_rtf: f64,
    pub requests_per_second: f64,
    pub progressive_streaming_requests: usize,
    pub exact_copy_bound_requests: usize,
    pub host_rss_start_bytes: Option<u64>,
    pub host_rss_peak_bytes: Option<u64>,
    pub host_rss_end_bytes: Option<u64>,
    pub caller_ttfa_ms: Distribution,
    pub request_wall_ms: Distribution,
    pub request_rtf: Distribution,
    pub runtime_first_audio_ms: Distribution,
    pub runtime_prefill_ms: Distribution,
    pub peak_request_device_bytes: u64,
    pub peak_request_host_bytes: u64,
    pub requests: Vec<RequestReport>,
}

#[derive(Debug, Serialize)]
pub struct RequestReport {
    pub ordinal: usize,
    pub corpus_id: String,
    pub language: String,
    pub packets: u64,
    pub codec_frames: u64,
    pub samples: u64,
    pub audio_seconds: f64,
    pub caller_ttfa_ms: f64,
    pub caller_wall_ms: f64,
    pub rtf: f64,
    pub progressive_streaming: bool,
    pub exact_copy_bounds: bool,
    pub runtime: RuntimeMetricsReport,
}

#[derive(Debug, Serialize)]
pub struct RuntimeMetricsReport {
    pub queue_ms: f64,
    pub prefill_ms: f64,
    pub first_codec_frame_ms: f64,
    pub first_audio_ms: f64,
    pub wall_ms: f64,
    pub talker_gpu_ms: f64,
    pub codec_gpu_ms: f64,
    pub peak_request_device_bytes: u64,
    pub peak_request_host_bytes: u64,
}

impl From<RequestMetrics> for RuntimeMetricsReport {
    fn from(metrics: RequestMetrics) -> Self {
        Self {
            queue_ms: micros_to_millis(metrics.queue_microseconds as f64),
            prefill_ms: micros_to_millis(metrics.prefill_microseconds as f64),
            first_codec_frame_ms: micros_to_millis(metrics.first_codec_frame_microseconds as f64),
            first_audio_ms: micros_to_millis(metrics.first_audio_microseconds as f64),
            wall_ms: micros_to_millis(metrics.wall_microseconds as f64),
            talker_gpu_ms: micros_to_millis(metrics.talker_gpu_microseconds),
            codec_gpu_ms: micros_to_millis(metrics.codec_gpu_microseconds),
            peak_request_device_bytes: metrics.peak_request_device_bytes,
            peak_request_host_bytes: metrics.peak_request_host_bytes,
        }
    }
}

fn micros_to_millis(value: f64) -> f64 {
    value / 1_000.0
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Distribution {
    pub minimum: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub maximum: f64,
    pub mean: f64,
}

impl Distribution {
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        Self {
            minimum: sorted[0],
            p50: percentile(&sorted, 0.50),
            p95: percentile(&sorted, 0.95),
            p99: percentile(&sorted, 0.99),
            maximum: sorted[sorted.len() - 1],
            mean: sorted.iter().sum::<f64>() / sorted.len() as f64,
        }
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    sorted[index]
}

#[cfg(test)]
mod tests {
    use super::Distribution;

    #[test]
    fn distributions_use_nearest_rank_without_hiding_tail_latency() {
        let distribution = Distribution::from_values(&[5.0, 1.0, 3.0, 2.0, 4.0]);
        assert_eq!(distribution.minimum, 1.0);
        assert_eq!(distribution.p50, 3.0);
        assert_eq!(distribution.p95, 5.0);
        assert_eq!(distribution.maximum, 5.0);
        assert_eq!(distribution.mean, 3.0);
    }
}
