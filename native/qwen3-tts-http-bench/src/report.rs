use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::Serialize;
use serde_json::Value;

use crate::model::{
    BackendProfile, BenchmarkConfig, NormalizedSampling, SCHEMA_VERSION, SamplingAudit,
};
use crate::{BenchError, Result};

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PacketRecord {
    pub schema_version: &'static str,
    pub request_index: usize,
    pub workload_id: String,
    pub backend: BackendProfile,
    pub kind: PacketKind,
    pub sequence: u64,
    pub arrival_ms: f64,
    pub inter_arrival_ms: Option<f64>,
    pub payload_bytes: u64,
    pub payload_sha256: String,
    pub byte_offset: u64,
    pub first_codec_frame: Option<u64>,
    pub first_sample: Option<u64>,
    pub sample_count: Option<u64>,
    pub codec_frames: Option<u64>,
    pub is_first: bool,
    pub is_final: Option<bool>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PacketKind {
    NativeAudioPacket,
    RawPcmTransportArrival,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct FailureRecord {
    pub code: String,
    pub message: String,
    pub response_body_bytes: Option<u64>,
    pub response_body_sha256: Option<String>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(transparent)]
pub(crate) struct SamplingParityQualifying(bool);

impl SamplingParityQualifying {
    const fn get(self) -> bool {
        self.0
    }
}

impl From<bool> for SamplingParityQualifying {
    fn from(value: bool) -> Self {
        Self(value)
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RequestRecord {
    pub schema_version: &'static str,
    pub request_index: usize,
    pub workload_id: String,
    pub backend: BackendProfile,
    pub text_sha256: String,
    pub voice_description_sha256: String,
    pub request_body_sha256: String,
    pub normalized_sampling: NormalizedSampling,
    pub normalized_sampling_sha256: String,
    pub sampling_parity_qualifying: SamplingParityQualifying,
    pub sampling_parity_non_qualifying_reasons: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_description: Option<String>,
    pub language: String,
    pub streaming: bool,
    pub success: bool,
    pub http_status: Option<u16>,
    pub server_request_id: Option<String>,
    pub server_seed: Option<String>,
    pub ttfa_ms: Option<f64>,
    pub wall_ms: Option<f64>,
    pub sample_rate_hz: Option<u64>,
    pub samples: Option<u64>,
    pub audio_sha256: Option<String>,
    pub audio_seconds: Option<f64>,
    pub rtf: Option<f64>,
    pub response_bytes: Option<u64>,
    pub packet_count: u64,
    pub continuity_valid: bool,
    pub final_flag_seen: Option<bool>,
    pub finish_reason: Option<String>,
    pub natural_eos: Option<bool>,
    pub length_limited: Option<bool>,
    pub end_metrics: Option<Value>,
    pub failure: Option<FailureRecord>,
}

pub(crate) struct FailureMetadata {
    pub index: usize,
    pub backend: BackendProfile,
    pub workload_id: String,
    pub text_sha256: String,
    pub voice_description_sha256: String,
    pub request_body_sha256: String,
    pub sampling_audit: SamplingAudit,
    pub language: String,
    pub streaming: bool,
    pub log_text: Option<(String, String)>,
}

impl RequestRecord {
    #[must_use]
    pub(crate) fn failure(
        metadata: FailureMetadata,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        let (text, voice_description) = metadata
            .log_text
            .map_or((None, None), |(text, voice)| (Some(text), Some(voice)));
        Self {
            schema_version: SCHEMA_VERSION,
            request_index: metadata.index,
            workload_id: metadata.workload_id,
            backend: metadata.backend,
            text_sha256: metadata.text_sha256,
            voice_description_sha256: metadata.voice_description_sha256,
            request_body_sha256: metadata.request_body_sha256,
            normalized_sampling: metadata.sampling_audit.normalized,
            normalized_sampling_sha256: metadata.sampling_audit.normalized_sha256,
            sampling_parity_qualifying: metadata.sampling_audit.parity_qualifying.into(),
            sampling_parity_non_qualifying_reasons: metadata.sampling_audit.non_qualifying_reasons,
            text,
            voice_description,
            language: metadata.language,
            streaming: metadata.streaming,
            success: false,
            http_status: None,
            server_request_id: None,
            server_seed: None,
            ttfa_ms: None,
            wall_ms: None,
            sample_rate_hz: None,
            samples: None,
            audio_sha256: None,
            audio_seconds: None,
            rtf: None,
            response_bytes: None,
            packet_count: 0,
            continuity_valid: false,
            final_flag_seen: None,
            finish_reason: None,
            natural_eos: None,
            length_limited: None,
            end_metrics: None,
            failure: Some(FailureRecord {
                code: code.into(),
                message: message.into(),
                response_body_bytes: None,
                response_body_sha256: None,
            }),
        }
    }
}

#[derive(Debug, Serialize)]
struct Summary<'a> {
    schema_version: &'static str,
    endpoint: &'a str,
    backend: BackendProfile,
    sglang_model: Option<&'a str>,
    concurrency: &'static str,
    synchronized_batch_width: usize,
    warmups: usize,
    planned_requests: usize,
    completed_requests: usize,
    successful_requests: usize,
    failed_requests: usize,
    natural_eos_requests: usize,
    length_limited_requests: usize,
    eos_unknown_requests: usize,
    sampling_parity_qualifying_requests: usize,
    sampling_parity_non_qualifying_requests: usize,
    normalized_sampling_sha256s: Vec<String>,
    benchmark_wall_seconds: f64,
    attempted_requests_per_second: f64,
    throughput_requests_per_second: f64,
    total_audio_seconds: f64,
    aggregate_rtf: Option<f64>,
    summed_request_wall_rtf: Option<f64>,
    ttfa_ms: Option<Distribution>,
    wall_ms: Option<Distribution>,
    request_rtf: Option<Distribution>,
}

#[derive(Clone, Debug, Serialize)]
struct Distribution {
    count: usize,
    min: f64,
    mean: f64,
    p50: f64,
    p90: f64,
    p95: f64,
    p99: f64,
    max: f64,
}

pub(crate) fn write_reports(
    config: &BenchmarkConfig,
    records: &mut [RequestRecord],
    packets: &mut [PacketRecord],
    benchmark_wall_seconds: f64,
) -> Result<()> {
    records.sort_by_key(|record| record.request_index);
    packets.sort_by(|left, right| {
        left.request_index
            .cmp(&right.request_index)
            .then(left.sequence.cmp(&right.sequence))
    });
    fs::create_dir_all(&config.output_dir)?;
    let requests_path = config.output_dir.join("requests.jsonl");
    let packets_path = config.output_dir.join("packets.jsonl");
    let summary_path = config.output_dir.join("summary.json");
    for path in [&requests_path, &packets_path, &summary_path] {
        if path.exists() {
            return Err(BenchError::Configuration(format!(
                "refusing to overwrite existing report {}",
                path.display()
            )));
        }
    }
    write_jsonl(&requests_path, records)?;
    write_jsonl(&packets_path, packets)?;

    let successful: Vec<_> = records.iter().filter(|record| record.success).collect();
    let total_audio_seconds = successful
        .iter()
        .filter_map(|record| record.audio_seconds)
        .sum::<f64>();
    let summed_request_wall_seconds = successful
        .iter()
        .filter_map(|record| record.wall_ms)
        .sum::<f64>()
        / 1_000.0;
    let normalized_sampling_sha256s = records
        .iter()
        .map(|record| record.normalized_sampling_sha256.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let summary = Summary {
        schema_version: SCHEMA_VERSION,
        endpoint: &config.endpoint,
        backend: config.profile,
        sglang_model: config.sglang_model.as_deref(),
        concurrency: config.concurrency.label(),
        synchronized_batch_width: config.concurrency.get(),
        warmups: config.warmups,
        planned_requests: config.requests,
        completed_requests: records.len(),
        successful_requests: successful.len(),
        failed_requests: records.len() - successful.len(),
        natural_eos_requests: successful
            .iter()
            .filter(|record| record.natural_eos == Some(true))
            .count(),
        length_limited_requests: successful
            .iter()
            .filter(|record| record.length_limited == Some(true))
            .count(),
        eos_unknown_requests: successful
            .iter()
            .filter(|record| record.natural_eos.is_none())
            .count(),
        sampling_parity_qualifying_requests: records
            .iter()
            .filter(|record| record.sampling_parity_qualifying.get())
            .count(),
        sampling_parity_non_qualifying_requests: records
            .iter()
            .filter(|record| !record.sampling_parity_qualifying.get())
            .count(),
        normalized_sampling_sha256s,
        benchmark_wall_seconds,
        attempted_requests_per_second: requests_per_second(records.len(), benchmark_wall_seconds),
        throughput_requests_per_second: requests_per_second(
            successful.len(),
            benchmark_wall_seconds,
        ),
        total_audio_seconds,
        aggregate_rtf: ratio(benchmark_wall_seconds, total_audio_seconds),
        summed_request_wall_rtf: ratio(summed_request_wall_seconds, total_audio_seconds),
        ttfa_ms: distribution(successful.iter().filter_map(|record| record.ttfa_ms)),
        wall_ms: distribution(successful.iter().filter_map(|record| record.wall_ms)),
        request_rtf: distribution(successful.iter().filter_map(|record| record.rtf)),
    };
    let mut writer = create_new(&summary_path)?;
    serde_json::to_writer_pretty(&mut writer, &summary)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn requests_per_second(requests: usize, wall_seconds: f64) -> f64 {
    if wall_seconds > 0.0 {
        f64::from(u32::try_from(requests).expect("request count was validated to fit u32"))
            / wall_seconds
    } else {
        0.0
    }
}

fn ratio(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator > 0.0).then_some(numerator / denominator)
}

fn write_jsonl<T: Serialize>(path: &Path, values: &[T]) -> Result<()> {
    let mut writer = create_new(path)?;
    for value in values {
        serde_json::to_writer(&mut writer, value)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn create_new(path: &Path) -> Result<BufWriter<File>> {
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                BenchError::Configuration(format!(
                    "refusing to overwrite existing report {}",
                    path.display()
                ))
            } else {
                BenchError::Io(error)
            }
        })?;
    Ok(BufWriter::new(file))
}

fn distribution(values: impl Iterator<Item = f64>) -> Option<Distribution> {
    let mut values: Vec<f64> = values.filter(|value| value.is_finite()).collect();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let count = u32::try_from(values.len()).expect("report value count fits u32");
    let mean = values.iter().sum::<f64>() / f64::from(count);
    Some(Distribution {
        count: values.len(),
        min: values[0],
        mean,
        p50: percentile(&values, 50, 100),
        p90: percentile(&values, 90, 100),
        p95: percentile(&values, 95, 100),
        p99: percentile(&values, 99, 100),
        max: values[values.len() - 1],
    })
}

fn percentile(values: &[f64], numerator: u32, denominator: u32) -> f64 {
    let intervals = u64::try_from(values.len() - 1).expect("usize fits u64");
    let scaled_rank = intervals * u64::from(numerator);
    let lower =
        usize::try_from(scaled_rank / u64::from(denominator)).expect("percentile rank fits usize");
    let upper = (lower + 1).min(values.len() - 1);
    if lower == upper {
        values[lower]
    } else {
        let remainder = u32::try_from(scaled_rank % u64::from(denominator))
            .expect("percentile remainder fits u32");
        let weight = f64::from(remainder) / f64::from(denominator);
        values[lower] * (1.0 - weight) + values[upper] * weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_is_linearly_interpolated() {
        let values = [1.0, 2.0, 3.0, 4.0];
        assert!((percentile(&values, 50, 100) - 2.5).abs() < f64::EPSILON);
        assert!((percentile(&values, 95, 100) - 3.85).abs() < 1e-12);
    }

    #[test]
    fn aggregate_rtf_uses_scenario_wall_time_not_summed_request_time() {
        let benchmark_wall_seconds = 2.0;
        let summed_request_wall_seconds = 6.0;
        let total_audio_seconds = 10.0;

        assert_eq!(
            ratio(benchmark_wall_seconds, total_audio_seconds),
            Some(0.2)
        );
        assert_eq!(
            ratio(summed_request_wall_seconds, total_audio_seconds),
            Some(0.6)
        );
    }

    #[test]
    fn throughput_counts_only_successful_requests() {
        assert!((requests_per_second(4, 2.0) - 2.0).abs() < f64::EPSILON);
        assert!((requests_per_second(3, 2.0) - 1.5).abs() < f64::EPSILON);
        assert!(requests_per_second(3, 0.0).abs() < f64::EPSILON);
    }
}
