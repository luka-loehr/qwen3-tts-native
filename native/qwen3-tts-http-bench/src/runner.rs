use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::sync::Barrier;

use crate::http::{HttpBody, ResponseHead, TimedBytes};
use crate::model::{
    BackendProfile, BenchmarkConfig, SAMPLE_RATE_HZ, SCHEMA_VERSION, SamplingAudit, WorkloadEntry,
    load_workload, sha256_hex,
};
use crate::multipart::{MultipartReader, Part, boundary_from_content_type};
use crate::report::{FailureMetadata, PacketKind, PacketRecord, RequestRecord, write_reports};
use crate::url::Endpoint;
use crate::wav::validate_wav;
use crate::{BenchError, Result};

#[derive(Clone)]
struct PreparedRequest {
    index: usize,
    entry: WorkloadEntry,
    body: Vec<u8>,
    body_sha256: String,
    sampling_audit: SamplingAudit,
}

struct Outcome {
    record: RequestRecord,
    packets: Vec<PacketRecord>,
}

/// Executes warmups and synchronized measured batches, then writes JSON evidence.
///
/// # Errors
///
/// Returns an error for invalid configuration or workloads, failed warmups,
/// loopback resolution failures, and report I/O or serialization failures.
pub async fn run_benchmark(config: BenchmarkConfig) -> Result<()> {
    validate_config(&config)?;
    let endpoint = Arc::new(Endpoint::parse_loopback(&config.endpoint).await?);
    let workload = load_workload(&config.workload_path)?;

    let warmups = prepare_requests(&config, &workload, config.warmups)?;
    if !warmups.is_empty() {
        let outcomes = run_phase(&config, Arc::clone(&endpoint), warmups).await;
        if let Some(failure) = outcomes.iter().find(|outcome| !outcome.record.success) {
            let detail = failure
                .record
                .failure
                .as_ref()
                .map_or("unknown warmup failure", |item| item.message.as_str());
            return Err(BenchError::Validation(format!(
                "warmup request {:?} failed: {detail}",
                failure.record.workload_id
            )));
        }
    }

    let measured = prepare_requests(&config, &workload, config.requests)?;
    let benchmark_start = Instant::now();
    let outcomes = run_phase(&config, endpoint, measured).await;
    let benchmark_wall_seconds = benchmark_start.elapsed().as_secs_f64();
    let mut records = Vec::with_capacity(outcomes.len());
    let mut packets = Vec::new();
    for outcome in outcomes {
        records.push(outcome.record);
        packets.extend(outcome.packets);
    }
    write_reports(&config, &mut records, &mut packets, benchmark_wall_seconds)?;
    Ok(())
}

fn validate_config(config: &BenchmarkConfig) -> Result<()> {
    if config.requests == 0 {
        return Err(BenchError::Configuration(
            "--requests must be greater than zero".to_owned(),
        ));
    }
    if u32::try_from(config.requests).is_err() || u32::try_from(config.warmups).is_err() {
        return Err(BenchError::Configuration(
            "--requests and --warmups must each fit in an unsigned 32-bit integer".to_owned(),
        ));
    }
    if config.timeout_seconds == 0 {
        return Err(BenchError::Configuration(
            "--timeout-seconds must be greater than zero".to_owned(),
        ));
    }
    if config.profile == BackendProfile::SglangOmni
        && config.sglang_model.as_deref().is_none_or(str::is_empty)
    {
        return Err(BenchError::Configuration(
            "--sglang-model is required for the sglang-omni profile".to_owned(),
        ));
    }
    Ok(())
}

fn prepare_requests(
    config: &BenchmarkConfig,
    workload: &[WorkloadEntry],
    count: usize,
) -> Result<Vec<PreparedRequest>> {
    (0..count)
        .map(|index| {
            let entry = workload[index % workload.len()].clone();
            let body = entry.request_body(config.profile, config.sglang_model.as_deref())?;
            let body_sha256 = sha256_hex(&body);
            let sampling_audit = entry.sampling_audit();
            Ok(PreparedRequest {
                index,
                entry,
                body,
                body_sha256,
                sampling_audit,
            })
        })
        .collect()
}

async fn run_phase(
    config: &BenchmarkConfig,
    endpoint: Arc<Endpoint>,
    requests: Vec<PreparedRequest>,
) -> Vec<Outcome> {
    let mut outcomes = Vec::with_capacity(requests.len());
    for batch in requests.chunks(config.concurrency.get()) {
        outcomes.extend(run_batch(config, Arc::clone(&endpoint), batch).await);
    }
    outcomes
}

async fn run_batch(
    config: &BenchmarkConfig,
    endpoint: Arc<Endpoint>,
    requests: &[PreparedRequest],
) -> Vec<Outcome> {
    let mut connected = Vec::with_capacity(requests.len());
    let mut outcomes = Vec::new();
    for request in requests {
        let connection = tokio::time::timeout(
            Duration::from_secs(config.timeout_seconds),
            connect_endpoint(&endpoint),
        )
        .await;
        match connection {
            Ok(Ok(stream)) => connected.push((request.clone(), stream)),
            Ok(Err(error)) => outcomes.push(failed_outcome(
                config,
                request,
                "connect_error",
                error.to_string(),
            )),
            Err(_) => outcomes.push(failed_outcome(
                config,
                request,
                "connect_timeout",
                format!(
                    "connection timed out after {} seconds",
                    config.timeout_seconds
                ),
            )),
        }
    }
    if connected.is_empty() {
        return outcomes;
    }

    let barrier = Arc::new(Barrier::new(connected.len()));
    let mut tasks = Vec::with_capacity(connected.len());
    for (request, stream) in connected {
        let endpoint = Arc::clone(&endpoint);
        let barrier = Arc::clone(&barrier);
        let profile = config.profile;
        let timeout_seconds = config.timeout_seconds;
        let log_prompt_text = config.log_prompt_text;
        let task_request = request.clone();
        tasks.push((
            task_request,
            tokio::spawn(async move {
                let result = tokio::time::timeout(
                    Duration::from_secs(timeout_seconds),
                    execute_connected(
                        profile,
                        log_prompt_text,
                        endpoint,
                        request.clone(),
                        stream,
                        barrier,
                    ),
                )
                .await;
                match result {
                    Ok(Ok(outcome)) => outcome,
                    Ok(Err(error)) => failed_outcome_values(
                        profile,
                        log_prompt_text,
                        &request,
                        "request_error",
                        error.to_string(),
                    ),
                    Err(_) => failed_outcome_values(
                        profile,
                        log_prompt_text,
                        &request,
                        "request_timeout",
                        format!("request timed out after {timeout_seconds} seconds"),
                    ),
                }
            }),
        ));
    }
    for (request, task) in tasks {
        match task.await {
            Ok(outcome) => outcomes.push(outcome),
            Err(error) => outcomes.push(failed_outcome_values(
                config.profile,
                config.log_prompt_text,
                &request,
                "task_failure",
                error.to_string(),
            )),
        }
    }
    outcomes
}

async fn connect_endpoint(endpoint: &Endpoint) -> Result<TcpStream> {
    let mut last_error = None;
    for address in &endpoint.addresses {
        match TcpStream::connect(address).await {
            Ok(stream) => {
                stream.set_nodelay(true)?;
                return Ok(stream);
            }
            Err(error) => last_error = Some(error),
        }
    }
    Err(last_error.map_or_else(
        || BenchError::Configuration("endpoint has no resolved addresses".to_owned()),
        BenchError::Io,
    ))
}

async fn execute_connected(
    profile: BackendProfile,
    log_prompt_text: bool,
    endpoint: Arc<Endpoint>,
    request: PreparedRequest,
    mut stream: TcpStream,
    barrier: Arc<Barrier>,
) -> Result<Outcome> {
    let accept = match profile {
        BackendProfile::Native => "multipart/mixed, audio/wav",
        BackendProfile::SglangOmni => "audio/pcm, application/octet-stream",
    };
    let head = format!(
        "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nAccept: {accept}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        endpoint.path_and_query,
        endpoint.authority,
        request.body.len()
    );
    let mut wire_request = Vec::with_capacity(head.len() + request.body.len());
    wire_request.extend_from_slice(head.as_bytes());
    wire_request.extend_from_slice(&request.body);

    barrier.wait().await;
    let t0 = Instant::now();
    stream.write_all(&wire_request).await?;
    let (response_head, body) = HttpBody::read_head(stream).await?;
    if !(200..300).contains(&response_head.status) {
        return http_failure(profile, log_prompt_text, &request, response_head, body, t0).await;
    }
    match profile {
        BackendProfile::Native => {
            parse_native(log_prompt_text, &request, response_head, body, t0).await
        }
        BackendProfile::SglangOmni => {
            parse_sglang_raw(log_prompt_text, &request, response_head, body, t0).await
        }
    }
}

async fn http_failure(
    profile: BackendProfile,
    log_prompt_text: bool,
    request: &PreparedRequest,
    head: ResponseHead,
    mut body: HttpBody,
    t0: Instant,
) -> Result<Outcome> {
    let mut hasher = Sha256::new();
    let mut body_bytes = 0_u64;
    while let Some(segment) = body.next_segment().await? {
        body_bytes += u64::try_from(segment.bytes.len()).expect("usize fits u64");
        hasher.update(&segment.bytes);
    }
    let mut outcome = failed_outcome_values(
        profile,
        log_prompt_text,
        request,
        "http_status",
        format!("endpoint returned HTTP status {}", head.status),
    );
    outcome.record.http_status = Some(head.status);
    outcome.record.wall_ms = Some(milliseconds(t0.elapsed()));
    outcome.record.response_bytes = Some(body.response_bytes());
    outcome.record.server_request_id = head.header("x-request-id").map(str::to_owned);
    if let Some(failure) = outcome.record.failure.as_mut() {
        failure.response_body_bytes = Some(body_bytes);
        failure.response_body_sha256 = Some(hex::encode(hasher.finalize()));
    }
    Ok(outcome)
}

async fn parse_native(
    log_prompt_text: bool,
    request: &PreparedRequest,
    head: ResponseHead,
    body: HttpBody,
    t0: Instant,
) -> Result<Outcome> {
    if head.header("x-request-id").is_none() {
        return Err(BenchError::Validation(
            "native response has no x-request-id header".to_owned(),
        ));
    }
    let content_type = head
        .header("content-type")
        .ok_or_else(|| BenchError::Validation("response has no Content-Type".to_owned()))?;
    if request.entry.stream {
        let boundary = boundary_from_content_type(content_type)?;
        parse_native_multipart(log_prompt_text, request, head, body, t0, &boundary).await
    } else {
        if !media_type_is(content_type, "audio/wav") {
            return Err(BenchError::Validation(
                "buffered native response is not audio/wav".to_owned(),
            ));
        }
        parse_native_wav(log_prompt_text, request, head, body, t0).await
    }
}

async fn parse_native_multipart(
    log_prompt_text: bool,
    request: &PreparedRequest,
    head: ResponseHead,
    body: HttpBody,
    t0: Instant,
    boundary: &str,
) -> Result<Outcome> {
    let mut reader = MultipartReader::new(body, boundary);
    let mut packets = Vec::new();
    let mut state = NativeStreamState::new(
        head.header("x-request-id")
            .expect("parse_native requires x-request-id")
            .to_owned(),
    );

    while let Some(part) = reader.next_part().await? {
        let content_type = part.header("content-type").ok_or_else(|| {
            BenchError::Validation("multipart part has no Content-Type".to_owned())
        })?;
        if media_type_is(content_type, "application/json") {
            let event: JsonEvent = serde_json::from_slice(&part.payload)?;
            state.handle_json_event(event)?;
        } else if media_type_is(content_type, "audio/pcm") {
            validate_native_audio_content_type(content_type)?;
            packets.push(state.handle_audio_part(&part, request, t0)?);
        } else {
            return Err(BenchError::Validation(format!(
                "unsupported multipart part Content-Type {content_type:?}"
            )));
        }
    }
    state.validate_complete()?;
    let first_audio_at = state
        .first_audio_at
        .expect("completion requires first audio");
    let wall_ms = milliseconds(t0.elapsed());
    let audio_seconds = audio_duration_seconds(state.expected_sample, SAMPLE_RATE_HZ)?;
    let reason = state
        .finish_reason
        .as_ref()
        .expect("completion requires a finish reason");
    let mut record = successful_record(log_prompt_text, request, BackendProfile::Native, &head);
    record.ttfa_ms = Some(milliseconds(first_audio_at.saturating_duration_since(t0)));
    record.wall_ms = Some(wall_ms);
    record.sample_rate_hz = Some(SAMPLE_RATE_HZ);
    record.samples = Some(state.expected_sample);
    record.audio_sha256 = Some(hex::encode(state.audio_hasher.clone().finalize()));
    record.audio_seconds = Some(audio_seconds);
    record.rtf = Some((wall_ms / 1_000.0) / audio_seconds);
    record.response_bytes = Some(reader.response_bytes());
    record.packet_count = state.expected_sequence;
    record.continuity_valid = true;
    record.final_flag_seen = Some(true);
    record.finish_reason = Some(reason.clone());
    record.natural_eos = Some(reason == "stop");
    record.length_limited = Some(reason == "length");
    record.end_metrics = state.end_metrics;
    Ok(Outcome { record, packets })
}

struct NativeStreamState {
    server_request_id: String,
    expected_sequence: u64,
    expected_sample: u64,
    expected_codec_frame: u64,
    byte_offset: u64,
    first_audio_at: Option<Instant>,
    previous_audio_at: Option<Instant>,
    start_seen: bool,
    end_seen: bool,
    final_seen: bool,
    finish_reason: Option<String>,
    end_metrics: Option<Value>,
    audio_hasher: Sha256,
}

impl NativeStreamState {
    fn new(server_request_id: String) -> Self {
        Self {
            server_request_id,
            expected_sequence: 0,
            expected_sample: 0,
            expected_codec_frame: 0,
            byte_offset: 0,
            first_audio_at: None,
            previous_audio_at: None,
            start_seen: false,
            end_seen: false,
            final_seen: false,
            finish_reason: None,
            end_metrics: None,
            audio_hasher: Sha256::new(),
        }
    }

    fn handle_json_event(&mut self, event: JsonEvent) -> Result<()> {
        validate_event_request_id(&self.server_request_id, event.request_id.as_deref())?;
        match event.event_type.as_str() {
            "start" if !self.start_seen && self.expected_sequence == 0 && !self.end_seen => {
                validate_start_audio(event.audio.as_ref())?;
                self.start_seen = true;
            }
            "end" if self.start_seen && self.expected_sequence > 0 && !self.end_seen => {
                let reason = event.finish_reason.ok_or_else(|| {
                    BenchError::Validation("end event has no finish_reason".to_owned())
                })?;
                validate_native_finish_reason(&reason)?;
                validate_end_metrics(
                    event.metrics.as_ref(),
                    self.expected_sequence,
                    self.expected_sample,
                )?;
                self.finish_reason = Some(reason);
                self.end_metrics = event.metrics;
                self.end_seen = true;
            }
            _ => {
                return Err(BenchError::Validation(
                    "multipart JSON events are missing, duplicated, or out of order".to_owned(),
                ));
            }
        }
        Ok(())
    }

    fn handle_audio_part(
        &mut self,
        part: &Part,
        request: &PreparedRequest,
        t0: Instant,
    ) -> Result<PacketRecord> {
        if !self.start_seen || self.end_seen || self.final_seen {
            return Err(BenchError::Validation(
                "audio packet is outside the start/end event interval".to_owned(),
            ));
        }
        let arrival = part
            .first_payload_at
            .ok_or_else(|| BenchError::Validation("audio packet payload is empty".to_owned()))?;
        let sequence = part_u64(part, "x-sequence")?;
        let first_codec_frame = part_u64(part, "x-first-codec-frame")?;
        let first_sample = part_u64(part, "x-first-sample")?;
        let sample_count = part_u64(part, "x-sample-count")?;
        let codec_frames = part_u64(part, "x-codec-frames")?;
        let is_final = part_bool(part, "x-final")?;
        let payload_bytes = u64::try_from(part.payload.len()).expect("usize fits u64");
        if sequence != self.expected_sequence
            || first_sample != self.expected_sample
            || first_codec_frame != self.expected_codec_frame
            || sample_count == 0
            || codec_frames == 0
            || sample_count.checked_mul(2) != Some(payload_bytes)
        {
            return Err(BenchError::Validation(
                "native audio packet sequence/sample/frame continuity is invalid".to_owned(),
            ));
        }
        self.first_audio_at.get_or_insert(arrival);
        self.audio_hasher.update(&part.payload);
        let inter_arrival_ms = self
            .previous_audio_at
            .map(|previous| milliseconds(arrival.saturating_duration_since(previous)));
        let record = PacketRecord {
            schema_version: SCHEMA_VERSION,
            request_index: request.index,
            workload_id: request.entry.id.clone(),
            backend: BackendProfile::Native,
            kind: PacketKind::NativeAudioPacket,
            sequence,
            arrival_ms: milliseconds(arrival.saturating_duration_since(t0)),
            inter_arrival_ms,
            payload_bytes,
            payload_sha256: sha256_hex(&part.payload),
            byte_offset: self.byte_offset,
            first_codec_frame: Some(first_codec_frame),
            first_sample: Some(first_sample),
            sample_count: Some(sample_count),
            codec_frames: Some(codec_frames),
            is_first: sequence == 0,
            is_final: Some(is_final),
        };
        self.expected_sequence += 1;
        self.expected_sample += sample_count;
        self.expected_codec_frame += codec_frames;
        self.byte_offset += payload_bytes;
        self.previous_audio_at = Some(arrival);
        self.final_seen = is_final;
        Ok(record)
    }

    fn validate_complete(&self) -> Result<()> {
        if !self.start_seen || !self.end_seen || !self.final_seen || self.first_audio_at.is_none() {
            return Err(BenchError::Validation(
                "multipart response lacks a start event, PCM payload, final audio flag, or end event"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

async fn parse_native_wav(
    log_prompt_text: bool,
    request: &PreparedRequest,
    head: ResponseHead,
    mut body: HttpBody,
    t0: Instant,
) -> Result<Outcome> {
    let mut wav = Vec::new();
    let mut arrivals = Vec::new();
    while let Some(segment) = body.next_segment().await? {
        let start = wav.len();
        wav.extend_from_slice(&segment.bytes);
        arrivals.push((start, wav.len(), segment.arrived));
    }
    let info = validate_wav(&wav)?;
    let first_pcm_at = arrivals
        .iter()
        .find(|(start, end, _)| *start <= info.data_offset && info.data_offset < *end)
        .map(|(_, _, arrived)| *arrived)
        .ok_or_else(|| BenchError::Wav("could not timestamp first PCM byte".to_owned()))?;
    let finish_reason = head
        .header("x-finish-reason")
        .ok_or_else(|| BenchError::Validation("WAV response has no x-finish-reason".to_owned()))?
        .to_owned();
    validate_native_finish_reason(&finish_reason)?;
    let wall_ms = milliseconds(t0.elapsed());
    let audio_seconds = audio_duration_seconds(info.sample_count, SAMPLE_RATE_HZ)?;
    let mut record = successful_record(log_prompt_text, request, BackendProfile::Native, &head);
    record.ttfa_ms = Some(milliseconds(first_pcm_at.saturating_duration_since(t0)));
    record.wall_ms = Some(wall_ms);
    record.sample_rate_hz = Some(SAMPLE_RATE_HZ);
    record.samples = Some(info.sample_count);
    record.audio_sha256 = Some(sha256_hex(
        &wav[info.data_offset..info.data_offset + info.data_bytes],
    ));
    record.audio_seconds = Some(audio_seconds);
    record.rtf = Some((wall_ms / 1_000.0) / audio_seconds);
    record.response_bytes = Some(body.response_bytes());
    record.packet_count = 0;
    record.continuity_valid = true;
    record.final_flag_seen = None;
    record.finish_reason = Some(finish_reason.clone());
    record.natural_eos = Some(finish_reason == "stop");
    record.length_limited = Some(finish_reason == "length");
    Ok(Outcome {
        record,
        packets: Vec::new(),
    })
}

async fn parse_sglang_raw(
    log_prompt_text: bool,
    request: &PreparedRequest,
    head: ResponseHead,
    mut body: HttpBody,
    t0: Instant,
) -> Result<Outcome> {
    let content_type = head
        .header("content-type")
        .ok_or_else(|| BenchError::Validation("response has no Content-Type".to_owned()))?;
    if !media_type_is(content_type, "audio/pcm")
        && !media_type_is(content_type, "application/octet-stream")
    {
        return Err(BenchError::Validation(format!(
            "SGLang-Omni streaming response has unsupported Content-Type {content_type:?}"
        )));
    }
    let sample_rate = sample_rate_from_headers(&head, content_type)?;
    let mut chunks: Vec<TimedBytes> = Vec::new();
    while let Some(segment) = body.next_segment().await? {
        if segment.bytes.is_empty() {
            continue;
        }
        if let Some(previous) = chunks.last_mut()
            && previous.arrived == segment.arrived
        {
            previous.bytes.extend_from_slice(&segment.bytes);
            continue;
        }
        chunks.push(segment);
    }
    let total_bytes = chunks.iter().map(|chunk| chunk.bytes.len()).sum::<usize>();
    if total_bytes == 0 || !total_bytes.is_multiple_of(2) {
        return Err(BenchError::Validation(
            "SGLang-Omni raw PCM must contain a non-empty whole number of 16-bit samples"
                .to_owned(),
        ));
    }
    let first_audio_at = chunks[0].arrived;
    let mut packets = Vec::with_capacity(chunks.len());
    let mut audio_hasher = Sha256::new();
    let mut byte_offset = 0_u64;
    let mut previous_at = None;
    for (sequence, chunk) in chunks.iter().enumerate() {
        audio_hasher.update(&chunk.bytes);
        let payload_bytes = u64::try_from(chunk.bytes.len()).expect("usize fits u64");
        let aligned = byte_offset.is_multiple_of(2) && payload_bytes.is_multiple_of(2);
        packets.push(PacketRecord {
            schema_version: SCHEMA_VERSION,
            request_index: request.index,
            workload_id: request.entry.id.clone(),
            backend: BackendProfile::SglangOmni,
            kind: PacketKind::RawPcmTransportArrival,
            sequence: u64::try_from(sequence).expect("usize fits u64"),
            arrival_ms: milliseconds(chunk.arrived.saturating_duration_since(t0)),
            inter_arrival_ms: previous_at.map(|previous: Instant| {
                milliseconds(chunk.arrived.saturating_duration_since(previous))
            }),
            payload_bytes,
            payload_sha256: sha256_hex(&chunk.bytes),
            byte_offset,
            first_codec_frame: None,
            first_sample: aligned.then_some(byte_offset / 2),
            sample_count: aligned.then_some(payload_bytes / 2),
            codec_frames: None,
            is_first: sequence == 0,
            // Raw PCM has no application-level terminal flag or sentinel.
            is_final: None,
        });
        byte_offset += payload_bytes;
        previous_at = Some(chunk.arrived);
    }
    let samples = u64::try_from(total_bytes / 2).expect("usize fits u64");
    let audio_seconds = audio_duration_seconds(samples, sample_rate)?;
    let wall_ms = milliseconds(t0.elapsed());
    let mut record = successful_record(log_prompt_text, request, BackendProfile::SglangOmni, &head);
    record.ttfa_ms = Some(milliseconds(first_audio_at.saturating_duration_since(t0)));
    record.wall_ms = Some(wall_ms);
    record.sample_rate_hz = Some(sample_rate);
    record.samples = Some(samples);
    record.audio_sha256 = Some(hex::encode(audio_hasher.finalize()));
    record.audio_seconds = Some(audio_seconds);
    record.rtf = Some((wall_ms / 1_000.0) / audio_seconds);
    record.response_bytes = Some(body.response_bytes());
    record.packet_count = u64::try_from(chunks.len()).expect("usize fits u64");
    record.continuity_valid = true;
    record.final_flag_seen = None;
    record.finish_reason = None;
    record.natural_eos = None;
    record.length_limited = None;
    Ok(Outcome { record, packets })
}

fn successful_record(
    log_prompt_text: bool,
    request: &PreparedRequest,
    backend: BackendProfile,
    head: &ResponseHead,
) -> RequestRecord {
    RequestRecord {
        schema_version: SCHEMA_VERSION,
        request_index: request.index,
        workload_id: request.entry.id.clone(),
        backend,
        text_sha256: request.entry.text_sha256(),
        voice_description_sha256: request.entry.voice_description_sha256(),
        request_body_sha256: request.body_sha256.clone(),
        normalized_sampling: request.sampling_audit.normalized.clone(),
        normalized_sampling_sha256: request.sampling_audit.normalized_sha256.clone(),
        sampling_parity_qualifying: request.sampling_audit.parity_qualifying.into(),
        sampling_parity_non_qualifying_reasons: request
            .sampling_audit
            .non_qualifying_reasons
            .clone(),
        text: log_prompt_text.then(|| request.entry.text.clone()),
        voice_description: log_prompt_text.then(|| request.entry.voice_description.clone()),
        language: request.entry.language.clone(),
        streaming: request.entry.stream,
        success: true,
        http_status: Some(head.status),
        server_request_id: head.header("x-request-id").map(str::to_owned),
        server_seed: head.header("x-qwen3-seed").map(str::to_owned),
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
        failure: None,
    }
}

fn failed_outcome(
    config: &BenchmarkConfig,
    request: &PreparedRequest,
    code: &str,
    message: String,
) -> Outcome {
    failed_outcome_values(
        config.profile,
        config.log_prompt_text,
        request,
        code,
        message,
    )
}

fn failed_outcome_values(
    profile: BackendProfile,
    log_prompt_text: bool,
    request: &PreparedRequest,
    code: &str,
    message: String,
) -> Outcome {
    let log_text = log_prompt_text.then(|| {
        (
            request.entry.text.clone(),
            request.entry.voice_description.clone(),
        )
    });
    Outcome {
        record: RequestRecord::failure(
            FailureMetadata {
                index: request.index,
                backend: profile,
                workload_id: request.entry.id.clone(),
                text_sha256: request.entry.text_sha256(),
                voice_description_sha256: request.entry.voice_description_sha256(),
                request_body_sha256: request.body_sha256.clone(),
                sampling_audit: request.sampling_audit.clone(),
                language: request.entry.language.clone(),
                streaming: request.entry.stream,
                log_text,
            },
            code,
            message,
        ),
        packets: Vec::new(),
    }
}

#[derive(Debug, Deserialize)]
struct JsonEvent {
    #[serde(rename = "type")]
    event_type: String,
    request_id: Option<String>,
    finish_reason: Option<String>,
    metrics: Option<Value>,
    audio: Option<JsonAudioDescription>,
}

#[derive(Debug, Deserialize)]
struct JsonAudioDescription {
    encoding: String,
    sample_rate_hz: u64,
    channels: u64,
    samples_per_codec_frame: u64,
}

fn validate_event_request_id(header: &str, event: Option<&str>) -> Result<()> {
    let event = event.ok_or_else(|| {
        BenchError::Validation("multipart JSON event has no request_id".to_owned())
    })?;
    if header != event {
        return Err(BenchError::Validation(
            "multipart event request_id differs from response header".to_owned(),
        ));
    }
    Ok(())
}

fn validate_start_audio(audio: Option<&JsonAudioDescription>) -> Result<()> {
    let audio = audio
        .ok_or_else(|| BenchError::Validation("start event has no audio description".to_owned()))?;
    if audio.encoding != "pcm_s16le"
        || audio.sample_rate_hz != SAMPLE_RATE_HZ
        || audio.channels != 1
        || audio.samples_per_codec_frame != 1_920
    {
        return Err(BenchError::Validation(
            "start event audio description is not native PCM16 mono at 24 kHz".to_owned(),
        ));
    }
    Ok(())
}

fn validate_native_audio_content_type(content_type: &str) -> Result<()> {
    let rate =
        content_type_parameter(content_type, "rate").and_then(|value| value.parse::<u64>().ok());
    let channels = content_type_parameter(content_type, "channels")
        .and_then(|value| value.parse::<u64>().ok());
    let format = content_type_parameter(content_type, "format");
    if rate != Some(SAMPLE_RATE_HZ)
        || channels != Some(1)
        || format.is_none_or(|value| !value.eq_ignore_ascii_case("s16le"))
    {
        return Err(BenchError::Validation(
            "native audio part Content-Type is not PCM16 mono at 24 kHz".to_owned(),
        ));
    }
    Ok(())
}

fn validate_native_finish_reason(reason: &str) -> Result<()> {
    if !matches!(reason, "stop" | "length") {
        return Err(BenchError::Validation(format!(
            "native finish reason {reason:?} is unsupported"
        )));
    }
    Ok(())
}

fn audio_duration_seconds(samples: u64, sample_rate: u64) -> Result<f64> {
    let samples = u32::try_from(samples).map_err(|_| {
        BenchError::Validation("PCM sample count exceeds the reportable range".to_owned())
    })?;
    let sample_rate = u32::try_from(sample_rate).map_err(|_| {
        BenchError::Validation("PCM sample rate exceeds the reportable range".to_owned())
    })?;
    Ok(f64::from(samples) / f64::from(sample_rate))
}

fn validate_end_metrics(metrics: Option<&Value>, packets: u64, samples: u64) -> Result<()> {
    let metrics = metrics
        .and_then(Value::as_object)
        .ok_or_else(|| BenchError::Validation("end event metrics are missing".to_owned()))?;
    if metrics.get("emitted_packets").and_then(Value::as_u64) != Some(packets)
        || metrics.get("emitted_samples").and_then(Value::as_u64) != Some(samples)
    {
        return Err(BenchError::Validation(
            "end event metrics disagree with observed packets or samples".to_owned(),
        ));
    }
    Ok(())
}

fn part_u64(part: &Part, name: &str) -> Result<u64> {
    part.header(name)
        .ok_or_else(|| BenchError::Validation(format!("audio part has no {name} header")))?
        .parse::<u64>()
        .map_err(|_| BenchError::Validation(format!("audio part {name} header is invalid")))
}

fn part_bool(part: &Part, name: &str) -> Result<bool> {
    part.header(name)
        .ok_or_else(|| BenchError::Validation(format!("audio part has no {name} header")))?
        .parse::<bool>()
        .map_err(|_| BenchError::Validation(format!("audio part {name} header is invalid")))
}

fn sample_rate_from_headers(head: &ResponseHead, content_type: &str) -> Result<u64> {
    let rate = head
        .header("x-sample-rate")
        .map(str::to_owned)
        .or_else(|| content_type_parameter(content_type, "rate").map(str::to_owned))
        .ok_or_else(|| {
            BenchError::Validation(
                "raw PCM response must declare x-sample-rate or Content-Type rate".to_owned(),
            )
        })?
        .parse::<u64>()
        .map_err(|_| BenchError::Validation("raw PCM sample rate is invalid".to_owned()))?;
    if rate == 0 || rate > 768_000 {
        return Err(BenchError::Validation(
            "raw PCM sample rate is outside 1..=768000 Hz".to_owned(),
        ));
    }
    Ok(rate)
}

fn content_type_parameter<'a>(content_type: &'a str, expected_name: &str) -> Option<&'a str> {
    content_type.split(';').skip(1).find_map(|parameter| {
        let (name, value) = parameter.trim().split_once('=')?;
        name.trim()
            .eq_ignore_ascii_case(expected_name)
            .then(|| value.trim().trim_matches('"'))
    })
}

fn media_type_is(value: &str, expected: &str) -> bool {
    value
        .split(';')
        .next()
        .is_some_and(|media_type| media_type.trim().eq_ignore_ascii_case(expected))
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn raw_pcm_rate_must_be_declared() {
        let missing = ResponseHead {
            status: 200,
            headers: BTreeMap::new(),
        };
        assert!(sample_rate_from_headers(&missing, "audio/pcm").is_err());
        let from_type = ResponseHead {
            status: 200,
            headers: BTreeMap::new(),
        };
        assert_eq!(
            sample_rate_from_headers(&from_type, "audio/pcm; rate=24000").unwrap(),
            24_000
        );
    }
}
