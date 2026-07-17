use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use qwen3_tts_http_bench::{BackendProfile, BenchmarkConfig, Concurrency, run_benchmark};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinSet;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new(label: &str) -> Self {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "qwen3-tts-http-bench-{label}-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone)]
struct ResponseSpec {
    fragments: Vec<Vec<u8>>,
    delay: Duration,
}

async fn spawn_server(responses: Vec<ResponseSpec>) -> (String, Arc<Mutex<Vec<Instant>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let arrivals = Arc::new(Mutex::new(Vec::new()));
    let task_arrivals = Arc::clone(&arrivals);
    tokio::spawn(async move {
        let mut tasks = JoinSet::new();
        for response in responses {
            let (mut stream, _) = listener.accept().await.unwrap();
            let arrivals = Arc::clone(&task_arrivals);
            tasks.spawn(async move {
                read_request(&mut stream).await;
                arrivals.lock().unwrap().push(Instant::now());
                for fragment in response.fragments {
                    stream.write_all(&fragment).await.unwrap();
                    if !response.delay.is_zero() {
                        tokio::time::sleep(response.delay).await;
                    }
                }
                stream.shutdown().await.unwrap();
            });
        }
        while tasks.join_next().await.is_some() {}
    });
    (format!("http://{address}/v1/test"), arrivals)
}

async fn read_request(stream: &mut TcpStream) {
    let mut bytes = Vec::new();
    let header_end = loop {
        if let Some(position) = find_bytes(&bytes, b"\r\n\r\n") {
            break position + 4;
        }
        let mut buffer = [0_u8; 256];
        let count = stream.read(&mut buffer).await.unwrap();
        assert!(count > 0);
        bytes.extend_from_slice(&buffer[..count]);
    };
    let headers = std::str::from_utf8(&bytes[..header_end]).unwrap();
    let content_length = headers
        .split("\r\n")
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap())
        })
        .unwrap();
    while bytes.len() - header_end < content_length {
        let mut buffer = [0_u8; 256];
        let count = stream.read(&mut buffer).await.unwrap();
        assert!(count > 0);
        bytes.extend_from_slice(&buffer[..count]);
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn workload(path: &Path) {
    fs::write(
        path,
        "{\"id\":\"fixture-001\",\"text\":\"Sensitive fixture prompt\",\"voice_description\":\"Calm test voice\",\"language\":\"English\",\"seed\":42,\"sampling\":{\"strategy\":\"sample\",\"temperature\":0.8,\"top_p\":0.95,\"top_k\":50,\"repetition_penalty\":1.05,\"predictor\":{\"strategy\":\"sample\",\"temperature\":0.9,\"top_p\":1.0,\"top_k\":50}},\"stream\":true}\n",
    )
    .unwrap();
}

fn config(
    directory: &TestDirectory,
    endpoint: String,
    profile: BackendProfile,
    requests: usize,
    concurrency: Concurrency,
) -> BenchmarkConfig {
    let workload_path = directory.0.join("workload.jsonl");
    workload(&workload_path);
    BenchmarkConfig {
        endpoint,
        profile,
        sglang_model: (profile == BackendProfile::SglangOmni)
            .then(|| "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign".to_owned()),
        workload_path,
        output_dir: directory.0.join("result"),
        phase_events_path: None,
        requests,
        warmups: 0,
        concurrency,
        timeout_seconds: 5,
        log_prompt_text: false,
    }
}

fn json_part(boundary: &str, payload: &Value, close: bool) -> Vec<u8> {
    part(
        boundary,
        "application/json",
        &[],
        &serde_json::to_vec(payload).unwrap(),
        close,
    )
}

fn audio_part(boundary: &str, sequence: u64, first_sample: u64, final_flag: bool) -> Vec<u8> {
    part(
        boundary,
        "audio/pcm;rate=24000;channels=1;format=s16le",
        &[
            ("X-Sequence", sequence.to_string()),
            ("X-First-Codec-Frame", sequence.to_string()),
            ("X-First-Sample", first_sample.to_string()),
            ("X-Sample-Count", "2".to_owned()),
            ("X-Codec-Frames", "1".to_owned()),
            ("X-Final", final_flag.to_string()),
        ],
        &[1, 0, 2, 0],
        false,
    )
}

fn part(
    boundary: &str,
    content_type: &str,
    headers: &[(&str, String)],
    payload: &[u8],
    close: bool,
) -> Vec<u8> {
    let mut output = format!(
        "--{boundary}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n",
        payload.len()
    )
    .into_bytes();
    for (name, value) in headers {
        output.extend_from_slice(format!("{name}: {value}\r\n").as_bytes());
    }
    output.extend_from_slice(b"\r\n");
    output.extend_from_slice(payload);
    output.extend_from_slice(b"\r\n");
    if close {
        output.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    }
    output
}

fn native_multipart_body(boundary: &str, sequence: u64) -> Vec<u8> {
    let request_id = "0198f65d-a679-7411-8f7c-151dbf0486be";
    let mut body = json_part(
        boundary,
        &serde_json::json!({
            "type":"start",
            "request_id":request_id,
            "audio":{
                "encoding":"pcm_s16le",
                "sample_rate_hz":24000,
                "channels":1,
                "samples_per_codec_frame":1920
            }
        }),
        false,
    );
    body.extend_from_slice(&audio_part(boundary, sequence, 0, true));
    body.extend_from_slice(&json_part(
        boundary,
        &serde_json::json!({
            "type":"end",
            "request_id":request_id,
            "finish_reason":"stop",
            "metrics":{"emitted_packets":1,"emitted_samples":2}
        }),
        true,
    ));
    body
}

fn chunked_response(content_type: &str, body: &[u8], body_chunk_sizes: &[usize]) -> Vec<u8> {
    let mut response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nTransfer-Encoding: chunked\r\nX-Request-Id: 0198f65d-a679-7411-8f7c-151dbf0486be\r\n\r\n"
    )
    .into_bytes();
    let mut offset = 0;
    for &requested in body_chunk_sizes {
        if offset == body.len() {
            break;
        }
        let count = requested.min(body.len() - offset);
        response.extend_from_slice(format!("{count:x}\r\n").as_bytes());
        response.extend_from_slice(&body[offset..offset + count]);
        response.extend_from_slice(b"\r\n");
        offset += count;
    }
    if offset < body.len() {
        let count = body.len() - offset;
        response.extend_from_slice(format!("{count:x}\r\n").as_bytes());
        response.extend_from_slice(&body[offset..]);
        response.extend_from_slice(b"\r\n");
    }
    response.extend_from_slice(b"0\r\nX-Fixture: complete\r\n\r\n");
    response
}

fn fragmented(bytes: &[u8], cuts: &[usize]) -> Vec<Vec<u8>> {
    let mut output = Vec::new();
    let mut offset = 0;
    for &count in cuts {
        if offset == bytes.len() {
            break;
        }
        let end = (offset + count).min(bytes.len());
        output.push(bytes[offset..end].to_vec());
        offset = end;
    }
    if offset < bytes.len() {
        output.push(bytes[offset..].to_vec());
    }
    output
}

fn read_request_records(directory: &TestDirectory) -> Vec<Value> {
    fs::read_to_string(directory.0.join("result/requests.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[tokio::test]
async fn parses_fragmented_chunked_multipart_without_logging_prompts() {
    let directory = TestDirectory::new("multipart");
    let boundary = "fixture-boundary";
    let body = native_multipart_body(boundary, 0);
    let response = chunked_response(
        &format!("multipart/mixed; boundary=\"{boundary}\""),
        &body,
        &[1, 2, 3, 5, 8, 13, 21],
    );
    let (endpoint, _) = spawn_server(vec![ResponseSpec {
        fragments: fragmented(&response, &[1, 7, 2, 31, 3, 17, 5]),
        delay: Duration::from_millis(1),
    }])
    .await;
    run_benchmark(config(
        &directory,
        endpoint,
        BackendProfile::Native,
        1,
        Concurrency::B1,
    ))
    .await
    .unwrap();

    let requests = read_request_records(&directory);
    assert_eq!(requests[0]["success"], true);
    assert_eq!(requests[0]["samples"], 2);
    assert_eq!(requests[0]["sampling_parity_qualifying"], true);
    assert_eq!(
        requests[0]["normalized_sampling"]["contract"],
        "qwen3-tts-native-sglang-common/v1"
    );
    assert_eq!(
        requests[0]["audio_sha256"],
        hex::encode(Sha256::digest([1_u8, 0, 2, 0]))
    );
    assert!(requests[0]["ttfa_ms"].as_f64().unwrap() > 0.0);
    let raw = fs::read_to_string(directory.0.join("result/requests.jsonl")).unwrap();
    assert!(!raw.contains("Sensitive fixture prompt"));
    assert!(!raw.contains("Calm test voice"));
    assert_eq!(
        fs::read_to_string(directory.0.join("result/packets.jsonl"))
            .unwrap()
            .lines()
            .count(),
        1
    );
}

#[tokio::test]
async fn multipart_wall_time_includes_delayed_chunked_terminator() {
    let directory = TestDirectory::new("multipart-terminator");
    let boundary = "terminator-boundary";
    let body = native_multipart_body(boundary, 0);
    let response = chunked_response(
        &format!("multipart/mixed; boundary={boundary}"),
        &body,
        &[body.len()],
    );
    let terminal = find_bytes(&response, b"0\r\nX-Fixture: complete\r\n\r\n").unwrap();
    let (endpoint, _) = spawn_server(vec![ResponseSpec {
        fragments: vec![response[..terminal].to_vec(), response[terminal..].to_vec()],
        delay: Duration::from_millis(20),
    }])
    .await;
    run_benchmark(config(
        &directory,
        endpoint,
        BackendProfile::Native,
        1,
        Concurrency::B1,
    ))
    .await
    .unwrap();
    let requests = read_request_records(&directory);
    assert_eq!(requests[0]["success"], true);
    assert!(requests[0]["wall_ms"].as_f64().unwrap() >= 15.0);
}

#[tokio::test]
async fn malformed_declared_boundary_becomes_a_request_failure() {
    let directory = TestDirectory::new("boundary");
    let body = native_multipart_body("actual-boundary", 0);
    let response = chunked_response("multipart/mixed; boundary=declared-boundary", &body, &[9]);
    let (endpoint, _) = spawn_server(vec![ResponseSpec {
        fragments: vec![response],
        delay: Duration::ZERO,
    }])
    .await;
    run_benchmark(config(
        &directory,
        endpoint,
        BackendProfile::Native,
        1,
        Concurrency::B1,
    ))
    .await
    .unwrap();
    let requests = read_request_records(&directory);
    assert_eq!(requests[0]["success"], false);
    assert!(
        requests[0]["failure"]["message"]
            .as_str()
            .unwrap()
            .contains("boundary")
    );
}

#[tokio::test]
async fn native_sequence_gap_is_rejected() {
    let directory = TestDirectory::new("gap");
    let boundary = "gap-boundary";
    let body = native_multipart_body(boundary, 1);
    let response = chunked_response(
        &format!("multipart/mixed; boundary={boundary}"),
        &body,
        &[body.len()],
    );
    let (endpoint, _) = spawn_server(vec![ResponseSpec {
        fragments: vec![response],
        delay: Duration::ZERO,
    }])
    .await;
    run_benchmark(config(
        &directory,
        endpoint,
        BackendProfile::Native,
        1,
        Concurrency::B1,
    ))
    .await
    .unwrap();
    let requests = read_request_records(&directory);
    assert_eq!(requests[0]["success"], false);
    assert!(
        requests[0]["failure"]["message"]
            .as_str()
            .unwrap()
            .contains("continuity")
    );
}

fn wav_fixture() -> Vec<u8> {
    let pcm = [1_u8, 0, 2, 0];
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&40_u32.to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&24_000_u32.to_le_bytes());
    wav.extend_from_slice(&48_000_u32.to_le_bytes());
    wav.extend_from_slice(&2_u16.to_le_bytes());
    wav.extend_from_slice(&16_u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&4_u32.to_le_bytes());
    wav.extend_from_slice(&pcm);
    wav
}

#[tokio::test]
async fn validates_buffered_wav_and_timestamps_pcm_not_headers() {
    let directory = TestDirectory::new("wav");
    let workload_path = directory.0.join("workload.jsonl");
    fs::write(
        &workload_path,
        "{\"id\":\"wav-001\",\"text\":\"Hello\",\"voice_description\":\"Calm\",\"language\":\"English\",\"stream\":false}\n",
    )
    .unwrap();
    let wav = wav_fixture();
    let headers = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nContent-Length: {}\r\nX-Request-Id: 0198f65d-a679-7411-8f7c-151dbf0486be\r\nX-Finish-Reason: stop\r\n\r\n",
        wav.len()
    );
    let (endpoint, _) = spawn_server(vec![ResponseSpec {
        fragments: vec![headers.into_bytes(), wav[..44].to_vec(), wav[44..].to_vec()],
        delay: Duration::from_millis(15),
    }])
    .await;
    let mut configuration = config(
        &directory,
        endpoint,
        BackendProfile::Native,
        1,
        Concurrency::B1,
    );
    fs::write(
        &workload_path,
        "{\"id\":\"wav-001\",\"text\":\"Hello\",\"voice_description\":\"Calm\",\"language\":\"English\",\"stream\":false}\n",
    )
    .unwrap();
    configuration.workload_path = workload_path;
    run_benchmark(configuration).await.unwrap();
    let requests = read_request_records(&directory);
    assert_eq!(requests[0]["success"], true);
    assert_eq!(requests[0]["samples"], 2);
    assert!(requests[0]["ttfa_ms"].as_f64().unwrap() >= 20.0);
}

#[tokio::test]
async fn sglang_raw_pcm_stream_is_progressive_across_fragmented_http_chunks() {
    let directory = TestDirectory::new("raw-pcm");
    let pcm = [1_u8, 0, 2, 0, 3, 0, 4, 0];
    let response = chunked_response("audio/pcm; rate=24000", &pcm, &[3, 1, 4]);
    let separator = find_bytes(&response, b"\r\n\r\n").unwrap() + 4;
    let (endpoint, _) = spawn_server(vec![ResponseSpec {
        fragments: vec![
            response[..separator].to_vec(),
            response[separator..separator + 8].to_vec(),
            response[separator + 8..].to_vec(),
        ],
        delay: Duration::from_millis(10),
    }])
    .await;
    run_benchmark(config(
        &directory,
        endpoint,
        BackendProfile::SglangOmni,
        1,
        Concurrency::B1,
    ))
    .await
    .unwrap();
    let requests = read_request_records(&directory);
    assert_eq!(requests[0]["success"], true);
    assert_eq!(requests[0]["samples"], 4);
    assert_eq!(requests[0]["sample_rate_hz"], 24_000);
    assert_eq!(requests[0]["sampling_parity_qualifying"], true);
    assert_eq!(
        requests[0]["audio_sha256"],
        hex::encode(Sha256::digest(pcm))
    );
    assert!(requests[0]["packet_count"].as_u64().unwrap() >= 2);
    let packets: Vec<Value> = fs::read_to_string(directory.0.join("result/packets.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(packets.iter().all(|packet| packet["is_final"].is_null()));
    assert!(packets[1]["inter_arrival_ms"].as_f64().unwrap() > 0.0);
}

#[tokio::test]
async fn b3_requests_cross_the_start_barrier_together() {
    let directory = TestDirectory::new("b3");
    let pcm = [1_u8, 0];
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nX-Sample-Rate: 24000\r\nContent-Length: {}\r\n\r\n",
        pcm.len()
    )
    .into_bytes();
    let mut wire = response;
    wire.extend_from_slice(&pcm);
    let responses = (0..3)
        .map(|_| ResponseSpec {
            fragments: vec![wire.clone()],
            delay: Duration::ZERO,
        })
        .collect();
    let (endpoint, arrivals) = spawn_server(responses).await;
    run_benchmark(config(
        &directory,
        endpoint,
        BackendProfile::SglangOmni,
        3,
        Concurrency::B3,
    ))
    .await
    .unwrap();
    let times = arrivals.lock().unwrap();
    assert_eq!(times.len(), 3);
    let earliest = *times.iter().min().unwrap();
    let latest = *times.iter().max().unwrap();
    assert!(latest.duration_since(earliest) < Duration::from_millis(100));
    assert!(
        read_request_records(&directory)
            .iter()
            .all(|record| record["success"] == true)
    );
}

#[tokio::test]
async fn warmups_are_validated_but_excluded_from_reports() {
    let directory = TestDirectory::new("warmup");
    let pcm = [1_u8, 0];
    let mut wire = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\nX-Sample-Rate: 24000\r\nContent-Length: {}\r\n\r\n",
        pcm.len()
    )
    .into_bytes();
    wire.extend_from_slice(&pcm);
    let response = ResponseSpec {
        fragments: vec![wire],
        delay: Duration::ZERO,
    };
    let (endpoint, _) = spawn_server(vec![response.clone(), response]).await;
    let mut configuration = config(
        &directory,
        endpoint,
        BackendProfile::SglangOmni,
        1,
        Concurrency::B1,
    );
    configuration.warmups = 1;
    run_benchmark(configuration).await.unwrap();
    assert_eq!(read_request_records(&directory).len(), 1);
    let summary: Value =
        serde_json::from_str(&fs::read_to_string(directory.0.join("result/summary.json")).unwrap())
            .unwrap();
    assert_eq!(summary["warmups"], 1);
    assert_eq!(summary["completed_requests"], 1);
    assert_eq!(summary["sampling_parity_qualifying_requests"], 1);
    assert_eq!(summary["sampling_parity_non_qualifying_requests"], 0);
    assert_eq!(
        summary["normalized_sampling_sha256s"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn phase_events_align_exactly_with_measured_wall_time_without_warmups() {
    let directory = TestDirectory::new("phase-events");
    let pcm = [1_u8, 0];
    let mut wire = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\nX-Sample-Rate: 24000\r\nContent-Length: {}\r\n\r\n",
        pcm.len()
    )
    .into_bytes();
    wire.extend_from_slice(&pcm);
    let (endpoint, _) = spawn_server(vec![ResponseSpec {
        fragments: vec![wire],
        delay: Duration::ZERO,
    }])
    .await;
    let phase_path = directory.0.join("evidence/phase-events.jsonl");
    let mut configuration = config(
        &directory,
        endpoint,
        BackendProfile::SglangOmni,
        1,
        Concurrency::B1,
    );
    configuration.phase_events_path = Some(phase_path.clone());
    run_benchmark(configuration).await.unwrap();

    let events: Vec<Value> = fs::read_to_string(phase_path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(events.len(), 4);
    assert_eq!(
        events
            .iter()
            .map(|event| event["event"].as_str().unwrap())
            .collect::<Vec<_>>(),
        [
            "warmup_start",
            "warmup_end",
            "measured_start",
            "measured_end"
        ]
    );
    for (sequence, event) in events.iter().enumerate() {
        assert_eq!(
            event["schema_version"],
            "qwen3-tts-http-bench-phase-events/v1"
        );
        assert_eq!(event["sequence"].as_u64().unwrap(), sequence as u64);
        assert!(event["wall_time_unix_ns"].as_u64().unwrap() > 0);
    }
    let monotonic: Vec<u64> = events
        .iter()
        .map(|event| event["monotonic_elapsed_ns"].as_u64().unwrap())
        .collect();
    assert!(monotonic.windows(2).all(|pair| pair[0] <= pair[1]));

    let measured_nanoseconds = monotonic[3] - monotonic[2];
    let summary: Value =
        serde_json::from_str(&fs::read_to_string(directory.0.join("result/summary.json")).unwrap())
            .unwrap();
    let measured_seconds = Duration::from_nanos(measured_nanoseconds).as_secs_f64();
    assert!(
        (summary["benchmark_wall_seconds"].as_f64().unwrap() - measured_seconds).abs()
            < f64::EPSILON
    );
}

#[tokio::test]
async fn phase_events_refuse_to_overwrite_before_any_request() {
    let directory = TestDirectory::new("phase-overwrite");
    let phase_path = directory.0.join("phase-events.jsonl");
    fs::write(&phase_path, "keep me\n").unwrap();
    let (endpoint, arrivals) = spawn_server(vec![ResponseSpec {
        fragments: vec![b"unused".to_vec()],
        delay: Duration::ZERO,
    }])
    .await;
    let mut configuration = config(
        &directory,
        endpoint,
        BackendProfile::SglangOmni,
        1,
        Concurrency::B1,
    );
    configuration.phase_events_path = Some(phase_path.clone());

    let error = run_benchmark(configuration).await.unwrap_err();
    assert!(error.to_string().contains("I/O error"));
    assert_eq!(fs::read_to_string(phase_path).unwrap(), "keep me\n");
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(arrivals.lock().unwrap().is_empty());
}

#[tokio::test]
async fn phase_events_cannot_alias_a_canonical_report() {
    let directory = TestDirectory::new("phase-collision");
    let mut configuration = config(
        &directory,
        "http://127.0.0.1:1/v1/test".to_owned(),
        BackendProfile::SglangOmni,
        1,
        Concurrency::B1,
    );
    configuration.phase_events_path = Some(directory.0.join("result/intermediate/../summary.json"));

    let error = run_benchmark(configuration).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("must not replace canonical report file summary.json")
    );
    assert!(!directory.0.join("result/summary.json").exists());
}

#[tokio::test]
async fn phase_events_cannot_replace_the_output_directory() {
    let directory = TestDirectory::new("phase-output-directory-collision");
    let (endpoint, arrivals) = spawn_server(vec![ResponseSpec {
        fragments: vec![b"unused".to_vec()],
        delay: Duration::ZERO,
    }])
    .await;
    let mut configuration = config(
        &directory,
        endpoint,
        BackendProfile::SglangOmni,
        1,
        Concurrency::B1,
    );
    configuration.phase_events_path = Some(configuration.output_dir.clone());

    let error = run_benchmark(configuration).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("--phase-events must not replace --output-dir")
    );
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(arrivals.lock().unwrap().is_empty());
}

#[tokio::test]
async fn failed_warmup_retains_a_synced_phase_prefix() {
    let directory = TestDirectory::new("phase-warmup-failure");
    let response = b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n".to_vec();
    let (endpoint, _) = spawn_server(vec![ResponseSpec {
        fragments: vec![response],
        delay: Duration::ZERO,
    }])
    .await;
    let phase_path = directory.0.join("phase-events.jsonl");
    let mut configuration = config(
        &directory,
        endpoint,
        BackendProfile::SglangOmni,
        1,
        Concurrency::B1,
    );
    configuration.warmups = 1;
    configuration.phase_events_path = Some(phase_path.clone());

    let error = run_benchmark(configuration).await.unwrap_err();
    assert!(error.to_string().contains("warmup request"));
    let events: Vec<Value> = fs::read_to_string(phase_path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["sequence"], 0);
    assert_eq!(events[0]["event"], "warmup_start");
}
