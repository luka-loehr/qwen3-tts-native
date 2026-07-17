# Qwen3-TTS HTTP Benchmark

`qwen3-tts-http-bench` is a standalone Rust client for reproducible external HTTP measurements of:

- the native Qwen3-TTS VoiceDesign API (`multipart/mixed` streaming PCM or buffered WAV), and
- SGLang-Omni VoiceDesign (`audio/pcm` or `application/octet-stream` raw PCM).

It uses a custom incremental HTTP/1.1 parser. No Python, Node.js, model runtime, or in-process server code participates in a measurement. The target must resolve exclusively to loopback addresses.

## Measurement semantics

All timings use the same monotonic `std::time::Instant` clock.

- `t0`: captured immediately before the first `write_all` that sends the HTTP request bytes. DNS resolution, TCP connection setup, workload parsing, and report serialization are excluded.
- Native streaming TTFA: arrival of the first non-empty PCM payload byte in an `audio/pcm` multipart part. HTTP headers and the JSON `start` part never count as audio.
- Native buffered-WAV TTFA: arrival of the first PCM byte in the RIFF `data` chunk. HTTP and WAV headers never count as audio.
- SGLang-Omni TTFA: arrival of the first non-empty raw PCM response-body byte after HTTP transfer decoding.
- Wall time: time from `t0` until the complete HTTP response body and protocol terminator have been consumed and validated.
- Audio duration: PCM sample count divided by the response sample rate. Native output is fixed at 24,000 Hz. SGLang output must declare `x-sample-rate` or a `rate` parameter on `Content-Type`; the client does not invent a sample rate.
- Per-request RTF: `wall_seconds / audio_seconds`. Values below 1.0 are faster than real time.
- Aggregate RTF: measured scenario wall time divided by the sum of successful audio durations. This is the end-to-end throughput RTF and remains meaningful for B3 and B6.
- Summed-request-wall RTF: sum of successful request wall times divided by the sum of their audio durations. This preserves the request-latency view but is not used as the parallel scenario throughput metric.
- Cadence: the interval between consecutive observed PCM arrivals. Native records application-level audio packets. Raw PCM records socket-delivery arrivals after HTTP transfer decoding and coalesces bytes obtained by the same socket read.
- `response_bytes`: all received HTTP bytes, including response headers and chunked-transfer framing.

For B1, B3, and B6, all TCP connections in a batch are established first. Tasks then wait at a shared barrier before independently taking `t0` and writing the request. The last partial batch uses the number of remaining requests.

## API profiles

### Native

Each workload item maps directly to the native endpoint:

| Workload field | Native request field |
| --- | --- |
| `text` | `text` |
| `voice_description` | `voice_description` |
| `language` | `language` |
| `seed` | `seed` |
| `max_duration_seconds` | `max_duration_seconds` |
| `sampling` | `sampling` |

`stream=true` adds `output_format="pcm_s16le"`; `stream=false` adds `output_format="wav"`.

Multipart validation requires ordered `start`, audio, and `end` parts; exact sequence, first-sample, and first-codec-frame continuity; correct PCM byte lengths; a final audio flag; matching request IDs; and end metrics that agree with observed packet and sample totals. `finish_reason="stop"` is a natural EOS and `finish_reason="length"` is reported as length-limited.

Buffered WAV validation checks the RIFF size, chunk boundaries and padding, PCM16 little-endian encoding, mono channel count, 24 kHz sample rate, byte rate, block alignment, data length, and `x-finish-reason`.

### SGLang-Omni 0.1.0 comparison profile

The comparison profile follows the official SGLang-Omni `/v1/audio/speech` VoiceDesign contract documented in [TTS Model Usage](https://sgl-project.github.io/sglang-omni/basic_usage/tts.html):

| Workload/config value | SGLang-Omni request field |
| --- | --- |
| `--sglang-model` | `model` |
| `text` | `input` |
| `voice_description` | `instructions` |
| `language` | `language` |
| fixed | `task_type="VoiceDesign"` |
| fixed | `voice="default"` |
| fixed | `stream=true` |
| fixed | `stream_format="audio"` |
| fixed | `response_format="pcm"` |
| `max_duration_seconds` | `max_new_tokens=ceil(seconds*12.5)` |
| `seed`, when present | `seed` |
| `sampling.strategy="sample"` | stock `do_sample=true` |
| `sampling.temperature` | `temperature` |
| `sampling.top_p` | `top_p` |
| `sampling.top_k` | `top_k` |
| `sampling.repetition_penalty` | `repetition_penalty` |
| predictor sample/0.9/1.0/50 | pinned stock defaults |

The pinned public speech schema does not expose `do_sample` or the four
predictor controls. Stock SGLang-Omni fixes them to sample/0.9/1.0/50, which is
the only predictor configuration this comparison profile accepts. The client
does not send undeclared fields that Pydantic would discard. Talker temperature,
top-p, top-k, repetition penalty, seed, and `max_new_tokens` are public request
fields and are sent explicitly. Greedy sampling and non-default predictor
settings fail before a request is sent.

Both implementations use 24 kHz audio with 1,920 samples per codec frame, or
exactly 12.5 frames per second. The canonical production workload sets
`max_duration_seconds=20.48`: Native maps that to 256 frames and the comparison
profile sends SGLang `max_new_tokens=256`. Production evidence is rejected if a
SGLang response reaches 255 frames (489,600 samples), keeping every accepted EOS
well clear of SGLang's length-before-EOS boundary. At EOS, stock SGLang also
runs one predictor step whose codes it discards, while Native stops before that
unnecessary step; this implementation difference is retained and documented.

SGLang raw streaming has no in-band start event, end event, usage object, finish reason, terminal sentinel, or final-packet flag. Consequently, successful transport completion is measurable, but natural EOS versus length termination is **unknown** and remains `null` in request records. The client never fabricates that distinction. Raw PCM is validated as non-empty PCM16 mono with an even total byte count and an explicitly declared sample rate.

## Workload format

The workload is deterministic UTF-8 JSONL. Blank lines are ignored; IDs must be unique.

```json
{"id":"english-calm-001","text":"Good morning.","voice_description":"A calm adult male voice with measured pacing.","language":"English","seed":42,"max_duration_seconds":20.48,"sampling":{"strategy":"sample","temperature":0.9,"top_p":1.0,"top_k":50,"repetition_penalty":1.05,"predictor":{"strategy":"sample","temperature":0.9,"top_p":1.0,"top_k":50}},"stream":true}
```

Fields:

| Field | Required | Description |
| --- | --- | --- |
| `id` | yes | Stable ASCII ID containing letters, digits, `.`, `_`, or `-` |
| `text` | yes | Text to synthesize |
| `voice_description` | yes | VoiceDesign instruction |
| `language` | no | Defaults to `auto` |
| `seed` | no | JSON-safe request seed |
| `max_duration_seconds` | no | Positive duration limit; required by the SGLang comparison profile and mapped at 12.5 codec frames/s |
| `sampling` | no | Strict common sampling object described below |
| `stream` | no | Defaults to `true`; SGLang comparison requires `true` |

When `--requests` exceeds the workload length, entries repeat in file order. Warmups use the same deterministic order and are validated but excluded from output.

### Sampling parity contract

Every request record contains an endpoint-neutral `normalized_sampling` object, its `normalized_sampling_sha256`, a `sampling_parity_qualifying` boolean, and explicit `sampling_parity_non_qualifying_reasons`. A Native/SGLang result is sampling-parity qualifying only when all effective common controls are explicit:

- `seed`;
- talker `strategy`, `temperature`, `top_p`, `top_k`, and `repetition_penalty`; and
- predictor/subtalker `strategy`, `temperature`, `top_p`, and `top_k`.

`strategy` defaults to `sample` inside an explicitly present stage object. With
`sample`, all listed controls must be present for qualification. Stock
SGLang-Omni 0.1.0 cannot select greedy mode through this HTTP API, and its
predictor controls are fixed to sample/0.9/1.0/50; any incompatible comparison
workload fails before I/O. Talker `repetition_penalty` remains portable.

Incomplete configurations may still be measured for diagnostics, but they are marked non-qualifying because server defaults can differ. Unknown fields fail during workload parsing. Comparison tooling should accept only records where `sampling_parity_qualifying=true` and should require matching `normalized_sampling_sha256` values across the two runs.

## Exact commands

Build the release binary:

```bash
cd native/qwen3-tts-http-bench
cargo build --release --locked
```

Run a 200-request native B1 benchmark:

```bash
./target/release/qwen3-tts-http-bench \
  --profile native \
  --endpoint http://127.0.0.1:8080/v1/voice-design/speech \
  --workload examples/workload.jsonl \
  --requests 200 \
  --warmups 24 \
  --concurrency B1 \
  --timeout-seconds 600 \
  --phase-events results/native-b1.phase-events.jsonl \
  --output-dir results/native-b1
```

Run a synchronized native B6 benchmark:

```bash
./target/release/qwen3-tts-http-bench \
  --profile native \
  --endpoint http://127.0.0.1:8080/v1/voice-design/speech \
  --workload examples/workload.jsonl \
  --requests 204 \
  --warmups 24 \
  --concurrency B6 \
  --phase-events results/native-b6.phase-events.jsonl \
  --output-dir results/native-b6
```

Run the same workload against SGLang-Omni:

```bash
./target/release/qwen3-tts-http-bench \
  --profile sglang-omni \
  --sglang-model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --endpoint http://127.0.0.1:8000/v1/audio/speech \
  --workload examples/workload.jsonl \
  --requests 200 \
  --warmups 24 \
  --concurrency B1 \
  --timeout-seconds 600 \
  --phase-events results/sglang-omni-b1.phase-events.jsonl \
  --output-dir results/sglang-omni-b1
```

Use a fresh output directory for every run. The client refuses to overwrite any existing report file.

## External telemetry alignment

`--phase-events PATH` opts into a separate, create-new JSONL evidence file for aligning host, container, process, GPU, power, and energy telemetry. The client refuses to overwrite the path and rejects aliases of the three canonical report files before sending any request. A successful run contains exactly these four records in order, including when the warmup count is zero:

1. `warmup_start`
2. `warmup_end`
3. `measured_start`
4. `measured_end`

Every record uses schema `qwen3-tts-http-bench-phase-events/v1` and contains a zero-based `sequence`, integer `wall_time_unix_ns`, and integer `monotonic_elapsed_ns`. Wall time is sampled from `SystemTime` for cross-process telemetry alignment. Monotonic elapsed time is sampled from `Instant` relative to an origin established immediately after the file is reserved; it is the authoritative clock for within-run intervals. The `benchmark_wall_seconds` value in `summary.json` uses the same `Instant` values captured for `measured_start` and `measured_end`.

The four records are held in memory and written, flushed, and synchronized only after `measured_end`, so phase-evidence I/O cannot enter the measured wall time. A failed warmup synchronizes the valid prefix containing `warmup_start`; an abnormal process termination can leave the already reserved file empty. Treat anything other than exactly four ordered records as an incomplete, non-qualifying run.

## Outputs

Each run writes exactly three canonical files under `--output-dir`:

- `requests.jsonl`: one raw measurement record per request, sorted by request index, including normalized sampling evidence and one `audio_sha256` over the complete decoded PCM payload.
- `packets.jsonl`: one Native application packet or SGLang raw-PCM transport arrival per line, sorted by request and sequence.
- `summary.json`: counts, successful and attempted throughput, scenario aggregate RTF, summed-request-wall RTF, and min/mean/p50/p90/p95/p99/max distributions for TTFA, wall time, and per-request RTF.

By default, reports contain workload IDs plus SHA-256 hashes of text, voice description, final request JSON, complete decoded PCM, individual audio packets, and the normalized sampling contract. They do not contain prompt text or PCM bytes. `--log-prompt-text` is an explicit opt-in for controlled local debugging.

HTTP error response bodies are never logged. Only their byte count and SHA-256 are retained.

The optional `--phase-events` file is auxiliary raw evidence, not a fourth canonical report, and does not alter any canonical schema.

## Fair comparisons

For a defensible Native-versus-SGLang comparison:

1. Use the same workload file and request count.
2. Use the same model checkpoint, language hints, warmup count, and concurrency tier.
3. Require `sampling_parity_qualifying=true` and identical `normalized_sampling_sha256` values in both runs.
4. Run on the same idle hardware with equivalent power and clock settings.
5. Record server/container digests and machine telemetry separately.
6. Compare B1 with B1, B3 with B3, and B6 with B6.
7. Report the SGLang natural-EOS field as unknown unless an external server-specific signal proves it.

This client measures transport and synthesis timing. It does not claim perceptual audio equivalence and does not calculate MOS, intelligibility, or speaker similarity.

## Verification

```bash
cargo fmt --all -- --check
cargo test --all-targets --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
```

The integration tests use local TCP fixture servers and cover fragmented HTTP headers, chunked transfer encoding, multipart boundaries split across reads, malformed boundaries, sequence gaps, buffered WAV delivery, fragmented raw PCM, prompt redaction, and synchronized B3 starts.
