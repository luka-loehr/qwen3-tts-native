# Benchmark Protocol

Benchmark results are immutable JSON files produced by the native executable.

## Microbenchmarks

The first milestone covers:

- BF16 argmax at talker vocabulary 3,072;
- BF16 argmax at predictor vocabulary 2,048;
- exact talker projection shapes;
- exact predictor projection shapes;
- CUDA runtime memory and device identity.

Every projection benchmark performs 100 warmups, then 5,000 measured launches.
Zero-filled input and weights provide a deterministic zero-output correctness
check without changing the GPU execution shape.

The reported cold latency includes the first cuBLAS call and therefore exposes
lazy library initialization. It is a startup cost, not a per-request target.

## Native artifact and weight-loader protocol

Artifact evidence uses the exact audited VoiceDesign snapshot revision
`5ecdb67327fd37bb2e042aab12ff7391903235d3`. A valid final run must demonstrate:

- a flat regular-file artifact with no symlinks or special files;
- all 404 VoiceDesign tensors and exactly 271 `decoder.*` speech-tokenizer
  tensors;
- no encoder tensor and no copy of the complete source tokenizer checkpoint;
- offline BF16 round-to-nearest-even conversion, plus an independently tested
  F32 validation path;
- canonical per-tensor name, component, dtype, shape, parameters, contiguous
  arena offset, byte count, and SHA-256;
- byte-identical output from two independent BF16 pack runs;
- contract-only and full whole-file SHA-256 loader modes;
- mapped file, owned host-copy, runtime-conversion, pinned staging, and device
  allocation bytes reported separately;
- bounded device upload with exact readback after all source mappings are
  released;
- a clean NVIDIA Compute Sanitizer result.

The canonical summary is
`results/native-artifact-loader-summary.json`. Files with `indexed` in their
name are the final tensor-index implementation. Temporary F32 and repeat
artifacts are deleted after validation; their small JSON and `time -v` reports
remain as provenance.

Weight loading is not neural inference. Artifact pack, mmap-open, file-hash, and
host-to-device copy timings must never be described as TTFA, RTF, streaming, or
audio-quality results.

## Native model-session protocol

The talker and code-predictor layer is qualified independently from PCM decoding
and network transport. A qualifying model-session report requires:

- one persistent `NativeTalkerModel` for the complete run;
- at least 200 measured warm requests after pool warmup;
- warm time-to-first-codec-frame p50, p95, p99, and maximum;
- warm TTFA p95 below 200 ms;
- separate tokenization, prompt-plan, acquire, create, reset, and prefill timing;
- exact full-sequence parity for sampled B1, B3, and B6;
- exact greedy B3 parity;
- true concurrent host threads with independent CUDA streams and cuBLAS handles;
- duplicate-seed equality and different-seed isolation;
- cancellation, drop, round-robin, and cross-thread session-lifetime checks;
- codec EOS before the configured 256-frame corpus limit;
- shared model weights reported once and session memory reported per request;
- an otherwise idle GPU for every throughput and latency measurement.

The model load duration is reported but excluded from warm TTFA. A report captured
while another CUDA process is consuming the GPU is diagnostic only and must not
be labelled qualifying.

The reproducible command and API contract are documented in
[`native/qwen3-tts-native/README.md`](../native/qwen3-tts-native/README.md).
The latest uncontaminated evidence is
[`results/native-talker-session-qualification.json`](results/native-talker-session-qualification.json).

## End-to-end candidate protocol

A final candidate requires:

- at least 20 warm requests;
- at least 200 measured requests;
- p50, p95, and p99 time to first audio;
- p50, p95, and p99 real-time factor;
- p99 inter-packet gap;
- concurrency 1, 3, and 6;
- raw-socket progressive delivery evidence;
- maximum resident host memory and CUDA memory;
- joules per generated audio minute;
- seeded token parity and decoder continuity;
- German, English, French, Italian, and best-effort Turkish fixtures.

Turkish is not listed as an officially supported VoiceDesign language and must be
reported as an empirical best-effort result, never as guaranteed support.

## Controlled Native versus SGLang comparison

A comparison is qualifying only when it measures the native release candidate
and the pinned stock SGLang-Omni comparator with one external client, one corpus,
and one definition of every metric. Coexistence measurements taken while another
CUDA workload is active are diagnostic and must not appear in comparison charts.

### Fixed subjects

The comparison record must pin all of the following:

- DGX Spark hardware identity, firmware, driver, CUDA runtime, kernel, and power
  mode;
- native Git commit, OCI manifest digest, image configuration digest, and model
  artifact hashes;
- SGLang-Omni tag and commit, SGLang image manifest digest, Python package lock,
  and every compatibility patch hash;
- Qwen3-TTS VoiceDesign model ID and immutable model revision
  `5ecdb67327fd37bb2e042aab12ff7391903235d3` on both sides;
- BF16 weights, language, text, voice description, seed, sampling parameters,
  termination policy, and every portable output limit;
- 24 kHz, signed 16-bit, mono PCM at the client-visible measurement boundary.

The stock SGLang-Omni series must remain byte-provenanced to upstream release
`0.1.0`. A compatibility patch required only to package that release on ARM64 is
allowed when it is listed and hashed. A patch that changes scheduling, model
execution, codec cadence, or streaming behavior creates a separate
`SGLang-Omni patched` subject and must never replace the stock series silently.

### Isolation and order

Before a qualifying series:

1. stop unrelated inference workloads and verify that no competing CUDA compute
   process remains;
2. record available host/unified memory, swap use, temperature, clocks, and
   power mode;
3. run exactly one server subject at a time with the same host, model revision,
   client binary, and localhost network path;
4. complete 24 unmeasured warm requests for the active concurrency bucket;
5. measure at least 200 successfully completed requests at each of B1, B3, and
   B6; Native requests must report natural EOS and no length limit, while stock
   SGLang PCM completion is accepted with EOS classification explicitly unknown
   because that endpoint exposes no finish reason;
6. alternate subject order between rounds, with at least two rounds per subject,
   so warm host state and thermal drift do not always favor one engine;
7. stop the subject, preserve its logs and raw measurements, and verify request
   retirement before starting the other subject.

Cold model load, first-request warm-up, and steady-state request measurements are
separate series. A failed setup trial is preserved in the audit log but excluded
from distributions. Retry counts and exclusion reasons are reported explicitly.

### Shared metric definitions

The external Rust HTTP client owns the comparison clock. It records `t0`
immediately before writing the request and uses one monotonic clock for both
subjects.

- **TTFA** is the elapsed time from `t0` until the first non-empty decoded PCM
  payload byte reaches the client. HTTP headers, multipart JSON, SSE framing,
  WAV headers, and zero-length chunks do not count as audio.
- **Request wall time** ends only after the complete valid response and transport
  terminator have arrived.
- **Audio duration** is decoded PCM samples divided by 24,000 Hz.
- **Request RTF** is request wall time divided by generated audio duration.
- **Aggregate RTF** is scenario wall time divided by the sum of generated audio
  seconds; it is reported together with requests per second.
- **Streaming window** is final PCM arrival minus first PCM arrival. Packet or
  transport-chunk arrival timestamps are preserved so a one-burst response is
  distinguishable from progressive synthesis.
- **Reliability** includes accepted, completed, natural-EOS, length-limited,
  cancelled, failed, timed-out, and malformed responses without collapsing
  categories.

Every latency distribution reports minimum, mean, p50, p95, p99, and maximum.
Raw per-request values remain available; charts must not be reconstructed from
rounded table values.

### Resource and energy capture

Resource sampling begins before warm-up and continues until all measured
requests retire. Use a sampling period no slower than 200 ms and preserve the
raw timestamped series.

- process and container resident memory are reported separately from mapped
  model bytes;
- DGX Spark unified/GPU memory is sampled from NVIDIA telemetry and must not be
  added to host RSS as though it were independent physical memory;
- CPU and GPU utilization, temperature, clocks, and board power are recorded
  when the driver exposes them;
- joules are integrated from the timestamped power series after subtracting a
  separately measured idle baseline, and are normalized per generated audio
  minute;
- an unavailable sensor is reported as unavailable, never replaced by an
  estimate presented as a measurement.

Container image compressed transfer size, unpacked size, startup-to-readiness,
idle memory, peak memory, and server process count are reported outside the
request-latency distributions.

The canonical single-run implementation is documented in
[`tools/README.md`](tools/README.md). It uses the Rust client's four-event phase
file, an explicit server-idle baseline, 100 ms configured sampling, a 200 ms
maximum observed-gap gate, per-PID cgroup RSS sampling, measured-only reduction,
and manifest-root-relative evidence paths.

### Evidence layout and publication gate

Each final comparison bundle contains:

- a machine-readable manifest with all pins, commands, timestamps, and hashes;
- the exact workload JSONL and its SHA-256;
- raw per-request JSONL and raw packet-arrival JSONL for both subjects;
- timestamped resource and power samples;
- bounded timestamped container stdout/stderr logs, their exact capture-window
  metadata, client summary JSON, container inspection, and image provenance;
- the generated tables, monochrome chart sources, and final rendered PDF;
- SHA-256 values for every evidence file.

The PDF is a view over this bundle, not the source of truth. Publication is
blocked if the workload hashes differ, either subject has fewer than 200 valid
requests in any required bucket, any Native valid request is not natural EOS, a
competing CUDA process was present, the first-PCM boundary differs, raw files are
missing, or a stock comparator contains an undisclosed execution patch. The
stock SGLang series must retain `eos_unknown`; tooling must not infer natural EOS
from a closed PCM response.
