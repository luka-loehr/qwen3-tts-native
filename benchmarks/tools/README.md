# Benchmark tools

These development-only tools capture and reduce controlled DGX Spark evidence.
They are not copied into the production inference image.

## Qualifying single-run controller

`run-qualifying-benchmark.sh` is the publication-oriented entry point. It
requires the engine, concurrency profile, round, running container, image,
external client, workload, endpoint, output directory, and manifest-relative
evidence prefix explicitly. It resolves the image reference to its local
content-addressed image ID and rejects a container backed by any other image.

The controller copies and hashes the exact client and workload used, captures
sanitized container and image inspection data, host and NVIDIA identity, Docker
version, and the benchmark command's stdout and stderr. It stages the run next
to the requested output path. Only a successfully reduced and checksum-verified
run is renamed atomically to the final path. A failed run is retained under a
unique `.failed.<timestamp>.<pid>` path for audit instead of being deleted.

```bash
benchmarks/tools/run-qualifying-benchmark.sh \
  --output-dir /evidence/runs/round-01/native/B1 \
  --evidence-prefix runs/round-01/native/B1 \
  --engine native \
  --profile B1 \
  --round 1 \
  --container qwen3-tts-native \
  --image luka-loehr/qwen3-tts-native@sha256:<PUBLISHED_DIGEST> \
  --client /opt/bench/qwen3-tts-http-bench \
  --workload native/qwen3-tts-http-bench/examples/workload.jsonl \
  --endpoint http://127.0.0.1:8080/v1/voice-design/speech \
  --requests 200 \
  --warmups 24 \
  --idle-baseline-seconds 15
```

For SGLang, use `--engine sglang` and provide its immutable served model ID with
`--sglang-model`. The prefix must be a normalized relative POSIX path: absolute
paths, empty components, backslashes, `.` components, and `..` components are
rejected. Its value is prepended to every `telemetry_evidence_paths` entry so the
generated `run-resource.json` can be inserted into a manifest rooted above all
runs without hand editing.

## Spark telemetry collector

`capture-spark-telemetry.sh` starts timestamped collection, records an explicit
server-idle baseline, and then runs one command. The default configured interval
is 100 ms. A qualifying reduction rejects an observed gap above 200 ms.

The raw files include:

- NVIDIA P-state, temperature, utilization, board power, and graphics clock;
- container cgroup memory, memory peak, process count, and CPU time;
- every extant PID from the container's `cgroup.procs`, including name, process
  start ticks, and `VmRSS`, plus the per-sample sum;
- host available memory and swap;
- per-process NVIDIA unified-memory accounting and whether each compute PID
  belongs to the target cgroup;
- the wrapped command's stdout, stderr, invocation, timestamps, and exit status.

The collector waits until the GPU, system, process-RSS, and GPU-process samplers
have each emitted a row before marking the idle start. After the fixed baseline,
it waits for samples at or beyond the idle end before starting the client. Thus
both idle boundaries are bracketed. After the wrapped client exits, it likewise
waits until every sampler has emitted a row at or beyond the command-finish
timestamp, which is later than the client's measured-end event. It fails if a
sampler dies, the output directory already exists, the selected container is not
running, cgroup v2 is unavailable, or a required command is missing.

The process-RSS sampler tolerates a process exiting during `/proc` inspection
without weakening evidence. It makes at most three immediate attempts inside
the original scheduled sample cycle, freshly enumerating `cgroup.procs` and
assigning an actual wall/UTC timestamp to every attempt. Per-process rows remain
buffered until an attempt finishes, so an abandoned attempt never contaminates
`process-rss.csv`. A retry never starts at or after the cycle deadline, and the
next-cycle sleep remains anchored to the original cycle start. If no coherent
attempt succeeds, only the final partial attempt is emitted with
`sample_complete=0`; the qualifying reducer rejects that run.

The collector can be used directly for a diagnostic command:

```bash
benchmarks/tools/capture-spark-telemetry.sh \
  --output-dir /evidence/diagnostic-native-b1 \
  --container qwen3-tts-native \
  --idle-baseline-seconds 15 \
  -- \
  native/qwen3-tts-http-bench/target/release/qwen3-tts-http-bench \
  --endpoint http://127.0.0.1:8080/v1/voice-design/speech \
  --profile native \
  --workload native/qwen3-tts-http-bench/examples/workload.jsonl \
  --output-dir /evidence/diagnostic-native-b1-client \
  --phase-events /evidence/diagnostic-native-b1/phase-events.jsonl \
  --requests 200 \
  --warmups 24 \
  --concurrency B1
```

Do not put credentials or secrets in the wrapped command because the escaped
command line is intentionally retained as provenance.

## Fail-closed resource reducer

`reduce-spark-run.sh` accepts only the client's exact four-record phase schema:
`warmup_start`, `warmup_end`, `measured_start`, and `measured_end`, with sequence
0 through 3 and nondecreasing Unix and monotonic nanosecond timestamps. It uses
wall timestamps to align telemetry and monotonic timestamps to verify the
client's scenario duration. A wall/monotonic mismatch above 5 ms is rejected.

Power at the measured and idle boundaries is linearly interpolated from the
bracketing NVIDIA `power.draw` samples. Gross energy is trapezoidally integrated
over wall-clock time. `average_power_w` and `peak_power_w` are gross measured
board power; report `energy_j` is
`max(0, gross measured joules - idle mean watts * measured wall seconds)`.
`resource-audit.json` retains idle mean, peak, samples and gross joules, measured
gross joules, both wall and monotonic durations, and the adjustment formula.

`process_rss_peak_bytes` is the peak measured-window sum of `VmRSS` for every
extant target-cgroup PID at one sample time. Like conventional summed process
RSS, it can count shared resident pages in more than one process. It is not
cgroup memory. The latter remains separately recorded as `memory.current` in the
audit. `gpu_unified_memory_peak_bytes` is the peak sum of NVIDIA `used_memory`
for compute PIDs that belong to the target cgroup; it is not added to host RSS.

Reduction fails when phase boundaries are incomplete, telemetry does not bracket
a window, any relevant observed gap exceeds 200 ms, process RSS is incomplete,
power or GPU-memory sensors report unavailable values, no target GPU process is
present, a competing CUDA PID appears, the client has fewer than 24 warmups or
200 successes, Native termination is not natural EOS, or stock SGLang EOS is
anything other than unknown. No sensor value is estimated or imputed.

## Tests

The reducer fixtures require no Docker or GPU:

```bash
bash benchmarks/tools/tests/test-process-rss-sampler.sh
bash benchmarks/tools/tests/test-reduce-spark-run.sh
```

They cover coherent retry after a short-lived PID, bounded persistent failure,
cycle-deadline enforcement, the known idle-adjusted energy calculation, and
rejection of an incomplete process-RSS sample, a fifth phase event, a gap above
200 ms, and a competing CUDA process.
