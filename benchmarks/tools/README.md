# Benchmark tools

These tools capture raw evidence for controlled DGX Spark measurements. They are
development-only and are not copied into the production image.

## Spark telemetry wrapper

`capture-spark-telemetry.sh` runs one command and records timestamped raw data at
200 ms intervals:

- NVIDIA P-state, temperature, utilization, board power, and graphics clock;
- container cgroup memory, memory peak, process count, and CPU time;
- host available memory and swap;
- per-process NVIDIA unified-memory accounting.

The wrapper fails if the output directory already exists, the selected container
is not running, cgroup v2 metrics are unavailable, or a required command is
missing. The wrapped command's stdout, stderr, shell-escaped invocation, start
time, finish time, and exit status are preserved.

```bash
benchmarks/tools/capture-spark-telemetry.sh \
  --output-dir benchmarks/evidence/native-b1-round-1 \
  --container qwen3-tts-native \
  -- \
  native/qwen3-tts-http-bench/target/release/qwen3-tts-http-bench \
  run \
  --config benchmarks/runs/native-b1.json
```

Do not place credentials or other secrets in the wrapped command because the
escaped command line is intentionally retained as benchmark provenance.

Board energy must be integrated from the raw `power_w` series. Subtract a
separately captured idle baseline, preserve the integration method, and report a
sensor as unavailable when the NVIDIA driver emits `N/A`.
