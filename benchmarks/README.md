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
