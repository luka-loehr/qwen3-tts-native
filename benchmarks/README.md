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
