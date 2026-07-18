# Changelog

All notable changes to Qwen3-TTS Native are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
for published releases.

## [Unreleased]

No changes yet.

## [0.4.0] - 2026-07-18

### Added

- Fast research vocoder mode (`QWEN3_TTS_CODEC_FAST=1`): numerically exact
  coalesced weight repack for the transposed-conv upsampler (cached once per
  process), block-parallel norms, per-(session, frame-count) CUDA-graph
  capture of the packet decode, and opt-in TF32 GEMMs
  (`QWEN3_TTS_CODEC_TF32=1`). The default codec path is untouched and its
  parity gates are unchanged.
- Flash-style decode attention and INT8 epilogue fusion (residual adds and a
  fused gate/up/SiLU kernel) in the INT8 research mode.
- Exact-parity parallel sampling kernel: bit-identical selected tokens and
  RNG state versus the previous kernel across 1.2 million randomized cases,
  20-30x faster on sampled configurations.

### Evidence-backed negative results

- RTN INT4 talker weights: ~12 percent output error on real matrices versus
  INT8's 0.9 percent, at most 1.6x at M=1 and slower than INT8 at M=6.
  Rejected.
- Codec INT8: exceeds the 1e-3 waveform-deviation budget by 10-3600x in
  every subset; the continuous vocoder output has no token-quantization
  cushion. Rejected.

### Measured on an idle DGX Spark (all 24 workload languages, natural EOS)

- INT8 + fast codec: single-stream RTF 0.287 with TTFA p50 55 ms; aggregate
  RTF 0.139 (B3) and 0.107 (B6); per-request wall RTF 0.37 (B3) and 0.54
  (B6).

## [0.3.0] - 2026-07-18

### Added

- Cross-request lockstep batched decoding (talker ABI v3). When several
  requests generate concurrently, one batched native call now performs the
  talker and 15-step code-predictor decode for every active session, so each
  shared weight matrix is read from memory once per codec frame instead of
  once per session. KV caches, sampling state, semantic histories, random
  states, and lifecycle leases remain request-local, and each session is
  still finished and delivered independently. Single-request behavior and the
  session-level ABI are unchanged; libraries predating ABI v3 fall back to
  the per-session decode path.
- CUDA-graph capture of the lockstep frame: the complete batched
  predictor-and-talker decode is captured once per session tuple and replayed
  thereafter, removing per-frame kernel-launch overhead. Session lifecycle
  events are external graph nodes, so codec staging and finish leases are
  unchanged.
- A batched grouped-query-attention kernel that reads per-session spans from
  device memory, and device-driven sampling indices, making the whole frame
  replay-safe.
- Opt-in INT8 weight-only decode (`QWEN3_TTS_INT8_DECODE=1`): every decode
  GEMM weight is quantized to per-output-channel symmetric INT8 at engine
  load, directly from the pinned BF16 artifact. Prefill stays BF16. The
  default engine remains BF16.
- First-frame priority: a session's first frame bypasses the lockstep
  rendezvous so first-audio latency does not pay a batching penalty.

### Measured on an idle DGX Spark (all 24 workload languages, natural EOS)

- BF16: single-stream RTF 0.72; aggregate RTF 0.41 (B3) and 0.36 (B6);
  per-request wall RTF 1.10 (B3) and 1.91 (B6) versus 1.65 and 3.22 for the
  unbatched v0.2.0 engine.
- INT8: single-stream RTF 0.47 with TTFA p50 60 ms; aggregate RTF 0.33 (B3)
  and 0.31 (B6); per-request wall RTF 0.90 (B3) and 1.69 (B6).

## [0.2.0] - 2026-07-18

### Changed

- Streaming PCM packets are converted to little-endian bytes with a single
  pre-sized buffer instead of per-sample appends.
- Public documentation no longer references unrelated private infrastructure
  by name, and release dates in `CITATION.cff` and this changelog match the
  actual `v0.1.0` tag date.
- Research paper: redesigned architecture and streaming-cadence figures,
  corrected performance-plot legends, expanded bibliography, and typography
  polish. Benchmark data and evidence bindings are unchanged and remain from
  the `v0.1.0` controlled comparison.

### Unchanged by design

- The talker decode GEMMs keep `CUBLAS_COMPUTE_32F`.
  `CUBLAS_COMPUTE_32F_FAST_16BF` was built and benchmarked on the target GB10
  and consistently regressed warm B1 aggregate RTF and TTFA by roughly
  20 percent, so it was reverted before release. The decode GEMM comment in
  `talker.cu` records this negative result.
- The speech-codec decoder keeps full FP32 cuBLAS GEMMs. TF32 was evaluated
  and rejected because it cannot satisfy the decoder parity thresholds
  (down to 5e-6 maximum absolute error) that gate this repository.

## [0.1.0] - 2026-07-18

### Added

- Native Rust and CUDA inference for the pinned
  `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` checkpoint.
- A strict artifact contract for the VoiceDesign talker, 15-step code
  predictor, and decoder-only speech-tokenizer weights, including pinned
  revisions, tensor inventories, and whole-file hashes.
- Custom `sm_121` CUDA/cuBLAS execution for the talker, code predictor, and
  stateful incremental neural speech decoder.
- Direct device-to-device handoff of ordered 16-codebook frames between the
  talker and decoder.
- Progressive 24 kHz mono signed-16-bit PCM with exact packet, codec-frame,
  sample-position, and caller-buffer accounting.
- Shared immutable models with independently owned request sessions, bounded
  pooling, adjacent-request prefill coalescing, backpressure, cancellation,
  deterministic retirement, and per-request memory metrics.
- A versioned public C ABI with typed status values, panic containment,
  caller-owned error buffers, explicit finish reasons, and lifecycle tests.
- A native Rust HTTP service with liveness, readiness, capabilities, metrics,
  progressive multipart PCM, buffered WAV, request IDs, cancellation, bounded
  limits, and graceful SIGINT/SIGTERM shutdown.
- Full-pipeline startup warm-up before listener binding and readiness.
- Official support for `Auto` and the ten language IDs exposed by the pinned
  model: Chinese, English, Japanese, Korean, German, French, Russian,
  Portuguese, Spanish, and Italian.
- Native correctness, parity, sanitizer, lifecycle, concurrency, natural-EOS,
  multilingual, HTTP, memory, energy, and performance evidence in
  [`benchmarks/results/`](benchmarks/results/).
- A qualifying 200-request single-stream natural-EOS endurance record and a
  24-entry official-language qualification record.
- A hardened production image definition for `linux/arm64` DGX Spark with
  pinned Rust, CUDA, cuBLAS, model, and BuildKit inputs; embedded pinned model
  weights; non-root execution; read-only-root support; and a minimal runtime
  inventory.
- Reproducible Apache-2.0 application and Rust dependency license reports,
  CycloneDX metadata, BuildKit SBOM/provenance support, model attribution, and
  third-party notices.
- A digest-specific image release checklist covering supply-chain, security,
  size, clean-pull, GPU behavior, performance, language, and promotion gates.
- Public project documentation, contribution standards, security reporting,
  and community conduct policy.

### Changed

- Session capacity is sized from the prompt and requested codec-frame budget
  and reused through bounded capacity classes instead of reserving the maximum
  sequence length for every request.
- Server startup now validates a real talker, predictor, device handoff, codec,
  final packet, finish reason, metrics, and request retirement before accepting
  traffic.
- Runtime and release documentation now distinguish direct native C-ABI,
  direct HTTP, final-container, fixed-length, natural-EOS, energy, and
  subjective audio-quality evidence.
- The repository is licensed under Apache License 2.0 with explicit model and
  third-party attribution boundaries.

### Release evidence

- The immutable container digest, semantic tag, `latest` alias, SBOM,
  provenance, signature, scan receipts, clean-pull proof, and GPU acceptance
  are recorded in the
  [`v0.1.0` GitHub release](https://github.com/luka-loehr/qwen3-tts-native/releases/tag/v0.1.0).
- The controlled two-engine, B1/B3/B6, two-round comparison is published in
  the [final benchmark report](reports/output/qwen3-tts-native-vs-sglang-stock-dgx-spark-2026-07-17-428307c-report.pdf).
- The complete English system description and evaluation are published as the
  [Qwen3-TTS Native research paper](research/paper/qwen3-tts-native-paper.pdf).
- The checked-in [release checklist](containers/RELEASE_CHECKLIST.md) remains
  the reusable fail-closed template; completed digest-specific receipts are
  release assets so the source revision embedded in the image stays immutable.

[Unreleased]: https://github.com/luka-loehr/qwen3-tts-native/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/luka-loehr/qwen3-tts-native/releases/tag/v0.1.0
