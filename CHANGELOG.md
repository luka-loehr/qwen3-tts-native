# Changelog

All notable changes to Qwen3-TTS Native are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
for published releases.

## [Unreleased]

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

### Release status

- No registry image digest, semantic release tag, or `latest` alias has been
  published as an accepted release yet.
- The first release remains blocked on completing every gate in
  [`containers/RELEASE_CHECKLIST.md`](containers/RELEASE_CHECKLIST.md) for the
  exact pushed candidate digest.
- No controlled SGLang comparison has been completed or claimed.

[Unreleased]: https://github.com/luka-loehr/qwen3-tts-native/commits/main
