# Qwen3-TTS Native

Native Rust and CUDA research runtime for
`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`, developed and measured on NVIDIA DGX
Spark.

The repository contains a real incremental text-to-code-to-PCM path. It loads
the official 1.7B VoiceDesign and speech-tokenizer decoder weights, generates
codec frames on the GPU, decodes them into 24 kHz mono PCM, and exposes the
stream through a versioned C ABI. Python and Node.js are not part of the
inference runtime.

> **Status:** working research implementation, not a production service. The
> public C ABI and native audio path are functional. Single-stream generation
> is faster than real time, while per-request B3/B6 latency and multilingual
> audio-quality qualification still require work.

## What is implemented

- Native Qwen3-TTS 1.7B VoiceDesign talker and 15-step code predictor.
- Native speech-tokenizer decoder with exact incremental PCM continuity.
- Rust orchestration around CUDA 13, cuBLAS, and narrow C ABIs.
- Shared immutable model weights with independently owned request sessions.
- Bounded session pooling and adjacent-request prefill coalescing.
- Public engine/request lifecycle, streaming polling, cancellation, metrics,
  typed failures, and panic containment.
- Exact-capacity PCM preflight without consuming or losing a packet.
- Contract, parity, concurrency, lifecycle, sanitizer, C-ABI smoke, memory,
  energy, and performance evidence.

The runtime emits signed 16-bit mono PCM at 24 kHz. One codec frame represents
1,920 samples, or 80 ms of audio. The first packet contains one frame; later
packets contain up to four frames.

## Verified DGX Spark result

The latest public C-ABI qualification used the official model revision
`5ecdb67327fd37bb2e042aab12ff7391903235d3`, 24 warmups, and 200 measured
requests at each concurrency level.

| Concurrency | Completed | TTFA p95 | Request RTF p50 | Aggregate RTF |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 200/200 | 78.24 ms | 0.765 | 0.767 |
| 3 | 200/200 | 186.42 ms | 1.800 | 0.601 |
| 6 | 200/200 | 364.62 ms | 3.557 | 0.594 |

All 600 measured requests completed with contiguous packet positions, exact
sample counts, untouched PCM tails, final-then-end-of-stream behavior, and
delivery metrics matching the packets observed by the C caller.

Interpret the concurrency numbers carefully: aggregate throughput remains
faster than real time at B3 and B6, but an individual request sharing the GPU
does not. The stricter single-stream RTF target below 0.50 is also not met yet.
See
[`benchmarks/results/native-runtime-public-c-abi-qualification.json`](benchmarks/results/native-runtime-public-c-abi-qualification.json)
for the complete measured record.

### Memory

- Shared VoiceDesign weights: 3,833,352,704 device bytes.
- Shared decoder weights: 457,292,548 device bytes.
- Peak additional device allocation per request: 47,042,708 bytes.
- Peak pinned host allocation per request: 46,080 bytes.
- Computed B6 device total: 4,572,901,500 bytes.
- Peak benchmark process RSS: 4,034,776 KiB.

### Functional audio evidence

The public C-ABI smoke generated valid 24 kHz s16le mono WAV output and proved
engine/request ownership, cancellation, invalid-input handling, packet
continuity, and metrics. Its 40-frame safety cap stopped the sample after 3.2
seconds and cut the final word short. It is therefore transport evidence, not a
completed audio-quality gate. The final quality corpus must run to the model's
natural end-of-sequence with only a generous emergency limit.

## Repository layout

| Path | Contents |
| --- | --- |
| `native/qwen3-tts-native` | Model contracts, tokenizer, artifact loader, VoiceDesign talker, code predictor, CUDA kernels, and session benchmark. |
| `native/qwen3-tts-native-codec` | Incremental neural speech-tokenizer decoder, shared model/session APIs, and CUDA implementation. |
| `native/qwen3-tts-runtime` | Scheduler, native backend, public C ABI, C header, smoke harness, and concurrency benchmark. |
| `native/qwen3-tts-bench` | Native benchmark utilities and WAV/report helpers. |
| `benchmarks/fixtures` | Small deterministic decoder parity fixtures; no model weights. |
| `benchmarks/results` | Checked measurement and qualification evidence. |
| `benchmarks/corpora` | Official-language and exploratory multilingual prompt corpora. |
| `notes` | Architecture, model-contract, artifact, codec, and toolchain decisions. |
| `containers` | Reproducible development-builder definition and container roadmap. |
| `tools` | Offline reference-fixture tooling; not used by the production runtime. |

## Model files

Model weights are deliberately not stored in Git or baked into an image. Use
the official
[`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
checkpoint and mount the prepared artifact read-only at runtime.

The qualifying artifact hashes are recorded in the benchmark JSON. This keeps
the source repository small, makes model provenance explicit, and avoids
silently redistributing multi-gigabyte third-party material.

## Toolchain

The tested target is Ubuntu 24.04 AArch64, NVIDIA GB10, CUDA 13.0.88, and real
SM 12.1 SASS. Rust 1.97.0 is pinned for reproducibility.

Build the local Rust tooling image:

```bash
docker build \
  --file containers/Dockerfile.builder \
  --tag qwen3-tts-native/builder:rust-1.97.0 \
  .
```

CUDA libraries are currently built with the pinned upstream
`nvcr.io/nvidia/tensorrt:25.11-py3` image. Exact component commands and
verification procedures live in the component READMEs:

- [`native/qwen3-tts-native/README.md`](native/qwen3-tts-native/README.md)
- [`native/qwen3-tts-native-codec/README.md`](native/qwen3-tts-native-codec/README.md)
- [`native/qwen3-tts-runtime/README.md`](native/qwen3-tts-runtime/README.md)

There is not yet a production runtime image. The Spark currently has only the
research builder image `codex/qwen3-tts-rust-builder:1.97.0`. A runtime image
will be added after the streaming ABI and performance work stabilizes; it will
contain only the three native libraries, public header, and a service/runner,
with weights mounted read-only rather than copied into the image.

## Public ABI

The stable entry surface is declared in
[`native/qwen3-tts-runtime/include/qwen3_tts_runtime.h`](native/qwen3-tts-runtime/include/qwen3_tts_runtime.h).
It covers:

- ABI version discovery;
- engine creation and destruction;
- request creation, cancellation, polling, metrics, and destruction;
- explicit status values and caller-owned error buffers.

The implementation is a library, not an HTTP or gRPC server. Network service
design remains deliberately outside this research milestone.

## Current boundaries

- No Ephraim backend, frontend, or production container is changed by this
  repository.
- The 0.6B model is out of scope.
- No permanent TTS daemon is currently installed on the Spark.
- B3/B6 per-request RTF below 1.0 and B1 RTF below 0.50 are not achieved yet.
- Multilingual intelligibility, speaker consistency, instruction adherence,
  and natural end-of-sequence still need final listening/ASR qualification.
- The checked sample WAV is intentionally excluded from Git; benchmark reports
  contain its format, byte count, and SHA-256 evidence.

Research artifacts should be promoted into another system only after latency,
memory, cancellation, concurrency, energy, and audio-quality gates pass.

## License

The application source in this repository is licensed under the Apache License
2.0. See [`LICENSE`](LICENSE). Embedded or redistributed third-party material,
including the pinned Qwen3-TTS model and NVIDIA CUDA runtime, remains subject to
its respective license and attribution; see [`licenses/`](licenses/).
