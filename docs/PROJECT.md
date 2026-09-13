# Project overview

## Scope

The project supports **VoiceDesign only**. It does not include voice cloning,
reference audio, speaker enrollment, the Base or CustomVoice checkpoints, the
speech-tokenizer encoder, or the retired 0.6B model.

## Supported languages and audio

The pinned model exposes ten explicit languages: `Chinese`, `English`,
`Japanese`, `Korean`, `German`, `French`, `Russian`, `Portuguese`, `Spanish`,
and `Italian`. `Auto` is also available. Values are case-insensitive at the
HTTP boundary; other languages (including Turkish) are rejected.

Audio is emitted as 24,000 Hz, mono, signed 16-bit little-endian PCM. Each
codec frame represents 1,920 samples, or 80 ms. The first streaming packet
contains one frame; subsequent packets contain up to four frames.

## Pipeline

```text
HTTP client
    |
    v
native Rust HTTP server (validation, limits, streaming, cancellation)
    |
    v
Rust scheduler and versioned C ABI
    |
    +--> VoiceDesign talker + 15-step code predictor (CUDA/cuBLAS)
    |          |
    |          `-- ordered device-to-device codec frames
    |
    `--> incremental neural speech decoder (CUDA/cuBLAS)
               |
               `-- progressive 24 kHz PCM
```

The process constructs one shared engine and performs a one-frame warm-up
through the complete native pipeline before binding the listener, so readiness
is a model-and-pipeline gate. The runtime image contains the pinned model
artifacts and excludes compilers, development packages, Python, Node.js,
PyTorch, SGLang, vLLM, TensorRT, cuDNN, NPP, cuSPARSE, and NCCL. See
[Architecture](ARCHITECTURE.md) and the
[container documentation](../containers/README.md).

## HTTP endpoints

| Method and path | Purpose |
| --- | --- |
| `GET /health/live` | Process and event-loop liveness. |
| `GET /health/ready` | Shared engine loaded, full native path warmed, and engine healthy. |
| `GET /v1/capabilities` | VoiceDesign-only languages, formats, and limits. |
| `POST /v1/voice-design/speech` | Progressive multipart PCM or buffered WAV synthesis. |
| `POST /v1/audio/speech` | Buffered-WAV compatibility endpoint (not a general OpenAI Audio API). |
| `DELETE /v1/requests/{request-id}` | Bounded cancellation of an admitted request. |
| `GET /metrics` | Prompt-free Prometheus request counters and engine-health gauge. |

Core fields of `POST /v1/voice-design/speech`:

| Field | Required | Default | Meaning |
| --- | ---: | --- | --- |
| `text` | yes | — | Non-empty UTF-8 text to synthesize. |
| `voice_description` | yes | — | Natural-language voice description; never a voice name, audio sample, or clone reference. |
| `language` | no | `auto` | `auto` or one of the ten explicit languages. |
| `stream` | no | inferred | `true` selects progressive multipart PCM; `false` selects buffered WAV. |
| `output_format` | no | inferred | `pcm_s16le` for streaming or `wav` for buffered output; must agree with `stream`. |
| `seed` | no | random | Optional reproducibility seed returned by the service. |
| `max_duration_seconds` | no | lower of 120 s and instance maximum | Safety ceiling; generation may stop earlier at natural codec EOS. |

Omitting both `stream` and `output_format` selects progressive multipart PCM.
The streaming response is `multipart/mixed`: a JSON start event, one or more
binary PCM parts, and exactly one JSON end or error event. Clients must parse
multipart boundaries. The full contract is in [API](API.md) and
[openapi.yaml](openapi.yaml).

## Deployment boundary and data handling

The standalone binary binds to `127.0.0.1:8080` by default; the image listens
on `0.0.0.0:8080` inside its container. The service does not terminate TLS or
authenticate clients, so public deployments belong behind an authenticated,
rate-limited proxy with timeouts.

Prompts, voice descriptions, generated audio, request IDs, and language values
are not written to normal logs or Prometheus labels. Transport encryption,
authentication, retention, and payload logging in the surrounding stack are the
operator's responsibility. See [Operations](OPERATIONS.md) and
[SECURITY.md](../SECURITY.md).

## Build and verification

Rust 1.97.0, CUDA 13.0.3, cuBLAS 13.1.1.3, Ubuntu 24.04 ARM64, and real
`sm_121` SASS are pinned for the production image. Reproducing the image also
requires the audited model artifact and generated release-metadata contexts.

The Dockerfile validates pinned artifact hashes, CUDA architecture, dynamic
dependencies, package inventory, non-root ownership, licenses, and SBOM inputs
during the build. Registry attestations, vulnerability scans, signature
verification, and GPU qualification are post-build release gates.

- [Production image and build command](../containers/README.md)
- [Release checklist](../containers/RELEASE_CHECKLIST.md)
- [HTTP server contract](../native/qwen3-tts-server/README.md)
- [Native runtime and C ABI](../native/qwen3-tts-runtime/README.md)
- [VoiceDesign talker and predictor](../native/qwen3-tts-native/README.md)
- [Incremental neural codec](../native/qwen3-tts-native-codec/README.md)
- [Release metadata generation](../tools/release-metadata/README.md)

## Repository layout

| Path | Contents |
| --- | --- |
| `native/qwen3-tts-native` | Artifact contract, tokenizer, VoiceDesign talker, code predictor, and CUDA kernels. |
| `native/qwen3-tts-native-codec` | Stateful incremental speech-tokenizer decoder. |
| `native/qwen3-tts-runtime` | Scheduler, native backend, versioned C ABI, and lifecycle tests. |
| `native/qwen3-tts-server` | Bounded Rust HTTP transport, healthcheck, metrics, streaming, and WAV output. |
| `native/qwen3-tts-bench` | Real-runtime qualification harness and report helpers. |
| `native/qwen3-tts-http-bench` | Standalone HTTP client for synchronized native and SGLang measurements. |
| `benchmarks` | Corpora, deterministic fixtures, protocols, and result records. |
| `reports` | Evidence validation and deterministic PDF report generation. |
| `research/paper` | Research paper PDF and LaTeX source. |
| `docs` | Quickstart, API, configuration, operations, architecture, and OpenAPI documentation. |
| `containers` | Reproducible builder, hardened runtime image, and release checklist. |
| `licenses` | Model provenance and third-party notices. |
| `tools` | Release metadata and image supply-chain tooling. |
| `notes` | Architecture, model-contract, artifact, codec, and toolchain decisions. |

## Research and citation

The [research paper](../research/paper/qwen3-tts-native-paper.pdf) describes
the system design, native implementation, streaming contract, controlled
evaluation, limitations, and licensing boundary; its LaTeX source is in
[`research/paper/`](../research/paper/). The
[benchmark report](../reports/output/qwen3-tts-native-vs-sglang-stock-dgx-spark-2026-07-17-428307c-report.pdf)
contains the per-round measurements.

Use [`CITATION.cff`](../CITATION.cff) when citing the software. When citing a
performance result, also record the release tag, image digest, hardware, model
revision, and evidence-manifest SHA-256 from the report.

## Contributing and security

Read [CONTRIBUTING.md](../CONTRIBUTING.md) before opening a change and report
vulnerabilities according to [SECURITY.md](../SECURITY.md). Participation is
governed by the [Code of Conduct](../CODE_OF_CONDUCT.md).
