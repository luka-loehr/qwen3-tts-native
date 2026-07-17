# Qwen3-TTS Native documentation

This documentation describes the production HTTP service for the native
Qwen3-TTS 1.7B VoiceDesign runtime. The supported model is
`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`; Base, CustomVoice, the 0.6B variants,
reference-audio conditioning, and voice cloning are outside the service
contract.

The inference path is implemented in Rust and CUDA. It does not use Python,
Node.js, PyTorch, SGLang, or vLLM at runtime.

> **Publication boundary:** deployment examples consume the complete
> digest-pinned `QWEN3_TTS_IMAGE` reference recorded in the `v0.1.0` GitHub
> release. They deliberately reject missing values and mutable tags. A local
> or candidate image is not a published release, even if it starts correctly.

## Documentation map

| Document | Purpose |
| --- | --- |
| [Project README](../README.md) | Scope, capabilities, published image, verified performance, and repository map. |
| [Quickstart](QUICKSTART.md) | Pull, run, verify, and call the digest-pinned container. |
| [API](API.md) | Human-readable HTTP and multipart streaming contract. |
| [OpenAPI](openapi.yaml) | Machine-readable OpenAPI 3.1 description of the HTTP surface. |
| [Configuration](CONFIGURATION.md) | Environment variables, defaults, intrinsic limits, and deployment boundaries. |
| [Operations](OPERATIONS.md) | Health, metrics, cancellation, shutdown, security, and troubleshooting. |
| [Architecture](ARCHITECTURE.md) | Implemented Rust/CUDA components, request lifecycle, scheduling, and data flow. |
| [Container guide](../containers/README.md) | Immutable inputs, reproducible image build, runtime contents, and hardening. |
| [Benchmark protocol](../benchmarks/README.md) | Qualification rules, metric definitions, telemetry, and evidence policy. |
| [Benchmark report](../reports/README.md) | Fail-closed evidence validation and deterministic PDF generation. |
| [Contributing](../CONTRIBUTING.md) | Development workflow, required checks, evidence rules, and review expectations. |
| [Security](../SECURITY.md) | Vulnerability reporting and the deployment security model. |
| [Changelog](../CHANGELOG.md) | Versioned project changes and release evidence links. |
| [Code of Conduct](../CODE_OF_CONDUCT.md) | Community participation standards and enforcement. |
| [License](../LICENSE) | Apache License 2.0 for the application source. |
| [Model provenance](../licenses/README.md) | Model identity, licensing, and third-party notice inventory. |

## Supported deployment

| Property | Supported value |
| --- | --- |
| Container platform | `linux/arm64` |
| GPU target | NVIDIA GB10 / DGX Spark, real `sm_121` SASS |
| CUDA userspace | CUDA 13.0.3 with cuBLAS 13.1.1.3 |
| Model | Qwen3-TTS 1.7B VoiceDesign only |
| Input | UTF-8 text plus a textual voice description |
| Languages | Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian |
| Streaming output | `multipart/mixed` with signed 16-bit, little-endian, 24 kHz mono PCM parts |
| Buffered output | RIFF/WAVE with signed 16-bit, little-endian, 24 kHz mono PCM |
| Maximum native concurrency | 6 active requests |

Other CPU architectures, CUDA architectures, and GPU families are not part of
the current image contract. The image contains the pinned model weights and
must not be started with replacement weights mounted over
`/opt/qwen3-tts/model`.

## Public surface

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health/live` | Process and HTTP-loop liveness. |
| `GET` | `/health/ready` | Native engine readiness. |
| `GET` | `/v1/capabilities` | Effective formats, languages, and limits. |
| `POST` | `/v1/voice-design/speech` | Native progressive PCM or buffered WAV synthesis. |
| `POST` | `/v1/audio/speech` | Narrow, buffered-WAV compatibility alias. |
| `DELETE` | `/v1/requests/{request_id}` | Request cancellation. |
| `GET` | `/metrics` | Prompt-free Prometheus counters and gauges. |

The service has no built-in authentication, authorization, TLS termination,
CORS policy, or tenant rate limiter. Bind it to loopback or place it behind an
authenticated reverse proxy before exposing it to another network. See
[Operations](OPERATIONS.md#security-boundary).

## Contract conventions

- JSON request objects reject unknown fields.
- A caller may supply an `x-request-id` UUID; otherwise the server creates a
  UUIDv7. The same ID is returned in synthesis response headers.
- Error responses sent before streaming begins use
  `application/problem+json`. Errors after a multipart response has started
  are terminal JSON parts inside the HTTP 200 stream.
- Generated audio is not cached: successful synthesis responses include
  `Cache-Control: no-store`.
- Duration limits are safety ceilings, not target durations. Natural codec
  end-of-sequence produces `finish_reason: "stop"`; reaching the configured
  frame ceiling produces `finish_reason: "length"`.

## Source of truth

The HTTP router and schemas live under `native/qwen3-tts-server`; the bounded
request scheduler and native backend live under `native/qwen3-tts-runtime`;
the talker/predictor and codec implementations live in their respective native
crates. The production image is defined by
`containers/Dockerfile.runtime`.

If prose and executable behavior ever differ, treat the implementation and its
contract tests as authoritative, then update both this documentation and
`openapi.yaml` in the same change.
