# Qwen3-TTS Native

Native Rust and CUDA inference for
[`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign),
designed and qualified for NVIDIA DGX Spark.

The project turns text plus a natural-language voice description into
progressive 24 kHz mono PCM. The complete inference path—prompt preparation,
the 1.7B VoiceDesign talker, its 15-step code predictor, device-to-device token
handoff, the neural speech decoder, scheduling, and HTTP delivery—runs in
native Rust and CUDA. Python, Node.js, PyTorch, SGLang, and vLLM are not part of
the runtime or production image.

> **Release `v0.1.0`:** source, the native service, the hardened image,
> benchmark report, and research paper are published together in the
> [`v0.1.0` GitHub release](https://github.com/luka-loehr/qwen3-tts-native/releases/tag/v0.1.0).
> Deploy only the complete immutable GHCR reference recorded there. A branch,
> local image, candidate tag, semantic tag, or `latest` alone is not a
> digest-pinned deployment identity.

## What this project provides

- Native Qwen3-TTS 1.7B VoiceDesign inference with custom CUDA kernels and
  cuBLAS execution on real `sm_121` SASS.
- Incremental speech-token generation and neural decoding without a Python or
  framework sidecar.
- Progressive multipart PCM delivery before synthesis completes, plus bounded
  buffered WAV output.
- One shared, warmed model engine with independently owned request sessions,
  bounded concurrency, backpressure, cancellation, and graceful shutdown.
- A versioned native C ABI beneath the HTTP service.
- Reproducible benchmark evidence, model provenance, third-party license
  reports, a CycloneDX SBOM, and BuildKit provenance/attestation support.
- A `linux/arm64` container that runs as an unprivileged user, includes the
  pinned model weights, and supports the documented hardened run profile with
  a read-only root filesystem and no Linux capabilities.

The project intentionally supports **VoiceDesign only**. It does not include
voice cloning, reference audio, speaker enrollment, the Base or CustomVoice
checkpoints, the speech-tokenizer encoder, or the retired 0.6B model.

## Supported languages and audio

The pinned model exposes ten explicit languages:

`Chinese`, `English`, `Japanese`, `Korean`, `German`, `French`, `Russian`,
`Portuguese`, `Spanish`, and `Italian`.

`Auto` is also available for automatic language selection. Values are
case-insensitive at the HTTP boundary. Languages outside this list are
rejected; in particular, Turkish is not represented by an explicit language ID
in the pinned VoiceDesign checkpoint and is not advertised as supported.

Audio is emitted as 24,000 Hz, mono, signed 16-bit little-endian PCM. Each
codec frame represents 1,920 samples, or 80 ms. The first streaming packet
contains one frame; subsequent packets contain up to four frames.

## Architecture

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

The process constructs one shared engine and performs a real one-frame warm-up
through the complete native pipeline before binding the listener. Readiness is
therefore a model-and-pipeline gate rather than a process-only check.

The runtime image contains the pinned VoiceDesign and decoder-only model
artifacts. It contains exactly the inference material it needs and excludes
compilers, development packages, Python, Node.js, PyTorch, SGLang, vLLM,
TensorRT, cuDNN, NPP, cuSPARSE, and NCCL. See the
[container documentation](containers/README.md) for the exact inputs, hashes,
labels, and build gates.

## Run the published image

The production image targets NVIDIA DGX Spark (`linux/arm64`, GB10,
`sm_121`). It is not a portable CPU image and is not qualified for x86-64 or a
different GPU architecture.

Copy the complete immutable image reference from the `v0.1.0` GitHub release,
then pull and run it by digest. The validation below intentionally fails for a
missing value, a mutable tag, or an image from another repository:

```bash
: "${QWEN3_TTS_IMAGE:?Set QWEN3_TTS_IMAGE from the v0.1.0 release notes}"
if [[ ! "$QWEN3_TTS_IMAGE" =~ ^ghcr.io/luka-loehr/qwen3-tts-native@sha256:[0-9a-f]{64}$ ]]; then
  printf 'Expected the immutable v0.1.0 GHCR reference, got: %s\n' \
    "$QWEN3_TTS_IMAGE" >&2
  exit 1
fi

docker pull "$QWEN3_TTS_IMAGE"

docker run --rm \
  --gpus device=0 \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m,uid=10001,gid=10001 \
  --pids-limit=256 \
  -p 127.0.0.1:8080:8080 \
  "$QWEN3_TTS_IMAGE"
```

Production execution uses the immutable digest. The human-readable release tag
is recorded in the release notes and OCI labels. Do not mount alternate
weights over `/opt/qwen3-tts/model`, as that would invalidate the model identity
recorded in the OCI metadata.

Wait for the complete native warm-up:

```bash
curl --fail --silent --show-error \
  http://127.0.0.1:8080/health/ready
```

Create a buffered WAV file:

```bash
curl --fail --silent --show-error \
  --header 'Content-Type: application/json' \
  --header 'x-request-id: 018f3df0-5a86-7e75-bec5-135764f0218a' \
  --data '{
    "text": "Good morning. This is a native voice-design test.",
    "voice_description": "A calm adult male voice with measured delivery and a warm low register.",
    "language": "english",
    "seed": 42,
    "max_duration_seconds": 30,
    "stream": false,
    "output_format": "wav"
  }' \
  --output speech.wav \
  http://127.0.0.1:8080/v1/voice-design/speech
```

Request progressive audio instead:

```bash
curl --fail --silent --show-error --no-buffer \
  --header 'Content-Type: application/json' \
  --data '{
    "text": "Guten Morgen. Dies ist ein nativer Streaming-Test.",
    "voice_description": "A calm adult male voice with clear, unhurried articulation.",
    "language": "german",
    "seed": 42,
    "max_duration_seconds": 30,
    "stream": true,
    "output_format": "pcm_s16le"
  }' \
  --output speech.multipart \
  http://127.0.0.1:8080/v1/voice-design/speech
```

The streaming response is `multipart/mixed`: a JSON start event, one or more
binary PCM parts, and exactly one JSON end or error event. Applications must
parse multipart boundaries; HTTP DATA-frame boundaries are not audio-packet
boundaries. See the [server contract](native/qwen3-tts-server/README.md) for
sampling controls, limits, packet headers, request IDs, and cancellation.

## HTTP API

| Method and path | Purpose |
| --- | --- |
| `GET /health/live` | Process and event-loop liveness. |
| `GET /health/ready` | Shared engine loaded, full native path warmed, and engine healthy. |
| `GET /v1/capabilities` | VoiceDesign-only languages, formats, and limits. |
| `POST /v1/voice-design/speech` | Progressive multipart PCM or buffered WAV synthesis. |
| `POST /v1/audio/speech` | Conservative buffered-WAV compatibility endpoint. |
| `DELETE /v1/requests/{request-id}` | Bounded cancellation of an admitted request. |
| `GET /metrics` | Prompt-free Prometheus request counters and engine-health gauge. |

`POST /v1/voice-design/speech` is the canonical synthesis endpoint. Its core
request fields are:

| Field | Required | Default | Meaning |
| --- | ---: | --- | --- |
| `text` | yes | — | Non-empty UTF-8 text to synthesize. |
| `voice_description` | yes | — | Natural-language VoiceDesign conditioning; never a voice name, audio sample, or clone reference. |
| `language` | no | `auto` | `auto` or one of the ten explicit model languages listed above. |
| `stream` | no | inferred | `true` selects progressive multipart PCM; `false` selects buffered WAV. |
| `output_format` | no | inferred | `pcm_s16le` for streaming or `wav` for buffered output; it must agree with `stream`. |
| `seed` | no | random | Optional reproducibility seed returned by the service. |
| `max_duration_seconds` | no | lower of 120 s and instance maximum | Safety ceiling; generation may stop earlier at natural codec EOS. |

Omitting both `stream` and `output_format` selects progressive multipart PCM.
The `/v1/audio/speech` compatibility endpoint instead accepts `input` and
`voice`, fixes the language to `auto`, and returns buffered WAV only; it is not
a general OpenAI Audio API implementation. See the
[complete HTTP API](docs/API.md) for validation, sampling, multipart framing,
response headers, errors, and the exact compatibility schema.

The standalone server binary binds to `127.0.0.1:8080` by default. The
production image listens on `0.0.0.0:8080` inside its container; the hardened
run command above publishes that port to host loopback only. The service does
not terminate TLS or provide an identity provider. Public deployments must
place it behind an authenticated, rate-limited proxy with request and response
timeouts. Review [SECURITY.md](SECURITY.md) before exposing the service.

### Data handling and deployment boundary

The native service does not add prompts, voice descriptions, generated audio,
request IDs, or language values to normal logs or Prometheus labels. That does
not make an unprotected deployment private: HTTP payloads are plaintext unless
a trusted proxy provides TLS, and proxies or clients can still record request
and response bodies. Operators are responsible for authentication,
authorization, tenant isolation, retention policy, access-controlled metrics,
and disabling payload logging throughout the surrounding stack. Do not send
sensitive text to a deployment whose transport and observability controls you
have not verified.

## Verified performance

### Controlled Native-versus-stock-SGLang comparison

The final schema-1.2 study completed two alternating-order rounds of B1, B3,
and B6 on one NVIDIA DGX Spark. All 2,600 measured requests succeeded after 24
warm-ups per cell, with no competing CUDA process. The ranges below span the
two accepted rounds:

| Profile | Native aggregate RTF | Stock aggregate RTF | Native TTFA p95 | Stock TTFA p95 |
| ---: | ---: | ---: | ---: | ---: |
| B1 | 0.800–0.803 | 0.497–0.499 | 93.89–95.58 ms | 2,691.51–2,703.91 ms |
| B3 | 0.641–0.642 | 0.186–0.197 | 215.55–216.81 ms | 2,720.93–2,814.63 ms |
| B6 | 0.617–0.618 | 0.102–0.112 | 405.38–406.04 ms | 2,873.49–3,145.88 ms |

Native peaked at 5.68 GB of observed GPU unified memory; stock SGLang peaked
at 108.90 GB. Stock SGLang achieved better aggregate throughput in every
profile. Native's measured advantages were progressive time to first audio
and approximately 19.2 times lower peak GPU unified-memory use. Stock delivery
was completion-buffered and exposed no authoritative EOS metadata, so its
TTFA and completion semantics are reported explicitly rather than treated as
equivalent to Native progressive delivery.

See the
[validated benchmark report](reports/output/qwen3-tts-native-vs-sglang-stock-dgx-spark-2026-07-17-428307c-report.pdf),
[English research paper](research/paper/qwen3-tts-native-paper.pdf), and
[benchmark protocol](benchmarks/README.md) for methodology, energy results,
limitations, and full traceability.

### Historical native baselines

The remaining results in this section are checked JSON evidence from direct
native runs on NVIDIA DGX Spark. They are historical pre-release baselines and
are not substituted for the controlled comparison above or the digest-specific
release acceptance.

### Full natural-end-of-sequence endurance

The native C ABI completed 200 measured single-stream requests after three
warm-ups. Every request reached natural codec EOS; none failed or hit the
512-frame emergency guard.

| Measurement | Result |
| --- | ---: |
| Completed requests | 200 / 200 |
| TTFA p50 / p95 / p99 | 74.01 / 76.95 / 79.82 ms |
| Request RTF p50 / p95 / p99 | 0.733 / 0.740 / 0.743 |
| Aggregate RTF | 0.734 |
| Generated audio | 926.72 s |
| Peak process RSS | 4,045,112 KiB |
| Peak additional device allocation per request | 141,285,524 bytes |

An RTF below 1.0 means synthesis completed faster than the generated audio's
playback duration. Full evidence:
[`native-runtime-natural-eos-endurance-a6bc32e.json`](benchmarks/results/native-runtime-natural-eos-endurance-a6bc32e.json).

### Official-language qualification

The multilingual native C-ABI run completed all 24 corpus entries covering all
ten explicit languages plus `Auto`. Every request streamed progressively,
preserved exact PCM copy bounds, and ended at natural codec EOS.

| Measurement | Result |
| --- | ---: |
| Completed corpus entries | 24 / 24 |
| TTFA p95 | 78.47 ms |
| Request RTF p50 / p95 | 0.745 / 0.763 |
| Aggregate RTF | 0.751 |
| Generated audio | 200.24 s |

Full evidence:
[`native-multilingual-natural-eos-ff061b6.json`](benchmarks/results/native-multilingual-natural-eos-ff061b6.json).
The saved WAV corpus is listening evidence, not an automated claim about
pronunciation, naturalness, or instruction adherence.

### Warmed HTTP server qualification

A direct native server run became ready after a full pipeline warm-up in
10.261 seconds. A German progressive request delivered its first audio in
77.884 ms and generated 4.72 seconds of audio at RTF 0.710. An Italian `Auto`
request returned a valid, unclipped 24 kHz PCM WAV at RTF 0.707. SIGTERM closed
the process and loopback port in under one second.

That historical run used a warm host filesystem cache and coexisted with an
already running SGLang service. Coexistence is not a performance comparison,
and this result is not used as the SGLang comparator. The controlled comparison
above uses a separate, complete two-engine, B1/B3/B6, two-round production
bundle that passed the schema-1.2 validator.

Full evidence:
[`native-server-startup-warmup-ce46acb.json`](benchmarks/results/native-server-startup-warmup-ce46acb.json).

### Fixed-length concurrency throughput

An earlier C-ABI scheduler qualification measured 200 fixed 320 ms requests at
each concurrency level. It is a packet-delivery and throughput test, not a
natural-EOS or audio-quality corpus.

| Concurrency | Completed | TTFA p95 | Request RTF p50 | Aggregate RTF |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 200 / 200 | 78.24 ms | 0.765 | 0.767 |
| 3 | 200 / 200 | 186.42 ms | 1.800 | 0.601 |
| 6 | 200 / 200 | 364.62 ms | 3.557 | 0.594 |

At B3 and B6, aggregate throughput was faster than real time while an
individual request was not. Full evidence:
[`native-runtime-public-c-abi-qualification.json`](benchmarks/results/native-runtime-public-c-abi-qualification.json).

Read [the benchmark protocol](benchmarks/README.md) before comparing or
republishing any result. In particular, model loading, artifact hashing, fixed
length scheduling, natural-EOS generation, HTTP transport, container overhead,
energy, and subjective listening results are separate measurements.

## Build and verification

Rust 1.97.0, CUDA 13.0.3, cuBLAS 13.1.1.3, Ubuntu 24.04 ARM64, and real
`sm_121` SASS are pinned for the production image. Reproducing the image also
requires the audited model artifact and generated release-metadata contexts.

Start with these documents:

- [Validated Native-versus-stock-SGLang benchmark report](reports/output/qwen3-tts-native-vs-sglang-stock-dgx-spark-2026-07-17-428307c-report.pdf)
- [English research paper and LaTeX source](research/paper/README.md)
- [Production image, immutable inputs, and build command](containers/README.md)
- [Exact release gates](containers/RELEASE_CHECKLIST.md)
- [HTTP server contract](native/qwen3-tts-server/README.md)
- [Native runtime and C ABI](native/qwen3-tts-runtime/README.md)
- [VoiceDesign talker and predictor](native/qwen3-tts-native/README.md)
- [Incremental neural codec](native/qwen3-tts-native-codec/README.md)
- [Benchmark protocol and evidence](benchmarks/README.md)
- [Release metadata generation](tools/release-metadata/README.md)

The Dockerfile validates the pinned artifact hashes, CUDA architecture,
dynamic dependencies, final package inventory, non-root ownership, licenses,
and SBOM inputs during the build. Registry attestations, vulnerability scans,
signature verification, clean pull, and digest-specific GPU qualification are
post-build release gates.

## Repository layout

| Path | Contents |
| --- | --- |
| `native/qwen3-tts-native` | Artifact contract, tokenizer, VoiceDesign talker, code predictor, and CUDA kernels. |
| `native/qwen3-tts-native-codec` | Stateful incremental speech-tokenizer decoder. |
| `native/qwen3-tts-runtime` | Scheduler, native backend, versioned C ABI, and lifecycle tests. |
| `native/qwen3-tts-server` | Bounded Rust HTTP transport, healthcheck, metrics, streaming, and WAV output. |
| `native/qwen3-tts-bench` | Real-runtime qualification harness and report helpers. |
| `native/qwen3-tts-http-bench` | Standalone external HTTP client for synchronized Native and SGLang measurements. |
| `benchmarks` | Corpora, deterministic fixtures, protocols, and immutable result records. |
| `reports` | Fail-closed evidence validation and deterministic black-and-white PDF generation. |
| `docs` | Quickstart, API, configuration, operations, architecture, and OpenAPI documentation. |
| `containers` | Reproducible builder, hardened runtime image, and release checklist. |
| `licenses` | Model provenance and third-party notices. |
| `tools/release-metadata` | Pinned license and CycloneDX metadata pipeline. |
| `notes` | Architecture, model-contract, artifact, codec, and toolchain decisions. |
| Root community files | Contribution, security, conduct, changelog, and Apache-2.0 license policies. |

## Research and citation

The [English research paper](research/paper/qwen3-tts-native-paper.pdf)
describes the system design, native implementation, streaming contract,
controlled evaluation, limitations, reproducibility, and licensing boundary.
Its complete LaTeX source is versioned under [`research/paper/`](research/paper/).
The independent [benchmark report](reports/output/qwen3-tts-native-vs-sglang-stock-dgx-spark-2026-07-17-428307c-report.pdf)
contains the detailed per-round measurements and evidence identity.

Use [`CITATION.cff`](CITATION.cff) when citing the software. When citing a
performance result, also record the release tag, immutable image digest,
hardware, model revision, and evidence-manifest SHA-256 from the report rather
than citing an unqualified branch or mutable container tag.

## Contributing and security

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a change. Documentation
and benchmark narratives must be written in English, measured evidence must
remain reproducible and unaltered, model weights and secrets must never enter
Git, and shared branches must never be force-pushed.

Report vulnerabilities privately according to [SECURITY.md](SECURITY.md).
Community participation is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License and model provenance

The application source is licensed under the
[Apache License 2.0](LICENSE). The pinned Qwen3-TTS model is separately
attributed and licensed by its upstream publisher under Apache-2.0. NVIDIA and
Ubuntu components, Rust dependencies, and other third-party material remain
subject to their respective terms. See the [license inventory](licenses/README.md) and
[third-party notices](licenses/THIRD_PARTY_NOTICES.md).
