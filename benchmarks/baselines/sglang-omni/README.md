# SGLang-Omni 0.1.0 Stock Comparator

This directory defines the version-pinned and auditable **stock SGLang-Omni
comparator** for Qwen3-TTS-12Hz-1.7B-VoiceDesign on an NVIDIA DGX Spark. It
exists only for a fair native-versus-framework benchmark. It is not part of the
native production runtime, and Python is intentionally confined to this
comparator image.

No model weights are stored in this repository. No measured result is embedded
in this directory. Every published number must point to a raw evidence bundle
captured on the target Spark.

## Pinned provenance

| Component | Pin |
|---|---|
| SGLang-Omni | tag `0.1.0`, commit `8e272bcb8832ef1a3865ec48255d36f4871ec885` |
| SGLang OCI base | `lmsysorg/sglang:v0.5.12.post1-cu130-runtime@sha256:8df56b542526f4fffd5372f7f65a583c7852e50442c1f43c9c3feddfd93944a4` |
| Linux ARM64 manifest | `sha256:f9860e7a07845585ccf082ec97bba712086bf10ef9ddaa317085a4a0316a4b8e` |
| SGLang build | `0.5.12.post1`, source commit `5a15cde858ea09b77116212a39356f2fc51b8584` |
| PyTorch / Transformers | `2.11.0` / `5.6.0` |
| Dependency installer | `uv 0.11.16`, ARM64 wheel SHA-256 `cfe1f06fb8f135a735a961065d5ee90f99cccf41749fb1f964edb5b3c3dae19b` |
| qwen-tts | `0.1.1`, wheel SHA-256 `11a290d8dabc7ef91a90c54478c8ab19b3edb1d85c0882313721892bdc4af15d` |
| System SoX | Ubuntu Noble ARM64 `14.4.2+git20190427-4build4` |
| VoiceDesign model | revision `5ecdb67327fd37bb2e042aab12ff7391903235d3` |

The OCI index was verified to contain both AMD64 and ARM64 manifests; this
comparator selects the ARM64 manifest. The base image itself fixes CPython 3.12,
CUDA 13.0, the SGLang runtime, and the compute dependency stack. Additional
Qwen packages are installed from [`requirements-qwen.txt`](requirements-qwen.txt)
with hashes and without dependency resolution.

SGLang-Omni 0.1.0 itself does not publish a complete dependency lock: several
ancillary packages use version ranges, and its declared graph contains mutually
incompatible Protobuf requirements (`s3prl` requires Protobuf 4.21 or newer,
while `descript-audiotools` requires a version below 3.20). Pip's strict resolver
therefore rejects the unmodified full declaration. The pinned upstream Spark
recipe uses uv to install that declared environment and then captures the known
incompatibilities. This comparator pins and hashes the same installer rather
than silently dropping unrelated dependencies.

Upstream pins the compute-critical Torch, Transformers, SGLang, relay, and
kernel versions. A qualifying run identifies the resolved comparator by its
immutable image ID or registry digest and attaches the complete `pip freeze`,
`pip check`, system-package list, and image inspection produced by
`scripts/capture-provenance.sh`. Rebuilding source without comparing those
records is not sufficient evidence of an identical benchmark environment.

## Stock source versus platform patch

The comparator applies exactly one patch:

- remove the unconditional `torchcodec==0.11.1` declaration because that
  release has no Linux ARM64 wheel;
- retain PyAV and the media stack already provided by the pinned SGLang image;
- change no scheduler, model, tokenizer, vocoder, API, or transport code.

The patch and its classification are recorded in
[`patches/manifest.json`](patches/manifest.json). There is no custom streaming
patch. If a progressive SGLang experiment is developed later, it must use a
different image tag and result series such as `sglang-omni-custom-streaming`.
It must never be reported as stock SGLang-Omni.

## Stock Qwen3-TTS streaming verdict

SGLang-Omni 0.1.0 exposes streaming HTTP transports, including SSE and raw
`audio/pcm`. That transport capability is not the same as progressive
Qwen3-TTS generation.

The stock 0.1.0 VoiceDesign source path is completion-buffered:

1. `sglang_omni/models/qwen3_tts/config.py` defines a linear
   `preprocessing -> tts_engine -> vocoder` pipeline. The engine has no
   `stream_to` edge to the vocoder.
2. `sglang_omni/models/qwen3_tts/model_runner.py::post_process_outputs`
   appends every generated codec frame to `data.output_codes`.
3. `sglang_omni/models/qwen3_tts/request_builders.py::apply_sglang_qwen3_tts_result`
   stacks the complete list, concatenates it, and copies the complete code
   tensor to CPU before constructing the vocoder payload.
4. `sglang_omni/models/qwen3_tts/stages.py::create_vocoder_executor` calls
   `tokenizer.decode` once with that complete code tensor and returns one
   waveform payload.
5. VoiceDesign requests also set the model prompt's `non_streaming_mode` to
   `true`; the upstream unit test explicitly checks this behavior.

Therefore, `stream=true` can cause the HTTP layer to return a streamed response,
but stock VoiceDesign does not progressively decode PCM while autoregressive
generation is still running. In this comparator, **TTFA is the timestamp of the
first actual PCM byte received by the client**, even when that byte arrives only
after full model generation. The external benchmark client must also record the
last-byte timestamp and packet-arrival distribution so the classification is
empirically confirmed on the target machine.

## Build on DGX Spark

From this directory:

```bash
./scripts/build.sh
IMAGE=qwen3-tts-sglang-omni:0.1.0-stock-spark ./scripts/preflight.sh
```

The preflight fails unless it sees ARM64, CUDA 13.0, the expected package
versions, an NVIDIA GB10 with compute capability 12.1, and the Qwen3-TTS
VoiceDesign pipeline imports.

## Acquire the exact model snapshot

Use the comparator image to download the public model at the fixed revision:

```bash
./scripts/download-model.sh /srv/models/qwen3-tts-voice-design-5ecdb673
```

The script records the revision and a SHA-256 manifest next to the downloaded
files. Alternatively, `MODEL_DIR` may point directly to a Hugging Face cache
snapshot directory whose basename is the full revision hash. The run script
then mounts the repository root as well, so the snapshot's links into `blobs`
remain valid. The run script refuses an unverifiable path by default.

## Start the stock server

```bash
MODEL_DIR=/srv/models/qwen3-tts-voice-design-5ecdb673 \
  ./scripts/run.sh
```

The server binds to `127.0.0.1:8000`, uses host IPC for the stock shared-memory
relay, mounts the model read-only, and enables Hugging Face offline mode. It
does not download weights at startup. The stock SGLang memory and CUDA-graph
defaults remain unchanged; any tuning creates a separate benchmark profile.

### Memory-admission caveat on a shared Spark

SGLang computes its static-pool admission headroom from the observed free
memory using the equivalent relationship
`post_free - pre_free * (1 - mem_fraction_static)`. A large unrelated GPU
process changes both the admission result and the amount of memory available to
the engine. The pinned Qwen3-TTS stage config sets the stock value to `0.85`.
On the target Spark, diagnostic starts with an approximately 98 GB Ephraim
process still resident failed even at explicit override fractions `0.05`,
`0.08`, and `0.10`. Those lower values are not the stock profile, and their
failures describe a contaminated machine state rather than SGLang capacity or
performance.

Do not tune `mem_fraction_static` merely to make the comparator coexist with
another inference service. Before the qualifying A/B run, stop every unrelated
CUDA process, wait for GPU memory and utilization to return to the recorded idle
baseline, and then launch exactly one engine. Preserve failed shared-machine
starts only as diagnostic logs; exclude them from benchmark aggregates.

Buffered WAV smoke request:

```bash
./scripts/request-buffered.sh /tmp/sglang-stock.wav
```

Raw PCM transport smoke request:

```bash
./scripts/request-raw-pcm.sh /tmp/sglang-stock.pcm
```

The raw-PCM script demonstrates the API shape. It does not, by itself, prove
progressive model output. Use the Rust HTTP benchmark client for byte-arrival
timestamps.

## Fair comparison protocol

A publishable comparison must satisfy all of the following gates:

1. Run native and SGLang on the same DGX Spark, driver, power mode, model
   revision, corpus, language hints, voice descriptions, seeds, BF16 precision,
   maximum codec frames, and sampling values.
2. Set talker sampling explicitly to temperature `0.9`, top-p `1.0`, top-k
   `50`, and repetition penalty `1.05`. Preserve the checkpoint's code-predictor
   defaults of temperature `0.9`, top-p `1.0`, and top-k `50` in both engines.
3. Never run the engines concurrently. Stop one engine, wait for GPU memory and
   utilization to return to idle, then start the other. An unrelated CUDA
   process invalidates performance qualification.
4. Record cold startup separately. Perform at least 24 unmeasured warmups per
   engine and workload before collecting data.
5. Measure at least 200 requests each at concurrency B1, B3, and B6. Alternate
   native-first and SGLang-first rounds to reduce drift and thermal bias.
6. Use the same external Rust client for both APIs. Measure request start,
   response headers, first non-empty PCM byte, every subsequent byte arrival,
   final byte, decoded sample count, and audio duration.
7. Report TTFA p50/p95/p99, end-to-end latency p50/p95/p99, RTF p50/p95/p99,
   throughput, error rate, cancellation behavior, and packet-gap p99. Do not
   derive SGLang TTFA from internal token timing.
8. Sample process RSS, GPU memory, utilization, temperature, clock, and board
   power at no less than 5 Hz. Report idle-adjusted energy per generated audio
   minute together with the raw time series.
9. Retain every request JSONL row, environment snapshot, image ID, package
   freeze, model manifest, command line, server log, and SHA-256 manifest.
10. Label stock SGLang as completion-buffered if the first PCM byte remains
    coupled to completed generation. Do not present HTTP chunked transfer as
    progressive synthesis.

Capture immutable build and machine provenance before a benchmark round:

```bash
./scripts/capture-provenance.sh /srv/benchmarks/sglang-stock/provenance
```

## License and attribution

SGLang-Omni is Apache-2.0. The Qwen3-TTS model revision is also published under
Apache-2.0. The comparator preserves upstream source history in
`/opt/sglang-omni` and records its sole packaging delta. The base container may
include additional third-party components under their respective licenses; the
captured image metadata and package freeze are required release evidence.

## Primary sources

- [SGLang-Omni 0.1.0 source tree](https://github.com/sgl-project/sglang-omni/tree/8e272bcb8832ef1a3865ec48255d36f4871ec885)
- [Official Qwen3-TTS usage and VoiceDesign configuration](https://github.com/sgl-project/sglang-omni/blob/8e272bcb8832ef1a3865ec48255d36f4871ec885/docs/basic_usage/tts.md)
- [Stock three-stage Qwen3-TTS pipeline](https://github.com/sgl-project/sglang-omni/blob/8e272bcb8832ef1a3865ec48255d36f4871ec885/sglang_omni/models/qwen3_tts/config.py)
- [Codec-frame collection in the model runner](https://github.com/sgl-project/sglang-omni/blob/8e272bcb8832ef1a3865ec48255d36f4871ec885/sglang_omni/models/qwen3_tts/model_runner.py)
- [Completion result adapter](https://github.com/sgl-project/sglang-omni/blob/8e272bcb8832ef1a3865ec48255d36f4871ec885/sglang_omni/models/qwen3_tts/request_builders.py)
- [Whole-tensor vocoder executor](https://github.com/sgl-project/sglang-omni/blob/8e272bcb8832ef1a3865ec48255d36f4871ec885/sglang_omni/models/qwen3_tts/stages.py)
- [Pinned ARM64 SGLang image](https://hub.docker.com/layers/lmsysorg/sglang/v0.5.12.post1-cu130-runtime/images/sha256-f9860e7a07845585ccf082ec97bba712086bf10ef9ddaa317085a4a0316a4b8e)
- [Pinned Qwen3-TTS VoiceDesign model revision](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/tree/5ecdb67327fd37bb2e042aab12ff7391903235d3)
