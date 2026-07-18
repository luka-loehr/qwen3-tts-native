# Native Qwen3-TTS Model Runtime

This crate builds and loads a reproducible native artifact for
`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` and executes its talker and 15-step code
predictor. It provides the production-oriented weight, ownership, prompt,
sampling, and incremental codec-frame layers. PCM decoding remains a separate
native crate and end-to-end delivery remains the responsibility of the public
runtime crate.

The runtime path uses Rust and a narrow CUDA C ABI. It has no Python, Node.js,
Candle, HTTP, or background-thread dependency.

## Artifact contract

The packer accepts only the audited Hugging Face snapshot revision
`5ecdb67327fd37bb2e042aab12ff7391903235d3`. It publishes a new directory by
atomic rename and refuses to overwrite an existing output.

The final BF16 artifact contains:

| Material | Tensors | Parameters | Tensor payload | File bytes |
| --- | ---: | ---: | ---: | ---: |
| VoiceDesign talker and code predictor, BF16 | 404 | 1,916,676,352 | 3,833,352,704 | 3,833,402,552 |
| Speech decoder only, BF16 | 271 | 114,323,137 | 228,646,274 | 228,678,506 |
| **Mapped model total** | **675** | **2,030,999,489** | **4,061,998,978** | **4,062,081,058** |

The speech-tokenizer encoder is never copied. The runtime artifact does not
contain the original 496-tensor, approximately 682 MB F32 tokenizer checkpoint.
BF16 conversion is performed offline with round-to-nearest-even. F32 decoder
artifacts are also supported for validation, without runtime conversion.

Every output is a regular file; symlinks, special files, path traversal, missing
material, changed shapes, changed dtypes, changed counts, changed byte lengths,
noncanonical hashes, unsafe tensor ordering, and noncontiguous arena offsets are
rejected before GPU allocation.

`manifest.json` records, for every tensor:

- canonical name and component;
- dtype and complete shape;
- parameter and byte counts;
- contiguous device-arena byte offset;
- SHA-256 of the individual tensor payload.

It also records whole-file byte counts and SHA-256 values. Full verification
hashes every material file; contract verification provides a faster trusted-disk
startup path while retaining all structural checks.

## Build

The tested Rust toolchain is 1.97.0. The tested CUDA toolchain is CUDA 13.0.88 in
`nvcr.io/nvidia/tensorrt:25.11-py3`, producing real SM 12.1 SASS.

```bash
docker run --rm \
  -v "$PWD:/work" \
  -w /work/native/qwen3-tts-native \
  codex/qwen3-tts-rust-builder:1.97.0 \
  sh -c 'cargo test --all-targets && cargo clippy --all-targets -- -D warnings && cargo build --release --bins'

docker run --rm \
  -v "$PWD:/work" \
  -w /work/native/qwen3-tts-native \
  nvcr.io/nvidia/tensorrt:25.11-py3 \
  sh -c 'cmake -S native -B build-sm121 -DCMAKE_BUILD_TYPE=Release && cmake --build build-sm121 --parallel'
```

No Rust, CUDA development package, Python package, or Node.js package needs to
be installed on the host.

## Pack and validate

```bash
native/qwen3-tts-native/target/release/pack_artifact \
  /path/to/huggingface/snapshots/5ecdb67327fd37bb2e042aab12ff7391903235d3 \
  /path/to/qwen3-tts-1.7b-voice-design-bf16 \
  --codec-dtype bf16 \
  --buffer-mib 8 \
  --report artifact-pack.json

native/qwen3-tts-native/target/release/load_artifact \
  /path/to/qwen3-tts-1.7b-voice-design-bf16 \
  --verification full \
  --staging-mib 8 \
  --report artifact-load.json
```

Use `--codec-dtype f32` only for numerical validation. BF16 is the intended
production decoder representation.

## Upload to CUDA

The upload CLI exercises the same reusable Rust and C ABI used by a future
runtime. It allocates one contiguous device arena and one fixed-capacity pinned
staging buffer. No tensor-sized host copy is created.

```bash
docker run --rm --gpus all \
  -v "$PWD:/work" \
  -v /path/to/artifacts:/artifacts:ro \
  -w /work/native/qwen3-tts-native \
  nvcr.io/nvidia/tensorrt:25.11-py3 \
  /work/native/qwen3-tts-native/target/release/upload_artifact \
  /artifacts/qwen3-tts-1.7b-voice-design-bf16 \
  /work/native/qwen3-tts-native/build-sm121/libqwen3_tts_cuda.so \
  --scope decoder \
  --device 0 \
  --staging-mib 8 \
  --report /work/decoder-upload.json
```

The CLI deliberately drops both source mappings after `finish` and before
readback. Exact first and final byte probes then demonstrate that the CUDA arena
owns an independent copy.

## Reusable API

The Rust library exposes:

- `NativeArtifact::open` for identity, file, contract, and tensor-index
  validation;
- namespace-safe tensor lookup for talker, code predictor, and speech decoder;
- descriptor lookup by canonical tensor name;
- `TensorDescriptor::as_c_view` for a borrowed `repr(C)` view containing name,
  hash, dtype, rank, shape pointer, arena offset, and byte count;
- aligned, bounded `TensorRef::chunks` iteration;
- explicit model memory metrics separating mapped files from owned host copies
  and runtime conversion scratch;
- `DeviceWeightBuffer::create`, `upload`, `finish`, `state_info`,
  `device_pointer`, `readback`, and `release`.

The descriptor view borrows its name and shape pointers. They remain valid only
while the originating Rust descriptor is alive and unmodified.

The CUDA library exposes an opaque `Qwen3TtsDeviceBuffer` through:

- `qwen3_tts_device_buffer_create`;
- `qwen3_tts_device_buffer_upload`;
- `qwen3_tts_device_buffer_finish`;
- `qwen3_tts_device_buffer_read`;
- `qwen3_tts_device_buffer_data`;
- `qwen3_tts_device_buffer_destroy`.

Calls are synchronous at the ownership boundary. The buffer is single-owner;
Rust RAII destroys the device allocation and pinned staging memory exactly once.

## Measured DGX Spark result

Measurements were taken on the NVIDIA GB10 with an 8 MiB staging buffer.

| Operation | Result | Peak host RSS |
| --- | ---: | ---: |
| BF16 artifact pack with per-tensor hashes | 24.05 s | 628,992 KiB |
| Contract-only open | 4 ms | 6,020 KiB |
| Full 4.06 GB SHA-256 verification | 10.567 s | 11,248 KiB |
| BF16 decoder native CUDA copy | 15.076 ms, 14.12 GiB/s | — |
| BF16 decoder upload section | 53.834 ms | 490,600 KiB |
| F32 decoder native CUDA copy | 51.920 ms, 8.20 GiB/s | — |
| F32 decoder upload section | 93.327 ms | 711,124 KiB |

The BF16 decoder allocated 228,646,274 device bytes plus 8,388,608 pinned
staging bytes. The loader reported 4,062,081,058 mapped file bytes, zero owned
host weight-copy bytes, and zero runtime conversion bytes. NVIDIA Compute
Sanitizer reported zero errors for the BF16 decoder upload.

Two independent BF16 pack runs were byte-identical:

- manifest: `9bb96a8d24bbb2d8933245e27083b8e7290346b776306dcb8a8f3aed68594527`;
- VoiceDesign weights: `391e8db219f292c515297cdceeb43e4eae67cdde35fa57e79a6a8a532fca0522`;
- BF16 decoder weights: `062caa0a31346422410e4c0d2494aec14be20553f8cb0b71a875329de99ce180`.

## Artifact-loader result boundaries

The full 4.06 GB model arena was not uploaded in one test. Only about 6.58 GB of
CUDA-managed unified memory was free before the BF16 decoder test, and later
about 3.57 GB was free before the F32 decoder test. Allocating the complete arena
next to unrelated live workloads would not have been a safe research action.
VoiceDesign tensors are nevertheless fully parsed, indexed, structurally
validated, mapped, and whole-file hashed.

These measurements qualify artifact creation, validation, and upload only. They
must not be presented as talker TTFA, RTF, streaming, or audio-quality results;
the execution measurements below use separate benchmark evidence.

## Talker and code predictor runtime

This crate is the Rust and CUDA research implementation of the
Qwen3-TTS-12Hz-1.7B-VoiceDesign talker and its 15-step code predictor. It loads
the official BF16 Safetensors checkpoint, prepares the official VoiceDesign
prompt, and incrementally emits one 16-codebook codec frame per call.

The component is intentionally independent of any surrounding production
services.
It emits codec tokens, not PCM audio. The speech-tokenizer decoder and network
transport are separate integration layers.

### Architecture

The runtime separates immutable model state from request state:

```text
Arc<NativeTalkerModel>
  |-- 404 immutable BF16 tensors on the GPU
  |-- model configuration and tokenizer
  `-- bounded inactive-session pool
          |
          `-- NativeTalkerSession per request
                |-- independent CUDA stream
                |-- independent cuBLAS handle
                |-- independent talker and predictor KV caches
                |-- independent workspace, RNG, cursor, and counters
                `-- caller-owned 16-codebook frame output
```

`NativeTalkerModel` is `Send + Sync + 'static`. `NativeTalkerSession` is
`Send + 'static` and deliberately not `Sync`: a session may move between host
threads, but only one thread may mutate a particular session at a time. Separate
sessions can execute `prefill` and `next_frame` concurrently on independent CUDA
streams. No global inference lock is used.

The inactive-session pool is locked only while a handle is acquired or returned.
The lock is never held during tokenization, prefill, sampling, predictor execution,
or talker execution. A failed native operation makes the session non-recyclable.

### Memory model

The immutable model weights are uploaded once per `NativeTalkerModel`:

| Allocation | Bytes |
| --- | ---: |
| Shared BF16 weights | 3,833,352,704 |
| Tensor count | 404 |

Each active session owns only its KV caches, workspace, RNG, and counters.
`max_sequence_length` is a hard safety limit, not an unconditional allocation.
The actual capacity is calculated as:

```text
prompt token count + requested maximum codec frames
```

It is rounded to a small 32-position reusable size class. A request that exceeds
its hard limit fails explicitly instead of truncating or silently growing into an
unbounded allocation.

For the qualification fixture with 26 prompt tokens and a 256-frame safety
budget, the allocated capacity is 288 positions:

| Per-session allocation | Bytes |
| --- | ---: |
| Talker KV cache | 33,030,144 |
| Predictor KV cache | 327,680 |
| Workspace | 21,544,172 |
| Total | 54,901,996 |
| Six active sessions | 329,411,976 |

Shared weight bytes are reported once. Per-session and aggregate session bytes
are reported separately by `benchmark-sessions`.

### Rust API

```rust,no_run
use std::sync::Arc;
use qwen3_tts_native::native_talker::{NativeTalkerModel, VoiceDesignRequest};

# fn example() -> anyhow::Result<()> {
let model: Arc<NativeTalkerModel> = NativeTalkerModel::load(
    "build-native/libqwen3_tts_cuda.so".as_ref(),
    "/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign".as_ref(),
    0,
)?;

let mut request = VoiceDesignRequest::new(
    "Hallo.",
    "A calm male voice, clear, relaxed, and natural.",
    "German",
);
request.max_frames = 256;
request.max_sequence_length = 1_024;

let mut session = model.start(request)?;
while let Some(frame) = session.next_frame()? {
    // frame.codes contains codebooks 0 through 15 for one 12.5 Hz frame.
}
# Ok(())
# }
```

Dropping a successfully prefetched session returns its native resources to the
bounded pool. `cancel()` marks a request as cancelled; dropping it is sufficient
to recycle the completed native call boundary. The session retains an `Arc` to
the model, so model weights and the dynamic library remain alive until the last
session is gone.

`SessionStartTiming` separates:

- tokenization;
- prompt-plan construction;
- session acquisition;
- native session creation;
- pooled-session reset;
- prefill.

Warm pool hits report zero native creation time.

### C ABI

The public header is `native/include/qwen3_tts_native.h`. The request lifecycle is
exposed through two opaque handle families.

Model operations:

- `qwen3_tts_model_create`
- `qwen3_tts_model_upload_tensor`
- `qwen3_tts_model_finalize`
- `qwen3_tts_model_destroy`

Session operations:

- `qwen3_tts_session_create`
- `qwen3_tts_session_reset`
- `qwen3_tts_session_prefill`
- `qwen3_tts_session_next_frame`
- `qwen3_tts_session_state_info`
- `qwen3_tts_session_destroy`

`qwen3_tts_session_next_frame` writes into a caller-owned array with
`QWEN3_TTS_CODEC_CODEBOOKS` entries. It returns the next semantic token and timing
metadata separately. A session handle never owns or duplicates model weights.

### Build

The project is built on the DGX Spark. A CUDA 13-compatible environment with
cuBLAS and an SM 12.1 compiler target is required.

```bash
cmake -S native -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --parallel
cargo build --locked --release
cargo test --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
```

The checkpoint path is supplied at runtime. Model files are never copied into the
repository or container image.

### Generate codec frames

```bash
target/release/qwen3-tts-native generate-codes \
  build-native/libqwen3_tts_cuda.so \
  /models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --text "Hallo." \
  --instruction "A calm male voice, clear, relaxed, and natural." \
  --language German \
  --max-frames 256 \
  --max-sequence 1024 \
  --seed 11
```

Official explicit language prompt IDs are covered for Chinese, English, German,
Italian, Portuguese, Spanish, Japanese, Korean, French, and Russian. `Auto` uses
the checkpoint's official automatic-language prefix. Languages outside that list
must be treated as empirical best effort.

### Session qualification

```bash
target/release/qwen3-tts-native benchmark-sessions \
  build-native/libqwen3_tts_cuda.so \
  /models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --text "Hallo." \
  --instruction "A calm male voice, clear, relaxed, and natural." \
  --language German \
  --max-frames 256 \
  --max-sequence 1024 \
  --warm-requests 200 \
  --rounds 3 \
  --output ../../benchmarks/results/native-talker-session-qualification.json
```

The command performs all of the following in one persistent model process:

- 200 measured warm requests with TTFA p50, p95, p99, and maximum;
- separate tokenization, prompt-plan, acquire, create, reset, and prefill timings;
- exact sampled parity for B1, B3, and B6 on real concurrent host threads;
- exact greedy B3 parity;
- duplicate-seed equality and different-seed isolation;
- cancellation and sibling-session isolation;
- single-thread round-robin interleaving;
- movement of an owned session to another host thread;
- pool-hit and per-session memory accounting;
- explicit failure when the corpus reaches the frame limit before codec EOS.

Run throughput qualification only on an otherwise idle GPU. Reports produced
while another CUDA workload is active must be labelled contaminated and must not
be promoted as qualifying evidence.

### Latest uncontaminated qualification

The tracked report is
[`benchmarks/results/native-talker-session-qualification.json`](../../benchmarks/results/native-talker-session-qualification.json).
It was captured with no competing CUDA compute process and contains 200 measured
warm requests plus three measured rounds for every concurrency level.

| Metric | Result |
| --- | ---: |
| Cold model load, excluded from warm TTFA | 9,392.57 ms |
| Warm requests / pool hits | 200 / 200 |
| Warm TTFA p50 | 61.36 ms |
| Warm TTFA p95 | 65.51 ms |
| Warm TTFA p99 | 66.45 ms |
| Warm TTFA maximum | 68.46 ms |
| Warm session-create p95 | 0.00 ms |
| Warm session-reset p95 | 0.011 ms |
| Warm prefill p95 | 19.35 ms |
| B1 mean aggregate RTF | 0.569 |
| B3 mean aggregate RTF | 0.428 |
| B6 mean aggregate RTF | 0.404 |

B1 TTFA was 61.06–62.39 ms in the measured concurrency rounds. B3 was
133.23–148.68 ms and B6 was 257.29–269.97 ms. The sub-200-ms warm TTFA gate is
the persistent B1 request gate; concurrent TTFA is reported independently rather
than hidden inside the aggregate throughput figure.

All measured sessions used capacity 288 for the 26-token prompt and 256-frame
safety budget. Full sampled sequences matched their standalone references for
B1, B3, and B6. Greedy B3, seed isolation, cancellation, sibling drop,
round-robin interleaving, cross-thread movement, and EOS-before-limit checks all
passed.

### Component boundary

This crate qualifies the VoiceDesign talker and code predictor. A complete audio
service still requires the native speech-tokenizer decoder, packetization, raw
socket streaming verification, audio-quality fixtures, energy measurement, and
production integration. Those layers must consume this API without moving model
weights into per-request state.
