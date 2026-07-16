# Native Qwen3-TTS Artifact and Weight Loader

This crate builds and loads a reproducible native artifact for
`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`. It is a production-oriented weight and
ownership layer, not a complete TTS inference engine. It does not execute the
talker, code predictor, or neural speech decoder and therefore makes no
time-to-first-audio, real-time-factor, or audio-quality claim.

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

## Limits of the result

The full 4.06 GB model arena was not uploaded in one test. Only about 6.58 GB of
CUDA-managed unified memory was free before the BF16 decoder test, and later
about 3.57 GB was free before the F32 decoder test. Allocating the complete arena
next to unrelated live workloads would not have been a safe research action.
VoiceDesign tensors are nevertheless fully parsed, indexed, structurally
validated, mapped, and whole-file hashed.

This milestone does not implement neural forward passes, KV caches, sampling,
streaming PCM, multilingual quality, concurrency, cancellation, TTFA, RTF, or
energy measurement. Those gates belong to the subsequent execution-engine work.
