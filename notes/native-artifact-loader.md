# Native Artifact Loader Validation Record

## Scope and source

This record covers the model-material pipeline for the native Qwen3-TTS 1.7B
VoiceDesign research runtime. Work was performed only in the standalone DGX
Spark playground. It did not modify the host's unrelated production services,
containers, system package database, or running inference service.

The source material was the official local Hugging Face cache for
`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`, pinned to revision
`5ecdb67327fd37bb2e042aab12ff7391903235d3`. The official Qwen3-TTS Python
source and the local MIT-licensed `qwen3-tts-rs` checkout were inspected for
namespace and architecture confirmation. No Candle dependency or external
implementation code was copied into this loader.

## Implemented boundary

The implementation consists of four independent layers:

1. an embedded immutable contract generated from the audited model inventories;
2. a Rust packer that flattens source symlinks, filters the tokenizer to
   `decoder.*`, performs optional offline BF16 conversion, hashes every tensor,
   and atomically publishes the artifact;
3. a Rust loader that validates identity, paths, files, aggregate contracts, and
   the complete tensor index before exposing borrowed mmap-backed tensor views;
4. an opaque CUDA device buffer that accepts bounded staged uploads and exposes
   state, metrics, readback, data pointer, and deterministic release.

There is no runtime dtype conversion. The BF16 decoder checkpoint is serialized
offline with round-to-nearest-even. The original tokenizer encoder and full F32
source checkpoint are not production material.

## Failure behavior

The packer refuses zero-sized buffers and existing outputs. It writes into a
process-specific staging directory, syncs files and directories, and publishes
by atomic rename. Incomplete staging material is removed on failure.

The loader rejects:

- a symlink artifact root;
- any listed or unlisted symlink below the root;
- non-regular material;
- absolute paths, empty paths, `.` components, or `..` traversal;
- missing required weight files;
- wrong model identity or revision encoding;
- wrong file sizes or whole-file hashes;
- wrong tensor counts, names, order, components, dtypes, shapes, parameters, or
  byte counts;
- noncontiguous arena offsets;
- uppercase, malformed, missing, or duplicate tensor hashes;
- any decoder tensor outside the `decoder.*` namespace.

The CUDA buffer rejects zero allocations, invalid staging capacity, out-of-range
uploads, overlapping or skipped sequential regions, upload after finish, and
readback outside the allocation. Rust owns the native handle uniquely and calls
destroy once through RAII or explicit `release`.

## Artifact result

The final regular-file artifact is
`qwen3-tts-1.7b-voice-design-bf16-indexed`. Its manifest is 275,866 bytes.

| Component | Tensor count | Payload bytes | File bytes | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| VoiceDesign BF16 | 404 | 3,833,352,704 | 3,833,402,552 | `391e8db219f292c515297cdceeb43e4eae67cdde35fa57e79a6a8a532fca0522` |
| Decoder-only BF16 | 271 | 228,646,274 | 228,678,506 | `062caa0a31346422410e4c0d2494aec14be20553f8cb0b71a875329de99ce180` |

The manifest SHA-256 is
`9bb96a8d24bbb2d8933245e27083b8e7290346b776306dcb8a8f3aed68594527`.
A second independent pack produced byte-identical manifest and weight files.
The repeat artifact was deleted after comparison.

## Memory ownership

The BF16 loader reports:

| Class | Bytes | Meaning |
| --- | ---: | --- |
| Mapped model files | 4,062,081,058 | Read-only virtual file mappings |
| Tensor payload views | 4,061,998,978 | Borrowed slices inside the mappings |
| Owned host weight copy | 0 | No model-sized heap allocation |
| Runtime dtype conversion | 0 | Conversion is offline |
| Pinned upload staging | 8,388,608 | Fixed reusable staging capacity |
| BF16 decoder CUDA arena | 228,646,274 | Independently owned device allocation |

Mapped byte count is not committed-RAM count. Contract-only open used 6,020 KiB
peak RSS; full file hashing used 11,248 KiB. The upload test deliberately dropped
both model mappings before device readback. Successful exact readback proves the
device arena does not borrow source mappings.

The first per-tensor hashing implementation retained touched mmap pages and
reached 4,433,728 KiB peak RSS. Eager `MADV_DONTNEED` after each immutable tensor
range reduced the final pack peak to 628,992 KiB without changing artifact bytes.

## GB10 validation

Rust 1.97 tests passed for all targets: 21 library tests and two binary inventory
tests. `cargo clippy --all-targets -- -D warnings` passed. The CUDA library built
with CUDA 13.0.88 as real SM 12.1 code.

| Gate | Result |
| --- | ---: |
| BF16 pack | 24.05 s, 628,992 KiB peak RSS |
| BF16 repeat pack | 24.26 s, 629,524 KiB peak RSS |
| Contract-only open | 4 ms, 6,020 KiB peak RSS |
| Full whole-file SHA-256 open | 10.567 s, 11,248 KiB peak RSS |
| BF16 decoder native upload | 15.076 ms, 14.12 GiB/s |
| BF16 decoder upload section | 53.834 ms |
| BF16 readback | Exact after source mmap release |
| NVIDIA Compute Sanitizer | 0 errors |
| F32 pack | 25.44 s, 621,068 KiB peak RSS |
| F32 decoder native upload | 51.920 ms, 8.20 GiB/s |
| F32 decoder upload section | 93.327 ms |
| F32 readback | Exact after source mmap release |

The F32 artifact was test-only and deleted after validation.

## Explicit limitations

The full VoiceDesign plus decoder CUDA arena was not allocated. CUDA reported
about 6.58 GB free before the BF16 decoder allocation, then about 3.57 GB before
the later F32 decoder allocation. A simultaneous 4.06 GB allocation would have
left unsafe headroom beside unrelated live workloads. Full VoiceDesign material
was still parsed, indexed, mapped, and whole-file hashed.

This milestone does not execute model layers and cannot establish TTFA, RTF,
streaming cadence, multilingual quality, voice adherence, cancellation,
concurrency, or energy. It is the validated material and ownership foundation
required before those claims can be measured honestly.
