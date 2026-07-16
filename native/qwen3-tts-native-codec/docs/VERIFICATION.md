# Verification Evidence

## Environment

- Date: 2026-07-16
- Device: NVIDIA GB10 in the DGX Spark
- CUDA compiler/runtime container: CUDA 13.0.88,
  `nvcr.io/nvidia/tensorrt:25.11-py3`
- CUDA architecture: SM 12.1 (`CMAKE_CUDA_ARCHITECTURES=121`)
- Rust: 1.97.0, edition 2024
- Model: Qwen3-TTS-12Hz-1.7B-VoiceDesign speech-tokenizer decoder
- BF16 artifact SHA-256:
  `062caa0a31346422410e4c0d2494aec14be20553f8cb0b71a875329de99ce180`
- Official oracle: Qwen tokenizer implementation, eager attention, F32
  execution, TF32 disabled

The official-reference container is an offline test oracle only. Python is not
part of the native runtime.

## Artifact contract

| Property | Verified value |
| --- | ---: |
| Canonical names | exactly 271 `decoder.*` tensors |
| Parameters | 114,323,137 |
| Source dtype | 271 BF16, 0 F32 |
| Tensor payload | 228,646,274 bytes |
| Safetensors file | 228,678,506 bytes |
| Execution dtype | F32 |
| Device weight allocation | 457,292,548 bytes |
| Upload staging limit | 8,388,608 bytes |

The native loader verifies the source shape product and byte length of every
tensor, rejects duplicates, requires the canonical model anchors, and does not
assume safetensors physical order.

## Official activation parity

The BF16 source tensors were loaded into the official Qwen decoder and expanded
to F32, matching the native execution precision. The native single-packet and
1+3 packet streams were compared independently.

| Checkpoint | Maximum absolute error | Allowed | Result |
| --- | ---: | ---: | --- |
| Decoder pre-convolution | 0.00002098 | 0.00005 | pass |
| Waveform block 1 | 0.00002503 | 0.00005 | pass |
| Waveform block 2 | 0.00006717 | 0.00010 | pass |
| Waveform block 3 | 0.00014892 | 0.00020 | pass |
| Waveform block 4 | 0.00092316 | 0.00100 | pass |
| Final SnakeBeta | 0.00154877 | 0.00200 | pass |
| Final pre-clamp | 0.00000298 | 0.000005 | pass |
| Final clamp | 0.00000298 | 0.000005 | pass |

The separate 83-frame transformer fixture crosses the 72-frame KV window with
uneven packet sizes 4/1/3/2. Its maximum absolute error is `1.86e-7`, and no
prefix is recomputed.

## PCM and lifecycle parity

The official reference contains 7,680 samples for four input frames.

| Scenario | Result |
| --- | --- |
| One four-frame packet vs official PCM | maximum 1 LSB |
| Split 1+3 packets vs official PCM | maximum 1 LSB |
| Four one-frame packets vs official PCM | maximum 1 LSB |
| Split 1+3 vs one packet | maximum 1 LSB |
| Reset replay vs first run | bit-exact |
| Short final | exactly 1,920 samples |
| Poisoned output suffix after short final | unchanged |
| Ring slots over four packets | `0, 1, 2, 0` |
| Packet after final | rejected |
| Warmup state | restored to zero |
| Warmup after stream progress | rejected |

The poisoned-suffix test specifically prevents the stale fixed-shape tail that
can occur when a short output reuses a larger slot.

## Legacy independent stream handles

The batch entry point was compared with standalone decoding of the same unique
stream inputs.

| Gate | Result |
| --- | --- |
| B=3, four one-frame packets per stream | standalone bit-exact |
| B=3 slot sequence | `0, 1, 2, 0` for every stream |
| B=3 reset replay | bit-exact |
| B=6 unique inputs | standalone bit-exact |
| B=6 final packet lengths | 1,920 / 3,840 / 5,760 samples |
| B=6 state counters | correct for all handles |
| Cross-request leakage | none detected |

The current batch implementation dispatches in array order. These results prove
state isolation, not fused-batch throughput.

## Shared model and concurrent sessions

The additive shared API loaded and warmed the 271 tensors once, then created
owned Rust sessions from one `Arc<NativeCodecModel>`. B=3 and B=6 were tested
both by deterministic interleaving and by real scoped host threads. Each
concurrent worker owned a separate CUDA stream, cuBLAS handle, events, KV,
histories, rings, and workspace. No inference lock was used.

| Gate | Result |
| --- | --- |
| Shared model Rust trait | `Send + Sync + 'static` |
| Owned session Rust trait | `Send + 'static`, deliberately not `Sync` |
| B=1 vs official PCM | maximum 1 LSB |
| B=1 reset replay | bit-exact |
| B=3 interleaved vs standalone | bit-exact |
| B=3 concurrent, 20 rounds | all streams and rounds bit-exact |
| B=3 reset replay | bit-exact |
| B=6 interleaved vs standalone | bit-exact |
| B=6 concurrent, 20 rounds | all streams and rounds bit-exact |
| Cancelled session | rejects further packets |
| Sibling after cancel/drop | bit-exact and unaffected |
| Session reference count after drops | returns to zero |

The B=3 concurrent wall time was 2,992.44 ms for 20 rounds. B=6 completed
20 rounds in 6,668.03 ms for 38.4 seconds of aggregate audio, an aggregate
wall RTF of 0.174. This measures the decoder only and is not an end-to-end TTS
throughput claim.

## Latency

Weights are loaded and the shared model is warmed before packet timing. Each
distribution contains 200 continuous real neural packets after 20 additional
packet warmups. PCM is delivered after every packet.

| Shared packet | Min | p50 | p95 | p99 | Max | p50 RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 frame / 80 ms | 8.04 ms | 12.01 ms | 12.97 ms | 16.14 ms | 16.26 ms | 0.150 |
| 4 frames / 320 ms | 49.67 ms | 62.20 ms | 65.97 ms | 68.86 ms | 70.22 ms | 0.194 |

Startup measurements use separate processes:

| Shared startup event | End-to-end |
| --- | ---: |
| Model load plus one-time warmup | 430.42 ms |
| First 80 ms chunk from a fresh session after model warmup | 8.14 ms |

The same build's immediately following legacy run measured 10.39 ms for its
first post-warmup chunk, 16.05 ms p50 for 80 ms packets, and 71.10 ms p50 for
320 ms packets. The machine was not exclusive, so this is evidence of no
regression, not a statistically controlled speedup claim.

## Memory

| Allocation group | Bytes |
| --- | ---: |
| Shared device weights, allocated once | 457,292,548 |
| Per-session device allocation | 35,034,920 |
| Per-session pinned host allocation | 46,080 |
| Transformer KV allocation reported by prototype | 7,077,888 |
| Convolution histories reported by prototype | 848,832 |
| Codec input ring | 384 |
| Device PCM ring | 46,080 |
| B=1 total device allocation | 492,327,468 |
| B=3 total device allocation | 562,397,308 |
| B=6 total device allocation | 667,502,068 |
| B=6 total pinned host allocation | 276,480 |

The exact persistent formula is
`457,292,548 + batch * 35,034,920` device bytes. Session memory reports exclude
shared weights by contract. The old B=1 total is unchanged, while additional
sessions add only mutable state rather than another 457 MB weight copy.

The Spark exposes unified memory, so `nvidia-smi` reports device memory as
`N/A`. These values come from exact owned allocation sizes in the runtime, not
from process RSS or a heuristic.

## Sanitizer and Rust gates

The shared BF16 load, model warmup, official PCM, B=1/B=3/B=6 interleaving,
real B=3/B=6 host-thread concurrency, reset/replay, cancel, drop, and sibling
isolation path was executed under:

```text
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99
```

Result:

```text
target passed: true
process exit code: 0
LEAK SUMMARY: 0 bytes leaked in 0 allocations
ERROR SUMMARY: 0 errors
```

Sanitizer used one concurrent round because memory instrumentation deliberately
distorts timing. The separate uninstrumented gate used 20 concurrent rounds.

Rust gates:

- two safetensors parser unit tests (F32 and BF16);
- one public-library integration test;
- compile-time `Send + Sync + 'static` model and `Send + 'static` session checks;
- one compile-fail Rustdoc check proving a session is not `Sync`;
- one compiling Rustdoc usage example;
- `cargo clippy --all-targets -- -D warnings`;
- `cargo build --release --locked`.

## Claim boundaries

- Latency numbers are from the real neural decoder, never the deterministic
  fixture.
- The decoder does not measure talker/token-generation latency.
- The batch API is not a fused kernel.
- Concurrent sessions share immutable weights but never mutable CUDA state.
- BF16 is the source artifact precision; execution weights are currently F32.
- No backend, frontend, production container, or network service was modified.

Machine-readable evidence is stored in:

- `results/shared-session-parity-2026-07-16.json`;
- `results/shared-neural-benchmark-2026-07-16.json`;
- `results/shared-session-sanitizer-2026-07-16.json`.
