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

## Independent stream handles

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

## Latency

Weights are loaded before packet timing. Each distribution contains 200
continuous real neural packets after 20 warmup packets. PCM is delivered after
every packet.

| Packet | Min | p50 | p95 | p99 | Max | p50 RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 frame / 80 ms | 7.70 ms | 11.65 ms | 12.66 ms | 14.65 ms | 15.02 ms | 0.146 |
| 4 frames / 320 ms | 50.42 ms | 62.09 ms | 65.83 ms | 67.92 ms | 69.51 ms | 0.194 |

Startup measurements use separate processes:

| Startup event | End-to-end |
| --- | ---: |
| First 80 ms chunk without warmup | 157.11 ms |
| Explicit maximum-packet warmup | 207.73 ms one time |
| First 80 ms user chunk after warmup | 7.88 ms |

## Memory

| Allocation group | Bytes |
| --- | ---: |
| Device weights | 457,292,548 |
| Total device allocation | 492,327,468 |
| Runtime allocation excluding weights | 35,034,920 |
| Transformer KV allocation reported by prototype | 7,077,888 |
| Convolution histories reported by prototype | 848,832 |
| Codec input ring | 384 |
| Device PCM ring | 46,080 |
| Pinned host PCM ring | 46,080 |

The Spark exposes unified memory, so `nvidia-smi` reports device memory as
`N/A`. These values come from exact owned allocation sizes in the runtime, not
from process RSS or a heuristic.

## Sanitizer and Rust gates

The full BF16 load, startup warmup, neural pipeline, split streaming, short
final, reset, and finalization test was executed under:

```text
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99
```

Result:

```text
LEAK SUMMARY: 0 bytes leaked in 0 allocations
ERROR SUMMARY: 0 errors
```

Rust gates:

- two safetensors parser unit tests (F32 and BF16);
- one public-library integration test;
- one compiling Rustdoc usage example;
- `cargo clippy --all-targets -- -D warnings`;
- `cargo build --release --locked`.

## Claim boundaries

- Latency numbers are from the real neural decoder, never the deterministic
  fixture.
- The decoder does not measure talker/token-generation latency.
- The batch API is not a fused kernel.
- BF16 is the source artifact precision; execution weights are currently F32.
- No backend, frontend, production container, or network service was modified.
