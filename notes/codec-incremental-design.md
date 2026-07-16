# Incremental Speech-Decoder State and Fixture Validation

## Scope

This note records the isolated native codec work for Qwen3-TTS 1.7B on the DGX
Spark. The implementation lives in `native/qwen3-tts-native-codec`. It does not
change the Ephraim backend, frontend, API, or production containers.

The current executable path is a deterministic CUDA fixture. It validates the
incremental contract and exact state capacity without claiming to execute the
speech-tokenizer decoder's neural weights.

## Packet geometry

- codec rate: 12.5 frames per second;
- codebooks: 16;
- packet size: four frames;
- packet duration: 320 ms;
- waveform rate: 24 kHz;
- samples per frame: 1,920;
- samples per packet: 7,680;
- PCM bytes per packet: 15,360 at signed 16-bit precision.

The decoder emits only samples finalized by the current packet. A
ConvTranspose1d layer with stride `s` and kernel width `2s` retains the final
`s` raw contribution positions. On the next packet those positions are added to
the first `s` new positions. Exactly `n * s` output positions are then final for
`n` new inputs, and the new `s`-position tail is retained. At end of stream the
remaining right tail is discarded, matching causal right trimming; it is never
emitted as extra audio.

## Persistent state

The transformer ring contains K and V for eight layers, 16 KV heads, a
72-frame window, and a head dimension of 64:

```text
2 * 8 * 16 * 72 * 64 = 1,179,648 BF16 values
1,179,648 * 2 bytes = 2,359,296 bytes
```

An earlier hand estimate of 4,718,592 bytes was wrong because it applied the
BF16 width twice. The compile-time assertions and runtime state report use the
correct value.

The contiguous BF16 convolution-history arena has this layout:

| Segment | BF16 values | Bytes |
| --- | ---: | ---: |
| Causal pre-convolution, 512 channels x 2 | 1,024 | 2,048 |
| Two ConvNeXt histories, 1,024 channels x 6 | 12,288 | 24,576 |
| Decoder input convolution, 1,024 channels x 6 | 6,144 | 12,288 |
| ConvTranspose tails for 768x8, 384x5, 192x4, 96x3 | 9,120 | 18,240 |
| Residual histories at dilations 1, 3, and 9 | 112,320 | 224,640 |
| Final convolution, 96 channels x 6 | 576 | 1,152 |
| **Total** | **141,472** | **282,944** |

Additional persistent or reusable device storage is:

| Allocation | Bytes |
| --- | ---: |
| Three-slot codec ring | 384 |
| Three-slot PCM ring | 46,080 |
| Deterministic fixture history | 376 |
| Reusable integer fixture scratch | 44,144 |
| **Device total including KV and convolution history** | **2,733,224** |
| Pinned host PCM ring | 46,080 |

All allocations are created once per context. Packet processing performs no
device or pinned-host allocation.

## Native ABI and lifecycle

The library exposes six versioned C functions: ABI version, create, destroy,
reset, state information, and deterministic packet processing. The ABI uses an
opaque context, fixed-width POD types, caller-owned error buffers, and signed
status codes. Hidden visibility prevents C++ or CUDA implementation symbols from
becoming part of the contract.

Each context owns a nonblocking CUDA stream, two timing events, all device state,
and its pinned ring. It tracks the absolute frame and sample positions, the KV
ring head, the next packet slot, and whether the stream was finalized. A packet
after finalization is rejected until reset. Context calls are single-owner and
not thread-safe.

## Deterministic fixture

The fixture deliberately avoids model-quality claims. It performs these native
operations:

1. copy one to four 16-codebook frames into the current CUDA ring slot;
2. derive a deterministic causal frame value using two retained history values;
3. update only the new positions in the exact-sized 72-frame BF16 KV ring;
4. apply two 2x expansions and stateful 8x, 5x, 4x, and 3x overlap stages;
5. clamp finalized samples into the CUDA PCM ring;
6. copy the packet through pinned memory and return it to the caller;
7. advance absolute positions and rotate the three-slot ring.

The Rust reference implements the same integer definition independently and
processes all 83 frames as one uninterrupted stream. The CUDA path uses uneven
packet sizes `1, 4, 2, 3, 4, 1, 3, 2` repeatedly. This crosses the 72-frame
window and creates 34 packet boundaries.

## Validation result

The final parity run produced:

- 159,360 expected and actual samples;
- zero maximum absolute sample error;
- zero mean squared error;
- infinite SNR, satisfying the 50 dB minimum;
- zero seam sample error;
- zero boundary-delta error;
- correct absolute positions, KV head, and rotating slots;
- correct post-final rejection and reset behavior.

The final 200-packet continuous benchmark used 20 warmup packets, no resets
between measured packets, and four frames per packet:

| Metric | GPU | End to end |
| --- | ---: | ---: |
| Minimum | 26.88 us | 32.00 us |
| p50 | 31.23 us | 36.58 us |
| p95 | 33.50 us | 38.75 us |
| p99 | 39.33 us | 44.69 us |
| Maximum | 39.74 us | 45.01 us |

These timings cover the deterministic state fixture only. They exclude neural
weight execution and must not be used as a Qwen3-TTS RTF or first-audio result.

## Remaining neural work

The separate artifact-loader milestone now supplies immutable, shape-validated
decoder weights, a BF16 offline conversion, bounded CUDA upload, and explicit
device ownership. Before the codec path can replace the existing tokenizer
decoder, it still needs:

1. the real RVQ projection, causal convolutions, transformer, ConvNeXt blocks,
   residual units, and output convolution;
2. native BF16/FP32 numerical parity against a trusted full-prefix decoder;
3. real-audio seam validation with at least 50 dB SNR where deterministic
   numerical parity is expected;
4. CUDA graph capture and packet-latency measurement for the complete decoder;
5. cancellation, multi-request contention, and memory-pressure tests;
6. backend integration only after the native decoder passes those gates.

Until then, the correct claim is that the persistent-state ABI, memory layout,
packet accounting, overlap semantics, and deterministic parity harness are
implemented and verified.
