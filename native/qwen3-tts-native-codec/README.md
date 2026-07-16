# Native Incremental Qwen3-TTS Codec Prototype

This crate isolates the state and packet ABI required by the Qwen3-TTS 1.7B
speech-tokenizer decoder on the DGX Spark. It is a native Rust/CUDA research
prototype. It contains no Python or Node.js runtime path.

## Implemented

- a versioned C ABI with an opaque per-stream context;
- a 72-frame, eight-layer BF16 transformer KV ring;
- one exact-sized BF16 arena for pre-convolution, ConvNeXt, residual-dilation,
  ConvTranspose-overlap, and final-convolution history;
- three-slot CUDA codec and PCM rings plus a pinned host PCM ring;
- persistent frame, sample, KV-head, ring-slot, and finalization state;
- deterministic CUDA fixture kernels with stateful 8x, 5x, 4x, and 3x overlap;
- an independent Rust full-stream reference;
- sample, seam, SNR, lifecycle, state-wrap, and latency validation.

One four-frame packet represents 320 ms of 24 kHz audio: 7,680 signed 16-bit
samples, or 15,360 bytes.

## Deliberate limitation

The deterministic fixture is not the neural speech decoder and does not produce
generated speech. It proves the ABI, persistent-state layout, packet accounting,
overlap behavior, and boundary invariance while full native tokenizer-decoder
kernels and weight loading are still unavailable. Fixture latency must never be
reported as Qwen3-TTS model latency or audio quality.

## Build

The CUDA target is compiled for the DGX Spark's SM 12.1 architecture.

```text
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --parallel
cargo build --release --locked
```

## Validate

```text
./target/release/qwen3-tts-native-codec \
  parity ./native/build/libqwen3_tts_codec_cuda.so

./target/release/qwen3-tts-native-codec \
  benchmark ./native/build/libqwen3_tts_codec_cuda.so 200
```

The parity command processes 83 deterministic frames through uneven packets,
crosses the 72-frame KV boundary, rotates all three packet slots, compares every
sample with the independent full-stream reference, checks packet seams, rejects
post-final input, and verifies reset behavior. The benchmark refuses fewer than
200 measured packets.

## ABI

The native library exposes a versioned C ABI with an opaque context handle,
fixed-width POD structures, integer status values, and caller-owned error
buffers. Only explicitly exported symbols have default visibility. A context is
owned by one stream and must not be called concurrently.

This subtree does not integrate with the Ephraim backend, frontend, or any
production container.
