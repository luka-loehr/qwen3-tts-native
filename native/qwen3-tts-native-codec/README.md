# Native Qwen3-TTS Speech-Tokenizer Decoder

This directory contains a real, incremental implementation of the
Qwen3-TTS 12 Hz speech-tokenizer decoder for the NVIDIA DGX Spark. Rust owns
artifact parsing and the host ABI. CUDA and cuBLAS execute the neural decoder.
The runtime does not invoke Python or Node.js.

The implementation accepts one to four frame-major speech-token packets
(`[frame][16]`, unsigned 16-bit), retains all causal state, and returns exactly
1,920 mono 24 kHz signed 16-bit samples per frame. Audio is copied to the caller
after every packet; it is not buffered until the stream is complete.

## Verified result

All figures below were measured on the DGX Spark with the indexed decoder-only
BF16 artifact (SHA-256
`062caa0a31346422410e4c0d2494aec14be20553f8cb0b71a875329de99ce180`).
Each steady-state latency distribution follows 20 warmup packets and contains
200 real neural measurements.
The machine-readable report is stored in
[`../../benchmarks/results/native-codec-decoder-bf16.json`](../../benchmarks/results/native-codec-decoder-bf16.json).

| Measurement | Result |
| --- | ---: |
| First 80 ms chunk without startup warmup | 157.11 ms |
| Explicit one-time startup warmup | 207.73 ms |
| First 80 ms chunk after startup warmup | 7.88 ms |
| 80 ms packets, end-to-end p50 / p95 / p99 | 11.65 / 12.66 / 14.65 ms |
| 80 ms packet p50 real-time factor | 0.146 (6.87x real time) |
| 320 ms packets, end-to-end p50 / p95 / p99 | 62.09 / 65.83 / 67.92 ms |
| 320 ms packet p50 real-time factor | 0.194 (5.16x real time) |
| Official-oracle PCM error | at most 1 signed 16-bit LSB |
| Decoder tensor payload in the BF16 artifact | 228,646,274 bytes |
| F32 execution weights on the GPU | 457,292,548 bytes |
| Total reported device allocation per state handle | 492,327,468 bytes |
| Pinned host PCM ring | 46,080 bytes |

The BF16 artifact is expanded to F32 on the GPU with a bounded 8 MiB staging
buffer. This preserves the already validated F32 execution kernels and keeps
the on-disk/mmap artifact small, but it is **not** a BF16 compute path. The
source and device byte counts are reported separately by the API.

## Neural pipeline

1. Split residual vector quantizer with 16 codebooks.
2. Stateful causal pre-convolution.
3. Eight transformer layers with RoPE and a persistent 72-frame sliding KV ring.
4. Two stateful 2x latent upsampling and ConvNeXt stages.
5. Causal waveform pre-convolution.
6. Four SnakeBeta, transposed-convolution, and residual stacks with strides
   8, 5, 4, and 3.
7. Final SnakeBeta, causal convolution, clamp, and signed 16-bit PCM conversion.

All pre-convolution windows, ConvNeXt windows, residual-dilation windows,
transposed-convolution overlap tails, the final convolution window, frame
positions, sample positions, and ring-slot positions persist in the opaque
state handle. Prefix audio is never recomputed.

## Public C ABI

The versioned ABI is declared in
[`native/include/qwen3_tts_codec.h`](native/include/qwen3_tts_codec.h).

| Entry point | Purpose |
| --- | --- |
| `qwen3_tts_codec_create_v1` / `destroy_v1` | Own one independent stream state. |
| `qwen3_tts_codec_load_model_v1` | Load 271 canonical `decoder.*` tensors from F32 or BF16 source data. |
| `qwen3_tts_codec_warmup_v1` | Initialize CUDA/cuBLAS before user traffic and restore fresh state. |
| `qwen3_tts_codec_process_packet_v1` | Decode 1-4 frames and return exactly `frames * 1920` samples. |
| `qwen3_tts_codec_process_batch_v1` | Dispatch 1-6 independent state handles in array order. |
| `qwen3_tts_codec_reset_v1` | Clear every causal state component for deterministic replay. |
| `qwen3_tts_codec_state_info_v1` | Report positions, ring indices, and owned memory. |
| `qwen3_tts_codec_model_info_v1` | Report source/device bytes, tensor counts, and source dtypes. |

`is_final` is metadata separate from the frame tensor. Once a final packet is
accepted, further packets are rejected until reset. The caller owns input and
output buffers. A context must not be called concurrently. The current batch
entry point proves isolation and gives an integration-stable API, but it is an
array-order reference dispatcher rather than a fused batch kernel.

## Build on the Spark

The pinned build targets CUDA 13 and SM 12.1.

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace" -w /workspace \
  nvcr.io/nvidia/tensorrt:25.11-py3 \
  bash -lc 'cmake -S native -B build/native \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121 && \
    cmake --build build/native -j2'

docker run --rm \
  -v "$PWD:/workspace" -w /workspace \
  codex/qwen3-tts-rust-builder:1.97.0 \
  sh -c '/usr/local/cargo/bin/cargo build --release --locked'
```

## Reproduce the principal gates

Set these paths for the isolated Spark checkout:

```bash
LIB=build/native/libqwen3_tts_codec_cuda.so
MODEL=/home/administrator/codex-playground-artifacts/\
qwen3-tts-1.7b-voice-design-bf16-indexed/speech_tokenizer/model.safetensors
FIXTURE=../../benchmarks/fixtures/decoder-reference-bf16
BIN=target/release/qwen3-tts-native-codec

$BIN neural-parity "$LIB" "$MODEL" "$FIXTURE"
$BIN decoder-parity "$LIB" "$MODEL" "$FIXTURE"
$BIN batch-parity "$LIB" "$MODEL" "$FIXTURE"
$BIN neural-cold-start "$LIB" "$MODEL"
$BIN neural-benchmark "$LIB" "$MODEL" 200
```

The checked gates cover official intermediate activations, official PCM,
single-packet versus 1+3 streaming, four one-frame packets, short final output,
stale-tail poisoning, finalization, reset replay, 72-frame KV wrap, three-slot
overwrite, B=3 reset/replay, B=6 mixed final lengths, and cross-request leakage.
NVIDIA Compute Sanitizer `memcheck --leak-check full` reports zero errors and
zero leaked bytes on the complete BF16 neural-parity run.

See [`docs/USAGE.md`](docs/USAGE.md) for Rust-library, neural CLI, batch, and C
examples; [`docs/VERIFICATION.md`](docs/VERIFICATION.md) for exact evidence;
and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for state and integration
rules.

## Scope boundary

This is the speech-tokenizer **decoder**, not the 1.7B text/talker model. It
cannot turn text or a voice description into codec frames by itself. The talker
and code-predictor runtime must provide correctly ordered `[frame][16]` tokens.
This research branch does not modify or connect to the Ephraim backend,
frontend, or production containers.

## Primary references

- [Official Qwen3-TTS repository](https://github.com/QwenLM/Qwen3-TTS)
- [Official 1.7B VoiceDesign model](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
- [Official speech-tokenizer files](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/tree/main/speech_tokenizer)
- [NVIDIA cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/contents.html)
- [NVIDIA Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
