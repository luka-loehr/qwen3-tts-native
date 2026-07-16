# Qwen3-TTS Native Research Playground

This standalone repository is the DGX Spark research environment for a native,
streaming implementation of Qwen3-TTS-12Hz-1.7B-VoiceDesign.

## Scope

- All compilation and execution happen on the DGX Spark.
- The Ephraim backend and frontend repositories remain untouched during research.
- The 0.6B model is out of scope.
- The production runtime must not depend on Python or Node.js.
- Model execution will use Rust for orchestration and native CUDA, cuBLAS, and
  TensorRT code behind a narrow C ABI.
- Research artifacts are promoted to the backend only after the quality,
  latency, memory, concurrency, cancellation, and streaming gates pass.

## Performance gates

- Warm time to first audio: p50 below 300 ms and p95 below 450 ms.
- End-to-end real-time factor: p50 below 0.50, with 0.45 as the engineering target.
- Four codec frames per audio packet: 320 ms or 15,360 bytes of mono 24 kHz s16 PCM.
- Inter-packet gap: p99 below 320 ms.
- Progressive delivery must be proven at the raw socket, not inferred from logs.
- At least 20 warmups and 200 measured requests per final candidate.
- Concurrency must be tested at 1, 3, and 6 requests.
- Seeded codec-token parity, exact sample counts, decoder continuity, multilingual
  intelligibility, speaker consistency, instruction adherence, memory, and energy
  are required quality gates.

## Current native baseline

The Rust checkpoint inspector validates the exact 1.7B VoiceDesign checkpoint:

- 404 BF16 tensors
- 1,916,676,352 parameters
- 3,833,352,704 bytes of tensor payload
- 3,483,101,184 bytes in the talker
- 350,251,520 bytes in the 15-step code predictor

The speech-tokenizer checkpoint contains 457,292,548 bytes of decoder weights and
224,937,216 bytes of encoder weights. The production synthesis path will omit the
unused encoder and initially convert the decoder to BF16.

The first SM 12.1 CUDA library provides a device probe, a BF16 vocabulary argmax
kernel, and exact-shape cuBLAS projection benchmarks. It contains native
`runtime.sm_121.cubin` and no PTX fallback.
