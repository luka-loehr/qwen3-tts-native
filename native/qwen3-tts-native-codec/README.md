# Qwen3-TTS native incremental codec research

This crate is an isolated Rust and CUDA prototype for the causal
Qwen3-TTS-Tokenizer-12Hz decoder used by the 1.7B VoiceDesign model.

The production target is four new codec frames per packet:

- 12.5 codec frames per second
- 1,920 waveform samples per codec frame
- 7,680 samples per full packet
- 15,360 bytes per full mono 24 kHz signed-16 PCM packet

The native context owns persistent CUDA allocations for the exact streaming
state geometry:

- a 72-frame, eight-layer transformer key/value ring;
- causal-convolution left histories;
- dilated residual histories;
- transposed-convolution overlap tails for strides 8, 5, 4, and 3;
- three-slot CUDA codec and PCM rings; and
- pinned host PCM staging slots.

## Research boundary

The initial deterministic fixture validates state transitions, packet
boundaries, sample counts, CUDA memory ownership, and the Rust/C ABI. It is not
the neural decoder and must never be presented as generated speech. Neural
weight loading and kernels become eligible only after fixture parity is exact.

No Python or Node.js runtime is used. Compilation and execution happen only in
the DGX Spark research environment. This subtree does not integrate with the
Ephraim backend, frontend, or production containers.

## ABI

The native library exposes a versioned C ABI with an opaque context handle,
fixed-width POD structures, integer status values, and caller-owned error
buffers. Only explicitly exported symbols have default visibility.
