# Native Streaming Architecture

## Runtime boundary

The production process will be a Rust service that owns HTTP streaming, request
queues, authentication hooks, cancellation, deterministic sampling state, and
backpressure. GPU execution is exposed through a versioned C ABI implemented in
CUDA/C++.

The native library will own:

- immutable BF16 weight arenas;
- per-request talker and predictor KV caches;
- the stateful speech-decoder cache;
- high-priority autoregressive and lower-priority vocoder CUDA streams;
- CUDA events between generation, decoder, and device-to-host transfer;
- CUDA graph instances for fixed decode steps;
- pinned PCM staging buffers.

## Exact model split

### Talker

- 28 decoder layers
- hidden size 2,048
- intermediate size 6,144
- 16 query heads and 8 KV heads
- head dimension 128
- per-head Q/K RMSNorm
- interleaved multimodal RoPE sections 24, 20, and 20
- codec vocabulary 3,072
- text vocabulary 151,936

### Code predictor

- 5 decoder layers
- hidden size 1,024
- intermediate size 3,072
- 16 query heads and 8 KV heads
- head dimension 64
- 15 distinct codec embeddings and LM heads
- codec vocabulary 2,048

Each talker token creates one two-position predictor prefill from the talker
hidden state and codebook-zero embedding, followed by 15 cached predictor decode
steps. The 16 codebook embeddings are summed to form the next talker input.

### Stateful speech decoder

- 16 residual codebooks
- 12.5 codec frames per second
- 1,920 waveform samples per frame
- 8 transformer layers at hidden size 512
- 16 query and 16 KV heads
- 72-frame sliding attention window
- causal pre-convolution with two retained frames
- two 2x causal transposed-convolution/ConvNeXt stages
- waveform upsampling stages 8x, 5x, 4x, and 3x

A request retains decoder KV state, causal convolution history, ConvNeXt history,
residual-unit dilation history, and transposed-convolution overlap tails. New
four-frame packets are decoded once; previously emitted prefixes are never
recomputed.

## Scheduling

1. Run talker prefill.
2. Generate one semantic code and its 15 residual codebooks on the high-priority
   autoregressive stream.
3. Every four frames, signal the lower-priority decoder stream.
4. Decode exactly those four new frames using persistent decoder state.
5. Copy 320 ms of PCM through a pinned ring buffer.
6. Release the network chunk as soon as the copy event completes.
7. Honor cancellation between every frame and packet without leaking caches.

## Cold-start policy

cuBLAS and every CUDA graph are warmed during process readiness. Per-request
latency must never include library initialization, engine construction, tactic
selection, or graph capture.
