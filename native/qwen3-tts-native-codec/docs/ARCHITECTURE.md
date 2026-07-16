# Architecture and Integration Contract

## Scope

The library implements only the Qwen3-TTS 12 Hz speech-tokenizer decoder. Its
input is a sequence of discrete speech-token frames produced by the talker and
code predictor. Its output is mono 24 kHz signed 16-bit PCM.

The native boundary is intentionally narrow:

```text
Rust caller
  Arc<NativeCodecModel> (shared immutable weights)
        |
        +--> NativeCodecSession A (stream, cuBLAS, KV, histories, rings)
        +--> NativeCodecSession B (stream, cuBLAS, KV, histories, rings)
        +--> NativeCodecSession N (stream, cuBLAS, KV, histories, rings)
                    |
                    v
       [frame][16] u16, 1-4 frames, final flag
                    |
                    v
       CUDA RVQ -> transformer -> upsampler -> waveform decoder
                    |
                    v
       exactly frame_count * 1920 mono s16 samples
```

There is no Python, Node.js, HTTP, JSON, or backend-specific type in the
execution path.

## Tensor and packet contract

- Input layout: frame-major `[frame_count][16]`.
- Input scalar: unsigned 16-bit integer.
- Accepted packet size: one through four frames.
- Frame rate: 12.5 frames per second; one frame represents 80 ms.
- Output rate: 24,000 samples per second.
- Output layout: contiguous mono signed 16-bit PCM.
- Output length: exactly 1,920 samples per input frame.
- Finalization: a separate integer flag, never encoded into the frame tensor.
- Backpressure surface: three packet slots per state handle.

The runtime writes only the valid output prefix. Passing a larger output
capacity does not cause stale slot data to be copied. This behavior is verified
by pre-filling the unused suffix with a sentinel and checking that it remains
unchanged after a one-frame final packet.

## Persistent state

Each opaque session owns the following causal state:

- two prior RVQ/pre-convolution positions;
- eight layers of K/V data in a 72-frame sliding ring;
- six prior positions for each of two ConvNeXt stages;
- six input positions for the waveform pre-convolution;
- overlap tails for transposed-convolution strides 8, 5, 4, and 3;
- residual-convolution histories for dilations 1, 3, and 9 in all four stages;
- six positions for the final waveform convolution;
- absolute frame and sample positions;
- current KV and three-slot packet-ring indices;
- finalization state.

No prior frame is passed back by the caller and no prefix is recomputed. Reset
zeros every history and restores all counters. A reset replay is bit-exact.

## Weight ownership

The loader accepts exactly 271 canonical `decoder.*` tensors. It validates
rank, shape products, byte length, dtype, required names, duplicate names, and
the final tensor count.

The Rust library exposes this boundary as the object-safe
`DecoderWeightProvider` trait. The built-in `DecoderWeights` safetensors reader
implements it, and a separate mmap/indexed artifact type can implement the same
trait without routing through the built-in reader.

F32 sources are copied directly. BF16 sources are copied through one reusable
8 MiB device staging buffer and converted to F32 by a CUDA kernel. Conversion
is stream ordered, works for tensors larger than the staging buffer, and does
not rely on safetensors file order. The execution allocation owns the resulting
weights after the provider callback returns.

The primary API stores those allocations in one `Qwen3TtsCodecModelV1`.
Sessions retain that model through a native atomic reference count and only
read its weight map. Dropping one session never frees or mutates weights.
Weights are released after both the public model handle and all retained
sessions are gone. Model loading and one-time warmup use a short lifecycle
mutex; inference never takes that mutex or any global lock.

The original context API remains exported for ABI compatibility. A legacy
context receives a private model internally, so its observable behavior and
memory report are unchanged. New integrations should use the shared model and
owned session APIs.

Current precision contract:

| Layer | Source | Execution |
| --- | --- | --- |
| Indexed artifact | BF16 | n/a |
| Native model weights | BF16 or F32 | F32 |
| Activations and causal histories | n/a | F32 |
| PCM | n/a | signed 16-bit |

F32 execution is deliberate because that path has official activation and PCM
parity. A future tensor-core BF16 execution path must receive its own oracle,
quality, latency, and memory evidence and must not silently reuse these claims.

## Streaming lifecycle

1. Load `NativeCodecLibrary` into an `Arc`.
2. Call `load_shared_model` once. It uploads all 271 tensors and performs the
   one-time model warmup before returning.
3. Call `Arc<NativeCodecModel>::start_session` for every active request.
4. Submit one to four frames through the owned session.
5. Deliver the returned PCM immediately.
6. Submit subsequent packets on the same session.
7. Set `is_final=1` only on the final packet.
8. Drop the session, cancel and drop it, or reset it before explicit reuse.

Shared-model warmup executes one maximum-size packet on a temporary internal
session, discards the result, destroys that session, and marks the model warm.
Public session creation is rejected until loading and warmup have completed.
The legacy context warmup contract remains unchanged.

## Multi-stream API

One shared model can back any number of independent sessions within available
memory. The validated runtime buckets are B=1, B=3, and B=6. Every session owns
a non-blocking CUDA stream, cuBLAS handle, events, KV state, convolution
histories, packet rings, workspace, and pinned host output ring.

`NativeCodecModel` is `Send + Sync`. `NativeCodecSession` is `Send + 'static`
but intentionally not `Sync`; mutable calls require exclusive access. Scoped
Rust workers can therefore process distinct sessions simultaneously with no
global inference lock. CUDA kernels may overlap when device resources permit.
Correctness never depends on overlap.

`process_batch_v1` and `session_process_batch_v1` remain array-order C ABI
dispatchers. Inputs may contain different frame counts and final flags. They
provide a convenient non-concurrent surface but are not fused CUDA batches.

The current implementation dispatches items in array order. It provides a
stable integration surface and verified state isolation, but it is not a fused
CUDA batch and should not be described as one. Graph capture or fused batching
may be added only for fixed `(batch, frame_count)` buckets; valid sample counts
must still trim every output independently.

## Error and ownership rules

- Status zero means success; negative values identify argument, CUDA, state,
  allocation, or model failures.
- Caller-owned error buffers receive a bounded null-terminated message.
- A session belongs to one stream and is not concurrently callable.
- The caller retains input/output allocation ownership.
- The shared model owns immutable device weights only.
- A session owns state, packet rings, workspace, events, stream, cuBLAS handle,
  and pinned host ring.
- Cancel synchronizes that session's stream and rejects further packets until
  reset or drop. It does not affect sibling sessions.
- Session destroy synchronizes only its stream, releases its state, and drops
  one model reference.

Exact persistent memory on the validated build is:

```text
shared device bytes = 457,292,548
per-session device bytes = 35,034,920
per-session pinned host bytes = 46,080
total device bytes(B) = 457,292,548 + B * 35,034,920
```

## Integration boundary

The intended backend adapter should translate its internal packet object into
the ABI without modifying frame order, tool schemas, prompts, or frontend
protocols. This playground contains no backend adapter and makes no network
calls. Promotion should happen only after the separate talker/code-predictor
runtime supplies real frames and end-to-end audio quality is evaluated.
