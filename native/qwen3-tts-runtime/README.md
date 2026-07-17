# Qwen3-TTS Native Runtime

This crate owns the public request lifecycle and connects the real native
Qwen3-TTS 1.7B VoiceDesign talker to the real neural speech-tokenizer decoder.
It exposes progressive 24 kHz mono signed 16-bit PCM through a versioned C ABI.

There is no placeholder inference path in the exported library. Test-only
scripted backends exercise scheduler failure and backpressure behavior without
being reachable through the public engine entry points.

## Implemented runtime

- Real `NativeBackend` with one shared talker model and one shared codec model.
- Independently owned talker/codec sessions per request.
- Bounded request concurrency and PCM ring buffers.
- Adjacent request prefill coalescing through `start_batch`.
- Concurrent session stepping through `step_batch`.
- One-frame first packet and up to four frames in subsequent packets.
- Exact contiguous frame, sample, and packet accounting.
- Explicit backpressure without dropping or overwriting audio.
- Cancellation, terminal-state enforcement, and deterministic retirement.
- Per-request queue, prefill, TTFA, wall, GPU-time, output, and memory metrics.
- Panic containment and typed status mapping at every exported FFI boundary.

## Public C ABI

The complete contract is declared in
[`include/qwen3_tts_runtime.h`](include/qwen3_tts_runtime.h). The exported
surface contains exactly:

- `qwen3_tts_runtime_abi_version_v1`;
- `qwen3_tts_engine_create_v1` / `qwen3_tts_engine_destroy_v1`;
- `qwen3_tts_request_start_v1`;
- `qwen3_tts_request_poll_v1`;
- `qwen3_tts_request_cancel_v1`;
- `qwen3_tts_request_metrics_v1`;
- `qwen3_tts_request_finish_reason_v1`;
- `qwen3_tts_request_destroy_v1`.

Callers provide versioned structures, PCM storage, and error buffers. A poll
preflights the complete packet capacity before touching the queue. An
undersized buffer therefore returns an error without consuming the packet.

The library distinguishes successful delivery, would-block, end-of-stream,
invalid input, invalid UTF-8, unsupported language, model, allocation, CUDA,
state, cancellation, and internal failures.

`qwen3_tts_request_start_v1` returns `QWEN3_TTS_RUNTIME_WOULD_BLOCK` and leaves
the output handle null when all configured request slots are occupied. After a
cancelled request has been destroyed successfully, its slot is retired and may
be reused immediately.

## Library layout

The runtime loads these native components:

- `libqwen3_tts_cuda.so`;
- `libqwen3_tts_codec_cuda.so`.

By default they are resolved beside `libqwen3_tts_runtime.so` using `dladdr`.
Set `QWEN3_TTS_LIBRARY_DIR` to an explicit directory when the three libraries
are not co-located.

The engine receives a model-root path containing the prepared VoiceDesign and
speech-tokenizer files. Weights are not embedded in the runtime library.

## Ownership and threading

`qwen3_tts_request_start_v1` copies the UTF-8 text and instruction. The caller
may release those input buffers when the function returns.

An active request retains the engine core, so it remains valid if the caller
destroys the public engine handle first. Destroying a request cancels it when
necessary and waits for bounded retirement. One thread may poll a request while
another requests cancellation. A request handle must not be polled concurrently
by multiple threads.

The engine owns shared immutable weights. Each active request owns its mutable
CUDA streams, cuBLAS handles, KV caches, decoder state, RNG state, cursors, and
PCM slots. Request teardown releases those allocations exactly once.

## Packet contract

Each codec frame represents 1,920 samples, or 80 ms at 24 kHz. The default
follow-up packet contains four frames (7,680 samples, 320 ms). A short final
packet is allowed.

For every delivered packet:

- `sequence` increases by one;
- `first_codec_frame` equals all previously delivered frames;
- `first_sample` equals `first_codec_frame * 1920`;
- `sample_count` equals `codec_frames * 1920`;
- caller storage after `sample_count` remains untouched;
- the final packet is followed by end-of-stream;
- `qwen3_tts_request_finish_reason_v1` returns `CODEC_EOS` for a natural model
  stop and `MAX_CODEC_FRAMES` when the configured safety ceiling truncates
  generation. It returns `NONE` before a terminal packet is produced.

## Verification

The Rust suite covers scheduler bounds, ABI layouts, failure mapping, panic
containment, backpressure, cancellation, retirement, start batching, and exact
delivery metrics. The strict C harness additionally covers malformed structure
sizes, malformed UTF-8, cancellation/destruction, undersized-buffer preflight,
engine destruction before a live request, packet continuity, and WAV output.

The public concurrency harness runs 24 warmups and 200 measured four-frame
requests at each of B1, B3, and B6. The qualifying run completed all 600
requests without a failure:

| Concurrency | TTFA p95 | Request RTF p50 | Aggregate RTF |
| ---: | ---: | ---: | ---: |
| 1 | 78.24 ms | 0.765 | 0.767 |
| 3 | 186.42 ms | 1.800 | 0.601 |
| 6 | 364.62 ms | 3.557 | 0.594 |

The complete evidence is stored in
[`../../benchmarks/results/native-runtime-public-c-abi-qualification.json`](../../benchmarks/results/native-runtime-public-c-abi-qualification.json).
Aggregate throughput is faster than real time at every tested level. B3/B6
per-request RTF and the stricter B1 RTF target below 0.50 remain open.

This fixed 320 ms harness measures scheduler throughput and packet delivery. It
does not claim that the model reached a natural end-of-sequence. The separate
`c_abi_endurance.c` harness is the release gate for complete requests. It runs
three full warmups followed by 200 full measured requests at B1, uses 512 codec
frames only as an emergency guard, and fails immediately unless every request
ends with `QWEN3_TTS_FINISH_REASON_CODEC_EOS`. It also validates packet
continuity, caller-buffer boundaries, terminal delivery, and exact metrics.

Build the release library first, then compile and run the endurance consumer:

```text
cargo build --release --locked --manifest-path native/qwen3-tts-runtime/Cargo.toml

cc -std=c11 -O3 -Wall -Wextra -Wpedantic -Werror \
  -I native/qwen3-tts-runtime/include \
  native/qwen3-tts-runtime/tests/c_abi_endurance.c \
  -L native/qwen3-tts-runtime/target/release \
  -Wl,-rpath,"$PWD/native/qwen3-tts-runtime/target/release" \
  -lqwen3_tts_runtime -lpthread -ldl \
  -o native/qwen3-tts-runtime/target/c_abi_endurance

QWEN3_TTS_LIBRARY_DIR=/path/to/native-libraries \
  native/qwen3-tts-runtime/target/c_abi_endurance \
  /path/to/qwen3-tts-1.7b-voice-design-bf16-indexed \
  /tmp/native-runtime-natural-eos-endurance.json
```

The output JSON contains all 200 request records plus aggregate TTFA, RTF,
memory, completion, and finish-reason evidence. A run is qualifying only when
`completed_requests`, `codec_eos_requests`, and `measured_requests` are all 200,
`failed_requests` and `max_codec_frames_requests` are zero, and
`all_packet_metric_and_finish_reason_invariants_passed` is true.

## Direct native smoke

`native_e2e_smoke` bypasses the scheduler and connects one incremental talker
session directly to one incremental decoder session. It is useful for model and
packet diagnosis:

```text
cargo run --release --locked --bin native_e2e_smoke -- \
  /path/to/libqwen3_tts_cuda.so \
  /path/to/libqwen3_tts_codec_cuda.so \
  /path/to/qwen3-tts-1.7b-voice-design-bf16-indexed \
  /tmp/native-e2e.wav \
  --text "Guten Morgen." \
  --instruction "A calm adult male voice with natural articulation." \
  --language German \
  --max-frames 256 \
  --packet-frames 4 \
  --seed 0 \
  --greedy \
  --report /tmp/native-e2e.json
```

`--max-frames` is an emergency ceiling, not a target duration. A low value can
cut speech mid-word. Quality runs must allow natural model end-of-sequence and
must treat a request that reaches the ceiling as truncated.

## Current boundaries

- This crate is a library, not an HTTP or gRPC service.
- It is not connected to the Ephraim backend, frontend, or production TTS
  container.
- There is no production runtime image yet.
- Talker generation and codec decoding still execute sequentially within one
  packet; overlapping those stages is the next performance milestone.
- The checked functional WAV proves transport and PCM validity, not complete
  multilingual intelligibility or voice-quality qualification.
