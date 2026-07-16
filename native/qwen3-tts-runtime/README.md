# Qwen3-TTS Native Runtime Contract

This crate owns the public request lifecycle and packet invariants for the
native Qwen3-TTS 1.7B VoiceDesign engine.

It currently provides:

- a versioned C ABI contract with caller-owned PCM output buffers;
- the official ten-language identifier set plus `Auto`;
- official talker and predictor sampling defaults;
- a bounded packet queue for explicit backpressure;
- terminal request-state enforcement and cancellation rules;
- contiguous frame, sample, and packet accounting;
- per-request TTFA, GPU-time, memory, and output metrics.

The crate intentionally does **not** provide placeholder or fixture neural
inference. Engine entry points declared in the C header are connected only after
the real talker/predictor and speech-tokenizer decoder pass reference parity.

## Ownership

`qwen3_tts_request_start_v1` copies UTF-8 text and instruction data. The engine
owns immutable weights and request caches. `qwen3_tts_request_poll_v1` copies
mono 24 kHz signed 16-bit PCM into a caller-owned buffer; internal pinned ring
buffers are never exposed. One thread polls a request while cancellation may be
requested from another thread.

## Packet contract

The default packet contains four codec frames, or 7,680 samples (320 ms). A
short final packet is allowed. Frame indices, sample offsets, and packet sequence
numbers must remain contiguous. Backpressure is signaled rather than dropping or
overwriting audio.
