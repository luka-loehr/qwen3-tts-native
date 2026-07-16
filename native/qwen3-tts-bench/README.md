# Native Qwen3-TTS qualification harness

This crate drives the versioned native runtime ABI. It does not contain a mock
model, a synthetic decoder, Python, Node.js, or an HTTP proxy. A run cannot
start unless the shared library exports every real engine/request symbol.

Before warm-up, it fills the configured request capacity, requires the next
start to return `WOULD_BLOCK`, cancels and destroys every live request, and
proves that the released capacity can immediately accept a new request.

The qualifying suite completes at least 200 requests at each configured
concurrency, defaults to `1,3,6`, polls audio progressively, poisons the unused
tail of every caller buffer, and fails if the runtime writes beyond the exact
packet sample count. It also verifies contiguous request, packet, codec-frame,
and sample positions and compares observed totals with runtime metrics.

```bash
cargo run --release -- suite \
  --library build/libqwen3_tts_runtime.so \
  --model-root /models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --corpus ../../benchmarks/corpora/qwen3-tts-voice-design.jsonl \
  --output ../../benchmarks/results/native-qualification.json \
  --audio-dir ../../benchmarks/results/native-qualification-audio
```

The suite gate requires all of the following:

- 200 completed requests for every concurrency with zero failures;
- progressive multi-packet output for at least 95 percent of requests;
- exact caller-buffer copy bounds and contiguous packet positions;
- bounded backpressure, cancellation, destruction, and capacity recovery;
- aggregate and p95 per-request real-time factors below `1.0`;
- p95 caller-observed time to first audio below `200 ms`.

`smoke` permits a short non-qualifying run while developing. Its JSON report
always sets `qualifying_run` and the final qualification gate to `false`.
Saved WAV files are listening artifacts, not an automated quality claim.
