# Native Qwen3-TTS qualification harness

This crate drives the versioned native runtime ABI. It does not contain a mock
model, a synthetic decoder, Python, Node.js, or an HTTP proxy. A run cannot
start unless the shared library exports every real engine/request symbol.
This includes the additive finish-reason query: every completed qualification
request must report `CODEC_EOS`. Reaching `max_codec_frames` is truncation and
fails the run rather than being counted as a completion.

Before warm-up, it fills the configured request capacity, requires the next
start to return `WOULD_BLOCK`, cancels and destroys every live request, and
proves that the released capacity can immediately accept a new request.

The qualifying suite completes at least 200 requests at each configured
concurrency, defaults to `1,3,6`, polls audio progressively, poisons the unused
tail of every caller buffer, and fails if the runtime writes beyond the exact
packet sample count. It also verifies contiguous request, packet, codec-frame,
and sample positions and compares observed totals with runtime metrics. Report
schema version 3 records `run_kind`, `finish_reason` for every request,
natural-EOS counts for every scenario, and explicit multilingual coverage.

```bash
cargo run --release -- suite \
  --library build/libqwen3_tts_runtime.so \
  --model-root /models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --corpus ../../benchmarks/corpora/qwen3-tts-voice-design.jsonl \
  --output ../../benchmarks/results/native-qualification.json \
  --audio-dir ../../benchmarks/results/native-qualification-audio
```

The independent multilingual release gate runs every corpus entry exactly once
at B1 and saves every listening artifact:

```bash
cargo run --release -- multilingual \
  --library build/libqwen3_tts_runtime.so \
  --model-root /models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --corpus ../../benchmarks/corpora/qwen3-tts-voice-design.jsonl \
  --output ../../benchmarks/results/native-multilingual.json \
  --audio-dir ../../benchmarks/results/native-multilingual-audio
```

`multilingual` passes only when all ten explicitly supported languages plus
`Auto` are observed, every corpus identifier runs exactly once, all requests
reach natural codec EOS, progressive delivery and copy bounds hold, and TTFA
and RTF remain inside their gates. It does not pretend to be the separate
200-request concurrency suite.

The suite gate requires all of the following:

- 200 completed requests for every concurrency with zero failures;
- natural codec EOS for every completed request, with zero length truncations;
- progressive multi-packet output for at least 95 percent of requests;
- exact caller-buffer copy bounds and contiguous packet positions;
- bounded backpressure, cancellation, destruction, and capacity recovery;
- aggregate and p95 per-request real-time factors below `1.0`;
- p95 caller-observed time to first audio below `200 ms`.

`smoke` permits a short non-qualifying run while developing. Its JSON report
always sets `qualifying_run` and the final qualification gate to `false`.
Saved WAV files are listening artifacts, not an automated quality claim.
