# Qwen3-TTS Native HTTP Server

This crate is the HTTP transport for the repository's public
`qwen3-tts-runtime` Rust API. It serves exactly the pinned Qwen3-TTS 1.7B
VoiceDesign model. It does not expose voice cloning, reference audio, speaker
enrolment, the Base model, or the retired 0.6B model.

## Surface

- `GET /health/live`
- `GET /health/ready`
- `GET /v1/capabilities`
- `GET /metrics`
- `POST /v1/voice-design/speech`
- `DELETE /v1/requests/{request_id}`
- `POST /v1/audio/speech` as a deliberately narrow buffered-WAV compatibility alias

## Startup readiness

The process constructs exactly one native engine before binding the listener.
Startup then runs one bounded, deterministic codec frame through the real
VoiceDesign talker, code predictor, device-to-device handoff, and neural codec.
The listener is not bound and readiness cannot become true until that warm-up
packet, end reason, retirement, and delivery metrics have all been validated.

The native endpoint defaults to progressive PCM streaming. `stream: false`
defaults to a fully buffered WAV response. Explicitly contradictory pairs such
as `stream: true` with `output_format: "wav"` are rejected.

All JSON objects reject unknown fields. A complete native request looks like
this:

```json
{
  "text": "Good morning. This is a native streaming test.",
  "voice_description": "A calm adult male voice with measured delivery and a warm low register.",
  "language": "english",
  "seed": 42,
  "max_duration_seconds": 30.0,
  "sampling": {
    "strategy": "sample",
    "temperature": 0.9,
    "top_p": 1.0,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "predictor": {
      "strategy": "sample",
      "temperature": 0.9,
      "top_p": 1.0,
      "top_k": 50
    }
  },
  "stream": true,
  "output_format": "pcm_s16le"
}
```

The caller may set `x-request-id` to a UUID before starting a request. This is
particularly important for buffered WAV because response headers arrive only
after synthesis. Duplicate active IDs return HTTP 409. Cancellation is
requested with `DELETE /v1/requests/{request_id}`. A request that fails to
retire becomes an explicit unhealthy tombstone: readiness fails, metrics keep
it visible, and later cancellation attempts return HTTP 503 instead of a
misleading acceptance.

Streaming uses `multipart/mixed`. A JSON start part is followed by binary
24 kHz mono signed-16-bit little-endian PCM parts as the native runtime emits
them, then exactly one JSON end or error part. Multipart boundaries, rather
than HTTP DATA-frame boundaries, define audio packets. Every audio part carries
`X-Sequence`, `X-First-Codec-Frame`, `X-First-Sample`, `X-Sample-Count`,
`X-Codec-Frames`, and `X-Final`. The server rejects gaps, overlaps, packets
after a final packet, and end-of-stream without explicit final audio.
The JSON end event reports `finish_reason: "stop"` for codec EOS and
`finish_reason: "length"` when `max_duration_seconds` is reached. Buffered WAV
responses expose the same value in `x-finish-reason`.

## Native integration

The executable requires these environment variables:

- `QWEN3_TTS_MODEL_ROOT`
- `QWEN3_TTS_TALKER_LIBRARY`
- `QWEN3_TTS_CODEC_LIBRARY`

Optional settings include `QWEN3_TTS_BIND` (default `127.0.0.1:8080`),
`QWEN3_TTS_DEVICE_INDEX`, `QWEN3_TTS_MAX_CONCURRENT_REQUESTS`,
`QWEN3_TTS_MAX_TEXT_BYTES`, `QWEN3_TTS_MAX_VOICE_DESCRIPTION_BYTES`, and
`QWEN3_TTS_MAX_DURATION_SECONDS`.

The server owns one long-lived `Scheduler<NativeBackend>`. Blocking native
polling runs outside Tokio's core workers. A one-packet HTTP channel preserves
backpressure: a slow or disconnected client stops polling and cancellation is
propagated to the native request.

SIGINT and SIGTERM close request admission atomically, cancel all admitted
requests, and start graceful HTTP shutdown. Cleanup budgets are validated to
fit inside the global deadline. An independent operating-system watchdog forces
exit code 124 if a blocking native or CUDA worker still prevents exit after the
deadline; failure to create that watchdog forces exit code 125 immediately.

## Audio and limits

- 24,000 Hz, mono, signed PCM16 little-endian
- 1,920 samples / 80 ms per codec frame
- first native packet: one frame
- later native packets: at most four frames / 320 ms
- intrinsic runtime ceiling: 8,192 frames / 655.36 seconds
- HTTP defaults: 32 KiB text, 4 KiB voice description, 120 seconds

Buffered PCM is bounded by the validated duration ceiling. A global semaphore
limits completed WAV bodies and remains held until the body reaches EOF or is
dropped, so slow clients cannot make response memory unbounded. A client that
never consumes or drops its response can deliberately occupy that bounded
capacity; production deployments must therefore enforce a TCP or reverse-proxy
response-write timeout as part of their edge configuration.

`auto`, Chinese, English, Japanese, Korean, German, French, Russian,
Portuguese, Spanish, and Italian are accepted. Turkish returns HTTP 422 because
the pinned VoiceDesign model has no explicit Turkish language ID.

## Scope boundary

TLS termination and concrete identity-provider integration belong in the
deployment proxy or a future Tower authentication layer. The server never logs
text or voice descriptions and does not add unbounded request queues.
