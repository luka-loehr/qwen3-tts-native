# HTTP API

The service exposes one native VoiceDesign endpoint, one deliberately narrow
compatibility endpoint, request cancellation, health, capabilities, and
metrics. There is no voice-clone, reference-audio, Base-model, CustomVoice, or
0.6B API.

All examples use `http://127.0.0.1:8080`. Put authentication and TLS at a
trusted proxy if the service is reachable beyond the local host.

## Endpoint summary

| Method | Path | Success type |
| --- | --- | --- |
| `GET` | `/health/live` | `application/json` |
| `GET` | `/health/ready` | `application/json` |
| `GET` | `/v1/capabilities` | `application/json` |
| `POST` | `/v1/voice-design/speech` | `multipart/mixed` or `audio/wav` |
| `POST` | `/v1/audio/speech` | `audio/wav` |
| `DELETE` | `/v1/requests/{request_id}` | `application/json` |
| `GET` | `/metrics` | `text/plain; version=0.0.4; charset=utf-8` |

The machine-readable contract is [openapi.yaml](openapi.yaml).

## Request identity

Both synthesis endpoints accept an optional `x-request-id` request header.
Its value must be an ASCII UUID. If omitted, the server generates a UUIDv7.
The chosen UUID is returned as `x-request-id` on successful synthesis
responses and on error responses that occur after the ID has been admitted.

An ID can identify only one active request. Reusing an active ID returns HTTP
409 with `code: "request_id_in_use"`. IDs are not persisted and are reusable
after the earlier request has fully retired.

## Native VoiceDesign synthesis

### `POST /v1/voice-design/speech`

The native endpoint accepts `application/json`. Unknown fields are rejected.

### Request body

| Field | Type | Required | Default | Contract |
| --- | --- | ---: | --- | --- |
| `text` | string | yes | — | Non-empty after trimming, no NUL, bounded by the configured UTF-8 byte limit. |
| `voice_description` | string | yes | — | Non-empty after trimming, no NUL, bounded by the configured UTF-8 byte limit. This is textual VoiceDesign conditioning, not a voice name or audio reference. |
| `language` | string | no | `auto` | Case-insensitive after trimming. Accepted values are listed below. |
| `seed` | integer | no | random | `0` through `9007199254740991`, inclusive. The selected seed is returned in `x-qwen3-seed` and the stream start event. |
| `max_duration_seconds` | number | no | lower of `120` and the configured maximum | At least `0.08` and no greater than the instance's `max_duration_seconds`. |
| `sampling` | object | no | sampling defaults | Talker and predictor controls described below. Unknown nested fields are rejected. |
| `stream` | boolean | no | inferred | Selects progressive PCM (`true`) or buffered WAV (`false`). |
| `output_format` | string | no | inferred | `pcm_s16le` or `wav`; it must agree with `stream`. |

Accepted language names are `auto`, `chinese`, `english`, `japanese`,
`korean`, `german`, `french`, `russian`, `portuguese`, `spanish`, and
`italian`. The parser is case-insensitive, so `German` and `german` are
equivalent. Turkish is not in the current model contract.

The duration ceiling is converted to codec frames as
`ceil(max_duration_seconds * 12.5)`. It is a safety limit, not a request to pad
the result. Generation normally stops at the model's codec end-of-sequence.

### Output-mode normalization

| `stream` | `output_format` | Result |
| --- | --- | --- |
| omitted or `true` | omitted or `pcm_s16le` | Progressive multipart PCM |
| omitted or `false` | `wav` | Buffered WAV |
| `false` | omitted | Buffered WAV |
| `true` | `wav` | HTTP 422 `unsupported_response_format` |
| `false` | `pcm_s16le` | HTTP 422 `unsupported_response_format` |

### Sampling

The default strategy is `sample`. The following values apply independently to
the talker and predictor unless noted otherwise:

| Field | Default | Accepted range |
| --- | ---: | --- |
| `sampling.strategy` | `sample` | `sample` or `greedy` |
| `sampling.temperature` | `0.9` | `0.01` through `2.0` |
| `sampling.top_p` | `1.0` | `1.17549435e-38` through `1.0` |
| `sampling.top_k` | `50` | `0` through `3072` |
| `sampling.repetition_penalty` | `1.05` | `0.1` through `2.0` |
| `sampling.predictor.strategy` | parent strategy | `sample` or `greedy` |
| `sampling.predictor.temperature` | `0.9` | `0.01` through `2.0` |
| `sampling.predictor.top_p` | `1.0` | `1.17549435e-38` through `1.0` |
| `sampling.predictor.top_k` | `50` | `0` through `2048` |

Every floating-point value must be finite. When either strategy is `greedy`,
omit `temperature`, `top_p`, and `top_k` at that same level. Supplying those
fields with a greedy strategy returns HTTP 422 `invalid_sampling`.
`repetition_penalty` applies only to the talker; predictor repetition penalty
is fixed internally at `1.0`.

Example sampled request:

```json
{
  "text": "Welcome to the native streaming API.",
  "voice_description": "A calm, confident technical narrator with precise articulation.",
  "language": "english",
  "seed": 42,
  "max_duration_seconds": 30,
  "sampling": {
    "strategy": "sample",
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "repetition_penalty": 1.05,
    "predictor": {
      "strategy": "sample",
      "temperature": 0.9,
      "top_p": 1.0,
      "top_k": 50
    }
  }
}
```

Example deterministic request:

```json
{
  "text": "This request uses greedy selection.",
  "voice_description": "A neutral adult narrator.",
  "language": "english",
  "seed": 42,
  "sampling": {
    "strategy": "greedy",
    "repetition_penalty": 1.05,
    "predictor": {
      "strategy": "greedy"
    }
  }
}
```

## Multipart stream

A progressive response has HTTP status 200 and a content type such as:

```http
Content-Type: multipart/mixed; boundary=qwen3tts-0198f65da67974118f7c151dbf0486be
Cache-Control: no-store
X-Accel-Buffering: no
X-Request-Id: 0198f65d-a679-7411-8f7c-151dbf0486be
X-Qwen3-Seed: 42
```

The body contains exactly one JSON start part, one or more binary audio parts,
and one terminal JSON end or error part. Every part has `Content-Type` and
`Content-Length`; clients must use the declared boundary and length rather
than searching binary audio for delimiter text.

### Start part

```http
--qwen3tts-0198f65da67974118f7c151dbf0486be
Content-Type: application/json
Content-Length: <bytes>

{"type":"start","request_id":"0198f65d-a679-7411-8f7c-151dbf0486be","model":"qwen3-tts-1.7b-voice-design","seed":42,"audio":{"encoding":"pcm_s16le","sample_rate_hz":24000,"channels":1,"samples_per_codec_frame":1920}}
```

Line endings on the wire are CRLF. The start part is queued before the native
worker begins returning audio.

### Audio parts

```http
--qwen3tts-0198f65da67974118f7c151dbf0486be
Content-Type: audio/pcm;rate=24000;channels=1;format=s16le
Content-Length: <bytes>
X-Sequence: 0
X-First-Codec-Frame: 0
X-First-Sample: 0
X-Sample-Count: 1920
X-Codec-Frames: 1
X-Final: false

<binary little-endian signed 16-bit PCM>
```

The first audio part contains one codec frame: 1,920 samples or 80 ms. Later
parts contain up to four frames: 7,680 samples or 320 ms. Sequence, codec-frame,
and sample positions are contiguous and zero-based. The final audio part has
`X-Final: true`.

### End part

After the final audio packet has been delivered and the native request has
retired, the server emits a JSON end part and closes the multipart boundary:

```json
{
  "type": "end",
  "request_id": "0198f65d-a679-7411-8f7c-151dbf0486be",
  "finish_reason": "stop",
  "metrics": {
    "queue_microseconds": 0,
    "prefill_microseconds": 100000,
    "first_codec_frame_microseconds": 150000,
    "first_audio_microseconds": 155000,
    "wall_microseconds": 2500000,
    "generated_codec_frames": 30,
    "emitted_samples": 57600,
    "emitted_packets": 9,
    "talker_gpu_microseconds": 1800000.0,
    "codec_gpu_microseconds": 300000.0,
    "peak_request_device_bytes": 47000000,
    "peak_request_host_bytes": 46080
  }
}
```

Metric values above are illustrative; the field names and units are the
contract. `finish_reason` is `stop` for model end-of-sequence and `length` when
the frame ceiling is reached.

### Error part after stream start

HTTP status and headers cannot change after streaming begins. A cancellation,
native failure, or retirement timeout therefore ends an already-started HTTP
200 stream with a JSON error part:

```json
{
  "type": "error",
  "request_id": "0198f65d-a679-7411-8f7c-151dbf0486be",
  "error": {
    "code": "request_cancelled",
    "detail": "the native request was cancelled"
  }
}
```

A client must treat a terminal error part, a missing terminal part, a broken
boundary, or discontinuous audio positions as a failed synthesis even though
the initial HTTP status was 200.

## Buffered WAV

Select buffered output with `"stream": false`, `"output_format": "wav"`, or
both. The server waits for generation and native retirement before returning
headers.

```http
HTTP/1.1 200 OK
Content-Type: audio/wav
Content-Length: <bytes>
Content-Disposition: inline; filename=voice-design.wav
Cache-Control: no-store
X-Request-Id: <uuid>
X-Qwen3-Seed: <integer>
X-Finish-Reason: stop
```

The body is a complete RIFF/WAVE file containing mono 24 kHz PCM16 audio.
`X-Finish-Reason` is `stop` or `length`. Buffered response capacity is bounded
to the configured native concurrency and remains reserved until the caller
consumes or drops the WAV body.

## Compatibility synthesis

### `POST /v1/audio/speech`

This alias intentionally supports only buffered WAV from the same VoiceDesign
model. It is not a general OpenAI Audio API implementation.

| Field | Type | Required | Default | Contract |
| --- | --- | ---: | --- | --- |
| `model` | string | yes | — | Must equal `qwen3-tts-1.7b-voice-design`. |
| `input` | string | yes | — | Maps to native `text`. |
| `voice` | string | yes | — | Maps to textual `voice_description`; it is not a named or cloned voice. |
| `response_format` | string | no | `wav` | The parser recognizes `wav` and `pcm`, but only `wav` is supported. |
| `speed` | number | no | `1.0` | Must be exactly `1.0`. |
| `stream` | boolean | no | `false` | Must be `false`. |

Language is fixed to `auto` on this alias. Use the native endpoint to choose a
language, stream PCM, provide a seed, set a duration ceiling, or tune sampling.

```bash
curl --fail-with-body \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "qwen3-tts-1.7b-voice-design",
    "input": "Hello from the compatibility endpoint.",
    "voice": "A calm and clear adult narrator.",
    "response_format": "wav"
  }' \
  --output compatibility.wav \
  http://127.0.0.1:8080/v1/audio/speech
```

## Cancellation

### `DELETE /v1/requests/{request_id}`

The path parameter must be a UUID. For an active request, the server signals
the process-wide child cancellation token and returns:

```http
HTTP/1.1 202 Accepted
Content-Type: application/json
```

```json
{
  "request_id": "0198f65d-a679-7411-8f7c-151dbf0486be",
  "status": "cancellation_requested"
}
```

Streaming clients that remain connected receive a terminal error part.
Buffered clients receive HTTP 409 `request_cancelled` if cancellation wins
before response headers. A request that has already retired returns HTTP 404
`request_not_found`.

If native retirement exceeded the safety deadline, the ID remains as an
unhealthy tombstone and cancellation returns HTTP 503
`native_retirement_timeout`. Restart the process; do not route new work to it.

## Health and capabilities

### `GET /health/live`

Always returns HTTP 200 while the HTTP process can answer:

```json
{"status":"live"}
```

Liveness does not imply that the GPU engine can accept work.

### `GET /health/ready`

Returns HTTP 200 when the shared engine is healthy and HTTP 503 otherwise:

```json
{
  "status": "ready",
  "model": "qwen3-tts-1.7b-voice-design",
  "model_kind": "voice_design",
  "sample_rate_hz": 24000,
  "engine_loaded": true
}
```

The unavailable variant uses `"status":"not_ready"` and
`"engine_loaded":false` with the same remaining fields.

### `GET /v1/capabilities`

Returns the deployed server limits and fixed audio/model contract:

```json
{
  "model": "qwen3-tts-1.7b-voice-design",
  "model_kind": "voice_design",
  "voice_clone": false,
  "sample_rate_hz": 24000,
  "channels": 1,
  "encoding": "pcm_s16le",
  "samples_per_codec_frame": 1920,
  "max_codec_frames": 8192,
  "max_concurrent_requests": 6,
  "max_text_bytes": 32768,
  "max_voice_description_bytes": 4096,
  "max_duration_seconds": 120.0,
  "languages": [
    "auto", "chinese", "english", "japanese", "korean", "german",
    "french", "russian", "portuguese", "spanish", "italian"
  ],
  "streaming": "multipart/mixed"
}
```

The example shows production-image defaults. `max_codec_frames` is the native
hard ceiling; `max_duration_seconds` is the effective deployment ceiling.

## Metrics

### `GET /metrics`

Returns Prometheus text format without prompt, text, voice-description,
language, request-ID, or user labels. The exported series are documented in
[Operations](OPERATIONS.md#metrics).

## Error contract

Errors produced before response streaming begins use
`Content-Type: application/problem+json`:

```json
{
  "type": "https://qwen3-tts.local/problems/unsupported_language",
  "title": "Request validation failed",
  "status": 422,
  "code": "unsupported_language",
  "detail": "language \"turkish\" is unsupported; accepted values are auto, chinese, english, japanese, korean, german, french, russian, portuguese, spanish, italian"
}
```

`request_id` is optional. Early JSON, semantic-validation, invalid-header, and
shutdown-admission errors omit it. Errors explicitly associated with a chosen
request, such as duplicate admission, native start failure, or cancellation,
include it. The problem `type` is an identifier; the server does not serve
documentation from that URL.

| HTTP | Codes | Meaning |
| ---: | --- | --- |
| 400 | `malformed_json`, `invalid_request_id` | Invalid JSON, unknown fields, missing/wrong JSON fields, or invalid UUID. |
| 404 | `request_not_found` | Cancellation target is not active or has retired. |
| 409 | `request_id_in_use`, `request_cancelled` | Duplicate active ID or cancellation before a buffered response. |
| 413 | `request_too_large` | JSON body exceeds the configured aggregate byte limit. |
| 415 | `unsupported_media_type` | Synthesis request is not `application/json`. |
| 422 | `invalid_text`, `invalid_voice_description`, `unsupported_language`, `invalid_max_duration`, `invalid_seed`, `invalid_sampling`, `unsupported_response_format`, `unsupported_model`, `unsupported_speed`, `invalid_request` | Semantically invalid request. |
| 429 | `capacity_exhausted` | Native or buffered-egress capacity is full. Includes `Retry-After: 1`. |
| 500 | `internal_error`, `worker_panicked`, `buffer_limit_exceeded`, `wav_encoding_failed` | Internal invariant, worker, bounded buffer, or WAV encoding failure. |
| 503 | `backend_unavailable`, `server_shutting_down`, `native_retirement_timeout` | Engine unavailable, admission closed for shutdown, or a native request failed to retire. |

Exact `detail` text is diagnostic and may become more specific. Integrations
should branch on HTTP status and `code`, not on `title`, `detail`, or the
problem-type URL.
