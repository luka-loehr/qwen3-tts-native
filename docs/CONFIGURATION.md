# Configuration

`qwen3-tts-server` is configured through environment variables. It has no
server CLI flags and does not read a configuration file. Configuration is
validated before model loading; the listener is bound only after the native
engine has loaded and completed warm-up.

The production container provides safe image defaults and fixed internal
paths. A directly executed binary uses the standalone defaults shown below.

## Server environment variables

| Variable | Required | Standalone default | Production image default | Validation and effect |
| --- | ---: | --- | --- | --- |
| `QWEN3_TTS_BIND` | no | `127.0.0.1:8080` | `0.0.0.0:8080` | Must parse as a socket address. Controls the HTTP listener. |
| `QWEN3_TTS_MODEL_ROOT` | yes | — | `/opt/qwen3-tts/model` | Non-empty path to the prepared 1.7B VoiceDesign artifact. |
| `QWEN3_TTS_TALKER_LIBRARY` | yes | — | `/opt/qwen3-tts/lib/libqwen3_tts_cuda.so` | Non-empty path to the native talker/predictor shared library. |
| `QWEN3_TTS_CODEC_LIBRARY` | yes | — | `/opt/qwen3-tts/lib/libqwen3_tts_codec_cuda.so` | Non-empty path to the native speech-decoder shared library. |
| `QWEN3_TTS_CUDART_LIBRARY` | no | automatic lookup | automatic lookup | Optional first CUDA runtime-library candidate for device packet staging. When absent, or if that candidate cannot load, the runtime tries `libcudart.so.13` and then `libcudart.so`. Do not override it in the published image. |
| `QWEN3_TTS_DEVICE_INDEX` | no | `0` | `0` | Signed integer CUDA device index passed to both native models. |
| `QWEN3_TTS_MAX_CONCURRENT_REQUESTS` | no | `3` | `6` | Integer from `1` through the native maximum of `6`. Controls native slots and buffered-WAV egress permits. |
| `QWEN3_TTS_MAX_TEXT_BYTES` | no | `32768` | `32768` | Integer from `1` through `1048576`. Measured as UTF-8 bytes, not characters. |
| `QWEN3_TTS_MAX_VOICE_DESCRIPTION_BYTES` | no | `4096` | `4096` | Integer from `1` through `262144`. Measured as UTF-8 bytes. |
| `QWEN3_TTS_MAX_DURATION_SECONDS` | no | `120` | `120` | Finite number from `0.08` through `655.36`. Sets the per-request maximum. |
| `RUST_LOG` | no | `info` | `info` | `tracing-subscriber` filter. Missing or invalid values fall back to `info`. |

The three path variables are required by the standalone executable even if a
library search path exists. An empty value is rejected. Missing, unreadable,
incompatible, or incorrect files fail native engine loading before the service
binds its port.

## Request-limit behavior

### Duration

The runtime produces 12.5 codec frames per second. A request duration ceiling
is converted with:

```text
max_codec_frames = ceil(max_duration_seconds * 12.5)
```

Each frame represents 1,920 mono samples at 24 kHz, or 80 ms. The intrinsic
frame ceiling is 8,192 frames (`655.36` seconds).

When a request omits `max_duration_seconds`, its default is the lower of 120
seconds and `QWEN3_TTS_MAX_DURATION_SECONDS`. Raising the deployment maximum
above 120 does not raise this per-request default; a caller must explicitly
request a longer ceiling.

The ceiling does not force output to that length. Natural codec
end-of-sequence stops earlier with `finish_reason: "stop"`; reaching the
ceiling returns `finish_reason: "length"`.

### JSON body

The Axum body limit is derived from the configured input limits:

```text
max_body_bytes = max_text_bytes + max_voice_description_bytes + 16384
```

With image defaults, the maximum JSON body is 53,248 bytes. This allowance
covers JSON structure and sampling fields in addition to the two bounded
strings. Exceeding it returns HTTP 413 `request_too_large`.

The service checks `text` and `voice_description` independently after JSON
parsing. Both must be non-empty after trimming and must not contain a NUL
character.

### Buffered audio

Buffered WAV uses a bounded PCM vector. Its maximum PCM payload is:

```text
ceil(max_duration_seconds * 12.5) * 1920 samples * 2 bytes
```

At the 120-second image default this is 5,760,000 PCM bytes, plus the WAV
header. The service also reserves one buffered-response egress permit per WAV
response until the HTTP body has been consumed or dropped. The permit count
equals `QWEN3_TTS_MAX_CONCURRENT_REQUESTS`.

### Concurrency and backpressure

The native scheduler owns a fixed number of request slots. There is no
unbounded server-side waiting queue: a request arriving when all slots are in
use returns HTTP 429 `capacity_exhausted` with `Retry-After: 1`.

Increasing concurrency raises per-request session-memory demand and can raise
individual latency even when aggregate throughput improves. The value cannot
exceed six because six is the compiled native batch and scheduler limit.

Within each request, the native runtime uses four codec frames per normal
packet and three bounded PCM ring slots. The first packet is deliberately one
frame for lower time to first audio. These internal values are fixed in the
current server and are not environment variables.

## Fixed service timeouts

The current executable does not expose timeout overrides:

| Timeout | Value | Behavior |
| --- | ---: | --- |
| Native poll | 100 ms | Maximum wait for one poll before the worker checks cancellation again. |
| Slow HTTP client | 5 s | Maximum wait to enqueue a multipart chunk to the one-slot HTTP stream channel. |
| Native retirement | 25 s | Maximum wait for a completed or cancelled native request to release its slot. |
| Graceful shutdown | 35 s | Process deadline after SIGINT or SIGTERM; a watchdog forces exit after it. |
| Container healthcheck I/O | 2 s | Loopback connect, read, and write timeout inside `qwen3-tts-healthcheck`. |

The configured cleanup invariant is
`slow_client_timeout + retirement_timeout < shutdown_timeout` (30 seconds is
less than 35 seconds). Changing these values requires a source change and a
new qualified image.

## Container runtime environment

The production image also defines variables used by the dynamic loader,
NVIDIA runtime, or container filesystem. They are not parsed as server
settings:

| Variable | Image value | Owner |
| --- | --- | --- |
| `LD_LIBRARY_PATH` | Native library directory, CUDA SBSA libraries, and NVIDIA driver mounts | ELF dynamic loader / NVIDIA runtime |
| `CUDA_MODULE_LOADING` | `EAGER` | CUDA runtime |
| `NVIDIA_VISIBLE_DEVICES` | `all` | NVIDIA Container Toolkit; `--gpus` should still narrow the device exposed to a deployment. |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | NVIDIA Container Toolkit |
| `HOME` | `/nonexistent` | Non-login runtime user |
| `TMPDIR` | `/tmp` | Writable tmpfs expected by the hardened run command |
| `QWEN3_TTS_LIBRARY_DIR` | `/opt/qwen3-tts/lib` | Component lookup for public C-ABI consumers. The HTTP server reads the two explicit library variables instead. |

Do not override `QWEN3_TTS_MODEL_ROOT`, the two library paths, or
`LD_LIBRARY_PATH` in the published image. The image was built and qualified
against its embedded, hash-verified model and native libraries.

## Safe deployment overrides

The most common supported overrides are listener publication, device index,
concurrency, request byte limits, duration ceiling, and log filtering:

```bash
docker run --rm \
  --gpus device=0 \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m,uid=10001,gid=10001 \
  --pids-limit=256 \
  --publish 127.0.0.1:8080:8080 \
  --env QWEN3_TTS_MAX_CONCURRENT_REQUESTS=3 \
  --env QWEN3_TTS_MAX_TEXT_BYTES=16384 \
  --env QWEN3_TTS_MAX_VOICE_DESCRIPTION_BYTES=2048 \
  --env QWEN3_TTS_MAX_DURATION_SECONDS=60 \
  --env RUST_LOG=qwen3_tts_server=info \
  'docker.io/luka-loehr/qwen3-tts-native@sha256:<PUBLISHED_DIGEST>'
```

Registry publication is pending; replace the explicit placeholder with the
accepted release digest.

## Network and security configuration

The standalone default binds to loopback. The image binds to all container
interfaces so Docker port publication works, but the recommended host mapping
is still `127.0.0.1:8080:8080`.

The server itself does not implement TLS, authentication, authorization, CORS,
tenant quotas, or a general rate limiter. Configure those controls in a
reverse proxy or service mesh. Never publish port 8080 directly on an
untrusted network.

See [Operations](OPERATIONS.md#security-boundary) for the complete runtime
hardening contract.
