# Quickstart

This guide starts the production Qwen3-TTS 1.7B VoiceDesign image on one NVIDIA
DGX Spark GPU and exercises readiness, buffered WAV, and progressive PCM.

## Prerequisites

- An NVIDIA DGX Spark or equivalent `linux/arm64` host with an NVIDIA GB10 GPU.
- A host driver compatible with the CUDA 13.0.3 userspace in the image.
- Docker Engine with the NVIDIA Container Toolkit configured.
- Access to the private image repository while the project remains private.

The image already contains the pinned VoiceDesign and decoder weights. Do not
download a model separately and do not mount model files into the container.

## 1. Select the immutable image

```bash
IMAGE='docker.io/luka-loehr/qwen3-tts-native@sha256:<published-digest>'
```

`<published-digest>` is deliberately a placeholder while registry publication
and clean-pull qualification are pending. Replace it with the release digest;
do not deploy a mutable `latest` tag.

```bash
docker pull "$IMAGE"
```

The published image must report `linux/arm64`:

```bash
docker image inspect "$IMAGE" \
  --format '{{.Os}}/{{.Architecture}} {{.Config.User}}'
```

Expected user: `10001:10001`.

## 2. Start the service

```bash
docker run --detach \
  --name qwen3-tts-native \
  --gpus device=0 \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m,uid=10001,gid=10001 \
  --pids-limit=256 \
  --publish 127.0.0.1:8080:8080 \
  "$IMAGE"
```

The process loads both native libraries and both weight sets, initializes the
GPU, and runs a real one-frame English warm-up before it binds the HTTP
listener. A started container is therefore not immediately the same as a ready
container.

Follow startup without exposing prompt data:

```bash
docker logs --follow qwen3-tts-native
```

Stop following the log after the `native VoiceDesign server ready` message.

## 3. Wait for readiness

```bash
until curl --fail --silent http://127.0.0.1:8080/health/ready; do
  sleep 1
done
```

The ready response is:

```json
{
  "status": "ready",
  "model": "qwen3-tts-1.7b-voice-design",
  "model_kind": "voice_design",
  "sample_rate_hz": 24000,
  "engine_loaded": true
}
```

Inspect the effective instance limits before accepting traffic:

```bash
curl --fail --silent http://127.0.0.1:8080/v1/capabilities
```

## 4. Generate a buffered WAV

```bash
curl --fail-with-body \
  --request POST \
  --header 'Content-Type: application/json' \
  --header 'x-request-id: 0198f65d-a679-7411-8f7c-151dbf0486be' \
  --data '{
    "text": "Guten Morgen. Dies ist ein ruhiger nativer Sprachtest.",
    "voice_description": "A calm adult male voice with a warm low register and measured delivery.",
    "language": "german",
    "seed": 42,
    "max_duration_seconds": 30,
    "stream": false,
    "output_format": "wav"
  }' \
  --output voice-design.wav \
  http://127.0.0.1:8080/v1/voice-design/speech
```

The file is mono, 24 kHz, signed 16-bit PCM in a RIFF/WAVE container. The
response also carries `x-request-id`, `x-qwen3-seed`, and `x-finish-reason`.

## 5. Observe progressive streaming

Streaming is the default for the native endpoint. Save the raw multipart body
and response headers as separate files:

```bash
curl --no-buffer --fail-with-body \
  --request POST \
  --header 'Content-Type: application/json' \
  --data '{
    "text": "Streaming begins before the full utterance has been generated.",
    "voice_description": "A composed technical narrator with clear articulation.",
    "language": "english",
    "seed": 43,
    "max_duration_seconds": 30
  }' \
  --dump-header stream.headers \
  --output stream.multipart \
  http://127.0.0.1:8080/v1/voice-design/speech
```

Do not treat `stream.multipart` as a PCM file. It contains a JSON start part,
one or more binary PCM parts, and a terminal JSON end or error part. A client
must parse the boundary from the response `Content-Type` header and honor each
part's `Content-Length`. See [Multipart stream](API.md#multipart-stream).

## 6. Cancel an active request

Choose an `x-request-id` before starting a long request, then cancel it from a
second connection:

```bash
curl --request DELETE \
  http://127.0.0.1:8080/v1/requests/0198f65d-a679-7411-8f7c-151dbf0486be
```

An active request returns HTTP 202 with
`{"status":"cancellation_requested"}`. Cancellation is idempotent inside the
native request object, but the HTTP endpoint returns 404 once the request has
fully retired and left the active map.

## 7. Stop cleanly

```bash
docker stop --time 40 qwen3-tts-native
docker rm qwen3-tts-native
```

`SIGTERM` closes request admission, cancels active work, and starts the bounded
35-second graceful-shutdown deadline. Do not use `docker kill` during normal
operation.

## Next steps

- Read the complete request and response contract in [API](API.md).
- Review all environment variables and limits in
  [Configuration](CONFIGURATION.md).
- Configure health probes, metrics, shutdown, and network controls using
  [Operations](OPERATIONS.md).
