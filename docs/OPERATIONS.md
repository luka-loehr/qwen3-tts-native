# Operations

This runbook covers the production Qwen3-TTS 1.7B VoiceDesign HTTP service on
NVIDIA DGX Spark. The process is stateless between restarts: model weights are
immutable image content, and requests, seeds, PCM buffers, and metrics live
only in process memory.

## Startup and readiness

Startup is intentionally fail-closed:

1. Parse and validate environment configuration.
2. Load the talker/predictor shared library and VoiceDesign weights.
3. Load the codec shared library and decoder weights.
4. Construct the bounded scheduler and shared native engine.
5. Run a real English warm-up request through talker, predictor, device
   handoff, and codec.
6. Verify its final packet, retirement, and metrics.
7. Bind the HTTP listener and log `native VoiceDesign server ready`.

Before step 7, the port is not listening. A container in Docker's `starting`
state is not ready for traffic.

The image healthcheck runs every 30 seconds, has a 3-second Docker timeout, a
45-second start period, and fails after three consecutive unsuccessful checks.
It calls the model-free native healthcheck binary against
`http://127.0.0.1:8080/health/ready`. The helper accepts only loopback `http://`
URLs, uses 2-second socket I/O timeouts, requires HTTP 200, and verifies both
`"status":"ready"` and `"engine_loaded":true`.

### Probe policy

Use the endpoints for distinct purposes:

| Probe | Success | Meaning |
| --- | --- | --- |
| `GET /health/live` | HTTP 200 | The HTTP process can answer. It does not inspect engine health. |
| `GET /health/ready` | HTTP 200 and `engine_loaded:true` | The shared native engine is healthy and can accept routing. |
| `GET /health/ready` | HTTP 503 | Stop routing immediately; restart if the state persists. |

For an orchestrator, use readiness to gate traffic. A startup probe must allow
the full model-load and warm-up interval. Use liveness conservatively: killing
a healthy process during an expected cold start only repeats expensive model
loading.

Manual checks:

```bash
curl --fail --silent http://127.0.0.1:8080/health/live
curl --fail --silent http://127.0.0.1:8080/health/ready
curl --fail --silent http://127.0.0.1:8080/v1/capabilities
```

Always compare `/v1/capabilities` with deployment expectations, especially
`max_concurrent_requests`, input byte limits, and `max_duration_seconds`.

## Metrics

`GET /metrics` returns Prometheus text format. Metrics reset on every process
restart and have no labels. In particular, prompt text, voice descriptions,
language, request IDs, and user data are not exported.

| Series | Type | Exact behavior |
| --- | --- | --- |
| `qwen3_tts_http_requests_total` | counter | Requests that successfully started in the native engine. Validation failures are not included. |
| `qwen3_tts_active_requests` | gauge | Admitted native requests not yet retired. A retirement-timeout tombstone intentionally remains active. |
| `qwen3_tts_streaming_requests_total` | counter | Successfully started progressive multipart requests. |
| `qwen3_tts_buffered_requests_total` | counter | Successfully started buffered-WAV requests. |
| `qwen3_tts_completed_requests_total` | counter | Requests that delivered successful terminal output and retired. |
| `qwen3_tts_failed_requests_total` | counter | Failed requests, including retirement timeouts. |
| `qwen3_tts_cancelled_requests_total` | counter | Requests cancelled by the caller, disconnect, slow-client handling, or shutdown and then retired. |
| `qwen3_tts_rejected_requests_total` | counter | Native start failures and buffered-egress permit exhaustion. It is not a count of every HTTP 4xx/5xx response. |
| `qwen3_tts_retirement_timeouts_total` | counter | Native requests that exceeded the 25-second retirement deadline. Any increase is an engine-health incident. |
| `qwen3_tts_emitted_samples_total` | counter | Samples from successfully completed requests. Partial samples from cancelled or failed requests are not added. |

The Prometheus endpoint does not expose latency, real-time factor, GPU memory,
GPU utilization, or energy. Per-request timing and memory fields are available
only in successful multipart end events. Collect host/GPU telemetry separately
and parse terminal events at the client or gateway if aggregate latency
histograms are required.

### Suggested operator alerts

These are deployment policies, not built-in thresholds:

- Page when readiness returns 503 for more than one probe interval.
- Page on any increase in `qwen3_tts_retirement_timeouts_total`; remove the
  instance from service and restart it.
- Alert on a sustained increase in `qwen3_tts_failed_requests_total`.
- Alert when `qwen3_tts_active_requests` remains at the configured capacity and
  rejection rate rises.
- Investigate elevated cancellation rate; it can indicate client disconnects,
  proxy buffering, or clients that do not consume multipart data quickly.
- Compare completed, failed, cancelled, and still-active work against accepted
  requests over process-lifetime windows, accounting for counter resets.

## Capacity and admission

The scheduler has a hard capacity from one through six active native requests.
It does not create an unbounded application queue. Exhaustion returns HTTP 429
`capacity_exhausted` with `Retry-After: 1`.

Clients should:

1. Limit their own in-flight requests.
2. Honor `Retry-After` and add randomized backoff.
3. Avoid retrying with the same active `x-request-id`.
4. Retry only when application semantics permit duplicate synthesis.

Buffered WAV has a second bounded resource: one egress permit per configured
concurrent request. A permit remains held while the response body is pending,
even after native generation finishes. Slow or abandoned WAV consumers can
therefore produce HTTP 429 without occupying a native generation slot.

Progressive output uses a one-chunk HTTP channel and a bounded native PCM ring.
This propagates backpressure instead of buffering an unbounded utterance. A
stream client that cannot accept a chunk within five seconds is cancelled and
retired.

## Request cancellation

Supply `x-request-id` before starting work when external cancellation may be
needed. Signal cancellation with:

```bash
REQUEST_ID='0198f65d-a679-7411-8f7c-151dbf0486be'
curl --request DELETE \
  "http://127.0.0.1:8080/v1/requests/$REQUEST_ID"
```

HTTP 202 means the signal was accepted. It does not guarantee that GPU cleanup
has already finished. Retirement releases the scheduler slot. A later DELETE
returns 404 after normal retirement.

For a progressive request, closing the response body also signals
cancellation. A connected client normally receives a terminal JSON error part.
For a buffered request, cancellation before response headers produces HTTP 409
`request_cancelled`.

## Shutdown and rolling replacement

The image declares `STOPSIGNAL SIGTERM`. SIGTERM and SIGINT perform the same
bounded sequence:

1. Serialize against request admission and close admission atomically.
2. Reject new synthesis with HTTP 503 `server_shutting_down`.
3. Propagate cancellation to every active request.
4. Allow Axum to finish graceful HTTP shutdown within 35 seconds.
5. Force process exit with code 124 if the deadline is exceeded. Failure to
   start the watchdog forces exit with code 125.

Give the container runtime more than the internal deadline:

```bash
docker stop --time 40 qwen3-tts-native
```

For a low-disruption rolling replacement, start the new digest, wait for its
readiness, stop sending new requests to the old instance at the load balancer,
wait for `qwen3_tts_active_requests` to reach zero, and only then send SIGTERM.
The service has no separate drain endpoint; SIGTERM cancels rather than
finishes active synthesis.

Avoid SIGKILL in routine operation because it bypasses bounded native cleanup.

## Retirement-timeout incident

Every completed, failed, or cancelled request must release its native session
within 25 seconds. If it does not:

- the engine is marked unhealthy;
- `/health/ready` returns 503;
- `qwen3_tts_retirement_timeouts_total` increments;
- the request UUID remains in the active map as a tombstone;
- `qwen3_tts_active_requests` deliberately does not decrement; and
- DELETE for that UUID returns HTTP 503 `native_retirement_timeout`.

There is no in-process repair path. Remove the instance from routing, preserve
its logs and host GPU diagnostics, and restart the container. If the event
recurs on the same image digest, quarantine that digest and collect the exact
request parameters without placing sensitive text in shared telemetry.

## Security boundary

The service is an inference component, not an internet-facing security layer.
It has no built-in authentication, authorization, TLS, CORS policy, tenant
isolation, or general request-rate limiter. Health, capabilities, synthesis,
cancellation, and metrics share one unauthenticated listener.

Minimum container controls:

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
  'docker.io/luka-loehr/qwen3-tts-native@sha256:<PUBLISHED_DIGEST>'
```

The digest is a publication placeholder until qualification finishes.

The qualified image contract also provides these controls:

- Runtime UID/GID `10001:10001`, no login shell, and no home directory.
- Root-owned model weights with mode `0444`.
- Pinned `linux/arm64` base images and real `sm_121` CUDA SASS without PTX
  fallback.
- No Python, Node.js, PyTorch, SGLang, vLLM, compilers, or development
  packages in the final runtime image.
- Application license, model provenance, third-party license report, and
  CycloneDX Rust SBOM inside the image.

Additional deployment controls remain mandatory:

- Pin the immutable image digest and verify release attestations/signature.
- Publish only to loopback or a private network segment.
- Terminate TLS and enforce authentication, authorization, quotas, and body
  limits at a trusted proxy.
- Restrict `/metrics`, health, and cancellation endpoints to operators or the
  internal control plane.
- Do not mount alternate files over `/opt/qwen3-tts/model` or native libraries.
- Do not place registry credentials, API keys, or request text in environment
  variables, image labels, command history, or proxy access logs.
- Treat text and voice descriptions as sensitive request data even though the
  built-in Prometheus metrics do not expose them.

## Release and rollback

Deploy by registry digest, not `latest` or a moving semantic-version tag. A
release is operationally accepted only after the exact pushed digest passes a
clean pull, image inspection, vulnerability review, signature/attestation
verification, hardened-container startup, readiness, native streaming,
buffered WAV, cancellation, shutdown, memory, and performance qualification.

Rollback means starting the previously accepted digest and routing traffic to
it. The service stores no local user state that needs schema migration.

## Troubleshooting

### Port never opens

Likely causes are invalid environment values, missing/incompatible native
libraries, model/config mismatch, CUDA initialization failure, weight loading
failure, or warm-up failure. Inspect container exit status and startup logs:

```bash
docker inspect qwen3-tts-native --format '{{.State.Status}} {{.State.ExitCode}} {{.State.Error}}'
docker logs qwen3-tts-native
```

Do not treat a connection refusal during warm-up as an HTTP outage; use the
container startup state and logs. Repeated failure on the same digest is not
fixed by extending readiness timeouts.

### Readiness changes from 200 to 503

The native engine was marked unhealthy, normally because backend availability
failed or a request did not retire. Stop routing, inspect
`qwen3_tts_retirement_timeouts_total`, capture host GPU state, and restart the
instance.

### HTTP 429

Check active requests, client read speed, and the effective concurrency from
`/v1/capabilities`. Consume or close WAV bodies promptly. Apply bounded client
backoff rather than immediately increasing concurrency.

### Multipart starts but audio fails

Parse the terminal part. The original HTTP status remains 200 after stream
headers, so a terminal `type: "error"`, missing final part, broken boundary, or
sequence discontinuity is the authoritative failure. Ensure reverse proxies
disable response buffering and preserve streaming; the service sends
`X-Accel-Buffering: no` but not every proxy honors it.

### WAV is shorter than requested

`max_duration_seconds` is a ceiling. A normal model EOS ends earlier with
`X-Finish-Reason: stop`. `X-Finish-Reason: length` means the ceiling was hit and
the utterance may be truncated.

### HTTP 400 on an apparently valid request

JSON objects deny unknown fields. Confirm `Content-Type: application/json`,
field names, JSON types, and absence of reference-audio or voice-clone fields.
Semantic constraints return 422; see [API errors](API.md#error-contract).

### Healthcheck works externally but Docker reports unhealthy

The image healthcheck deliberately resolves only loopback HTTP and requires the
ready payload, not just a 200 status. Inspect Docker health output and confirm
the service still uses the image's port and bind defaults:

```bash
docker inspect qwen3-tts-native --format '{{json .State.Health}}'
```
