#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

[[ $# -eq 3 ]] || release_die "usage: $0 RELEASE_RECORD BUILDER_DOCKER_INFO EVIDENCE_DIR"
readonly RELEASE_RECORD=$1
readonly BUILDER_DOCKER_INFO=$2
readonly EVIDENCE_DIR=$3
release_read_record "$RELEASE_RECORD"
release_require_file "$BUILDER_DOCKER_INFO"
[[ -n "${DOCKER_HOST:-}" ]] || release_die "DOCKER_HOST must select the separate clean-pull daemon"

for command_name in curl docker jq nvidia-smi; do
  release_require_command "$command_name"
done
[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "aarch64" ]] ||
  release_die "GPU acceptance requires Linux ARM64"

readonly DIGEST="$(release_record_value "$RELEASE_RECORD" '.digest')"
readonly REFERENCE="$RELEASE_IMAGE@$DIGEST"
readonly ANON_CONFIG="$(mktemp -d "${TMPDIR:-/tmp}/qwen3-tts-anonymous-docker.XXXXXX")"
readonly CONTAINER_NAME="qwen3-tts-release-${DIGEST#sha256:}"
readonly SHORT_CONTAINER_NAME="${CONTAINER_NAME:0:48}"
container_started=0
cleanup() {
  if (( container_started == 1 )); then
    docker --config "$ANON_CONFIG" rm --force "$SHORT_CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
  rm -rf "$ANON_CONFIG"
}
trap cleanup EXIT

release_require_new_directory "$EVIDENCE_DIR"
docker --config "$ANON_CONFIG" info --format '{{json .}}' >"$EVIDENCE_DIR/clean-daemon-info.json"
readonly BUILD_DAEMON_ID="$(jq -er '.ID | select(type == "string" and length > 0)' "$BUILDER_DOCKER_INFO")"
readonly CLEAN_DAEMON_ID="$(jq -er '.ID | select(type == "string" and length > 0)' "$EVIDENCE_DIR/clean-daemon-info.json")"
[[ "$CLEAN_DAEMON_ID" != "$BUILD_DAEMON_ID" ]] || release_die "clean pull cannot use the build daemon"
[[ -z "$(docker --config "$ANON_CONFIG" image ls --quiet)" ]] ||
  release_die "clean-pull daemon image store is not empty"
docker --config "$ANON_CONFIG" image ls --digests --no-trunc >"$EVIDENCE_DIR/pre-pull-images.txt"
docker --config "$ANON_CONFIG" pull "$REFERENCE" 2>&1 | tee "$EVIDENCE_DIR/pull.log"
docker --config "$ANON_CONFIG" image inspect "$REFERENCE" >"$EVIDENCE_DIR/image-inspect.json"

jq -e \
  --arg reference "$REFERENCE" \
  --arg revision "$(release_record_value "$RELEASE_RECORD" '.source_revision')" \
  --arg version "$(release_record_value "$RELEASE_RECORD" '.release_version')" '
    .[0].Os == "linux"
    and .[0].Architecture == "arm64"
    and .[0].Config.User == "10001:10001"
    and (.[0].RepoDigests | index($reference) != null)
    and .[0].Config.Labels["org.opencontainers.image.revision"] == $revision
    and .[0].Config.Labels["org.opencontainers.image.version"] == $version
    and .[0].Config.Labels["io.qwen3-tts.model.voice-clone"] == "false"
  ' "$EVIDENCE_DIR/image-inspect.json" >/dev/null || release_die "pulled image identity is invalid"
readonly UNPACKED_BYTES="$(jq -er '.[0].Size | numbers' "$EVIDENCE_DIR/image-inspect.json")"
(( UNPACKED_BYTES <= 10000000000 )) || release_die "unpacked image exceeds 10.0 GB"

docker --config "$ANON_CONFIG" run --detach \
  --name "$SHORT_CONTAINER_NAME" \
  --gpus 'device=0' \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m,uid=10001,gid=10001 \
  --pids-limit=256 \
  -p 127.0.0.1:18080:8080 \
  "$REFERENCE" >"$EVIDENCE_DIR/container-id.txt"
container_started=1

readonly START_SECONDS=$SECONDS
ready=0
while (( SECONDS - START_SECONDS <= 20 )); do
  if curl --fail --silent --show-error http://127.0.0.1:18080/health/ready \
    >"$EVIDENCE_DIR/readiness.json"; then
    ready=1
    break
  fi
  sleep 1
done
(( ready == 1 )) || release_die "container did not become ready within 20 seconds"
readonly READY_SECONDS=$((SECONDS - START_SECONDS))
jq -e '.status == "ready" and .engine_loaded == true and .model_kind == "voice_design"' \
  "$EVIDENCE_DIR/readiness.json" >/dev/null || release_die "readiness contract failed"
curl --fail --silent --show-error http://127.0.0.1:18080/v1/capabilities \
  >"$EVIDENCE_DIR/capabilities.json"
jq -e '
  .model_kind == "voice_design" and .voice_clone == false
  and .sample_rate_hz == 24000 and .channels == 1
  and .encoding == "pcm_s16le" and .streaming == "multipart/mixed"
' "$EVIDENCE_DIR/capabilities.json" >/dev/null || release_die "capabilities contract failed"

curl --fail --silent --show-error --no-buffer --max-time 120 \
  --dump-header "$EVIDENCE_DIR/stream.headers" \
  --output "$EVIDENCE_DIR/stream.multipart" \
  --header 'Content-Type: application/json' \
  --data '{"text":"Guten Morgen. Dies ist der finale Streaming-Test.","voice_description":"A calm adult male voice with measured delivery.","language":"german","seed":42,"max_duration_seconds":8}' \
  http://127.0.0.1:18080/v1/voice-design/speech
grep -Eiq '^content-type: multipart/mixed; boundary=qwen3tts-' "$EVIDENCE_DIR/stream.headers" ||
  release_die "stream response is not multipart/mixed"
grep -aFq 'audio/pcm;rate=24000;channels=1;format=s16le' "$EVIDENCE_DIR/stream.multipart" ||
  release_die "stream contains no PCM audio part"
grep -aFq '"type":"end"' "$EVIDENCE_DIR/stream.multipart" || release_die "stream has no terminal event"

docker --config "$ANON_CONFIG" inspect --size "$SHORT_CONTAINER_NAME" >"$EVIDENCE_DIR/container-inspect.json"
docker --config "$ANON_CONFIG" logs --timestamps "$SHORT_CONTAINER_NAME" >"$EVIDENCE_DIR/container.log" 2>&1
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used \
  --format=csv,noheader,nounits >"$EVIDENCE_DIR/nvidia-smi.csv"

jq -n --arg digest "$DIGEST" --argjson ready_seconds "$READY_SECONDS" \
  --argjson unpacked_bytes "$UNPACKED_BYTES" '{
    schema: "qwen3-tts-native/clean-pull-gpu-acceptance/v1",
    digest: $digest,
    pull: "anonymous-empty-store",
    hardened_runtime: "passed",
    gpu: "passed",
    streaming_pcm: "passed",
    readiness_seconds: $ready_seconds,
    unpacked_bytes: $unpacked_bytes
  }' >"$EVIDENCE_DIR/gpu-acceptance.json"
printf 'Anonymous clean-pull and GPU acceptance passed for %s\n' "$REFERENCE"
