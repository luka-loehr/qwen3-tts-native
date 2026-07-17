#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
# shellcheck source=../../benchmarks/tools/lib/process-rss-sampler.sh
source "$RELEASE_IMAGE_ROOT/benchmarks/tools/lib/process-rss-sampler.sh"

[[ $# -eq 3 ]] || release_die "usage: $0 RELEASE_RECORD BUILDER_DOCKER_INFO EVIDENCE_DIR"
readonly RELEASE_RECORD=$1
readonly BUILDER_DOCKER_INFO=$2
readonly EVIDENCE_DIR=$3
release_read_record "$RELEASE_RECORD"
release_require_file "$BUILDER_DOCKER_INFO"
[[ -n "${DOCKER_HOST:-}" ]] || release_die "DOCKER_HOST must select the separate clean-pull daemon"

for command_name in awk curl date docker find grep jq nvidia-smi sha256sum sleep sort sox stat; do
  release_require_command "$command_name"
done
[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "aarch64" ]] ||
  release_die "GPU acceptance requires Linux ARM64"

readonly DIGEST="$(release_record_value "$RELEASE_RECORD" '.digest')"
readonly REFERENCE="$RELEASE_IMAGE@$DIGEST"
readonly ANON_CONFIG="$(mktemp -d "${TMPDIR:-/tmp}/qwen3-tts-anonymous-docker.XXXXXX")"
readonly CONTAINER_NAME="qwen3-tts-release-${DIGEST#sha256:}"
readonly SHORT_CONTAINER_NAME="${CONTAINER_NAME:0:48}"
readonly SAMPLE_INTERVAL_MS=100
readonly SAMPLE_INTERVAL_NS=100000000
readonly MAXIMUM_SAMPLE_GAP_MS=200
readonly MAXIMUM_SAMPLE_GAP_NS=200000000
readonly COLD_RSS_LIMIT_BYTES=4509715660
readonly STEADY_SAMPLE_COUNT=10

container_created=0
rss_sampler_pid=0
cancellation_pid=0

cleanup() {
  local status=$?
  trap - EXIT
  if ((cancellation_pid > 0)); then
    kill "$cancellation_pid" >/dev/null 2>&1 || true
    wait "$cancellation_pid" >/dev/null 2>&1 || true
  fi
  if ((rss_sampler_pid > 0)); then
    kill "$rss_sampler_pid" >/dev/null 2>&1 || true
    wait "$rss_sampler_pid" >/dev/null 2>&1 || true
  fi
  if ((container_created == 1)); then
    docker --config "$ANON_CONFIG" rm --force "$SHORT_CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
  rm -rf "$ANON_CONFIG"
  exit "$status"
}
trap cleanup EXIT

wall_time_unix_ns() {
  local value
  value=$(date +%s%N)
  [[ "$value" =~ ^[0-9]{19}$ ]] || release_die "GNU date did not provide nanosecond Unix time"
  printf '%s\n' "$value"
}

timestamp_utc() {
  LC_ALL=C date --utc '+%Y-%m-%dT%H:%M:%S.%NZ'
}

sleep_until_next_sample() {
  local sample_started_ns=$1
  local now_ns remaining_ns remaining_seconds
  now_ns=$(wall_time_unix_ns)
  remaining_ns=$((sample_started_ns + SAMPLE_INTERVAL_NS - now_ns))
  ((remaining_ns > 0)) || return 0
  remaining_seconds=$(awk -v nanoseconds="$remaining_ns" 'BEGIN { printf "%.9f", nanoseconds / 1000000000 }')
  sleep "$remaining_seconds"
}

start_container() {
  local id_output=$1
  docker --config "$ANON_CONFIG" run --detach \
    --name "$SHORT_CONTAINER_NAME" \
    --gpus 'device=0' \
    --read-only \
    --cap-drop=ALL \
    --security-opt=no-new-privileges \
    --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m,uid=10001,gid=10001 \
    --pids-limit=256 \
    -p 127.0.0.1:18080:8080 \
    "$REFERENCE" >"$id_output"
  container_created=1
}

wait_for_readiness() {
  local output=$1
  local started_seconds=$2
  local ready=0
  while ((SECONDS - started_seconds <= 20)); do
    if curl --fail --silent --show-error --connect-timeout 1 --max-time 1 \
      http://127.0.0.1:18080/health/ready >"$output" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 0.1
  done
  ((ready == 1)) || release_die "container did not become ready within 20 seconds"
  jq -e '.status == "ready" and .engine_loaded == true and .model_kind == "voice_design"' \
    "$output" >/dev/null || release_die "readiness contract failed"
  printf '%s\n' "$((SECONDS - started_seconds))"
}

resolve_container_cgroup() {
  local container_pid cgroup_relative cgroup_dir
  container_pid=$(docker --config "$ANON_CONFIG" inspect --format '{{.State.Pid}}' "$SHORT_CONTAINER_NAME")
  [[ "$container_pid" =~ ^[1-9][0-9]*$ ]] || release_die "release container has no host PID"
  cgroup_relative=$(awk -F: '$1 == "0" { print $3; exit }' "/proc/$container_pid/cgroup")
  [[ -n "$cgroup_relative" ]] || release_die "cannot resolve release container cgroup v2 path"
  cgroup_dir="/sys/fs/cgroup$cgroup_relative"
  [[ -r "$cgroup_dir/cgroup.procs" ]] || release_die "release container cgroup process list is unavailable"
  printf '%s\n' "$cgroup_dir"
}

start_rss_sampler() {
  local cgroup_dir=$1
  local detail_output=$2
  local total_output=$3
  (
    trap 'exit 0' TERM INT
    while :; do
      cycle_started_ns=$(wall_time_unix_ns)
      sample_process_rss_cycle \
        "$cgroup_dir/cgroup.procs" /proc "$cycle_started_ns" "$SAMPLE_INTERVAL_NS" \
        "$detail_output" "$total_output" 3
      sleep_until_next_sample "$cycle_started_ns"
    done
  ) &
  rss_sampler_pid=$!
}

stop_rss_sampler() {
  ((rss_sampler_pid > 0)) || release_die "process RSS sampler was not running"
  kill -TERM "$rss_sampler_pid"
  wait "$rss_sampler_pid" || release_die "process RSS sampler did not stop cleanly"
  rss_sampler_pid=0
}

summarize_rss_samples() {
  local path=$1
  local minimum_samples=$2
  awk -F, -v minimum="$minimum_samples" -v maximum_gap="$MAXIMUM_SAMPLE_GAP_NS" '
    NR == 1 {
      if ($0 != "wall_time_unix_ns,timestamp_utc,listed_processes,sampled_processes,process_rss_sum_bytes,sample_complete") exit 2
      next
    }
    {
      if (NF != 6 || $1 !~ /^[0-9]+$/ || $3 !~ /^[0-9]+$/ || $4 !~ /^[0-9]+$/ ||
          $5 !~ /^[0-9]+$/ || $6 != 1 || $3 < 1 || $4 != $3 || $5 < 1) exit 2
      count++
      timestamp = $1 + 0
      if (previous > 0) {
        gap = timestamp - previous
        if (gap <= 0 || gap > maximum_gap) exit 2
        if (gap > largest_gap) largest_gap = gap
      }
      previous = timestamp
      sum += $5
      if ($5 > peak) peak = $5
    }
    END {
      if (count < minimum || peak < 1) exit 2
      printf "%.0f\t%.3f\t%d\t%.3f\n", peak, sum / count, count, largest_gap / 1000000
    }
  ' "$path" || release_die "process RSS samples are incomplete or exceed the 200 ms gap limit"
}

validate_stream() {
  local headers=$1
  local body=$2
  local require_natural_eos=$3
  grep -Eiq '^content-type: multipart/mixed; boundary=qwen3tts-' "$headers" ||
    release_die "stream response is not multipart/mixed"
  grep -aFq 'audio/pcm;rate=24000;channels=1;format=s16le' "$body" ||
    release_die "stream contains no PCM audio part"
  grep -aFq '"type":"end"' "$body" || release_die "stream has no terminal end event"
  if [[ "$require_natural_eos" == true ]]; then
    grep -aFq '"finish_reason":"stop"' "$body" || release_die "stream did not finish at natural EOS"
  fi
  if grep -aFq '"type":"error"' "$body"; then
    release_die "stream contains a terminal error event"
  fi
}

graceful_stop_and_remove() {
  local prefix=$1
  local started_seconds=$SECONDS
  docker --config "$ANON_CONFIG" stop --time 40 "$SHORT_CONTAINER_NAME" >"$EVIDENCE_DIR/$prefix-stop.txt"
  local elapsed=$((SECONDS - started_seconds))
  docker --config "$ANON_CONFIG" inspect "$SHORT_CONTAINER_NAME" >"$EVIDENCE_DIR/$prefix-stopped-inspect.json"
  jq -e '.[0].State.Status == "exited" and .[0].State.ExitCode == 0 and .[0].State.OOMKilled == false' \
    "$EVIDENCE_DIR/$prefix-stopped-inspect.json" >/dev/null ||
    release_die "release container did not exit cleanly after SIGTERM"
  docker --config "$ANON_CONFIG" logs --timestamps "$SHORT_CONTAINER_NAME" \
    >"$EVIDENCE_DIR/$prefix-container.log" 2>&1
  docker --config "$ANON_CONFIG" rm "$SHORT_CONTAINER_NAME" >"$EVIDENCE_DIR/$prefix-rm.txt"
  container_created=0
  printf '%s\n' "$elapsed"
}

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
((UNPACKED_BYTES <= 10000000000)) || release_die "unpacked image exceeds 10.0 GB"

printf '%s\n' \
  'wall_time_unix_ns,timestamp_utc,pid,process_start_ticks,process_name,rss_bytes' \
  >"$EVIDENCE_DIR/cold-process-rss.csv"
printf '%s\n' \
  'wall_time_unix_ns,timestamp_utc,listed_processes,sampled_processes,process_rss_sum_bytes,sample_complete' \
  >"$EVIDENCE_DIR/cold-process-rss-total.csv"
readonly COLD_START_SECONDS=$SECONDS
start_container "$EVIDENCE_DIR/container-id.txt"
readonly COLD_CGROUP_DIR="$(resolve_container_cgroup)"
start_rss_sampler \
  "$COLD_CGROUP_DIR" \
  "$EVIDENCE_DIR/cold-process-rss.csv" \
  "$EVIDENCE_DIR/cold-process-rss-total.csv"
readonly READY_SECONDS="$(wait_for_readiness "$EVIDENCE_DIR/readiness.json" "$COLD_START_SECONDS")"
stop_rss_sampler
IFS=$'\t' read -r COLD_RSS_PEAK_BYTES COLD_RSS_MEAN_BYTES COLD_RSS_SAMPLES COLD_RSS_MAX_GAP_MS \
  <<<"$(summarize_rss_samples "$EVIDENCE_DIR/cold-process-rss-total.csv" 2)"
((COLD_RSS_PEAK_BYTES <= COLD_RSS_LIMIT_BYTES)) || release_die "cold-load process RSS exceeds 4.2 GiB"

printf '%s\n' \
  'wall_time_unix_ns,timestamp_utc,pid,process_start_ticks,process_name,rss_bytes' \
  >"$EVIDENCE_DIR/steady-process-rss.csv"
printf '%s\n' \
  'wall_time_unix_ns,timestamp_utc,listed_processes,sampled_processes,process_rss_sum_bytes,sample_complete' \
  >"$EVIDENCE_DIR/steady-process-rss-total.csv"
for ((steady_index = 0; steady_index < STEADY_SAMPLE_COUNT; steady_index++)); do
  cycle_started_ns=$(wall_time_unix_ns)
  sample_process_rss_cycle \
    "$COLD_CGROUP_DIR/cgroup.procs" /proc "$cycle_started_ns" "$SAMPLE_INTERVAL_NS" \
    "$EVIDENCE_DIR/steady-process-rss.csv" "$EVIDENCE_DIR/steady-process-rss-total.csv" 3
  sleep_until_next_sample "$cycle_started_ns"
done
IFS=$'\t' read -r STEADY_RSS_PEAK_BYTES STEADY_RSS_MEAN_BYTES STEADY_RSS_SAMPLES STEADY_RSS_MAX_GAP_MS \
  <<<"$(summarize_rss_samples "$EVIDENCE_DIR/steady-process-rss-total.csv" "$STEADY_SAMPLE_COUNT")"

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
  --data '{"text":"Guten Morgen. Dies ist der finale Streaming-Test.","voice_description":"A calm adult male voice with measured delivery.","language":"german","seed":42,"max_duration_seconds":12}' \
  http://127.0.0.1:18080/v1/voice-design/speech
validate_stream "$EVIDENCE_DIR/stream.headers" "$EVIDENCE_DIR/stream.multipart" true

curl --fail --silent --show-error --max-time 120 \
  --dump-header "$EVIDENCE_DIR/buffered.headers" \
  --output "$EVIDENCE_DIR/buffered.wav" \
  --header 'Content-Type: application/json' \
  --data '{"text":"Guten Morgen. Dies ist ein ruhiger Test der gepufferten Audioausgabe.","voice_description":"A calm adult male voice with measured delivery.","language":"german","seed":43,"max_duration_seconds":12,"stream":false}' \
  http://127.0.0.1:18080/v1/voice-design/speech
grep -Eiq '^content-type: audio/wav' "$EVIDENCE_DIR/buffered.headers" ||
  release_die "buffered response is not audio/wav"
grep -Eiq '^x-finish-reason: (stop|length)' "$EVIDENCE_DIR/buffered.headers" ||
  release_die "buffered response has no valid finish reason"
readonly BUFFERED_BYTES="$(stat -c '%s' "$EVIDENCE_DIR/buffered.wav")"
readonly DECLARED_BUFFERED_BYTES="$(awk -F: 'tolower($1) == "content-length" { gsub(/[[:space:]\r]/, "", $2); print $2; exit }' "$EVIDENCE_DIR/buffered.headers")"
[[ "$DECLARED_BUFFERED_BYTES" =~ ^[0-9]+$ && "$DECLARED_BUFFERED_BYTES" == "$BUFFERED_BYTES" ]] ||
  release_die "buffered WAV Content-Length does not match the retained file"
((BUFFERED_BYTES > 44)) || release_die "buffered WAV contains no PCM payload"

sox --i "$EVIDENCE_DIR/buffered.wav" >"$EVIDENCE_DIR/sox-info.txt"
readonly WAV_RATE="$(sox --i -r "$EVIDENCE_DIR/buffered.wav")"
readonly WAV_CHANNELS="$(sox --i -c "$EVIDENCE_DIR/buffered.wav")"
readonly WAV_BITS="$(sox --i -b "$EVIDENCE_DIR/buffered.wav")"
readonly WAV_ENCODING="$(sox --i -e "$EVIDENCE_DIR/buffered.wav")"
readonly WAV_SAMPLES="$(sox --i -s "$EVIDENCE_DIR/buffered.wav")"
[[ "$WAV_RATE" == 24000 && "$WAV_CHANNELS" == 1 && "$WAV_BITS" == 16 &&
  "$WAV_ENCODING" == "Signed Integer PCM" && "$WAV_SAMPLES" =~ ^[1-9][0-9]*$ ]] ||
  release_die "buffered WAV is not non-empty 24 kHz mono signed-16-bit PCM"
sox "$EVIDENCE_DIR/buffered.wav" -n stat 2>"$EVIDENCE_DIR/sox-stat.txt"
if grep -Eiq 'warn.*clipp|clipped' "$EVIDENCE_DIR/sox-stat.txt"; then
  release_die "SoX reported a clipping warning"
fi
readonly WAV_MIN_AMPLITUDE="$(awk -F: '$1 ~ /Minimum amplitude/ { gsub(/[[:space:]]/, "", $2); print $2; exit }' "$EVIDENCE_DIR/sox-stat.txt")"
readonly WAV_MAX_AMPLITUDE="$(awk -F: '$1 ~ /Maximum amplitude/ { gsub(/[[:space:]]/, "", $2); print $2; exit }' "$EVIDENCE_DIR/sox-stat.txt")"
readonly WAV_RMS_AMPLITUDE="$(awk -F: '$1 ~ /RMS[[:space:]]+amplitude/ { gsub(/[[:space:]]/, "", $2); print $2; exit }' "$EVIDENCE_DIR/sox-stat.txt")"
for amplitude in "$WAV_MIN_AMPLITUDE" "$WAV_MAX_AMPLITUDE" "$WAV_RMS_AMPLITUDE"; do
  [[ "$amplitude" =~ ^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] ||
    release_die "SoX reported a non-finite amplitude"
done
awk -v minimum="$WAV_MIN_AMPLITUDE" -v maximum="$WAV_MAX_AMPLITUDE" -v rms="$WAV_RMS_AMPLITUDE" \
  'BEGIN { exit !(minimum >= -1.0 && maximum <= 1.0 && rms > 0.0) }' ||
  release_die "buffered WAV signal is empty or outside finite PCM amplitude bounds"

readonly CANCELLATION_ID=0198f65d-a679-7411-8f7c-151dbf0486be
(
  set +e
  curl --silent --show-error --no-buffer --max-time 120 \
    --dump-header "$EVIDENCE_DIR/cancellation-stream.headers" \
    --output "$EVIDENCE_DIR/cancellation-stream.multipart" \
    --header 'Content-Type: application/json' \
    --header "x-request-id: $CANCELLATION_ID" \
    --data '{"text":"Read this extended technical report slowly and clearly. Describe architecture, deployment controls, monitoring, failure recovery, benchmarking, release procedures, and operational safeguards in detail.","voice_description":"A calm technical narrator with clear and unhurried delivery.","language":"english","seed":44,"max_duration_seconds":120}' \
    http://127.0.0.1:18080/v1/voice-design/speech
  status=$?
  printf '%s\n' "$status" >"$EVIDENCE_DIR/cancellation-stream.curl-status"
  exit "$status"
) &
cancellation_pid=$!
cancel_stream_ready=0
for ((attempt = 0; attempt < 200; attempt++)); do
  if grep -Eq '^HTTP/[^ ]+ 200' "$EVIDENCE_DIR/cancellation-stream.headers" 2>/dev/null; then
    cancel_stream_ready=1
    break
  fi
  kill -0 "$cancellation_pid" >/dev/null 2>&1 || break
  sleep 0.1
done
((cancel_stream_ready == 1)) || release_die "cancellation test stream did not become active"
readonly CANCELLATION_HTTP_STATUS="$(curl --silent --show-error --max-time 5 \
  --request DELETE \
  --output "$EVIDENCE_DIR/cancellation-response.json" \
  --write-out '%{http_code}' \
  "http://127.0.0.1:18080/v1/requests/$CANCELLATION_ID")"
[[ "$CANCELLATION_HTTP_STATUS" == 202 ]] || release_die "active cancellation did not return HTTP 202"
jq -e --arg request_id "$CANCELLATION_ID" \
  '.request_id == $request_id and .status == "cancellation_requested"' \
  "$EVIDENCE_DIR/cancellation-response.json" >/dev/null || release_die "cancellation response contract failed"
for ((attempt = 0; attempt < 300; attempt++)); do
  kill -0 "$cancellation_pid" >/dev/null 2>&1 || break
  sleep 0.1
done
if kill -0 "$cancellation_pid" >/dev/null 2>&1; then
  release_die "cancelled request did not terminate within 30 seconds"
fi
wait "$cancellation_pid" || release_die "cancelled streaming client did not finish cleanly"
cancellation_pid=0
grep -aFq '"type":"error"' "$EVIDENCE_DIR/cancellation-stream.multipart" ||
  release_die "cancelled stream has no terminal error event"
grep -aFq '"code":"request_cancelled"' "$EVIDENCE_DIR/cancellation-stream.multipart" ||
  release_die "cancelled stream has no request_cancelled error"

readonly -a LANGUAGES=(auto chinese english japanese korean german french russian portuguese spanish italian)
readonly -a LANGUAGE_TEXTS=(
  'This is a calm automatic-language synthesis check.'
  '你好，这是一次平静的中文语音测试。'
  'This is a calm English speech synthesis check.'
  'こんにちは。これは落ち着いた日本語の音声テストです。'
  '안녕하세요. 이것은 차분한 한국어 음성 테스트입니다.'
  'Guten Tag. Dies ist ein ruhiger deutscher Sprachtest.'
  'Bonjour. Ceci est un test vocal français calme.'
  'Здравствуйте. Это спокойная проверка русской речи.'
  'Olá. Este é um teste calmo de voz em português.'
  'Hola. Esta es una prueba tranquila de voz en español.'
  'Buongiorno. Questa è una prova vocale italiana calma.'
)
install -d -m 0755 "$EVIDENCE_DIR/languages"
for index in "${!LANGUAGES[@]}"; do
  language=${LANGUAGES[$index]}
  payload=$(jq -nc \
    --arg text "${LANGUAGE_TEXTS[$index]}" \
    --arg language "$language" \
    --argjson seed "$((100 + index))" \
    '{text: $text, voice_description: "A calm adult narrator with measured delivery.", language: $language, seed: $seed, max_duration_seconds: 12}')
  curl --fail --silent --show-error --no-buffer --max-time 120 \
    --dump-header "$EVIDENCE_DIR/languages/$language.headers" \
    --output "$EVIDENCE_DIR/languages/$language.multipart" \
    --header 'Content-Type: application/json' \
    --data "$payload" \
    http://127.0.0.1:18080/v1/voice-design/speech
  validate_stream \
    "$EVIDENCE_DIR/languages/$language.headers" \
    "$EVIDENCE_DIR/languages/$language.multipart" \
    true
done

curl --fail --silent --show-error \
  --dump-header "$EVIDENCE_DIR/metrics.headers" \
  http://127.0.0.1:18080/metrics >"$EVIDENCE_DIR/metrics.prom"
grep -Eiq '^content-type: text/plain; version=0.0.4; charset=utf-8' "$EVIDENCE_DIR/metrics.headers" ||
  release_die "metrics endpoint has an unexpected content type"
readonly -a METRIC_NAMES=(
  qwen3_tts_http_requests_total
  qwen3_tts_active_requests
  qwen3_tts_streaming_requests_total
  qwen3_tts_buffered_requests_total
  qwen3_tts_completed_requests_total
  qwen3_tts_failed_requests_total
  qwen3_tts_cancelled_requests_total
  qwen3_tts_rejected_requests_total
  qwen3_tts_retirement_timeouts_total
  qwen3_tts_emitted_samples_total
)
for metric_name in "${METRIC_NAMES[@]}"; do
  [[ "$(awk -v name="$metric_name" '$1 == name { count++ } END { print count + 0 }' "$EVIDENCE_DIR/metrics.prom")" == 1 ]] ||
    release_die "metrics endpoint is missing or duplicates $metric_name"
done
[[ "$(awk '/^#/ { next } NF > 0 { count++ } END { print count + 0 }' "$EVIDENCE_DIR/metrics.prom")" == "${#METRIC_NAMES[@]}" ]] ||
  release_die "metrics endpoint exposes samples outside the ten approved metric names"
awk '
  /^#/ { next }
  /\{/ { exit 2 }
  NF != 2 { exit 2 }
  $1 !~ /^[A-Za-z_:][A-Za-z0-9_:]*$/ { exit 2 }
  $2 !~ /^[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$/ { exit 2 }
' "$EVIDENCE_DIR/metrics.prom" || release_die "metrics contain labels or malformed samples"
if grep -Eiq 'voice_description|request[_-]?id|language|Guten Morgen|calm adult' "$EVIDENCE_DIR/metrics.prom"; then
  release_die "metrics contain prompt, request, voice, or language material"
fi
[[ "$(awk '$1 == "qwen3_tts_active_requests" { print $2 }' "$EVIDENCE_DIR/metrics.prom")" == 0 ]] ||
  release_die "a request remained active after clean-pull acceptance"
awk '$1 == "qwen3_tts_cancelled_requests_total" { found = 1; if ($2 < 1) exit 2 } END { exit !found }' \
  "$EVIDENCE_DIR/metrics.prom" || release_die "cancellation was not reflected in metrics"

docker --config "$ANON_CONFIG" inspect --size "$SHORT_CONTAINER_NAME" >"$EVIDENCE_DIR/container-inspect.json"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used \
  --format=csv,noheader,nounits >"$EVIDENCE_DIR/nvidia-smi.csv"
readonly FIRST_STOP_SECONDS="$(graceful_stop_and_remove initial)"

readonly RESTART_START_SECONDS=$SECONDS
start_container "$EVIDENCE_DIR/restart-container-id.txt"
readonly RESTART_READY_SECONDS="$(wait_for_readiness "$EVIDENCE_DIR/restart-readiness.json" "$RESTART_START_SECONDS")"
readonly RESTART_STOP_SECONDS="$(graceful_stop_and_remove restart)"

readonly LANGUAGES_JSON="$(printf '%s\n' "${LANGUAGES[@]}" | jq -R . | jq -s .)"
jq -n \
  --arg digest "$DIGEST" \
  --argjson ready_seconds "$READY_SECONDS" \
  --argjson restart_ready_seconds "$RESTART_READY_SECONDS" \
  --argjson unpacked_bytes "$UNPACKED_BYTES" \
  --argjson cold_limit "$COLD_RSS_LIMIT_BYTES" \
  --argjson cold_peak "$COLD_RSS_PEAK_BYTES" \
  --argjson cold_mean "$COLD_RSS_MEAN_BYTES" \
  --argjson cold_samples "$COLD_RSS_SAMPLES" \
  --argjson cold_gap "$COLD_RSS_MAX_GAP_MS" \
  --argjson steady_peak "$STEADY_RSS_PEAK_BYTES" \
  --argjson steady_mean "$STEADY_RSS_MEAN_BYTES" \
  --argjson steady_samples "$STEADY_RSS_SAMPLES" \
  --argjson steady_gap "$STEADY_RSS_MAX_GAP_MS" \
  --argjson buffered_bytes "$BUFFERED_BYTES" \
  --argjson wav_samples "$WAV_SAMPLES" \
  --argjson first_stop_seconds "$FIRST_STOP_SECONDS" \
  --argjson restart_stop_seconds "$RESTART_STOP_SECONDS" \
  --argjson languages "$LANGUAGES_JSON" '{
    schema: "qwen3-tts-native/clean-pull-gpu-acceptance/v2",
    digest: $digest,
    pull: "anonymous-empty-store",
    hardened_runtime: "passed",
    gpu: "passed",
    streaming_pcm: "passed",
    readiness_seconds: $ready_seconds,
    unpacked_bytes: $unpacked_bytes,
    cold_start: {
      status: "passed",
      process_rss_definition: "peak 100 ms sampled sum of VmRSS for every process in the container cgroup from process start through first readiness",
      process_rss_peak_bytes: $cold_peak,
      process_rss_mean_bytes: $cold_mean,
      process_rss_limit_bytes: $cold_limit,
      samples: $cold_samples,
      sample_interval_ms: 100,
      maximum_observed_gap_ms: $cold_gap
    },
    post_ready_steady_rss: {
      status: "measured_for_review",
      process_rss_peak_bytes: $steady_peak,
      process_rss_mean_bytes: $steady_mean,
      samples: $steady_samples,
      sample_interval_ms: 100,
      maximum_observed_gap_ms: $steady_gap
    },
    buffered_wav: {
      status: "passed",
      bytes: $buffered_bytes,
      sample_rate_hz: 24000,
      channels: 1,
      bits_per_sample: 16,
      encoding: "signed_integer_pcm",
      samples: $wav_samples,
      finite_signal: "passed",
      non_empty_signal: "passed",
      sox_clipping_warning: "none"
    },
    cancellation: "passed",
    prompt_free_metrics: "passed",
    language_natural_eos: {
      status: "passed",
      completed: ($languages | length),
      languages: $languages
    },
    restart: {status: "passed", readiness_seconds: $restart_ready_seconds},
    graceful_sigterm: {
      status: "passed",
      initial_stop_seconds: $first_stop_seconds,
      restart_stop_seconds: $restart_stop_seconds,
      exit_code: 0
    }
  }' >"$EVIDENCE_DIR/gpu-acceptance.json"

(
  cd "$EVIDENCE_DIR"
  while IFS= read -r -d '' path; do
    sha256sum "${path#./}"
  done < <(find . -type f ! -name SHA256SUMS -print0 | sort -z)
) >"$EVIDENCE_DIR/SHA256SUMS"
(
  cd "$EVIDENCE_DIR"
  sha256sum --check --strict SHA256SUMS >/dev/null
)

printf 'Anonymous clean-pull and enhanced GPU acceptance passed for %s\n' "$REFERENCE"
