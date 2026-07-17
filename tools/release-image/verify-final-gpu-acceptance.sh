#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

[[ $# -eq 5 ]] || release_die \
  "usage: $0 RELEASE_RECORD CLEAN_GPU_RECEIPT NATIVE_B1_RUN_DIR NATIVE_B6_RUN_DIR EVIDENCE_DIR"
readonly RELEASE_RECORD=$1
readonly CLEAN_GPU_RECEIPT=$2
readonly B1_RUN_DIR=$3
readonly B6_RUN_DIR=$4
readonly EVIDENCE_DIR=$5

release_read_record "$RELEASE_RECORD"
release_require_file "$CLEAN_GPU_RECEIPT"
release_require_directory "$B1_RUN_DIR"
release_require_directory "$B6_RUN_DIR"

for command_name in awk cmp find grep jq mktemp sed sha256sum sox sort stat uniq wc; do
  release_require_command "$command_name"
done

readonly DIGEST="$(release_record_value "$RELEASE_RECORD" '.digest')"
readonly VERSION="$(release_record_value "$RELEASE_RECORD" '.release_version')"
readonly SOURCE_REVISION="$(release_record_value "$RELEASE_RECORD" '.source_revision')"
readonly REFERENCE="$RELEASE_IMAGE@$DIGEST"
readonly CLEAN_EVIDENCE_DIR="$(cd "$(dirname "$CLEAN_GPU_RECEIPT")" && pwd)"
readonly B6_GPU_UNIFIED_MEMORY_LIMIT_BYTES=6000000000
readonly COLD_RSS_LIMIT_BYTES=4509715660
readonly MAXIMUM_TELEMETRY_GAP_NS=200000000
readonly WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/qwen3-tts-final-gpu.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

require_regular_file() {
  local path=$1
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] ||
    release_die "required non-empty regular evidence file is missing or unsafe: $path"
}

require_regular_file_allow_empty() {
  local path=$1
  [[ -f "$path" && ! -L "$path" ]] ||
    release_die "required regular evidence file is missing or unsafe: $path"
}

validate_complete_inventory() {
  local directory=$1
  local label=$2
  local inventory="$directory/SHA256SUMS"
  local actual="$WORK_DIR/$label.actual"
  local listed="$WORK_DIR/$label.listed"
  local duplicates="$WORK_DIR/$label.duplicates"

  [[ -d "$directory" && ! -L "$directory" ]] || release_die "$label evidence directory is unsafe"
  require_regular_file "$inventory"
  [[ -z "$(find "$directory" -type l -print -quit)" ]] || release_die "$label evidence contains a symlink"
  (
    cd "$directory"
    sha256sum --check --strict SHA256SUMS >/dev/null
  ) || release_die "$label SHA256SUMS verification failed"

  (
    cd "$directory"
    find . -type f ! -name SHA256SUMS -print | sed 's#^\./##' | LC_ALL=C sort
  ) >"$actual"
  awk '
    {
      if ($0 !~ /^[0-9a-f]+  / || length($1) != 64) exit 2
      path = substr($0, 67)
      if (path == "" || path ~ /^\// || path ~ /(^|\/)\.\.($|\/)/ || path ~ /\\/) exit 2
      print path
    }
  ' "$inventory" | LC_ALL=C sort >"$listed" || release_die "$label SHA256SUMS contains an unsafe entry"
  uniq -d "$listed" >"$duplicates"
  [[ ! -s "$duplicates" ]] || release_die "$label SHA256SUMS contains duplicate paths"
  cmp -s "$actual" "$listed" || release_die "$label SHA256SUMS is not a complete file inventory"
}

validate_image_inspect() {
  local inspect_path=$1
  local resolved_id=${2:-}
  require_regular_file "$inspect_path"
  jq -e \
    --arg reference "$REFERENCE" \
    --arg revision "$SOURCE_REVISION" \
    --arg version "$VERSION" \
    --arg resolved_id "$resolved_id" '
      type == "array" and length == 1
      and .[0].Os == "linux" and .[0].Architecture == "arm64"
      and .[0].Config.User == "10001:10001"
      and (.[0].RepoDigests | type == "array" and index($reference) != null)
      and .[0].Config.Labels["org.opencontainers.image.revision"] == $revision
      and .[0].Config.Labels["org.opencontainers.image.version"] == $version
      and .[0].Config.Labels["io.qwen3-tts.model.voice-clone"] == "false"
      and ($resolved_id == "" or .[0].Id == $resolved_id)
    ' "$inspect_path" >/dev/null || release_die "image inspection is not bound to the release digest and labels: $inspect_path"
}

validate_gap_window() {
  local path=$1
  local start_ns=$2
  local end_ns=$3
  [[ "$start_ns" =~ ^[0-9]+$ && "$end_ns" =~ ^[0-9]+$ && "$end_ns" -gt "$start_ns" ]] ||
    release_die "invalid telemetry window for $path"
  awk -F, -v start="$start_ns" -v end="$end_ns" -v maximum="$MAXIMUM_TELEMETRY_GAP_NS" '
    NR == 1 { next }
    {
      if ($1 !~ /^[0-9]+$/) exit 2
      current = $1 + 0
      if (count > 0 && current <= previous) exit 2
      if (count > 0 && previous <= end && current >= start) {
        gap = current - previous
        if (gap > maximum) exit 2
        if (gap > largest) largest = gap
      }
      if (count == 0) first = current
      previous = current
      count++
    }
    END {
      if (count < 2 || first > start || previous < end) exit 2
      printf "%.3f\n", largest / 1000000
    }
  ' "$path" || release_die "telemetry does not bracket its window or exceeds a 200 ms gap: $path"
}

validate_audit_sources() {
  local run_dir=$1
  local audit_path="$run_dir/resource-audit.json"
  local row path expected_sha expected_bytes actual_sha actual_bytes
  while IFS= read -r row; do
    path=$(jq -er '.path' <<<"$row")
    expected_sha=$(jq -er '.sha256' <<<"$row")
    expected_bytes=$(jq -er '.bytes' <<<"$row")
    [[ "$path" != /* && "$path" != *..* && "$path" != *\\* ]] ||
      release_die "resource audit contains an unsafe source path"
    require_regular_file_allow_empty "$run_dir/$path"
    actual_sha=$(sha256sum "$run_dir/$path" | awk '{ print $1 }')
    actual_bytes=$(wc -c <"$run_dir/$path" | awk '{ print $1 }')
    [[ "$actual_sha" == "$expected_sha" && "$actual_bytes" == "$expected_bytes" ]] ||
      release_die "resource audit source identity differs from retained evidence: $path"
  done < <(jq -ec '.source_files[]' "$audit_path")
}

validate_clean_acceptance() {
  local metric_name
  local -a expected_metric_names=(
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
  [[ "$(basename "$CLEAN_GPU_RECEIPT")" == gpu-acceptance.json ]] ||
    release_die "clean GPU receipt must be named gpu-acceptance.json"
  validate_complete_inventory "$CLEAN_EVIDENCE_DIR" clean
  jq -e --arg digest "$DIGEST" '
    .schema == "qwen3-tts-native/clean-pull-gpu-acceptance/v2"
    and .digest == $digest
    and .pull == "anonymous-empty-store"
    and .hardened_runtime == "passed"
    and .gpu == "passed"
    and .streaming_pcm == "passed"
    and (.readiness_seconds | type == "number" and . <= 20)
    and (.unpacked_bytes | type == "number" and . > 0 and . <= 10000000000)
    and .cold_start.status == "passed"
    and .cold_start.process_rss_limit_bytes == 4509715660
    and (.cold_start.process_rss_peak_bytes | type == "number")
    and .cold_start.process_rss_peak_bytes > 0
    and .cold_start.process_rss_peak_bytes <= .cold_start.process_rss_limit_bytes
    and .cold_start.sample_interval_ms == 100
    and (.cold_start.maximum_observed_gap_ms | type == "number" and . <= 200)
    and .post_ready_steady_rss.status == "measured_for_review"
    and (.post_ready_steady_rss.process_rss_peak_bytes | type == "number" and . > 0)
    and (.post_ready_steady_rss.process_rss_mean_bytes | type == "number" and . > 0)
    and (.post_ready_steady_rss.samples | type == "number" and . >= 10)
    and .post_ready_steady_rss.sample_interval_ms == 100
    and (.post_ready_steady_rss.maximum_observed_gap_ms | type == "number" and . <= 200)
    and .buffered_wav.status == "passed"
    and .buffered_wav.sample_rate_hz == 24000
    and .buffered_wav.channels == 1
    and .buffered_wav.bits_per_sample == 16
    and .buffered_wav.encoding == "signed_integer_pcm"
    and (.buffered_wav.samples | type == "number" and . > 0)
    and .buffered_wav.finite_signal == "passed"
    and .buffered_wav.non_empty_signal == "passed"
    and .buffered_wav.sox_clipping_warning == "none"
    and .cancellation == "passed"
    and .prompt_free_metrics == "passed"
    and .language_natural_eos.status == "passed"
    and .language_natural_eos.completed == 11
    and .language_natural_eos.languages == [
      "auto", "chinese", "english", "japanese", "korean", "german",
      "french", "russian", "portuguese", "spanish", "italian"
    ]
    and .restart.status == "passed"
    and (.restart.readiness_seconds | type == "number" and . <= 20)
    and .graceful_sigterm.status == "passed"
    and .graceful_sigterm.exit_code == 0
  ' "$CLEAN_GPU_RECEIPT" >/dev/null || release_die "enhanced clean-pull receipt does not authorize final GPU acceptance"

  validate_image_inspect "$CLEAN_EVIDENCE_DIR/image-inspect.json"
  for path in \
    readiness.json restart-readiness.json buffered.wav sox-info.txt sox-stat.txt \
    cancellation-response.json cancellation-stream.multipart metrics.prom \
    cold-process-rss-total.csv steady-process-rss-total.csv \
    initial-stopped-inspect.json restart-stopped-inspect.json; do
    require_regular_file "$CLEAN_EVIDENCE_DIR/$path"
  done
  jq -e '.status == "ready" and .engine_loaded == true' "$CLEAN_EVIDENCE_DIR/readiness.json" >/dev/null ||
    release_die "initial readiness evidence is invalid"
  jq -e '.status == "ready" and .engine_loaded == true' "$CLEAN_EVIDENCE_DIR/restart-readiness.json" >/dev/null ||
    release_die "restart readiness evidence is invalid"
  jq -e '.[0].State.Status == "exited" and .[0].State.ExitCode == 0 and .[0].State.OOMKilled == false' \
    "$CLEAN_EVIDENCE_DIR/initial-stopped-inspect.json" >/dev/null || release_die "initial SIGTERM evidence is invalid"
  jq -e '.[0].State.Status == "exited" and .[0].State.ExitCode == 0 and .[0].State.OOMKilled == false' \
    "$CLEAN_EVIDENCE_DIR/restart-stopped-inspect.json" >/dev/null || release_die "restart SIGTERM evidence is invalid"
  jq -e '.status == "cancellation_requested"' "$CLEAN_EVIDENCE_DIR/cancellation-response.json" >/dev/null ||
    release_die "cancellation evidence is invalid"
  grep -aFq '"code":"request_cancelled"' "$CLEAN_EVIDENCE_DIR/cancellation-stream.multipart" ||
    release_die "cancelled stream evidence has no request_cancelled event"
  if grep -Eiq 'voice_description|request[_-]?id|language|Guten Morgen|calm adult|\{' "$CLEAN_EVIDENCE_DIR/metrics.prom"; then
    release_die "retained metrics evidence contains labels or prompt/request material"
  fi
  for metric_name in "${expected_metric_names[@]}"; do
    [[ "$(awk -v name="$metric_name" '$1 == name { count++ } END { print count + 0 }' \
      "$CLEAN_EVIDENCE_DIR/metrics.prom")" == 1 ]] ||
      release_die "retained metrics evidence is missing or duplicates $metric_name"
  done
  [[ "$(awk '/^#/ { next } NF > 0 { count++ } END { print count + 0 }' \
    "$CLEAN_EVIDENCE_DIR/metrics.prom")" == "${#expected_metric_names[@]}" ]] ||
    release_die "retained metrics evidence has samples outside the ten approved names"
  for language in auto chinese english japanese korean german french russian portuguese spanish italian; do
    require_regular_file "$CLEAN_EVIDENCE_DIR/languages/$language.headers"
    require_regular_file "$CLEAN_EVIDENCE_DIR/languages/$language.multipart"
    grep -aFq 'audio/pcm;rate=24000;channels=1;format=s16le' \
      "$CLEAN_EVIDENCE_DIR/languages/$language.multipart" || release_die "$language smoke contains no PCM"
    grep -aFq '"finish_reason":"stop"' "$CLEAN_EVIDENCE_DIR/languages/$language.multipart" ||
      release_die "$language smoke is not natural EOS: $language"
  done

  [[ "$(sox --i -r "$CLEAN_EVIDENCE_DIR/buffered.wav")" == 24000 ]] || release_die "retained WAV rate changed"
  [[ "$(sox --i -c "$CLEAN_EVIDENCE_DIR/buffered.wav")" == 1 ]] || release_die "retained WAV channels changed"
  [[ "$(sox --i -b "$CLEAN_EVIDENCE_DIR/buffered.wav")" == 16 ]] || release_die "retained WAV depth changed"
  [[ "$(sox --i -e "$CLEAN_EVIDENCE_DIR/buffered.wav")" == "Signed Integer PCM" ]] ||
    release_die "retained WAV encoding changed"
  sox "$CLEAN_EVIDENCE_DIR/buffered.wav" -n stat 2>"$WORK_DIR/sox-recheck.txt"
  if grep -Eiq 'warn.*clipp|clipped' "$WORK_DIR/sox-recheck.txt"; then
    release_die "retained WAV produces a SoX clipping warning"
  fi
}

validate_run() {
  local run_dir=$1
  local profile=$2
  local minimum_requests=$3
  local exact_requests=$4
  local label=${profile,,}
  local invocation="$run_dir/provenance/invocation.json"
  local image_inspect="$run_dir/provenance/image-inspect.json"
  local container_inspect="$run_dir/provenance/container-inspect.sanitized.json"
  local summary="$run_dir/client/summary.json"
  local requests="$run_dir/client/requests.jsonl"
  local resource="$run_dir/run-resource.json"
  local audit="$run_dir/resource-audit.json"
  local image_id planned_requests warmups aggregate_rtf ttfa_p95
  local process_rss_peak gpu_memory_peak internal_device_peak internal_host_peak
  local idle_start idle_end measured_start measured_end max_gap

  validate_complete_inventory "$run_dir" "$label"
  for path in "$invocation" "$image_inspect" "$container_inspect" "$summary" "$requests" "$resource" "$audit" \
    "$run_dir/raw/run.txt" "$run_dir/raw/gpu.csv" "$run_dir/raw/system.csv" \
    "$run_dir/raw/process-rss-total.csv" "$run_dir/raw/gpu-process-summary.csv"; do
    require_regular_file "$path"
  done

  jq -e \
    --arg profile "$profile" \
    --arg reference "$REFERENCE" \
    --argjson minimum_requests "$minimum_requests" \
    --argjson exact_requests "$exact_requests" '
      .schema_version == "qwen3-tts-qualifying-run/v1"
      and .engine == "native" and .profile == $profile
      and (.round | type == "number" and .round >= 1)
      and .image.reference == $reference
      and (.image.resolved_id | test("^sha256:[0-9a-f]{64}$"))
      and .request.endpoint == "http://127.0.0.1:8080/v1/voice-design/speech"
      and .request.warmups == 24
      and (if $exact_requests then .request.requests == $minimum_requests else .request.requests >= $minimum_requests end)
      and .telemetry.configured_sample_interval_ms == 100
      and .telemetry.maximum_qualifying_observed_gap_ms == 200
      and .tooling_repository.tracked_files_clean == true
    ' "$invocation" >/dev/null || release_die "$profile invocation is not digest-specific qualifying evidence"
  image_id=$(jq -er '.image.resolved_id' "$invocation")
  validate_image_inspect "$image_inspect" "$image_id"
  jq -e --arg image_id "$image_id" '
    type == "array" and length == 1 and .[0].Image == $image_id
    and .[0].State.Running == true and .[0].HostConfig.ReadonlyRootfs == true
  ' "$container_inspect" >/dev/null || release_die "$profile container evidence is not bound to the tested image"

  jq -e \
    --arg profile "$profile" \
    --argjson minimum_requests "$minimum_requests" \
    --argjson exact_requests "$exact_requests" '
      .schema_version == "qwen3-tts-http-bench/v1"
      and .backend == "native" and .concurrency == $profile
      and .warmups == 24
      and (if $exact_requests then .planned_requests == $minimum_requests else .planned_requests >= $minimum_requests end)
      and .completed_requests == .planned_requests
      and .successful_requests == .planned_requests
      and .failed_requests == 0
      and .natural_eos_requests == .successful_requests
      and .length_limited_requests == 0 and .eos_unknown_requests == 0
      and .sampling_parity_qualifying_requests == .completed_requests
      and .sampling_parity_non_qualifying_requests == 0
      and (.aggregate_rtf | type == "number" and .aggregate_rtf > 0)
      and (.ttfa_ms.p95 | type == "number" and .ttfa_ms.p95 >= 0)
    ' "$summary" >/dev/null || release_die "$profile client summary is not complete natural-EOS evidence"
  planned_requests=$(jq -er '.planned_requests' "$summary")
  warmups=$(jq -er '.warmups' "$summary")
  aggregate_rtf=$(jq -er '.aggregate_rtf' "$summary")
  ttfa_p95=$(jq -er '.ttfa_ms.p95' "$summary")
  if [[ "$profile" == B1 ]]; then
    awk -v rtf="$aggregate_rtf" -v ttfa="$ttfa_p95" 'BEGIN { exit !(rtf < 1.0 && ttfa < 200.0) }' ||
      release_die "B1 does not satisfy aggregate RTF < 1 and TTFA p95 < 200 ms"
  fi

  jq -e -s --argjson expected "$planned_requests" '
    length == $expected
    and all(.[ ];
      .backend == "native" and .success == true and .streaming == true
      and .finish_reason == "stop" and .natural_eos == true and .length_limited == false
      and .failure == null and .continuity_valid == true and .final_flag_seen == true
      and (.end_metrics.peak_request_device_bytes | type == "number" and . >= 0 and . == floor)
      and (.end_metrics.peak_request_host_bytes | type == "number" and . >= 0 and . == floor)
    )
  ' "$requests" >/dev/null || release_die "$profile request records are incomplete or lack internal memory fields"
  internal_device_peak=$(jq -ers '[.[].end_metrics.peak_request_device_bytes] | max' "$requests")
  internal_host_peak=$(jq -ers '[.[].end_metrics.peak_request_host_bytes] | max' "$requests")

  jq -e --arg profile "$profile" --argjson round "$(jq -er '.round' "$invocation")" '
    .engine_id == "native" and .profile_id == $profile and .round == $round
    and (.process_rss_peak_bytes | type == "number" and . > 0)
    and (.gpu_unified_memory_peak_bytes | type == "number" and . > 0)
    and .sampling_interval_ms == 100
    and .competing_cuda_processes == 0
  ' "$resource" >/dev/null || release_die "$profile run-resource does not satisfy telemetry identity and isolation"
  process_rss_peak=$(jq -er '.process_rss_peak_bytes' "$resource")
  gpu_memory_peak=$(jq -er '.gpu_unified_memory_peak_bytes' "$resource")
  if [[ "$profile" == B6 ]]; then
    ((gpu_memory_peak <= B6_GPU_UNIFIED_MEMORY_LIMIT_BYTES)) ||
      release_die "B6 observed GPU unified-memory peak exceeds 6,000,000,000 bytes"
  fi

  jq -e \
    --arg profile "$profile" \
    --argjson round "$(jq -er '.round' "$invocation")" \
    --argjson rss "$process_rss_peak" \
    --argjson gpu "$gpu_memory_peak" '
      .schema_version == "qwen3-tts-spark-resource-audit/v1"
      and .engine_id == "native" and .profile_id == $profile and .round == $round
      and .sampling.configured_interval_ms == 100
      and .sampling.maximum_allowed_observed_gap_ms == 200
      and .memory.process_rss_peak_bytes == $rss
      and .memory.gpu_unified_memory_peak_bytes == $gpu
      and (.source_files | type == "array" and length > 0)
    ' "$audit" >/dev/null || release_die "$profile resource audit differs from run-resource"
  validate_audit_sources "$run_dir"
  [[ "$(grep -Fxc 'sample_interval_ms=100' "$run_dir/raw/run.txt")" == 1 ]] ||
    release_die "$profile raw telemetry did not use a 100 ms interval"
  [[ "$(grep -Fxc 'maximum_qualifying_gap_ms=200' "$run_dir/raw/run.txt")" == 1 ]] ||
    release_die "$profile raw telemetry does not declare the 200 ms maximum gap"

  idle_start=$(jq -er '.phase_boundaries.idle_start_wall_time_unix_ns' "$audit")
  idle_end=$(jq -er '.phase_boundaries.idle_end_wall_time_unix_ns' "$audit")
  measured_start=$(jq -er '.phase_boundaries.measured_start_wall_time_unix_ns' "$audit")
  measured_end=$(jq -er '.phase_boundaries.measured_end_wall_time_unix_ns' "$audit")
  max_gap=$(printf '%s\n' \
    "$(validate_gap_window "$run_dir/raw/gpu.csv" "$idle_start" "$idle_end")" \
    "$(validate_gap_window "$run_dir/raw/gpu.csv" "$measured_start" "$measured_end")" \
    "$(validate_gap_window "$run_dir/raw/process-rss-total.csv" "$measured_start" "$measured_end")" \
    "$(validate_gap_window "$run_dir/raw/gpu-process-summary.csv" "$idle_start" "$measured_end")" \
    "$(validate_gap_window "$run_dir/raw/system.csv" "$measured_start" "$measured_end")" |
    awk '$1 > maximum { maximum = $1 } END { printf "%.3f", maximum }')

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$planned_requests" "$warmups" "$aggregate_rtf" "$ttfa_p95" \
    "$process_rss_peak" "$gpu_memory_peak" "$internal_device_peak" "$internal_host_peak" "$max_gap"
}

validate_clean_acceptance
IFS=$'\t' read -r B1_REQUESTS B1_WARMUPS B1_AGGREGATE_RTF B1_TTFA_P95 \
  B1_PROCESS_RSS B1_GPU_MEMORY B1_INTERNAL_DEVICE B1_INTERNAL_HOST B1_MAX_GAP \
  <<<"$(validate_run "$B1_RUN_DIR" B1 200 true)"
IFS=$'\t' read -r B6_REQUESTS B6_WARMUPS B6_AGGREGATE_RTF B6_TTFA_P95 \
  B6_PROCESS_RSS B6_GPU_MEMORY B6_INTERNAL_DEVICE B6_INTERNAL_HOST B6_MAX_GAP \
  <<<"$(validate_run "$B6_RUN_DIR" B6 240 false)"

release_require_new_directory "$EVIDENCE_DIR"
readonly CLEAN_RECEIPT_SHA256="$(release_sha256_file "$CLEAN_GPU_RECEIPT")"
readonly B1_INVENTORY_SHA256="$(release_sha256_file "$B1_RUN_DIR/SHA256SUMS")"
readonly B6_INVENTORY_SHA256="$(release_sha256_file "$B6_RUN_DIR/SHA256SUMS")"
readonly COLD_RSS_PEAK="$(jq -er '.cold_start.process_rss_peak_bytes' "$CLEAN_GPU_RECEIPT")"
readonly STEADY_RSS_PEAK="$(jq -er '.post_ready_steady_rss.process_rss_peak_bytes' "$CLEAN_GPU_RECEIPT")"
readonly STEADY_RSS_MEAN="$(jq -er '.post_ready_steady_rss.process_rss_mean_bytes' "$CLEAN_GPU_RECEIPT")"

jq -n \
  --arg digest "$DIGEST" --arg version "$VERSION" --arg source_revision "$SOURCE_REVISION" \
  --arg clean_receipt_sha256 "$CLEAN_RECEIPT_SHA256" \
  --arg b1_inventory_sha256 "$B1_INVENTORY_SHA256" --arg b6_inventory_sha256 "$B6_INVENTORY_SHA256" \
  --argjson cold_rss_peak "$COLD_RSS_PEAK" --argjson cold_rss_limit "$COLD_RSS_LIMIT_BYTES" \
  --argjson steady_rss_peak "$STEADY_RSS_PEAK" --argjson steady_rss_mean "$STEADY_RSS_MEAN" \
  --argjson b1_requests "$B1_REQUESTS" --argjson b1_warmups "$B1_WARMUPS" \
  --argjson b1_rtf "$B1_AGGREGATE_RTF" --argjson b1_ttfa "$B1_TTFA_P95" \
  --argjson b1_rss "$B1_PROCESS_RSS" --argjson b1_gpu "$B1_GPU_MEMORY" \
  --argjson b1_internal_device "$B1_INTERNAL_DEVICE" --argjson b1_internal_host "$B1_INTERNAL_HOST" \
  --argjson b1_gap "$B1_MAX_GAP" \
  --argjson b6_requests "$B6_REQUESTS" --argjson b6_warmups "$B6_WARMUPS" \
  --argjson b6_rtf "$B6_AGGREGATE_RTF" --argjson b6_ttfa "$B6_TTFA_P95" \
  --argjson b6_rss "$B6_PROCESS_RSS" --argjson b6_gpu "$B6_GPU_MEMORY" \
  --argjson b6_gpu_limit "$B6_GPU_UNIFIED_MEMORY_LIMIT_BYTES" \
  --argjson b6_internal_device "$B6_INTERNAL_DEVICE" --argjson b6_internal_host "$B6_INTERNAL_HOST" \
  --argjson b6_gap "$B6_MAX_GAP" '{
    schema: "qwen3-tts-native/final-gpu-acceptance/v1",
    status: "passed",
    digest: $digest,
    release_version: $version,
    source_revision: $source_revision,
    evidence: {
      clean_pull_receipt_sha256: $clean_receipt_sha256,
      b1_sha256sums_sha256: $b1_inventory_sha256,
      b6_sha256sums_sha256: $b6_inventory_sha256
    },
    clean_acceptance: {
      cold_start_process_rss_peak_bytes: $cold_rss_peak,
      cold_start_process_rss_limit_bytes: $cold_rss_limit,
      post_ready_steady_rss_status: "measured_for_review",
      post_ready_steady_process_rss_peak_bytes: $steady_rss_peak,
      post_ready_steady_process_rss_mean_bytes: $steady_rss_mean,
      buffered_wav_sox: "passed",
      cancellation: "passed",
      prompt_free_metrics: "passed",
      language_natural_eos: "11/11",
      restart_readiness: "passed",
      graceful_sigterm: "passed"
    },
    b1: {
      planned_completed_successful_requests: $b1_requests,
      warmups: $b1_warmups,
      failed_requests: 0,
      natural_eos_requests: $b1_requests,
      aggregate_rtf: $b1_rtf,
      ttfa_p95_ms: $b1_ttfa,
      process_rss_peak_bytes: $b1_rss,
      gpu_unified_memory_peak_bytes: $b1_gpu,
      maximum_observed_telemetry_gap_ms: $b1_gap
    },
    b6: {
      planned_completed_successful_requests: $b6_requests,
      warmups: $b6_warmups,
      failed_requests: 0,
      natural_eos_requests: $b6_requests,
      aggregate_rtf: $b6_rtf,
      ttfa_p95_ms: $b6_ttfa,
      process_rss_peak_bytes: $b6_rss,
      gpu_unified_memory_peak_bytes: $b6_gpu,
      gpu_unified_memory_limit_bytes: $b6_gpu_limit,
      maximum_observed_telemetry_gap_ms: $b6_gap
    },
    internal_request_memory: {
      definition: "maximum server end-event per-request accounting; never substituted for observed process or GPU unified memory",
      b1_peak_request_device_bytes: $b1_internal_device,
      b1_peak_request_host_bytes: $b1_internal_host,
      b6_peak_request_device_bytes: $b6_internal_device,
      b6_peak_request_host_bytes: $b6_internal_host,
      substituted_for_observed_total: false
    },
    telemetry: {
      configured_interval_ms: 100,
      maximum_allowed_observed_gap_ms: 200,
      competing_cuda_processes: 0
    }
  }' >"$EVIDENCE_DIR/final-gpu-acceptance.json"

printf 'Final digest-bound B1/B6 GPU acceptance passed for %s\n' "$REFERENCE"
