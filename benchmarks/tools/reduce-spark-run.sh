#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  reduce-spark-run.sh --run-dir DIR --engine native|sglang \
    --profile B1|B3|B6 --round POSITIVE_INTEGER \
    --evidence-prefix MANIFEST_RELATIVE_POSIX_PATH

Validates and reduces one completed qualifying-run directory. The directory must
contain raw telemetry, the four-event client phase file, and client summary. The
script writes run-resource.json and resource-audit.json without overwriting.
USAGE
}

die() {
  echo "$*" >&2
  exit 65
}

run_dir=
engine=
profile=
round=
evidence_prefix=

while (($#)); do
  case "$1" in
    --run-dir)
      (($# >= 2)) || { usage; exit 64; }
      run_dir=$2
      shift 2
      ;;
    --engine)
      (($# >= 2)) || { usage; exit 64; }
      engine=$2
      shift 2
      ;;
    --profile)
      (($# >= 2)) || { usage; exit 64; }
      profile=$2
      shift 2
      ;;
    --round)
      (($# >= 2)) || { usage; exit 64; }
      round=$2
      shift 2
      ;;
    --evidence-prefix)
      (($# >= 2)) || { usage; exit 64; }
      evidence_prefix=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 64
      ;;
  esac
done

[[ -n "$run_dir" && -n "$engine" && -n "$profile" && -n "$round" && -n "$evidence_prefix" ]] || {
  usage
  exit 64
}
[[ "$engine" == native || "$engine" == sglang ]] || die "engine must be native or sglang"
[[ "$profile" =~ ^B(1|3|6)$ ]] || die "profile must be B1, B3, or B6"
[[ "$round" =~ ^[1-9][0-9]*$ ]] || die "round must be a positive integer"
[[ "$evidence_prefix" != /* && "$evidence_prefix" != */ && "$evidence_prefix" != *//* && \
  "$evidence_prefix" != *\\* && "$evidence_prefix" != *$'\n'* ]] || die "evidence prefix must be a normalized relative POSIX path"
IFS=/ read -r -a evidence_components <<<"$evidence_prefix"
for component in "${evidence_components[@]}"; do
  [[ -n "$component" && "$component" != . && "$component" != .. ]] || die "evidence prefix contains an unsafe path component"
done
[[ -d "$run_dir" && ! -L "$run_dir" ]] || die "run directory is missing or is a symlink: $run_dir"

for command_name in awk jq sha256sum; do
  command -v "$command_name" >/dev/null || die "required command is unavailable: $command_name"
done

raw_dir="$run_dir/raw"
client_dir="$run_dir/client"
phase_path="$raw_dir/phase-events.jsonl"
run_metadata_path="$raw_dir/run.txt"
gpu_path="$raw_dir/gpu.csv"
system_path="$raw_dir/system.csv"
rss_path="$raw_dir/process-rss.csv"
rss_total_path="$raw_dir/process-rss-total.csv"
gpu_processes_path="$raw_dir/gpu-processes.csv"
gpu_summary_path="$raw_dir/gpu-process-summary.csv"
summary_path="$client_dir/summary.json"
command_stdout_path="$raw_dir/command.stdout"
command_stderr_path="$raw_dir/command.stderr"
resource_path="$run_dir/run-resource.json"
audit_path="$run_dir/resource-audit.json"

require_regular_file() {
  local path=$1
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "required non-empty regular file is missing or unsafe: $path"
}

require_regular_file_allow_empty() {
  local path=$1
  [[ -f "$path" && ! -L "$path" ]] || die "required regular file is missing or unsafe: $path"
}

for path in \
  "$phase_path" "$run_metadata_path" "$gpu_path" "$system_path" \
  "$rss_path" "$rss_total_path" "$gpu_processes_path" "$gpu_summary_path" \
  "$summary_path"; do
  require_regular_file "$path"
done
require_regular_file_allow_empty "$command_stdout_path"
require_regular_file_allow_empty "$command_stderr_path"
[[ ! -e "$resource_path" && ! -e "$audit_path" ]] || die "refusing to overwrite an existing reduction"

expected_phase_schema=qwen3-tts-http-bench-phase-events/v1
jq -e -s --arg schema "$expected_phase_schema" '
  length == 4 and
  [.[].sequence] == [0, 1, 2, 3] and
  [.[].event] == ["warmup_start", "warmup_end", "measured_start", "measured_end"] and
  all(.[ ];
    type == "object" and
    (keys | sort) == ["event", "monotonic_elapsed_ns", "schema_version", "sequence", "wall_time_unix_ns"] and
    .schema_version == $schema and
    (.sequence | type) == "number" and .sequence == (.sequence | floor) and
    (.wall_time_unix_ns | type) == "number" and
    .wall_time_unix_ns == (.wall_time_unix_ns | floor) and .wall_time_unix_ns >= 0 and
    (.monotonic_elapsed_ns | type) == "number" and
    .monotonic_elapsed_ns == (.monotonic_elapsed_ns | floor) and .monotonic_elapsed_ns >= 0
  )
' "$phase_path" >/dev/null || die "phase-events.jsonl does not satisfy the exact v1 four-event contract"

phase_rows=()
while IFS= read -r phase_row; do
  phase_rows+=("$phase_row")
done < <(jq -r '[.sequence, .event, .wall_time_unix_ns, .monotonic_elapsed_ns] | @tsv' "$phase_path")
[[ ${#phase_rows[@]} -eq 4 ]] || die "phase-events.jsonl must contain exactly four records"

declare -a phase_wall phase_monotonic
expected_events=(warmup_start warmup_end measured_start measured_end)
for index in 0 1 2 3; do
  IFS=$'\t' read -r sequence event wall_ns monotonic_ns <<<"${phase_rows[$index]}"
  [[ "$sequence" == "$index" ]] || die "phase sequence is not contiguous"
  [[ "$event" == "${expected_events[$index]}" ]] || die "phase event order is invalid"
  [[ "$wall_ns" =~ ^[0-9]+$ && "$monotonic_ns" =~ ^[0-9]+$ ]] || die "phase timestamps must be integer nanoseconds"
  phase_wall[index]=$wall_ns
  phase_monotonic[index]=$monotonic_ns
done
for index in 1 2 3; do
  ((phase_wall[index] >= phase_wall[index - 1])) || die "phase wall-clock timestamps are not monotonic"
  ((phase_monotonic[index] >= phase_monotonic[index - 1])) || die "phase monotonic timestamps are not monotonic"
done
((phase_wall[3] > phase_wall[2])) || die "measured wall-clock window must have positive duration"
((phase_monotonic[3] > phase_monotonic[2])) || die "measured monotonic window must have positive duration"

run_value() {
  local key=$1
  local path=$2
  local count value
  count=$(awk -F= -v wanted="$key" '$1 == wanted { count++ } END { print count + 0 }' "$path")
  [[ "$count" == 1 ]] || die "run metadata must contain exactly one $key field"
  value=$(awk -F= -v wanted="$key" '$1 == wanted { sub(/^[^=]*=/, ""); print; exit }' "$path")
  [[ -n "$value" ]] || die "run metadata field is empty: $key"
  printf '%s\n' "$value"
}

command_status=$(run_value exit_status "$run_metadata_path")
sample_interval_ms=$(run_value sample_interval_ms "$run_metadata_path")
maximum_gap_ms=$(run_value maximum_qualifying_gap_ms "$run_metadata_path")
idle_baseline_seconds=$(run_value idle_baseline_seconds "$run_metadata_path")
idle_start_ns=$(run_value idle_baseline_start_wall_time_unix_ns "$run_metadata_path")
idle_end_ns=$(run_value idle_baseline_end_wall_time_unix_ns "$run_metadata_path")

[[ "$command_status" == 0 ]] || die "benchmark client did not exit successfully"
[[ "$sample_interval_ms" =~ ^[0-9]+$ && "$sample_interval_ms" -le 100 ]] || die "configured sample interval is not qualifying"
[[ "$maximum_gap_ms" == 200 ]] || die "maximum qualifying telemetry gap must be exactly 200 ms"
[[ "$idle_baseline_seconds" =~ ^[0-9]+$ && "$idle_baseline_seconds" -ge 15 ]] || die "idle baseline declaration is shorter than 15 seconds"
[[ "$idle_start_ns" =~ ^[0-9]+$ && "$idle_end_ns" =~ ^[0-9]+$ ]] || die "idle baseline timestamps are invalid"
((idle_end_ns > idle_start_ns)) || die "idle baseline window must have positive duration"
((idle_end_ns <= phase_wall[0])) || die "idle baseline overlaps client warmups"

idle_duration_seconds=$(awk -v start="$idle_start_ns" -v end="$idle_end_ns" 'BEGIN { printf "%.9f", (end - start) / 1000000000 }')
awk -v duration="$idle_duration_seconds" 'BEGIN { exit !(duration >= 15.0) }' || die "observed idle baseline is shorter than 15 seconds"

expected_backend=native
[[ "$engine" == sglang ]] && expected_backend=sglang-omni
jq -e \
  --arg backend "$expected_backend" \
  --arg profile "$profile" \
  --arg engine "$engine" '
    .schema_version == "qwen3-tts-http-bench/v1" and
    .backend == $backend and .concurrency == $profile and
    (.warmups | type) == "number" and .warmups >= 24 and
    (.planned_requests | type) == "number" and .planned_requests >= 200 and
    .completed_requests == .planned_requests and
    .successful_requests >= 200 and
    .failed_requests == (.completed_requests - .successful_requests) and
    .sampling_parity_qualifying_requests == .completed_requests and
    .sampling_parity_non_qualifying_requests == 0 and
    (.benchmark_wall_seconds | type) == "number" and .benchmark_wall_seconds > 0 and
    (if $engine == "native" then
      .natural_eos_requests == .successful_requests and
      .length_limited_requests == 0 and .eos_unknown_requests == 0
    else
      .natural_eos_requests == 0 and .length_limited_requests == 0 and
      .eos_unknown_requests == .successful_requests
    end)
  ' "$summary_path" >/dev/null || die "client summary is not a qualifying $engine/$profile run"

summary_wall_seconds=$(jq -r '.benchmark_wall_seconds' "$summary_path")
phase_monotonic_seconds=$(awk -v start="${phase_monotonic[2]}" -v end="${phase_monotonic[3]}" \
  'BEGIN { printf "%.9f", (end - start) / 1000000000 }')
phase_wall_seconds=$(awk -v start="${phase_wall[2]}" -v end="${phase_wall[3]}" \
  'BEGIN { printf "%.9f", (end - start) / 1000000000 }')
awk -v client="$summary_wall_seconds" -v phase="$phase_monotonic_seconds" '
  BEGIN { delta = client - phase; if (delta < 0) delta = -delta; exit !(delta <= 0.005) }
' || die "client summary wall time differs from the measured monotonic phase by more than 5 ms"
awk -v wall="$phase_wall_seconds" -v monotonic="$phase_monotonic_seconds" '
  BEGIN { delta = wall - monotonic; if (delta < 0) delta = -delta; exit !(delta <= 0.005) }
' || die "wall and monotonic measured windows differ by more than 5 ms"

check_header() {
  local path=$1
  local expected=$2
  local actual
  IFS= read -r actual <"$path"
  [[ "$actual" == "$expected" ]] || die "unexpected CSV header in $path"
}

check_header "$gpu_path" \
  'wall_time_unix_ns,timestamp_utc,gpu_index,gpu_uuid,pstate,temperature_c,gpu_util_percent,memory_util_percent,power_w,graphics_clock_mhz'
check_header "$rss_total_path" \
  'wall_time_unix_ns,timestamp_utc,listed_processes,sampled_processes,process_rss_sum_bytes,sample_complete'
check_header "$rss_path" \
  'wall_time_unix_ns,timestamp_utc,pid,process_start_ticks,process_name,rss_bytes'
check_header "$gpu_summary_path" \
  'wall_time_unix_ns,timestamp_utc,query_ok,gpu_compute_processes,target_container_gpu_processes,competing_cuda_processes,target_unified_memory_mib'
check_header "$gpu_processes_path" \
  'wall_time_unix_ns,timestamp_utc,pid,in_target_container,process_name,unified_memory_mib'
check_header "$system_path" \
  'wall_time_unix_ns,timestamp_utc,uptime_s,cgroup_memory_bytes,cgroup_memory_peak_bytes,cgroup_pids,cgroup_cpu_usec,host_mem_available_kib,host_swap_free_kib'

temporary_dir=$(mktemp -d "$run_dir/.reduce.XXXXXX")
cleanup() {
  rm -rf "$temporary_dir"
}
trap cleanup EXIT INT TERM
max_gap_ns=$((maximum_gap_ms * 1000000))

power_metrics_path="$temporary_dir/power.tsv"
awk -F, \
  -v idle_start="$idle_start_ns" -v idle_end="$idle_end_ns" \
  -v measured_start="${phase_wall[2]}" -v measured_end="${phase_wall[3]}" \
  -v max_gap="$max_gap_ns" '
  function fail(message) { print message > "/dev/stderr"; failed = 1; exit 2 }
  function trim(value) { gsub(/^[[:space:]]+|[[:space:]]+$/, "", value); return value }
  function interpolate(point,    i, ratio) {
    for (i = 1; i < count; i++) {
      if (times[i] <= point && times[i + 1] >= point) {
        ratio = (point - times[i]) / (times[i + 1] - times[i])
        return watts[i] + ratio * (watts[i + 1] - watts[i])
      }
    }
    fail("power telemetry does not bracket a reduction boundary")
  }
  function reduce_window(start, end, key,    i, previous_time, previous_power, current_power, duration) {
    if (times[1] > start || times[count] < end) fail("power telemetry does not bracket the complete window")
    for (i = 1; i < count; i++) {
      if (times[i] <= end && times[i + 1] >= start && times[i + 1] - times[i] > max_gap)
        fail("power telemetry contains an observed gap greater than 200 ms")
    }
    previous_time = start
    previous_power = interpolate(start)
    result_peak[key] = previous_power
    result_samples[key] = 0
    result_energy[key] = 0
    for (i = 1; i <= count; i++) {
      if (times[i] >= start && times[i] <= end) result_samples[key]++
      if (times[i] <= start || times[i] >= end) continue
      duration = (times[i] - previous_time) / 1000000000
      result_energy[key] += (previous_power + watts[i]) * duration / 2
      previous_time = times[i]
      previous_power = watts[i]
      if (previous_power > result_peak[key]) result_peak[key] = previous_power
    }
    current_power = interpolate(end)
    duration = (end - previous_time) / 1000000000
    result_energy[key] += (previous_power + current_power) * duration / 2
    if (current_power > result_peak[key]) result_peak[key] = current_power
    result_duration[key] = (end - start) / 1000000000
    if (result_samples[key] < 2 || result_duration[key] <= 0) fail("power window has insufficient samples")
  }
  NR == 1 { next }
  {
    timestamp = trim($1)
    power = trim($9)
    if (timestamp !~ /^[0-9]+$/ || power !~ /^[0-9]+([.][0-9]+)?$/) fail("power telemetry contains an unavailable or malformed sensor value")
    count++
    times[count] = timestamp + 0
    watts[count] = power + 0
    if (count > 1 && times[count] <= times[count - 1]) fail("power telemetry timestamps are not strictly increasing")
  }
  END {
    if (failed) exit 2
    if (count < 2) fail("power telemetry has fewer than two samples")
    reduce_window(idle_start + 0, idle_end + 0, "idle")
    reduce_window(measured_start + 0, measured_end + 0, "measured")
    printf "%.9f\t%.9f\t%.9f\t%d\t%.9f\t%.9f\t%.9f\t%d\n", \
      result_energy["idle"], result_energy["idle"] / result_duration["idle"], \
      result_peak["idle"], result_samples["idle"], result_energy["measured"], \
      result_energy["measured"] / result_duration["measured"], \
      result_peak["measured"], result_samples["measured"]
  }
' "$gpu_path" >"$power_metrics_path" || die "power telemetry reduction failed"
IFS=$'\t' read -r idle_gross_energy_j idle_average_power_w idle_peak_power_w idle_power_samples \
  measured_gross_energy_j measured_average_power_w measured_peak_power_w measured_power_samples \
  <"$power_metrics_path"

adjusted_energy_j=$(awk \
  -v gross="$measured_gross_energy_j" -v idle="$idle_average_power_w" -v duration="$phase_wall_seconds" '
  BEGIN { value = gross - idle * duration; if (value < 0) value = 0; printf "%.9f", value }
')

rss_metrics_path="$temporary_dir/rss.tsv"
awk -F, -v start="${phase_wall[2]}" -v end="${phase_wall[3]}" -v max_gap="$max_gap_ns" '
  function fail(message) { print message > "/dev/stderr"; failed = 1; exit 2 }
  NR == 1 { next }
  {
    if ($1 !~ /^[0-9]+$/ || $3 !~ /^[0-9]+$/ || $4 !~ /^[0-9]+$/ ||
        $5 !~ /^[0-9]+$/ || $6 !~ /^[01]$/) fail("process RSS summary contains a malformed value")
    count++
    time[count] = $1 + 0
    if (count > 1 && time[count] <= time[count - 1]) fail("process RSS timestamps are not strictly increasing")
    if ($1 + 0 >= start + 0 && $1 + 0 <= end + 0) {
      inside++
      if ($6 != 1 || $3 < 1 || $4 != $3 || $5 < 1) fail("process RSS sample is incomplete or empty in the measured window")
      if ($5 > peak) peak = $5
    }
  }
  END {
    if (failed) exit 2
    if (count < 2 || time[1] > start || time[count] < end) fail("process RSS telemetry does not bracket the measured window")
    for (i = 1; i < count; i++)
      if (time[i] <= end && time[i + 1] >= start && time[i + 1] - time[i] > max_gap)
        fail("process RSS telemetry contains an observed gap greater than 200 ms")
    if (inside < 1 || peak < 1) fail("no process RSS sample exists in the measured window")
    printf "%.0f\t%d\n", peak, inside
  }
' "$rss_total_path" >"$rss_metrics_path" || die "process RSS telemetry reduction failed"
IFS=$'\t' read -r process_rss_peak_bytes process_rss_samples <"$rss_metrics_path"

gpu_process_metrics_path="$temporary_dir/gpu-process.tsv"
awk -F, -v audit_start="$idle_start_ns" -v start="${phase_wall[2]}" \
  -v end="${phase_wall[3]}" -v max_gap="$max_gap_ns" '
  function fail(message) { print message > "/dev/stderr"; failed = 1; exit 2 }
  NR == 1 { next }
  {
    if ($1 !~ /^[0-9]+$/ || $3 !~ /^[01]$/ || $4 !~ /^[0-9]+$/ ||
        $5 !~ /^[0-9]+$/ || $6 !~ /^[0-9]+$/) fail("GPU process summary contains a malformed value")
    count++
    time[count] = $1 + 0
    if (count > 1 && time[count] <= time[count - 1]) fail("GPU process timestamps are not strictly increasing")
    if ($1 + 0 >= audit_start + 0 && $1 + 0 <= end + 0) {
      audit_inside++
      if ($3 != 1) fail("GPU process query failed between idle baseline and measured end")
      if ($6 > competing) competing = $6
    }
    if ($1 + 0 >= start + 0 && $1 + 0 <= end + 0) {
      inside++
      if ($5 < 1) fail("target container has no GPU process in the measured window")
      if ($7 !~ /^[0-9]+([.][0-9]+)?$/ || $7 <= 0) fail("GPU unified-memory sensor is unavailable in the measured window")
      if ($7 > peak_mib) peak_mib = $7
    }
  }
  END {
    if (failed) exit 2
    if (count < 2 || time[1] > audit_start || time[count] < end) fail("GPU process telemetry does not bracket idle start through measured end")
    for (i = 1; i < count; i++)
      if (time[i] <= end && time[i + 1] >= audit_start && time[i + 1] - time[i] > max_gap)
        fail("GPU process telemetry contains an observed gap greater than 200 ms")
    if (audit_inside < 1 || inside < 1 || peak_mib <= 0) fail("no complete GPU process window exists")
    printf "%.0f\t%d\t%d\n", peak_mib * 1048576, competing, inside
  }
' "$gpu_summary_path" >"$gpu_process_metrics_path" || die "GPU process telemetry reduction failed"
IFS=$'\t' read -r gpu_unified_memory_peak_bytes competing_cuda_processes gpu_process_samples \
  <"$gpu_process_metrics_path"
[[ "$competing_cuda_processes" == 0 ]] || die "competing CUDA processes were observed between idle baseline and measured end"

system_metrics_path="$temporary_dir/system.tsv"
awk -F, -v start="${phase_wall[2]}" -v end="${phase_wall[3]}" -v max_gap="$max_gap_ns" '
  function fail(message) { print message > "/dev/stderr"; failed = 1; exit 2 }
  NR == 1 { next }
  {
    if ($1 !~ /^[0-9]+$/ || $4 !~ /^[0-9]+$/ || $8 !~ /^[0-9]+$/ || $9 !~ /^[0-9]+$/)
      fail("system telemetry contains a malformed value")
    count++
    time[count] = $1 + 0
    if (count > 1 && time[count] <= time[count - 1]) fail("system telemetry timestamps are not strictly increasing")
    if ($1 + 0 >= start + 0 && $1 + 0 <= end + 0) {
      inside++
      if ($4 > cgroup_peak) cgroup_peak = $4
      if (host_available == 0 || $8 < host_available) host_available = $8
      if (inside == 1) swap_free_start = $9
      swap_free_end = $9
    }
  }
  END {
    if (failed) exit 2
    if (count < 2 || time[1] > start || time[count] < end) fail("system telemetry does not bracket the measured window")
    for (i = 1; i < count; i++)
      if (time[i] <= end && time[i + 1] >= start && time[i + 1] - time[i] > max_gap)
        fail("system telemetry contains an observed gap greater than 200 ms")
    if (inside < 1 || cgroup_peak < 1) fail("no system telemetry sample exists in the measured window")
    printf "%.0f\t%.0f\t%.0f\t%.0f\t%d\n", cgroup_peak, host_available, swap_free_start, swap_free_end, inside
  }
' "$system_path" >"$system_metrics_path" || die "system telemetry reduction failed"
IFS=$'\t' read -r cgroup_memory_current_peak_bytes host_mem_available_min_kib \
  host_swap_free_start_kib host_swap_free_end_kib system_samples <"$system_metrics_path"

jq -n \
  --arg engine_id "$engine" --arg profile_id "$profile" --argjson round "$round" \
  --argjson rss "$process_rss_peak_bytes" --argjson gpu_memory "$gpu_unified_memory_peak_bytes" \
  --argjson average_power "$measured_average_power_w" --argjson peak_power "$measured_peak_power_w" \
  --argjson energy "$adjusted_energy_j" --argjson interval "$sample_interval_ms" \
  --argjson competing "$competing_cuda_processes" --arg prefix "$evidence_prefix" '
  {
    engine_id: $engine_id,
    profile_id: $profile_id,
    round: $round,
    process_rss_peak_bytes: $rss,
    gpu_unified_memory_peak_bytes: $gpu_memory,
    average_power_w: $average_power,
    peak_power_w: $peak_power,
    energy_j: $energy,
    sampling_interval_ms: $interval,
    competing_cuda_processes: $competing,
    telemetry_evidence_paths: [
      ($prefix + "/raw/gpu.csv"),
      ($prefix + "/raw/system.csv"),
      ($prefix + "/raw/process-rss.csv"),
      ($prefix + "/raw/process-rss-total.csv"),
      ($prefix + "/raw/gpu-processes.csv"),
      ($prefix + "/raw/gpu-process-summary.csv"),
      ($prefix + "/raw/phase-events.jsonl"),
      ($prefix + "/raw/run.txt")
    ]
  }
' >"$temporary_dir/run-resource.json"

file_digest_object() {
  local path=$1
  local relative=${path#"$run_dir/"}
  local digest bytes
  digest=$(sha256sum "$path" | awk '{ print $1 }')
  bytes=$(wc -c <"$path" | awk '{ print $1 }')
  jq -n --arg path "$relative" --arg sha256 "$digest" --argjson bytes "$bytes" \
    '{path: $path, sha256: $sha256, bytes: $bytes}'
}

digests_path="$temporary_dir/digests.jsonl"
for path in \
  "$phase_path" "$run_metadata_path" "$gpu_path" "$system_path" "$rss_path" \
  "$rss_total_path" "$gpu_processes_path" "$gpu_summary_path" "$summary_path" \
  "$command_stdout_path" "$command_stderr_path"; do
  file_digest_object "$path" >>"$digests_path"
done

jq -n \
  --arg schema_version qwen3-tts-spark-resource-audit/v1 \
  --arg engine_id "$engine" --arg profile_id "$profile" --argjson round "$round" \
  --arg idle_start_ns "$idle_start_ns" --arg idle_end_ns "$idle_end_ns" \
  --arg measured_start_ns "${phase_wall[2]}" --arg measured_end_ns "${phase_wall[3]}" \
  --argjson idle_duration "$idle_duration_seconds" \
  --argjson measured_duration "$phase_monotonic_seconds" \
  --argjson measured_wall_duration "$phase_wall_seconds" \
  --argjson idle_samples "$idle_power_samples" --argjson measured_power_samples "$measured_power_samples" \
  --argjson rss_samples "$process_rss_samples" --argjson gpu_samples "$gpu_process_samples" \
  --argjson system_samples "$system_samples" --argjson idle_mean "$idle_average_power_w" \
  --argjson idle_peak "$idle_peak_power_w" --argjson idle_gross_energy "$idle_gross_energy_j" \
  --argjson measured_mean "$measured_average_power_w" --argjson measured_peak "$measured_peak_power_w" \
  --argjson measured_gross_energy "$measured_gross_energy_j" --argjson measured_adjusted_energy "$adjusted_energy_j" \
  --argjson rss_peak "$process_rss_peak_bytes" --argjson gpu_memory_peak "$gpu_unified_memory_peak_bytes" \
  --argjson cgroup_peak "$cgroup_memory_current_peak_bytes" \
  --argjson host_available "$host_mem_available_min_kib" \
  --argjson swap_start "$host_swap_free_start_kib" --argjson swap_end "$host_swap_free_end_kib" \
  --argjson interval "$sample_interval_ms" --argjson max_gap "$maximum_gap_ms" \
  --slurpfile sources "$digests_path" '
  {
    schema_version: $schema_version,
    engine_id: $engine_id,
    profile_id: $profile_id,
    round: $round,
    phase_boundaries: {
      idle_start_wall_time_unix_ns: $idle_start_ns,
      idle_end_wall_time_unix_ns: $idle_end_ns,
      measured_start_wall_time_unix_ns: $measured_start_ns,
      measured_end_wall_time_unix_ns: $measured_end_ns,
      idle_duration_seconds: $idle_duration,
      measured_monotonic_duration_seconds: $measured_duration,
      measured_wall_duration_seconds: $measured_wall_duration
    },
    sampling: {
      configured_interval_ms: $interval,
      maximum_allowed_observed_gap_ms: $max_gap,
      idle_power_samples: $idle_samples,
      measured_power_samples: $measured_power_samples,
      measured_process_rss_samples: $rss_samples,
      measured_gpu_process_samples: $gpu_samples,
      measured_system_samples: $system_samples
    },
    power: {
      source: "NVIDIA board power.draw",
      integration: "linear boundary interpolation and trapezoidal integration",
      idle_average_power_w: $idle_mean,
      idle_peak_power_w: $idle_peak,
      idle_gross_energy_j: $idle_gross_energy,
      measured_average_power_w: $measured_mean,
      measured_peak_power_w: $measured_peak,
      measured_gross_energy_j: $measured_gross_energy,
      measured_idle_adjusted_energy_j: $measured_adjusted_energy,
      idle_adjustment: "max(0, measured gross energy - idle mean power * measured wall-clock duration)"
    },
    memory: {
      process_rss_definition: "peak measured-window sum of VmRSS for every extant PID in the target container cgroup",
      process_rss_peak_bytes: $rss_peak,
      gpu_unified_memory_definition: "peak measured-window sum of NVIDIA used_memory for target-container compute PIDs",
      gpu_unified_memory_peak_bytes: $gpu_memory_peak,
      cgroup_memory_definition: "peak measured-window memory.current; distinct from process RSS",
      cgroup_memory_current_peak_bytes: $cgroup_peak,
      host_mem_available_min_kib: $host_available,
      host_swap_free_start_kib: $swap_start,
      host_swap_free_end_kib: $swap_end
    },
    source_files: $sources
  }
' >"$temporary_dir/resource-audit.json"

mv "$temporary_dir/run-resource.json" "$resource_path"
mv "$temporary_dir/resource-audit.json" "$audit_path"
