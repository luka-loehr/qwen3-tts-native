#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  capture-spark-telemetry.sh --output-dir DIR --container NAME \
    --idle-baseline-seconds SECONDS [--sample-interval-ms MILLISECONDS] \
    [--gpu-index INDEX] -- COMMAND [ARG...]

Starts telemetry, records a fixed server-idle baseline, and then runs COMMAND.
The configured sampling interval must be 100 ms or faster; the qualifying-run
reducer rejects observed gaps greater than 200 ms. The output directory must not
already exist. Do not put secrets in COMMAND.
USAGE
}

output_dir=
container_name=
idle_baseline_seconds=
sample_interval_ms=100
gpu_index=0

while (($#)); do
  case "$1" in
    --output-dir)
      (($# >= 2)) || { usage; exit 64; }
      output_dir=$2
      shift 2
      ;;
    --container)
      (($# >= 2)) || { usage; exit 64; }
      container_name=$2
      shift 2
      ;;
    --idle-baseline-seconds)
      (($# >= 2)) || { usage; exit 64; }
      idle_baseline_seconds=$2
      shift 2
      ;;
    --sample-interval-ms)
      (($# >= 2)) || { usage; exit 64; }
      sample_interval_ms=$2
      shift 2
      ;;
    --gpu-index)
      (($# >= 2)) || { usage; exit 64; }
      gpu_index=$2
      shift 2
      ;;
    --)
      shift
      break
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

[[ -n "$output_dir" && -n "$container_name" && -n "$idle_baseline_seconds" && $# -gt 0 ]] || {
  usage
  exit 64
}
[[ "$idle_baseline_seconds" =~ ^[0-9]+$ && "$idle_baseline_seconds" -ge 15 ]] || {
  echo "idle baseline must be an integer of at least 15 seconds" >&2
  exit 64
}
[[ "$sample_interval_ms" =~ ^[0-9]+$ && "$sample_interval_ms" -ge 20 && "$sample_interval_ms" -le 100 ]] || {
  echo "sample interval must be an integer from 20 through 100 ms" >&2
  exit 64
}
[[ "$gpu_index" =~ ^[0-9]+$ ]] || {
  echo "GPU index must be a non-negative integer" >&2
  exit 64
}
[[ ! -e "$output_dir" ]] || {
  echo "output directory already exists: $output_dir" >&2
  exit 73
}

for command_name in awk date docker getconf nvidia-smi sed sleep tail wc; do
  command -v "$command_name" >/dev/null || {
    echo "required command is unavailable: $command_name" >&2
    exit 69
  }
done

container_pid=$(docker inspect --format '{{.State.Pid}}' "$container_name")
[[ "$container_pid" =~ ^[1-9][0-9]*$ ]] || {
  echo "container is not running: $container_name" >&2
  exit 69
}

cgroup_relative=$(awk -F: '$1 == "0" { print $3 }' "/proc/$container_pid/cgroup")
[[ -n "$cgroup_relative" ]] || {
  echo "cannot resolve cgroup v2 path for container: $container_name" >&2
  exit 70
}
cgroup_dir="/sys/fs/cgroup$cgroup_relative"
[[ -r "$cgroup_dir/memory.current" && -r "$cgroup_dir/pids.current" ]] || {
  echo "container cgroup telemetry is unavailable: $cgroup_dir" >&2
  exit 69
}

page_size=$(getconf PAGESIZE)
[[ "$page_size" =~ ^[1-9][0-9]*$ ]] || {
  echo "cannot determine the host page size" >&2
  exit 70
}
sample_interval_seconds=$(awk -v milliseconds="$sample_interval_ms" 'BEGIN { printf "%.3f", milliseconds / 1000 }')

wall_time_unix_ns() {
  local value
  value=$(date +%s%N)
  [[ "$value" =~ ^[0-9]{19}$ ]] || {
    echo "GNU date did not provide a nanosecond Unix timestamp" >&2
    return 70
  }
  printf '%s\n' "$value"
}

timestamp_utc() {
  date --utc --iso-8601=ns
}

csv_quote() {
  local value=${1//$'\r'/ }
  value=${value//$'\n'/ }
  value=${value//\"/\"\"}
  printf '"%s"' "$value"
}

pid_is_in_target_cgroup() {
  local candidate=$1
  [[ -r "/proc/$candidate/cgroup" ]] || return 1
  awk -F: -v expected="$cgroup_relative" '$1 == "0" && $3 == expected { found = 1 } END { exit !found }' \
    "/proc/$candidate/cgroup"
}

mkdir -p "$output_dir"

{
  printf 'container=%s\n' "$container_name"
  printf 'container_pid=%s\n' "$container_pid"
  printf 'cgroup=%s\n' "$cgroup_relative"
  printf 'started_at=%s\n' "$(date --iso-8601=ns)"
  printf 'started_wall_time_unix_ns=%s\n' "$(wall_time_unix_ns)"
  printf 'sample_interval_ms=%s\n' "$sample_interval_ms"
  printf 'maximum_qualifying_gap_ms=200\n'
  printf 'idle_baseline_seconds=%s\n' "$idle_baseline_seconds"
  printf 'gpu_index=%s\n' "$gpu_index"
  printf 'page_size_bytes=%s\n' "$page_size"
  printf 'command='
  printf '%q ' "$@"
  printf '\n'
} >"$output_dir/run.txt"

printf '%s\n' \
  'wall_time_unix_ns,timestamp_utc,gpu_index,gpu_uuid,pstate,temperature_c,gpu_util_percent,memory_util_percent,power_w,graphics_clock_mhz' \
  >"$output_dir/gpu.csv"
printf '%s\n' \
  'wall_time_unix_ns,timestamp_utc,uptime_s,cgroup_memory_bytes,cgroup_memory_peak_bytes,cgroup_pids,cgroup_cpu_usec,host_mem_available_kib,host_swap_free_kib' \
  >"$output_dir/system.csv"
printf '%s\n' \
  'wall_time_unix_ns,timestamp_utc,pid,process_start_ticks,process_name,rss_bytes' \
  >"$output_dir/process-rss.csv"
printf '%s\n' \
  'wall_time_unix_ns,timestamp_utc,listed_processes,sampled_processes,process_rss_sum_bytes,sample_complete' \
  >"$output_dir/process-rss-total.csv"
printf '%s\n' 'wall_time_unix_ns,timestamp_utc,pid,in_target_container,process_name,unified_memory_mib' \
  >"$output_dir/gpu-processes.csv"
printf '%s\n' \
  'wall_time_unix_ns,timestamp_utc,query_ok,gpu_compute_processes,target_container_gpu_processes,competing_cuda_processes,target_unified_memory_mib' \
  >"$output_dir/gpu-process-summary.csv"

sampler_pids=()

(
  while :; do
    sample_ns=$(wall_time_unix_ns) || exit $?
    sample_utc=$(timestamp_utc)
    if sample=$(nvidia-smi \
      --id="$gpu_index" \
      --query-gpu=index,uuid,pstate,temperature.gpu,utilization.gpu,utilization.memory,power.draw,clocks.current.graphics \
      --format=csv,noheader,nounits 2>/dev/null); then
      printf '%s,%s,%s\n' "$sample_ns" "$sample_utc" "$sample"
    else
      printf '%s,%s,NA,NA,NA,NA,NA,NA,NA,NA\n' "$sample_ns" "$sample_utc"
    fi
    sleep "$sample_interval_seconds"
  done
) >>"$output_dir/gpu.csv" &
sampler_pids+=("$!")

(
  while :; do
    sample_ns=$(wall_time_unix_ns) || exit $?
    sample_utc=$(timestamp_utc)
    uptime_s=$(awk '{ print $1 }' /proc/uptime)
    memory_bytes=$(<"$cgroup_dir/memory.current")
    if [[ -r "$cgroup_dir/memory.peak" ]]; then
      memory_peak_bytes=$(<"$cgroup_dir/memory.peak")
    else
      memory_peak_bytes=NA
    fi
    pids=$(<"$cgroup_dir/pids.current")
    cpu_usec=$(awk '$1 == "usage_usec" { print $2 }' "$cgroup_dir/cpu.stat")
    host_mem_kib=$(awk '$1 == "MemAvailable:" { print $2 }' /proc/meminfo)
    host_swap_kib=$(awk '$1 == "SwapFree:" { print $2 }' /proc/meminfo)
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$sample_ns" "$sample_utc" "$uptime_s" "$memory_bytes" "$memory_peak_bytes" \
      "$pids" "$cpu_usec" "$host_mem_kib" "$host_swap_kib"
    sleep "$sample_interval_seconds"
  done
) >>"$output_dir/system.csv" &
sampler_pids+=("$!")

(
  while :; do
    sample_ns=$(wall_time_unix_ns) || exit $?
    sample_utc=$(timestamp_utc)
    mapfile -t process_ids <"$cgroup_dir/cgroup.procs"
    listed=${#process_ids[@]}
    sampled=0
    read_failures=0
    rss_sum=0
    for process_id in "${process_ids[@]}"; do
      if [[ ! "$process_id" =~ ^[1-9][0-9]*$ ]]; then
        read_failures=$((read_failures + 1))
        continue
      fi
      status_path="/proc/$process_id/status"
      stat_path="/proc/$process_id/stat"
      if ! IFS= read -r stat_line_before <"$stat_path" 2>/dev/null; then
        if [[ -d "/proc/$process_id" ]]; then
          read_failures=$((read_failures + 1))
        else
          listed=$((listed - 1))
        fi
        continue
      fi
      if ! process_name=$(awk '$1 == "Name:" { sub(/^[^:]+:[[:space:]]*/, ""); print; exit }' \
        "$status_path" 2>/dev/null) ||
        ! rss_kib=$(awk '$1 == "VmRSS:" { print $2; exit }' "$status_path" 2>/dev/null) ||
        ! IFS= read -r stat_line_after <"$stat_path" 2>/dev/null; then
        if [[ -d "/proc/$process_id" ]]; then
          read_failures=$((read_failures + 1))
        else
          listed=$((listed - 1))
        fi
        continue
      fi
      stat_rest_before=${stat_line_before##*) }
      stat_rest_after=${stat_line_after##*) }
      start_ticks_before=$(awk '{ print $20 }' <<<"$stat_rest_before")
      start_ticks_after=$(awk '{ print $20 }' <<<"$stat_rest_after")
      if [[ ! "$rss_kib" =~ ^[0-9]+$ || ! "$start_ticks_before" =~ ^[0-9]+$ ||
        "$start_ticks_before" != "$start_ticks_after" ]]; then
        if [[ -d "/proc/$process_id" ]]; then
          read_failures=$((read_failures + 1))
        else
          listed=$((listed - 1))
        fi
        continue
      fi
      rss_bytes=$((rss_kib * 1024))
      printf '%s,%s,%s,%s,' "$sample_ns" "$sample_utc" "$process_id" "$start_ticks_before"
      csv_quote "$process_name"
      printf ',%s\n' "$rss_bytes"
      rss_sum=$((rss_sum + rss_bytes))
      sampled=$((sampled + 1))
    done
    sample_complete=0
    ((listed > 0 && read_failures == 0 && listed == sampled)) && sample_complete=1
    printf '%s,%s,%s,%s,%s,%s\n' \
      "$sample_ns" "$sample_utc" "$listed" "$sampled" "$rss_sum" "$sample_complete" \
      >>"$output_dir/process-rss-total.csv"
    sleep "$sample_interval_seconds"
  done
) >>"$output_dir/process-rss.csv" &
sampler_pids+=("$!")

(
  while :; do
    sample_ns=$(wall_time_unix_ns) || exit $?
    sample_utc=$(timestamp_utc)
    query_ok=1
    process_count=0
    target_count=0
    competing_count=0
    target_memory_mib=0
    target_memory_available=1
    if ! process_rows=$(nvidia-smi \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>/dev/null); then
      query_ok=0
      process_rows=
    fi
    while IFS=',' read -r process_id process_name used_memory_mib; do
      process_id=$(sed 's/^[[:space:]]*//;s/[[:space:]]*$//' <<<"$process_id")
      [[ -n "$process_id" ]] || continue
      process_name=$(sed 's/^[[:space:]]*//;s/[[:space:]]*$//' <<<"$process_name")
      used_memory_mib=$(sed 's/^[[:space:]]*//;s/[[:space:]]*$//' <<<"$used_memory_mib")
      process_count=$((process_count + 1))
      in_target=0
      if pid_is_in_target_cgroup "$process_id"; then
        in_target=1
        target_count=$((target_count + 1))
        if [[ "$used_memory_mib" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
          if ((target_memory_available)); then
            target_memory_mib=$(awk -v total="$target_memory_mib" -v value="$used_memory_mib" \
              'BEGIN { printf "%.6f", total + value }')
          fi
        else
          target_memory_available=0
        fi
      else
        competing_count=$((competing_count + 1))
      fi
      {
        printf '%s,%s,%s,%s,' "$sample_ns" "$sample_utc" "$process_id" "$in_target"
        csv_quote "$process_name"
        printf ',%s\n' "$used_memory_mib"
      } >>"$output_dir/gpu-processes.csv"
    done <<<"$process_rows"
    ((target_memory_available)) || target_memory_mib=NA
    printf '%s,%s,%s,%s,%s,%s,%s\n' \
      "$sample_ns" "$sample_utc" "$query_ok" "$process_count" "$target_count" \
      "$competing_count" "$target_memory_mib"
    sleep "$sample_interval_seconds"
  done
) >>"$output_dir/gpu-process-summary.csv" &
sampler_pids+=("$!")

# ShellCheck does not resolve function names registered through trap.
# shellcheck disable=SC2329
cleanup() {
  local sampler_pid
  for sampler_pid in "${sampler_pids[@]}"; do
    kill "$sampler_pid" 2>/dev/null || true
  done
  for sampler_pid in "${sampler_pids[@]}"; do
    wait "$sampler_pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

samplers_are_alive() {
  local sampler_pid
  for sampler_pid in "${sampler_pids[@]}"; do
    kill -0 "$sampler_pid" 2>/dev/null || return 1
  done
}

wait_for_initial_samples() {
  local attempt line_count path ready
  for ((attempt = 0; attempt < 200; attempt++)); do
    samplers_are_alive || return 1
    ready=1
    for path in gpu.csv system.csv process-rss-total.csv gpu-process-summary.csv; do
      line_count=$(wc -l <"$output_dir/$path")
      ((line_count >= 2)) || ready=0
    done
    ((ready)) && return 0
    sleep 0.05
  done
  return 1
}

wait_for_samples_after() {
  local boundary_ns=$1
  local attempt latest_ns path ready
  for ((attempt = 0; attempt < 200; attempt++)); do
    samplers_are_alive || return 1
    ready=1
    for path in gpu.csv system.csv process-rss-total.csv gpu-process-summary.csv; do
      latest_ns=$(tail -n 1 "$output_dir/$path" | awk -F, '{ print $1 }')
      [[ "$latest_ns" =~ ^[0-9]+$ ]] || {
        ready=0
        continue
      }
      ((latest_ns >= boundary_ns)) || ready=0
    done
    ((ready)) && return 0
    sleep 0.05
  done
  return 1
}

wait_for_initial_samples || {
  echo "telemetry samplers did not produce initial rows within 10 seconds" >&2
  exit 70
}
baseline_start_ns=$(wall_time_unix_ns)
printf 'idle_baseline_start_wall_time_unix_ns=%s\n' "$baseline_start_ns" >>"$output_dir/run.txt"
sleep "$idle_baseline_seconds"
baseline_end_ns=$(wall_time_unix_ns)
printf 'idle_baseline_end_wall_time_unix_ns=%s\n' "$baseline_end_ns" >>"$output_dir/run.txt"
wait_for_samples_after "$baseline_end_ns" || {
  echo "telemetry samplers did not bracket the idle baseline end within 10 seconds" >&2
  exit 70
}

set +e
"$@" >"$output_dir/command.stdout" 2>"$output_dir/command.stderr"
status=$?
set -e

command_finished_ns=$(wall_time_unix_ns)
printf 'command_finished_at=%s\ncommand_finished_wall_time_unix_ns=%s\nexit_status=%s\n' \
  "$(date --iso-8601=ns)" "$command_finished_ns" "$status" \
  >>"$output_dir/run.txt"
wait_for_samples_after "$command_finished_ns" || {
  echo "telemetry samplers did not bracket the wrapped command end within 10 seconds" >&2
  exit 70
}
printf 'finished_at=%s\nfinished_wall_time_unix_ns=%s\n' \
  "$(date --iso-8601=ns)" "$(wall_time_unix_ns)" >>"$output_dir/run.txt"
exit "$status"
