#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  capture-spark-telemetry.sh --output-dir DIR --container NAME -- COMMAND [ARG...]

Runs COMMAND while sampling DGX Spark and container telemetry every 200 ms.
The output directory must not already exist. Do not put secrets in COMMAND.
USAGE
}

output_dir=
container_name=

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

[[ -n "$output_dir" && -n "$container_name" && $# -gt 0 ]] || {
  usage
  exit 64
}
[[ ! -e "$output_dir" ]] || {
  echo "output directory already exists: $output_dir" >&2
  exit 73
}

for command_name in awk date docker nvidia-smi sed; do
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

mkdir -p "$output_dir"

{
  printf 'container=%s\n' "$container_name"
  printf 'container_pid=%s\n' "$container_pid"
  printf 'cgroup=%s\n' "$cgroup_relative"
  printf 'started_at=%s\n' "$(date --iso-8601=ns)"
  printf 'sample_interval_ms=200\n'
  printf 'command='
  printf '%q ' "$@"
  printf '\n'
} >"$output_dir/run.txt"

printf '%s\n' \
  'timestamp,pstate,temperature_c,gpu_util_percent,memory_util_percent,power_w,graphics_clock_mhz' \
  >"$output_dir/gpu.csv"
printf '%s\n' \
  'timestamp,uptime_s,cgroup_memory_bytes,cgroup_memory_peak_bytes,cgroup_pids,cgroup_cpu_usec,host_mem_available_kib,host_swap_free_kib' \
  >"$output_dir/system.csv"
printf '%s\n' 'timestamp,pid,process_name,unified_memory_mib' \
  >"$output_dir/gpu-processes.csv"

sampler_pids=()

nvidia-smi \
  --query-gpu=timestamp,pstate,temperature.gpu,utilization.gpu,utilization.memory,power.draw,clocks.current.graphics \
  --format=csv,noheader,nounits \
  --loop-ms=200 >>"$output_dir/gpu.csv" &
sampler_pids+=("$!")

(
  while :; do
    timestamp=$(date --iso-8601=ns)
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
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$timestamp" "$uptime_s" "$memory_bytes" "$memory_peak_bytes" \
      "$pids" "$cpu_usec" "$host_mem_kib" "$host_swap_kib"
    sleep 0.2
  done
) >>"$output_dir/system.csv" &
sampler_pids+=("$!")

(
  while :; do
    timestamp=$(date --iso-8601=ns)
    while IFS= read -r process; do
      [[ -n "$process" ]] && printf '%s,%s\n' "$timestamp" "$process"
    done < <(
      nvidia-smi \
        --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null || true
    )
    sleep 0.2
  done
) >>"$output_dir/gpu-processes.csv" &
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

set +e
"$@" >"$output_dir/command.stdout" 2>"$output_dir/command.stderr"
status=$?
set -e

printf 'finished_at=%s\nexit_status=%s\n' "$(date --iso-8601=ns)" "$status" \
  >>"$output_dir/run.txt"
exit "$status"
