#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
reducer=$(realpath "$script_dir/../reduce-spark-run.sh")
temporary_dir=$(mktemp -d)
trap 'rm -rf "$temporary_dir"' EXIT

base_ns=1700000000000000000
idle_start_ns=$((base_ns + 100000000))
idle_end_ns=$((idle_start_ns + 15000000000))
warmup_start_ns=$((idle_end_ns + 100000000))
warmup_end_ns=$((warmup_start_ns + 1000000000))
measured_start_ns=$((warmup_end_ns + 100000000))
measured_end_ns=$((measured_start_ns + 2000000000))
telemetry_end_ns=$((measured_end_ns + 100000000))

create_fixture() {
  local directory=$1
  local competing=$2
  local mode=$3
  local nanoseconds power process_count rss_complete rss_sampled sample_competing timestamp_utc
  mkdir -p "$directory/raw" "$directory/client"

  printf '%s\n' \
    'wall_time_unix_ns,timestamp_utc,gpu_index,gpu_uuid,pstate,temperature_c,gpu_util_percent,memory_util_percent,power_w,graphics_clock_mhz' \
    >"$directory/raw/gpu.csv"
  printf '%s\n' \
    'wall_time_unix_ns,timestamp_utc,uptime_s,cgroup_memory_bytes,cgroup_memory_peak_bytes,cgroup_pids,cgroup_cpu_usec,host_mem_available_kib,host_swap_free_kib' \
    >"$directory/raw/system.csv"
  printf '%s\n' \
    'wall_time_unix_ns,timestamp_utc,pid,process_start_ticks,process_name,rss_bytes' \
    >"$directory/raw/process-rss.csv"
  printf '%s\n' \
    'wall_time_unix_ns,timestamp_utc,listed_processes,sampled_processes,process_rss_sum_bytes,sample_complete' \
    >"$directory/raw/process-rss-total.csv"
  printf '%s\n' \
    'wall_time_unix_ns,timestamp_utc,pid,in_target_container,process_name,unified_memory_mib' \
    >"$directory/raw/gpu-processes.csv"
  printf '%s\n' \
    'wall_time_unix_ns,timestamp_utc,query_ok,gpu_compute_processes,target_container_gpu_processes,competing_cuda_processes,target_unified_memory_mib' \
    >"$directory/raw/gpu-process-summary.csv"

  for ((nanoseconds = base_ns; nanoseconds <= telemetry_end_ns; nanoseconds += 100000000)); do
    if [[ "$mode" == gap && \
      ("$nanoseconds" == $((measured_start_ns + 500000000)) || \
       "$nanoseconds" == $((measured_start_ns + 600000000))) ]]; then
      continue
    fi
    power=50
    ((nanoseconds >= measured_start_ns)) && power=100
    timestamp_utc=fixture
    [[ "$mode" == comma-timestamp ]] && timestamp_utc='2026-07-17T15:05:56,123456789Z'
    sample_competing=$competing
    if [[ "$mode" == warmup-competitor ]]; then
      sample_competing=0
      if ((nanoseconds >= warmup_start_ns && nanoseconds < measured_start_ns)); then
        sample_competing=1
      fi
    fi
    printf '%s,%s,0,GPU-fixture,P0,40,50,20,%s,1000\n' "$nanoseconds" "$timestamp_utc" "$power" \
      >>"$directory/raw/gpu.csv"
    printf '%s,%s,1.0,2000,2500,2,1000,100000,200000\n' "$nanoseconds" "$timestamp_utc" \
      >>"$directory/raw/system.csv"
    printf '%s,%s,101,1,"server-main",400\n' "$nanoseconds" "$timestamp_utc" \
      >>"$directory/raw/process-rss.csv"
    printf '%s,%s,102,2,"server-worker",600\n' "$nanoseconds" "$timestamp_utc" \
      >>"$directory/raw/process-rss.csv"
    rss_sampled=2
    rss_complete=1
    if [[ "$mode" == incomplete-rss && \
      "$nanoseconds" == $((measured_start_ns + 500000000)) ]]; then
      rss_sampled=1
      rss_complete=0
    fi
    printf '%s,%s,2,%s,1000,%s\n' \
      "$nanoseconds" "$timestamp_utc" "$rss_sampled" "$rss_complete" \
      >>"$directory/raw/process-rss-total.csv"
    printf '%s,%s,101,1,"server-main",4096\n' "$nanoseconds" "$timestamp_utc" \
      >>"$directory/raw/gpu-processes.csv"
    process_count=1
    if ((sample_competing)); then
      printf '%s,%s,999,0,"competitor",1024\n' "$nanoseconds" "$timestamp_utc" \
        >>"$directory/raw/gpu-processes.csv"
      process_count=2
    fi
    printf '%s,%s,1,%s,1,%s,4096\n' \
      "$nanoseconds" "$timestamp_utc" "$process_count" "$sample_competing" \
      >>"$directory/raw/gpu-process-summary.csv"
  done

  {
    printf 'container=fixture\n'
    printf 'container_pid=1\n'
    printf 'cgroup=/fixture\n'
    printf 'started_at=fixture\n'
    printf 'started_wall_time_unix_ns=%s\n' "$base_ns"
    printf 'sample_interval_ms=100\n'
    printf 'maximum_qualifying_gap_ms=200\n'
    printf 'idle_baseline_seconds=15\n'
    printf 'idle_baseline_start_wall_time_unix_ns=%s\n' "$idle_start_ns"
    printf 'idle_baseline_end_wall_time_unix_ns=%s\n' "$idle_end_ns"
    printf 'exit_status=0\n'
  } >"$directory/raw/run.txt"
  : >"$directory/raw/command.stdout"
  : >"$directory/raw/command.stderr"

  {
    printf '{"schema_version":"qwen3-tts-http-bench-phase-events/v1","sequence":0,"event":"warmup_start","wall_time_unix_ns":%s,"monotonic_elapsed_ns":0}\n' \
      "$warmup_start_ns"
    printf '{"schema_version":"qwen3-tts-http-bench-phase-events/v1","sequence":1,"event":"warmup_end","wall_time_unix_ns":%s,"monotonic_elapsed_ns":1000000000}\n' \
      "$warmup_end_ns"
    printf '{"schema_version":"qwen3-tts-http-bench-phase-events/v1","sequence":2,"event":"measured_start","wall_time_unix_ns":%s,"monotonic_elapsed_ns":1100000000}\n' \
      "$measured_start_ns"
    printf '{"schema_version":"qwen3-tts-http-bench-phase-events/v1","sequence":3,"event":"measured_end","wall_time_unix_ns":%s,"monotonic_elapsed_ns":3100000000}\n' \
      "$measured_end_ns"
  } >"$directory/raw/phase-events.jsonl"
  if [[ "$mode" == extra-phase ]]; then
    printf '{"schema_version":"qwen3-tts-http-bench-phase-events/v1","sequence":4,"event":"measured_end","wall_time_unix_ns":%s,"monotonic_elapsed_ns":3100000000}\n' \
      "$measured_end_ns" >>"$directory/raw/phase-events.jsonl"
  fi

  jq -n '{
    schema_version: "qwen3-tts-http-bench/v1",
    backend: "native",
    concurrency: "B1",
    warmups: 24,
    planned_requests: 200,
    completed_requests: 200,
    successful_requests: 200,
    failed_requests: 0,
    natural_eos_requests: 200,
    length_limited_requests: 0,
    eos_unknown_requests: 0,
    sampling_parity_qualifying_requests: 200,
    sampling_parity_non_qualifying_requests: 0,
    benchmark_wall_seconds: 2.0
  }' >"$directory/client/summary.json"
}

valid_dir="$temporary_dir/valid"
create_fixture "$valid_dir" 0 valid
bash "$reducer" --run-dir "$valid_dir" --engine native --profile B1 --round 1 \
  --evidence-prefix runs/round-01/native/B1
jq -e '
  .engine_id == "native" and .profile_id == "B1" and .round == 1 and
  .process_rss_peak_bytes == 1000 and
  .gpu_unified_memory_peak_bytes == 4294967296 and
  .average_power_w == 100 and .peak_power_w == 100 and .energy_j == 100 and
  .sampling_interval_ms == 100 and .competing_cuda_processes == 0 and
  (.telemetry_evidence_paths | all(startswith("runs/round-01/native/B1/raw/")))
' "$valid_dir/run-resource.json" >/dev/null
jq -e '
  .power.idle_average_power_w == 50 and
  .power.idle_gross_energy_j == 750 and
  .power.measured_gross_energy_j == 200 and
  .power.measured_idle_adjusted_energy_j == 100 and
  .memory.cgroup_memory_current_peak_bytes == 2000
' "$valid_dir/resource-audit.json" >/dev/null

competing_dir="$temporary_dir/competing"
create_fixture "$competing_dir" 1 valid
if bash "$reducer" --run-dir "$competing_dir" --engine native --profile B1 --round 1 \
  --evidence-prefix runs/round-01/native/B1 >/dev/null 2>&1; then
  echo "expected competing-CUDA fixture to fail" >&2
  exit 1
fi

warmup_competing_dir="$temporary_dir/warmup-competing"
create_fixture "$warmup_competing_dir" 0 warmup-competitor
if bash "$reducer" --run-dir "$warmup_competing_dir" --engine native --profile B1 --round 1 \
  --evidence-prefix runs/round-01/native/B1 >/dev/null 2>&1; then
  echo "expected warmup-only competing-CUDA fixture to fail" >&2
  exit 1
fi

phase_dir="$temporary_dir/extra-phase"
create_fixture "$phase_dir" 0 extra-phase
if bash "$reducer" --run-dir "$phase_dir" --engine native --profile B1 --round 1 \
  --evidence-prefix runs/round-01/native/B1 >/dev/null 2>&1; then
  echo "expected extra-phase fixture to fail" >&2
  exit 1
fi

gap_dir="$temporary_dir/gap"
create_fixture "$gap_dir" 0 gap
if bash "$reducer" --run-dir "$gap_dir" --engine native --profile B1 --round 1 \
  --evidence-prefix runs/round-01/native/B1 >/dev/null 2>&1; then
  echo "expected telemetry-gap fixture to fail" >&2
  exit 1
fi

incomplete_rss_dir="$temporary_dir/incomplete-rss"
create_fixture "$incomplete_rss_dir" 0 incomplete-rss
if bash "$reducer" --run-dir "$incomplete_rss_dir" --engine native --profile B1 --round 1 \
  --evidence-prefix runs/round-01/native/B1 >/dev/null 2>&1; then
  echo "expected incomplete process-RSS fixture to fail" >&2
  exit 1
fi

comma_timestamp_dir="$temporary_dir/comma-timestamp"
create_fixture "$comma_timestamp_dir" 0 comma-timestamp
if bash "$reducer" --run-dir "$comma_timestamp_dir" --engine native --profile B1 --round 1 \
  --evidence-prefix runs/round-01/native/B1 >/dev/null 2>&1; then
  echo "expected comma-timestamp fixture to fail" >&2
  exit 1
fi

echo "reduce-spark-run fixtures passed"
