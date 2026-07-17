#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=benchmarks/tools/lib/process-rss-sampler.sh
source "$script_dir/../lib/process-rss-sampler.sh"

temporary_dir=$(mktemp -d)
trap 'rm -rf "$temporary_dir"' EXIT

base_ns=1700000000000000000
clock_path="$temporary_dir/clock"

reset_clock() {
  printf '0\n' >"$clock_path"
}

wall_time_unix_ns() {
  local tick
  tick=$(<"$clock_path")
  tick=$((tick + 1))
  printf '%s\n' "$tick" >"$clock_path"
  printf '%s\n' "$((base_ns + tick * 1000000))"
}

timestamp_utc() {
  printf 'fixture-attempt-%s\n' "$(<"$clock_path")"
}

stat_line() {
  local pid=$1
  local name=$2
  local start_ticks=$3
  local field
  printf '%s (%s) S' "$pid" "$name"
  for ((field = 2; field < 20; field++)); do
    printf ' 0'
  done
  printf ' %s\n' "$start_ticks"
}

create_valid_process() {
  local proc_root=$1
  local pid=$2
  local name=$3
  local rss_kib=$4
  local start_ticks=$5
  mkdir -p "$proc_root/$pid"
  printf 'Name:\t%s\nVmRSS:\t%s kB\n' "$name" "$rss_kib" >"$proc_root/$pid/status"
  stat_line "$pid" "$name" "$start_ticks" >"$proc_root/$pid/stat"
}

assert_file_equals() {
  local expected=$1
  local path=$2
  local actual
  actual=$(<"$path")
  if [[ "$actual" != "$expected" ]]; then
    printf 'unexpected content in %s\nexpected: %s\nactual:   %s\n' \
      "$path" "$expected" "$actual" >&2
    exit 1
  fi
}

run_transient_exit_case() {
  local directory="$temporary_dir/transient"
  local proc_root="$directory/proc"
  local cgroup_procs="$directory/cgroup.procs"
  local detail="$directory/process-rss.csv"
  local total="$directory/process-rss-total.csv"
  local stderr_path="$directory/stderr"
  local fifo="$proc_root/102/stat"
  local short_lived_stat writer_pid expected_ns

  mkdir -p "$proc_root/102"
  create_valid_process "$proc_root" 101 server-main 100 10001
  printf 'Name:\tshort-lived\n' >"$proc_root/102/status"
  mkfifo "$fifo"
  short_lived_stat=$(stat_line 102 short-lived 10002)
  printf '101\n102\n' >"$cgroup_procs"
  : >"$detail"
  : >"$total"
  reset_clock

  # The first attempt has already enumerated PID 102 when it leaves the cgroup.
  # Its malformed status makes that attempt incomplete; the fresh retry sees
  # only PID 101. The FIFO makes the ordering deterministic without sleeps.
  (
    printf '%s\n' "$short_lived_stat" >"$fifo"
    printf '101\n' >"$cgroup_procs"
    printf '%s\n' "$short_lived_stat" >"$fifo"
  ) &
  writer_pid=$!

  sample_process_rss_cycle \
    "$cgroup_procs" "$proc_root" "$base_ns" 100000000 "$detail" "$total" 3 \
    2>"$stderr_path"
  wait "$writer_pid"

  expected_ns=$((base_ns + 2000000))
  assert_file_equals \
    "$expected_ns,fixture-attempt-2,101,10001,\"server-main\",102400" "$detail"
  assert_file_equals \
    "$expected_ns,fixture-attempt-2,1,1,102400,1" "$total"
  [[ ! -s "$stderr_path" ]] || {
    echo "transient process exit produced stderr" >&2
    exit 1
  }
  [[ "$(<"$clock_path")" == 2 ]] || {
    echo "transient process exit did not use exactly two attempts" >&2
    exit 1
  }
}

run_persistent_failure_case() {
  local directory="$temporary_dir/persistent"
  local proc_root="$directory/proc"
  local cgroup_procs="$directory/cgroup.procs"
  local detail="$directory/process-rss.csv"
  local total="$directory/process-rss-total.csv"
  local stderr_path="$directory/stderr"
  local expected_ns

  mkdir -p "$proc_root/202"
  create_valid_process "$proc_root" 201 server-main 200 20001
  printf 'Name:\tbroken-worker\n' >"$proc_root/202/status"
  stat_line 202 broken-worker 20002 >"$proc_root/202/stat"
  printf '201\n202\n' >"$cgroup_procs"
  : >"$detail"
  : >"$total"
  reset_clock

  sample_process_rss_cycle \
    "$cgroup_procs" "$proc_root" "$base_ns" 100000000 "$detail" "$total" 3 \
    2>"$stderr_path"

  expected_ns=$((base_ns + 3000000))
  assert_file_equals \
    "$expected_ns,fixture-attempt-3,201,20001,\"server-main\",204800" "$detail"
  assert_file_equals \
    "$expected_ns,fixture-attempt-3,2,1,204800,0" "$total"
  [[ ! -s "$stderr_path" ]] || {
    echo "persistent process read failure produced noisy stderr" >&2
    exit 1
  }
  [[ "$(<"$clock_path")" == 3 ]] || {
    echo "persistent failure did not stop after three attempts" >&2
    exit 1
  }
}

run_deadline_case() {
  local directory="$temporary_dir/deadline"
  local proc_root="$directory/proc"
  local cgroup_procs="$directory/cgroup.procs"
  local detail="$directory/process-rss.csv"
  local total="$directory/process-rss-total.csv"
  local expected_ns

  mkdir -p "$proc_root/302"
  create_valid_process "$proc_root" 301 server-main 300 30001
  printf 'Name:\tbroken-worker\n' >"$proc_root/302/status"
  stat_line 302 broken-worker 30002 >"$proc_root/302/stat"
  printf '301\n302\n' >"$cgroup_procs"
  : >"$detail"
  : >"$total"
  reset_clock

  # Attempt one starts at +1 ms. The next actual wall-time read is +2 ms,
  # exactly the cycle deadline, so no second attempt may begin.
  sample_process_rss_cycle \
    "$cgroup_procs" "$proc_root" "$base_ns" 2000000 "$detail" "$total" 3

  expected_ns=$((base_ns + 1000000))
  assert_file_equals \
    "$expected_ns,fixture-attempt-1,301,30001,\"server-main\",307200" "$detail"
  assert_file_equals \
    "$expected_ns,fixture-attempt-1,2,1,307200,0" "$total"
  [[ "$(<"$clock_path")" == 2 ]] || {
    echo "deadline case did not inspect the retry deadline exactly once" >&2
    exit 1
  }
}

run_transient_exit_case
run_persistent_failure_case
run_deadline_case

echo "process-RSS sampler retry fixtures passed"
