#!/usr/bin/env bash

# Capture one coherent process-RSS sample for a cgroup. Callers must provide
# wall_time_unix_ns and timestamp_utc functions. Detail rows are buffered until
# an attempt finishes, so a retry can never leak abandoned rows into evidence.
sample_process_rss_cycle() {
  if (($# != 7)); then
    echo "sample_process_rss_cycle requires 7 arguments" >&2
    return 64
  fi

  local cgroup_procs_path=$1
  local proc_root=$2
  local cycle_started_ns=$3
  local sample_interval_ns=$4
  local detail_output_path=$5
  local total_output_path=$6
  local maximum_attempts=$7

  [[ "$cycle_started_ns" =~ ^[0-9]+$ && "$sample_interval_ns" =~ ^[1-9][0-9]*$ &&
    "$maximum_attempts" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid process-RSS sampling configuration" >&2
    return 64
  }

  local cycle_deadline_ns=$((cycle_started_ns + sample_interval_ns))
  local attempt attempt_ns attempt_utc process_id status_path stat_path
  local stat_line_before stat_line_after stat_rest_before stat_rest_after
  local process_name rss_kib start_ticks_before start_ticks_after rss_bytes detail_row
  local listed sampled read_failures rss_sum sample_complete
  local final_ns='' final_utc='' final_listed=0 final_sampled=0 final_rss_sum=0 final_complete=0
  local escaped_process_name
  local -a process_ids=()
  local -a attempt_rows=()
  local -a final_rows=()

  for ((attempt = 1; attempt <= maximum_attempts; attempt++)); do
    attempt_ns=$(wall_time_unix_ns) || return $?
    [[ "$attempt_ns" =~ ^[0-9]{19}$ ]] || {
      echo "process-RSS sampler received an invalid wall timestamp" >&2
      return 70
    }

    # A retry belongs to the current scheduled cycle only. The first attempt is
    # always made; later attempts never begin at or beyond the cycle deadline.
    if ((attempt > 1 && attempt_ns >= cycle_deadline_ns)); then
      break
    fi
    attempt_utc=$(timestamp_utc) || return $?

    process_ids=()
    attempt_rows=()
    read_failures=0
    if [[ -r "$cgroup_procs_path" ]]; then
      while IFS= read -r process_id || [[ -n "$process_id" ]]; do
        process_ids+=("$process_id")
      done <"$cgroup_procs_path"
    else
      read_failures=1
    fi

    listed=${#process_ids[@]}
    sampled=0
    rss_sum=0
    for process_id in "${process_ids[@]}"; do
      if [[ ! "$process_id" =~ ^[1-9][0-9]*$ ]]; then
        read_failures=$((read_failures + 1))
        continue
      fi

      status_path="$proc_root/$process_id/status"
      stat_path="$proc_root/$process_id/stat"
      if ! IFS= read -r stat_line_before 2>/dev/null <"$stat_path"; then
        read_failures=$((read_failures + 1))
        continue
      fi
      if ! process_name=$(awk '$1 == "Name:" { sub(/^[^:]+:[[:space:]]*/, ""); print; exit }' \
        "$status_path" 2>/dev/null); then
        read_failures=$((read_failures + 1))
        continue
      fi
      if ! rss_kib=$(awk '$1 == "VmRSS:" { print $2; exit }' "$status_path" 2>/dev/null); then
        read_failures=$((read_failures + 1))
        continue
      fi
      if ! IFS= read -r stat_line_after 2>/dev/null <"$stat_path"; then
        read_failures=$((read_failures + 1))
        continue
      fi

      stat_rest_before=${stat_line_before##*) }
      stat_rest_after=${stat_line_after##*) }
      start_ticks_before=$(awk '{ print $20 }' <<<"$stat_rest_before")
      start_ticks_after=$(awk '{ print $20 }' <<<"$stat_rest_after")
      if [[ -z "$process_name" || ! "$rss_kib" =~ ^[0-9]+$ ||
        ! "$start_ticks_before" =~ ^[0-9]+$ || "$start_ticks_before" != "$start_ticks_after" ]]; then
        read_failures=$((read_failures + 1))
        continue
      fi

      rss_bytes=$((rss_kib * 1024))
      escaped_process_name=${process_name//$'\r'/ }
      escaped_process_name=${escaped_process_name//$'\n'/ }
      escaped_process_name=${escaped_process_name//\"/\"\"}
      printf -v detail_row '%s,%s,%s,%s,"%s",%s' \
        "$attempt_ns" "$attempt_utc" "$process_id" "$start_ticks_before" \
        "$escaped_process_name" "$rss_bytes"
      attempt_rows+=("$detail_row")
      rss_sum=$((rss_sum + rss_bytes))
      sampled=$((sampled + 1))
    done

    sample_complete=0
    ((listed > 0 && read_failures == 0 && listed == sampled)) && sample_complete=1
    final_ns=$attempt_ns
    final_utc=$attempt_utc
    final_listed=$listed
    final_sampled=$sampled
    final_rss_sum=$rss_sum
    final_complete=$sample_complete
    final_rows=("${attempt_rows[@]}")
    ((sample_complete)) && break
  done

  # Only the accepted or final failed attempt is emitted. A failed attempt is
  # explicit and remains fatal to the qualifying reducer; no value is imputed.
  for detail_row in "${final_rows[@]}"; do
    printf '%s\n' "$detail_row" >>"$detail_output_path"
  done
  printf '%s,%s,%s,%s,%s,%s\n' \
    "$final_ns" "$final_utc" "$final_listed" "$final_sampled" "$final_rss_sum" \
    "$final_complete" >>"$total_output_path"
}
