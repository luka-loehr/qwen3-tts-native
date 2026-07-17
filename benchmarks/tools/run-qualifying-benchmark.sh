#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  run-qualifying-benchmark.sh \
    --output-dir DIR --engine native|sglang --profile B1|B3|B6 --round N \
    --container NAME --image IMAGE --client FILE --workload FILE \
    --endpoint LOOPBACK_URL --requests N --warmups N \
    --idle-baseline-seconds N --evidence-prefix MANIFEST_RELATIVE_POSIX_PATH \
    [--sglang-model MODEL] [--timeout-seconds N] \
    [--sample-interval-ms N] [--gpu-index N]

Runs one production-qualifying client scenario against an already-running
container. IMAGE is resolved to a content-addressed local image ID and must equal
the image backing CONTAINER. The final output path is published atomically only
after telemetry reduction and hashing succeed.
USAGE
}

die() {
  echo "$*" >&2
  exit 65
}

output_dir=
engine=
profile=
round=
container_name=
image_reference=
client_path=
workload_path=
endpoint=
requests=
warmups=
idle_baseline_seconds=
evidence_prefix=
sglang_model=
timeout_seconds=600
sample_interval_ms=100
gpu_index=0

while (($#)); do
  case "$1" in
    --output-dir|--engine|--profile|--round|--container|--image|--client|--workload|--endpoint|--requests|--warmups|--idle-baseline-seconds|--evidence-prefix|--sglang-model|--timeout-seconds|--sample-interval-ms|--gpu-index)
      (($# >= 2)) || { usage; exit 64; }
      option=$1
      value=$2
      shift 2
      case "$option" in
        --output-dir) output_dir=$value ;;
        --engine) engine=$value ;;
        --profile) profile=$value ;;
        --round) round=$value ;;
        --container) container_name=$value ;;
        --image) image_reference=$value ;;
        --client) client_path=$value ;;
        --workload) workload_path=$value ;;
        --endpoint) endpoint=$value ;;
        --requests) requests=$value ;;
        --warmups) warmups=$value ;;
        --idle-baseline-seconds) idle_baseline_seconds=$value ;;
        --evidence-prefix) evidence_prefix=$value ;;
        --sglang-model) sglang_model=$value ;;
        --timeout-seconds) timeout_seconds=$value ;;
        --sample-interval-ms) sample_interval_ms=$value ;;
        --gpu-index) gpu_index=$value ;;
      esac
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

for required_value in \
  "$output_dir" "$engine" "$profile" "$round" "$container_name" "$image_reference" \
  "$client_path" "$workload_path" "$endpoint" "$requests" "$warmups" \
  "$idle_baseline_seconds" "$evidence_prefix"; do
  [[ -n "$required_value" ]] || { usage; exit 64; }
done

[[ "$engine" == native || "$engine" == sglang ]] || die "engine must be native or sglang"
[[ "$profile" =~ ^B(1|3|6)$ ]] || die "profile must be B1, B3, or B6"
[[ "$round" =~ ^[1-9][0-9]*$ ]] || die "round must be a positive integer"
[[ "$requests" =~ ^[0-9]+$ && "$requests" -ge 200 ]] || die "requests must be an integer of at least 200"
[[ "$warmups" =~ ^[0-9]+$ && "$warmups" -ge 24 ]] || die "warmups must be an integer of at least 24"
[[ "$idle_baseline_seconds" =~ ^[0-9]+$ && "$idle_baseline_seconds" -ge 15 ]] || die "idle baseline must be at least 15 seconds"
[[ "$timeout_seconds" =~ ^[1-9][0-9]*$ ]] || die "timeout must be a positive integer"
[[ "$sample_interval_ms" =~ ^[0-9]+$ && "$sample_interval_ms" -ge 20 && "$sample_interval_ms" -le 100 ]] || die "sample interval must be from 20 through 100 ms"
[[ "$gpu_index" =~ ^[0-9]+$ ]] || die "GPU index must be a non-negative integer"
if [[ "$endpoint" =~ ^http://(127\.0\.0\.1|localhost|\[::1\]):([1-9][0-9]{0,4})(/[^[:space:]?#]*)?$ ]]; then
  endpoint_port=${BASH_REMATCH[2]}
  ((endpoint_port <= 65535)) || die "endpoint port must be from 1 through 65535"
else
  die "endpoint must be an HTTP loopback URL without query parameters or fragments"
fi
[[ "$evidence_prefix" != /* && "$evidence_prefix" != */ && "$evidence_prefix" != *//* && \
  "$evidence_prefix" != *\\* && "$evidence_prefix" != *$'\n'* ]] || die "evidence prefix must be a normalized relative POSIX path"
IFS=/ read -r -a evidence_components <<<"$evidence_prefix"
for component in "${evidence_components[@]}"; do
  [[ -n "$component" && "$component" != . && "$component" != .. ]] || die "evidence prefix contains an unsafe path component"
done
if [[ "$engine" == sglang ]]; then
  [[ -n "$sglang_model" ]] || die "--sglang-model is required for the sglang engine"
else
  [[ -z "$sglang_model" ]] || die "--sglang-model is only valid for the sglang engine"
fi

for command_name in \
  awk basename chmod cp date dirname docker find git install jq mkdir mv nvidia-smi \
  readlink realpath sha256sum sort stat uname wc; do
  command -v "$command_name" >/dev/null || die "required command is unavailable: $command_name"
done

script_path=$(readlink -f "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_path")
capture_path="$script_dir/capture-spark-telemetry.sh"
reducer_path="$script_dir/reduce-spark-run.sh"
[[ -x "$capture_path" && -x "$reducer_path" ]] || die "benchmark collector or reducer is not executable"
[[ -f "$client_path" && ! -L "$client_path" && -x "$client_path" ]] || die "client must be an executable regular file, not a symlink"
[[ -f "$workload_path" && ! -L "$workload_path" && -s "$workload_path" ]] || die "workload must be a non-empty regular file, not a symlink"

output_dir=$(realpath -m "$output_dir")
output_parent=$(dirname "$output_dir")
output_name=$(basename "$output_dir")
mkdir -p "$output_parent"
[[ -d "$output_parent" && ! -L "$output_parent" ]] || die "output parent is missing or is a symlink"
[[ ! -e "$output_dir" ]] || die "output path already exists: $output_dir"
staging_dir="$output_parent/.${output_name}.partial.$$"
[[ ! -e "$staging_dir" ]] || die "staging path already exists: $staging_dir"

published=0
preserve_failed_run() {
  local status=$?
  local failed_path timestamp
  if ((status != 0 && published == 0)) && [[ -d "$staging_dir" ]]; then
    timestamp=$(date --utc +%Y%m%dT%H%M%SZ)
    failed_path="$output_dir.failed.$timestamp.$$"
    if [[ ! -e "$failed_path" ]]; then
      mv "$staging_dir" "$failed_path"
      echo "failed run preserved at: $failed_path" >&2
    else
      echo "failed run remains at staging path: $staging_dir" >&2
    fi
  fi
  exit "$status"
}
trap preserve_failed_run EXIT

mkdir -p "$staging_dir/provenance" "$staging_dir/input"
copied_client="$staging_dir/input/qwen3-tts-http-bench"
copied_workload="$staging_dir/input/workload.jsonl"
install -m 0555 "$client_path" "$copied_client"
install -m 0444 "$workload_path" "$copied_workload"
install -m 0444 "$script_path" "$staging_dir/provenance/run-qualifying-benchmark.sh"
copied_capture="$staging_dir/provenance/capture-spark-telemetry.sh"
copied_reducer="$staging_dir/provenance/reduce-spark-run.sh"
install -m 0555 "$capture_path" "$copied_capture"
install -m 0555 "$reducer_path" "$copied_reducer"

original_client_sha256=$(sha256sum "$client_path" | awk '{ print $1 }')
copied_client_sha256=$(sha256sum "$copied_client" | awk '{ print $1 }')
original_workload_sha256=$(sha256sum "$workload_path" | awk '{ print $1 }')
copied_workload_sha256=$(sha256sum "$copied_workload" | awk '{ print $1 }')
[[ "$original_client_sha256" == "$copied_client_sha256" ]] || die "client changed while it was copied"
[[ "$original_workload_sha256" == "$copied_workload_sha256" ]] || die "workload changed while it was copied"

image_id=$(docker image inspect --format '{{.Id}}' "$image_reference")
container_image_id=$(docker container inspect --format '{{.Image}}' "$container_name")
container_id=$(docker container inspect --format '{{.Id}}' "$container_name")
container_running=$(docker container inspect --format '{{.State.Running}}' "$container_name")
container_pid=$(docker container inspect --format '{{.State.Pid}}' "$container_name")
[[ "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]] || die "image reference did not resolve to a content-addressed image ID"
[[ "$container_image_id" == "$image_id" ]] || die "container does not run the explicitly selected image"
[[ "$container_id" =~ ^[0-9a-f]{64}$ ]] || die "container ID is invalid"
[[ "$container_running" == true && "$container_pid" =~ ^[1-9][0-9]*$ ]] || die "container is not running"

docker image inspect "$image_reference" | jq '
  map({
    Id, RepoTags, RepoDigests, Created, Architecture, Os, Size,
    Config: {
      User: .Config.User,
      ExposedPorts: .Config.ExposedPorts,
      Entrypoint: .Config.Entrypoint,
      Cmd: .Config.Cmd,
      WorkingDir: .Config.WorkingDir,
      Labels: .Config.Labels,
      StopSignal: .Config.StopSignal,
      EnvironmentVariableNames: [.Config.Env[]? | split("=")[0]]
    },
    RootFS: .RootFS,
    Metadata: .Metadata
  })
' >"$staging_dir/provenance/image-inspect.json"

docker container inspect "$container_name" | jq '
  map({
    Id, Created, Path, Args, State: {
      Status: .State.Status,
      Running: .State.Running,
      Pid: .State.Pid,
      StartedAt: .State.StartedAt
    },
    Image,
    Config: {
      Hostname: .Config.Hostname,
      User: .Config.User,
      ExposedPorts: .Config.ExposedPorts,
      Entrypoint: .Config.Entrypoint,
      Cmd: .Config.Cmd,
      WorkingDir: .Config.WorkingDir,
      Labels: .Config.Labels,
      StopSignal: .Config.StopSignal,
      EnvironmentVariableNames: [.Config.Env[]? | split("=")[0]]
    },
    HostConfig: {
      Runtime: .HostConfig.Runtime,
      NetworkMode: .HostConfig.NetworkMode,
      Memory: .HostConfig.Memory,
      MemorySwap: .HostConfig.MemorySwap,
      NanoCpus: .HostConfig.NanoCpus,
      PidsLimit: .HostConfig.PidsLimit,
      ReadonlyRootfs: .HostConfig.ReadonlyRootfs,
      DeviceRequests: .HostConfig.DeviceRequests,
      PortBindings: .HostConfig.PortBindings
    },
    Mounts: [.Mounts[]? | {Type, Destination, Mode, RW, Propagation}],
    NetworkSettings: {Ports: .NetworkSettings.Ports}
  })
' >"$staging_dir/provenance/container-inspect.sanitized.json"

"$copied_client" --version >"$staging_dir/provenance/client-version.txt"
uname -a >"$staging_dir/provenance/uname.txt"
nvidia-smi -L >"$staging_dir/provenance/nvidia-smi-list.txt"
nvidia-smi -q >"$staging_dir/provenance/nvidia-smi-query.txt"
docker version >"$staging_dir/provenance/docker-version.txt"
if [[ -f /etc/os-release ]]; then
  cp /etc/os-release "$staging_dir/provenance/os-release.txt"
fi

repository_root=$(realpath "$script_dir/../..")
repository_commit=
repository_clean=false
if git -C "$repository_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  repository_commit=$(git -C "$repository_root" rev-parse HEAD)
  git -C "$repository_root" status --short --branch >"$staging_dir/provenance/repository-status.txt"
  if [[ -z "$(git -C "$repository_root" status --porcelain --untracked-files=no)" ]]; then
    repository_clean=true
  fi
fi

jq -n \
  --arg schema_version qwen3-tts-qualifying-run/v1 \
  --arg engine "$engine" --arg profile "$profile" --argjson round "$round" \
  --arg container_name "$container_name" --arg container_id "$container_id" \
  --arg image_reference "$image_reference" --arg image_id "$image_id" \
  --arg endpoint "$endpoint" --argjson requests "$requests" --argjson warmups "$warmups" \
  --argjson idle_baseline_seconds "$idle_baseline_seconds" \
  --argjson timeout_seconds "$timeout_seconds" --argjson sample_interval_ms "$sample_interval_ms" \
  --argjson gpu_index "$gpu_index" --arg sglang_model "$sglang_model" \
  --arg evidence_prefix "$evidence_prefix" \
  --arg client_sha256 "$copied_client_sha256" --arg workload_sha256 "$copied_workload_sha256" \
  --arg repository_commit "$repository_commit" --argjson repository_clean "$repository_clean" '
  {
    schema_version: $schema_version,
    engine: $engine,
    profile: $profile,
    round: $round,
    container: {name: $container_name, id: $container_id},
    image: {reference: $image_reference, resolved_id: $image_id},
    client: {path: "input/qwen3-tts-http-bench", sha256: $client_sha256},
    workload: {path: "input/workload.jsonl", sha256: $workload_sha256},
    evidence_prefix: $evidence_prefix,
    request: {
      endpoint: $endpoint,
      requests: $requests,
      warmups: $warmups,
      timeout_seconds: $timeout_seconds,
      sglang_model: (if $sglang_model == "" then null else $sglang_model end)
    },
    telemetry: {
      idle_baseline_seconds: $idle_baseline_seconds,
      configured_sample_interval_ms: $sample_interval_ms,
      maximum_qualifying_observed_gap_ms: 200,
      gpu_index: $gpu_index
    },
    tooling_repository: {
      commit: (if $repository_commit == "" then null else $repository_commit end),
      tracked_files_clean: $repository_clean
    }
  }
' >"$staging_dir/provenance/invocation.json"

client_profile=native
[[ "$engine" == sglang ]] && client_profile=sglang-omni
client_command=(
  "$copied_client"
  --endpoint "$endpoint"
  --profile "$client_profile"
  --workload "$copied_workload"
  --output-dir "$staging_dir/client"
  --phase-events "$staging_dir/raw/phase-events.jsonl"
  --requests "$requests"
  --warmups "$warmups"
  --concurrency "$profile"
  --timeout-seconds "$timeout_seconds"
)
if [[ "$engine" == sglang ]]; then
  client_command+=(--sglang-model "$sglang_model")
fi

"$copied_capture" \
  --output-dir "$staging_dir/raw" \
  --container "$container_name" \
  --idle-baseline-seconds "$idle_baseline_seconds" \
  --sample-interval-ms "$sample_interval_ms" \
  --gpu-index "$gpu_index" \
  -- "${client_command[@]}"

for path in \
  "$staging_dir/client/summary.json" "$staging_dir/client/requests.jsonl" \
  "$staging_dir/client/packets.jsonl"; do
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "client did not produce complete canonical evidence: $path"
done

"$copied_reducer" \
  --run-dir "$staging_dir" \
  --engine "$engine" \
  --profile "$profile" \
  --round "$round" \
  --evidence-prefix "$evidence_prefix"

(
  cd "$staging_dir"
  while IFS= read -r -d '' path; do
    sha256sum "${path#./}"
  done < <(find . -type f ! -name SHA256SUMS -print0 | sort -z)
) >"$staging_dir/SHA256SUMS"

(
  cd "$staging_dir"
  sha256sum --check --strict SHA256SUMS >/dev/null
)

mv "$staging_dir" "$output_dir"
published=1
trap - EXIT
printf 'qualifying run published at: %s\n' "$output_dir"
