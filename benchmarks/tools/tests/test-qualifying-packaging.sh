#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_tools_dir=$(realpath "$script_dir/..")
temporary_dir=$(mktemp -d)
trap 'rm -rf "$temporary_dir"' EXIT

fixture_repository="$temporary_dir/repository"
fixture_tools="$fixture_repository/benchmarks/tools"
fixture_bin="$temporary_dir/bin"
mkdir -p "$fixture_tools/lib" "$fixture_bin"

cp "$source_tools_dir/run-qualifying-benchmark.sh" "$fixture_tools/"
cp "$source_tools_dir/lib/process-rss-sampler.sh" "$fixture_tools/lib/"
chmod 0555 "$fixture_tools/run-qualifying-benchmark.sh"

cat >"$fixture_tools/capture-spark-telemetry.sh" <<'CAPTURE'
#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/process-rss-sampler.sh"
declare -F sample_process_rss_cycle >/dev/null

output_dir=
while (($#)); do
  case "$1" in
    --output-dir)
      output_dir=$2
      shift 2
      ;;
    --)
      break
      ;;
    *)
      shift
      ;;
  esac
done
[[ -n "$output_dir" ]]
mkdir -p "$output_dir" "$(dirname "$output_dir")/client"
printf '{}\n' >"$(dirname "$output_dir")/client/summary.json"
printf '{}\n' >"$(dirname "$output_dir")/client/requests.jsonl"
printf '{}\n' >"$(dirname "$output_dir")/client/packets.jsonl"
printf 'fixture\n' >"$output_dir/command.stdout"
: >"$output_dir/command.stderr"
CAPTURE

cat >"$fixture_tools/reduce-spark-run.sh" <<'REDUCER'
#!/usr/bin/env bash
set -euo pipefail
run_dir=
while (($#)); do
  case "$1" in
    --run-dir)
      run_dir=$2
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
[[ -n "$run_dir" ]]
printf '{}\n' >"$run_dir/run-resource.json"
printf '{}\n' >"$run_dir/resource-audit.json"
REDUCER
chmod 0555 "$fixture_tools/capture-spark-telemetry.sh" "$fixture_tools/reduce-spark-run.sh"

cat >"$fixture_bin/realpath" <<'REALPATH'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == -m ]]; then
  parent=$(/usr/bin/dirname "$2")
  name=$(/usr/bin/basename "$2")
  printf '%s/%s\n' "$(/bin/realpath "$parent")" "$name"
else
  exec /bin/realpath "$@"
fi
REALPATH

cat >"$fixture_bin/docker" <<'DOCKER'
#!/usr/bin/env bash
set -euo pipefail
case "${1:-}:${2:-}" in
  image:inspect)
    if [[ "${3:-}" == --format ]]; then
      printf '%s\n' "$FIXTURE_IMAGE_ID"
    else
      jq -n --arg image_id "$FIXTURE_IMAGE_ID" --arg reference "$FIXTURE_IMAGE_REFERENCE" '[{
        Id: $image_id,
        RepoTags: [$reference],
        RepoDigests: [],
        Created: "fixture",
        Architecture: "arm64",
        Os: "linux",
        Size: 1,
        Config: {
          User: "", ExposedPorts: null, Entrypoint: [], Cmd: [], WorkingDir: "",
          Labels: {}, StopSignal: "", Env: []
        },
        RootFS: {Type: "layers", Layers: []},
        Metadata: {}
      }]'
    fi
    ;;
  container:inspect)
    if [[ "${3:-}" == --format ]]; then
      case "$4" in
        '{{.Image}}') printf '%s\n' "$FIXTURE_IMAGE_ID" ;;
        '{{.Id}}') printf '%064d\n' 5 ;;
        '{{.State.Running}}') printf 'true\n' ;;
        '{{.State.Pid}}') printf '123\n' ;;
        *) exit 2 ;;
      esac
    else
      jq -n --arg image_id "$FIXTURE_IMAGE_ID" '[{
        Id: ("5" * 64), Created: "fixture", Path: "fixture", Args: [],
        State: {Status: "running", Running: true, Pid: 123, StartedAt: "fixture"},
        Image: $image_id,
        Config: {
          Hostname: "fixture", User: "", ExposedPorts: null, Entrypoint: [], Cmd: [],
          WorkingDir: "", Labels: {}, StopSignal: "", Env: []
        },
        HostConfig: {
          Runtime: "nvidia", NetworkMode: "host", Memory: 0, MemorySwap: 0,
          NanoCpus: 0, PidsLimit: 0, ReadonlyRootfs: false, DeviceRequests: [],
          PortBindings: null
        },
        Mounts: [], NetworkSettings: {Ports: {}}
      }]'
    fi
    ;;
  logs:*)
    shift
    timestamps=false
    since=
    until=
    container=
    while (($#)); do
      case "$1" in
        --timestamps)
          timestamps=true
          shift
          ;;
        --since)
          since=$2
          shift 2
          ;;
        --until)
          until=$2
          shift 2
          ;;
        *)
          [[ -z "$container" ]] || exit 2
          container=$1
          shift
          ;;
      esac
    done
    [[ "$timestamps" == true && "$since" =~ ^[0-9]+$ && "$until" =~ ^[0-9]+$ ]]
    ((until > since))
    [[ "$container" == fixture-native ]]
    if [[ "${FIXTURE_DOCKER_LOG_FAIL:-false}" == true ]]; then
      printf 'fixture Docker log failure\n' >&2
      exit 42
    fi
    printf '2026-07-17T00:00:00.000000000Z fixture server log\n'
    ;;
  version:*)
    printf 'fixture Docker version\n'
    ;;
  *)
    exit 2
    ;;
esac
DOCKER

cat >"$fixture_bin/date" <<'DATE'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --utc ]]; then
  shift
  exec /bin/date -u "$@"
fi
exec /bin/date "$@"
DATE

cat >"$fixture_bin/nvidia-smi" <<'NVIDIA'
#!/usr/bin/env bash
set -euo pipefail
printf 'fixture NVIDIA data for %s\n' "${1:-query}"
NVIDIA
chmod 0555 \
  "$fixture_bin/realpath" "$fixture_bin/docker" "$fixture_bin/date" \
  "$fixture_bin/nvidia-smi"

client="$temporary_dir/qwen3-tts-http-bench"
cat >"$client" <<'CLIENT'
#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-}" == --version ]]
printf 'fixture-client 1.0\n'
CLIENT
chmod 0555 "$client"
workload="$temporary_dir/workload.jsonl"
printf '{"fixture":true}\n' >"$workload"

FIXTURE_IMAGE_ID="sha256:$(printf '%064d' 7)"
export FIXTURE_IMAGE_ID
export FIXTURE_IMAGE_REFERENCE='fixture/qwen3-tts-native:candidate'
test_path="$fixture_bin:/usr/bin:/bin:/usr/sbin:/sbin"
output_dir="$temporary_dir/published-run"

PATH="$test_path" bash "$fixture_tools/run-qualifying-benchmark.sh" \
  --output-dir "$output_dir" \
  --engine native \
  --profile B1 \
  --round 1 \
  --container fixture-native \
  --image "$FIXTURE_IMAGE_REFERENCE" \
  --client "$client" \
  --workload "$workload" \
  --endpoint http://127.0.0.1:8080/v1/voice-design/speech \
  --requests 200 \
  --warmups 24 \
  --idle-baseline-seconds 15 \
  --evidence-prefix runs/round-01/native/B1

copied_sampler="$output_dir/provenance/lib/process-rss-sampler.sh"
[[ -f "$copied_sampler" && ! -L "$copied_sampler" ]] || {
  echo "published process-RSS sampler is not a regular non-symlink file" >&2
  exit 1
}
cmp "$fixture_tools/lib/process-rss-sampler.sh" "$copied_sampler"

if copied_mode=$(stat -c '%a' "$copied_sampler" 2>/dev/null) &&
  [[ "$copied_mode" =~ ^[0-9]+$ ]]; then
  :
else
  copied_mode=$(stat -f '%Lp' "$copied_sampler")
fi
[[ "$copied_mode" == 444 ]] || {
  echo "published process-RSS sampler mode is $copied_mode, expected 444" >&2
  exit 1
}

sampler_sha256=$(sha256sum "$copied_sampler" | awk '{ print $1 }')
grep -Fqx \
  "$sampler_sha256  provenance/lib/process-rss-sampler.sh" \
  "$output_dir/SHA256SUMS"
(
  cd "$output_dir"
  sha256sum --check --strict SHA256SUMS >/dev/null
)

server_log="$output_dir/provenance/server.log"
server_log_window="$output_dir/provenance/server-log-window.json"
[[ -f "$server_log" && ! -L "$server_log" ]] || {
  echo "published server log is not a regular non-symlink file" >&2
  exit 1
}
grep -Fqx \
  '2026-07-17T00:00:00.000000000Z fixture server log' \
  "$server_log"
jq -e \
  --arg container_id "$(printf '%064d' 5)" '
  .schema_version == "qwen3-tts-server-log-window/v1" and
  .container == {name: "fixture-native", id: $container_id} and
  (.since_unix_seconds | type == "number") and
  (.until_unix_seconds > .since_unix_seconds)
' "$server_log_window" >/dev/null
server_log_sha256=$(sha256sum "$server_log" | awk '{ print $1 }')
server_log_window_sha256=$(sha256sum "$server_log_window" | awk '{ print $1 }')
grep -Fqx "$server_log_sha256  provenance/server.log" "$output_dir/SHA256SUMS"
grep -Fqx \
  "$server_log_window_sha256  provenance/server-log-window.json" \
  "$output_dir/SHA256SUMS"

if FIXTURE_DOCKER_LOG_FAIL=true PATH="$test_path" \
  bash "$fixture_tools/run-qualifying-benchmark.sh" \
  --output-dir "$temporary_dir/rejected-log-run" \
  --engine native \
  --profile B1 \
  --round 1 \
  --container fixture-native \
  --image "$FIXTURE_IMAGE_REFERENCE" \
  --client "$client" \
  --workload "$workload" \
  --endpoint http://127.0.0.1:8080/v1/voice-design/speech \
  --requests 200 \
  --warmups 24 \
  --idle-baseline-seconds 15 \
  --evidence-prefix runs/round-01/native/B1 \
  >"$temporary_dir/rejected-log.stdout" \
  2>"$temporary_dir/rejected-log.stderr"; then
  echo "controller accepted a failed Docker log capture" >&2
  exit 1
fi
grep -Fq 'failed to capture bounded container logs' "$temporary_dir/rejected-log.stderr"
[[ ! -e "$temporary_dir/rejected-log-run" ]] || {
  echo "controller published a run with failed Docker log capture" >&2
  exit 1
}
[[ $(find "$temporary_dir" -maxdepth 1 -type d -name 'rejected-log-run.failed.*' | wc -l) -eq 1 ]] || {
  echo "controller did not preserve exactly one failed Docker log capture" >&2
  exit 1
}

mv "$fixture_tools/lib/process-rss-sampler.sh" "$fixture_tools/lib/process-rss-sampler.real.sh"
ln -s process-rss-sampler.real.sh "$fixture_tools/lib/process-rss-sampler.sh"
if PATH="$test_path" bash "$fixture_tools/run-qualifying-benchmark.sh" \
  --output-dir "$temporary_dir/rejected-run" \
  --engine native \
  --profile B1 \
  --round 1 \
  --container fixture-native \
  --image "$FIXTURE_IMAGE_REFERENCE" \
  --client "$client" \
  --workload "$workload" \
  --endpoint http://127.0.0.1:8080/v1/voice-design/speech \
  --requests 200 \
  --warmups 24 \
  --idle-baseline-seconds 15 \
  --evidence-prefix runs/round-01/native/B1 \
  >"$temporary_dir/rejected.stdout" 2>"$temporary_dir/rejected.stderr"; then
  echo "controller accepted a symlinked process-RSS sampler library" >&2
  exit 1
fi
grep -Fq \
  'process-RSS sampler library must be a regular file, not a symlink' \
  "$temporary_dir/rejected.stderr"
[[ ! -e "$temporary_dir/rejected-run" ]] || {
  echo "controller staged output before rejecting a symlinked sampler" >&2
  exit 1
}

echo "qualifying controller provenance packaging fixtures passed"
