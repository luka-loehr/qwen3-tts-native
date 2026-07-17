#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 MODEL_DIR" >&2
  exit 64
fi

model_dir="$(mkdir -p -- "$1" && cd -- "$1" && pwd)"
image="${IMAGE:-qwen3-tts-sglang-omni:0.1.0-stock-spark}"
model_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
revision="5ecdb67327fd37bb2e042aab12ff7391903235d3"

docker_args=(
  run --rm
  --entrypoint hf
  --user "$(id -u):$(id -g)"
  --volume "${model_dir}:/model"
)
if [[ -n "${HF_TOKEN:-}" ]]; then
  docker_args+=(--env HF_TOKEN)
fi

docker "${docker_args[@]}" "${image}" \
  download "${model_id}" \
    --revision "${revision}" \
    --local-dir /model

printf '%s\n' "${revision}" > "${model_dir}/.baseline-model-revision"
(
  cd -- "${model_dir}"
  find . -type f \
    ! -path './.cache/*' \
    ! -name '.baseline-model-revision' \
    ! -name '.baseline-files.sha256' \
    -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum
) > "${model_dir}/.baseline-files.sha256"

echo "Downloaded ${model_id}@${revision} to ${model_dir}"
echo "Recorded file hashes in ${model_dir}/.baseline-files.sha256"
