#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -z "${MODEL_DIR:-}" ]]; then
  echo "MODEL_DIR must point to Qwen3-TTS-12Hz-1.7B-VoiceDesign@5ecdb673..." >&2
  exit 64
fi

model_dir="$(cd -- "${MODEL_DIR}" && pwd)"
expected_revision="5ecdb67327fd37bb2e042aab12ff7391903235d3"
revision_file="${model_dir}/.baseline-model-revision"
snapshot_name="$(basename -- "${model_dir}")"

if [[ -f "${revision_file}" ]]; then
  actual_revision="$(tr -d '[:space:]' < "${revision_file}")"
  if [[ "${actual_revision}" != "${expected_revision}" ]]; then
    echo "model revision mismatch: ${actual_revision}" >&2
    exit 65
  fi
elif [[ "${snapshot_name}" != "${expected_revision}" && "${ALLOW_UNVERIFIED_MODEL:-0}" != "1" ]]; then
  echo "model revision is not verifiable; use download-model.sh or set ALLOW_UNVERIFIED_MODEL=1" >&2
  exit 65
fi

if [[ -f "${model_dir}/.baseline-files.sha256" ]]; then
  (cd -- "${model_dir}" && sha256sum --quiet --check .baseline-files.sha256)
fi

image="${IMAGE:-qwen3-tts-sglang-omni:0.1.0-stock-spark}"
container_name="${CONTAINER_NAME:-qwen3-tts-sglang-stock}"
port="${PORT:-8000}"
cache_dir="${CACHE_DIR:-${TMPDIR:-/tmp}/qwen3-tts-sglang-cache}"
mkdir -p -- "${cache_dir}"

model_volume="${model_dir}:/models/voice-design:ro"
container_model_path="/models/voice-design"
if [[ "${snapshot_name}" == "${expected_revision}" \
      && "$(basename -- "$(dirname -- "${model_dir}")")" == "snapshots" ]]; then
  repository_root="$(cd -- "${model_dir}/../.." && pwd)"
  model_volume="${repository_root}:/models/hf-repository:ro"
  container_model_path="/models/hf-repository/snapshots/${expected_revision}"
fi

exec docker run --rm \
  --name "${container_name}" \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1:-1 \
  --stop-timeout 30 \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env HF_HOME=/cache/huggingface \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --volume "${model_volume}" \
  --volume "${cache_dir}:/cache" \
  "${image}" \
  sgl-omni serve \
    --model-path "${container_model_path}" \
    --config /opt/baseline/qwen3_tts_1_7b_voicedesign.yaml \
    --host 127.0.0.1 \
    --port "${port}"
