#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
context_dir="$(cd -- "${script_dir}/.." && pwd)"
image="${IMAGE:-qwen3-tts-sglang-omni:0.1.0-stock-spark}"

docker build \
  --pull \
  --platform linux/arm64 \
  --tag "${image}" \
  "${context_dir}"

docker image inspect \
  --format 'image={{.Id}} created={{.Created}} size={{.Size}}' \
  "${image}"
