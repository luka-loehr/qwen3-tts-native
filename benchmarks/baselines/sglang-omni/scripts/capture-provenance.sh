#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 64
fi

output_dir="$(mkdir -p -- "$1" && cd -- "$1" && pwd)"
image="${IMAGE:-qwen3-tts-sglang-omni:0.1.0-stock-spark}"
base="lmsysorg/sglang:v0.5.12.post1-cu130-runtime@sha256:8df56b542526f4fffd5372f7f65a583c7852e50442c1f43c9c3feddfd93944a4"

docker image inspect "${image}" > "${output_dir}/image-inspect.json"
docker buildx imagetools inspect "${base}" > "${output_dir}/base-image-manifest.txt"
docker run --rm --entrypoint python3 "${image}" -m pip freeze --all \
  > "${output_dir}/pip-freeze.txt"
docker run --rm --entrypoint dpkg-query "${image}" \
  -W '-f=${Package}\t${Version}\n' sox libsox-fmt-all \
  > "${output_dir}/system-packages.txt"
docker run --rm --entrypoint git "${image}" \
  -C /opt/sglang-omni status --short --branch \
  > "${output_dir}/upstream-git-status.txt"
docker run --rm --entrypoint git "${image}" \
  -C /opt/sglang-omni diff -- pyproject.toml \
  > "${output_dir}/applied-packaging-patch.diff"
nvidia-smi -q > "${output_dir}/nvidia-smi-q.txt"

(
  cd -- "${output_dir}"
  find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum
) > "${output_dir}/SHA256SUMS"

echo "Captured comparator provenance in ${output_dir}"
