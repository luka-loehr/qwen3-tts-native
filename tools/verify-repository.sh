#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REDOCLY_VERSION="2.38.0"
readonly MANIFESTS=(
  "native/qwen3-tts-native/Cargo.toml"
  "native/qwen3-tts-native-codec/Cargo.toml"
  "native/qwen3-tts-runtime/Cargo.toml"
  "native/qwen3-tts-server/Cargo.toml"
  "native/qwen3-tts-bench/Cargo.toml"
  "native/qwen3-tts-http-bench/Cargo.toml"
)

cd "$ROOT"

for command_name in cargo git npx; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'required verification command is unavailable: %s\n' "$command_name" >&2
    exit 1
  fi
done

for manifest in "${MANIFESTS[@]}"; do
  printf '==> cargo metadata: %s\n' "$manifest"
  cargo metadata \
    --locked \
    --no-deps \
    --format-version 1 \
    --manifest-path "$manifest" >/dev/null

  printf '==> cargo fmt: %s\n' "$manifest"
  cargo fmt --all --manifest-path "$manifest" -- --check

  printf '==> cargo test: %s\n' "$manifest"
  cargo test --all-targets --all-features --locked --manifest-path "$manifest"

  printf '==> cargo clippy: %s\n' "$manifest"
  cargo clippy \
    --all-targets \
    --all-features \
    --locked \
    --manifest-path "$manifest" \
    -- \
    -D warnings
done

printf '==> shell syntax\n'
while IFS= read -r -d '' script; do
  bash -n "$script"
done < <(git ls-files -z -- '*.sh')

printf '==> release-image policy tests\n'
./tools/release-image/test-release-tools.sh

printf '==> OpenAPI 3.1 lint with Redocly CLI %s\n' "$REDOCLY_VERSION"
REDOCLY_TELEMETRY=off \
REDOCLY_SUPPRESS_UPDATE_NOTICE=true \
  npx --yes "@redocly/cli@${REDOCLY_VERSION}" \
  lint --extends=spec docs/openapi.yaml

printf '==> Git whitespace and conflict markers\n'
git diff --check

printf 'Repository verification passed. GPU and release-image qualification remain separate gates.\n'
