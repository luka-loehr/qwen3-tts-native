#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

readonly WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/qwen3-tts-release-tests.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

jq -n \
  --arg image "$RELEASE_IMAGE" \
  --arg digest "sha256:$(printf 'a%.0s' {1..64})" \
  --arg model "$RELEASE_MODEL_ID" \
  --arg revision "$RELEASE_MODEL_REVISION" '{
    schema: "qwen3-tts-native/release-record/v1",
    image: $image,
    digest: $digest,
    release_version: "v0.1.0",
    candidate_tag: "v0.1.0-vd1.7b-cu13.0.3-sm121",
    git_tag: "git-0123456789ab-model-5ecdb67",
    source_revision: "0123456789abcdef0123456789abcdef01234567",
    platform: "linux/arm64",
    model: {id: $model, revision: $revision}
  }' >"$WORK_DIR/valid.json"
release_read_record "$WORK_DIR/valid.json"

jq '.image = "ghcr.io/example/other"' "$WORK_DIR/valid.json" >"$WORK_DIR/wrong-image.json"
if (release_read_record "$WORK_DIR/wrong-image.json") >/dev/null 2>&1; then
  release_die "wrong-image release record was accepted"
fi
jq '.digest = "sha256:1234"' "$WORK_DIR/valid.json" >"$WORK_DIR/wrong-digest.json"
if (release_read_record "$WORK_DIR/wrong-digest.json") >/dev/null 2>&1; then
  release_die "malformed digest was accepted"
fi

for script in "$SCRIPT_DIR"/*.sh; do
  bash -n "$script"
done
if command -v shellcheck >/dev/null 2>&1; then
  shellcheck -x --severity=error "$SCRIPT_DIR"/*.sh
fi
[[ -x "$SCRIPT_DIR/clean-pull-gpu-acceptance.sh" ]] ||
  release_die "clean-pull GPU acceptance script is not executable"
[[ -x "$SCRIPT_DIR/verify-final-gpu-acceptance.sh" ]] ||
  release_die "final GPU acceptance verifier is not executable"

grep -Fq -- '--provenance=mode=max' "$SCRIPT_DIR/build-and-push.sh"
grep -Fq -- 'docker/buildkit-syft-scanner@sha256:' "$SCRIPT_DIR/build-and-push.sh"
grep -Fq -- 'high/critical release blockers' "$SCRIPT_DIR/verify-supply-chain.sh"
grep -Fq -- 'db update' "$SCRIPT_DIR/verify-supply-chain.sh"
grep -Fq -- 'provenance contains a credential-like field' "$SCRIPT_DIR/verify-supply-chain.sh"
grep -Fq -- 'image store is not empty' "$SCRIPT_DIR/clean-pull-gpu-acceptance.sh"
grep -Fq -- '--gpus' "$SCRIPT_DIR/clean-pull-gpu-acceptance.sh"
grep -Fq -- '--cap-drop=ALL' "$SCRIPT_DIR/clean-pull-gpu-acceptance.sh"
grep -Fq -- 'qwen3-tts-native/clean-pull-gpu-acceptance/v2' \
  "$SCRIPT_DIR/clean-pull-gpu-acceptance.sh"
grep -Fq -- 'COLD_RSS_LIMIT_BYTES=4509715660' "$SCRIPT_DIR/clean-pull-gpu-acceptance.sh"
grep -Fq -- 'readonly COLD_START_SECONDS=$SECONDS' "$SCRIPT_DIR/clean-pull-gpu-acceptance.sh"
grep -Fq -- 'samples outside the ten approved metric names' "$SCRIPT_DIR/clean-pull-gpu-acceptance.sh"
grep -Fq -- 'sox_clipping_warning: "none"' "$SCRIPT_DIR/clean-pull-gpu-acceptance.sh"
grep -Fq -- 'cancellation_requested' "$SCRIPT_DIR/clean-pull-gpu-acceptance.sh"
grep -Fq -- 'prompt_free_metrics: "passed"' "$SCRIPT_DIR/clean-pull-gpu-acceptance.sh"
grep -Fq -- 'readonly -a LANGUAGES=(auto chinese english japanese korean german french russian portuguese spanish italian)' \
  "$SCRIPT_DIR/clean-pull-gpu-acceptance.sh"
grep -Fq -- 'graceful_sigterm' "$SCRIPT_DIR/clean-pull-gpu-acceptance.sh"
grep -Fq -- 'sha256sum --check --strict SHA256SUMS' "$SCRIPT_DIR/clean-pull-gpu-acceptance.sh"
grep -Fq -- 'qwen3-tts-native/final-gpu-acceptance/v1' \
  "$SCRIPT_DIR/verify-final-gpu-acceptance.sh"
grep -Fq -- 'B6_GPU_UNIFIED_MEMORY_LIMIT_BYTES=6000000000' \
  "$SCRIPT_DIR/verify-final-gpu-acceptance.sh"
grep -Fq -- 'B1 does not satisfy aggregate RTF < 1 and TTFA p95 < 200 ms' \
  "$SCRIPT_DIR/verify-final-gpu-acceptance.sh"
grep -Fq -- 'validate_complete_inventory "$run_dir" "$label"' \
  "$SCRIPT_DIR/verify-final-gpu-acceptance.sh"
grep -Fq -- 'validate_run "$B1_RUN_DIR" B1 200 true' \
  "$SCRIPT_DIR/verify-final-gpu-acceptance.sh"
grep -Fq -- 'validate_run "$B6_RUN_DIR" B6 240 false' \
  "$SCRIPT_DIR/verify-final-gpu-acceptance.sh"
grep -Fq -- 'substituted_for_observed_total: false' \
  "$SCRIPT_DIR/verify-final-gpu-acceptance.sh"
grep -Fq -- 'samples outside the ten approved names' \
  "$SCRIPT_DIR/verify-final-gpu-acceptance.sh"
grep -Fq -- 'FINAL_GPU_RECEIPT' "$SCRIPT_DIR/promote.sh"
grep -Fq -- 'final_gpu_acceptance: "verified"' "$SCRIPT_DIR/promote.sh"
grep -Fq -- 'cosign-verification.json' "$SCRIPT_DIR/promote.sh"
printf 'Release-image tool tests passed.\n'
