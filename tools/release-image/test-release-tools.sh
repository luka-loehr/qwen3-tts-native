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

grep -Fq -- '--provenance=mode=max' "$SCRIPT_DIR/build-and-push.sh"
grep -Fq -- 'docker/buildkit-syft-scanner@sha256:' "$SCRIPT_DIR/build-and-push.sh"
grep -Fq -- 'high/critical release blockers' "$SCRIPT_DIR/verify-supply-chain.sh"
grep -Fq -- 'db update' "$SCRIPT_DIR/verify-supply-chain.sh"
grep -Fq -- 'provenance contains a credential-like field' "$SCRIPT_DIR/verify-supply-chain.sh"
grep -Fq -- 'image store is not empty' "$SCRIPT_DIR/clean-pull-gpu-acceptance.sh"
grep -Fq -- '--gpus' "$SCRIPT_DIR/clean-pull-gpu-acceptance.sh"
grep -Fq -- '--cap-drop=ALL' "$SCRIPT_DIR/clean-pull-gpu-acceptance.sh"
grep -Fq -- 'cosign-verification.json' "$SCRIPT_DIR/promote.sh"
printf 'Release-image tool tests passed.\n'
