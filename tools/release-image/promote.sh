#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

[[ $# -eq 4 ]] || release_die "usage: $0 RELEASE_RECORD SUPPLY_RECEIPT GPU_RECEIPT EVIDENCE_DIR"
readonly RELEASE_RECORD=$1
readonly SUPPLY_RECEIPT=$2
readonly GPU_RECEIPT=$3
readonly EVIDENCE_DIR=$4
release_read_record "$RELEASE_RECORD"
release_require_file "$SUPPLY_RECEIPT"
release_require_file "$GPU_RECEIPT"
release_require_command docker
release_require_command jq

readonly DIGEST="$(release_record_value "$RELEASE_RECORD" '.digest')"
readonly VERSION="$(release_record_value "$RELEASE_RECORD" '.release_version')"
readonly CANDIDATE_TAG="$(release_record_value "$RELEASE_RECORD" '.candidate_tag')"
readonly GIT_TAG="$(release_record_value "$RELEASE_RECORD" '.git_tag')"
readonly REFERENCE="$RELEASE_IMAGE@$DIGEST"
readonly COSIGN_BIN="${COSIGN_BIN:-cosign}"
release_require_exact_tool_version "$COSIGN_BIN" "$COSIGN_VERSION"

jq -e --arg digest "$DIGEST" '
  .schema == "qwen3-tts-native/supply-chain-verification/v1" and .digest == $digest
  and .buildkit_sbom == "verified" and .buildkit_provenance == "verified"
  and .source_secrets == "none" and .high_critical_vulnerabilities == 0
' "$SUPPLY_RECEIPT" >/dev/null || release_die "supply-chain receipt does not authorize this digest"
jq -e --arg digest "$DIGEST" '
  .schema == "qwen3-tts-native/clean-pull-gpu-acceptance/v1" and .digest == $digest
  and .pull == "anonymous-empty-store" and .hardened_runtime == "passed"
  and .gpu == "passed" and .streaming_pcm == "passed"
' "$GPU_RECEIPT" >/dev/null || release_die "GPU receipt does not authorize this digest"

release_require_new_directory "$EVIDENCE_DIR"
"$COSIGN_BIN" verify \
  --certificate-identity "$RELEASE_CERTIFICATE_IDENTITY" \
  --certificate-oidc-issuer "$RELEASE_CERTIFICATE_ISSUER" \
  --annotations "org.opencontainers.image.version=$VERSION" \
  --annotations 'org.opencontainers.image.source=https://github.com/luka-loehr/qwen3-tts-native' \
  --output json "$REFERENCE" >"$EVIDENCE_DIR/cosign-verification.json"
jq -e 'type == "array" and length >= 1' "$EVIDENCE_DIR/cosign-verification.json" >/dev/null ||
  release_die "Cosign returned no verified signature"

for tag in "$CANDIDATE_TAG" "$GIT_TAG"; do
  [[ "$(docker buildx imagetools inspect "$RELEASE_IMAGE:$tag" --format '{{json .}}' \
    | jq -er '.manifest.digest')" == "$DIGEST" ]] || release_die "pre-promotion tag drift: $tag"
done
docker buildx imagetools create --tag "$RELEASE_IMAGE:$VERSION" "$REFERENCE"
[[ "$(docker buildx imagetools inspect "$RELEASE_IMAGE:$VERSION" --format '{{json .}}' \
  | jq -er '.manifest.digest')" == "$DIGEST" ]] || release_die "semantic-version promotion changed digest"
docker buildx imagetools create --tag "$RELEASE_IMAGE:latest" "$REFERENCE"
[[ "$(docker buildx imagetools inspect "$RELEASE_IMAGE:latest" --format '{{json .}}' \
  | jq -er '.manifest.digest')" == "$DIGEST" ]] || release_die "latest promotion changed digest"

jq -n --arg digest "$DIGEST" --arg version "$VERSION" '{
  schema: "qwen3-tts-native/promotion/v1",
  digest: $digest,
  version_tag: $version,
  latest: "same-digest",
  cosign: "verified"
}' >"$EVIDENCE_DIR/promotion.json"
printf 'Promoted %s and latest to %s\n' "$VERSION" "$REFERENCE"
