#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

[[ $# -eq 2 ]] || release_die "usage: $0 RELEASE_RECORD EVIDENCE_DIR"
readonly RELEASE_RECORD=$1
readonly EVIDENCE_DIR=$2
release_read_record "$RELEASE_RECORD"

for command_name in docker git jq; do
  release_require_command "$command_name"
done
readonly TOOLS_DIR="$(release_tools_dir)"
readonly GRYPE_BIN="${GRYPE_BIN:-$TOOLS_DIR/bin/grype}"
readonly GITLEAKS_BIN="${GITLEAKS_BIN:-$TOOLS_DIR/bin/gitleaks}"
release_require_exact_tool_version "$GRYPE_BIN" "$GRYPE_VERSION"
release_require_exact_tool_version "$GITLEAKS_BIN" "$GITLEAKS_VERSION"

readonly DIGEST="$(release_record_value "$RELEASE_RECORD" '.digest')"
readonly SOURCE_REVISION="$(release_record_value "$RELEASE_RECORD" '.source_revision')"
readonly CANDIDATE_TAG="$(release_record_value "$RELEASE_RECORD" '.candidate_tag')"
readonly GIT_TAG="$(release_record_value "$RELEASE_RECORD" '.git_tag')"
readonly REFERENCE="$RELEASE_IMAGE@$DIGEST"

cd "$RELEASE_IMAGE_ROOT"
[[ "$(git rev-parse HEAD)" == "$SOURCE_REVISION" ]] || release_die "checkout does not match release record"
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || release_die "release checkout is not clean"
release_require_new_directory "$EVIDENCE_DIR"

for tag in "$CANDIDATE_TAG" "$GIT_TAG"; do
  remote_digest="$(docker buildx imagetools inspect "$RELEASE_IMAGE:$tag" \
    --format '{{json .}}' | jq -er '.manifest.digest')"
  [[ "$remote_digest" == "$DIGEST" ]] || release_die "tag drift detected: $tag"
done

docker buildx imagetools inspect "$REFERENCE" --raw >"$EVIDENCE_DIR/oci-index.json"
jq -e '
  [.manifests[] | select(.annotations["vnd.docker.reference.type"] != "attestation-manifest")]
    as $runtime
  | ($runtime | length) == 1
    and $runtime[0].platform.os == "linux"
    and $runtime[0].platform.architecture == "arm64"
  and ([.manifests[] | select(
    .annotations["vnd.docker.reference.type"] == "attestation-manifest"
  )] | length) >= 1
' "$EVIDENCE_DIR/oci-index.json" >/dev/null || release_die "invalid runtime/attestation manifest set"

readonly PLATFORM_DIGEST="$(jq -er '
  [.manifests[] | select(.annotations["vnd.docker.reference.type"] != "attestation-manifest")]
  | .[0].digest | select(test("^sha256:[0-9a-f]{64}$"))
' "$EVIDENCE_DIR/oci-index.json")"
docker buildx imagetools inspect "$RELEASE_IMAGE@$PLATFORM_DIGEST" --raw \
  >"$EVIDENCE_DIR/platform-manifest.json"
readonly COMPRESSED_BYTES="$(jq -er '.config.size + ([.layers[].size] | add)' \
  "$EVIDENCE_DIR/platform-manifest.json")"
(( COMPRESSED_BYTES <= 6000000000 )) || release_die "compressed image exceeds 6.0 GB"

docker buildx imagetools inspect "$REFERENCE" --format '{{json .SBOM}}' \
  >"$EVIDENCE_DIR/buildkit-sbom.json"
docker buildx imagetools inspect "$REFERENCE" --format '{{json .Provenance}}' \
  >"$EVIDENCE_DIR/buildkit-provenance.json"
jq -e '.SPDX.spdxVersion | startswith("SPDX-")' \
  "$EVIDENCE_DIR/buildkit-sbom.json" >/dev/null || release_die "BuildKit SPDX SBOM is absent or invalid"
jq -e -f "$SCRIPT_DIR/max-provenance.jq" \
  "$EVIDENCE_DIR/buildkit-provenance.json" >/dev/null || \
  release_die "maximum BuildKit provenance is absent or incomplete"
jq -f "$SCRIPT_DIR/normalize-provenance.jq" \
  "$EVIDENCE_DIR/buildkit-provenance.json" \
  >"$EVIDENCE_DIR/provenance-parameters.json"
if jq -e '
  [paths(scalars) as $p
   | {key: ($p | map(tostring) | join(".")), value: getpath($p)}
   | select(
       (.key | test("password|passwd|token|secret|credential|api.?key"; "i"))
       or ((.value | type) == "string" and
           (.value | test("(^|[^A-Za-z])(ghp_|github_pat_|glpat-|AKIA|sk-[A-Za-z0-9]|/home/|/Users/|/root/)")))
     )] | length > 0
' "$EVIDENCE_DIR/provenance-parameters.json" >/dev/null; then
  release_die "provenance contains a credential-like field, secret, or private absolute path"
fi

"$GITLEAKS_BIN" git "$RELEASE_IMAGE_ROOT" \
  --no-banner --no-color --redact=100 --report-format json \
  --report-path "$EVIDENCE_DIR/gitleaks.json"
jq -e 'type == "array" and length == 0' "$EVIDENCE_DIR/gitleaks.json" >/dev/null ||
  release_die "Gitleaks report is not an empty JSON array"

"$GRYPE_BIN" db update >"$EVIDENCE_DIR/grype-db-update.txt"
"$GRYPE_BIN" db status >"$EVIDENCE_DIR/grype-db-status.txt"
"$GRYPE_BIN" "$REFERENCE" --output json >"$EVIDENCE_DIR/grype.json"
jq -e '.matches | type == "array"' "$EVIDENCE_DIR/grype.json" >/dev/null ||
  release_die "Grype did not produce a valid result"
readonly BLOCKERS="$(jq '[.matches[] | select(
  .vulnerability.severity == "High" or .vulnerability.severity == "Critical"
)] | length' "$EVIDENCE_DIR/grype.json")"
(( BLOCKERS == 0 )) || release_die "Grype found $BLOCKERS high/critical release blockers"

jq -n \
  --arg digest "$DIGEST" \
  --arg platform_digest "$PLATFORM_DIGEST" \
  --argjson compressed_bytes "$COMPRESSED_BYTES" \
  --arg grype_version "$GRYPE_VERSION" \
  --arg gitleaks_version "$GITLEAKS_VERSION" '{
    schema: "qwen3-tts-native/supply-chain-verification/v1",
    digest: $digest,
    platform_digest: $platform_digest,
    compressed_bytes: $compressed_bytes,
    buildkit_sbom: "verified",
    buildkit_provenance: "verified",
    source_secrets: "none",
    high_critical_vulnerabilities: 0,
    tools: {grype: $grype_version, gitleaks: $gitleaks_version}
  }' >"$EVIDENCE_DIR/supply-chain-verified.json"

printf 'Supply-chain verification passed for %s\n' "$REFERENCE"
