#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

usage() {
  printf 'Usage: RELEASE_VERSION=vX.Y.Z %s MODEL_CONTEXT RELEASE_CONTEXT EVIDENCE_DIR\n' "$0" >&2
  exit 2
}

[[ $# -eq 3 ]] || usage
readonly MODEL_CONTEXT=$1
readonly RELEASE_CONTEXT=$2
readonly EVIDENCE_DIR=$3
readonly RELEASE_VERSION=${RELEASE_VERSION:-}

release_validate_version "$RELEASE_VERSION"
[[ "$RELEASE_VERSION" =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] ||
  release_die "the final image version must not contain prerelease or build metadata"
[[ "$MODEL_CONTEXT" = /* ]] || release_die "model context must be an absolute path"
[[ "$RELEASE_CONTEXT" = /* ]] || release_die "release context must be an absolute path"
release_require_directory "$MODEL_CONTEXT"
release_require_directory "$RELEASE_CONTEXT"

for command_name in docker git jq tee uname; do
  release_require_command "$command_name"
done
[[ "$(uname -s)" == "Linux" ]] || release_die "the release image must be built on Linux"
[[ "$(uname -m)" == "aarch64" ]] || release_die "the release image must be built natively on ARM64"

cd "$RELEASE_IMAGE_ROOT"
[[ "$(git branch --show-current)" == "main" ]] || release_die "release builds require branch main"
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || release_die "release worktree is not clean"
readonly SOURCE_REVISION="$(git rev-parse HEAD)"
release_validate_commit "$SOURCE_REVISION"
readonly ORIGIN_MAIN="$(git rev-parse --verify refs/remotes/origin/main)"
[[ "$SOURCE_REVISION" == "$ORIGIN_MAIN" ]] ||
  release_die "HEAD must equal the explicitly fetched origin/main"

readonly BUILD_DATE="$(git show -s --format=%cI "$SOURCE_REVISION")"
readonly SOURCE_DATE_EPOCH="$(git show -s --format=%ct "$SOURCE_REVISION")"
readonly CANDIDATE_TAG="${RELEASE_VERSION}-vd1.7b-cu13.0.3-sm121"
readonly GIT_TAG="git-${SOURCE_REVISION:0:12}-model-5ecdb67"
readonly CANDIDATE_REFERENCE="$RELEASE_IMAGE:$CANDIDATE_TAG"
readonly GIT_REFERENCE="$RELEASE_IMAGE:$GIT_TAG"

[[ "$(release_sha256_file "$MODEL_CONTEXT/manifest.json")" == "$RELEASE_MODEL_MANIFEST_SHA256" ]] ||
  release_die "model manifest hash mismatch"
[[ "$(release_sha256_file "$MODEL_CONTEXT/model.safetensors")" == "$RELEASE_MODEL_VOICE_SHA256" ]] ||
  release_die "VoiceDesign weight hash mismatch"
[[ "$(release_sha256_file "$MODEL_CONTEXT/speech_tokenizer/model.safetensors")" == "$RELEASE_MODEL_DECODER_SHA256" ]] ||
  release_die "decoder weight hash mismatch"
"$RELEASE_IMAGE_ROOT/tools/release-metadata/validate.sh" "$RELEASE_CONTEXT"

release_require_new_directory "$EVIDENCE_DIR"
docker info --format '{{json .}}' >"$EVIDENCE_DIR/builder-docker-info.json"
jq -e '.Architecture == "aarch64" and (.DockerRootDir | type == "string" and length > 0)' \
  "$EVIDENCE_DIR/builder-docker-info.json" >/dev/null ||
  release_die "Docker daemon is not an ARM64 daemon with an identifiable content store"
docker buildx inspect --bootstrap >"$EVIDENCE_DIR/buildx-inspect.txt"

readonly BUILD_METADATA="$EVIDENCE_DIR/build-metadata.json"
docker buildx build \
  --pull \
  --platform linux/arm64 \
  --file containers/Dockerfile.runtime \
  --build-context "model=$MODEL_CONTEXT" \
  --build-context "release-metadata=$RELEASE_CONTEXT" \
  --build-arg "VERSION=$RELEASE_VERSION" \
  --build-arg "VCS_REF=$SOURCE_REVISION" \
  --build-arg "BUILD_DATE=$BUILD_DATE" \
  --build-arg "SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH" \
  --tag "$CANDIDATE_REFERENCE" \
  --tag "$GIT_REFERENCE" \
  --provenance=mode=max \
  --attest \
    'type=sbom,generator=docker.io/docker/buildkit-syft-scanner@sha256:79e7b013cbec16bbb436f312819a49a4a57752b2270c1a9332ae1a10fcc82a68' \
  --metadata-file "$BUILD_METADATA" \
  --push \
  . 2>&1 | tee "$EVIDENCE_DIR/build.log"

readonly DIGEST="$(jq -er '
  ."containerimage.digest"
  | select(test("^sha256:[0-9a-f]{64}$"))
' "$BUILD_METADATA")"
release_validate_digest "$DIGEST"

for reference in "$CANDIDATE_REFERENCE" "$GIT_REFERENCE"; do
  remote_digest="$(docker buildx imagetools inspect "$reference" \
    --format '{{json .}}' | jq -er '.manifest.digest')"
  [[ "$remote_digest" == "$DIGEST" ]] ||
    release_die "remote tag does not resolve to the pushed digest: $reference"
done

jq -n \
  --arg image "$RELEASE_IMAGE" \
  --arg digest "$DIGEST" \
  --arg release_version "$RELEASE_VERSION" \
  --arg candidate_tag "$CANDIDATE_TAG" \
  --arg git_tag "$GIT_TAG" \
  --arg source_revision "$SOURCE_REVISION" \
  --arg build_date "$BUILD_DATE" \
  --arg model_id "$RELEASE_MODEL_ID" \
  --arg model_revision "$RELEASE_MODEL_REVISION" \
  --arg manifest_sha256 "$RELEASE_MODEL_MANIFEST_SHA256" \
  --arg voice_sha256 "$RELEASE_MODEL_VOICE_SHA256" \
  --arg decoder_sha256 "$RELEASE_MODEL_DECODER_SHA256" '
  {
    schema: "qwen3-tts-native/release-record/v1",
    image: $image,
    digest: $digest,
    release_version: $release_version,
    candidate_tag: $candidate_tag,
    git_tag: $git_tag,
    source_revision: $source_revision,
    build_date: $build_date,
    platform: "linux/arm64",
    model: {
      id: $model_id,
      revision: $model_revision,
      manifest_sha256: $manifest_sha256,
      voice_sha256: $voice_sha256,
      decoder_sha256: $decoder_sha256,
      voice_clone: false
    }
  }
' >"$EVIDENCE_DIR/release-record.json"

release_read_record "$EVIDENCE_DIR/release-record.json"
printf 'Pushed immutable candidate %s@%s\n' "$RELEASE_IMAGE" "$DIGEST"
printf 'Release record: %s\n' "$EVIDENCE_DIR/release-record.json"
