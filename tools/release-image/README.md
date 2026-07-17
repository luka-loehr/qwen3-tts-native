# Release image tooling

These fail-closed scripts implement the production GHCR path. Docker builds,
image scans, clean pulls, and GPU acceptance run only on the DGX Spark. The Mac
may run `test-release-tools.sh`, Git operations, and GitHub control-plane
commands; it must not build or run the model image.

## Fixed order

From the final `main` commit on the Spark, while the repository and package are
still private:

```bash
git fetch origin main
git switch main
git pull --ff-only origin main
test -z "$(git status --porcelain=v1 --untracked-files=all)"

RELEASE_ROOT=/absolute/new/path/qwen3-tts-v0.1.0
MODEL_CONTEXT=/absolute/path/to/qwen3-tts-1.7b-voice-design-bf16-indexed
mkdir -p "$RELEASE_ROOT"

./tools/release-image/bootstrap-tools.sh
./tools/release-metadata/bootstrap-tools.sh
SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)" \
  ./tools/release-metadata/generate.sh "$RELEASE_ROOT/metadata"

# From a trusted control host with `gh`, authenticate this Spark Docker client
# without putting the token in a command argument:
gh auth token | ssh spark-host \
  'docker login ghcr.io --username luka-loehr --password-stdin'
RELEASE_VERSION=v0.1.0 ./tools/release-image/build-and-push.sh \
  "$MODEL_CONTEXT" \
  "$RELEASE_ROOT/metadata" \
  "$RELEASE_ROOT/build"

./tools/release-image/verify-supply-chain.sh \
  "$RELEASE_ROOT/build/release-record.json" \
  "$RELEASE_ROOT/supply-chain"
docker logout ghcr.io
```

The build produces only the immutable candidate and Git/model tags. It records
the real OCI index digest in `release-record.json`. Supply-chain verification
rejects tag drift, a non-ARM64 runtime, absent BuildKit SPDX/SLSA evidence,
credential-like provenance, private absolute provenance paths, Git-history
secrets, high/critical vulnerabilities, or a compressed image above 6.0 GB.

Make the repository and package public, then sign the accepted digest from the
Mac (replace the digest with the value in the record):

```bash
gh repo edit luka-loehr/qwen3-tts-native \
  --visibility public --accept-visibility-change-consequences
gh api --method PATCH /user/packages/container/qwen3-tts-native \
  -f visibility=public
REPO=luka-loehr/qwen3-tts-native
SOURCE_REVISION=$(git rev-parse origin/main)
DIGEST='sha256:<PUBLISHED_DIGEST>'
gh workflow run sign-ghcr-image.yml --repo "$REPO" --ref main \
  --field version=v0.1.0 \
  --field digest="$DIGEST"

RUN_ID=
for attempt in $(seq 1 30); do
  RUN_ID=$(gh run list --repo "$REPO" --workflow sign-ghcr-image.yml \
    --branch main --event workflow_dispatch --limit 20 \
    --json databaseId,displayTitle,headSha | jq -r \
    --arg title "Sign v0.1.0 at $DIGEST" --arg sha "$SOURCE_REVISION" \
    '[.[] | select(.displayTitle == $title and .headSha == $sha)]
     | first | .databaseId // empty')
  test -n "$RUN_ID" && break
  sleep 2
done
test -n "$RUN_ID"
gh run watch "$RUN_ID" --repo "$REPO" --exit-status
gh run view "$RUN_ID" --repo "$REPO" \
  --json conclusion,event,headBranch,headSha | jq -e --arg sha "$SOURCE_REVISION" '
  .conclusion == "success" and .event == "workflow_dispatch"
  and .headBranch == "main" and .headSha == $sha
'
```

Run `clean-pull-gpu-acceptance.sh` on a separate Spark Docker daemon selected
by `DOCKER_HOST`. Its daemon ID must differ from the build daemon and its image
store must be empty. The script supplies an empty temporary Docker config, so
the pull is genuinely anonymous. It then validates the immutable image under
the hardened runtime and performs a real VoiceDesign multipart PCM request:

```bash
export DOCKER_HOST=unix:///run/qwen3-tts-clean/docker.sock
./tools/release-image/clean-pull-gpu-acceptance.sh \
  "$RELEASE_ROOT/build/release-record.json" \
  "$RELEASE_ROOT/build/builder-docker-info.json" \
  "$RELEASE_ROOT/gpu-acceptance"
```

Finally, promote only after both receipts and the keyless signature verify:

```bash
unset DOCKER_HOST
test "$(docker info --format '{{.ID}}')" = \
  "$(jq -r .ID "$RELEASE_ROOT/build/builder-docker-info.json")"
# Repeat the control-host stdin login used for the private candidate build.
gh auth token | ssh spark-host \
  'docker login ghcr.io --username luka-loehr --password-stdin'
export RELEASE_IMAGE_TOOLS_DIR="$HOME/.cache/qwen3-tts-release-image"
COSIGN_BIN="$RELEASE_IMAGE_TOOLS_DIR/bin/cosign" \
  ./tools/release-image/promote.sh \
  "$RELEASE_ROOT/build/release-record.json" \
  "$RELEASE_ROOT/supply-chain/supply-chain-verified.json" \
  "$RELEASE_ROOT/gpu-acceptance/gpu-acceptance.json" \
  "$RELEASE_ROOT/promotion"
docker logout ghcr.io
```

`promote.sh` moves `v0.1.0` first and `latest` last, verifies both resolve to
the original digest, and never rebuilds the image.
