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

The build produces only an immutable version-and-model candidate tag and a
source-commit-and-model-revision tag. It records
the real OCI index digest in `release-record.json`. Supply-chain verification
rejects tag drift, a non-ARM64 runtime, absent BuildKit SPDX/SLSA evidence,
credential-like provenance, private absolute provenance paths, Git-history
secrets, high/critical vulnerabilities, or a compressed image above 6.0 GB.

After the private supply-chain scan passes, sign the accepted digest while both
the repository and package are still private. Keep the repository private until
GPU acceptance, promoted tags, Git tag, and GitHub release are complete.
Replace the digest below with the value in the release record:

```bash
test "$(gh repo view luka-loehr/qwen3-tts-native \
  --json visibility --jq .visibility)" = PRIVATE
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

# The accepted digest is now signed. Expose only the package so the following
# clean pull can be genuinely anonymous; keep the repository private.
gh api --method PATCH /user/packages/container/qwen3-tts-native \
  -f visibility=public
test "$(gh api /user/packages/container/qwen3-tts-native \
  --jq .visibility)" = public
```

Run `clean-pull-gpu-acceptance.sh` on a separate Spark Docker daemon selected
by `DOCKER_HOST`. Its daemon ID must differ from the build daemon and its image
store must be empty. The script supplies an empty temporary Docker config, so
the pull is genuinely anonymous. It then validates the immutable image under
the hardened runtime. Its version-2 receipt retains 100 ms cold-start and
post-ready process-RSS samples, enforces readiness within 20 seconds and the
4.2 GiB cold process-RSS ceiling, validates a buffered WAV with SoX, exercises
cancellation and prompt-free metrics, requires progressive natural EOS for
`auto` and all ten explicit languages, and proves restart/readiness plus
graceful SIGTERM shutdown:

```bash
export DOCKER_HOST=unix:///run/qwen3-tts-clean/docker.sock
./tools/release-image/clean-pull-gpu-acceptance.sh \
  "$RELEASE_ROOT/build/release-record.json" \
  "$RELEASE_ROOT/build/builder-docker-info.json" \
  "$RELEASE_ROOT/gpu-acceptance"
```

Post-ready steady RSS is recorded as `measured_for_review`; the receipt does
not turn the historical 768 MiB review reference into a false automated pass.
Next bind the same digest to complete qualifying Native B1 and B6 runs:

```bash
./tools/release-image/verify-final-gpu-acceptance.sh \
  "$RELEASE_ROOT/build/release-record.json" \
  "$RELEASE_ROOT/gpu-acceptance/gpu-acceptance.json" \
  /absolute/path/to/digest-specific/native/B1 \
  /absolute/path/to/digest-specific/native/B6 \
  "$RELEASE_ROOT/final-gpu-acceptance"
```

Both run directories require a complete valid `SHA256SUMS` inventory and the
exact `ghcr.io/luka-loehr/qwen3-tts-native@sha256:...` image identity with
matching source/version labels. B1 is exactly 200 successful natural-EOS
requests after 24 warmups, with aggregate RTF below 1 and TTFA p95 below
200 ms. B6 is at least 240 successful natural-EOS requests after 24 warmups.
Both require zero failures, zero competing CUDA processes, a configured 100 ms
telemetry interval, and no qualifying observed telemetry gap above 200 ms.
The observed B6 GPU unified-memory peak must not exceed 6,000,000,000 bytes.
Internal `peak_request_device_bytes` and `peak_request_host_bytes` are reported
separately and never substituted for observed total memory.

Finally, promote only after the supply-chain, enhanced clean-pull, final B1/B6,
and keyless-signature receipts verify:

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
  "$RELEASE_ROOT/final-gpu-acceptance/final-gpu-acceptance.json" \
  "$RELEASE_ROOT/promotion"
docker logout ghcr.io
```

`promote.sh` moves `v0.1.0` first and `latest` last, verifies both resolve to
the original digest, and never rebuilds the image. Create the annotated Git
tag and complete GitHub release while the repository is still private. Make
the repository public only after those records exist, then verify an anonymous
clone, release-asset access, and digest pull from clean contexts.
