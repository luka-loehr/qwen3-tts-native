# Release image tooling

These fail-closed scripts implement the production GHCR path. Docker builds,
image scans, clean pulls, and GPU acceptance run only on the DGX Spark. The Mac
may run `test-release-tools.sh`, Git operations, and GitHub control-plane
commands; it must not build or run the model image.

## Fixed order

From the final public `main` commit on the Spark:

```bash
git fetch origin main
git switch main
git pull --ff-only origin main
test -z "$(git status --porcelain=v1 --untracked-files=all)"

./tools/release-image/bootstrap-tools.sh
./tools/release-metadata/bootstrap-tools.sh
SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)" \
  ./tools/release-metadata/generate.sh /home/administrator/qwen3-tts-release/metadata

gh auth token | docker login ghcr.io --username luka-loehr --password-stdin
RELEASE_VERSION=v0.1.0 ./tools/release-image/build-and-push.sh \
  /home/administrator/codex-playground-artifacts/qwen3-tts-1.7b-voice-design-bf16-indexed \
  /home/administrator/qwen3-tts-release/metadata \
  /home/administrator/qwen3-tts-release/build

./tools/release-image/verify-supply-chain.sh \
  /home/administrator/qwen3-tts-release/build/release-record.json \
  /home/administrator/qwen3-tts-release/supply-chain
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
gh workflow run sign-ghcr-image.yml --ref main \
  --field version=v0.1.0 \
  --field digest=sha256:<PUBLISHED_DIGEST>
gh run watch --exit-status
```

Run `clean-pull-gpu-acceptance.sh` on a separate Spark Docker daemon selected
by `DOCKER_HOST`. Its daemon ID must differ from the build daemon and its image
store must be empty. The script supplies an empty temporary Docker config, so
the pull is genuinely anonymous. It then validates the immutable image under
the hardened runtime and performs a real VoiceDesign multipart PCM request:

```bash
export DOCKER_HOST=unix:///run/qwen3-tts-clean/docker.sock
./tools/release-image/clean-pull-gpu-acceptance.sh \
  /home/administrator/qwen3-tts-release/build/release-record.json \
  /home/administrator/qwen3-tts-release/build/builder-docker-info.json \
  /home/administrator/qwen3-tts-release/gpu-acceptance
```

Finally, promote only after both receipts and the keyless signature verify:

```bash
export RELEASE_IMAGE_TOOLS_DIR="$HOME/.cache/qwen3-tts-release-image"
COSIGN_BIN="$RELEASE_IMAGE_TOOLS_DIR/bin/cosign" \
  ./tools/release-image/promote.sh \
  /home/administrator/qwen3-tts-release/build/release-record.json \
  /home/administrator/qwen3-tts-release/supply-chain/supply-chain-verified.json \
  /home/administrator/qwen3-tts-release/gpu-acceptance/gpu-acceptance.json \
  /home/administrator/qwen3-tts-release/promotion
```

`promote.sh` moves `v0.1.0` first and `latest` last, verifies both resolve to
the original digest, and never rebuilds the image.
