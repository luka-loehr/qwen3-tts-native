#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

for command_name in curl install mktemp tar; do
  release_require_command "$command_name"
done

[[ "$(uname -s)" == "Linux" ]] || release_die "release scanners must be bootstrapped on Linux"
[[ "$(uname -m)" == "aarch64" ]] || release_die "release scanners require Linux ARM64"

readonly TOOLS_DIR="$(release_tools_dir)"
readonly WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/qwen3-tts-release-tools.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

download() {
  local url=$1
  local output=$2
  curl --proto '=https' --tlsv1.2 --fail --location --show-error \
    --output "$output" "$url"
}

verify_checksum_file() {
  local expected=$1
  local file=$2
  [[ "$(release_sha256_file "$file")" == "$expected" ]] ||
    release_die "checksum manifest authentication failed: $(basename "$file")"
}

install_release_archive() {
  local project=$1
  local version=$2
  local manifest_sha=$3
  local binary=$4
  local archive="${project}_${version}_linux_arm64.tar.gz"
  local manifest="${project}_${version}_checksums.txt"
  local base_url="https://github.com/$(
    case "$project" in
      grype) printf 'anchore/grype' ;;
      gitleaks) printf 'gitleaks/gitleaks' ;;
      *) release_die "unsupported release project: $project" ;;
    esac
  )/releases/download/v${version}"

  download "$base_url/$manifest" "$WORK_DIR/$manifest"
  verify_checksum_file "$manifest_sha" "$WORK_DIR/$manifest"
  download "$base_url/$archive" "$WORK_DIR/$archive"

  (
    cd "$WORK_DIR"
    grep -E "^[0-9a-f]{64}[[:space:]]+${archive}$" "$manifest" \
      | sha256sum --check --strict -
  ) || release_die "release archive authentication failed: $archive"

  tar -xzf "$WORK_DIR/$archive" -C "$WORK_DIR" "$binary"
  install -D -m 0555 "$WORK_DIR/$binary" "$TOOLS_DIR/bin/$binary"
}

install_release_archive \
  grype "$GRYPE_VERSION" "$GRYPE_CHECKSUMS_SHA256" grype
install_release_archive \
  gitleaks "$GITLEAKS_VERSION" "$GITLEAKS_CHECKSUMS_SHA256" gitleaks

download \
  "https://github.com/sigstore/cosign/releases/download/v${COSIGN_VERSION}/cosign-linux-arm64" \
  "$WORK_DIR/cosign"
[[ "$(release_sha256_file "$WORK_DIR/cosign")" == "$COSIGN_LINUX_ARM64_SHA256" ]] ||
  release_die "Cosign release binary authentication failed"
install -D -m 0555 "$WORK_DIR/cosign" "$TOOLS_DIR/bin/cosign"

release_require_exact_tool_version "$TOOLS_DIR/bin/grype" "$GRYPE_VERSION"
release_require_exact_tool_version "$TOOLS_DIR/bin/gitleaks" "$GITLEAKS_VERSION"
release_require_exact_tool_version "$TOOLS_DIR/bin/cosign" "$COSIGN_VERSION"
printf 'Pinned release scanners are ready in %s\n' "$TOOLS_DIR/bin"
