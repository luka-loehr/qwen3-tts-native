#!/usr/bin/env bash

set -euo pipefail

readonly RELEASE_IMAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly RELEASE_IMAGE="ghcr.io/luka-loehr/qwen3-tts-native"
readonly RELEASE_MODEL_ID="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
readonly RELEASE_MODEL_REVISION="5ecdb67327fd37bb2e042aab12ff7391903235d3"
readonly RELEASE_MODEL_MANIFEST_SHA256="9bb96a8d24bbb2d8933245e27083b8e7290346b776306dcb8a8f3aed68594527"
readonly RELEASE_MODEL_VOICE_SHA256="391e8db219f292c515297cdceeb43e4eae67cdde35fa57e79a6a8a532fca0522"
readonly RELEASE_MODEL_DECODER_SHA256="062caa0a31346422410e4c0d2494aec14be20553f8cb0b71a875329de99ce180"
readonly RELEASE_CERTIFICATE_IDENTITY="https://github.com/luka-loehr/qwen3-tts-native/.github/workflows/sign-ghcr-image.yml@refs/heads/main"
readonly RELEASE_CERTIFICATE_ISSUER="https://token.actions.githubusercontent.com"

# shellcheck source=versions.env
source "$(dirname "${BASH_SOURCE[0]}")/versions.env"

release_die() {
  printf 'release-image: %s\n' "$*" >&2
  exit 1
}

release_require_command() {
  command -v "$1" >/dev/null 2>&1 || release_die "required command is missing: $1"
}

release_require_file() {
  [[ -f "$1" ]] || release_die "required file is missing: $1"
}

release_require_directory() {
  [[ -d "$1" ]] || release_die "required directory is missing: $1"
}

release_require_new_directory() {
  local path=$1
  [[ "$path" = /* ]] || release_die "evidence directory must be absolute: $path"
  [[ ! -e "$path" ]] || release_die "evidence path already exists: $path"
  install -d -m 0755 "$path"
}

release_validate_version() {
  [[ "$1" =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z]+([.-][0-9A-Za-z]+)*)?(\+[0-9A-Za-z]+([.-][0-9A-Za-z]+)*)?$ ]] ||
    release_die "version must be a v-prefixed semantic version: $1"
}

release_validate_digest() {
  [[ "$1" =~ ^sha256:[0-9a-f]{64}$ ]] ||
    release_die "digest must match sha256:<64 lowercase hexadecimal characters>"
}

release_validate_commit() {
  [[ "$1" =~ ^[0-9a-f]{40}$ ]] ||
    release_die "source revision must be a 40-character lowercase Git commit"
}

release_sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{ print $1; exit }'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{ print $1; exit }'
  else
    release_die "sha256sum or shasum is required"
  fi
}

release_tools_dir() {
  if [[ -n "${RELEASE_IMAGE_TOOLS_DIR:-}" ]]; then
    printf '%s\n' "$RELEASE_IMAGE_TOOLS_DIR"
  elif [[ -n "${XDG_CACHE_HOME:-}" ]]; then
    printf '%s/qwen3-tts-release-image\n' "$XDG_CACHE_HOME"
  elif [[ -n "${HOME:-}" ]]; then
    printf '%s/.cache/qwen3-tts-release-image\n' "$HOME"
  else
    release_die "set RELEASE_IMAGE_TOOLS_DIR, XDG_CACHE_HOME, or HOME"
  fi
}

release_read_record() {
  local record=$1
  release_require_file "$record"
  jq -e \
    --arg image "$RELEASE_IMAGE" \
    --arg model_id "$RELEASE_MODEL_ID" \
    --arg model_revision "$RELEASE_MODEL_REVISION" '
      .schema == "qwen3-tts-native/release-record/v1"
      and .image == $image
      and (.digest | test("^sha256:[0-9a-f]{64}$"))
      and (.release_version | test("^v[0-9]+\\.[0-9]+\\.[0-9]+([+-][0-9A-Za-z.-]+)?$"))
      and (.source_revision | test("^[0-9a-f]{40}$"))
      and (.candidate_tag | type == "string" and length > 0)
      and (.git_tag | type == "string" and length > 0)
      and .platform == "linux/arm64"
      and .model.id == $model_id
      and .model.revision == $model_revision
    ' "$record" >/dev/null || release_die "invalid release record: $record"
}

release_record_value() {
  jq -er "$2" "$1"
}

release_require_exact_tool_version() {
  local command_path=$1
  local expected=$2
  local actual
  case "$(basename "$command_path")" in
    grype)
      actual=$("$command_path" version | awk -F': *' '$1 == "Version" { print $2; exit }')
      ;;
    gitleaks)
      actual=$("$command_path" version | awk '{ print $NF; exit }')
      actual=${actual#v}
      ;;
    cosign)
      actual=$("$command_path" version 2>/dev/null | awk -F': *' '$1 ~ /GitVersion/ { gsub(/^v/, "", $2); print $2; exit }')
      ;;
    *)
      release_die "unsupported pinned tool: $command_path"
      ;;
  esac
  [[ "$actual" == "$expected" ]] ||
    release_die "$(basename "$command_path") $expected is required; found ${actual:-unknown}"
}
