#!/bin/sh

set -eu

if [ -z "${SCRIPT_DIR:-}" ]; then
    printf '%s\n' 'SCRIPT_DIR must be set before sourcing lib.sh' >&2
    exit 1
fi

# shellcheck source=versions.env
. "$SCRIPT_DIR/versions.env"

die() {
    printf 'release-metadata: %s\n' "$*" >&2
    exit 1
}

cache_root() {
    if [ -n "${RELEASE_METADATA_TOOLS_DIR:-}" ]; then
        printf '%s\n' "$RELEASE_METADATA_TOOLS_DIR"
    elif [ -n "${XDG_CACHE_HOME:-}" ]; then
        printf '%s/qwen3-tts-release-metadata/cargo-%s-about-%s-cyclonedx-%s\n' \
            "$XDG_CACHE_HOME" \
            "$CARGO_VERSION" \
            "$CARGO_ABOUT_VERSION" \
            "$CARGO_CYCLONEDX_VERSION"
    elif [ -n "${HOME:-}" ]; then
        printf '%s/.cache/qwen3-tts-release-metadata/cargo-%s-about-%s-cyclonedx-%s\n' \
            "$HOME" \
            "$CARGO_VERSION" \
            "$CARGO_ABOUT_VERSION" \
            "$CARGO_CYCLONEDX_VERSION"
    else
        die 'set RELEASE_METADATA_TOOLS_DIR, XDG_CACHE_HOME, or HOME'
    fi
}

TOOLS_DIR=$(cache_root)
CARGO_ABOUT_BIN=${CARGO_ABOUT_BIN:-"$TOOLS_DIR/bin/cargo-about"}
CARGO_CYCLONEDX_BIN=${CARGO_CYCLONEDX_BIN:-"$TOOLS_DIR/bin/cargo-cyclonedx"}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command is missing: $1"
}

second_word() {
    awk '{ print $2; exit }'
}

require_base_tools() {
    require_command cargo
    require_command awk
    require_command cmp
    require_command jq
    require_command mktemp
    require_command tar

    actual_cargo=$(cargo --version | second_word)
    [ "$actual_cargo" = "$CARGO_VERSION" ] || \
        die "cargo $CARGO_VERSION is required; found $actual_cargo"

    actual_jq=$(jq --version)
    case "$actual_jq" in
        "jq-$JQ_VERSION"|"jq-$JQ_VERSION-"*) ;;
        *) die "jq $JQ_VERSION is required; found $actual_jq" ;;
    esac
}

require_generator_tools() {
    require_base_tools

    [ -x "$CARGO_ABOUT_BIN" ] || \
        die "cargo-about is missing; run $SCRIPT_DIR/bootstrap-tools.sh"
    [ -x "$CARGO_CYCLONEDX_BIN" ] || \
        die "cargo-cyclonedx is missing; run $SCRIPT_DIR/bootstrap-tools.sh"

    actual_about=$("$CARGO_ABOUT_BIN" --version | second_word)
    [ "$actual_about" = "$CARGO_ABOUT_VERSION" ] || \
        die "cargo-about $CARGO_ABOUT_VERSION is required; found $actual_about"

    actual_cyclonedx=$("$CARGO_CYCLONEDX_BIN" cyclonedx --version | second_word)
    [ "$actual_cyclonedx" = "$CARGO_CYCLONEDX_VERSION" ] || \
        die "cargo-cyclonedx $CARGO_CYCLONEDX_VERSION is required; found $actual_cyclonedx"
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{ print $1; exit }'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{ print $1; exit }'
    else
        die 'sha256sum or shasum is required'
    fi
}
