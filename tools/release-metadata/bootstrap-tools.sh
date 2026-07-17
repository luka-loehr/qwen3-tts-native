#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH='' cd -P "$(dirname "$0")" && pwd)
# shellcheck source=lib.sh
. "$SCRIPT_DIR/lib.sh"

require_base_tools
mkdir -p "$TOOLS_DIR"

about_version=''
if [ -x "$CARGO_ABOUT_BIN" ]; then
    about_version=$("$CARGO_ABOUT_BIN" --version | second_word)
fi
if [ "$about_version" != "$CARGO_ABOUT_VERSION" ]; then
    cargo install \
        --root "$TOOLS_DIR" \
        --version "$CARGO_ABOUT_VERSION" \
        --locked \
        --force \
        --features cli \
        cargo-about
fi

cyclonedx_version=''
if [ -x "$CARGO_CYCLONEDX_BIN" ]; then
    cyclonedx_version=$("$CARGO_CYCLONEDX_BIN" cyclonedx --version | second_word)
fi
if [ "$cyclonedx_version" != "$CARGO_CYCLONEDX_VERSION" ]; then
    cargo install \
        --root "$TOOLS_DIR" \
        --version "$CARGO_CYCLONEDX_VERSION" \
        --locked \
        --force \
        cargo-cyclonedx
fi

require_generator_tools
printf 'Pinned release tools are ready in %s\n' "$TOOLS_DIR"
