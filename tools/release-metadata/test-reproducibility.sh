#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH='' cd -P "$(dirname "$0")" && pwd)
# shellcheck source=lib.sh
. "$SCRIPT_DIR/lib.sh"

require_generator_tools

TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/qwen3-tts-release-metadata-test.XXXXXX")
cleanup() {
    case "${TEST_ROOT:-}" in
        "${TMPDIR:-/tmp}"/qwen3-tts-release-metadata-test.*) rm -rf "$TEST_ROOT" ;;
        '') ;;
        *) printf 'Refusing to clean unexpected test directory: %s\n' "$TEST_ROOT" >&2 ;;
    esac
}
trap cleanup EXIT HUP INT TERM

export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1704067200}"

"$SCRIPT_DIR/generate.sh" "$TEST_ROOT/run-one"
"$SCRIPT_DIR/generate.sh" "$TEST_ROOT/run-two"

for artifact in \
    APPLICATION-LICENSE.txt \
    RUST-THIRD-PARTY-LICENSES.html \
    RUST-SBOM.cdx.json
do
    cmp -s "$TEST_ROOT/run-one/$artifact" "$TEST_ROOT/run-two/$artifact" || \
        die "artifact is not byte-reproducible: $artifact"
done

"$SCRIPT_DIR/validate.sh" "$TEST_ROOT/run-one"
"$SCRIPT_DIR/validate.sh" "$TEST_ROOT/run-two"
printf 'Release metadata is byte-reproducible for SOURCE_DATE_EPOCH=%s\n' "$SOURCE_DATE_EPOCH"
