#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH='' cd -P "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH='' cd -P "$SCRIPT_DIR/../.." && pwd)
# shellcheck source=lib.sh
. "$SCRIPT_DIR/lib.sh"

usage() {
    printf 'Usage: SOURCE_DATE_EPOCH=<unix-seconds> %s OUTPUT_DIRECTORY\n' "$0" >&2
    exit 2
}

[ "$#" -eq 1 ] || usage
[ -n "${SOURCE_DATE_EPOCH:-}" ] || die 'SOURCE_DATE_EPOCH is required'
case "$SOURCE_DATE_EPOCH" in
    *[!0-9]*|'') die 'SOURCE_DATE_EPOCH must contain only decimal Unix seconds' ;;
esac

require_generator_tools

OUTPUT_REQUEST=$1
OUTPUT_PARENT=$(dirname "$OUTPUT_REQUEST")
OUTPUT_NAME=$(basename "$OUTPUT_REQUEST")
case "$OUTPUT_NAME" in
    ''|.|..) die 'OUTPUT_DIRECTORY must name a new child directory' ;;
esac
mkdir -p "$OUTPUT_PARENT"
OUTPUT_PARENT=$(CDPATH='' cd -P "$OUTPUT_PARENT" && pwd)
OUTPUT_DIR="$OUTPUT_PARENT/$OUTPUT_NAME"
[ ! -e "$OUTPUT_DIR" ] || die "output directory already exists: $OUTPUT_DIR"

MANIFEST="$REPO_ROOT/native/qwen3-tts-server/Cargo.toml"
LOCKFILE="$REPO_ROOT/native/qwen3-tts-server/Cargo.lock"
[ -f "$MANIFEST" ] || die "release manifest is missing: $MANIFEST"
[ -f "$LOCKFILE" ] || die "release lockfile is missing: $LOCKFILE"
[ -f "$REPO_ROOT/LICENSE" ] || die 'repository LICENSE is missing'

LOCK_HASH_BEFORE=$(sha256_file "$LOCKFILE")

WORK_DIR=$(mktemp -d "$OUTPUT_PARENT/.release-metadata.work.XXXXXX")
cleanup() {
    if [ -n "${WORK_DIR:-}" ]; then
        case "$WORK_DIR" in
            "$OUTPUT_PARENT"/.release-metadata.work.*) rm -rf "$WORK_DIR" ;;
            *) printf 'Refusing to clean unexpected work directory: %s\n' "$WORK_DIR" >&2 ;;
        esac
    fi
}
trap cleanup EXIT HUP INT TERM

PUBLISH_DIR="$WORK_DIR/output"
MIRROR_ROOT="$WORK_DIR/source"
mkdir -p "$PUBLISH_DIR" "$MIRROR_ROOT/native"

# This is the only dependency-network step. The lockfile makes archive
# identities immutable; every generator below runs offline afterwards.
cargo fetch \
    --locked \
    --manifest-path "$MANIFEST" \
    --target "$RELEASE_TARGET"

[ "$(sha256_file "$LOCKFILE")" = "$LOCK_HASH_BEFORE" ] || \
    die 'cargo fetch changed the release lockfile'

cargo metadata \
    --format-version 1 \
    --locked \
    --offline \
    --all-features \
    --filter-platform "$RELEASE_TARGET" \
    --manifest-path "$MANIFEST" \
    >/dev/null

# cargo-cyclonedx always writes beside its manifest. Mirror only the four
# crates shipped in the final image, excluding Cargo build artifacts, so the
# generator cannot write into or mutate the source tree.
for crate in \
    qwen3-tts-native \
    qwen3-tts-native-codec \
    qwen3-tts-runtime \
    qwen3-tts-server
do
    source_crate="$REPO_ROOT/native/$crate"
    mirror_crate="$MIRROR_ROOT/native/$crate"
    archive="$WORK_DIR/$crate.tar"
    [ -d "$source_crate" ] || die "release crate is missing: $source_crate"
    mkdir -p "$mirror_crate"
    tar \
        -C "$source_crate" \
        --exclude='./target' \
        --exclude='target' \
        --exclude='*.cdx.json' \
        -cf "$archive" \
        .
    tar -C "$mirror_crate" -xf "$archive"
    rm -f "$archive"
done

MIRROR_MANIFEST="$MIRROR_ROOT/native/qwen3-tts-server/Cargo.toml"
MIRROR_SBOM="$MIRROR_ROOT/native/qwen3-tts-server/RUST-SBOM.cdx.json"

cp "$REPO_ROOT/LICENSE" "$PUBLISH_DIR/APPLICATION-LICENSE.txt"

"$CARGO_ABOUT_BIN" about generate \
    --config "$SCRIPT_DIR/about.toml" \
    --manifest-path "$MIRROR_MANIFEST" \
    --all-features \
    --target "$RELEASE_TARGET" \
    --frozen \
    --fail \
    --output-file "$PUBLISH_DIR/RUST-THIRD-PARTY-LICENSES.html" \
    "$SCRIPT_DIR/about.hbs"

env \
    SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
    CARGO_NET_OFFLINE=true \
    "$CARGO_CYCLONEDX_BIN" cyclonedx \
        --manifest-path "$MIRROR_MANIFEST" \
        --format json \
        --all-features \
        --target "$RELEASE_TARGET" \
        --license-strict \
        --license-accept-named 'Apache-2.0 / MIT' \
        --license-accept-named 'MIT/Apache-2.0' \
        --spec-version "$CYCLONEDX_SPEC_VERSION" \
        --override-filename RUST-SBOM.cdx

[ -s "$MIRROR_SBOM" ] || die 'cargo-cyclonedx did not produce the expected SBOM'

# serialNumber is optional and random. cargo-cyclonedx also identifies local
# path packages with the absolute temporary mirror path. Remove the serial,
# map the mirror to a stable logical source root, and sort object keys.
jq --sort-keys '
    walk(
        if type == "string" then
            gsub("path\\+file://.*/source/native/"; "path+file:///source/native/")
            | gsub("download_url=file://.*/source/native/"; "download_url=file:///source/native/")
        else
            .
        end
    )
    | del(.serialNumber)
' "$MIRROR_SBOM" \
    >"$PUBLISH_DIR/RUST-SBOM.cdx.json"

[ "$(sha256_file "$LOCKFILE")" = "$LOCK_HASH_BEFORE" ] || \
    die 'release metadata generation changed the source lockfile'

chmod 0644 \
    "$PUBLISH_DIR/APPLICATION-LICENSE.txt" \
    "$PUBLISH_DIR/RUST-THIRD-PARTY-LICENSES.html" \
    "$PUBLISH_DIR/RUST-SBOM.cdx.json"

"$SCRIPT_DIR/validate.sh" "$PUBLISH_DIR"

# OUTPUT_DIR was required not to exist, and WORK_DIR is on the same
# filesystem, so this directory rename publishes all three files atomically.
mv "$PUBLISH_DIR" "$OUTPUT_DIR"
printf 'Generated reproducible release metadata in %s\n' "$OUTPUT_DIR"
