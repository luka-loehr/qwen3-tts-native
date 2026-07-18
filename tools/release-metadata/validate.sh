#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH='' cd -P "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH='' cd -P "$SCRIPT_DIR/../.." && pwd)
# shellcheck source=lib.sh
. "$SCRIPT_DIR/lib.sh"

usage() {
    printf 'Usage: %s OUTPUT_DIRECTORY\n' "$0" >&2
    exit 2
}

[ "$#" -eq 1 ] || usage
OUTPUT_DIR=$1
[ -d "$OUTPUT_DIR" ] || die "output directory does not exist: $OUTPUT_DIR"

require_command cmp
require_command grep
require_command jq

expected_count=0
for entry in "$OUTPUT_DIR"/* "$OUTPUT_DIR"/.[!.]* "$OUTPUT_DIR"/..?*; do
    [ -e "$entry" ] || continue
    [ ! -L "$entry" ] || die "symlinks are not allowed in release metadata: $entry"
    [ -f "$entry" ] || die "only regular files are allowed in release metadata: $entry"
    case "$(basename "$entry")" in
        APPLICATION-LICENSE.txt|RUST-THIRD-PARTY-LICENSES.html|RUST-SBOM.cdx.json)
            expected_count=$((expected_count + 1))
            ;;
        *)
            die "unexpected release metadata file: $entry"
            ;;
    esac
done
[ "$expected_count" -eq 3 ] || die 'release metadata must contain exactly the three required files'

APPLICATION_LICENSE="$OUTPUT_DIR/APPLICATION-LICENSE.txt"
LICENSE_REPORT="$OUTPUT_DIR/RUST-THIRD-PARTY-LICENSES.html"
SBOM="$OUTPUT_DIR/RUST-SBOM.cdx.json"

for required in "$APPLICATION_LICENSE" "$LICENSE_REPORT" "$SBOM"; do
    [ -s "$required" ] || die "required release metadata file is empty: $required"
done

cmp -s "$REPO_ROOT/LICENSE" "$APPLICATION_LICENSE" || \
    die 'APPLICATION-LICENSE.txt is not byte-for-byte identical to the repository LICENSE'
grep -Fq 'Apache License' "$APPLICATION_LICENSE" || \
    die 'APPLICATION-LICENSE.txt does not contain the approved Apache license heading'
grep -Fq 'Version 2.0, January 2004' "$APPLICATION_LICENSE" || \
    die 'APPLICATION-LICENSE.txt does not contain the approved Apache-2.0 version marker'

grep -Fq '<!doctype html>' "$LICENSE_REPORT" || die 'license report is missing its HTML doctype'
grep -Fq '<html lang="en">' "$LICENSE_REPORT" || die 'license report must declare English content'
grep -Fq '<h1>Rust third-party licenses</h1>' "$LICENSE_REPORT" || \
    die 'license report is missing its expected title'
grep -Fq '<section class="license"' "$LICENSE_REPORT" || \
    die 'license report contains no license text sections'
grep -Fq 'data-source="registry+' "$LICENSE_REPORT" || \
    die 'license report contains no registry-sourced third-party crates'
for first_party in \
    'qwen3-tts-native 0.4.0' \
    'qwen3-tts-native-codec 0.4.0' \
    'qwen3-tts-runtime 0.4.0' \
    'qwen3-tts-server 0.4.0'
do
    if grep -Fq "$first_party" "$LICENSE_REPORT"; then
        die "first-party package was listed as a third party: $first_party"
    fi
done

if grep -Fq '.release-metadata.work.' "$SBOM"; then
    die 'RUST-SBOM.cdx.json leaks its temporary generation path'
fi
grep -Fq 'path+file:///source/native/qwen3-tts-server#0.4.0' "$SBOM" || \
    die 'RUST-SBOM.cdx.json is missing its canonical local root reference'

jq -e --arg spec "$CYCLONEDX_SPEC_VERSION" '
    def bomrefs:
        [.. | objects | .["bom-ref"]? | select(type == "string" and length > 0)];
    .bomFormat == "CycloneDX"
    and .specVersion == $spec
    and (.version | type == "number" and . >= 1)
    and (.metadata | type == "object")
    and (.metadata.timestamp | type == "string" and length > 0)
    and (.metadata.component.name == "qwen3-tts-server")
    and (.metadata.component.version == "0.4.0")
    and (.components | type == "array" and length > 0)
    and ([.components[] | select(
        (.type | type) != "string"
        or (.name | type) != "string"
        or (.version | type) != "string"
    )] | length == 0)
    and ((bomrefs | length) == (bomrefs | unique | length))
    and ((([.dependencies[]? | .ref] + [.dependencies[]?.dependsOn[]?]) - (bomrefs | unique)) | length == 0)
' "$SBOM" >/dev/null || die 'RUST-SBOM.cdx.json failed CycloneDX structural and reference validation'

printf 'Validated release metadata in %s\n' "$OUTPUT_DIR"
