# License Inventory

This directory contains source-controlled license and provenance material for
the native runtime image and release publications.

## Source-controlled material

- model/APACHE-2.0.txt: official Apache License 2.0 text applying to the pinned
  upstream Qwen model.
- model/MODEL_CARD.md: release-specific model identity, scope, hashes,
  limitations, and attribution.
- model/SOURCE.json: machine-readable pinned source and artifact hashes.
- THIRD_PARTY_NOTICES.md: model, NVIDIA base-image, and Rust dependency
  boundaries.
- Bitstream-Vera-LICENSE.txt: copyright, permission, restriction, and warranty
  notice for the font subsets embedded in the benchmark report only; these
  fonts are not part of the inference image.
- application/README.md: record of the repository-wide Apache-2.0 application
  license and its relationship to the separately attributed model grant.

The model license was checked against both the official Apache license text and
the official Qwen3-TTS upstream license. The pinned Hugging Face model metadata
declares apache-2.0.

## Generated release material

Do not hand-author dependency inventories. Before building a release image,
generate these files from the locked Cargo dependency graph in a clean
environment:

    release-metadata/APPLICATION-LICENSE.txt
    release-metadata/RUST-THIRD-PARTY-LICENSES.html
    release-metadata/RUST-SBOM.cdx.json

`APPLICATION-LICENSE.txt` must be a byte-for-byte copy of the root-level
`LICENSE`; it is not generated from Cargo. Pass that directory as the
`release-metadata` named BuildKit context. BuildKit's image-level SBOM and
provenance attestations are additional requirements; they do not replace the
Rust dependency notice or the model manifest.
