# Reproducible release metadata

This directory generates and validates the three files required by the native
runtime image:

- `APPLICATION-LICENSE.txt` — an exact byte-for-byte copy of the repository's
  owner-approved Apache-2.0 `LICENSE`;
- `RUST-THIRD-PARTY-LICENSES.html` — full license texts for registry or Git
  crates in the locked ARM64 Linux release graph;
- `RUST-SBOM.cdx.json` — a canonical CycloneDX 1.5 JSON SBOM for that graph.

The pipeline uses POSIX shell, Cargo tools, and `jq`. It does not require or
invoke Python, Node.js, or an on-device model.

## Release graph

`native/qwen3-tts-server/Cargo.toml` is the release root because the final
container ships the server and healthcheck and the server graph includes the
runtime, talker host, and codec host through local path dependencies. The
benchmark crate and development-only dependencies are not part of the final
image and are intentionally excluded. Build dependencies remain included.

The target is fixed to `aarch64-unknown-linux-gnu`, matching the DGX Spark
runtime image. Adding a new local release crate requires adding it to the
auditable mirror list in `generate.sh`.

## Pinned tools

`versions.env` is the single source of truth:

| Tool or format | Pinned value |
| --- | --- |
| Cargo | 1.95.0 |
| cargo-about | 0.9.1 |
| cargo-cyclonedx | 0.5.9 |
| jq | 1.7.1 |
| CycloneDX | 1.5 JSON |
| Cargo target | aarch64-unknown-linux-gnu |

The scripts fail closed on a version mismatch. Platform builds of `jq` may
append a vendor suffix, such as `jq-1.7.1-apple`, but the semantic version must
remain exactly 1.7.1.

## Exact usage

Run these commands from any directory in a clean checkout:

```sh
./tools/release-metadata/bootstrap-tools.sh

export SOURCE_DATE_EPOCH="$(git -C /absolute/path/to/qwen3-tts-native show -s --format=%ct HEAD)"
./tools/release-metadata/generate.sh /absolute/path/to/release-metadata
./tools/release-metadata/validate.sh /absolute/path/to/release-metadata
```

The output directory must not already exist. Generation happens in a hidden
temporary sibling directory. Only after all three artifacts pass validation is
the complete directory published with one same-filesystem rename.

The default tool prefix is a versioned directory under `XDG_CACHE_HOME` or the
user cache directory. To make it explicit in CI:

```sh
export RELEASE_METADATA_TOOLS_DIR=/opt/qwen3-tts-release-tools
./tools/release-metadata/bootstrap-tools.sh
```

`bootstrap-tools.sh` installs both Cargo plugins with an exact `--version`, the
upstream `--locked` dependency graph, and an isolated `--root`. It never
installs a floating `latest` version.

## Network and lockfile behavior

`generate.sh` performs one `cargo fetch --locked` step for the complete pinned
lockfile graph. Fetching the complete graph is required because Cargo 1.95 can
otherwise omit host-side metadata packages needed by the frozen license pass.
It then filters metadata and generated output to the pinned ARM64 target while
running cargo-about and cargo-cyclonedx offline.
The source `Cargo.lock` SHA-256 is checked before and after every phase.

Because cargo-cyclonedx writes beside its input manifest, the script creates a
temporary `target`-free mirror of only the four shipped crates. The generator
therefore cannot create an SBOM in the source tree. Its random optional
CycloneDX `serialNumber` is removed, temporary local package paths are mapped
to the logical root `/source/native/`, and JSON object keys are sorted. The
timestamp still comes from the required `SOURCE_DATE_EPOCH`.

## Validation

`validate.sh` fails unless all of the following are true:

- the directory contains exactly the three expected regular files and no
  symlinks;
- the application license is byte-identical to the authoritative root
  `LICENSE` and contains the Apache-2.0 version markers;
- the English HTML report contains license text and at least one
  registry-sourced crate, while no local first-party crate is presented as a
  third party;
- the SBOM is valid JSON with the expected CycloneDX version, root component,
  non-empty typed component list, unique BOM references, and no dangling
  dependency references.

`cargo-about --fail` additionally rejects every license that is not satisfied
by the explicit allowlist in `about.toml`. A dependency license change therefore
requires review and a source-controlled policy update; it cannot silently pass.
The SBOM generator runs in strict SPDX mode with two explicit compatibility
exceptions for the legacy strings `Apache-2.0 / MIT` and `MIT/Apache-2.0`
reported by `fnv` and `serde_urlencoded`; their accepted license choices are
still independently enforced by cargo-about.

## Reproducibility gate

Run the full generator twice and compare every byte:

```sh
./tools/release-metadata/test-reproducibility.sh
```

The test uses `SOURCE_DATE_EPOCH=1704067200` unless the caller provides another
decimal Unix timestamp. Both runs are independently validated before their
three outputs are compared.

The dependency report is compliance evidence, not legal advice. The image-level
BuildKit SBOM and provenance attestations remain separate release requirements.
