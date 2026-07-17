# Manifest-derived paper data

`evidence_placeholders.tex` is intentionally buildable before final evidence
exists. `tools/finalize_evidence.py` replaces it atomically from a validated
schema-v1.2 production evidence bundle; manual entry is forbidden.

When `\FinalEvidenceAvailabletrue` is set, the following files are mandatory:

- `native-runs.dat`
- `sglang-runs.dat`

Each file must be ASCII whitespace-separated data with this header and exactly
six rows, ordered by round and then concurrency:

```text
concurrency round ttfa_p95_ms aggregate_rtf
```

The two rounds must each contain concurrency values `1`, `3`, and `6`. Values
remain at source precision. PGFPlots handles display rounding.

Run the finalizer from the paper directory:

```bash
make finalize-evidence \
  MANIFEST=/absolute/path/to/validated-evidence-root/manifest.json
```

The command deterministically writes all 12 performance rows, all 12 resource
rows, both six-row plot files, source and artifact identities, and fixed
English evidence summaries. Registry compressed size is `N/A` when the
manifest has no digest-bound value. The command never estimates a missing
measurement and never translates or rewrites prose with a language model.
