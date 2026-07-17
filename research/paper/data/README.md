# Manifest-derived paper data

`evidence_placeholders.tex` is intentionally buildable before final evidence
exists. A release finalizer must replace it atomically from the validated
production manifest; manual entry is forbidden.

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
