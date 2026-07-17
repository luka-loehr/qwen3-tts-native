# Benchmark Evidence

This directory stores immutable, reviewable evidence captured from qualifying or
diagnostic DGX Spark runs. Evidence is organized by subject and the repository
commit that produced it.

Provenance captures establish the exact software and container state. Performance
captures additionally contain the raw request, packet, telemetry, and run-manifest
files consumed by the report generator. A provenance directory is not, by itself,
a performance result.

Every evidence set includes a `SHA256SUMS` file. Verify a set from within its
directory with:

```bash
shasum -a 256 -c SHA256SUMS
```

Generated PDFs are views over verified evidence. The JSON, JSONL, telemetry, and
hash manifests remain the source of truth.
