# Benchmark report pipeline

This directory contains the deterministic, fail-closed report pipeline for the
Qwen3 TTS Native versus SGLang benchmark. It is reporting tooling only. Python
and ReportLab are not part of the production inference runtime or container.

The pipeline deliberately does not ship a benchmark PDF. A final PDF is only
valid after real evidence has been collected, validated, rendered, and visually
inspected page by page.

## Directory layout

```text
reports/
  evidence.schema.json        Versioned manifest and evidence contract
  generate_report.py          Validator and PDF generator
  fixtures/layout_only/       Synthetic layout fixture, never benchmark evidence
  output/                     Final, reviewed production PDFs only
  tests/                      Validator and rendering tests
  tmp/pdfs/                   Intermediate renders and local QA artifacts
```

## Production evidence bundle

The generator accepts one JSON manifest. Production uses schema version `1.1`
and consumes the Rust benchmark client's files directly. There is no normalized
measurement export and no hand-edited intermediate statistics file.

A production bundle contains:

- the exact benchmark input workload JSONL;
- one unmodified client `summary.json`, `requests.jsonl`, and `packets.jsonl`
  for every engine, profile, and round;
- raw timestamped telemetry files for every run;
- one `run_resources` manifest record for every run, with digest-bound links to
  its raw telemetry;
- one manifest containing immutable system, model, implementation, workload,
  methodology, and evidence-file metadata.

Each evidence descriptor records its role, relative path, byte length, SHA-256,
and, for run-specific evidence, engine, profile, and round. Parsed client files
must be JSON or JSONL. Raw provenance may additionally be CSV, TXT, LOG, STDOUT,
or STDERR. All files must remain inside the manifest directory. Symlinks,
absolute paths, path
traversal, duplicate records, non-finite numbers, unrecognized fields, digest
mismatches, and missing run triples are rejected.

The workload JSONL is the canonical workload identity. For each profile and
round, Native and SGLang must have exactly the same ordered request indices,
workload IDs, text hashes, VoiceDesign hashes, languages, streaming mode, and
normalized sampling hashes. The validator recomputes text, VoiceDesign, and
normalized-sampling SHA-256 values rather than trusting the summaries. It also
rejects an implementation whose repository, model revision, precision, or
model-manifest digest differs from the shared model declaration.

Production evidence must include the B1, B3, and B6 profiles at concurrency 1,
3, and 6. It must contain at least two contiguous rounds per engine and profile,
at least 24 warmups per run, and at least 200 successful measured requests in
every run. Warmups are declared separately and are never included in measured
rows.

`run_resources` contains measured run-level process RSS, GPU-visible unified
memory, mean and peak power, integrated energy, telemetry sampling interval, and
the observed count of competing CUDA processes. Production is rejected when
sampling is slower than 200 ms, competing CUDA work is present, or a telemetry
path is absent, unverified, or assigned to another run.

Raw telemetry may start before warmup so the audit trail includes startup and
thermal state. The values placed in `run_resources` must nevertheless be
reduced over the measured scenario window only. Warmup energy and warmup-only
peaks must not be charged to measured audio. The collection controller therefore
needs unambiguous measured-phase boundaries that can be aligned with telemetry
timestamps. If those boundaries are unavailable, power and energy results are
not publishable. Cgroup memory is container memory; it must not be relabeled as
process RSS without a separate process-level measurement.

## Measurement semantics

Each successful client request records:

- time to first playable PCM audio (TTFA);
- total request latency and emitted audio duration;
- request real-time factor (request wall time / audio duration);
- packet count and every inter-packet interval;
- response size;
- protocol continuity and explicit termination fields.

Failed, cancelled, and timed-out requests remain in the evidence and contribute
to reliability results. Performance percentiles only use successful requests.
All definitions, clock sources, measurement tools, sampling intervals, run
ordering, and environmental controls are mandatory manifest fields.

The client summary is verified against raw requests. The report distinguishes:

- aggregate RTF: sum of scenario wall times divided by total successful audio
  duration;
- summed-request-wall RTF: sum of successful request wall times divided by
  total successful audio duration;
- successful throughput: successful requests divided by scenario wall time;
- attempted throughput: all completed attempt records divided by scenario wall
  time.

Only aggregate RTF is the parallel scenario throughput RTF for B3 and B6.
Summed-request-wall RTF is retained as a request-latency view and is never
substituted for aggregate RTF.

Termination policy is deliberately asymmetric and fail-closed. Every successful
Native request must declare `finish_reason="stop"`, `natural_eos=true`, and
`length_limited=false`. The stock SGLang raw-PCM API exposes no finish reason, so
every successful SGLang request must retain `natural_eos=null`,
`length_limited=null`, and `finish_reason=null`. The generator never imputes
natural EOS from transport completion.

## Generate a production report

Use the Codex bundled PDF runtime when available:

```bash
/Users/luka/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12 \
  reports/generate_report.py /absolute/path/to/evidence/manifest.json
```

The default output is
`reports/output/<benchmark-id>.pdf`. An explicit production output can be set
with `--output`. Validate without generating a PDF with:

```bash
python3 reports/generate_report.py evidence/manifest.json --validate-only
```

The PDF uses vector tables and charts. Engine identity is conveyed through
black-and-white fills, hatch patterns, line styles, and point markers so the
report remains readable in grayscale and for readers who cannot rely on color.

## Test fixture safety

`fixtures/layout_only` is synthetic, uses the isolated legacy schema version
`1.0`, and is marked `evidence_kind` equal to `test_fixture`. Version `1.0` is
forbidden for production. The default command rejects the fixture. Rendering
requires the explicit `--allow-test-fixture` flag, adds a prominent fixture
banner to every page, and refuses to write into `reports/output/`.

Example layout-only render:

```bash
python3 reports/generate_report.py \
  reports/fixtures/layout_only/manifest.json \
  --allow-test-fixture \
  --output reports/tmp/pdfs/layout-fixture.pdf
```

This output is not benchmark evidence and must never be published as a result.

## Determinism and release QA

The generator normalizes PDF metadata, uses fixed typography and page geometry,
sorts all evidence before aggregation, and enables ReportLab invariant mode.
Given identical evidence and the same ReportLab version, output bytes are
stable.

Before release:

1. Run the test suite.
2. Validate the production manifest with `--validate-only`.
3. Generate the production PDF under `reports/output/`.
4. Render every page to PNG under `reports/tmp/pdfs/` with `pdftoppm`.
5. Inspect every page for clipping, overlap, unreadable labels, broken tables,
   missing glyphs, incorrect pagination, and chart ambiguity.
6. Check the PDF with `pdfinfo` and extract text as a secondary sanity check.
7. Keep the PDF only after visual review has zero defects.

Example rendering command:

```bash
pdftoppm -png reports/output/qwen3-tts-native-vs-sglang.pdf \
  reports/tmp/pdfs/qwen3-tts-native-vs-sglang
```

## Report contents

The generated report includes:

1. title and evidence identity;
2. executive summary;
3. methodology and fairness controls;
4. system, model, implementation, and version inventory;
5. TTFA distributions;
6. RTF distributions;
7. request and audio throughput;
8. process RSS and GPU-visible unified memory;
9. power and energy;
10. streaming cadence;
11. reliability;
12. startup time and image size;
13. limitations;
14. raw evidence and artifact digest appendix.

The report does not infer missing values, interpolate failed measurements,
classify unknown SGLang EOS, or silently compare unequal workloads.
