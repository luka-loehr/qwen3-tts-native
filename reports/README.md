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

The generator accepts one JSON manifest. Production uses schema version `1.2`
and consumes the Rust benchmark client's files directly. There is no normalized
measurement export and no hand-edited intermediate statistics file.

A production bundle contains:

- the exact benchmark input workload JSONL;
- one unmodified client `summary.json`, `requests.jsonl`, and `packets.jsonl`
  for every engine, profile, and round;
- raw timestamped telemetry files for every run;
- one `run_resources` manifest record for every run, with digest-bound links to
  its raw telemetry;
- one digest-bound model-artifact manifest per implementation;
- optional, separately digest-bound OCI registry metadata when registry digest
  or compressed-size claims are reported;
- one manifest containing immutable system, model, implementation, workload,
  methodology, and evidence-file metadata.

Each evidence descriptor records its role, relative path, byte length, SHA-256,
and, for run-specific evidence, engine, profile, and round. Parsed client files
must be JSON or JSONL. Raw provenance may additionally be CSV, TXT, LOG, STDOUT,
or STDERR. All files must remain inside the manifest directory. Symlinks,
absolute paths, path
traversal, duplicate records, non-finite numbers, unrecognized fields, digest
mismatches, and missing run triples are rejected.

The workload JSONL is the canonical workload identity. Its exact ordered row
seeds are repeated as `workload.ordered_seeds` and cross-validated; there is no
meaningful single corpus seed. For each profile and
round, Native and SGLang must have exactly the same ordered request indices,
workload IDs, text hashes, VoiceDesign hashes, languages, streaming mode, and
normalized sampling hashes. The validator recomputes text, VoiceDesign, and
normalized-sampling SHA-256 values rather than trusting the summaries.

The top-level model declares only the genuinely common repository, revision,
and variant. Precision, parameter count, optional artifact-manifest digest, and
complete weight metadata belong to each implementation's separately
digest-bound `model_artifact`. Native and Stock values may differ and are never
forced equal. The artifact evidence also binds the metadata to the tested local
Docker image ID. A missing Stock artifact file fails validation rather than
copying the Native declaration.

Local Docker identity and OCI registry identity are separate. `local_image.id`
is Docker's local `.Id`; `local_image.unpacked_size_bytes` is Docker inspect's
local unpacked/virtual `Size`. An optional registry manifest digest and optional
compressed size appear only with their own registry evidence. Neither value is
inferred from the local image ID or local size.

Every production workload entry must set `stream=true` and
`max_duration_seconds=20.48` exactly. At 24 kHz with 1,920 samples per codec
frame, 20.48 seconds is the shared 256-frame request ceiling. Missing values,
nearby floating-point values, and per-entry duration differences are rejected;
the production report never compares an uncapped request with a capped one.

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

Raw telemetry may start before warmup so the audit trail includes the idle and
pre-warmup thermal state. The values placed in `run_resources` must nevertheless be
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
All measured definitions, clock sources, measurement tools, sampling intervals,
run ordering, and environmental controls are mandatory manifest fields. These
runs use already-running containers, so they do not support a startup-time or
canonical launch-command claim; schema v1.2 rejects unverified scalar fields for
either claim.

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

Stock SGLang has an additional conservative boundary gate. For every successful
response, the validator sums the already validated raw-PCM packet payload bytes,
divides that byte count by two for signed 16-bit mono samples, and requires the
result to match both the request `samples` field and `audio_seconds * 24,000`.
It deliberately does not use `response_bytes`, which includes HTTP headers and
chunked-transfer framing rather than audio alone.
The result must be strictly below 489,600 samples and the recorded duration must
be strictly below 20.40 seconds. This is strictly shorter than 255 codec frames;
a frame-aligned response can therefore contain at most 254 frames.

The exclusive 255-frame boundary is one 80 ms frame below the nominal 256-frame
request ceiling. With `max_new_tokens=256`, the stock adapter's
token-to-decodable-frame accounting can be off by one: a 255-frame waveform can
be the observable result of reaching the 256-token generation limit. The
255-frame result is therefore rejected rather than misclassified as natural
completion. Because the boundary is exclusive, the largest frame-aligned value
that passes is 254 frames, or 20.32 seconds, two frames below the nominal
20.48-second ceiling. Passing this shorter-audio gate only excludes the known
length-boundary case; it does not prove natural EOS, so all accepted stock
SGLang EOS fields remain unknown.

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
12. local container footprint and optional registry provenance;
13. limitations;
14. raw evidence and artifact digest appendix.

The report does not infer missing values, interpolate failed measurements,
classify unknown SGLang EOS, or silently compare unequal workloads.
