# Production manifest assembler

`assemble_production_manifest.py` turns one canonical workload and exactly
twelve completed `run-qualifying-benchmark.sh` directories into a create-new
`evidence.schema.json` version `1.1` production manifest. The tool uses only the
Python standard library. It never edits, repairs, normalizes, or copies source
evidence.

The manifest is intentionally assembled only after all benchmark runs finish.
No system, model, implementation, methodology, limitation, author, timestamp,
or startup claim is inferred from the host. Every such claim must be supplied
in the explicit configuration file.

## Expected evidence set

The manifest output directory is the evidence root. Both the central workload
and the runs root must already be inside it. A recommended layout is:

```text
evidence-root/
  workload/
    workload.jsonl
  runs/
    round-01/
      native/{B1,B3,B6}/
      sglang/{B1,B3,B6}/
    round-02/
      native/{B1,B3,B6}/
      sglang/{B1,B3,B6}/
  manifest.json                 created by the assembler
```

Directory names are not trusted as run identity. The assembler discovers runs
through `provenance/invocation.json` and `run-resource.json`, then requires the
embedded engine, profile, round, evidence prefix, image, workload, client, and
telemetry identities to agree. The final set must be exactly:

- engines `native` and `sglang`;
- profiles `B1`, `B3`, and `B6`;
- rounds `1` and `2`;
- one and only one directory for each of the twelve triples.

Failed staging directories, a third round, a duplicate identity, or a missing
run cause assembly to fail. Every directory and file below the runs root must
belong to one of the twelve discovered run directories; unowned partial logs,
empty staging directories, and other leftover artifacts are rejected.

## Required metadata configuration

Pass one UTF-8 JSON object containing exactly these top-level keys:

```json
{
  "report": {},
  "system": {},
  "model": {},
  "workload": {},
  "implementations": [],
  "methodology": {},
  "limitations": []
}
```

Each value must already satisfy its corresponding definition in
`reports/evidence.schema.json`. Use the schema as the field-level template; do
not copy placeholder values into production evidence. In particular:

- `report.generated_at` is an explicit evidence timestamp, not the time the
  assembler happens to run;
- `system` must contain the observed Spark identity, software versions,
  physical unified memory, power source, and notes;
- `model` must contain the immutable repository revision, precision, model
  manifest digest, and every declared weight digest and byte length;
- both `implementations` must contain the actually tested image ID in
  `image_digest`, measured startup time, source commit, command digest, and
  exact runtime description;
- `methodology` must define all clocks and metrics and must set the actual
  telemetry interval used by every run;
- `limitations` must state the real comparison limitations, including the
  stock SGLang completion-buffered boundary where applicable.

The configuration's `workload.corpus_sha256` must equal the byte-for-byte
SHA-256 of the supplied central workload. Production workload metadata is
fixed to mono 24 kHz PCM16 streaming, at least 24 warmups, at least 200
successful measured requests per run, exactly two rounds, and exactly the B1,
B3, and B6 profiles.

Every canonical workload row must explicitly set `stream=true` and
`max_duration_seconds=20.48`. For successful stock SGLang requests, the
assembler also enforces the conservative report boundary: fewer than 255 codec
frames, fewer than 489,600 PCM samples, and less than 20.40 seconds. The SGLang
EOS fields must remain unknown. Successful Native requests must declare
natural EOS, no length limit, and `finish_reason="stop"`.

## Assemble once

Run from the repository root:

```bash
python3 reports/tools/assemble_production_manifest.py \
  --config /absolute/path/to/production-metadata.json \
  --workload /absolute/path/to/evidence-root/workload/workload.jsonl \
  --runs-root /absolute/path/to/evidence-root/runs \
  --output /absolute/path/to/evidence-root/manifest.json
```

The output parent must already exist. `manifest.json` must not exist. The tool
opens it with create-new semantics, writes deterministic UTF-8 JSON, flushes it
to disk, and refuses every overwrite. The generated top-level fields are only
`schema_version`, `evidence_kind`, `evidence_files`, and `run_resources`; all
claim-bearing static sections are copied unchanged from the configuration.

After assembly, run the complete semantic report validator:

```bash
python3 reports/generate_report.py \
  /absolute/path/to/evidence-root/manifest.json \
  --validate-only
```

The assembler validates the final object against the repository's schema before
writing, while `generate_report.py` additionally performs full request/packet
arithmetic, workload parity, statistical, and report-specific validation.

## Integrity and path policy

For every run, the assembler requires and recalculates the complete
`SHA256SUMS` inventory. The inventory must cover every regular run file exactly
once except `SHA256SUMS` itself. Missing entries, extra entries, malformed
paths, changed bytes, and duplicate paths fail assembly.

Every non-empty artifact whose extension is representable by schema v1.1
(`json`, `jsonl`, `csv`, `txt`, `log`, `stdout`, or `stderr`) receives an
`evidence_files` descriptor with its observed SHA-256 and byte count. The three
canonical client artifacts receive their dedicated roles. Schema v1.1 cannot
represent extensionless binaries, shell scripts, `SHA256SUMS`, or zero-byte
streams as evidence descriptors; those files are still mandatory where the
controller always produces them and remain fully verified through the run's
complete checksum inventory.

Absolute manifest paths, traversal, backslashes, non-normalized paths, symlink
files, symlink directories, files outside the evidence root, mismatched
`evidence_prefix` values, and output symlinks are rejected. Raw telemetry paths
in each `run-resource.json` must equal the reducer's canonical eight-file set
and must resolve to digest-bound raw descriptors for the same run.

Additional production gates include:

- one clean immutable benchmark-tooling commit and one client digest across all
  twelve runs;
- one resolved container image ID and one image reference per engine;
- resolved image IDs equal to their configured implementation image digests;
- identical captured workload bytes in all twelve runs;
- configured, invoked, and reduced sampling intervals agree;
- no competing CUDA process and no unclean benchmark-tooling worktree;
- all eleven `resource-audit.json` source-file SHA-256/byte records agree with
  the recalculated run inventory, and its sampling, power, energy, RSS, and GPU
  memory reductions agree with `run-resource.json`;
- run, resource audit, summary, and invocation identities agree.

## Tests

The tests use only temporary directories and require no Docker or GPU:

```bash
python3 -m unittest discover \
  -s reports/tools/tests \
  -p 'test_*.py' \
  -v

ruff check --no-cache reports/tools
ruff format --check --no-cache reports/tools
```

The fixtures exercise a complete twelve-run, 2,400-request matrix and explicit
rejection of missing and extra runs, checksum changes, symlinks, identity
conflicts, a non-20.48-second workload, the stock SGLang 255-frame boundary,
image-digest drift, stale internally referenced audit digests, and output
overwrites.
