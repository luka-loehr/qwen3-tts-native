# Production manifest assembler

`assemble_production_manifest.py` turns one canonical workload, two
implementation-artifact manifests, and exactly twelve completed qualifying-run
directories into a create-new `evidence.schema.json` version `1.2` production
manifest. It uses only the Python standard library and never repairs, normalizes,
or rewrites source evidence.

Production schema v1.2 separates three identities that must not be conflated:

- the local Docker reference, local Docker `.Id`, and Docker's inspected local
  unpacked/virtual `Size`;
- an optional OCI registry reference, registry manifest digest, and optional
  compressed transfer size; and
- the model artifact actually visible to each implementation at benchmark time.

The qualifying runs use already-running containers. They contain no startup
measurement and no canonical launch-command record. Consequently v1.2 reports
neither `startup_ms` nor `command_sha256`; manually supplied scalars are rejected.

## Evidence-root layout

```text
evidence-root/
  workload/workload.jsonl
  artifacts/
    native/model-artifact.json
    sglang/model-artifact.json
    native/registry-image.json       optional
    sglang/registry-image.json       optional
  runs/
    round-01/{native,sglang}/{B1,B3,B6}/
    round-02/{native,sglang}/{B1,B3,B6}/
  manifest.json                     created by the assembler
```

Directory names are not trusted as run identity. The assembler discovers runs
through `provenance/invocation.json` and `run-resource.json`, then requires the
embedded engine, profile, round, evidence prefix, local image, workload, client,
and telemetry identities to agree. The final set is exactly two engines, three
profiles, two rounds, and twelve unique triples. Failed staging directories,
unowned files, missing runs, duplicate identities, and extra runs fail assembly.

## Static configuration

The UTF-8 configuration contains exactly:

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

Use `reports/evidence.schema.json` as the field-level contract. In v1.2:

- `model` contains only the genuinely common repository, revision, and variant;
- every implementation has its own required `model_artifact`, which may differ
  in parameter count, precision, manifest, weight files, bytes, and hashes;
- `local_image.id` is a Docker `.Id`, not an OCI registry manifest digest;
- `local_image.unpacked_size_bytes` is Docker inspect `Size`, not registry
  compressed size;
- optional `registry_image` metadata requires separate digest-bound evidence;
- `methodology` describes measured clocks and metrics but has no startup claim.

`workload.corpus_sha256` must equal the byte-for-byte SHA-256 of the central
workload. `workload.ordered_seeds` must exactly equal the ordered `seed` values
in every workload row. There is no corpus-wide seed. Every row requires
`stream=true`, an integer seed, and `max_duration_seconds=20.48` exactly.
Schema v1.2 does not contain the old optional workload-level
`voice_description_sha256` or `generation` summaries; the canonical per-row
VoiceDesign and sampling values remain in the workload and client evidence.

Production remains fixed to mono 24 kHz PCM16 streaming, at least 24 warmups
per run, at least 200 measured requests per run, exactly B1/B3/B6, and exactly
two rounds. Successful Native requests require natural EOS, no length limit,
and `finish_reason="stop"`. Stock SGLang EOS stays unknown and successful audio
must remain below the exclusive 255-frame / 489,600-sample / 20.40-second gate.

## Required model-artifact evidence

Each implementation's `model_artifact.evidence.path` is a normalized JSON path
inside the evidence root. `model_artifact.evidence.sha256` is the exact file
digest. The minimal Stock file is:

```json
{
  "schema_version": "qwen3-tts-model-artifact/v1",
  "implementation_id": "sglang",
  "local_image_id": "sha256:<64 lowercase hex Docker .Id>",
  "repository": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
  "revision": "5ecdb67327fd37bb2e042aab12ff7391903235d3",
  "variant": "1.7B VoiceDesign",
  "parameter_count": 2087233793,
  "precision": ["bfloat16", "float32"],
  "manifest_sha256": null,
  "weight_files": [
    {
      "path": "model.safetensors",
      "sha256": "391e8db219f292c515297cdceeb43e4eae67cdde35fa57e79a6a8a532fca0522",
      "bytes": 3833402552,
      "parameter_count": 1916676352,
      "precision": "bfloat16"
    },
    {
      "path": "speech_tokenizer/model.safetensors",
      "sha256": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
      "bytes": 682293092,
      "parameter_count": 170557441,
      "precision": "float32"
    }
  ],
  "source": {
    "kind": "read_only_bind_mount",
    "container_path": "/models/hf-repository",
    "read_only": true,
    "host_path": "/srv/qwen3-tts/model-cache/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "snapshot_path": "snapshots/5ecdb67327fd37bb2e042aab12ff7391903235d3",
    "revision_ref_path": "refs/main"
  }
}
```

The matching implementation configuration repeats the claim-bearing artifact
fields verbatim and adds the evidence reference:

```json
{
  "model_artifact": {
    "repository": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "revision": "5ecdb67327fd37bb2e042aab12ff7391903235d3",
    "variant": "1.7B VoiceDesign",
    "parameter_count": 2087233793,
    "precision": ["bfloat16", "float32"],
    "manifest_sha256": null,
    "weight_files": [
      {
        "path": "model.safetensors",
        "sha256": "391e8db219f292c515297cdceeb43e4eae67cdde35fa57e79a6a8a532fca0522",
        "bytes": 3833402552,
        "parameter_count": 1916676352,
        "precision": "bfloat16"
      },
      {
        "path": "speech_tokenizer/model.safetensors",
        "sha256": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
        "bytes": 682293092,
        "parameter_count": 170557441,
        "precision": "float32"
      }
    ],
    "evidence": {
      "path": "artifacts/sglang/model-artifact.json",
      "sha256": "<SHA-256 of the exact JSON file>"
    }
  }
}
```

The assembler cross-validates exact metadata equality, parameter-count sums,
the sorted unique precision set, common repository/revision/variant, and the
artifact's `local_image_id` against the tested local Docker ID. Stock SGLang
must identify its observed read-only bind mount. Missing or changed Stock
artifact evidence fails explicitly; Native values are never substituted.

Native uses the same file format with `implementation_id="native"`, source kind
`container_image`, its BF16 decoder-only artifact, and its actual artifact
manifest digest. A null `manifest_sha256` explicitly means no single artifact
manifest was observed; it is not a placeholder digest.

Optional registry metadata uses `qwen3-tts-registry-image/v1` and records
`implementation_id`, `local_image_id`, `reference`, `manifest_digest`, and
optional `compressed_size_bytes`. The implementation's `registry_image.evidence`
must reference that exact digest-inventoried file. Registry claims are omitted
when this evidence does not exist.

## Assemble once

### Recommended one-shot production finalization

Copy `production-declarations.example.json` outside the repository, replace
every `PENDING_...` value with an exact observed or release-authorized claim,
and review the English prose. The declarations intentionally omit workload
digests and seeds, local Docker IDs and sizes, model-artifact claims, and
optional registry-image claims. `prepare_production_metadata.py` derives those
fields only from the canonical workload, the complete checksum inventories of
all twelve runs, and the digest-bound files under `artifacts/`.

After all twelve qualifying runs are present, finalize the complete evidence
boundary with one command from the repository root:

```bash
python3 reports/tools/finalize_production_evidence.py \
  --declarations /absolute/path/to/reviewed-production-declarations.json \
  --evidence-root /absolute/path/to/evidence-root \
  --paper-root /absolute/path/to/qwen3-tts-native/research/paper
```

The command performs the complete manifest assembler validation, the complete
production report validation and PDF build, and the paper evidence finalizer
before publishing any result. It then creates, without overwrite:

- `evidence-root/production-metadata.json`;
- `evidence-root/manifest.json`;
- `evidence-root/<benchmark-id>-report.pdf`; and
- the three managed files under `research/paper/data/`.

An unresolved declaration placeholder, missing or extra run, checksum drift,
inconsistent image identity, invalid model or registry artifact, report error,
or paper protocol mismatch rejects the operation. Hidden staging files are
removed on failure. If publishing fails after staging, the command removes only
the create-new files it owns and restores the original pending paper boundary.
It refuses an already finalized paper or any existing output, so reruns cannot
silently replace accepted evidence.

To inspect only the expanded metadata before finalization, use:

```bash
python3 reports/tools/prepare_production_metadata.py \
  --declarations /absolute/path/to/reviewed-production-declarations.json \
  --evidence-root /absolute/path/to/evidence-root \
  --output /absolute/path/to/production-metadata.json
```

The lower-level commands remain available for forensic or stepwise review:

```bash
python3 reports/tools/assemble_production_manifest.py \
  --config /absolute/path/to/production-metadata.json \
  --workload /absolute/path/to/evidence-root/workload/workload.jsonl \
  --runs-root /absolute/path/to/evidence-root/runs \
  --output /absolute/path/to/evidence-root/manifest.json

python3 reports/generate_report.py \
  /absolute/path/to/evidence-root/manifest.json \
  --validate-only
```

The output parent must exist and `manifest.json` must not. The assembler uses
create-new semantics, deterministic UTF-8 JSON, flushes the output, and refuses
overwrites. Claim-bearing static sections are copied unchanged only after all
cross-validation succeeds. Observed evidence descriptors and run resources are
derived from the files.

## Integrity policy

Every run's complete `SHA256SUMS` inventory is recalculated. Every representable
non-empty artifact receives an evidence descriptor; binaries, shell scripts,
the checksum file, and zero-byte streams remain covered by the complete run
inventory even where they do not receive a descriptor. Absolute paths,
traversal, non-normalized paths, backslashes, symlinks, files outside the
evidence root, and duplicate paths are rejected.

Additional gates include one clean immutable tooling commit, one client digest,
one local image ID/reference per engine, exact invocation and Docker-inspect ID
agreement, exact Docker-inspect local `Size`, identical workload bytes, aligned
sampling intervals, no competing CUDA process, complete resource-audit source
digests, and matching invocation/resource/summary identities.

## Tests

```bash
python3 -m unittest discover -s reports/tools/tests -p 'test_*.py' -v
ruff check --no-cache reports/tools
ruff format --check --no-cache reports/tools
```

The fixtures exercise the complete twelve-run matrix plus missing/extra runs,
checksum mutation, symlinks, identity conflicts, duration and SGLang boundary
violations, local-image ID/size drift, ordered-seed drift, missing and changed
Stock artifacts, invalid parameter sums, registry-evidence drift, stale audit
digests, and output overwrite protection.
