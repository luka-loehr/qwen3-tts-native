# Contributing to Qwen3-TTS Native

Thank you for helping improve Qwen3-TTS Native. This project combines a native
model runtime, CUDA kernels, a public C ABI, an HTTP service, reproducible
benchmarks, and a hardened release image. Changes must preserve the boundaries
and evidence standards that make those layers auditable.

By participating, you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md). Security-sensitive findings must follow
the private process in [SECURITY.md](SECURITY.md), not a public issue.

## Project scope

The supported inference target is exactly
`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` at the pinned revision documented in
[`containers/README.md`](containers/README.md). The production surface accepts
text and a natural-language voice description and returns progressive PCM or a
buffered WAV.

The following are outside the current project scope:

- voice cloning, reference audio, speaker enrollment, or speaker databases;
- the Base, CustomVoice, and 0.6B checkpoints;
- the speech-tokenizer encoder;
- Python, Node.js, PyTorch, SGLang, vLLM, TensorRT, or another model framework
  in the inference runtime or final image;
- x86-64, a GPU architecture other than `sm_121`, or CPU-only inference;
- Ephraim backend or frontend integration;
- languages not exposed by the pinned VoiceDesign checkpoint.

Offline reference tooling may use a language appropriate to the task, but it
must remain outside the production runtime, be clearly labeled, and never
become a hidden inference dependency.

Discuss a proposed scope expansion before implementing it. A contribution
must not silently broaden the model, platform, language, or deployment claims.

## Documentation language and structure

All repository documentation must be written in clear, professional English.
This includes READMEs, Markdown files, architecture notes, release notes,
benchmark narratives, examples, public API documentation, and user-facing
error text. Do not submit machine-translated text without reviewing and
rewriting it yourself.

Use this structure for component documentation where applicable:

1. purpose and precise scope;
2. implemented behavior;
3. public contract and ownership rules;
4. build or usage commands with pinned prerequisites;
5. verified evidence, linked to its machine-readable record;
6. limitations and explicit non-claims;
7. links to adjacent components and primary upstream sources.

Keep commands executable, use repository-relative links, and distinguish
current behavior from planned work. Do not describe a planned release, image,
benchmark, feature, platform, or quality result as completed.

### Public contracts and release documentation

An HTTP contract change must update the implementation tests,
[`docs/openapi.yaml`](docs/openapi.yaml), [`docs/API.md`](docs/API.md), the
server README, and every affected root or quickstart example in the same
change. State defaults and invalid field combinations explicitly. Keep the
canonical VoiceDesign fields (`text`, `voice_description`, `language`,
`stream`, and `output_format`) distinct from the narrower compatibility
endpoint fields (`input`, `voice`, and `response_format`). Never describe a
textual voice description as a speaker name, reference sample, or clone.

Before an image is published and accepted, make deployment examples require a
caller-supplied immutable reference and reject a missing value or mutable tag.
After acceptance, direct readers to the exact reference in the GitHub release;
mutable tags may be shown only as secondary convenience aliases. Do not make
an image public or claim that it can be pulled merely because a local candidate
exists.

Release-facing performance text must identify whether evidence came from a
direct source build, a local candidate image, or a clean pull of an exact
registry digest. Link the raw machine-readable evidence and preserve its source
commit, hardware, model identity, workload, and limitations. A final release
claim requires the digest-specific gates; an earlier direct-runtime result may
remain documented only when it is clearly labeled as a baseline.

## Repository safety

Never commit or attach any of the following:

- model weights, derived multi-gigabyte model artifacts, or unapproved model
  files;
- access tokens, API keys, registry credentials, SSH material, passwords,
  cookies, signed URLs, private endpoints, or `.env` files;
- personal, student, customer, or production data;
- voice samples without documented rights and explicit repository approval;
- generated build trees, core dumps, profiling traces, or temporary audio;
- third-party code or binary material without compatible licensing and
  attribution.

If a secret is exposed, stop work, revoke it, and report the incident privately
according to [SECURITY.md](SECURITY.md). Removing it in a later commit is not
sufficient.

Model artifacts used for GPU qualification must remain outside Git and must be
identified by the exact revision, manifest hash, and weight hashes recorded by
the repository's artifact contract.

## Git workflow

1. Start from the latest `main` and create a focused branch.
2. Keep commits small, coherent, and reviewable.
3. Do not mix unrelated cleanup with functional or benchmark changes.
4. Rebase or merge only when it does not discard another contributor's work.
5. Never force-push a shared branch. Force-pushing `main`, a release branch, or
   a branch another contributor is using is prohibited.
6. Never use destructive history or worktree commands to remove changes you do
   not own.
7. Commit generated evidence only when its source, command, environment, and
   validation are documented.

Use descriptive imperative commit subjects, for example:

```text
feat(server): reject unsupported output combinations
fix(codec): preserve overlap state across final packets
bench(runtime): record 200-request natural-EOS qualification
docs(container): document immutable digest deployment
```

## Engineering requirements

Changes to native or server code must preserve these invariants unless a
reviewed design explicitly replaces them:

- one shared immutable model, with independently owned mutable request state;
- bounded concurrency, queues, PCM storage, request duration, and shutdown;
- progressive delivery without dropped, duplicated, or reordered samples;
- explicit cancellation, finish reason, error mapping, and resource
  retirement;
- no global inference lock around GPU execution;
- no unbounded host or device allocation derived from untrusted input;
- no prompt, voice description, or generated audio in normal logs or metrics;
- no Python, Node.js, or model-serving framework in the production inference
  path;
- exact model identity and the ten official languages plus `Auto` only;
- deterministic behavior for an explicitly supplied seed, within the stated
  platform and artifact contract.

Unsafe boundaries must be narrow, documented, checked before dereferencing,
and covered by success and failure tests. Rust code should remain warning-free
under the lints declared by each crate. CUDA changes must preserve explicit
ownership, stream, error, synchronization, and architecture contracts.

## Test tiers

Every pull request must state which tiers were run, the exact commands, and the
result. If a tier is not relevant or unavailable, say why; never imply that it
passed.

### Canonical repository verification

From the repository root, run the complete platform-neutral gate:

```bash
./tools/verify-repository.sh
```

The script requires the pinned Rust toolchain, Node.js 22.12 or newer, and
`npx`. Node.js is used only to run the pinned OpenAPI development linter; it is
not an inference or image dependency. The gate runs metadata, formatting,
tests, and Clippy for every shipped manifest:

```text
native/qwen3-tts-native/Cargo.toml
native/qwen3-tts-native-codec/Cargo.toml
native/qwen3-tts-runtime/Cargo.toml
native/qwen3-tts-server/Cargo.toml
native/qwen3-tts-bench/Cargo.toml
native/qwen3-tts-http-bench/Cargo.toml
```

For each manifest, the exact Cargo command family is:

```bash
cargo metadata --locked --no-deps --format-version 1 --manifest-path <manifest>
cargo fmt --all --manifest-path <manifest> -- --check
cargo test --all-targets --all-features --locked --manifest-path <manifest>
cargo clippy --all-targets --all-features --locked \
  --manifest-path <manifest> -- -D warnings
```

It also runs `bash -n` over every tracked shell script, validates
`docs/openapi.yaml` with `@redocly/cli@2.38.0` and the OpenAPI specification
ruleset, and finishes with `git diff --check`.

### Tier 0: documentation and static hygiene

Required for every change:

- check Markdown structure, repository-relative links, and English language;
- run `cargo fmt --check` for every affected Rust crate;
- inspect `git diff --check` and confirm no generated or secret material was
  added;
- verify that changed shell scripts pass syntax checks and ShellCheck when the
  pinned tool is available.

Documentation-only changes may stop at Tier 0 when they do not alter commands,
contracts, release metadata, benchmark interpretation, or generated files.

At minimum, finish a documentation-only change with:

```bash
git diff --check
git status --short
```

Inspect every changed repository-relative link and every command block against
the current tree. If the change modifies an API contract, executable command,
release process, or benchmark interpretation, it is not a prose-only change;
run the canonical repository verification and any affected higher tier.

### Tier 1: local component tests

Required for affected Rust components and CPU-testable behavior. The canonical
script above runs the commands below for all six manifests; to isolate one
crate, use its explicit manifest path:

```bash
cargo test --all-targets --locked \
  --manifest-path native/qwen3-tts-server/Cargo.toml

cargo clippy --all-targets --locked \
  --manifest-path native/qwen3-tts-server/Cargo.toml -- -D warnings
```

Run the equivalent commands for every affected crate. Add targeted contract,
failure, lifecycle, and boundary tests rather than relying only on a happy-path
smoke. A local compile does not qualify CUDA execution or model behavior.

### Tier 2: isolated DGX Spark GPU qualification

Required for talker, predictor, codec, scheduler, artifact, memory, or
performance changes:

Build both CUDA libraries from the repository root on the DGX Spark with the
same architecture contract used by the production image:

```bash
cmake -S native/qwen3-tts-native/native \
  -B build/verify-talker -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121-real
cmake --build build/verify-talker --parallel

cmake -S native/qwen3-tts-native-codec/native \
  -B build/verify-codec -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121-real
cmake --build build/verify-codec --parallel
```

Inspect both shared libraries with `cuobjdump --list-elf`,
`cuobjdump --list-ptx`, and `cuobjdump --dump-sass --gpu-architecture sm_121`.
Qualifying artifacts contain real `sm_121` SASS and no PTX fallback. Run the
component-specific real-model commands documented in the talker, codec,
runtime, server, and benchmark READMEs; preserve their machine-readable output.

- build from a clean, identified commit in an isolated worktree;
- use the pinned model revision and verify all artifact hashes;
- compile and inspect real `sm_121` SASS, with no unintended PTX fallback;
- record the CUDA, driver, hardware, library, and compiler identities;
- disclose every competing GPU process and reject a performance run when the
  protocol requires an otherwise idle GPU;
- run relevant parity, lifecycle, cancellation, concurrency, memory, and
  Compute Sanitizer gates;
- run real natural-EOS requests rather than presenting a fixed frame cap as a
  completed utterance;
- remove only temporary resources created by the qualification.

Do not run production GPU qualification on a developer laptop or substitute a
simulator, synthetic decoder, local model, or differently sized checkpoint.

### Tier 3: release-image qualification

Required before a registry digest or release tag is promoted. Complete every
item in [`containers/RELEASE_CHECKLIST.md`](containers/RELEASE_CHECKLIST.md)
against the exact pushed digest, including:

Run the Dockerfile static check with the same read-only model and generated
release-metadata contexts used by the candidate build:

```bash
docker buildx build --check \
  --platform linux/arm64 \
  --file containers/Dockerfile.runtime \
  --build-context model=/absolute/path/to/pinned-model-context \
  --build-context release-metadata=/absolute/path/to/release-metadata \
  .
```

Then run the complete `--provenance=mode=max --sbom=true --push` candidate
command from [`containers/README.md`](containers/README.md#candidate-build).
The static check does not replace an image build or any digest-specific gate.

- reproducible metadata and license policy;
- SBOM and provenance attestations;
- image inventory, architecture, dynamic-link, and non-root checks;
- vulnerability scan and signature verification;
- clean pull by digest;
- hardened read-only GPU execution;
- server lifecycle, streaming, WAV, cancellation, and shutdown tests;
- at least 200 natural-EOS requests and the complete language matrix;
- container-versus-direct performance and memory regression checks.

A successful source build or locally tagged image is not a release result.

### Release-document finalization

Release documentation has one source of truth for each claim:

- the `v0.1.0` GitHub release records the immutable
  `ghcr.io/luka-loehr/qwen3-tts-native@sha256:...` reference;
- `reports/output/` contains only a schema-validated, visually reviewed
  production report, never a fixture or partial matrix;
- `CHANGELOG.md` records the release date and links the accepted report and
  immutable evidence identity;
- deployment examples consume `QWEN3_TTS_IMAGE` from the GitHub release and
  reject mutable tags instead of embedding an unverified digest.

Do not copy candidate values into public prose while qualification is running.
Finalize all claim-bearing documents from the same accepted manifest and image
digest, run the documentation checks, and review the resulting diff before the
semantic tag is created.

## Benchmark integrity

Benchmark evidence is part of the product contract. Follow
[`benchmarks/README.md`](benchmarks/README.md) and preserve these rules:

- Commit the raw machine-readable report, not a manually retyped substitute.
- Never edit measured values to smooth, normalize, translate, or improve a
  result. Correct a harness and run it again under a new evidence filename.
- Record the exact source commit, model revision and hashes, native library or
  image digest, hardware, CUDA stack, workload, seed policy, warm-ups,
  repetitions, concurrency, limits, and competing workload state.
- Define TTFA at the observed audio boundary and define RTF as generation wall
  time divided by generated audio duration.
- Separate cold model load from warm request latency.
- Separate microbenchmarks, artifact loading, fixed-length scheduler tests,
  natural-EOS generation, HTTP transport, final-container performance, energy,
  and subjective listening evaluation.
- Report failed, cancelled, truncated, and non-EOS requests; never remove them
  from a distribution without documenting the exclusion.
- Preserve units and include enough raw data to recompute percentiles and
  aggregates.
- Treat WAV format, finite signal, and no clipping as transport/signal checks,
  not proof of intelligibility, pronunciation, naturalness, or instruction
  adherence.

A comparison with SGLang or another runtime is acceptable only when both sides
use the same hardware state, model and revision, precision, text and voice
descriptions, language, sampling policy, seeds, output duration/EOS rules,
concurrency, warm-up policy, TTFA boundary, and RTF formula. Report both raw
records and disclose any unavoidable difference. A service merely running at
the same time is coexistence evidence, not a benchmark comparison.

## Pull request checklist

Before requesting review, confirm that:

- the change stays within the stated model and platform scope;
- documentation and user-facing text are professional English;
- no secret, model file, personal data, temporary audio, or build artifact is
  present;
- public contracts and failure behavior are documented;
- relevant test tiers and exact commands are reported truthfully;
- benchmark claims link to immutable evidence and state their limitations;
- licensing and third-party attribution are complete;
- the branch was pushed normally, without force;
- `git diff --check` passes and the worktree contains only intended changes.

## Licensing

By contributing, you agree that your contribution is licensed under the
repository's [Apache License 2.0](LICENSE). You are responsible for ensuring
that you have the right to submit the work and any included third-party
material. Model and dependency terms remain separately attributed in
[`licenses/`](licenses/).
