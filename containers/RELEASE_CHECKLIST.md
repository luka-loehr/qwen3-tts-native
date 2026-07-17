# Runtime image release checklist

Every item applies to the exact candidate digest. A mutable tag must never be
published as a shortcut around an incomplete gate.

## Source and licensing

- [ ] The build uses a reviewed, clean Git commit on the release branch.
- [ ] The Docker build context contains no backend, frontend, credential,
      secret, user data, or unrelated repository material.
- [ ] The root `LICENSE` is the approved Apache License 2.0 text.
- [ ] Every shipped Cargo package declares `Apache-2.0`.
- [ ] The OCI application license label is exactly `Apache-2.0`.
- [ ] Qwen model license, model card, source record, and attribution are
      present and reviewed.
- [ ] NVIDIA and Ubuntu license material inherited from the pinned base, plus
      copied cuBLAS package notices, remains present.

## Reproducible dependency evidence

- [ ] The release `Cargo.lock` is unchanged after dependency review.
- [ ] `tools/release-metadata/test-reproducibility.sh` passes twice byte for
      byte at the release commit timestamp.
- [ ] The generated application license is byte-identical to root `LICENSE`.
- [ ] The pinned cargo-about deny-by-default scan has no unknown or unapproved
      license.
- [ ] The CycloneDX 1.5 SBOM passes structural and reference validation.
- [ ] BuildKit SBOM generation is enabled.
- [ ] Maximum provenance is enabled and contains no secret build argument.

## Pinned toolchain and native artifacts

- [ ] Dockerfile frontend, Rust builder, CUDA devel, and CUDA base resolve to
      the source-controlled digests.
- [ ] `qwen3-tts-server` is the reviewed production server.
- [ ] `qwen3-tts-healthcheck` is a model-free loopback client.
- [ ] Runtime, talker, and codec shared libraries are stripped AArch64 ELF
      artifacts.
- [ ] `cuobjdump --list-elf` reports only `sm_121` cubins for both CUDA
      libraries.
- [ ] `cuobjdump --list-ptx` reports no PTX fallback for either library.
- [ ] `cuobjdump --dump-sass --gpu-architecture sm_121` emits non-empty SASS.
- [ ] Every executable and shared library resolves all dynamic dependencies.

## Minimal final image

- [ ] The final image derives from the pinned CUDA 13.0.3 ARM64 `base` image.
- [ ] Only the pinned cuBLAS/cuBLASLt runtime payload is added from the CUDA
      devel image.
- [ ] Python, Node.js, PyTorch, SGLang, vLLM, TensorRT, cuDNN, NPP, cuSPARSE,
      NCCL, compilers, and development packages are absent.
- [ ] The image runs as UID/GID `10001`, never root.
- [ ] The root filesystem is read-only with only a bounded `/tmp` tmpfs.
- [ ] All Linux capabilities are dropped and no-new-privileges is enabled.

## Model identity

- [ ] Model ID is exactly `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`.
- [ ] Revision is exactly `5ecdb67327fd37bb2e042aab12ff7391903235d3`.
- [ ] Manifest SHA-256 is
      `9bb96a8d24bbb2d8933245e27083b8e7290346b776306dcb8a8f3aed68594527`.
- [ ] VoiceDesign SHA-256 is
      `391e8db219f292c515297cdceeb43e4eae67cdde35fa57e79a6a8a532fca0522`.
- [ ] Decoder SHA-256 is
      `062caa0a31346422410e4c0d2494aec14be20553f8cb0b71a875329de99ce180`.
- [ ] Full native artifact validation passes during the build.
- [ ] The final image contains exactly two Safetensors files, each once.
- [ ] No Base, CustomVoice, voice-cloning, reference-audio, cloned-speaker, or
      speech-tokenizer encoder material is present.

## Registry and supply-chain integrity

- [ ] The candidate is pushed directly as exactly one `linux/arm64` platform.
- [ ] Registry credentials were never passed as build arguments or copied into
      a layer.
- [ ] The remote manifest contains BuildKit SBOM and provenance attestations.
- [ ] A current vulnerability scan reports no unresolved critical or high
      release blocker.
- [ ] The candidate digest is signed and the signature verifies.
- [ ] Registry compressed size is no more than 6.0 GB.
- [ ] Locally unpacked image size is no more than 10.0 GB.
- [ ] A clean Spark pull by digest succeeds without a cached model layer.

## Runtime behavior

- [ ] The shared engine loads exactly once.
- [ ] Readiness remains unavailable until the real talker, predictor,
      device-to-device handoff, and neural codec warm-up succeeds.
- [ ] Cold readiness on an otherwise idle Spark is no more than 20 seconds.
- [ ] Liveness remains independent of generation capacity.
- [ ] Healthchecks do not map model files, initialize CUDA, allocate GPU
      memory, or construct a second engine.
- [ ] Streaming delivers audio before generation completion.
- [ ] Buffered WAV is 24 kHz mono signed-16-bit PCM with exact RIFF lengths.
- [ ] Cancellation, capacity backpressure, natural EOS, max-length EOS,
      retirement, restart, and SIGTERM shutdown pass.
- [ ] The hardened read-only/cap-drop/no-new-privileges run succeeds.

## Performance and quality

- [ ] Cold-load peak host RSS is no more than 4.2 GiB.
- [ ] Post-load steady host RSS is measured and reviewed against 768 MiB.
- [ ] B6 CUDA device allocation is no more than 4.65 GB.
- [ ] Warm streaming TTFA p95 is below 200 ms.
- [ ] B1 request RTF remains below 1.0.
- [ ] Container TTFA and RTF regress no more than 3 percent against the same
      commit run directly on the Spark.
- [ ] At least 200 full natural-EOS requests complete without failure.
- [ ] All ten explicit languages plus `auto` complete with valid progressive
      audio and natural EOS.
- [ ] Automated SoX validation reports correct format, finite signal, and no
      clipping warning.
- [ ] Human intelligibility, pronunciation, instruction adherence, and speaker
      consistency are reviewed separately from transport/performance gates.

## Promotion

- [ ] The immutable version/model tag points to the accepted digest.
- [ ] The Git/model tag points to the same digest.
- [ ] The semantic-version alias is moved only after every gate passes.
- [ ] `latest` is moved last and points to that exact digest.
- [ ] Release notes and deployment documentation record the immutable digest.
