# Toolchain and Host Change Audit

## Host

- Operating system: Ubuntu 24.04 on AArch64
- GPU: NVIDIA GB10, compute capability 12.1
- Driver: 580.159.03
- No Rust, CUDA compiler, TensorRT development package, Python package, or Node.js
  package was installed on the host.
- The production inference container was not stopped or modified for this milestone.

## Pinned container inputs

### TensorRT and CUDA

- Registry reference: `nvcr.io/nvidia/tensorrt:25.11-py3`
- ARM64 manifest digest:
  `sha256:6a507190873a87ee42205d5b2392491078aca971b1b25cda5f1053d72366bb89`
- CUDA compiler: 13.0.88
- TensorRT: 10.14.1
- CMake: 3.24.0
- Compiler: GCC 13.3
- Output architecture: real SM 12.1 SASS only

### Rust

- Upstream image: `rust:1.97.0-bookworm`
- ARM64 manifest digest:
  `sha256:4334151ca823b4a52237f3a2d1b16973d6c3c0dd2a7e0ec66a9a8947713d4fa4`
- Local builder image ID:
  `sha256:5d2de1857cd88619bd31ad5d85f2b042238bd0c646d7c041684b85460fde30a5`
- The local builder adds only the official Rust 1.97 `rustfmt` and `clippy`
  components.
- Cargo downloads are isolated in the ignored playground-local `.cargo-cache`.

### Qwen reference image

- Local image ID:
  `sha256:296794fef3e60314d5231963b03692d955d2886789f6d017c4de441ef7bc5447`
- Upstream source was read for behavioral parity; it was not executed to build the
  native runtime.

## Filesystem changes

During research, temporary files were isolated in a dedicated Spark-side
playground and separate Git worktrees. Those locations are not runtime inputs
and are removed as part of the release-host cleanup. Docker image layers remain
inside Docker-managed storage. The research phase did not modify the backend
checkout, frontend checkout, or production container configuration.
