# Third-Party Notices

## Qwen3-TTS model material

The image embeds the pinned Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign model revision
5ecdb67327fd37bb2e042aab12ff7391903235d3 and a decoder-only BF16 derivative of
its speech-tokenizer checkpoint.

Upstream project:

https://github.com/QwenLM/Qwen3-TTS

Upstream model:

https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/tree/5ecdb67327fd37bb2e042aab12ff7391903235d3

License: Apache-2.0. The full license and the runtime model card are included in
the model subdirectory.

## NVIDIA CUDA runtime

The final image derives from NVIDIA's official CUDA 13.0.3 `base` image for
Ubuntu 24.04 ARM64 at platform-manifest digest
`sha256:56d9d8183e2181a20be6b0d3801d1f056a0e75c17706df939ba207b126e1cb9c`.
It additionally copies only the runtime files and notices from
`libcublas-13-0=13.1.1.3-1` in the pinned official CUDA devel image. NVIDIA and
operating-system license material inherited from those inputs must remain
present. The runtime image must not copy TensorRT, cuDNN, NPP, cuSPARSE, NCCL,
compiler, Python, or Node.js material from build stages.

Official image:

https://hub.docker.com/r/nvidia/cuda

## Rust dependencies

The checked Cargo.lock files are the dependency authority for the native Rust
components. A complete machine-generated dependency notice is intentionally not
checked in as a handwritten approximation.

Every release build must provide both of these generated, non-empty files
through the release-metadata BuildKit context:

- RUST-THIRD-PARTY-LICENSES.html
- RUST-SBOM.cdx.json

The same context must also provide APPLICATION-LICENSE.txt as a byte-for-byte
copy of the repository's root-level Apache-2.0 `LICENSE` file.

The Dockerfile copies all three into the image and fails when any is absent or
empty. The release process must also run a deny-by-default license-policy check
over the locked Cargo graph and review all unknown, unlicensed, copyleft, or
non-OSI findings before publication.
