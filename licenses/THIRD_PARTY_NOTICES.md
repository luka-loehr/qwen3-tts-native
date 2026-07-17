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

The final image derives from NVIDIA's official CUDA 13.0.2 runtime image for
Ubuntu 24.04 ARM64. NVIDIA and operating-system license material inherited from
that base image must remain present. The runtime image must not copy TensorRT,
cuDNN, compiler, Python, or Node.js material from build stages.

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
