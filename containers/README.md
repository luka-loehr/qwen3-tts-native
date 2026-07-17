# Containers

## Current image inventory

The DGX Spark currently has one project-specific development image:

`codex/qwen3-tts-rust-builder:1.97.0`

It is a Rust builder, not a TTS runtime image. The checked
`Dockerfile.builder` reproduces it from the pinned ARM64 Rust 1.97.0 base and
adds only `rustfmt` and `clippy`.

```bash
docker build \
  --file containers/Dockerfile.builder \
  --tag qwen3-tts-native/builder:rust-1.97.0 \
  .
```

CUDA 13.0.88 and SM 12.1 compilation currently use the pinned upstream image:

`nvcr.io/nvidia/tensorrt:25.11-py3`

The upstream image includes tools that are useful during compilation. Python
inside that development image is not part of the native inference path.

## Runtime image boundary

There is intentionally no production runtime image yet. The current artifact
is a C-callable library stack rather than a network daemon, and the streaming
pipeline is still being optimized and qualified.

The eventual runtime image should contain only:

- `libqwen3_tts_runtime.so`;
- `libqwen3_tts_cuda.so`;
- `libqwen3_tts_codec_cuda.so`;
- the versioned public C header;
- one small native service or verification runner;
- required CUDA runtime libraries.

Model weights must be mounted read-only. They must not be copied into the image
or committed to Git. The image should run without Python or Node.js and should
publish immutable OCI labels for source revision, model contract, CUDA target,
and ABI version.
