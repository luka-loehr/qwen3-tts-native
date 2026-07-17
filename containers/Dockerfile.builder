# syntax=docker/dockerfile:1.7
FROM rust:1.97.0-bookworm@sha256:4334151ca823b4a52237f3a2d1b16973d6c3c0dd2a7e0ec66a9a8947713d4fa4

LABEL org.opencontainers.image.title="Qwen3-TTS Native Rust Builder"
LABEL org.opencontainers.image.description="Pinned Rust builder for the Qwen3-TTS native research runtime"
LABEL org.opencontainers.image.source="https://github.com/luka-loehr/qwen3-tts-native"

RUN rustup component add rustfmt
RUN rustup component add clippy

WORKDIR /workspace
CMD ["cargo", "--version"]
