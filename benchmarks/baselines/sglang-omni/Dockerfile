FROM lmsysorg/sglang:v0.5.12.post1-cu130-runtime@sha256:8df56b542526f4fffd5372f7f65a583c7852e50442c1f43c9c3feddfd93944a4

ARG TARGETARCH
ARG SGLANG_OMNI_COMMIT=8e272bcb8832ef1a3865ec48255d36f4871ec885

LABEL org.opencontainers.image.title="Qwen3-TTS SGLang-Omni benchmark comparator" \
      org.opencontainers.image.description="Stock SGLang-Omni 0.1.0 comparator for Qwen3-TTS 1.7B VoiceDesign on DGX Spark" \
      org.opencontainers.image.source="https://github.com/sgl-project/sglang-omni" \
      org.opencontainers.image.revision="8e272bcb8832ef1a3865ec48255d36f4871ec885" \
      org.opencontainers.image.version="0.1.0-stock-spark" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.qwen3-tts-native.comparator.profile="stock-official" \
      io.qwen3-tts-native.comparator.runtime-patches="none" \
      io.qwen3-tts-native.comparator.packaging-patches="arm64-torchcodec-only"

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/cache/huggingface \
    PYTHONUNBUFFERED=1

RUN test "${TARGETARCH}" = "arm64"

# The Python `sox` package may invoke the system executable. These Ubuntu
# packages are build inputs only for the Python comparator, never for the native
# Rust/CUDA production image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      sox=14.4.2+git20190427-4build4 \
      libsox-fmt-all=14.4.2+git20190427-4build4 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git init sglang-omni \
    && cd sglang-omni \
    && git remote add origin https://github.com/sgl-project/sglang-omni.git \
    && git fetch --depth 1 origin refs/tags/0.1.0:refs/tags/0.1.0 \
    && git checkout --detach refs/tags/0.1.0 \
    && test "$(git rev-parse HEAD)" = "${SGLANG_OMNI_COMMIT}" \
    && test "$(git describe --tags --exact-match)" = "0.1.0"

COPY patches/0001-arm64-drop-torchcodec-packaging-dependency.patch /tmp/patches/
RUN cd /opt/sglang-omni \
    && git apply --check /tmp/patches/0001-arm64-drop-torchcodec-packaging-dependency.patch \
    && git apply /tmp/patches/0001-arm64-drop-torchcodec-packaging-dependency.patch \
    && git diff --check \
    && test "$(git diff --name-only)" = "pyproject.toml"

# SGLang-Omni 0.1.0's declared transitive dependency graph contains mutually
# incompatible protobuf constraints. The upstream Spark recipe uses uv, whose
# resolver installs that declared set and reports the resulting incompatibilities
# for audit. Pin and hash the exact installer used by the validated Spark build.
COPY requirements-build.txt /tmp/requirements-build.txt
RUN python3 -m pip install \
      --no-cache-dir \
      --no-deps \
      --force-reinstall \
      --require-hashes \
      --requirement /tmp/requirements-build.txt \
    && test "$(uv --version)" = "uv 0.11.16 (aarch64-unknown-linux-gnu)"

# Install the complete stock dependency declaration after applying the sole
# ARM64 packaging patch. Compute-critical versions are pinned by upstream and
# the OCI base. Resolved ancillary versions and `pip check` are captured by the
# provenance script; the immutable comparator image digest is the runtime pin.
WORKDIR /opt/sglang-omni
RUN UV_NO_CACHE=1 uv pip install --system --no-build-isolation -e .

COPY requirements-qwen.txt /tmp/requirements-qwen.txt
RUN python3 -m pip install \
      --no-cache-dir \
      --no-build-isolation \
      --no-deps \
      --require-hashes \
      --requirement /tmp/requirements-qwen.txt

COPY qwen3_tts_1_7b_voicedesign.yaml /opt/baseline/qwen3_tts_1_7b_voicedesign.yaml

RUN python3 -c 'import platform; from importlib.metadata import version; from sglang_omni.models.qwen3_tts.compat import apply_qwen_tts_transformers_compatibility_patches; apply_qwen_tts_transformers_compatibility_patches(); import librosa, qwen_tts, sglang_omni, torch, transformers; assert platform.machine() in {"aarch64", "arm64"}; assert version("uv") == "0.11.16"; assert version("sglang") == "0.5.12.post1"; assert version("sglang-omni") == "0.1.0"; assert version("qwen-tts") == "0.1.1"; assert version("transformers") == "5.6.0"; assert version("torch").split("+")[0] == "2.11.0"'

EXPOSE 8000
CMD ["sgl-omni", "--help"]
