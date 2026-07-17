#!/usr/bin/env bash
set -Eeuo pipefail

image="${IMAGE:-qwen3-tts-sglang-omni:0.1.0-stock-spark}"

docker run --rm -i \
  --gpus all \
  --entrypoint python3 \
  "${image}" - <<'PY'
import platform
from importlib.metadata import version

import torch

from sglang_omni.models.qwen3_tts.compat import (
    apply_qwen_tts_transformers_compatibility_patches,
)

apply_qwen_tts_transformers_compatibility_patches()

expected = {
    "sglang": "0.5.12.post1",
    "sglang-omni": "0.1.0",
    "transformers": "5.6.0",
    "qwen-tts": "0.1.1",
    "einops": "0.8.1",
    "onnxruntime": "1.26.0",
    "sox": "1.5.0",
}
actual = {name: version(name) for name in expected}

assert platform.machine() in {"aarch64", "arm64"}, platform.machine()
assert version("torch").split("+")[0] == "2.11.0", version("torch")
assert actual == expected, actual
assert torch.version.cuda is not None and torch.version.cuda.startswith("13.0")
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (12, 1)

from qwen_tts import Qwen3TTSTokenizer  # noqa: F401,E402
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig  # noqa: E402

print("architecture:", platform.machine())
print("torch:", version("torch"))
print("cuda_runtime:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("compute_capability:", torch.cuda.get_device_capability(0))
for name, value in sorted(actual.items()):
    print(f"{name}: {value}")
print("pipeline:", Qwen3TTSPipelineConfig.__name__)
PY
