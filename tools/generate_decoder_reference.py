#!/usr/bin/env python3
"""Generate offline parity fixtures with the official Qwen decoder.

This tool is not part of the native runtime. It is executed only in the pinned
official-reference container and may be removed after refreshing fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from qwen_tts.core import Qwen3TTSTokenizerV2Model


def deterministic_codes(frame_count: int) -> np.ndarray:
    state = 0x6D2B79F5
    codes = np.zeros((1, 16, frame_count), dtype=np.int64)
    for frame in range(frame_count):
        for codebook in range(16):
            state = (state * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
            codes[0, codebook, frame] = (state >> 8) & 2047
    return codes


def save_tensor(directory: Path, name: str, tensor: torch.Tensor, manifest: dict) -> None:
    array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().astype("<f4")
    filename = f"{name}.f32le"
    payload = array.tobytes(order="C")
    (directory / filename).write_bytes(payload)
    manifest["checkpoints"][name] = {
        "file": filename,
        "dtype": "F32_LE",
        "shape": list(array.shape),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--transformer-only", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.frames <= 128:
        raise ValueError("frames must be between 1 and 128")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = Qwen3TTSTokenizerV2Model.from_pretrained(
        args.model,
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    ).to("cuda:0")
    model.eval()
    decoder = model.decoder
    codes_np = deterministic_codes(args.frames)
    codes = torch.from_numpy(codes_np).to(device="cuda:0", dtype=torch.long)

    manifest = {
        "schema_version": 1,
        "description": "Official Qwen3-TTS 12Hz decoder forward checkpoints",
        "reference_implementation": "QwenLM/Qwen3-TTS qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py",
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign speech_tokenizer",
        "device": torch.cuda.get_device_name(0),
        "torch_dtype": "float32",
        "float32_precision": "TF32 disabled for deterministic kernel parity",
        "attention": "eager causal sliding-window 72",
        "frame_count": args.frames,
        "codebook_count": 16,
        "samples_per_frame": 1920,
        "checkpoints": {},
    }

    codes_u16 = codes_np.transpose(0, 2, 1).astype("<u2").copy()
    codes_payload = codes_u16.tobytes(order="C")
    (args.output / "codes.u16le").write_bytes(codes_payload)
    manifest["codes"] = {
        "file": "codes.u16le",
        "dtype": "U16_LE",
        "shape": list(codes_u16.shape),
        "sha256": hashlib.sha256(codes_payload).hexdigest(),
        "bytes": len(codes_payload),
    }

    hidden = decoder.quantizer.decode(codes)
    save_tensor(args.output, "01-rvq", hidden, manifest)
    hidden = decoder.pre_conv(hidden).transpose(1, 2)
    save_tensor(args.output, "02-pre-conv", hidden, manifest)
    hidden = decoder.pre_transformer(inputs_embeds=hidden).last_hidden_state
    save_tensor(args.output, "03-transformer", hidden, manifest)
    if args.transformer_only:
        manifest["scope"] = "RVQ, causal pre-convolution, and transformer only"
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return
    hidden = hidden.permute(0, 2, 1)
    for stage, blocks in enumerate(decoder.upsample, start=1):
        for block in blocks:
            hidden = block(hidden)
        save_tensor(args.output, f"{3 + stage:02d}-latent-upsample-{stage}", hidden, manifest)

    wav = hidden
    wav = decoder.decoder[0](wav)
    save_tensor(args.output, "06-decoder-pre-conv", wav, manifest)
    for stage in range(1, 5):
        wav = decoder.decoder[stage](wav)
        save_tensor(args.output, f"{6 + stage:02d}-decoder-block-{stage}", wav, manifest)
    wav = decoder.decoder[5](wav)
    save_tensor(args.output, "11-final-snake", wav, manifest)
    wav = decoder.decoder[6](wav)
    save_tensor(args.output, "12-final-pre-clamp", wav, manifest)
    wav = wav.clamp(min=-1, max=1)
    save_tensor(args.output, "13-final-clamp", wav, manifest)

    pcm = torch.round(wav.squeeze(0).squeeze(0) * 32767.0).clamp(-32768, 32767)
    pcm_np = pcm.to(device="cpu", dtype=torch.int16).contiguous().numpy().astype("<i2")
    pcm_payload = pcm_np.tobytes(order="C")
    (args.output / "reference-pcm.s16le").write_bytes(pcm_payload)
    manifest["pcm"] = {
        "file": "reference-pcm.s16le",
        "dtype": "S16_LE",
        "shape": list(pcm_np.shape),
        "sha256": hashlib.sha256(pcm_payload).hexdigest(),
        "bytes": len(pcm_payload),
        "conversion": "round(clamp(waveform,-1,1)*32767), clamp to signed 16-bit",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
