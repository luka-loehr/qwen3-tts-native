# Immutable Model Contract

The native runtime must reject a model artifact before allocating GPU memory if any pinned configuration field or tensor metadata differs from the audited Qwen3-TTS-12Hz-1.7B-VoiceDesign snapshot.

## Validation scope

`validate-artifact` checks:

- the 1.7B VoiceDesign model type, token IDs, talker dimensions, predictor dimensions, RoPE sections, and exact language-ID map;
- the exact generation defaults used for both semantic and residual-codebook sampling;
- the speech-tokenizer sample rates, 1,920-sample frame expansion, decoder dimensions, 72-frame attention window, and upsampling ratios;
- every expected tensor name, dtype, shape, parameter count, and payload byte count in the 404-tensor BF16 VoiceDesign checkpoint;
- every expected tensor name, dtype, shape, parameter count, and payload byte count in the 496-tensor F32 speech-tokenizer checkpoint;
- the decoder-only tensor count and the projected BF16 decoder payload for production planning.

The current Hugging Face snapshot is a research cache whose files are symlinks. It validates only when `--allow-symlinks` is explicit, and the report then sets `production_material` to `false`. A production artifact must contain flattened regular files and pass without that flag.

## Language boundary

The pinned model defines explicit codec-language IDs for German, English, French, Italian, Chinese, Spanish, Japanese, Korean, Russian, and Portuguese. It does not define a Turkish language ID. Turkish remains an empirical intelligibility and voice-consistency gate; the runtime and documentation must not present it as guaranteed native support.

## Usage

```bash
cargo run --release --bin validate_artifact -- \
  /models/qwen3-tts-1.7b-voice-design
```

For the audited research cache only:

```bash
cargo run --release --bin validate_artifact -- \
  /path/to/huggingface/snapshot \
  --allow-symlinks \
  --output artifact-validation.json
```
