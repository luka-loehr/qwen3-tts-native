# Qwen3-TTS 1.7B VoiceDesign — Runtime Model Card

## Identity

- Upstream model: Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
- Pinned upstream revision: 5ecdb67327fd37bb2e042aab12ff7391903235d3
- Runtime artifact: qwen3-tts-native-1.7b-voice-design
- Upstream license: Apache License 2.0
- Model type: text and natural-language voice description to 24 kHz speech

The authoritative upstream model page is:

https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/tree/5ecdb67327fd37bb2e042aab12ff7391903235d3

## Material included in the image

The image contains only the pinned 1.7B VoiceDesign talker/code predictor and
the decoder-only part of the Qwen3-TTS 12 Hz speech tokenizer.

It deliberately does not contain:

- a Qwen3-TTS Base checkpoint;
- a CustomVoice or voice-cloning checkpoint;
- cloned-speaker embeddings or reference recordings;
- the speech-tokenizer encoder;
- the original F32 speech-tokenizer checkpoint.

The decoder was converted offline to BF16 with round-to-nearest-even. No model
conversion is performed when the service starts.

## Verified artifact hashes

| Material | SHA-256 |
| --- | --- |
| Artifact manifest | 9bb96a8d24bbb2d8933245e27083b8e7290346b776306dcb8a8f3aed68594527 |
| VoiceDesign weights | 391e8db219f292c515297cdceeb43e4eae67cdde35fa57e79a6a8a532fca0522 |
| Decoder-only BF16 weights | 062caa0a31346422410e4c0d2494aec14be20553f8cb0b71a875329de99ce180 |

The embedded manifest additionally records every tensor name, dtype, shape,
byte range, component, and tensor-level SHA-256.

## Inputs and outputs

Inputs are UTF-8 text, a natural-language voice instruction, an official
language selection, and bounded generation settings. The model emits
progressive signed 16-bit mono PCM at 24 kHz. One codec frame corresponds to
1,920 samples, or 80 milliseconds of audio.

The upstream Qwen3-TTS family documents support for Chinese, English, Japanese,
Korean, German, French, Russian, Portuguese, Spanish, and Italian. Shipping a
language in the API does not by itself constitute a complete intelligibility or
speaker-consistency qualification.

## Intended use

This artifact is intended for native, GPU-accelerated VoiceDesign speech
generation on NVIDIA DGX Spark-class ARM64 systems. Users describe the desired
voice with text. The image is not intended to clone a real person's voice.

## Limitations and safety

- Generated speech can contain pronunciation, factual, prosodic, or language
  errors and must be reviewed when accuracy matters.
- A generated voice can accidentally resemble a real person. Applications
  should prohibit impersonation, fraud, and deceptive attribution.
- Voice instructions are not guaranteed to be followed perfectly.
- The current checked benchmark proves native transport, memory, and latency
  properties; it is not a universal perceptual-quality certification.
- Downstream deployments are responsible for consent, disclosure, content
  policy, and all applicable privacy and voice-likeness laws.

## Upstream attribution

Qwen3-TTS was developed by the Qwen team. The upstream project and technical
report are available at:

- https://github.com/QwenLM/Qwen3-TTS
- https://arxiv.org/abs/2601.15621

The complete Apache License 2.0 text is distributed beside this model card.

