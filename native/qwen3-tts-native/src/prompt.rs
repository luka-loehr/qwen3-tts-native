use anyhow::{Result, ensure};

use crate::config::ModelConfig;
use crate::tokenizer::Qwen2Tokenizer;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextSource {
    Token(u32),
    TtsBos,
    TtsEos,
    TtsPad,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmbeddingStep {
    pub text: Option<TextSource>,
    pub codec: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VoiceDesignPrompt {
    pub instruction_ids: Vec<u32>,
    pub assistant_ids: Vec<u32>,
    pub prefill: Vec<EmbeddingStep>,
    pub trailing_text: Vec<TextSource>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextMode {
    Streaming,
    NonStreaming,
}

impl VoiceDesignPrompt {
    pub fn tokenize(
        tokenizer: &Qwen2Tokenizer,
        config: &ModelConfig,
        text: &str,
        instruction: &str,
        language: &str,
        mode: TextMode,
    ) -> Result<Self> {
        let instruction_text = format!("<|im_start|>user\n{instruction}<|im_end|>\n");
        let assistant_text =
            format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n");
        let instruction_ids = if instruction.is_empty() {
            Vec::new()
        } else {
            tokenizer.encode(&instruction_text)?
        };
        let assistant_ids = tokenizer.encode(&assistant_text)?;
        Self::from_token_ids(config, instruction_ids, assistant_ids, language, mode)
    }

    pub fn from_token_ids(
        config: &ModelConfig,
        instruction_ids: Vec<u32>,
        assistant_ids: Vec<u32>,
        language: &str,
        mode: TextMode,
    ) -> Result<Self> {
        ensure!(assistant_ids.len() >= 8, "assistant prompt is too short");
        let language_id = config.language_id(language)?;
        let talker = &config.talker_config;

        let mut codec_prefix = if let Some(language_id) = language_id {
            vec![
                talker.codec_think_id,
                talker.codec_think_bos_id,
                language_id,
                talker.codec_think_eos_id,
            ]
        } else {
            vec![
                talker.codec_nothink_id,
                talker.codec_think_bos_id,
                talker.codec_think_eos_id,
            ]
        };
        codec_prefix.push(talker.codec_pad_id);
        codec_prefix.push(talker.codec_bos_id);

        let mut prefill = instruction_ids
            .iter()
            .copied()
            .map(|token| EmbeddingStep {
                text: Some(TextSource::Token(token)),
                codec: None,
            })
            .collect::<Vec<_>>();

        prefill.extend(
            assistant_ids[..3]
                .iter()
                .copied()
                .map(|token| EmbeddingStep {
                    text: Some(TextSource::Token(token)),
                    codec: None,
                }),
        );

        for (index, codec) in codec_prefix[..codec_prefix.len() - 1]
            .iter()
            .copied()
            .enumerate()
        {
            let text = if index + 1 == codec_prefix.len() - 1 {
                TextSource::TtsBos
            } else {
                TextSource::TtsPad
            };
            prefill.push(EmbeddingStep {
                text: Some(text),
                codec: Some(codec),
            });
        }

        match mode {
            TextMode::Streaming => {
                prefill.push(EmbeddingStep {
                    text: Some(TextSource::Token(assistant_ids[3])),
                    codec: codec_prefix.last().copied(),
                });
                let mut trailing_text = assistant_ids[4..assistant_ids.len() - 5]
                    .iter()
                    .copied()
                    .map(TextSource::Token)
                    .collect::<Vec<_>>();
                trailing_text.push(TextSource::TtsEos);
                Ok(Self {
                    instruction_ids,
                    assistant_ids,
                    prefill,
                    trailing_text,
                })
            }
            TextMode::NonStreaming => {
                for token in &assistant_ids[3..assistant_ids.len() - 5] {
                    prefill.push(EmbeddingStep {
                        text: Some(TextSource::Token(*token)),
                        codec: Some(talker.codec_pad_id),
                    });
                }
                prefill.push(EmbeddingStep {
                    text: Some(TextSource::TtsEos),
                    codec: Some(talker.codec_pad_id),
                });
                prefill.push(EmbeddingStep {
                    text: Some(TextSource::TtsPad),
                    codec: Some(talker.codec_bos_id),
                });
                Ok(Self {
                    instruction_ids,
                    assistant_ids,
                    prefill,
                    trailing_text: vec![TextSource::TtsPad],
                })
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CodecFrame(pub [u32; 16]);

impl CodecFrame {
    pub fn from_predictor(codebook_zero: u32, residual: &[u32]) -> Result<Self> {
        ensure!(
            residual.len() == 15,
            "predictor must return exactly 15 residual codebooks"
        );
        let mut frame = [0_u32; 16];
        frame[0] = codebook_zero;
        frame[1..].copy_from_slice(residual);
        Ok(Self(frame))
    }

    pub fn is_eos(&self, config: &ModelConfig) -> bool {
        self.0[0] == config.talker_config.codec_eos_token_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictor_frame_requires_fifteen_residual_tokens() {
        assert!(CodecFrame::from_predictor(7, &[1; 14]).is_err());
        let frame = CodecFrame::from_predictor(7, &[1; 15]).unwrap();
        assert_eq!(frame.0[0], 7);
        assert_eq!(frame.0[15], 1);
    }
}
