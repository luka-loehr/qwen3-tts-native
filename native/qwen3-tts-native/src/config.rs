use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

pub const VOICE_DESIGN_MODEL_TYPE: &str = "voice_design";

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ModelConfig {
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,
    pub tts_model_type: String,
    pub tokenizer_type: String,
    pub talker_config: TalkerConfig,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct TalkerConfig {
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub code_predictor_config: PredictorConfig,
    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_language_id: BTreeMap<String, u32>,
    pub codec_nothink_id: u32,
    pub codec_pad_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,
    pub codec_think_id: u32,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_code_groups: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub position_id_per_seconds: usize,
    pub rms_norm_eps: f32,
    pub rope_scaling: RopeScaling,
    pub rope_theta: f32,
    pub text_hidden_size: usize,
    pub text_vocab_size: usize,
    pub vocab_size: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct RopeScaling {
    pub interleaved: bool,
    pub mrope_section: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct PredictorConfig {
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_code_groups: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: usize,
}

impl ModelConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read model config {}", path.display()))?;
        let config: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid model config {}", path.display()))?;
        config.validate_voice_design_1_7b()?;
        Ok(config)
    }

    pub fn validate_voice_design_1_7b(&self) -> Result<()> {
        ensure!(
            self.tts_model_type == VOICE_DESIGN_MODEL_TYPE,
            "expected VoiceDesign model, found {:?}",
            self.tts_model_type
        );
        ensure!(
            self.tokenizer_type == "qwen3_tts_tokenizer_12hz",
            "unsupported speech tokenizer {:?}",
            self.tokenizer_type
        );

        let talker = &self.talker_config;
        ensure!(
            !talker.attention_bias,
            "talker attention bias is unsupported"
        );
        ensure!(
            talker.attention_dropout == 0.0,
            "talker dropout must be zero"
        );
        ensure!(
            talker.hidden_size == 2_048,
            "talker hidden size must be 2048"
        );
        ensure!(
            talker.intermediate_size == 6_144,
            "talker intermediate size must be 6144"
        );
        ensure!(talker.num_hidden_layers == 28, "talker must have 28 layers");
        ensure!(
            talker.num_attention_heads == 16 && talker.num_key_value_heads == 8,
            "talker must use 16 query and 8 KV heads"
        );
        ensure!(talker.head_dim == 128, "talker head dimension must be 128");
        ensure!(talker.num_code_groups == 16, "talker must use 16 codebooks");
        ensure!(
            talker.vocab_size == 3_072,
            "talker codec vocabulary must be 3072"
        );
        ensure!(
            talker.text_vocab_size == 151_936 && talker.text_hidden_size == 2_048,
            "unexpected talker text embedding dimensions"
        );
        ensure!(
            talker.hidden_act == "silu",
            "talker activation must be SiLU"
        );
        ensure!(
            talker.rope_scaling.interleaved && talker.rope_scaling.mrope_section == [24, 20, 20],
            "talker must use interleaved MRoPE sections [24, 20, 20]"
        );

        let predictor = &talker.code_predictor_config;
        ensure!(
            !predictor.attention_bias,
            "predictor attention bias is unsupported"
        );
        ensure!(
            predictor.attention_dropout == 0.0,
            "predictor dropout must be zero"
        );
        ensure!(
            predictor.hidden_size == 1_024,
            "predictor hidden size must be 1024"
        );
        ensure!(
            predictor.intermediate_size == 3_072,
            "predictor intermediate size must be 3072"
        );
        ensure!(
            predictor.num_hidden_layers == 5,
            "predictor must have 5 layers"
        );
        ensure!(
            predictor.num_attention_heads == 16 && predictor.num_key_value_heads == 8,
            "predictor must use 16 query and 8 KV heads"
        );
        ensure!(
            predictor.head_dim == 128,
            "predictor head dimension must be 128"
        );
        ensure!(
            predictor.num_code_groups == 16,
            "predictor must use 16 codebooks"
        );
        ensure!(
            predictor.vocab_size == 2_048,
            "predictor vocabulary must be 2048"
        );
        ensure!(
            predictor.hidden_act == "silu",
            "predictor activation must be SiLU"
        );

        if talker.codec_language_id.contains_key("turkish") {
            bail!("the pinned model unexpectedly claims an explicit Turkish language ID");
        }
        Ok(())
    }

    pub fn language_id(&self, language: &str) -> Result<Option<u32>> {
        if language.eq_ignore_ascii_case("auto") {
            return Ok(None);
        }
        self.talker_config
            .codec_language_id
            .get(&language.to_ascii_lowercase())
            .copied()
            .map(Some)
            .with_context(|| format!("unsupported explicit language {language:?}"))
    }
}
