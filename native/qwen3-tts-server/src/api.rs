use std::fmt;

use qwen3_tts_runtime::{GenerationConfig, Language, RequestInput};
use serde::{Deserialize, Serialize};

use crate::config::{ServerConfig, duration_to_frames};
use crate::engine::EngineSynthesisRequest;

pub const MODEL_ID: &str = "qwen3-tts-1.7b-voice-design";
pub const MAX_SAFE_JSON_INTEGER: u64 = (1_u64 << 53) - 1;

pub const ACCEPTED_LANGUAGES: [&str; 11] = [
    "auto",
    "chinese",
    "english",
    "japanese",
    "korean",
    "german",
    "french",
    "russian",
    "portuguese",
    "spanish",
    "italian",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    PcmS16le,
    Wav,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputMode {
    StreamPcm,
    BufferedWav,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingStrategy {
    #[default]
    Sample,
    Greedy,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictorSamplingOptions {
    pub strategy: Option<SamplingStrategy>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingOptions {
    #[serde(default)]
    pub strategy: SamplingStrategy,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub repetition_penalty: Option<f32>,
    pub predictor: Option<PredictorSamplingOptions>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechRequestBody {
    pub text: String,
    pub voice_description: String,
    #[serde(default = "default_language")]
    pub language: String,
    pub seed: Option<u64>,
    pub max_duration_seconds: Option<f64>,
    pub sampling: Option<SamplingOptions>,
    pub stream: Option<bool>,
    pub output_format: Option<OutputFormat>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiSpeechRequestBody {
    pub model: String,
    pub input: String,
    pub voice: String,
    #[serde(default = "default_openai_response_format")]
    pub response_format: OpenAiResponseFormat,
    #[serde(default = "default_speed")]
    pub speed: f32,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiResponseFormat {
    #[default]
    Wav,
    Pcm,
}

#[derive(Clone, Debug)]
pub struct PreparedSpeech {
    pub engine_request: EngineSynthesisRequest,
    pub output: OutputMode,
    pub seed: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationError {
    pub code: &'static str,
    pub detail: String,
}

impl ValidationError {
    fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl std::error::Error for ValidationError {}

impl SpeechRequestBody {
    /// Validates and maps the HTTP body to the public native runtime contract.
    ///
    /// # Errors
    ///
    /// Returns a typed validation error for invalid limits, language, output,
    /// duration, seed, or sampling settings.
    pub fn prepare(self, config: &ServerConfig) -> Result<PreparedSpeech, ValidationError> {
        validate_text(&self.text, config.max_text_bytes)?;
        validate_voice_description(&self.voice_description, config.max_voice_description_bytes)?;
        let language = parse_language(&self.language)?;
        let duration = self
            .max_duration_seconds
            .unwrap_or(config.default_duration_seconds);
        if !duration.is_finite() || duration < 0.08 || duration > config.max_duration_seconds {
            return Err(ValidationError::new(
                "invalid_max_duration",
                format!(
                    "max_duration_seconds must be in 0.08..={}",
                    config.max_duration_seconds
                ),
            ));
        }
        let max_codec_frames = duration_to_frames(duration);
        let seed = self.seed.unwrap_or_else(random_json_safe_seed);
        if seed > MAX_SAFE_JSON_INTEGER {
            return Err(ValidationError::new(
                "invalid_seed",
                format!("seed must be in 0..={MAX_SAFE_JSON_INTEGER}"),
            ));
        }
        let output = normalize_output(self.stream, self.output_format)?;
        let generation =
            prepare_sampling(self.sampling.unwrap_or_default(), max_codec_frames, seed)?;
        Ok(PreparedSpeech {
            engine_request: EngineSynthesisRequest {
                input: RequestInput {
                    text: self.text,
                    instruct: self.voice_description,
                    language,
                },
                generation,
            },
            output,
            seed,
        })
    }
}

impl OpenAiSpeechRequestBody {
    /// Maps the narrow compatibility body to the native request shape.
    ///
    /// # Errors
    ///
    /// Returns a validation error for a non-VoiceDesign model, non-unit speed,
    /// streaming, or non-WAV output.
    pub fn into_native(self) -> Result<SpeechRequestBody, ValidationError> {
        if self.model != MODEL_ID {
            return Err(ValidationError::new(
                "unsupported_model",
                format!("model must be {MODEL_ID:?}"),
            ));
        }
        if !self.speed.is_finite() || (self.speed - 1.0).abs() > f32::EPSILON {
            return Err(ValidationError::new(
                "unsupported_speed",
                "the native VoiceDesign server supports only speed=1.0",
            ));
        }
        let (stream, output_format) = match (self.stream, self.response_format) {
            (false, OpenAiResponseFormat::Wav) => (false, OutputFormat::Wav),
            _ => {
                return Err(ValidationError::new(
                    "unsupported_response_format",
                    "the conservative compatibility alias supports only buffered wav; use /v1/voice-design/speech for streaming PCM",
                ));
            }
        };
        Ok(SpeechRequestBody {
            text: self.input,
            voice_description: self.voice,
            language: default_language(),
            seed: None,
            max_duration_seconds: None,
            sampling: None,
            stream: Some(stream),
            output_format: Some(output_format),
        })
    }
}

fn validate_text(text: &str, limit: usize) -> Result<(), ValidationError> {
    if text.trim().is_empty() {
        return Err(ValidationError::new(
            "invalid_text",
            "text must not be empty or whitespace-only",
        ));
    }
    if text.len() > limit {
        return Err(ValidationError::new(
            "invalid_text",
            format!("text is {} UTF-8 bytes; maximum is {limit}", text.len()),
        ));
    }
    if text.contains('\0') {
        return Err(ValidationError::new(
            "invalid_text",
            "text must not contain NUL characters",
        ));
    }
    Ok(())
}

fn validate_voice_description(description: &str, limit: usize) -> Result<(), ValidationError> {
    if description.trim().is_empty() {
        return Err(ValidationError::new(
            "invalid_voice_description",
            "voice_description is required for the VoiceDesign model",
        ));
    }
    if description.len() > limit {
        return Err(ValidationError::new(
            "invalid_voice_description",
            format!(
                "voice_description is {} UTF-8 bytes; maximum is {limit}",
                description.len()
            ),
        ));
    }
    if description.contains('\0') {
        return Err(ValidationError::new(
            "invalid_voice_description",
            "voice_description must not contain NUL characters",
        ));
    }
    Ok(())
}

fn parse_language(value: &str) -> Result<Language, ValidationError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(Language::Auto),
        "chinese" => Ok(Language::Chinese),
        "english" => Ok(Language::English),
        "japanese" => Ok(Language::Japanese),
        "korean" => Ok(Language::Korean),
        "german" => Ok(Language::German),
        "french" => Ok(Language::French),
        "russian" => Ok(Language::Russian),
        "portuguese" => Ok(Language::Portuguese),
        "spanish" => Ok(Language::Spanish),
        "italian" => Ok(Language::Italian),
        unsupported => Err(ValidationError::new(
            "unsupported_language",
            format!(
                "language {unsupported:?} is unsupported; accepted values are {}",
                ACCEPTED_LANGUAGES.join(", ")
            ),
        )),
    }
}

fn normalize_output(
    stream: Option<bool>,
    format: Option<OutputFormat>,
) -> Result<OutputMode, ValidationError> {
    let normalized = match (stream, format) {
        (None | Some(true), None | Some(OutputFormat::PcmS16le)) => OutputMode::StreamPcm,
        (None | Some(false), Some(OutputFormat::Wav)) | (Some(false), None) => {
            OutputMode::BufferedWav
        }
        (Some(true), Some(OutputFormat::Wav)) => {
            return Err(ValidationError::new(
                "unsupported_response_format",
                "stream=true requires output_format=pcm_s16le",
            ));
        }
        (Some(false), Some(OutputFormat::PcmS16le)) => {
            return Err(ValidationError::new(
                "unsupported_response_format",
                "stream=false requires output_format=wav",
            ));
        }
    };
    Ok(normalized)
}

fn prepare_sampling(
    sampling: SamplingOptions,
    max_codec_frames: u32,
    seed: u64,
) -> Result<GenerationConfig, ValidationError> {
    let defaults = GenerationConfig::default();
    validate_strategy_fields(
        sampling.strategy,
        sampling.temperature,
        sampling.top_p,
        sampling.top_k,
        "sampling",
    )?;
    let temperature = sampling.temperature.unwrap_or(defaults.temperature);
    let top_p = sampling.top_p.unwrap_or(defaults.top_p);
    let top_k = sampling.top_k.unwrap_or(defaults.top_k);
    let repetition_penalty = sampling
        .repetition_penalty
        .unwrap_or(defaults.repetition_penalty);
    validate_float(temperature, 0.01, 2.0, "sampling.temperature")?;
    validate_float(top_p, f32::MIN_POSITIVE, 1.0, "sampling.top_p")?;
    validate_float(repetition_penalty, 0.1, 2.0, "sampling.repetition_penalty")?;
    if top_k > 3_072 {
        return Err(invalid_sampling("sampling.top_k must be in 0..=3072"));
    }

    let predictor = sampling.predictor.unwrap_or_default();
    let predictor_strategy = predictor.strategy.unwrap_or(sampling.strategy);
    validate_strategy_fields(
        predictor_strategy,
        predictor.temperature,
        predictor.top_p,
        predictor.top_k,
        "sampling.predictor",
    )?;
    let predictor_temperature = predictor
        .temperature
        .unwrap_or(defaults.predictor_temperature);
    let predictor_top_p = predictor.top_p.unwrap_or(defaults.predictor_top_p);
    let predictor_top_k = predictor.top_k.unwrap_or(defaults.predictor_top_k);
    validate_float(
        predictor_temperature,
        0.01,
        2.0,
        "sampling.predictor.temperature",
    )?;
    validate_float(
        predictor_top_p,
        f32::MIN_POSITIVE,
        1.0,
        "sampling.predictor.top_p",
    )?;
    if predictor_top_k > 2_048 {
        return Err(invalid_sampling(
            "sampling.predictor.top_k must be in 0..=2048",
        ));
    }

    Ok(GenerationConfig {
        max_codec_frames,
        seed,
        temperature,
        top_p,
        repetition_penalty,
        top_k,
        do_sample: u32::from(sampling.strategy == SamplingStrategy::Sample),
        predictor_temperature,
        predictor_top_p,
        predictor_top_k,
        predictor_do_sample: u32::from(predictor_strategy == SamplingStrategy::Sample),
        ..defaults
    })
}

fn validate_strategy_fields(
    strategy: SamplingStrategy,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    path: &str,
) -> Result<(), ValidationError> {
    if strategy == SamplingStrategy::Greedy
        && (temperature.is_some() || top_p.is_some() || top_k.is_some())
    {
        return Err(invalid_sampling(format!(
            "{path} temperature/top_p/top_k must be omitted for greedy strategy"
        )));
    }
    Ok(())
}

fn validate_float(
    value: f32,
    minimum: f32,
    maximum: f32,
    path: &str,
) -> Result<(), ValidationError> {
    if !value.is_finite() || value < minimum || value > maximum {
        return Err(invalid_sampling(format!(
            "{path} must be finite and in {minimum}..={maximum}"
        )));
    }
    Ok(())
}

fn invalid_sampling(detail: impl Into<String>) -> ValidationError {
    ValidationError::new("invalid_sampling", detail)
}

fn default_language() -> String {
    "auto".to_owned()
}

fn default_openai_response_format() -> OpenAiResponseFormat {
    OpenAiResponseFormat::Wav
}

fn default_speed() -> f32 {
    1.0
}

fn random_json_safe_seed() -> u64 {
    rand::random::<u64>() & MAX_SAFE_JSON_INTEGER
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_body() -> SpeechRequestBody {
        SpeechRequestBody {
            text: "Guten Morgen".to_owned(),
            voice_description: "A calm adult male voice".to_owned(),
            language: "German".to_owned(),
            seed: Some(42),
            max_duration_seconds: Some(10.0),
            sampling: None,
            stream: None,
            output_format: None,
        }
    }

    #[test]
    fn default_output_is_progressive_pcm() {
        let prepared = valid_body().prepare(&ServerConfig::default()).unwrap();
        assert_eq!(prepared.output, OutputMode::StreamPcm);
        assert_eq!(prepared.engine_request.generation.max_codec_frames, 125);
        assert_eq!(prepared.engine_request.input.language, Language::German);
    }

    #[test]
    fn wav_infers_buffered_mode_and_conflicting_modes_are_rejected() {
        let mut body = valid_body();
        body.output_format = Some(OutputFormat::Wav);
        assert_eq!(
            body.prepare(&ServerConfig::default()).unwrap().output,
            OutputMode::BufferedWav
        );

        let mut conflict = valid_body();
        conflict.stream = Some(true);
        conflict.output_format = Some(OutputFormat::Wav);
        assert_eq!(
            conflict.prepare(&ServerConfig::default()).unwrap_err().code,
            "unsupported_response_format"
        );
    }

    #[test]
    fn turkish_is_a_semantic_unsupported_language_error() {
        let mut body = valid_body();
        body.language = "Turkish".to_owned();
        assert_eq!(
            body.prepare(&ServerConfig::default()).unwrap_err().code,
            "unsupported_language"
        );
    }

    #[test]
    fn voice_description_is_required() {
        let mut body = valid_body();
        body.voice_description = "  ".to_owned();
        assert_eq!(
            body.prepare(&ServerConfig::default()).unwrap_err().code,
            "invalid_voice_description"
        );
    }

    #[test]
    fn greedy_rejects_meaningless_sampling_parameters() {
        let mut body = valid_body();
        body.sampling = Some(SamplingOptions {
            strategy: SamplingStrategy::Greedy,
            temperature: Some(0.9),
            ..SamplingOptions::default()
        });
        assert_eq!(
            body.prepare(&ServerConfig::default()).unwrap_err().code,
            "invalid_sampling"
        );
    }
}
