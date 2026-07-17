use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::{BenchError, Result};

pub const SCHEMA_VERSION: &str = "qwen3-tts-http-bench/v1";
pub const SAMPLE_RATE_HZ: u64 = 24_000;
const SAMPLING_CONTRACT: &str = "qwen3-tts-native-sglang-common/v1";

/// Request/response contract implemented by the target endpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum BackendProfile {
    Native,
    SglangOmni,
}

impl BackendProfile {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::SglangOmni => "sglang-omni",
        }
    }
}

impl std::str::FromStr for BackendProfile {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "native" => Ok(Self::Native),
            "sglang-omni" | "sglang" => Ok(Self::SglangOmni),
            _ => Err("profile must be native or sglang-omni".to_owned()),
        }
    }
}

impl fmt::Display for BackendProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

/// Supported synchronized benchmark batch widths.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum Concurrency {
    B1,
    B3,
    B6,
}

impl Concurrency {
    #[must_use]
    pub const fn get(self) -> usize {
        match self {
            Self::B1 => 1,
            Self::B3 => 3,
            Self::B6 => 6,
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::B1 => "B1",
            Self::B3 => "B3",
            Self::B6 => "B6",
        }
    }
}

impl std::str::FromStr for Concurrency {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.to_ascii_uppercase().as_str() {
            "1" | "B1" => Ok(Self::B1),
            "3" | "B3" => Ok(Self::B3),
            "6" | "B6" => Ok(Self::B6),
            _ => Err("concurrency must be B1, B3, or B6".to_owned()),
        }
    }
}

impl fmt::Display for Concurrency {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

/// Fully specified benchmark invocation.
#[derive(Clone, Debug)]
pub struct BenchmarkConfig {
    pub endpoint: String,
    pub profile: BackendProfile,
    pub sglang_model: Option<String>,
    pub workload_path: PathBuf,
    pub output_dir: PathBuf,
    pub phase_events_path: Option<PathBuf>,
    pub requests: usize,
    pub warmups: usize,
    pub concurrency: Concurrency,
    pub timeout_seconds: u64,
    pub log_prompt_text: bool,
}

/// Endpoint-neutral sampling strategy shared by Native and SGLang-Omni.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingStrategy {
    #[default]
    Sample,
    Greedy,
}

/// Sampling controls for the main semantic/talker stage.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingOptions {
    #[serde(default)]
    pub strategy: SamplingStrategy,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predictor: Option<PredictorSamplingOptions>,
}

/// Sampling controls for Native's predictor and `SGLang`'s subtalker stage.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PredictorSamplingOptions {
    #[serde(default)]
    pub strategy: SamplingStrategy,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,
}

impl SamplingOptions {
    fn validate(&self, line: usize) -> Result<()> {
        validate_stage(
            line,
            "sampling",
            self.strategy,
            self.temperature,
            self.top_p,
            self.top_k,
        )?;
        if let Some(value) = self.repetition_penalty
            && (!value.is_finite() || value <= 0.0)
        {
            return Err(BenchError::Workload(format!(
                "line {line}: sampling.repetition_penalty must be finite and positive"
            )));
        }
        if let Some(predictor) = &self.predictor {
            validate_stage(
                line,
                "sampling.predictor",
                predictor.strategy,
                predictor.temperature,
                predictor.top_p,
                predictor.top_k,
            )?;
        }
        Ok(())
    }
}

fn validate_stage(
    line: usize,
    field: &str,
    strategy: SamplingStrategy,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
) -> Result<()> {
    if strategy == SamplingStrategy::Greedy
        && (temperature.is_some() || top_p.is_some() || top_k.is_some())
    {
        return Err(BenchError::Workload(format!(
            "line {line}: {field} temperature/top_p/top_k are invalid with strategy=greedy because the servers would ignore them"
        )));
    }
    if let Some(value) = temperature
        && (!value.is_finite() || value <= 0.0)
    {
        return Err(BenchError::Workload(format!(
            "line {line}: {field}.temperature must be finite and positive"
        )));
    }
    if let Some(value) = top_p
        && (!value.is_finite() || value <= 0.0 || value > 1.0)
    {
        return Err(BenchError::Workload(format!(
            "line {line}: {field}.top_p must be finite and in (0, 1]"
        )));
    }
    if top_k == Some(0) {
        return Err(BenchError::Workload(format!(
            "line {line}: {field}.top_k must be greater than zero"
        )));
    }
    Ok(())
}

/// Canonical, endpoint-neutral generation settings written into every report.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct NormalizedSampling {
    pub contract: &'static str,
    pub seed: Option<u64>,
    pub talker: Option<NormalizedTalkerSampling>,
    pub predictor: Option<NormalizedPredictorSampling>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct NormalizedTalkerSampling {
    pub strategy: SamplingStrategy,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u64>,
    pub repetition_penalty: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct NormalizedPredictorSampling {
    pub strategy: SamplingStrategy,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u64>,
}

#[derive(Clone, Debug)]
pub(crate) struct SamplingAudit {
    pub normalized: NormalizedSampling,
    pub normalized_sha256: String,
    pub parity_qualifying: bool,
    pub non_qualifying_reasons: Vec<String>,
}

/// One deterministic JSONL workload item.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkloadEntry {
    pub id: String,
    pub text: String,
    pub voice_description: String,
    #[serde(default = "default_language")]
    pub language: String,
    pub seed: Option<u64>,
    pub max_duration_seconds: Option<f64>,
    pub sampling: Option<SamplingOptions>,
    #[serde(default = "default_stream")]
    pub stream: bool,
}

fn default_language() -> String {
    "auto".to_owned()
}

const fn default_stream() -> bool {
    true
}

impl WorkloadEntry {
    pub(crate) fn validate(&self, line: usize) -> Result<()> {
        if self.id.trim().is_empty() {
            return Err(BenchError::Workload(format!(
                "line {line}: id must not be empty"
            )));
        }
        if self.id.len() > 128 || !self.id.bytes().all(is_safe_id_byte) {
            return Err(BenchError::Workload(format!(
                "line {line}: id must contain at most 128 ASCII letters, digits, '.', '_', or '-'"
            )));
        }
        if self.text.trim().is_empty() {
            return Err(BenchError::Workload(format!(
                "line {line}: text must not be empty"
            )));
        }
        if self.voice_description.trim().is_empty() {
            return Err(BenchError::Workload(format!(
                "line {line}: voice_description must not be empty"
            )));
        }
        if let Some(duration) = self.max_duration_seconds
            && (!duration.is_finite() || duration <= 0.0)
        {
            return Err(BenchError::Workload(format!(
                "line {line}: max_duration_seconds must be finite and positive"
            )));
        }
        if let Some(sampling) = &self.sampling {
            sampling.validate(line)?;
        }
        Ok(())
    }

    pub(crate) fn request_body(
        &self,
        profile: BackendProfile,
        sglang_model: Option<&str>,
    ) -> Result<Vec<u8>> {
        match profile {
            BackendProfile::Native => self.native_request_body(),
            BackendProfile::SglangOmni => self.sglang_request_body(sglang_model),
        }
    }

    fn native_request_body(&self) -> Result<Vec<u8>> {
        let mut object = Map::new();
        object.insert("text".to_owned(), Value::String(self.text.clone()));
        object.insert(
            "voice_description".to_owned(),
            Value::String(self.voice_description.clone()),
        );
        object.insert("language".to_owned(), Value::String(self.language.clone()));
        if let Some(seed) = self.seed {
            object.insert("seed".to_owned(), Value::from(seed));
        }
        if let Some(duration) = self.max_duration_seconds {
            object.insert("max_duration_seconds".to_owned(), Value::from(duration));
        }
        if let Some(sampling) = &self.sampling {
            object.insert("sampling".to_owned(), serde_json::to_value(sampling)?);
        }
        object.insert("stream".to_owned(), Value::Bool(self.stream));
        object.insert(
            "output_format".to_owned(),
            Value::String(if self.stream { "pcm_s16le" } else { "wav" }.to_owned()),
        );
        Ok(serde_json::to_vec(&Value::Object(object))?)
    }

    fn sglang_request_body(&self, model: Option<&str>) -> Result<Vec<u8>> {
        if !self.stream {
            return Err(BenchError::Workload(format!(
                "entry {:?}: sglang-omni comparison requires stream=true raw PCM",
                self.id
            )));
        }
        if self.max_duration_seconds.is_some() {
            return Err(BenchError::Workload(format!(
                "entry {:?}: max_duration_seconds has no exact SGLang-Omni API equivalent",
                self.id
            )));
        }
        let model = model.ok_or_else(|| {
            BenchError::Configuration(
                "--sglang-model is required for the sglang-omni profile".to_owned(),
            )
        })?;
        let mut object = Map::new();
        object.insert("model".to_owned(), Value::String(model.to_owned()));
        object.insert("voice".to_owned(), Value::String("default".to_owned()));
        object.insert("input".to_owned(), Value::String(self.text.clone()));
        object.insert(
            "instructions".to_owned(),
            Value::String(self.voice_description.clone()),
        );
        object.insert(
            "task_type".to_owned(),
            Value::String("VoiceDesign".to_owned()),
        );
        object.insert("language".to_owned(), Value::String(self.language.clone()));
        object.insert("stream".to_owned(), Value::Bool(true));
        object.insert(
            "response_format".to_owned(),
            Value::String("pcm".to_owned()),
        );
        if let Some(seed) = self.seed {
            object.insert("seed".to_owned(), Value::from(seed));
        }
        if let Some(sampling) = &self.sampling {
            object.insert(
                "do_sample".to_owned(),
                Value::Bool(sampling.strategy == SamplingStrategy::Sample),
            );
            insert_optional(&mut object, "temperature", sampling.temperature);
            insert_optional(&mut object, "top_p", sampling.top_p);
            insert_optional(&mut object, "top_k", sampling.top_k);
            insert_optional(
                &mut object,
                "repetition_penalty",
                sampling.repetition_penalty,
            );
            if let Some(predictor) = &sampling.predictor {
                object.insert(
                    "subtalker_dosample".to_owned(),
                    Value::Bool(predictor.strategy == SamplingStrategy::Sample),
                );
                insert_optional(&mut object, "subtalker_temperature", predictor.temperature);
                insert_optional(&mut object, "subtalker_top_p", predictor.top_p);
                insert_optional(&mut object, "subtalker_top_k", predictor.top_k);
            }
        }
        Ok(serde_json::to_vec(&Value::Object(object))?)
    }

    #[must_use]
    pub(crate) fn sampling_audit(&self) -> SamplingAudit {
        let normalized = NormalizedSampling {
            contract: SAMPLING_CONTRACT,
            seed: self.seed,
            talker: self
                .sampling
                .as_ref()
                .map(|sampling| NormalizedTalkerSampling {
                    strategy: sampling.strategy,
                    temperature: sampling.temperature,
                    top_p: sampling.top_p,
                    top_k: sampling.top_k,
                    repetition_penalty: sampling.repetition_penalty,
                }),
            predictor: self
                .sampling
                .as_ref()
                .and_then(|sampling| sampling.predictor.as_ref())
                .map(|predictor| NormalizedPredictorSampling {
                    strategy: predictor.strategy,
                    temperature: predictor.temperature,
                    top_p: predictor.top_p,
                    top_k: predictor.top_k,
                }),
        };
        let mut reasons = Vec::new();
        if self.seed.is_none() {
            reasons.push("seed is not explicit".to_owned());
        }
        match &self.sampling {
            None => reasons.push("sampling object is not explicit".to_owned()),
            Some(sampling) => {
                if sampling.strategy == SamplingStrategy::Sample {
                    require_explicit(
                        &mut reasons,
                        "sampling.temperature",
                        sampling.temperature.is_some(),
                    );
                    require_explicit(&mut reasons, "sampling.top_p", sampling.top_p.is_some());
                    require_explicit(&mut reasons, "sampling.top_k", sampling.top_k.is_some());
                }
                require_explicit(
                    &mut reasons,
                    "sampling.repetition_penalty",
                    sampling.repetition_penalty.is_some(),
                );
                match &sampling.predictor {
                    None => reasons.push("sampling.predictor is not explicit".to_owned()),
                    Some(predictor) if predictor.strategy == SamplingStrategy::Sample => {
                        require_explicit(
                            &mut reasons,
                            "sampling.predictor.temperature",
                            predictor.temperature.is_some(),
                        );
                        require_explicit(
                            &mut reasons,
                            "sampling.predictor.top_p",
                            predictor.top_p.is_some(),
                        );
                        require_explicit(
                            &mut reasons,
                            "sampling.predictor.top_k",
                            predictor.top_k.is_some(),
                        );
                    }
                    Some(_) => {}
                }
            }
        }
        let normalized_sha256 = sha256_hex(
            &serde_json::to_vec(&normalized)
                .expect("normalized sampling serialization cannot fail"),
        );
        SamplingAudit {
            normalized,
            normalized_sha256,
            parity_qualifying: reasons.is_empty(),
            non_qualifying_reasons: reasons,
        }
    }

    #[must_use]
    pub(crate) fn text_sha256(&self) -> String {
        sha256_hex(self.text.as_bytes())
    }

    #[must_use]
    pub(crate) fn voice_description_sha256(&self) -> String {
        sha256_hex(self.voice_description.as_bytes())
    }
}

fn insert_optional<T>(object: &mut Map<String, Value>, name: &str, value: Option<T>)
where
    Value: From<T>,
{
    if let Some(value) = value {
        object.insert(name.to_owned(), Value::from(value));
    }
}

fn require_explicit(reasons: &mut Vec<String>, name: &str, present: bool) {
    if !present {
        reasons.push(format!("{name} is not explicit"));
    }
}

const fn is_safe_id_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')
}

pub(crate) fn load_workload(path: &Path) -> Result<Vec<WorkloadEntry>> {
    let contents = fs::read_to_string(path)?;
    let mut entries = Vec::new();
    let mut ids = HashSet::new();
    for (index, raw) in contents.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let entry: WorkloadEntry = serde_json::from_str(line).map_err(|error| {
            BenchError::Workload(format!("line {line_number}: invalid JSON: {error}"))
        })?;
        entry.validate(line_number)?;
        if !ids.insert(entry.id.clone()) {
            return Err(BenchError::Workload(format!(
                "line {line_number}: duplicate id {:?}",
                entry.id
            )));
        }
        entries.push(entry);
    }
    if entries.is_empty() {
        return Err(BenchError::Workload(
            "workload must contain at least one non-empty JSONL record".to_owned(),
        ));
    }
    Ok(entries)
}

#[must_use]
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn portable_sampling() -> SamplingOptions {
        SamplingOptions {
            strategy: SamplingStrategy::Sample,
            temperature: Some(0.8),
            top_p: Some(0.95),
            top_k: Some(50),
            repetition_penalty: Some(1.05),
            predictor: Some(PredictorSamplingOptions {
                strategy: SamplingStrategy::Sample,
                temperature: Some(0.9),
                top_p: Some(1.0),
                top_k: Some(50),
            }),
        }
    }

    #[test]
    fn concurrency_accepts_only_named_batch_widths() {
        assert_eq!("B1".parse(), Ok(Concurrency::B1));
        assert_eq!("3".parse(), Ok(Concurrency::B3));
        assert_eq!("B6".parse(), Ok(Concurrency::B6));
        assert!("2".parse::<Concurrency>().is_err());
    }

    #[test]
    fn request_body_does_not_include_benchmark_id() {
        let entry = WorkloadEntry {
            id: "safe-id".to_owned(),
            text: "Hello".to_owned(),
            voice_description: "Calm".to_owned(),
            language: "English".to_owned(),
            seed: Some(7),
            max_duration_seconds: None,
            sampling: None,
            stream: true,
        };
        let body: Value =
            serde_json::from_slice(&entry.request_body(BackendProfile::Native, None).unwrap())
                .unwrap();
        assert!(body.get("id").is_none());
        assert_eq!(body["output_format"], "pcm_s16le");
    }

    #[test]
    fn sglang_voice_design_mapping_uses_official_field_names() {
        let entry = WorkloadEntry {
            id: "safe-id".to_owned(),
            text: "Hello".to_owned(),
            voice_description: "Calm".to_owned(),
            language: "English".to_owned(),
            seed: Some(42),
            max_duration_seconds: None,
            sampling: Some(portable_sampling()),
            stream: true,
        };
        let body: Value = serde_json::from_slice(
            &entry
                .request_body(BackendProfile::SglangOmni, Some("Qwen/VoiceDesign"))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["input"], "Hello");
        assert_eq!(body["instructions"], "Calm");
        assert_eq!(body["task_type"], "VoiceDesign");
        assert_eq!(body["response_format"], "pcm");
        assert_eq!(body["stream"], true);
        assert_eq!(body["do_sample"], true);
        assert_eq!(body["temperature"], 0.8);
        assert_eq!(body["top_p"], 0.95);
        assert_eq!(body["top_k"], 50);
        assert_eq!(body["repetition_penalty"], 1.05);
        assert_eq!(body["subtalker_dosample"], true);
        assert_eq!(body["subtalker_temperature"], 0.9);
        assert_eq!(body["subtalker_top_p"], 1.0);
        assert_eq!(body["subtalker_top_k"], 50);
        let audit = entry.sampling_audit();
        assert!(audit.parity_qualifying);
        assert!(audit.non_qualifying_reasons.is_empty());
    }

    #[test]
    fn incomplete_sampling_is_explicitly_non_qualifying() {
        let entry: WorkloadEntry = serde_json::from_value(serde_json::json!({
            "id": "safe-id",
            "text": "Hello",
            "voice_description": "Calm",
            "stream": true
        }))
        .unwrap();
        let audit = entry.sampling_audit();
        assert!(!audit.parity_qualifying);
        assert_eq!(
            audit.non_qualifying_reasons,
            ["seed is not explicit", "sampling object is not explicit"]
        );
    }

    #[test]
    fn sampling_schema_rejects_unknown_fields() {
        let error = serde_json::from_value::<WorkloadEntry>(serde_json::json!({
            "id": "safe-id",
            "text": "Hello",
            "voice_description": "Calm",
            "sampling": {"temperature": 0.8, "typo_top_p": 0.9}
        }))
        .unwrap_err();
        assert!(error.to_string().contains("unknown field `typo_top_p`"));
    }

    #[test]
    fn greedy_sampling_rejects_controls_servers_would_ignore() {
        let entry: WorkloadEntry = serde_json::from_value(serde_json::json!({
            "id": "safe-id",
            "text": "Hello",
            "voice_description": "Calm",
            "sampling": {"strategy": "greedy", "temperature": 0.8}
        }))
        .unwrap();
        let error = entry.validate(1).unwrap_err();
        assert!(error.to_string().contains("would ignore them"));
    }
}
