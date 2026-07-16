use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail};
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;

const VOICE_DESIGN_INVENTORY: &str =
    include_str!("../../../benchmarks/results/voice-design-model-inventory.json");
const SPEECH_TOKENIZER_INVENTORY: &str =
    include_str!("../../../benchmarks/results/speech-tokenizer-model-inventory.json");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodecWeightDtype {
    F32,
    Bf16,
}

impl CodecWeightDtype {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "f32" => Ok(Self::F32),
            "bf16" => Ok(Self::Bf16),
            _ => bail!("unsupported codec weight dtype {value:?}; expected f32 or bf16"),
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::Bf16 => "BF16",
        }
    }

    pub const fn safetensors_dtype(self) -> Dtype {
        match self {
            Self::F32 => Dtype::F32,
            Self::Bf16 => Dtype::BF16,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub parameters: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug)]
pub struct TensorContract {
    pub label: String,
    pub tensors: BTreeMap<String, TensorSpec>,
    pub tensor_count: usize,
    pub parameter_count: u64,
    pub payload_bytes: u64,
    pub exact_file_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContractSummary {
    pub tensor_count: usize,
    pub parameter_count: u64,
    pub payload_bytes: u64,
}

pub fn voice_design_contract() -> Result<TensorContract> {
    parse_inventory(
        VOICE_DESIGN_INVENTORY,
        "voice-design-1.7b",
        |_| true,
        None,
        true,
    )
}

pub fn speech_tokenizer_contract() -> Result<TensorContract> {
    parse_inventory(
        SPEECH_TOKENIZER_INVENTORY,
        "speech-tokenizer",
        |_| true,
        None,
        true,
    )
}

pub fn speech_decoder_contract(dtype: CodecWeightDtype) -> Result<TensorContract> {
    parse_inventory(
        SPEECH_TOKENIZER_INVENTORY,
        &format!("speech-decoder-{}", dtype.label().to_ascii_lowercase()),
        |name| name.starts_with("decoder."),
        Some(dtype.safetensors_dtype()),
        false,
    )
}

pub fn validate_tensors(
    tensors: &SafeTensors<'_>,
    contract: &TensorContract,
) -> Result<ContractSummary> {
    if tensors.len() != contract.tensor_count {
        bail!(
            "{} tensor count mismatch: expected {}, found {}",
            contract.label,
            contract.tensor_count,
            tensors.len()
        );
    }

    let actual_names = tensors.names().into_iter().collect::<BTreeSet<_>>();
    let expected_names = contract
        .tensors
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if actual_names != expected_names {
        let missing = expected_names
            .difference(&actual_names)
            .take(8)
            .copied()
            .collect::<Vec<_>>();
        let unexpected = actual_names
            .difference(&expected_names)
            .take(8)
            .copied()
            .collect::<Vec<_>>();
        bail!(
            "{} tensor names differ; missing {missing:?}, unexpected {unexpected:?}",
            contract.label
        );
    }

    let mut parameter_count = 0_u64;
    let mut payload_bytes = 0_u64;
    for (name, spec) in &contract.tensors {
        let tensor = tensors
            .tensor(name)
            .with_context(|| format!("{} is missing tensor {name}", contract.label))?;
        if tensor.dtype() != spec.dtype {
            bail!(
                "{} tensor {name} dtype mismatch: expected {:?}, found {:?}",
                contract.label,
                spec.dtype,
                tensor.dtype()
            );
        }
        if tensor.shape() != spec.shape {
            bail!(
                "{} tensor {name} shape mismatch: expected {:?}, found {:?}",
                contract.label,
                spec.shape,
                tensor.shape()
            );
        }
        if tensor.data().len() as u64 != spec.bytes {
            bail!(
                "{} tensor {name} byte mismatch: expected {}, found {}",
                contract.label,
                spec.bytes,
                tensor.data().len()
            );
        }
        parameter_count = parameter_count
            .checked_add(spec.parameters)
            .context("contract parameter count overflowed")?;
        payload_bytes = payload_bytes
            .checked_add(spec.bytes)
            .context("contract payload byte count overflowed")?;
    }

    if parameter_count != contract.parameter_count || payload_bytes != contract.payload_bytes {
        bail!(
            "{} aggregate totals do not match its contract",
            contract.label
        );
    }
    Ok(ContractSummary {
        tensor_count: tensors.len(),
        parameter_count,
        payload_bytes,
    })
}

fn parse_inventory(
    encoded: &str,
    label: &str,
    include: impl Fn(&str) -> bool,
    dtype_override: Option<Dtype>,
    retain_exact_file_bytes: bool,
) -> Result<TensorContract> {
    let inventory: Value = serde_json::from_str(encoded)
        .with_context(|| format!("embedded {label} inventory is invalid JSON"))?;
    let entries = inventory["tensors"]
        .as_array()
        .with_context(|| format!("{label} inventory has no tensor array"))?;

    let mut tensors = BTreeMap::new();
    let mut parameter_count = 0_u64;
    let mut payload_bytes = 0_u64;

    for entry in entries {
        let name = entry["name"]
            .as_str()
            .with_context(|| format!("{label} inventory tensor has no name"))?;
        if !include(name) {
            continue;
        }
        let source_dtype = parse_dtype(
            entry["dtype"]
                .as_str()
                .with_context(|| format!("{label} tensor {name} has no dtype"))?,
        )?;
        let dtype = dtype_override.unwrap_or(source_dtype);
        let shape = entry["shape"]
            .as_array()
            .with_context(|| format!("{label} tensor {name} shape is not an array"))?
            .iter()
            .map(|dimension| {
                let value = dimension
                    .as_u64()
                    .with_context(|| format!("{label} tensor {name} dimension is invalid"))?;
                usize::try_from(value)
                    .with_context(|| format!("{label} tensor {name} dimension is too large"))
            })
            .collect::<Result<Vec<_>>>()?;
        let parameters = shape.iter().try_fold(1_u64, |product, dimension| {
            product
                .checked_mul(*dimension as u64)
                .with_context(|| format!("{label} tensor {name} shape overflows"))
        })?;
        let inventory_parameters = entry["parameters"]
            .as_u64()
            .with_context(|| format!("{label} tensor {name} has no parameter count"))?;
        if parameters != inventory_parameters {
            bail!(
                "{label} tensor {name} shape has {parameters} parameters, inventory claims {inventory_parameters}"
            );
        }
        let bytes = parameters
            .checked_mul(bytes_per_element(dtype)?)
            .with_context(|| format!("{label} tensor {name} byte count overflows"))?;
        if dtype_override.is_none() {
            let inventory_bytes = entry["bytes"]
                .as_u64()
                .with_context(|| format!("{label} tensor {name} has no byte count"))?;
            if bytes != inventory_bytes {
                bail!(
                    "{label} tensor {name} dtype/shape implies {bytes} bytes, inventory claims {inventory_bytes}"
                );
            }
        }

        let spec = TensorSpec {
            name: name.to_owned(),
            dtype,
            shape,
            parameters,
            bytes,
        };
        if tensors.insert(name.to_owned(), spec).is_some() {
            bail!("{label} inventory contains duplicate tensor {name}");
        }
        parameter_count = parameter_count
            .checked_add(parameters)
            .context("contract parameter count overflowed")?;
        payload_bytes = payload_bytes
            .checked_add(bytes)
            .context("contract payload byte count overflowed")?;
    }

    if tensors.is_empty() {
        bail!("{label} contract selected no tensors");
    }

    let tensor_count = tensors.len();
    let exact_file_bytes = if retain_exact_file_bytes {
        Some(
            inventory["file_bytes"]
                .as_u64()
                .with_context(|| format!("{label} inventory has no file_bytes"))?,
        )
    } else {
        None
    };

    Ok(TensorContract {
        label: label.to_owned(),
        tensors,
        tensor_count,
        parameter_count,
        payload_bytes,
        exact_file_bytes,
    })
}

fn parse_dtype(label: &str) -> Result<Dtype> {
    match label {
        "BF16" => Ok(Dtype::BF16),
        "F32" => Ok(Dtype::F32),
        _ => bail!("unsupported inventory dtype {label:?}"),
    }
}

fn bytes_per_element(dtype: Dtype) -> Result<u64> {
    match dtype {
        Dtype::BF16 => Ok(2),
        Dtype::F32 => Ok(4),
        _ => bail!("unsupported contract dtype {dtype:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_contract_totals_are_exact() {
        let voice = voice_design_contract().expect("voice contract");
        assert_eq!(voice.tensor_count, 404);
        assert_eq!(voice.parameter_count, 1_916_676_352);
        assert_eq!(voice.payload_bytes, 3_833_352_704);
        assert_eq!(voice.exact_file_bytes, Some(3_833_402_552));

        let tokenizer = speech_tokenizer_contract().expect("tokenizer contract");
        assert_eq!(tokenizer.tensor_count, 496);
        assert_eq!(tokenizer.parameter_count, 170_557_441);
        assert_eq!(tokenizer.payload_bytes, 682_229_764);
        assert_eq!(tokenizer.exact_file_bytes, Some(682_293_092));
    }

    #[test]
    fn decoder_contract_omits_encoder_and_supports_both_dtypes() {
        let f32 = speech_decoder_contract(CodecWeightDtype::F32).expect("f32 decoder");
        let bf16 = speech_decoder_contract(CodecWeightDtype::Bf16).expect("bf16 decoder");

        assert_eq!(f32.tensor_count, 271);
        assert_eq!(f32.parameter_count, 114_323_137);
        assert_eq!(f32.payload_bytes, 457_292_548);
        assert_eq!(bf16.tensor_count, 271);
        assert_eq!(bf16.parameter_count, f32.parameter_count);
        assert_eq!(bf16.payload_bytes, 228_646_274);
        assert!(f32.tensors.keys().all(|name| name.starts_with("decoder.")));
        assert!(
            bf16.tensors
                .values()
                .all(|tensor| tensor.dtype == Dtype::BF16)
        );
        assert!(
            f32.tensors
                .keys()
                .all(|name| bf16.tensors.contains_key(name))
        );
    }

    #[test]
    fn codec_dtype_parser_is_strict() {
        assert_eq!(
            CodecWeightDtype::parse("bf16").expect("bf16"),
            CodecWeightDtype::Bf16
        );
        assert_eq!(
            CodecWeightDtype::parse("f32").expect("f32"),
            CodecWeightDtype::F32
        );
        assert!(CodecWeightDtype::parse("fp16").is_err());
        assert!(CodecWeightDtype::parse("BF16").is_err());
    }
}
