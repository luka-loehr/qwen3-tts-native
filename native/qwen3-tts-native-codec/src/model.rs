use serde_json::Value;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum TensorDType {
    F32 = 1,
    Bf16 = 2,
}

impl TensorDType {
    fn from_safetensors(value: &str) -> Result<Self, io::Error> {
        match value {
            "F32" => Ok(Self::F32),
            "BF16" => Ok(Self::Bf16),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported tensor dtype {other}"),
            )),
        }
    }

    pub fn bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorEntry {
    pub dtype: TensorDType,
    pub shape: Vec<u64>,
    pub data_start: usize,
    pub data_end: usize,
}

impl TensorEntry {
    pub fn byte_len(&self) -> usize {
        self.data_end - self.data_start
    }

    pub fn element_count(&self) -> Result<usize, io::Error> {
        self.shape.iter().try_fold(1_usize, |total, dimension| {
            let dimension = usize::try_from(*dimension).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "tensor dimension exceeds usize")
            })?;
            total.checked_mul(dimension).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "tensor element count overflow")
            })
        })
    }
}

pub struct SafetensorsFile {
    storage: Vec<u8>,
    tensors: BTreeMap<String, TensorEntry>,
}

impl SafetensorsFile {
    pub fn open(path: &Path) -> Result<Self, Box<dyn Error>> {
        let mut file = File::open(path)?;
        let file_len = usize::try_from(file.metadata()?.len())?;
        if file_len < 8 {
            return Err(
                io::Error::new(io::ErrorKind::InvalidData, "truncated safetensors file").into(),
            );
        }

        let mut storage = Vec::with_capacity(file_len);
        file.read_to_end(&mut storage)?;
        let header_len = usize::try_from(u64::from_le_bytes(storage[..8].try_into()?))?;
        let data_base = 8_usize.checked_add(header_len).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "safetensors header length overflow",
            )
        })?;
        if data_base > storage.len() {
            return Err(
                io::Error::new(io::ErrorKind::InvalidData, "truncated safetensors header").into(),
            );
        }

        let header: Value = serde_json::from_slice(&storage[8..data_base])?;
        let object = header.as_object().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "safetensors header is not an object",
            )
        })?;
        let mut tensors = BTreeMap::new();
        for (name, value) in object {
            if name == "__metadata__" {
                continue;
            }
            let tensor = value.as_object().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("tensor {name} is not an object"),
                )
            })?;
            let dtype = TensorDType::from_safetensors(
                tensor.get("dtype").and_then(Value::as_str).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("tensor {name} has no dtype"),
                    )
                })?,
            )?;
            let shape = tensor
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("tensor {name} has no shape"),
                    )
                })?
                .iter()
                .map(|dimension| {
                    dimension.as_u64().ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("tensor {name} has a non-integer dimension"),
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let offsets = tensor
                .get("data_offsets")
                .and_then(Value::as_array)
                .filter(|offsets| offsets.len() == 2)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("tensor {name} has invalid offsets"),
                    )
                })?;
            let relative_start = usize::try_from(offsets[0].as_u64().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("tensor {name} has invalid start offset"),
                )
            })?)?;
            let relative_end = usize::try_from(offsets[1].as_u64().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("tensor {name} has invalid end offset"),
                )
            })?)?;
            let data_start = data_base.checked_add(relative_start).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("tensor {name} start offset overflow"),
                )
            })?;
            let data_end = data_base.checked_add(relative_end).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("tensor {name} end offset overflow"),
                )
            })?;
            if data_start > data_end || data_end > storage.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("tensor {name} data range is outside the file"),
                )
                .into());
            }
            let entry = TensorEntry {
                dtype,
                shape,
                data_start,
                data_end,
            };
            let expected = entry
                .element_count()?
                .checked_mul(dtype.bytes())
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "tensor byte count overflow")
                })?;
            if entry.byte_len() != expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "tensor {name} byte count is {}, expected {expected}",
                        entry.byte_len()
                    ),
                )
                .into());
            }
            tensors.insert(name.clone(), entry);
        }

        Ok(Self { storage, tensors })
    }

    pub fn tensor(&self, name: &str) -> Option<(&TensorEntry, &[u8])> {
        self.tensors
            .get(name)
            .map(|entry| (entry, &self.storage[entry.data_start..entry.data_end]))
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn decoder_tensor_count(&self) -> usize {
        self.tensors
            .keys()
            .filter(|name| name.starts_with("decoder."))
            .count()
    }

    pub fn decoder_payload_bytes(&self) -> usize {
        self.tensors
            .iter()
            .filter(|(name, _)| name.starts_with("decoder."))
            .map(|(_, entry)| entry.byte_len())
            .sum()
    }

    pub fn decoder_dtype_counts(&self) -> (usize, usize) {
        self.tensors
            .iter()
            .filter(|(name, _)| name.starts_with("decoder."))
            .fold((0, 0), |(f32_count, bf16_count), (_, entry)| {
                match entry.dtype {
                    TensorDType::F32 => (f32_count + 1, bf16_count),
                    TensorDType::Bf16 => (f32_count, bf16_count + 1),
                }
            })
    }
}

#[cfg(test)]
mod tests {
    use super::{SafetensorsFile, TensorDType};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parses_minimal_f32_file() {
        let header = r#"{"decoder.test":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let padded = format!("{header:<128}");
        let mut bytes = (padded.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(padded.as_bytes());
        bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        bytes.extend_from_slice(&2.0_f32.to_le_bytes());
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("qwen3-tts-codec-{nonce}.safetensors"));
        fs::write(&path, bytes).expect("write fixture");
        let model = SafetensorsFile::open(&path).expect("parse fixture");
        fs::remove_file(path).expect("remove fixture");
        let (entry, payload) = model.tensor("decoder.test").expect("tensor exists");
        assert_eq!(entry.dtype, TensorDType::F32);
        assert_eq!(entry.shape, [2]);
        assert_eq!(payload.len(), 8);
        assert_eq!(model.decoder_tensor_count(), 1);
    }
}
