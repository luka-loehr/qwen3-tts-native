use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

pub trait WeightProvider {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>>;
    fn tensor_names(&self) -> Result<Vec<String>>;
}

pub struct SafeTensorProvider {
    path: PathBuf,
    mapping: Mmap,
}

impl SafeTensorProvider {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open checkpoint {}", path.display()))?;
        // SAFETY: the mapping is read-only and the provider retains both the
        // mapping and the immutable backing file for its full lifetime.
        let mapping = unsafe { Mmap::map(&file) }
            .with_context(|| format!("failed to map checkpoint {}", path.display()))?;
        SafeTensors::deserialize(&mapping)
            .with_context(|| format!("invalid Safetensors checkpoint {}", path.display()))?;
        Ok(Self {
            path: path.to_owned(),
            mapping,
        })
    }

    pub fn expect_bf16(&self, name: &str, shape: &[usize]) -> Result<TensorView<'_>> {
        let tensor = self.tensor(name)?;
        ensure!(tensor.dtype() == Dtype::BF16, "{name} must be BF16");
        ensure!(
            tensor.shape() == shape,
            "{name}: expected shape {shape:?}, found {:?}",
            tensor.shape()
        );
        Ok(tensor)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl WeightProvider for SafeTensorProvider {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>> {
        let tensors = SafeTensors::deserialize(&self.mapping)
            .with_context(|| format!("invalid Safetensors checkpoint {}", self.path.display()))?;
        tensors
            .tensor(name)
            .with_context(|| format!("checkpoint is missing tensor {name}"))
    }

    fn tensor_names(&self) -> Result<Vec<String>> {
        let tensors = SafeTensors::deserialize(&self.mapping)
            .with_context(|| format!("invalid Safetensors checkpoint {}", self.path.display()))?;
        Ok(tensors.names().into_iter().map(str::to_owned).collect())
    }
}
