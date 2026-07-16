use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};

pub fn write_pcm16_mono(path: &Path, sample_rate: u32, samples: &[i16]) -> Result<()> {
    let data_bytes = samples
        .len()
        .checked_mul(size_of::<i16>())
        .context("WAV payload length overflow")?;
    let data_bytes = u32::try_from(data_bytes).context("WAV payload exceeds RIFF limit")?;
    let riff_bytes = 36_u32
        .checked_add(data_bytes)
        .context("WAV RIFF length overflow")?;
    if sample_rate == 0 {
        bail!("WAV sample rate must be non-zero");
    }

    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut output = BufWriter::new(file);
    output.write_all(b"RIFF")?;
    output.write_all(&riff_bytes.to_le_bytes())?;
    output.write_all(b"WAVEfmt ")?;
    output.write_all(&16_u32.to_le_bytes())?;
    output.write_all(&1_u16.to_le_bytes())?;
    output.write_all(&1_u16.to_le_bytes())?;
    output.write_all(&sample_rate.to_le_bytes())?;
    output.write_all(&(sample_rate * 2).to_le_bytes())?;
    output.write_all(&2_u16.to_le_bytes())?;
    output.write_all(&16_u16.to_le_bytes())?;
    output.write_all(b"data")?;
    output.write_all(&data_bytes.to_le_bytes())?;
    for sample in samples {
        output.write_all(&sample.to_le_bytes())?;
    }
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::write_pcm16_mono;

    #[test]
    fn wav_header_declares_exact_payload() {
        let path = std::env::temp_dir().join(format!(
            "qwen3-tts-bench-wav-header-{}.wav",
            std::process::id()
        ));
        write_pcm16_mono(&path, 24_000, &[1, -2, 3]).unwrap();
        let bytes = fs::read(&path).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(u32::from_le_bytes(bytes[40..44].try_into().unwrap()), 6);
        assert_eq!(bytes.len(), 50);
    }
}
