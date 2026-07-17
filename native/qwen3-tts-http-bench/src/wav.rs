use crate::model::SAMPLE_RATE_HZ;
use crate::{BenchError, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WavInfo {
    pub sample_count: u64,
    pub data_offset: usize,
    pub data_bytes: usize,
}

pub(crate) fn validate_wav(bytes: &[u8]) -> Result<WavInfo> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(BenchError::Wav(
            "response is not a RIFF/WAVE file".to_owned(),
        ));
    }
    let riff_size = read_u32(bytes, 4)? as usize;
    if riff_size.checked_add(8) != Some(bytes.len()) {
        return Err(BenchError::Wav(
            "RIFF size does not match the response length".to_owned(),
        ));
    }

    let mut cursor = 12;
    let mut format_seen = false;
    let mut data = None;
    while cursor < bytes.len() {
        if bytes.len() - cursor < 8 {
            return Err(BenchError::Wav("truncated RIFF chunk header".to_owned()));
        }
        let chunk_id = &bytes[cursor..cursor + 4];
        let chunk_size = read_u32(bytes, cursor + 4)? as usize;
        let payload_start = cursor + 8;
        let payload_end = payload_start
            .checked_add(chunk_size)
            .ok_or_else(|| BenchError::Wav("RIFF chunk size overflow".to_owned()))?;
        if payload_end > bytes.len() {
            return Err(BenchError::Wav(
                "RIFF chunk extends past the response".to_owned(),
            ));
        }
        match chunk_id {
            b"fmt " => {
                if format_seen {
                    return Err(BenchError::Wav("duplicate fmt chunk".to_owned()));
                }
                validate_format(&bytes[payload_start..payload_end])?;
                format_seen = true;
            }
            b"data" => {
                if data.is_some() {
                    return Err(BenchError::Wav("duplicate data chunk".to_owned()));
                }
                if chunk_size == 0 || !chunk_size.is_multiple_of(2) {
                    return Err(BenchError::Wav(
                        "PCM data must contain a non-empty whole number of 16-bit samples"
                            .to_owned(),
                    ));
                }
                data = Some((payload_start, chunk_size));
            }
            _ => {}
        }
        cursor = payload_end
            .checked_add(chunk_size & 1)
            .ok_or_else(|| BenchError::Wav("RIFF padding overflow".to_owned()))?;
        if cursor > bytes.len() {
            return Err(BenchError::Wav("missing RIFF chunk padding".to_owned()));
        }
    }
    if cursor != bytes.len() {
        return Err(BenchError::Wav(
            "RIFF chunks do not consume the full response".to_owned(),
        ));
    }
    if !format_seen {
        return Err(BenchError::Wav("WAV file has no fmt chunk".to_owned()));
    }
    let (data_offset, data_bytes) =
        data.ok_or_else(|| BenchError::Wav("WAV file has no PCM data chunk".to_owned()))?;
    Ok(WavInfo {
        sample_count: u64::try_from(data_bytes / 2).expect("usize fits u64"),
        data_offset,
        data_bytes,
    })
}

fn validate_format(bytes: &[u8]) -> Result<()> {
    if bytes.len() < 16 {
        return Err(BenchError::Wav(
            "fmt chunk is shorter than 16 bytes".to_owned(),
        ));
    }
    let audio_format = read_u16(bytes, 0)?;
    let channels = read_u16(bytes, 2)?;
    let sample_rate = read_u32(bytes, 4)?;
    let byte_rate = read_u32(bytes, 8)?;
    let block_align = read_u16(bytes, 12)?;
    let bits_per_sample = read_u16(bytes, 14)?;
    if audio_format != 1
        || channels != 1
        || u64::from(sample_rate) != SAMPLE_RATE_HZ
        || byte_rate != sample_rate * 2
        || block_align != 2
        || bits_per_sample != 16
    {
        return Err(BenchError::Wav(
            "WAV must be PCM16 little-endian mono at 24 kHz".to_owned(),
        ));
    }
    Ok(())
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let value = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| BenchError::Wav("truncated 16-bit WAV field".to_owned()))?;
    Ok(u16::from_le_bytes(
        value.try_into().expect("slice length was checked"),
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| BenchError::Wav("truncated 32-bit WAV field".to_owned()))?;
    Ok(u32::from_le_bytes(
        value.try_into().expect("slice length was checked"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(data: &[u8]) -> Vec<u8> {
        let riff_size = 36 + u32::try_from(data.len()).unwrap();
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&riff_size.to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&24_000_u32.to_le_bytes());
        wav.extend_from_slice(&48_000_u32.to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&u32::try_from(data.len()).unwrap().to_le_bytes());
        wav.extend_from_slice(data);
        wav
    }

    #[test]
    fn validates_native_wav() {
        let wav = fixture(&[1, 0, 255, 255]);
        assert_eq!(
            validate_wav(&wav).unwrap(),
            WavInfo {
                sample_count: 2,
                data_offset: 44,
                data_bytes: 4,
            }
        );
    }

    #[test]
    fn rejects_wrong_declared_size() {
        let mut wav = fixture(&[1, 0]);
        wav[4] = 0;
        assert!(validate_wav(&wav).is_err());
    }
}
