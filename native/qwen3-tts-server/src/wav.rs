use std::mem::size_of;

use qwen3_tts_runtime::SAMPLE_RATE;

const WAV_HEADER_BYTES: usize = 44;
const PCM16_BYTES_PER_SAMPLE_U32: u32 = 2;
const PCM16_BLOCK_ALIGN_U16: u16 = 2;

pub fn encode_pcm16_mono(pcm_s16le: &[u8]) -> Result<Vec<u8>, &'static str> {
    if !pcm_s16le.len().is_multiple_of(size_of::<i16>()) {
        return Err("PCM16 payload must contain complete samples");
    }
    let data_bytes = u32::try_from(pcm_s16le.len()).map_err(|_| "WAV payload is too large")?;
    let riff_bytes = data_bytes
        .checked_add(36)
        .ok_or("WAV RIFF length overflowed")?;
    let byte_rate = SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE_U32;
    let mut wav = Vec::with_capacity(WAV_HEADER_BYTES + pcm_s16le.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&riff_bytes.to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&PCM16_BLOCK_ALIGN_U16.to_le_bytes());
    wav.extend_from_slice(&16_u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_bytes.to_le_bytes());
    wav.extend_from_slice(pcm_s16le);
    Ok(wav)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header_describes_exact_native_pcm() {
        let pcm = [1_u8, 0, 255, 255];
        let wav = encode_pcm16_mono(&pcm).unwrap();
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 24_000);
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 4);
        assert_eq!(&wav[44..], pcm.as_slice());
    }
}
