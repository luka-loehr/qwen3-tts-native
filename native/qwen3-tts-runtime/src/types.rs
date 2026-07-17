use std::fmt;

use crate::SAMPLES_PER_CODEC_FRAME;

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeStatus {
    Ok = 0,
    WouldBlock = 1,
    EndOfStream = 2,
    InvalidArgument = -1,
    InvalidUtf8 = -2,
    UnsupportedLanguage = -3,
    Model = -4,
    Allocation = -5,
    Cuda = -6,
    State = -7,
    Cancelled = -8,
    Internal = -9,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Language {
    Auto = 0,
    Chinese = 1,
    English = 2,
    Japanese = 3,
    Korean = 4,
    German = 5,
    French = 6,
    Russian = 7,
    Portuguese = 8,
    Spanish = 9,
    Italian = 10,
}

impl Language {
    pub const fn as_official_name(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Chinese => "Chinese",
            Self::English => "English",
            Self::Japanese => "Japanese",
            Self::Korean => "Korean",
            Self::German => "German",
            Self::French => "French",
            Self::Russian => "Russian",
            Self::Portuguese => "Portuguese",
            Self::Spanish => "Spanish",
            Self::Italian => "Italian",
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngineConfig {
    pub struct_size: u32,
    pub device_index: i32,
    pub max_concurrent_requests: u32,
    pub packet_frames: u32,
    pub pcm_ring_slots: u32,
    pub max_text_bytes: u32,
    pub max_instruct_bytes: u32,
    pub flags: u32,
    pub reserved: [u64; 8],
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            device_index: 0,
            max_concurrent_requests: 3,
            packet_frames: 4,
            pcm_ring_slots: 3,
            max_text_bytes: 64 * 1024,
            max_instruct_bytes: 16 * 1024,
            flags: 0,
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationConfig {
    pub struct_size: u32,
    pub max_codec_frames: u32,
    pub seed: u64,
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub top_k: u32,
    pub do_sample: u32,
    pub predictor_temperature: f32,
    pub predictor_top_p: f32,
    pub predictor_top_k: u32,
    pub predictor_do_sample: u32,
    pub reserved: [u64; 8],
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            max_codec_frames: 4_096,
            seed: 0,
            temperature: 0.9,
            top_p: 1.0,
            repetition_penalty: 1.05,
            top_k: 50,
            do_sample: 1,
            predictor_temperature: 0.9,
            predictor_top_p: 1.0,
            predictor_top_k: 50,
            predictor_do_sample: 1,
            reserved: [0; 8],
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestInput {
    pub text: String,
    pub instruct: String,
    pub language: Language,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RequestInputError {
    EmptyText,
    TextTooLarge { actual: usize, maximum: usize },
    InstructTooLarge { actual: usize, maximum: usize },
}

impl fmt::Display for RequestInputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyText => formatter.write_str("text must not be empty"),
            Self::TextTooLarge { actual, maximum } => {
                write!(formatter, "text is {actual} bytes; maximum is {maximum}")
            }
            Self::InstructTooLarge { actual, maximum } => write!(
                formatter,
                "voice instruction is {actual} bytes; maximum is {maximum}"
            ),
        }
    }
}

impl std::error::Error for RequestInputError {}

impl RequestInput {
    pub fn validate(&self, config: &EngineConfig) -> Result<(), RequestInputError> {
        if self.text.is_empty() {
            return Err(RequestInputError::EmptyText);
        }
        if self.text.len() > config.max_text_bytes as usize {
            return Err(RequestInputError::TextTooLarge {
                actual: self.text.len(),
                maximum: config.max_text_bytes as usize,
            });
        }
        if self.instruct.len() > config.max_instruct_bytes as usize {
            return Err(RequestInputError::InstructTooLarge {
                actual: self.instruct.len(),
                maximum: config.max_instruct_bytes as usize,
            });
        }
        Ok(())
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestPhase {
    Queued = 0,
    Prefilling = 1,
    Generating = 2,
    Draining = 3,
    Completed = 4,
    Cancelled = 5,
    Failed = 6,
}

impl RequestPhase {
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Failed)
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FinishReason {
    #[default]
    None = 0,
    CodecEos = 1,
    MaxCodecFrames = 2,
}

impl FinishReason {
    pub const fn is_terminal(self) -> bool {
        !matches!(self, Self::None)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AudioPacketDescriptor {
    pub request_id: u64,
    pub sequence: u64,
    pub first_codec_frame: u64,
    pub first_sample: u64,
    pub codec_frames: u32,
    pub sample_count: u32,
    pub sample_rate: u32,
    pub channels: u32,
    pub is_final: u32,
    pub reserved: u32,
    pub talker_gpu_microseconds: f32,
    pub codec_gpu_microseconds: f32,
    pub end_to_end_microseconds: f32,
}

impl AudioPacketDescriptor {
    pub fn validate(&self, packet_frames: u32) -> Result<(), &'static str> {
        if self.codec_frames == 0 {
            return Err("audio packet must contain at least one codec frame");
        }
        if self.codec_frames > packet_frames {
            return Err("audio packet exceeds configured packet frame count");
        }
        if self.sample_count != self.codec_frames * SAMPLES_PER_CODEC_FRAME {
            return Err("audio packet sample count does not match codec frame count");
        }
        if self.sample_rate != crate::SAMPLE_RATE || self.channels != 1 {
            return Err("audio packet must be mono 24 kHz PCM");
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RequestMetrics {
    pub queue_microseconds: u64,
    pub prefill_microseconds: u64,
    pub first_codec_frame_microseconds: u64,
    pub first_audio_microseconds: u64,
    pub wall_microseconds: u64,
    pub generated_codec_frames: u64,
    pub emitted_samples: u64,
    pub emitted_packets: u64,
    pub talker_gpu_microseconds: f64,
    pub codec_gpu_microseconds: f64,
    pub peak_request_device_bytes: u64,
    pub peak_request_host_bytes: u64,
}
