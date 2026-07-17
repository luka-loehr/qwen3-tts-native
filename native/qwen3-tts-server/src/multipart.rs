use bytes::{Bytes, BytesMut};
use serde::Serialize;
use uuid::Uuid;

use crate::api::MODEL_ID;
use crate::engine::{EngineFinishReason, EngineMetrics, EnginePacket};

#[derive(Serialize)]
struct StartEvent<'a> {
    #[serde(rename = "type")]
    event_type: &'static str,
    request_id: Uuid,
    model: &'a str,
    seed: u64,
    audio: AudioDescription,
}

#[derive(Serialize)]
struct AudioDescription {
    encoding: &'static str,
    sample_rate_hz: u32,
    channels: u32,
    samples_per_codec_frame: u32,
}

#[derive(Serialize)]
struct EndEvent {
    #[serde(rename = "type")]
    event_type: &'static str,
    request_id: Uuid,
    finish_reason: EngineFinishReason,
    metrics: EngineMetrics,
}

pub fn boundary(request_id: Uuid) -> String {
    format!("qwen3tts-{}", request_id.simple())
}

pub fn start_part(boundary: &str, request_id: Uuid, seed: u64) -> Bytes {
    let payload = serde_json::to_vec(&StartEvent {
        event_type: "start",
        request_id,
        model: MODEL_ID,
        seed,
        audio: AudioDescription {
            encoding: "pcm_s16le",
            sample_rate_hz: 24_000,
            channels: 1,
            samples_per_codec_frame: 1_920,
        },
    })
    .expect("serializing a fixed start event cannot fail");
    part(boundary, "application/json", &[], &payload, false)
}

pub fn audio_part(boundary: &str, packet: &EnginePacket) -> Bytes {
    let headers = [
        ("X-Sequence", packet.sequence.to_string()),
        ("X-First-Codec-Frame", packet.first_codec_frame.to_string()),
        ("X-First-Sample", packet.first_sample.to_string()),
        ("X-Sample-Count", packet.sample_count.to_string()),
        ("X-Codec-Frames", packet.codec_frames.to_string()),
        ("X-Final", packet.is_final.to_string()),
    ];
    part(
        boundary,
        "audio/pcm;rate=24000;channels=1;format=s16le",
        &headers,
        &packet.pcm_s16le,
        false,
    )
}

pub fn end_part(
    boundary: &str,
    request_id: Uuid,
    finish_reason: EngineFinishReason,
    metrics: EngineMetrics,
) -> Bytes {
    let payload = serde_json::to_vec(&EndEvent {
        event_type: "end",
        request_id,
        finish_reason,
        metrics,
    })
    .expect("serializing a fixed end event cannot fail");
    part(boundary, "application/json", &[], &payload, true)
}

pub fn error_part(boundary: &str, payload: &serde_json::Value) -> Bytes {
    let payload = serde_json::to_vec(payload).expect("serializing an error value cannot fail");
    part(boundary, "application/json", &[], &payload, true)
}

fn part(
    boundary: &str,
    content_type: &str,
    headers: &[(&str, String)],
    payload: &[u8],
    close: bool,
) -> Bytes {
    let mut output = BytesMut::with_capacity(payload.len() + 512);
    output.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    output.extend_from_slice(format!("Content-Type: {content_type}\r\n").as_bytes());
    output.extend_from_slice(format!("Content-Length: {}\r\n", payload.len()).as_bytes());
    for (name, value) in headers {
        output.extend_from_slice(name.as_bytes());
        output.extend_from_slice(b": ");
        output.extend_from_slice(value.as_bytes());
        output.extend_from_slice(b"\r\n");
    }
    output.extend_from_slice(b"\r\n");
    output.extend_from_slice(payload);
    output.extend_from_slice(b"\r\n");
    if close {
        output.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    }
    output.freeze()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_audio_part_has_explicit_application_level_boundaries() {
        let packet = EnginePacket {
            sequence: 2,
            first_codec_frame: 5,
            first_sample: 9_600,
            codec_frames: 1,
            sample_count: 1_920,
            sample_rate: 24_000,
            channels: 1,
            is_final: false,
            pcm_s16le: vec![0, 255, 1, 254],
        };
        let bytes = audio_part("test-boundary", &packet);
        assert!(bytes.starts_with(b"--test-boundary\r\n"));
        assert!(
            bytes
                .windows(20)
                .any(|window| window == b"X-Sample-Count: 1920")
        );
        assert!(bytes.ends_with(&[0, 255, 1, 254, b'\r', b'\n']));
    }
}
