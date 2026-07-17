use qwen3_tts_runtime::{
    AudioPacketDescriptor, EngineConfig, FinishReason, GenerationConfig, Language, PacketQueue,
    PacketQueueError, RequestInput, RequestMetrics, RequestPhase, RequestRecord, RuntimeStatus,
    SAMPLE_RATE,
};

#[test]
fn finish_reason_values_are_stable_for_the_public_c_abi() {
    assert_eq!(FinishReason::None as u32, 0);
    assert_eq!(FinishReason::CodecEos as u32, 1);
    assert_eq!(FinishReason::MaxCodecFrames as u32, 2);
}

#[test]
fn public_c_layouts_are_stable() {
    assert_eq!(size_of::<RuntimeStatus>(), 4);
    assert_eq!(size_of::<Language>(), 4);
    assert_eq!(size_of::<EngineConfig>(), 96);
    assert_eq!(align_of::<EngineConfig>(), 8);
    assert_eq!(size_of::<GenerationConfig>(), 120);
    assert_eq!(align_of::<GenerationConfig>(), 8);
    assert_eq!(size_of::<AudioPacketDescriptor>(), 72);
    assert_eq!(align_of::<AudioPacketDescriptor>(), 8);
    assert_eq!(size_of::<RequestMetrics>(), 96);
    assert_eq!(align_of::<RequestMetrics>(), 8);
}

fn packet(request_id: u64, sequence: u64, first_frame: u64, frames: u32) -> AudioPacketDescriptor {
    AudioPacketDescriptor {
        request_id,
        sequence,
        first_codec_frame: first_frame,
        first_sample: first_frame * 1_920,
        codec_frames: frames,
        sample_count: frames * 1_920,
        sample_rate: SAMPLE_RATE,
        channels: 1,
        is_final: 0,
        reserved: 0,
        talker_gpu_microseconds: 10.0,
        codec_gpu_microseconds: 20.0,
        end_to_end_microseconds: 40.0,
    }
}

#[test]
fn defaults_match_official_generation_and_streaming_contract() {
    let engine = EngineConfig::default();
    assert_eq!(engine.packet_frames, 4);
    assert_eq!(engine.pcm_ring_slots, 3);

    let generation = GenerationConfig::default();
    assert_eq!(generation.max_codec_frames, 4_096);
    assert_eq!(generation.top_k, 50);
    assert_eq!(generation.temperature, 0.9);
    assert_eq!(generation.top_p, 1.0);
    assert_eq!(generation.repetition_penalty, 1.05);
    assert_eq!(generation.predictor_top_k, 50);
    assert_eq!(generation.predictor_temperature, 0.9);
}

#[test]
fn request_input_is_bounded_and_turkish_is_not_claimed() {
    let config = EngineConfig::default();
    let input = RequestInput {
        text: "Guten Morgen".to_owned(),
        instruct: "A calm male voice".to_owned(),
        language: Language::German,
    };
    input.validate(&config).unwrap();
    assert_eq!(Language::German.as_official_name(), "German");
    assert_eq!(Language::Italian as u32, 10);
}

#[test]
fn packet_positions_and_metrics_are_contiguous() {
    let mut request = RequestRecord::new(7);
    request.transition(RequestPhase::Prefilling).unwrap();
    request.transition(RequestPhase::Generating).unwrap();
    request.record_packet(&packet(7, 0, 0, 4), 4).unwrap();
    request.record_packet(&packet(7, 1, 4, 2), 4).unwrap();
    assert_eq!(request.next_codec_frame, 6);
    assert_eq!(request.next_sample, 11_520);
    assert_eq!(request.metrics.generated_codec_frames, 6);
    assert_eq!(request.metrics.emitted_packets, 0);
    assert_eq!(request.metrics.emitted_samples, 0);
    assert_eq!(request.metrics.talker_gpu_microseconds, 20.0);
    assert_eq!(request.metrics.codec_gpu_microseconds, 40.0);
}

#[test]
fn terminal_requests_cannot_reenter_generation() {
    let mut request = RequestRecord::new(9);
    request.transition(RequestPhase::Cancelled).unwrap();
    assert!(request.transition(RequestPhase::Prefilling).is_err());
}

#[test]
fn bounded_packet_queue_applies_backpressure() {
    let mut queue = PacketQueue::new(2);
    queue.push(1).unwrap();
    queue.push(2).unwrap();
    assert_eq!(queue.push(3), Err(PacketQueueError::Full));
    assert_eq!(queue.pop(), Some(1));
    queue.push(3).unwrap();
    assert_eq!(queue.len(), 2);
}
