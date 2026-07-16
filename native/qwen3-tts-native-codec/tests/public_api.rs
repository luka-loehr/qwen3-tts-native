use qwen3_tts_native_codec::{
    CODEBOOKS, DecoderTensorDType, DecoderWeights, MAX_BATCH_STREAMS, MAX_PACKET_FRAMES,
    NativeCodecLibrary, PacketResult, SAMPLES_PER_FRAME,
};

#[test]
fn exports_runtime_and_weight_provider_contract() {
    let _load_native = NativeCodecLibrary::load;
    let _open_weights = DecoderWeights::open;
    assert_eq!(CODEBOOKS, 16);
    assert_eq!(MAX_PACKET_FRAMES, 4);
    assert_eq!(MAX_BATCH_STREAMS, 6);
    assert_eq!(SAMPLES_PER_FRAME, 1920);
    assert_eq!(DecoderTensorDType::Bf16.bytes(), 2);
    assert_eq!(DecoderTensorDType::F32.bytes(), 4);
    assert!(std::mem::size_of::<PacketResult>() > 0);
}
