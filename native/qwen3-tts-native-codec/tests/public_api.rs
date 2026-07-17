use qwen3_tts_native_codec::{
    CODEBOOKS, DEVICE_PACKET_ABI_VERSION, DecoderTensorDType, DecoderWeightProvider,
    DecoderWeightTensor, DecoderWeights, MAX_BATCH_STREAMS, MAX_PACKET_FRAMES, NativeCodecLibrary,
    NativeCodecModel, NativeCodecSession, PacketResult, PendingDevicePacket, SAMPLES_PER_FRAME,
    STATUS_INVALID_ARGUMENT,
};

fn assert_send_sync_static<T: Send + Sync + 'static>() {}
fn assert_send_static<T: Send + 'static>() {}

struct ExternalArtifactProvider;

impl DecoderWeightProvider for ExternalArtifactProvider {
    fn decoder_tensor_names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        Box::new(std::iter::empty())
    }

    fn decoder_tensor(&self, _name: &str) -> Option<DecoderWeightTensor<'_>> {
        None
    }
}

#[test]
fn exports_runtime_and_weight_provider_contract() {
    let _load_native = NativeCodecLibrary::load;
    let _library_supports_device_packets = NativeCodecLibrary::supports_device_packets;
    let _model_supports_device_packets = NativeCodecModel::supports_device_packets;
    let _session_supports_device_packets = NativeCodecSession::supports_device_packets;
    let _begin_device_packet = NativeCodecSession::begin_device_packet;
    let _codes_consumed_event = PendingDevicePacket::codes_consumed_event;
    let _try_finish_device_packet = PendingDevicePacket::try_finish;
    let _finish_device_packet = PendingDevicePacket::finish;
    let _open_weights = DecoderWeights::open;
    let external_provider = ExternalArtifactProvider;
    let _provider_object: &dyn DecoderWeightProvider = &external_provider;
    assert_eq!(CODEBOOKS, 16);
    assert_eq!(MAX_PACKET_FRAMES, 4);
    assert_eq!(MAX_BATCH_STREAMS, 6);
    assert_eq!(SAMPLES_PER_FRAME, 1920);
    assert_eq!(DEVICE_PACKET_ABI_VERSION, 2);
    assert_eq!(STATUS_INVALID_ARGUMENT, -1);
    assert_eq!(DecoderTensorDType::Bf16.bytes(), 2);
    assert_eq!(DecoderTensorDType::F32.bytes(), 4);
    assert!(std::mem::size_of::<PacketResult>() > 0);
    assert_send_sync_static::<NativeCodecModel>();
    assert_send_static::<NativeCodecSession>();
}
