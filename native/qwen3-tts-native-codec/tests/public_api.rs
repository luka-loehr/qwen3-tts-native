use qwen3_tts_native_codec::{
    CODEBOOKS, DecoderTensorDType, DecoderWeightProvider, DecoderWeightTensor, DecoderWeights,
    MAX_BATCH_STREAMS, MAX_PACKET_FRAMES, NativeCodecLibrary, NativeCodecModel, NativeCodecSession,
    PacketResult, SAMPLES_PER_FRAME,
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
    let _open_weights = DecoderWeights::open;
    let external_provider = ExternalArtifactProvider;
    let _provider_object: &dyn DecoderWeightProvider = &external_provider;
    assert_eq!(CODEBOOKS, 16);
    assert_eq!(MAX_PACKET_FRAMES, 4);
    assert_eq!(MAX_BATCH_STREAMS, 6);
    assert_eq!(SAMPLES_PER_FRAME, 1920);
    assert_eq!(DecoderTensorDType::Bf16.bytes(), 2);
    assert_eq!(DecoderTensorDType::F32.bytes(), 4);
    assert!(std::mem::size_of::<PacketResult>() > 0);
    assert_send_sync_static::<NativeCodecModel>();
    assert_send_static::<NativeCodecSession>();
}
