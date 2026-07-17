use crate::model::{DecoderWeightProvider, TensorDType};
use libloading::Library;
use std::cell::Cell;
use std::error::Error;
use std::ffi::{CStr, CString, c_char, c_void};
use std::marker::PhantomData;
use std::path::Path;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;

pub const CODEBOOKS: usize = 16;
pub const MAX_PACKET_FRAMES: usize = 4;
pub const SAMPLES_PER_FRAME: usize = 1920;
pub const MAX_PACKET_SAMPLES: usize = MAX_PACKET_FRAMES * SAMPLES_PER_FRAME;
pub const MAX_BATCH_STREAMS: usize = 6;
pub const DEVICE_PACKET_ABI_VERSION: i32 = 2;
pub const STATUS_INVALID_ARGUMENT: i32 = -1;
pub const STATUS_STATE: i32 = -3;
pub const STATUS_MODEL: i32 = -5;

#[repr(C)]
pub struct Context {
    _private: [u8; 0],
}

#[repr(C)]
pub struct SharedModelHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct SessionHandle {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Config {
    pub device_index: i32,
    pub ring_slots: i32,
    pub max_packet_frames: i32,
    pub reserved: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StateInfo {
    pub frame_position: u64,
    pub emitted_samples: u64,
    pub device_bytes: u64,
    pub host_pinned_bytes: u64,
    pub transformer_kv_bytes: u64,
    pub convolution_history_bytes: u64,
    pub codec_ring_bytes: u64,
    pub pcm_ring_bytes: u64,
    pub kv_ring_head: u32,
    pub next_ring_slot: u32,
    pub ring_slots: u32,
    pub max_packet_frames: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PacketResult {
    pub first_frame_position: u64,
    pub first_sample_position: u64,
    pub frame_count: u32,
    pub sample_count: u32,
    pub ring_slot: u32,
    pub is_final: u32,
    pub gpu_microseconds: f32,
    pub end_to_end_microseconds: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DevicePacketBeginV2 {
    struct_size: u32,
    reserved: u32,
    device_codec_frames: *const u16,
    frame_count: u32,
    reserved_2: u32,
    producer_ready_event: *mut c_void,
    reserved_3: u64,
}

impl DevicePacketBeginV2 {
    fn new(
        device_codec_frames: *const u16,
        frame_count: usize,
        producer_ready_event: *mut c_void,
    ) -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            reserved: 0,
            device_codec_frames,
            frame_count: frame_count as u32,
            reserved_2: 0,
            producer_ready_event,
            reserved_3: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DevicePacketBeginResultV2 {
    struct_size: u32,
    reserved: u32,
    ticket_id: u64,
    codes_consumed_event: *mut c_void,
    reserved_2: u64,
}

impl DevicePacketBeginResultV2 {
    fn new() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            reserved: 0,
            ticket_id: 0,
            codes_consumed_event: std::ptr::null_mut(),
            reserved_2: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DevicePacketFinishV2 {
    struct_size: u32,
    reserved: u32,
    ticket_id: u64,
    is_final: i32,
    reserved_2: u32,
    pcm_output: *mut i16,
    pcm_capacity_samples: usize,
    result: *mut PacketResult,
    reserved_3: u64,
}

impl DevicePacketFinishV2 {
    fn new(
        ticket_id: u64,
        is_final: bool,
        pcm_output: *mut i16,
        pcm_capacity_samples: usize,
        result: *mut PacketResult,
    ) -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            reserved: 0,
            ticket_id,
            is_final: i32::from(is_final),
            reserved_2: 0,
            pcm_output,
            pcm_capacity_samples,
            result,
            reserved_3: 0,
        }
    }
}

// These assertions mirror native/include/qwen3_tts_codec.h on the 64-bit
// Linux deployment target. A layout drift is therefore a compile-time error,
// before a mismatched request can cross the dynamic-library boundary.
#[cfg(target_pointer_width = "64")]
const _: () = {
    use std::mem::{align_of, offset_of, size_of};

    assert!(size_of::<PacketResult>() == 40);
    assert!(align_of::<PacketResult>() == 8);
    assert!(offset_of!(PacketResult, first_frame_position) == 0);
    assert!(offset_of!(PacketResult, first_sample_position) == 8);
    assert!(offset_of!(PacketResult, frame_count) == 16);
    assert!(offset_of!(PacketResult, sample_count) == 20);
    assert!(offset_of!(PacketResult, ring_slot) == 24);
    assert!(offset_of!(PacketResult, is_final) == 28);
    assert!(offset_of!(PacketResult, gpu_microseconds) == 32);
    assert!(offset_of!(PacketResult, end_to_end_microseconds) == 36);

    assert!(size_of::<DevicePacketBeginV2>() == 40);
    assert!(align_of::<DevicePacketBeginV2>() == 8);
    assert!(offset_of!(DevicePacketBeginV2, struct_size) == 0);
    assert!(offset_of!(DevicePacketBeginV2, reserved) == 4);
    assert!(offset_of!(DevicePacketBeginV2, device_codec_frames) == 8);
    assert!(offset_of!(DevicePacketBeginV2, frame_count) == 16);
    assert!(offset_of!(DevicePacketBeginV2, reserved_2) == 20);
    assert!(offset_of!(DevicePacketBeginV2, producer_ready_event) == 24);
    assert!(offset_of!(DevicePacketBeginV2, reserved_3) == 32);

    assert!(size_of::<DevicePacketBeginResultV2>() == 32);
    assert!(align_of::<DevicePacketBeginResultV2>() == 8);
    assert!(offset_of!(DevicePacketBeginResultV2, struct_size) == 0);
    assert!(offset_of!(DevicePacketBeginResultV2, reserved) == 4);
    assert!(offset_of!(DevicePacketBeginResultV2, ticket_id) == 8);
    assert!(offset_of!(DevicePacketBeginResultV2, codes_consumed_event) == 16);
    assert!(offset_of!(DevicePacketBeginResultV2, reserved_2) == 24);

    assert!(size_of::<DevicePacketFinishV2>() == 56);
    assert!(align_of::<DevicePacketFinishV2>() == 8);
    assert!(offset_of!(DevicePacketFinishV2, struct_size) == 0);
    assert!(offset_of!(DevicePacketFinishV2, reserved) == 4);
    assert!(offset_of!(DevicePacketFinishV2, ticket_id) == 8);
    assert!(offset_of!(DevicePacketFinishV2, is_final) == 16);
    assert!(offset_of!(DevicePacketFinishV2, reserved_2) == 20);
    assert!(offset_of!(DevicePacketFinishV2, pcm_output) == 24);
    assert!(offset_of!(DevicePacketFinishV2, pcm_capacity_samples) == 32);
    assert!(offset_of!(DevicePacketFinishV2, result) == 40);
    assert!(offset_of!(DevicePacketFinishV2, reserved_3) == 48);
};

pub type BatchOutput = Vec<(Vec<i16>, PacketResult)>;

#[repr(C)]
struct BatchItem {
    context: *mut Context,
    codec_frames: *const u16,
    frame_count: u32,
    is_final: i32,
    pcm_output: *mut i16,
    pcm_capacity_samples: usize,
    result: *mut PacketResult,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct TensorView {
    pub data: *const c_void,
    pub byte_length: u64,
    pub shape: [u64; 4],
    pub rank: u32,
    pub dtype: u32,
}

type TensorAtFn = unsafe extern "C" fn(
    *mut c_void,
    u64,
    *mut *const c_char,
    *mut TensorView,
    *mut c_char,
    usize,
) -> i32;

#[repr(C)]
pub struct WeightProvider {
    pub abi_version: u32,
    pub reserved: u32,
    pub tensor_count: u64,
    pub user_data: *mut c_void,
    pub tensor_at: Option<TensorAtFn>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ModelInfo {
    pub source_bytes: u64,
    pub device_bytes: u64,
    pub parameter_count: u64,
    pub tensor_count: u32,
    pub source_dtype_f32_count: u32,
    pub source_dtype_bf16_count: u32,
    pub loaded: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ModelMemoryInfo {
    pub source_bytes: u64,
    pub shared_weight_device_bytes: u64,
    pub parameter_count: u64,
    pub transient_upload_device_bytes: u64,
    pub tensor_count: u32,
    pub warmup_completed: u32,
    pub active_session_count: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SessionMemoryInfo {
    pub device_bytes: u64,
    pub host_pinned_bytes: u64,
    pub transformer_kv_bytes: u64,
    pub convolution_history_bytes: u64,
    pub codec_ring_bytes: u64,
    pub pcm_ring_bytes: u64,
    pub workspace_device_bytes: u64,
    pub reserved: u64,
}

type AbiVersionFn = unsafe extern "C" fn() -> i32;
type CreateFn = unsafe extern "C" fn(*const Config, *mut *mut Context, *mut c_char, usize) -> i32;
type DestroyFn = unsafe extern "C" fn(*mut Context, *mut c_char, usize) -> i32;
type ResetFn = unsafe extern "C" fn(*mut Context, *mut c_char, usize) -> i32;
type WarmupFn = unsafe extern "C" fn(*mut Context, *mut c_char, usize) -> i32;
type StateInfoFn = unsafe extern "C" fn(*const Context, *mut StateInfo, *mut c_char, usize) -> i32;
type ProcessFn = unsafe extern "C" fn(
    *mut Context,
    *const u16,
    u32,
    i32,
    *mut i16,
    usize,
    *mut PacketResult,
    *mut c_char,
    usize,
) -> i32;
type BatchProcessFn = unsafe extern "C" fn(*mut BatchItem, u32, *mut c_char, usize) -> i32;
type LoadModelFn = unsafe extern "C" fn(
    *mut Context,
    *const WeightProvider,
    *mut ModelInfo,
    *mut c_char,
    usize,
) -> i32;
type ModelInfoFn = unsafe extern "C" fn(*const Context, *mut ModelInfo, *mut c_char, usize) -> i32;
type FrontendFn = unsafe extern "C" fn(
    *mut Context,
    *const u16,
    u32,
    *mut f32,
    usize,
    *mut f32,
    usize,
    *mut c_char,
    usize,
) -> i32;
type TransformerFn =
    unsafe extern "C" fn(*mut Context, *const u16, u32, *mut f32, usize, *mut c_char, usize) -> i32;
type LatentFn = unsafe extern "C" fn(
    *mut Context,
    *const u16,
    u32,
    *mut f32,
    usize,
    *mut f32,
    usize,
    *mut c_char,
    usize,
) -> i32;
type DecoderCheckpointFn = unsafe extern "C" fn(
    *mut Context,
    *const u16,
    u32,
    u32,
    *mut f32,
    usize,
    *mut c_char,
    usize,
) -> i32;
type SharedModelCreateFn =
    unsafe extern "C" fn(i32, *mut *mut SharedModelHandle, *mut c_char, usize) -> i32;
type SharedModelDestroyFn = unsafe extern "C" fn(*mut SharedModelHandle, *mut c_char, usize) -> i32;
type SharedModelLoadFn = unsafe extern "C" fn(
    *mut SharedModelHandle,
    *const WeightProvider,
    *mut ModelInfo,
    *mut c_char,
    usize,
) -> i32;
type SharedModelWarmupFn = unsafe extern "C" fn(*mut SharedModelHandle, *mut c_char, usize) -> i32;
type SharedModelInfoFn =
    unsafe extern "C" fn(*const SharedModelHandle, *mut ModelInfo, *mut c_char, usize) -> i32;
type SharedModelMemoryInfoFn =
    unsafe extern "C" fn(*const SharedModelHandle, *mut ModelMemoryInfo, *mut c_char, usize) -> i32;
type SessionCreateFn = unsafe extern "C" fn(
    *mut SharedModelHandle,
    *mut *mut SessionHandle,
    *mut c_char,
    usize,
) -> i32;
type SessionDestroyFn = unsafe extern "C" fn(*mut SessionHandle, *mut c_char, usize) -> i32;
type SessionResetFn = unsafe extern "C" fn(*mut SessionHandle, *mut c_char, usize) -> i32;
type SessionCancelFn = unsafe extern "C" fn(*mut SessionHandle, *mut c_char, usize) -> i32;
type SessionStateInfoFn =
    unsafe extern "C" fn(*const SessionHandle, *mut StateInfo, *mut c_char, usize) -> i32;
type SessionMemoryInfoFn =
    unsafe extern "C" fn(*const SessionHandle, *mut SessionMemoryInfo, *mut c_char, usize) -> i32;
type SessionProcessFn = unsafe extern "C" fn(
    *mut SessionHandle,
    *const u16,
    u32,
    i32,
    *mut i16,
    usize,
    *mut PacketResult,
    *mut c_char,
    usize,
) -> i32;
type SessionDevicePacketBeginFn = unsafe extern "C" fn(
    *mut SessionHandle,
    *const DevicePacketBeginV2,
    *mut DevicePacketBeginResultV2,
    *mut c_char,
    usize,
) -> i32;
type SessionDevicePacketFinishFn = unsafe extern "C" fn(
    *mut SessionHandle,
    *const DevicePacketFinishV2,
    *mut c_char,
    usize,
) -> i32;

#[derive(Clone, Copy)]
struct DevicePacketApi {
    begin: SessionDevicePacketBeginFn,
    finish: SessionDevicePacketFinishFn,
}

pub struct Api {
    _library: Library,
    abi_version: AbiVersionFn,
    create: CreateFn,
    destroy: DestroyFn,
    reset: ResetFn,
    warmup: WarmupFn,
    state_info: StateInfoFn,
    load_model: LoadModelFn,
    model_info: ModelInfoFn,
    frontend: FrontendFn,
    transformer: TransformerFn,
    latent: LatentFn,
    decoder_checkpoint: DecoderCheckpointFn,
    process: ProcessFn,
    batch_process: BatchProcessFn,
    fixture_process: ProcessFn,
    shared_model_create: Option<SharedModelCreateFn>,
    shared_model_destroy: Option<SharedModelDestroyFn>,
    shared_model_load: Option<SharedModelLoadFn>,
    shared_model_warmup: Option<SharedModelWarmupFn>,
    shared_model_info: Option<SharedModelInfoFn>,
    shared_model_memory_info: Option<SharedModelMemoryInfoFn>,
    session_create: Option<SessionCreateFn>,
    session_destroy: Option<SessionDestroyFn>,
    session_reset: Option<SessionResetFn>,
    session_cancel: Option<SessionCancelFn>,
    session_state_info: Option<SessionStateInfoFn>,
    session_memory_info: Option<SessionMemoryInfoFn>,
    session_process: Option<SessionProcessFn>,
    device_packet: Option<DevicePacketApi>,
}

unsafe fn load_symbol<T: Copy>(library: &Library, symbol: &[u8]) -> Result<T, Box<dyn Error>> {
    let loaded = unsafe { library.get::<T>(symbol)? };
    Ok(*loaded)
}

unsafe fn load_optional_symbol<T: Copy>(library: &Library, symbol: &[u8]) -> Option<T> {
    unsafe { library.get::<T>(symbol) }
        .ok()
        .map(|loaded| *loaded)
}

fn validate_device_packet_symbol_set(
    reported_version: Option<i32>,
    has_begin: bool,
    has_finish: bool,
) -> Result<bool, String> {
    match (reported_version, has_begin, has_finish) {
        (None, false, false) => Ok(false),
        (Some(DEVICE_PACKET_ABI_VERSION), true, true) => Ok(true),
        (Some(version), true, true) => Err(format!(
            "device packet ABI version mismatch: expected {DEVICE_PACKET_ABI_VERSION}, got {version}"
        )),
        (version, begin, finish) => Err(format!(
            "incomplete device packet ABI v2 symbol set: abi_version_v2={}, begin_v2={begin}, finish_v2={finish}",
            version.is_some()
        )),
    }
}

impl Api {
    pub fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        let library = unsafe { Library::new(path)? };
        let abi_version = unsafe { load_symbol(&library, b"qwen3_tts_codec_abi_version_v1\0")? };
        let create = unsafe { load_symbol(&library, b"qwen3_tts_codec_create_v1\0")? };
        let destroy = unsafe { load_symbol(&library, b"qwen3_tts_codec_destroy_v1\0")? };
        let reset = unsafe { load_symbol(&library, b"qwen3_tts_codec_reset_v1\0")? };
        let warmup = unsafe { load_symbol(&library, b"qwen3_tts_codec_warmup_v1\0")? };
        let state_info = unsafe { load_symbol(&library, b"qwen3_tts_codec_state_info_v1\0")? };
        let load_model = unsafe { load_symbol(&library, b"qwen3_tts_codec_load_model_v1\0")? };
        let model_info = unsafe { load_symbol(&library, b"qwen3_tts_codec_model_info_v1\0")? };
        let frontend =
            unsafe { load_symbol(&library, b"qwen3_tts_codec_debug_frontend_packet_v1\0")? };
        let transformer =
            unsafe { load_symbol(&library, b"qwen3_tts_codec_debug_transformer_packet_v1\0")? };
        let latent = unsafe { load_symbol(&library, b"qwen3_tts_codec_debug_latent_packet_v1\0")? };
        let decoder_checkpoint =
            unsafe { load_symbol(&library, b"qwen3_tts_codec_debug_decoder_checkpoint_v1\0")? };
        let process = unsafe { load_symbol(&library, b"qwen3_tts_codec_process_packet_v1\0")? };
        let batch_process =
            unsafe { load_symbol(&library, b"qwen3_tts_codec_process_batch_v1\0")? };
        let fixture_process =
            unsafe { load_symbol(&library, b"qwen3_tts_codec_process_fixture_packet_v1\0")? };
        let shared_model_create =
            unsafe { load_optional_symbol(&library, b"qwen3_tts_codec_shared_model_create_v1\0") };
        let shared_model_destroy =
            unsafe { load_optional_symbol(&library, b"qwen3_tts_codec_shared_model_destroy_v1\0") };
        let shared_model_load =
            unsafe { load_optional_symbol(&library, b"qwen3_tts_codec_shared_model_load_v1\0") };
        let shared_model_warmup =
            unsafe { load_optional_symbol(&library, b"qwen3_tts_codec_shared_model_warmup_v1\0") };
        let shared_model_info =
            unsafe { load_optional_symbol(&library, b"qwen3_tts_codec_shared_model_info_v1\0") };
        let shared_model_memory_info = unsafe {
            load_optional_symbol(&library, b"qwen3_tts_codec_shared_model_memory_info_v1\0")
        };
        let session_create =
            unsafe { load_optional_symbol(&library, b"qwen3_tts_codec_session_create_v1\0") };
        let session_destroy =
            unsafe { load_optional_symbol(&library, b"qwen3_tts_codec_session_destroy_v1\0") };
        let session_reset =
            unsafe { load_optional_symbol(&library, b"qwen3_tts_codec_session_reset_v1\0") };
        let session_cancel =
            unsafe { load_optional_symbol(&library, b"qwen3_tts_codec_session_cancel_v1\0") };
        let session_state_info =
            unsafe { load_optional_symbol(&library, b"qwen3_tts_codec_session_state_info_v1\0") };
        let session_memory_info =
            unsafe { load_optional_symbol(&library, b"qwen3_tts_codec_session_memory_info_v1\0") };
        let session_process = unsafe {
            load_optional_symbol(&library, b"qwen3_tts_codec_session_process_packet_v1\0")
        };
        let device_packet_abi_version: Option<AbiVersionFn> =
            unsafe { load_optional_symbol(&library, b"qwen3_tts_codec_abi_version_v2\0") };
        let device_packet_begin: Option<SessionDevicePacketBeginFn> = unsafe {
            load_optional_symbol(
                &library,
                b"qwen3_tts_codec_session_process_device_packet_begin_v2\0",
            )
        };
        let device_packet_finish: Option<SessionDevicePacketFinishFn> = unsafe {
            load_optional_symbol(
                &library,
                b"qwen3_tts_codec_session_process_device_packet_finish_v2\0",
            )
        };
        let reported_device_packet_version =
            device_packet_abi_version.map(|version| unsafe { version() });
        let supports_device_packets = validate_device_packet_symbol_set(
            reported_device_packet_version,
            device_packet_begin.is_some(),
            device_packet_finish.is_some(),
        )
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidData, message))?;
        let device_packet = if supports_device_packets {
            let (Some(begin), Some(finish)) = (device_packet_begin, device_packet_finish) else {
                unreachable!("validated v2 symbols must be complete")
            };
            Some(DevicePacketApi { begin, finish })
        } else {
            None
        };
        Ok(Self {
            _library: library,
            abi_version,
            create,
            destroy,
            reset,
            warmup,
            state_info,
            load_model,
            model_info,
            frontend,
            transformer,
            latent,
            decoder_checkpoint,
            process,
            batch_process,
            fixture_process,
            shared_model_create,
            shared_model_destroy,
            shared_model_load,
            shared_model_warmup,
            shared_model_info,
            shared_model_memory_info,
            session_create,
            session_destroy,
            session_reset,
            session_cancel,
            session_state_info,
            session_memory_info,
            session_process,
            device_packet,
        })
    }

    pub fn abi_version(&self) -> i32 {
        unsafe { (self.abi_version)() }
    }

    pub fn device_packet_abi_version(&self) -> Option<i32> {
        self.device_packet.map(|_| DEVICE_PACKET_ABI_VERSION)
    }

    pub fn supports_device_packets(&self) -> bool {
        self.device_packet.is_some()
    }

    pub fn create_codec(&self, device_index: i32) -> Result<Codec<'_>, String> {
        let config = Config {
            device_index,
            ring_slots: 3,
            max_packet_frames: MAX_PACKET_FRAMES as i32,
            reserved: 0,
        };
        let mut context = std::ptr::null_mut();
        let mut error = [0 as c_char; 512];
        let status =
            unsafe { (self.create)(&config, &mut context, error.as_mut_ptr(), error.len()) };
        status_result(status, &error)?;
        if context.is_null() {
            return Err("create returned a null context".to_owned());
        }
        Ok(Codec { api: self, context })
    }

    pub fn load_shared_model(
        self: &Arc<Self>,
        device_index: i32,
        weights: &dyn DecoderWeightProvider,
    ) -> Result<Arc<NativeCodecModel>, String> {
        let create = self
            .shared_model_create
            .ok_or_else(|| "shared model ABI is unavailable".to_owned())?;
        let load = self
            .shared_model_load
            .ok_or_else(|| "shared model ABI is unavailable".to_owned())?;
        let warmup = self
            .shared_model_warmup
            .ok_or_else(|| "shared model ABI is unavailable".to_owned())?;
        if self.shared_model_destroy.is_none()
            || self.shared_model_info.is_none()
            || self.shared_model_memory_info.is_none()
            || self.session_create.is_none()
            || self.session_destroy.is_none()
            || self.session_reset.is_none()
            || self.session_cancel.is_none()
            || self.session_state_info.is_none()
            || self.session_memory_info.is_none()
            || self.session_process.is_none()
        {
            return Err("shared model ABI is incomplete".to_owned());
        }

        let mut handle = std::ptr::null_mut();
        let mut error = [0 as c_char; 512];
        let status = unsafe { create(device_index, &mut handle, error.as_mut_ptr(), error.len()) };
        status_result(status, &error)?;
        if handle.is_null() {
            return Err("shared model create returned a null handle".to_owned());
        }
        let model = Arc::new(NativeCodecModel {
            api: Arc::clone(self),
            handle,
        });

        let mut provider_state = ModelProvider::new(weights)?;
        let provider = WeightProvider {
            abi_version: 1,
            reserved: 0,
            tensor_count: provider_state.names.len() as u64,
            user_data: (&mut provider_state as *mut ModelProvider<'_>).cast::<c_void>(),
            tensor_at: Some(model_tensor_at),
        };
        let mut info = ModelInfo::default();
        error.fill(0);
        let status = unsafe {
            load(
                model.handle,
                &provider,
                &mut info,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        status_result(status, &error)?;
        error.fill(0);
        let status = unsafe { warmup(model.handle, error.as_mut_ptr(), error.len()) };
        status_result(status, &error)?;
        Ok(model)
    }

    pub fn process_batch(
        &self,
        codecs: &mut [&mut Codec<'_>],
        frames: &[&[[u16; CODEBOOKS]]],
        finals: &[bool],
    ) -> Result<BatchOutput, (i32, String)> {
        if codecs.is_empty()
            || codecs.len() > MAX_BATCH_STREAMS
            || frames.len() != codecs.len()
            || finals.len() != codecs.len()
        {
            return Err((
                -1,
                "batch vectors must contain 1-6 matching items".to_owned(),
            ));
        }
        let mut pcm = frames
            .iter()
            .map(|packet| vec![0_i16; packet.len() * SAMPLES_PER_FRAME])
            .collect::<Vec<_>>();
        let mut results = vec![PacketResult::default(); codecs.len()];
        let mut items = Vec::with_capacity(codecs.len());
        for index in 0..codecs.len() {
            items.push(BatchItem {
                context: codecs[index].context,
                codec_frames: frames[index].as_ptr().cast::<u16>(),
                frame_count: frames[index].len() as u32,
                is_final: i32::from(finals[index]),
                pcm_output: pcm[index].as_mut_ptr(),
                pcm_capacity_samples: pcm[index].len(),
                result: &mut results[index],
            });
        }
        let mut error = [0 as c_char; 512];
        let status = unsafe {
            (self.batch_process)(
                items.as_mut_ptr(),
                items.len() as u32,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err((status, error_text(&error)));
        }
        Ok(pcm.into_iter().zip(results).collect())
    }
}

pub struct Codec<'a> {
    api: &'a Api,
    context: *mut Context,
}

impl Codec<'_> {
    pub fn load_model(&mut self, model: &dyn DecoderWeightProvider) -> Result<ModelInfo, String> {
        let mut provider_state = ModelProvider::new(model)?;
        let provider = WeightProvider {
            abi_version: 1,
            reserved: 0,
            tensor_count: provider_state.names.len() as u64,
            user_data: (&mut provider_state as *mut ModelProvider<'_>).cast::<c_void>(),
            tensor_at: Some(model_tensor_at),
        };
        let mut output = ModelInfo::default();
        let mut error = [0 as c_char; 512];
        let status = unsafe {
            (self.api.load_model)(
                self.context,
                &provider,
                &mut output,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        status_result(status, &error)?;
        Ok(output)
    }

    pub fn model_info(&self) -> Result<ModelInfo, String> {
        let mut output = ModelInfo::default();
        let mut error = [0 as c_char; 512];
        let status = unsafe {
            (self.api.model_info)(self.context, &mut output, error.as_mut_ptr(), error.len())
        };
        status_result(status, &error)?;
        Ok(output)
    }

    pub fn debug_frontend(
        &mut self,
        frames: &[[u16; CODEBOOKS]],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let mut rvq = vec![0.0_f32; frames.len() * 512];
        let mut preconv = vec![0.0_f32; frames.len() * 1024];
        let mut error = [0 as c_char; 512];
        let status = unsafe {
            (self.api.frontend)(
                self.context,
                frames.as_ptr().cast::<u16>(),
                frames.len() as u32,
                rvq.as_mut_ptr(),
                rvq.len(),
                preconv.as_mut_ptr(),
                preconv.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        status_result(status, &error)?;
        Ok((rvq, preconv))
    }

    pub fn debug_transformer(&mut self, frames: &[[u16; CODEBOOKS]]) -> Result<Vec<f32>, String> {
        let mut output = vec![0.0_f32; frames.len() * 1024];
        let mut error = [0 as c_char; 512];
        let status = unsafe {
            (self.api.transformer)(
                self.context,
                frames.as_ptr().cast::<u16>(),
                frames.len() as u32,
                output.as_mut_ptr(),
                output.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        status_result(status, &error)?;
        Ok(output)
    }

    pub fn debug_latent(
        &mut self,
        frames: &[[u16; CODEBOOKS]],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let mut stage_one = vec![0.0_f32; frames.len() * 2 * 1024];
        let mut stage_two = vec![0.0_f32; frames.len() * 4 * 1024];
        let mut error = [0 as c_char; 512];
        let status = unsafe {
            (self.api.latent)(
                self.context,
                frames.as_ptr().cast::<u16>(),
                frames.len() as u32,
                stage_one.as_mut_ptr(),
                stage_one.len(),
                stage_two.as_mut_ptr(),
                stage_two.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        status_result(status, &error)?;
        Ok((stage_one, stage_two))
    }

    pub fn debug_decoder_checkpoint(
        &mut self,
        frames: &[[u16; CODEBOOKS]],
        checkpoint: u32,
        output_elements: usize,
    ) -> Result<Vec<f32>, String> {
        let mut output = vec![0.0_f32; output_elements];
        let mut error = [0 as c_char; 512];
        let status = unsafe {
            (self.api.decoder_checkpoint)(
                self.context,
                frames.as_ptr().cast::<u16>(),
                frames.len() as u32,
                checkpoint,
                output.as_mut_ptr(),
                output.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        status_result(status, &error)?;
        Ok(output)
    }

    pub fn reset(&mut self) -> Result<(), String> {
        let mut error = [0 as c_char; 512];
        let status = unsafe { (self.api.reset)(self.context, error.as_mut_ptr(), error.len()) };
        status_result(status, &error)
    }

    pub fn warmup(&mut self) -> Result<(), String> {
        let mut error = [0 as c_char; 512];
        let status = unsafe { (self.api.warmup)(self.context, error.as_mut_ptr(), error.len()) };
        status_result(status, &error)
    }

    pub fn state_info(&self) -> Result<StateInfo, String> {
        let mut output = StateInfo::default();
        let mut error = [0 as c_char; 512];
        let status = unsafe {
            (self.api.state_info)(self.context, &mut output, error.as_mut_ptr(), error.len())
        };
        status_result(status, &error)?;
        Ok(output)
    }

    pub fn process(
        &mut self,
        frames: &[[u16; CODEBOOKS]],
        is_final: bool,
    ) -> Result<(Vec<i16>, PacketResult), (i32, String)> {
        let mut pcm = vec![0_i16; frames.len() * SAMPLES_PER_FRAME];
        let result = self.process_into(frames, is_final, &mut pcm)?;
        Ok((pcm, result))
    }

    pub fn process_into(
        &mut self,
        frames: &[[u16; CODEBOOKS]],
        is_final: bool,
        pcm: &mut [i16],
    ) -> Result<PacketResult, (i32, String)> {
        let mut result = PacketResult::default();
        let mut error = [0 as c_char; 512];
        let status = unsafe {
            (self.api.process)(
                self.context,
                frames.as_ptr().cast::<u16>(),
                frames.len() as u32,
                i32::from(is_final),
                pcm.as_mut_ptr(),
                pcm.len(),
                &mut result,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err((status, error_text(&error)));
        }
        Ok(result)
    }

    pub fn process_fixture(
        &mut self,
        frames: &[[u16; CODEBOOKS]],
        is_final: bool,
    ) -> Result<(Vec<i16>, PacketResult), (i32, String)> {
        let mut pcm = vec![0_i16; frames.len() * SAMPLES_PER_FRAME];
        let mut result = PacketResult::default();
        let mut error = [0 as c_char; 512];
        let status = unsafe {
            (self.api.fixture_process)(
                self.context,
                frames.as_ptr().cast::<u16>(),
                frames.len() as u32,
                i32::from(is_final),
                pcm.as_mut_ptr(),
                pcm.len(),
                &mut result,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err((status, error_text(&error)));
        }
        Ok((pcm, result))
    }
}

pub struct NativeCodecModel {
    api: Arc<Api>,
    handle: *mut SharedModelHandle,
}

// The native model is immutable after construction. Its only mutable fields
// are atomic lifetime counters and a lifecycle mutex that is never held during
// inference. Device weights are read-only across independent CUDA streams.
unsafe impl Send for NativeCodecModel {}
unsafe impl Sync for NativeCodecModel {}

impl NativeCodecModel {
    pub fn supports_device_packets(&self) -> bool {
        self.api.supports_device_packets()
    }

    pub fn model_info(&self) -> Result<ModelInfo, String> {
        let info = self
            .api
            .shared_model_info
            .ok_or_else(|| "shared model ABI is unavailable".to_owned())?;
        let mut output = ModelInfo::default();
        let mut error = [0 as c_char; 512];
        let status = unsafe { info(self.handle, &mut output, error.as_mut_ptr(), error.len()) };
        status_result(status, &error)?;
        Ok(output)
    }

    pub fn memory_info(&self) -> Result<ModelMemoryInfo, String> {
        let memory_info = self
            .api
            .shared_model_memory_info
            .ok_or_else(|| "shared model ABI is unavailable".to_owned())?;
        let mut output = ModelMemoryInfo::default();
        let mut error = [0 as c_char; 512];
        let status =
            unsafe { memory_info(self.handle, &mut output, error.as_mut_ptr(), error.len()) };
        status_result(status, &error)?;
        Ok(output)
    }

    pub fn start_session(self: &Arc<Self>) -> Result<NativeCodecSession, String> {
        let create = self
            .api
            .session_create
            .ok_or_else(|| "shared session ABI is unavailable".to_owned())?;
        let mut handle = std::ptr::null_mut();
        let mut error = [0 as c_char; 512];
        let status = unsafe { create(self.handle, &mut handle, error.as_mut_ptr(), error.len()) };
        status_result(status, &error)?;
        if handle.is_null() {
            return Err("session create returned a null handle".to_owned());
        }
        Ok(NativeCodecSession {
            model: Arc::clone(self),
            handle,
            not_sync: PhantomData,
        })
    }
}

impl Drop for NativeCodecModel {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Some(destroy) = self.api.shared_model_destroy {
            let mut error = [0 as c_char; 512];
            unsafe {
                destroy(self.handle, error.as_mut_ptr(), error.len());
            }
        }
        self.handle = std::ptr::null_mut();
    }
}

/// An owned mutable decoder stream.
///
/// Sessions are `Send + 'static`, so each one can be moved into an independent
/// worker thread. They intentionally are not `Sync`; packet, reset, and cancel
/// operations require exclusive mutable access.
///
/// ```compile_fail
/// use qwen3_tts_native_codec::NativeCodecSession;
/// fn require_sync<T: Sync>() {}
/// require_sync::<NativeCodecSession>();
/// ```
pub struct NativeCodecSession {
    model: Arc<NativeCodecModel>,
    handle: *mut SessionHandle,
    not_sync: PhantomData<Cell<()>>,
}

unsafe impl Send for NativeCodecSession {}

impl NativeCodecSession {
    pub fn model(&self) -> &Arc<NativeCodecModel> {
        &self.model
    }

    pub fn supports_device_packets(&self) -> bool {
        self.model.supports_device_packets()
    }

    pub fn reset(&mut self) -> Result<(), String> {
        let reset = self
            .model
            .api
            .session_reset
            .ok_or_else(|| "shared session ABI is unavailable".to_owned())?;
        let mut error = [0 as c_char; 512];
        let status = unsafe { reset(self.handle, error.as_mut_ptr(), error.len()) };
        status_result(status, &error)
    }

    pub fn cancel(&mut self) -> Result<(), String> {
        let cancel = self
            .model
            .api
            .session_cancel
            .ok_or_else(|| "shared session ABI is unavailable".to_owned())?;
        let mut error = [0 as c_char; 512];
        let status = unsafe { cancel(self.handle, error.as_mut_ptr(), error.len()) };
        status_result(status, &error)
    }

    pub fn state_info(&self) -> Result<StateInfo, String> {
        let state_info = self
            .model
            .api
            .session_state_info
            .ok_or_else(|| "shared session ABI is unavailable".to_owned())?;
        let mut output = StateInfo::default();
        let mut error = [0 as c_char; 512];
        let status =
            unsafe { state_info(self.handle, &mut output, error.as_mut_ptr(), error.len()) };
        status_result(status, &error)?;
        Ok(output)
    }

    pub fn memory_info(&self) -> Result<SessionMemoryInfo, String> {
        let memory_info = self
            .model
            .api
            .session_memory_info
            .ok_or_else(|| "shared session ABI is unavailable".to_owned())?;
        let mut output = SessionMemoryInfo::default();
        let mut error = [0 as c_char; 512];
        let status =
            unsafe { memory_info(self.handle, &mut output, error.as_mut_ptr(), error.len()) };
        status_result(status, &error)?;
        Ok(output)
    }

    pub fn process(
        &mut self,
        frames: &[[u16; CODEBOOKS]],
        is_final: bool,
    ) -> Result<(Vec<i16>, PacketResult), (i32, String)> {
        let mut pcm = vec![0_i16; frames.len() * SAMPLES_PER_FRAME];
        let result = self.process_into(frames, is_final, &mut pcm)?;
        Ok((pcm, result))
    }

    pub fn process_into(
        &mut self,
        frames: &[[u16; CODEBOOKS]],
        is_final: bool,
        pcm: &mut [i16],
    ) -> Result<PacketResult, (i32, String)> {
        let Some(process) = self.model.api.session_process else {
            return Err((-1, "shared session ABI is unavailable".to_owned()));
        };
        let mut result = PacketResult::default();
        let mut error = [0 as c_char; 512];
        let status = unsafe {
            process(
                self.handle,
                frames.as_ptr().cast::<u16>(),
                frames.len() as u32,
                i32::from(is_final),
                pcm.as_mut_ptr(),
                pcm.len(),
                &mut result,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err((status, error_text(&error)));
        }
        Ok(result)
    }

    /// Starts a packet from codec codes already resident on the session's CUDA
    /// device and returns without synchronizing the decoder stream.
    ///
    /// The returned guard exclusively borrows this codec session. The producer
    /// can begin its next packet after CUDA observes `codes_consumed_event`,
    /// while this session continues decoding the current packet.
    ///
    /// # Safety
    ///
    /// `device_codec_frames` must identify a contiguous CUDA device allocation
    /// containing `frame_count * CODEBOOKS` `u16` values on this session's CUDA
    /// device. `producer_ready_event` must be a valid CUDA event whose record
    /// has already been enqueued. The caller must keep that event and the source
    /// allocation alive until the guard's `codes_consumed_event` completes, or
    /// until a successful `finish`, `reset`, or `cancel` has synchronized the
    /// pending work. If cleanup reports an error, the caller must conservatively
    /// retain both resources until its own CUDA synchronization confirms safety.
    pub unsafe fn begin_device_packet(
        &mut self,
        device_codec_frames: *const u16,
        frame_count: usize,
        producer_ready_event: *mut c_void,
    ) -> Result<PendingDevicePacket<'_>, (i32, String)> {
        required_device_packet_samples(frame_count)?;
        if device_codec_frames.is_null() {
            return Err((
                STATUS_INVALID_ARGUMENT,
                "device codec frame pointer must not be null".to_owned(),
            ));
        }
        if producer_ready_event.is_null() {
            return Err((
                STATUS_INVALID_ARGUMENT,
                "producer-ready CUDA event must not be null".to_owned(),
            ));
        }
        let Some(device_packet) = self.model.api.device_packet else {
            return Err((
                STATUS_STATE,
                "device packet ABI v2 is unavailable".to_owned(),
            ));
        };

        let request =
            DevicePacketBeginV2::new(device_codec_frames, frame_count, producer_ready_event);
        let mut output = DevicePacketBeginResultV2::new();
        let mut error = [0 as c_char; 512];
        let status = unsafe {
            (device_packet.begin)(
                self.handle,
                &request,
                &mut output,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err((status, error_text(&error)));
        }

        let Some(codes_consumed_event) = NonNull::new(output.codes_consumed_event) else {
            let cleanup = self.reset().err();
            return Err((
                STATUS_STATE,
                malformed_begin_output_message("returned a null codes-consumed event", cleanup),
            ));
        };
        if output.ticket_id == 0 {
            let cleanup = self.reset().err();
            return Err((
                STATUS_STATE,
                malformed_begin_output_message("returned ticket zero", cleanup),
            ));
        }

        Ok(PendingDevicePacket {
            session: self,
            finish: device_packet.finish,
            ticket_id: output.ticket_id,
            codes_consumed_event,
            frame_count,
            active: true,
            not_send_sync: PhantomData,
        })
    }
}

/// An in-flight device-code packet tied to one exclusively borrowed session.
///
/// The guard must stay on its creating host thread because its CUDA event and
/// raw device-resource lifetime contract are thread-local integration details.
///
/// ```compile_fail
/// use qwen3_tts_native_codec::PendingDevicePacket;
/// fn require_send<T: Send>() {}
/// fn reject_send<'a>() { require_send::<PendingDevicePacket<'a>>(); }
/// ```
///
/// ```compile_fail
/// use qwen3_tts_native_codec::PendingDevicePacket;
/// fn require_sync<T: Sync>() {}
/// fn reject_sync<'a>() { require_sync::<PendingDevicePacket<'a>>(); }
/// ```
///
/// Dropping an active guard attempts a synchronous reset and then a cancel
/// fallback, but cannot report cleanup failure. Unsafe callers that need proof
/// that producer resources are reusable must wait on `codes_consumed_event` or
/// call `finish`, `reset`, or `cancel` explicitly and handle the result.
#[must_use = "finish, reset, or cancel the pending device packet explicitly"]
pub struct PendingDevicePacket<'a> {
    session: &'a mut NativeCodecSession,
    finish: SessionDevicePacketFinishFn,
    ticket_id: u64,
    codes_consumed_event: NonNull<c_void>,
    frame_count: usize,
    active: bool,
    not_send_sync: PhantomData<Rc<()>>,
}

impl PendingDevicePacket<'_> {
    pub fn ticket_id(&self) -> u64 {
        self.ticket_id
    }

    pub fn required_pcm_samples(&self) -> usize {
        self.frame_count * SAMPLES_PER_FRAME
    }

    /// Returns the session-owned CUDA event recorded immediately after the D2D
    /// codec-code snapshot. Its handle is valid while the session lives, but
    /// this ticket's recorded completion may be replaced by the next begin.
    pub fn codes_consumed_event(&self) -> *mut c_void {
        self.codes_consumed_event.as_ptr()
    }

    /// Attempts to finish without consuming the guard.
    ///
    /// A too-small PCM slice is rejected in Rust before entering the C ABI, so
    /// the caller can resize it and retry this same guard. Other native errors
    /// should be treated as terminal; dropping the guard then performs cleanup.
    pub fn try_finish(
        &mut self,
        is_final: bool,
        pcm: &mut [i16],
    ) -> Result<PacketResult, (i32, String)> {
        if !self.active {
            return Err((
                STATUS_STATE,
                "device packet is no longer pending".to_owned(),
            ));
        }
        validate_device_pcm_capacity(self.frame_count, pcm.len())?;

        let mut result = PacketResult::default();
        let request = DevicePacketFinishV2::new(
            self.ticket_id,
            is_final,
            pcm.as_mut_ptr(),
            pcm.len(),
            &mut result,
        );
        let mut error = [0 as c_char; 512];
        let status = unsafe {
            (self.finish)(
                self.session.handle,
                &request,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err((status, error_text(&error)));
        }
        self.active = false;

        let expected_samples = self.required_pcm_samples();
        if result.frame_count as usize != self.frame_count
            || result.sample_count as usize != expected_samples
        {
            return Err((
                STATUS_STATE,
                format!(
                    "device finish returned inconsistent metadata: expected {} frames/{expected_samples} samples, got {} frames/{} samples",
                    self.frame_count, result.frame_count, result.sample_count
                ),
            ));
        }
        Ok(result)
    }

    /// Finishes this ticket and releases the exclusive session borrow.
    ///
    /// Use `try_finish` when the caller wants to recover from a too-small PCM
    /// buffer without resetting the pending ticket.
    pub fn finish(
        mut self,
        is_final: bool,
        pcm: &mut [i16],
    ) -> Result<PacketResult, (i32, String)> {
        self.try_finish(is_final, pcm)
    }

    /// Cancels pending work and releases the session borrow. The session remains
    /// cancelled until its owner explicitly calls `NativeCodecSession::reset`.
    pub fn cancel(mut self) -> Result<(), String> {
        let result = self.session.cancel();
        if result.is_ok() {
            self.active = false;
        }
        result
    }

    /// Drains pending work, resets stream state, and releases the session borrow.
    pub fn reset(mut self) -> Result<(), String> {
        let result = self.session.reset();
        if result.is_ok() {
            self.active = false;
        }
        result
    }
}

impl Drop for PendingDevicePacket<'_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        if self.session.reset().is_err() {
            let _ = self.session.cancel();
        }
        self.active = false;
    }
}

impl Drop for NativeCodecSession {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Some(destroy) = self.model.api.session_destroy {
            let mut error = [0 as c_char; 512];
            unsafe {
                destroy(self.handle, error.as_mut_ptr(), error.len());
            }
        }
        self.handle = std::ptr::null_mut();
    }
}

struct ModelProvider<'a> {
    model: &'a dyn DecoderWeightProvider,
    names: Vec<CString>,
}

impl<'a> ModelProvider<'a> {
    fn new(model: &'a dyn DecoderWeightProvider) -> Result<Self, String> {
        let names = model
            .decoder_tensor_names()
            .map(|name| CString::new(name).map_err(|_| format!("tensor name contains NUL: {name}")))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { model, names })
    }
}

unsafe extern "C" fn model_tensor_at(
    user_data: *mut c_void,
    index: u64,
    name: *mut *const c_char,
    output: *mut TensorView,
    _error: *mut c_char,
    _error_capacity: usize,
) -> i32 {
    if user_data.is_null() || name.is_null() || output.is_null() {
        return STATUS_MODEL;
    }
    let provider = unsafe { &*(user_data.cast::<ModelProvider<'_>>()) };
    let Some(name_value) = provider.names.get(index as usize) else {
        return STATUS_MODEL;
    };
    let Ok(name_str) = name_value.to_str() else {
        return STATUS_MODEL;
    };
    let Some(tensor) = provider.model.decoder_tensor(name_str) else {
        return STATUS_MODEL;
    };
    if tensor.shape.len() > 4 {
        return STATUS_MODEL;
    }
    let mut shape = [0_u64; 4];
    shape[..tensor.shape.len()].copy_from_slice(tensor.shape);
    unsafe {
        *name = name_value.as_ptr();
        *output = TensorView {
            data: tensor.data.as_ptr().cast::<c_void>(),
            byte_length: tensor.data.len() as u64,
            shape,
            rank: tensor.shape.len() as u32,
            dtype: match tensor.dtype {
                TensorDType::F32 => 1,
                TensorDType::Bf16 => 2,
            },
        };
    }
    0
}

impl Drop for Codec<'_> {
    fn drop(&mut self) {
        if self.context.is_null() {
            return;
        }
        let mut error = [0 as c_char; 512];
        unsafe {
            (self.api.destroy)(self.context, error.as_mut_ptr(), error.len());
        }
        self.context = std::ptr::null_mut();
    }
}

fn required_device_packet_samples(frame_count: usize) -> Result<usize, (i32, String)> {
    if !(1..=MAX_PACKET_FRAMES).contains(&frame_count) {
        return Err((
            STATUS_INVALID_ARGUMENT,
            format!("device packet must contain 1-{MAX_PACKET_FRAMES} frames"),
        ));
    }
    frame_count.checked_mul(SAMPLES_PER_FRAME).ok_or_else(|| {
        (
            STATUS_INVALID_ARGUMENT,
            "device packet sample count overflowed usize".to_owned(),
        )
    })
}

fn validate_device_pcm_capacity(
    frame_count: usize,
    pcm_capacity_samples: usize,
) -> Result<usize, (i32, String)> {
    let required = required_device_packet_samples(frame_count)?;
    if pcm_capacity_samples < required {
        return Err((
            STATUS_INVALID_ARGUMENT,
            format!(
                "PCM output capacity is too small: need {required} samples, got {pcm_capacity_samples}"
            ),
        ));
    }
    Ok(required)
}

fn malformed_begin_output_message(reason: &str, cleanup_error: Option<String>) -> String {
    match cleanup_error {
        Some(cleanup_error) => format!(
            "device packet begin {reason}; defensive session reset also failed: {cleanup_error}"
        ),
        None => format!("device packet begin {reason}; session was reset defensively"),
    }
}

fn status_result(status: i32, error: &[c_char]) -> Result<(), String> {
    if status == 0 {
        Ok(())
    } else {
        Err(format!("status {status}: {}", error_text(error)))
    }
}

fn error_text(error: &[c_char]) -> String {
    if error.is_empty() {
        return String::new();
    }
    unsafe { CStr::from_ptr(error.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn device_packet_abi_layout_matches_c_header_on_64_bit() {
        if cfg!(target_pointer_width = "64") {
            assert_eq!(size_of::<PacketResult>(), 40);
            assert_eq!(align_of::<PacketResult>(), 8);
            assert_eq!(offset_of!(PacketResult, first_frame_position), 0);
            assert_eq!(offset_of!(PacketResult, first_sample_position), 8);
            assert_eq!(offset_of!(PacketResult, frame_count), 16);
            assert_eq!(offset_of!(PacketResult, sample_count), 20);
            assert_eq!(offset_of!(PacketResult, ring_slot), 24);
            assert_eq!(offset_of!(PacketResult, is_final), 28);
            assert_eq!(offset_of!(PacketResult, gpu_microseconds), 32);
            assert_eq!(offset_of!(PacketResult, end_to_end_microseconds), 36);

            assert_eq!(size_of::<DevicePacketBeginV2>(), 40);
            assert_eq!(align_of::<DevicePacketBeginV2>(), 8);
            assert_eq!(offset_of!(DevicePacketBeginV2, struct_size), 0);
            assert_eq!(offset_of!(DevicePacketBeginV2, reserved), 4);
            assert_eq!(offset_of!(DevicePacketBeginV2, device_codec_frames), 8);
            assert_eq!(offset_of!(DevicePacketBeginV2, frame_count), 16);
            assert_eq!(offset_of!(DevicePacketBeginV2, reserved_2), 20);
            assert_eq!(offset_of!(DevicePacketBeginV2, producer_ready_event), 24);
            assert_eq!(offset_of!(DevicePacketBeginV2, reserved_3), 32);

            assert_eq!(size_of::<DevicePacketBeginResultV2>(), 32);
            assert_eq!(align_of::<DevicePacketBeginResultV2>(), 8);
            assert_eq!(offset_of!(DevicePacketBeginResultV2, struct_size), 0);
            assert_eq!(offset_of!(DevicePacketBeginResultV2, reserved), 4);
            assert_eq!(offset_of!(DevicePacketBeginResultV2, ticket_id), 8);
            assert_eq!(
                offset_of!(DevicePacketBeginResultV2, codes_consumed_event),
                16
            );
            assert_eq!(offset_of!(DevicePacketBeginResultV2, reserved_2), 24);

            assert_eq!(size_of::<DevicePacketFinishV2>(), 56);
            assert_eq!(align_of::<DevicePacketFinishV2>(), 8);
            assert_eq!(offset_of!(DevicePacketFinishV2, struct_size), 0);
            assert_eq!(offset_of!(DevicePacketFinishV2, reserved), 4);
            assert_eq!(offset_of!(DevicePacketFinishV2, ticket_id), 8);
            assert_eq!(offset_of!(DevicePacketFinishV2, is_final), 16);
            assert_eq!(offset_of!(DevicePacketFinishV2, reserved_2), 20);
            assert_eq!(offset_of!(DevicePacketFinishV2, pcm_output), 24);
            assert_eq!(offset_of!(DevicePacketFinishV2, pcm_capacity_samples), 32);
            assert_eq!(offset_of!(DevicePacketFinishV2, result), 40);
            assert_eq!(offset_of!(DevicePacketFinishV2, reserved_3), 48);
        }
    }

    #[test]
    fn device_packet_symbol_set_is_atomic() {
        assert!(!validate_device_packet_symbol_set(None, false, false).unwrap());
        assert!(
            validate_device_packet_symbol_set(Some(DEVICE_PACKET_ABI_VERSION), true, true).unwrap()
        );

        let mismatch = validate_device_packet_symbol_set(Some(7), true, true).unwrap_err();
        assert!(mismatch.contains("version mismatch"));
        for partial in [
            (Some(DEVICE_PACKET_ABI_VERSION), false, false),
            (Some(DEVICE_PACKET_ABI_VERSION), true, false),
            (Some(DEVICE_PACKET_ABI_VERSION), false, true),
            (None, true, true),
            (None, true, false),
            (None, false, true),
        ] {
            let error =
                validate_device_packet_symbol_set(partial.0, partial.1, partial.2).unwrap_err();
            assert!(error.contains("incomplete device packet ABI v2 symbol set"));
        }
    }

    #[test]
    fn device_packet_struct_constructors_set_the_exact_contract() {
        let device_codes = NonNull::<u16>::dangling().as_ptr();
        let ready_event = NonNull::<u8>::dangling().as_ptr().cast::<c_void>();
        let begin = DevicePacketBeginV2::new(device_codes, 4, ready_event);
        assert_eq!(begin.struct_size as usize, size_of::<DevicePacketBeginV2>());
        assert_eq!(begin.reserved, 0);
        assert_eq!(begin.device_codec_frames, device_codes);
        assert_eq!(begin.frame_count, 4);
        assert_eq!(begin.reserved_2, 0);
        assert_eq!(begin.producer_ready_event, ready_event);
        assert_eq!(begin.reserved_3, 0);

        let output = DevicePacketBeginResultV2::new();
        assert_eq!(
            output.struct_size as usize,
            size_of::<DevicePacketBeginResultV2>()
        );
        assert_eq!(output.reserved, 0);
        assert_eq!(output.ticket_id, 0);
        assert!(output.codes_consumed_event.is_null());
        assert_eq!(output.reserved_2, 0);

        let mut pcm = [0_i16; 1];
        let mut result = PacketResult::default();
        let result_ptr = &mut result as *mut PacketResult;
        let finish = DevicePacketFinishV2::new(42, true, pcm.as_mut_ptr(), pcm.len(), result_ptr);
        assert_eq!(
            finish.struct_size as usize,
            size_of::<DevicePacketFinishV2>()
        );
        assert_eq!(finish.reserved, 0);
        assert_eq!(finish.ticket_id, 42);
        assert_eq!(finish.is_final, 1);
        assert_eq!(finish.reserved_2, 0);
        assert_eq!(finish.pcm_output, pcm.as_mut_ptr());
        assert_eq!(finish.pcm_capacity_samples, 1);
        assert_eq!(finish.result, result_ptr);
        assert_eq!(finish.reserved_3, 0);
    }

    #[test]
    fn pcm_capacity_rejection_is_retryable_before_ffi() {
        assert!(required_device_packet_samples(0).is_err());
        assert!(required_device_packet_samples(MAX_PACKET_FRAMES + 1).is_err());
        assert_eq!(required_device_packet_samples(1).unwrap(), 1920);
        assert_eq!(required_device_packet_samples(4).unwrap(), 7680);

        let too_small = validate_device_pcm_capacity(3, 5759).unwrap_err();
        assert_eq!(too_small.0, STATUS_INVALID_ARGUMENT);
        assert!(too_small.1.contains("need 5760 samples"));
        assert_eq!(validate_device_pcm_capacity(3, 5760).unwrap(), 5760);
    }
}
