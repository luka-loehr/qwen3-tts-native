use crate::model::{DecoderWeightProvider, TensorDType};
use libloading::Library;
use std::cell::Cell;
use std::error::Error;
use std::ffi::{CStr, CString, c_char, c_void};
use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;

pub const CODEBOOKS: usize = 16;
pub const MAX_PACKET_FRAMES: usize = 4;
pub const SAMPLES_PER_FRAME: usize = 1920;
pub const MAX_PACKET_SAMPLES: usize = MAX_PACKET_FRAMES * SAMPLES_PER_FRAME;
pub const MAX_BATCH_STREAMS: usize = 6;
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
        })
    }

    pub fn abi_version(&self) -> i32 {
        unsafe { (self.abi_version)() }
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
