use crate::model::{SafetensorsFile, TensorDType};
use libloading::Library;
use std::error::Error;
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::Path;

pub const CODEBOOKS: usize = 16;
pub const MAX_PACKET_FRAMES: usize = 4;
pub const SAMPLES_PER_FRAME: usize = 1920;
pub const MAX_PACKET_SAMPLES: usize = MAX_PACKET_FRAMES * SAMPLES_PER_FRAME;
pub const STATUS_STATE: i32 = -3;
pub const STATUS_MODEL: i32 = -5;

#[repr(C)]
pub struct Context {
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
#[derive(Clone, Copy, Default)]
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
#[derive(Clone, Copy, Default)]
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

type AbiVersionFn = unsafe extern "C" fn() -> i32;
type CreateFn = unsafe extern "C" fn(*const Config, *mut *mut Context, *mut c_char, usize) -> i32;
type DestroyFn = unsafe extern "C" fn(*mut Context, *mut c_char, usize) -> i32;
type ResetFn = unsafe extern "C" fn(*mut Context, *mut c_char, usize) -> i32;
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

pub struct Api {
    _library: Library,
    abi_version: AbiVersionFn,
    create: CreateFn,
    destroy: DestroyFn,
    reset: ResetFn,
    state_info: StateInfoFn,
    load_model: LoadModelFn,
    model_info: ModelInfoFn,
    frontend: FrontendFn,
    transformer: TransformerFn,
    latent: LatentFn,
    process: ProcessFn,
}

unsafe fn load_symbol<T: Copy>(library: &Library, symbol: &[u8]) -> Result<T, Box<dyn Error>> {
    let loaded = unsafe { library.get::<T>(symbol)? };
    Ok(*loaded)
}

impl Api {
    pub fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        let library = unsafe { Library::new(path)? };
        let abi_version = unsafe { load_symbol(&library, b"qwen3_tts_codec_abi_version_v1\0")? };
        let create = unsafe { load_symbol(&library, b"qwen3_tts_codec_create_v1\0")? };
        let destroy = unsafe { load_symbol(&library, b"qwen3_tts_codec_destroy_v1\0")? };
        let reset = unsafe { load_symbol(&library, b"qwen3_tts_codec_reset_v1\0")? };
        let state_info = unsafe { load_symbol(&library, b"qwen3_tts_codec_state_info_v1\0")? };
        let load_model = unsafe { load_symbol(&library, b"qwen3_tts_codec_load_model_v1\0")? };
        let model_info = unsafe { load_symbol(&library, b"qwen3_tts_codec_model_info_v1\0")? };
        let frontend =
            unsafe { load_symbol(&library, b"qwen3_tts_codec_debug_frontend_packet_v1\0")? };
        let transformer =
            unsafe { load_symbol(&library, b"qwen3_tts_codec_debug_transformer_packet_v1\0")? };
        let latent = unsafe { load_symbol(&library, b"qwen3_tts_codec_debug_latent_packet_v1\0")? };
        let process =
            unsafe { load_symbol(&library, b"qwen3_tts_codec_process_fixture_packet_v1\0")? };
        Ok(Self {
            _library: library,
            abi_version,
            create,
            destroy,
            reset,
            state_info,
            load_model,
            model_info,
            frontend,
            transformer,
            latent,
            process,
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
}

pub struct Codec<'a> {
    api: &'a Api,
    context: *mut Context,
}

impl Codec<'_> {
    pub fn load_model(&mut self, model: &SafetensorsFile) -> Result<ModelInfo, String> {
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

    pub fn reset(&mut self) -> Result<(), String> {
        let mut error = [0 as c_char; 512];
        let status = unsafe { (self.api.reset)(self.context, error.as_mut_ptr(), error.len()) };
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
        Ok((pcm, result))
    }
}

struct ModelProvider<'a> {
    model: &'a SafetensorsFile,
    names: Vec<CString>,
}

impl<'a> ModelProvider<'a> {
    fn new(model: &'a SafetensorsFile) -> Result<Self, String> {
        let names = model
            .tensor_names()
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
    let Some((entry, data)) = provider.model.tensor(name_str) else {
        return STATUS_MODEL;
    };
    if entry.shape.len() > 4 {
        return STATUS_MODEL;
    }
    let mut shape = [0_u64; 4];
    shape[..entry.shape.len()].copy_from_slice(&entry.shape);
    unsafe {
        *name = name_value.as_ptr();
        *output = TensorView {
            data: data.as_ptr().cast::<c_void>(),
            byte_length: data.len() as u64,
            shape,
            rank: entry.shape.len() as u32,
            dtype: match entry.dtype {
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
