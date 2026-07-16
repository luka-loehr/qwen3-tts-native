use libloading::Library;
use std::error::Error;
use std::ffi::{CStr, c_char};
use std::path::Path;

pub const CODEBOOKS: usize = 16;
pub const MAX_PACKET_FRAMES: usize = 4;
pub const SAMPLES_PER_FRAME: usize = 1920;
pub const MAX_PACKET_SAMPLES: usize = MAX_PACKET_FRAMES * SAMPLES_PER_FRAME;
pub const STATUS_STATE: i32 = -3;

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

pub struct Api {
    _library: Library,
    abi_version: AbiVersionFn,
    create: CreateFn,
    destroy: DestroyFn,
    reset: ResetFn,
    state_info: StateInfoFn,
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
        let process =
            unsafe { load_symbol(&library, b"qwen3_tts_codec_process_fixture_packet_v1\0")? };
        Ok(Self {
            _library: library,
            abi_version,
            create,
            destroy,
            reset,
            state_info,
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
