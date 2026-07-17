use std::env;
use std::ffi::{CStr, OsString, c_char, c_int, c_void};
use std::ptr;
use std::sync::Arc;

use libloading::{Library, Symbol};
use qwen3_tts_native_codec::{CODEBOOKS, MAX_PACKET_FRAMES};

const CUDA_SUCCESS: c_int = 0;
const CUDA_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;
const CUDA_STREAM_NON_BLOCKING: u32 = 1;
const CUDA_EVENT_DISABLE_TIMING: u32 = 2;
const CUDA_13_MINIMUM_RUNTIME_VERSION: c_int = 13_000;
const PACKET_BYTES: usize = CODEBOOKS * MAX_PACKET_FRAMES * size_of::<u16>();

type CudaSetDevice = unsafe extern "C" fn(c_int) -> c_int;
type CudaMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
type CudaFree = unsafe extern "C" fn(*mut c_void) -> c_int;
type CudaStreamCreateWithFlags = unsafe extern "C" fn(*mut *mut c_void, u32) -> c_int;
type CudaStreamDestroy = unsafe extern "C" fn(*mut c_void) -> c_int;
type CudaStreamSynchronize = unsafe extern "C" fn(*mut c_void) -> c_int;
type CudaStreamWaitEvent = unsafe extern "C" fn(*mut c_void, *mut c_void, u32) -> c_int;
type CudaEventCreateWithFlags = unsafe extern "C" fn(*mut *mut c_void, u32) -> c_int;
type CudaEventDestroy = unsafe extern "C" fn(*mut c_void) -> c_int;
type CudaEventRecord = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
type CudaMemcpyAsync =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int, *mut c_void) -> c_int;
type CudaGetErrorString = unsafe extern "C" fn(c_int) -> *const c_char;
type CudaRuntimeGetVersion = unsafe extern "C" fn(*mut c_int) -> c_int;

pub(crate) struct CudaRuntime {
    _library: Library,
    set_device: CudaSetDevice,
    malloc: CudaMalloc,
    free: CudaFree,
    stream_create_with_flags: CudaStreamCreateWithFlags,
    stream_destroy: CudaStreamDestroy,
    stream_synchronize: CudaStreamSynchronize,
    stream_wait_event: CudaStreamWaitEvent,
    event_create_with_flags: CudaEventCreateWithFlags,
    event_destroy: CudaEventDestroy,
    event_record: CudaEventRecord,
    memcpy_async: CudaMemcpyAsync,
    get_error_string: CudaGetErrorString,
}

impl CudaRuntime {
    pub(crate) fn load() -> Result<Arc<Self>, String> {
        let candidates = cudart_candidates();
        let mut failures = Vec::new();
        for candidate in candidates {
            let library = match unsafe { Library::new(&candidate) } {
                Ok(library) => library,
                Err(error) => {
                    failures.push(format!("{}: {error}", candidate.to_string_lossy()));
                    continue;
                }
            };
            match unsafe { Self::from_library(library) } {
                Ok(runtime) => return Ok(Arc::new(runtime)),
                Err(error) => failures.push(format!("{}: {error}", candidate.to_string_lossy())),
            }
        }
        Err(format!(
            "failed to load CUDA runtime ({})",
            failures.join("; ")
        ))
    }

    unsafe fn from_library(library: Library) -> Result<Self, String> {
        let set_device = unsafe { load_symbol(&library, b"cudaSetDevice\0")? };
        let malloc = unsafe { load_symbol(&library, b"cudaMalloc\0")? };
        let free = unsafe { load_symbol(&library, b"cudaFree\0")? };
        let stream_create_with_flags =
            unsafe { load_symbol(&library, b"cudaStreamCreateWithFlags\0")? };
        let stream_destroy = unsafe { load_symbol(&library, b"cudaStreamDestroy\0")? };
        let stream_synchronize = unsafe { load_symbol(&library, b"cudaStreamSynchronize\0")? };
        let stream_wait_event = unsafe { load_symbol(&library, b"cudaStreamWaitEvent\0")? };
        let event_create_with_flags =
            unsafe { load_symbol(&library, b"cudaEventCreateWithFlags\0")? };
        let event_destroy = unsafe { load_symbol(&library, b"cudaEventDestroy\0")? };
        let event_record = unsafe { load_symbol(&library, b"cudaEventRecord\0")? };
        let memcpy_async = unsafe { load_symbol(&library, b"cudaMemcpyAsync\0")? };
        let get_error_string = unsafe { load_symbol(&library, b"cudaGetErrorString\0")? };
        let runtime_get_version =
            unsafe { load_symbol::<CudaRuntimeGetVersion>(&library, b"cudaRuntimeGetVersion\0")? };

        let mut runtime_version = 0;
        let status = unsafe { runtime_get_version(&mut runtime_version) };
        if status != CUDA_SUCCESS {
            return Err(format!(
                "cudaRuntimeGetVersion failed with CUDA status {status}"
            ));
        }
        if runtime_version < CUDA_13_MINIMUM_RUNTIME_VERSION {
            return Err(format!(
                "CUDA runtime {runtime_version} is older than required CUDA 13.0"
            ));
        }

        Ok(Self {
            _library: library,
            set_device,
            malloc,
            free,
            stream_create_with_flags,
            stream_destroy,
            stream_synchronize,
            stream_wait_event,
            event_create_with_flags,
            event_destroy,
            event_record,
            memcpy_async,
            get_error_string,
        })
    }

    fn check(&self, operation: &str, status: c_int) -> Result<(), String> {
        if status == CUDA_SUCCESS {
            return Ok(());
        }
        let message = unsafe {
            let pointer = (self.get_error_string)(status);
            if pointer.is_null() {
                format!("CUDA status {status}")
            } else {
                CStr::from_ptr(pointer).to_string_lossy().into_owned()
            }
        };
        Err(format!("{operation}: {message} ({status})"))
    }

    fn select_device(&self, device_index: c_int) -> Result<(), String> {
        self.check("select CUDA packet-stager device", unsafe {
            (self.set_device)(device_index)
        })
    }
}

unsafe fn load_symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, String> {
    let symbol: Symbol<'_, T> = unsafe { library.get(name) }.map_err(|error| {
        format!(
            "missing CUDA runtime symbol {}: {error}",
            String::from_utf8_lossy(name).trim_end_matches('\0')
        )
    })?;
    Ok(*symbol)
}

fn cudart_candidates() -> Vec<OsString> {
    let mut candidates = Vec::new();
    if let Some(path) = env::var_os("QWEN3_TTS_CUDART_LIBRARY") {
        candidates.push(path);
    }
    candidates.push(OsString::from("libcudart.so.13"));
    candidates.push(OsString::from("libcudart.so"));
    candidates
}

#[derive(Clone, Copy)]
pub(crate) struct StagedPacketView {
    pub(crate) device_codes: *const u16,
    pub(crate) frame_count: usize,
    pub(crate) ready_event: *mut c_void,
}

pub(crate) struct CudaPacketStager {
    runtime: Arc<CudaRuntime>,
    device_index: c_int,
    device_codes: *mut c_void,
    stream: *mut c_void,
    copied_event: *mut c_void,
    staged_frames: usize,
    poisoned: bool,
}

// Every session owns its stager exclusively. Moving it between scheduler
// workers is safe; concurrent access is prevented by Rust's mutable borrow.
unsafe impl Send for CudaPacketStager {}

impl CudaPacketStager {
    pub(crate) fn new(runtime: Arc<CudaRuntime>, device_index: c_int) -> Result<Self, String> {
        runtime.select_device(device_index)?;
        let mut stager = Self {
            runtime,
            device_index,
            device_codes: ptr::null_mut(),
            stream: ptr::null_mut(),
            copied_event: ptr::null_mut(),
            staged_frames: 0,
            poisoned: false,
        };
        let status = unsafe { (stager.runtime.malloc)(&mut stager.device_codes, PACKET_BYTES) };
        stager
            .runtime
            .check("allocate CUDA packet staging buffer", status)?;
        let status = unsafe {
            (stager.runtime.stream_create_with_flags)(&mut stager.stream, CUDA_STREAM_NON_BLOCKING)
        };
        stager
            .runtime
            .check("create CUDA packet staging stream", status)?;
        let status = unsafe {
            (stager.runtime.event_create_with_flags)(
                &mut stager.copied_event,
                CUDA_EVENT_DISABLE_TIMING,
            )
        };
        stager
            .runtime
            .check("create CUDA packet staging event", status)?;
        Ok(stager)
    }

    pub(crate) fn device_bytes(&self) -> u64 {
        PACKET_BYTES as u64
    }

    pub(crate) fn staged_frames(&self) -> usize {
        self.staged_frames
    }

    /// Copies one Talker frame into the next packet slot without host staging.
    ///
    /// # Safety
    ///
    /// Both CUDA handles must be live on `source_device`, and `device_codes`
    /// must expose at least `CODEBOOKS` u16 values until the returned event has
    /// completed.
    pub(crate) unsafe fn stage_frame(
        &mut self,
        device_codes: *const u16,
        code_count: usize,
        producer_ready_event: *mut c_void,
        source_device: c_int,
    ) -> Result<*mut c_void, String> {
        if self.poisoned {
            return Err("CUDA packet stager is poisoned".to_owned());
        }
        if source_device != self.device_index {
            return Err(format!(
                "Talker device {source_device} does not match stager device {}",
                self.device_index
            ));
        }
        if device_codes.is_null() || producer_ready_event.is_null() {
            return Err("Talker device codes and ready event are required".to_owned());
        }
        if code_count != CODEBOOKS {
            return Err(format!(
                "Talker returned {code_count} codes; expected {CODEBOOKS}"
            ));
        }
        if self.staged_frames >= MAX_PACKET_FRAMES {
            return Err("CUDA packet staging buffer is full".to_owned());
        }
        self.runtime.select_device(self.device_index)?;
        let wait_status =
            unsafe { (self.runtime.stream_wait_event)(self.stream, producer_ready_event, 0) };
        if let Err(error) = self.runtime.check("wait for Talker frame", wait_status) {
            self.poisoned = true;
            return Err(error);
        }
        let byte_offset = self.staged_frames * CODEBOOKS * size_of::<u16>();
        let destination = unsafe { self.device_codes.cast::<u8>().add(byte_offset) }.cast();
        let copy_status = unsafe {
            (self.runtime.memcpy_async)(
                destination,
                device_codes.cast(),
                CODEBOOKS * size_of::<u16>(),
                CUDA_MEMCPY_DEVICE_TO_DEVICE,
                self.stream,
            )
        };
        if let Err(error) = self.runtime.check("stage Talker codec frame", copy_status) {
            self.poisoned = true;
            return Err(error);
        }
        let event_status = unsafe { (self.runtime.event_record)(self.copied_event, self.stream) };
        if let Err(error) = self
            .runtime
            .check("record staged Talker frame", event_status)
        {
            self.poisoned = true;
            return Err(error);
        }
        self.staged_frames += 1;
        Ok(self.copied_event)
    }

    pub(crate) fn packet_view(&self) -> Result<StagedPacketView, String> {
        if self.poisoned {
            return Err("CUDA packet stager is poisoned".to_owned());
        }
        if self.staged_frames == 0 {
            return Err("CUDA packet staging buffer is empty".to_owned());
        }
        Ok(StagedPacketView {
            device_codes: self.device_codes.cast(),
            frame_count: self.staged_frames,
            ready_event: self.copied_event,
        })
    }

    /// Transfers buffer ownership back to the stager after the Codec snapshot.
    /// Subsequent frame copies are ordered behind `consumer_done_event`.
    ///
    /// # Safety
    ///
    /// The event must be live, recorded on the same CUDA device, and cover the
    /// Codec's final read from `packet_view().device_codes`.
    pub(crate) unsafe fn release_after_consumer(
        &mut self,
        consumer_done_event: *mut c_void,
    ) -> Result<(), String> {
        if self.poisoned {
            return Err("CUDA packet stager is poisoned".to_owned());
        }
        if self.staged_frames == 0 {
            return Err("no staged packet is awaiting a consumer".to_owned());
        }
        if consumer_done_event.is_null() {
            return Err("Codec consume event is required".to_owned());
        }
        self.runtime.select_device(self.device_index)?;
        let status =
            unsafe { (self.runtime.stream_wait_event)(self.stream, consumer_done_event, 0) };
        if let Err(error) = self.runtime.check("wait for Codec packet snapshot", status) {
            self.poisoned = true;
            return Err(error);
        }
        self.staged_frames = 0;
        Ok(())
    }
}

impl Drop for CudaPacketStager {
    fn drop(&mut self) {
        let _ = self.runtime.select_device(self.device_index);
        if !self.stream.is_null() {
            unsafe {
                (self.runtime.stream_synchronize)(self.stream);
            }
        }
        if !self.copied_event.is_null() {
            unsafe {
                (self.runtime.event_destroy)(self.copied_event);
            }
            self.copied_event = ptr::null_mut();
        }
        if !self.stream.is_null() {
            unsafe {
                (self.runtime.stream_destroy)(self.stream);
            }
            self.stream = ptr::null_mut();
        }
        if !self.device_codes.is_null() {
            unsafe {
                (self.runtime.free)(self.device_codes);
            }
            self.device_codes = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_stager_owns_exactly_four_codec_frames() {
        assert_eq!(CODEBOOKS, 16);
        assert_eq!(MAX_PACKET_FRAMES, 4);
        assert_eq!(PACKET_BYTES, 128);
    }

    #[test]
    fn cuda_runtime_candidates_prefer_an_explicit_override_when_present() {
        let candidates = cudart_candidates();
        assert!(candidates.iter().any(|path| path == "libcudart.so.13"));
        assert!(candidates.iter().any(|path| path == "libcudart.so"));
    }
}
