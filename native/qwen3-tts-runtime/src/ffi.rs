use std::env;
use std::ffi::{CStr, OsStr, c_char, c_void};
use std::fmt;
use std::mem::{align_of, zeroed};
use std::os::unix::ffi::OsStrExt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;
use std::time::Duration;

use crate::ffi_support::{
    FfiError, RequestInputV1, clear_error_buffer, copy_engine_config, copy_model_root,
    copy_request_input, write_error_buffer,
};
use crate::{
    AudioPacketDescriptor, BackendError, EngineConfig, NativeBackend, PollError, PollOutcome,
    RequestHandle, RequestMetrics, RuntimeStatus, SAMPLES_PER_CODEC_FRAME, Scheduler,
    SchedulerError,
};

const LIBRARY_DIRECTORY_ENV: &str = "QWEN3_TTS_LIBRARY_DIR";
const TALKER_LIBRARY_FILENAME: &str = "libqwen3_tts_cuda.so";
const CODEC_LIBRARY_FILENAME: &str = "libqwen3_tts_codec_cuda.so";
const REQUEST_RETIRE_TIMEOUT: Duration = Duration::from_secs(30);

type RuntimeScheduler = Scheduler<NativeBackend>;

struct EngineCore {
    scheduler: RuntimeScheduler,
    config: EngineConfig,
}

#[repr(C)]
pub struct Qwen3TtsEngineV1 {
    core: Arc<EngineCore>,
}

#[repr(C)]
pub struct Qwen3TtsRequestV1 {
    request: RequestHandle,
    _engine: Arc<EngineCore>,
}

#[derive(Debug)]
struct CallError {
    status: RuntimeStatus,
    message: String,
}

impl CallError {
    fn new(status: RuntimeStatus, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn invalid_argument(message: impl Into<String>) -> Self {
        Self::new(RuntimeStatus::InvalidArgument, message)
    }
}

impl fmt::Display for CallError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl From<FfiError> for CallError {
    fn from(error: FfiError) -> Self {
        Self::new(error.status, error.message)
    }
}

impl From<BackendError> for CallError {
    fn from(error: BackendError) -> Self {
        Self::new(error.status(), error.message())
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qwen3_tts_engine_create_v1(
    model_root_utf8: *const u8,
    model_root_bytes: usize,
    config: *const EngineConfig,
    output: *mut *mut Qwen3TtsEngineV1,
    error: *mut c_char,
    error_capacity: usize,
) -> i32 {
    ffi_call(error, error_capacity, || {
        unsafe { prepare_handle_output(output, "engine output pointer is null or misaligned")? };
        let model_root = unsafe { copy_model_root(model_root_utf8, model_root_bytes)? };
        let config = unsafe { copy_engine_config(config)? };
        let library_directory = component_library_directory()?;
        let backend = NativeBackend::load(
            &library_directory.join(TALKER_LIBRARY_FILENAME),
            &library_directory.join(CODEC_LIBRARY_FILENAME),
            Path::new(&model_root),
            config.device_index,
        )?;
        let scheduler = RuntimeScheduler::new(config, backend).map_err(map_scheduler_error)?;
        let engine = Box::new(Qwen3TtsEngineV1 {
            core: Arc::new(EngineCore { scheduler, config }),
        });
        unsafe { output.write(Box::into_raw(engine)) };
        Ok(RuntimeStatus::Ok)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qwen3_tts_engine_destroy_v1(
    engine: *mut Qwen3TtsEngineV1,
    error: *mut c_char,
    error_capacity: usize,
) -> i32 {
    ffi_call(error, error_capacity, || {
        unsafe { validate_mut_handle(engine, "engine pointer is null or misaligned")? };
        drop(unsafe { Box::from_raw(engine) });
        Ok(RuntimeStatus::Ok)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qwen3_tts_request_start_v1(
    engine: *mut Qwen3TtsEngineV1,
    input: *const RequestInputV1,
    output: *mut *mut Qwen3TtsRequestV1,
    error: *mut c_char,
    error_capacity: usize,
) -> i32 {
    ffi_call(error, error_capacity, || {
        unsafe { prepare_handle_output(output, "request output pointer is null or misaligned")? };
        let engine = unsafe { borrow_handle(engine, "engine pointer is null or misaligned")? };
        let (input, generation) = unsafe { copy_request_input(input, &engine.core.config)? };
        let request = engine
            .core
            .scheduler
            .start(input, generation)
            .map_err(map_scheduler_error)?;
        let request = Box::new(Qwen3TtsRequestV1 {
            request,
            _engine: Arc::clone(&engine.core),
        });
        unsafe { output.write(Box::into_raw(request)) };
        Ok(RuntimeStatus::Ok)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qwen3_tts_request_poll_v1(
    request: *mut Qwen3TtsRequestV1,
    timeout_milliseconds: u32,
    pcm_output: *mut i16,
    pcm_capacity_samples: usize,
    packet: *mut AudioPacketDescriptor,
    error: *mut c_char,
    error_capacity: usize,
) -> i32 {
    ffi_call(error, error_capacity, || {
        let request = unsafe { borrow_handle(request, "request pointer is null or misaligned")? };
        unsafe { prepare_value_output(packet, "packet output pointer is null or misaligned")? };
        let required_capacity = (request._engine.config.packet_frames as usize)
            .checked_mul(SAMPLES_PER_CODEC_FRAME as usize)
            .ok_or_else(|| CallError::invalid_argument("configured PCM capacity overflowed"))?;
        if pcm_capacity_samples < required_capacity {
            return Err(CallError::invalid_argument(
                "PCM output capacity is smaller than the configured packet size",
            ));
        }
        unsafe { validate_pcm_output(pcm_output, pcm_capacity_samples)? };

        match request
            .request
            .poll(Duration::from_millis(u64::from(timeout_milliseconds)))
        {
            Ok(PollOutcome::Packet(audio)) => {
                let samples = audio.pcm();
                unsafe {
                    ptr::copy_nonoverlapping(samples.as_ptr(), pcm_output, samples.len());
                    packet.write(audio.descriptor);
                }
                Ok(RuntimeStatus::Ok)
            }
            Ok(PollOutcome::WouldBlock) => Ok(RuntimeStatus::WouldBlock),
            Ok(PollOutcome::EndOfStream) => Ok(RuntimeStatus::EndOfStream),
            Err(PollError::Cancelled) => Err(CallError::new(
                RuntimeStatus::Cancelled,
                "request was cancelled",
            )),
            Err(PollError::Failed(error)) => Err(error.into()),
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qwen3_tts_request_cancel_v1(
    request: *mut Qwen3TtsRequestV1,
    error: *mut c_char,
    error_capacity: usize,
) -> i32 {
    ffi_call(error, error_capacity, || {
        let request = unsafe { borrow_handle(request, "request pointer is null or misaligned")? };
        request.request.cancel().map_err(map_scheduler_error)?;
        Ok(RuntimeStatus::Ok)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qwen3_tts_request_metrics_v1(
    request: *const Qwen3TtsRequestV1,
    output: *mut RequestMetrics,
    error: *mut c_char,
    error_capacity: usize,
) -> i32 {
    ffi_call(error, error_capacity, || {
        let request =
            unsafe { borrow_const_handle(request, "request pointer is null or misaligned")? };
        unsafe { prepare_value_output(output, "metrics output pointer is null or misaligned")? };
        unsafe { output.write(request.request.metrics()) };
        Ok(RuntimeStatus::Ok)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn qwen3_tts_request_destroy_v1(
    request: *mut Qwen3TtsRequestV1,
    error: *mut c_char,
    error_capacity: usize,
) -> i32 {
    ffi_call(error, error_capacity, || {
        unsafe { validate_mut_handle(request, "request pointer is null or misaligned")? };
        let request = unsafe { Box::from_raw(request) };
        let retired = request
            .request
            .cancel_and_wait(REQUEST_RETIRE_TIMEOUT)
            .map_err(map_scheduler_error)?;
        drop(request);
        if !retired {
            return Err(CallError::new(
                RuntimeStatus::State,
                "request cancellation did not retire within 30 seconds",
            ));
        }
        Ok(RuntimeStatus::Ok)
    })
}

fn ffi_call(
    error: *mut c_char,
    error_capacity: usize,
    operation: impl FnOnce() -> Result<RuntimeStatus, CallError>,
) -> i32 {
    if !unsafe { clear_error_buffer(error, error_capacity) } {
        return RuntimeStatus::InvalidArgument as i32;
    }
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(status)) => status as i32,
        Ok(Err(call_error)) => {
            unsafe { write_error_buffer(error, error_capacity, &call_error.message) };
            call_error.status as i32
        }
        Err(_) => {
            unsafe { write_error_buffer(error, error_capacity, "runtime FFI call panicked") };
            RuntimeStatus::Internal as i32
        }
    }
}

fn component_library_directory() -> Result<PathBuf, CallError> {
    if let Some(directory) = env::var_os(LIBRARY_DIRECTORY_ENV) {
        if directory.is_empty() {
            return Err(CallError::invalid_argument(format!(
                "{LIBRARY_DIRECTORY_ENV} must not be empty"
            )));
        }
        return Ok(PathBuf::from(directory));
    }
    runtime_library_directory().map_err(|message| CallError::new(RuntimeStatus::Model, message))
}

fn runtime_library_directory() -> Result<PathBuf, String> {
    let mut information = unsafe { zeroed::<libc::Dl_info>() };
    let symbol = crate::qwen3_tts_runtime_abi_version_v1 as *const () as *const c_void;
    let found = unsafe { libc::dladdr(symbol, &mut information) };
    if found == 0 || information.dli_fname.is_null() {
        return Err("failed to locate the runtime shared library with dladdr".to_owned());
    }
    let bytes = unsafe { CStr::from_ptr(information.dli_fname) }.to_bytes();
    let path = PathBuf::from(OsStr::from_bytes(bytes));
    path.parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "runtime shared library has no parent directory".to_owned())
}

fn map_scheduler_error(error: SchedulerError) -> CallError {
    let status = match error {
        SchedulerError::InvalidConfiguration(_)
        | SchedulerError::InvalidGeneration(_)
        | SchedulerError::InvalidInput(_) => RuntimeStatus::InvalidArgument,
        SchedulerError::Full | SchedulerError::Closed => RuntimeStatus::State,
        SchedulerError::Worker(_) => RuntimeStatus::Internal,
    };
    CallError::new(status, error.to_string())
}

unsafe fn prepare_handle_output<T>(
    output: *mut *mut T,
    message: &'static str,
) -> Result<(), CallError> {
    validate_mut_pointer(output, message)?;
    unsafe { output.write(ptr::null_mut()) };
    Ok(())
}

unsafe fn prepare_value_output<T: Default>(
    output: *mut T,
    message: &'static str,
) -> Result<(), CallError> {
    validate_mut_pointer(output, message)?;
    unsafe { output.write(T::default()) };
    Ok(())
}

unsafe fn borrow_handle<'a, T>(pointer: *mut T, message: &'static str) -> Result<&'a T, CallError> {
    validate_mut_handle(pointer, message)?;
    Ok(unsafe { &*pointer })
}

unsafe fn borrow_const_handle<'a, T>(
    pointer: *const T,
    message: &'static str,
) -> Result<&'a T, CallError> {
    validate_const_pointer(pointer, message)?;
    Ok(unsafe { &*pointer })
}

unsafe fn validate_mut_handle<T>(pointer: *mut T, message: &'static str) -> Result<(), CallError> {
    validate_mut_pointer(pointer, message)
}

fn validate_mut_pointer<T>(pointer: *mut T, message: &'static str) -> Result<(), CallError> {
    validate_const_pointer(pointer.cast_const(), message)
}

fn validate_const_pointer<T>(pointer: *const T, message: &'static str) -> Result<(), CallError> {
    if pointer.is_null() || !(pointer as usize).is_multiple_of(align_of::<T>()) {
        return Err(CallError::invalid_argument(message));
    }
    Ok(())
}

unsafe fn validate_pcm_output(pointer: *mut i16, capacity: usize) -> Result<(), CallError> {
    if capacity == 0 {
        return Err(CallError::invalid_argument(
            "PCM output capacity must not be zero",
        ));
    }
    validate_mut_pointer(pointer, "PCM output pointer is null or misaligned")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequestInputError;

    #[test]
    fn scheduler_errors_map_to_stable_public_statuses() {
        let cases = [
            (
                SchedulerError::InvalidConfiguration("invalid"),
                RuntimeStatus::InvalidArgument,
            ),
            (
                SchedulerError::InvalidGeneration("invalid"),
                RuntimeStatus::InvalidArgument,
            ),
            (
                SchedulerError::InvalidInput(RequestInputError::EmptyText),
                RuntimeStatus::InvalidArgument,
            ),
            (SchedulerError::Full, RuntimeStatus::State),
            (SchedulerError::Closed, RuntimeStatus::State),
            (
                SchedulerError::Worker("failed".to_owned()),
                RuntimeStatus::Internal,
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(map_scheduler_error(error).status, expected);
        }
    }

    #[test]
    fn invalid_error_buffer_contract_returns_invalid_argument() {
        let mut byte = 1 as c_char;
        assert_eq!(
            ffi_call(&mut byte, 0, || Ok(RuntimeStatus::Ok)),
            RuntimeStatus::InvalidArgument as i32
        );
        assert_eq!(byte, 1 as c_char);
    }

    #[test]
    fn panic_is_contained_at_the_ffi_boundary() {
        let mut error = [0 as c_char; 64];
        assert_eq!(
            ffi_call(error.as_mut_ptr(), error.len(), || panic!("boom")),
            RuntimeStatus::Internal as i32
        );
        let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
        assert_eq!(message, "runtime FFI call panicked");
    }
}
