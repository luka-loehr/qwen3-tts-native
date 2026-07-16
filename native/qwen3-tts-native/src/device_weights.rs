use std::ffi::{CStr, c_char, c_void};
use std::path::Path;
use std::ptr::NonNull;

use anyhow::{Context, Result, bail};
use libloading::Library;

const ERROR_CAPACITY: usize = 512;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct RawUploadMetrics {
    device_index: i32,
    reserved: i32,
    allocation_bytes: u64,
    pinned_staging_bytes: u64,
    uploaded_bytes: u64,
    upload_calls: u64,
    free_before_bytes: u64,
    free_after_allocation_bytes: u64,
    allocation_microseconds: f32,
    upload_microseconds: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct WeightUploadMetrics {
    pub device_index: i32,
    pub allocation_bytes: u64,
    pub pinned_staging_bytes: u64,
    pub uploaded_bytes: u64,
    pub upload_calls: u64,
    pub free_before_bytes: u64,
    pub free_after_allocation_bytes: u64,
    pub allocation_microseconds: f32,
    pub upload_microseconds: f32,
}

impl From<RawUploadMetrics> for WeightUploadMetrics {
    fn from(raw: RawUploadMetrics) -> Self {
        Self {
            device_index: raw.device_index,
            allocation_bytes: raw.allocation_bytes,
            pinned_staging_bytes: raw.pinned_staging_bytes,
            uploaded_bytes: raw.uploaded_bytes,
            upload_calls: raw.upload_calls,
            free_before_bytes: raw.free_before_bytes,
            free_after_allocation_bytes: raw.free_after_allocation_bytes,
            allocation_microseconds: raw.allocation_microseconds,
            upload_microseconds: raw.upload_microseconds,
        }
    }
}

type Create = unsafe extern "C" fn(
    i32,
    u64,
    u64,
    *mut *mut c_void,
    *mut RawUploadMetrics,
    *mut c_char,
    usize,
) -> i32;
type Upload = unsafe extern "C" fn(*mut c_void, u64, *const c_void, u64, *mut c_char, usize) -> i32;
type Finish = unsafe extern "C" fn(*mut c_void, *mut RawUploadMetrics, *mut c_char, usize) -> i32;
type Read = unsafe extern "C" fn(*mut c_void, u64, *mut c_void, u64, *mut c_char, usize) -> i32;
type Data = unsafe extern "C" fn(*const c_void) -> *const c_void;
type Destroy = unsafe extern "C" fn(*mut c_void);

pub struct DeviceWeightBuffer {
    handle: NonNull<c_void>,
    upload_fn: Upload,
    finish_fn: Finish,
    read_fn: Read,
    data_fn: Data,
    destroy_fn: Destroy,
    allocation_metrics: WeightUploadMetrics,
    finished: bool,
    _library: Library,
}

impl DeviceWeightBuffer {
    pub fn create(
        library_path: &Path,
        device_index: i32,
        allocation_bytes: u64,
        staging_bytes: u64,
    ) -> Result<Self> {
        // SAFETY: Symbol signatures are pinned by qwen3_tts_native.h and the
        // loaded Library remains owned by this object for the handle lifetime.
        let library = unsafe { Library::new(library_path) }
            .with_context(|| format!("failed to load {}", library_path.display()))?;
        let create = unsafe {
            *library
                .get::<Create>(b"qwen3_tts_device_buffer_create\0")
                .context("missing qwen3_tts_device_buffer_create symbol")?
        };
        let upload_fn = unsafe {
            *library
                .get::<Upload>(b"qwen3_tts_device_buffer_upload\0")
                .context("missing qwen3_tts_device_buffer_upload symbol")?
        };
        let finish_fn = unsafe {
            *library
                .get::<Finish>(b"qwen3_tts_device_buffer_finish\0")
                .context("missing qwen3_tts_device_buffer_finish symbol")?
        };
        let read_fn = unsafe {
            *library
                .get::<Read>(b"qwen3_tts_device_buffer_read\0")
                .context("missing qwen3_tts_device_buffer_read symbol")?
        };
        let data_fn = unsafe {
            *library
                .get::<Data>(b"qwen3_tts_device_buffer_data\0")
                .context("missing qwen3_tts_device_buffer_data symbol")?
        };
        let destroy_fn = unsafe {
            *library
                .get::<Destroy>(b"qwen3_tts_device_buffer_destroy\0")
                .context("missing qwen3_tts_device_buffer_destroy symbol")?
        };

        let mut handle = std::ptr::null_mut();
        let mut metrics = RawUploadMetrics::default();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: All output pointers and capacities are valid for the call.
        let status = unsafe {
            create(
                device_index,
                allocation_bytes,
                staging_bytes,
                &mut handle,
                &mut metrics,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ensure_success(status, &error)?;
        let handle = NonNull::new(handle).context("CUDA create returned a null handle")?;
        Ok(Self {
            handle,
            upload_fn,
            finish_fn,
            read_fn,
            data_fn,
            destroy_fn,
            allocation_metrics: metrics.into(),
            finished: false,
            _library: library,
        })
    }

    pub const fn allocation_metrics(&self) -> WeightUploadMetrics {
        self.allocation_metrics
    }

    pub fn device_pointer(&self) -> NonNull<c_void> {
        // SAFETY: The buffer handle is valid until Drop.
        let pointer = unsafe { (self.data_fn)(self.handle.as_ptr()) };
        NonNull::new(pointer.cast_mut()).expect("valid device buffer has a non-null pointer")
    }

    pub fn upload(&mut self, offset_bytes: u64, source: &[u8]) -> Result<()> {
        if self.finished {
            bail!("cannot upload after device buffer was finalized");
        }
        let bytes = u64::try_from(source.len()).context("upload length does not fit u64")?;
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: The native call synchronously copies the complete source
        // slice into its owned pinned buffer before returning.
        let status = unsafe {
            (self.upload_fn)(
                self.handle.as_ptr(),
                offset_bytes,
                source.as_ptr().cast(),
                bytes,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ensure_success(status, &error)
    }

    pub fn finish(&mut self) -> Result<WeightUploadMetrics> {
        let mut metrics = RawUploadMetrics::default();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: The handle and output storage remain valid for the call.
        let status = unsafe {
            (self.finish_fn)(
                self.handle.as_ptr(),
                &mut metrics,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ensure_success(status, &error)?;
        self.finished = true;
        Ok(metrics.into())
    }

    pub fn readback(&self, offset_bytes: u64, bytes: usize) -> Result<Vec<u8>> {
        let mut output = vec![0_u8; bytes];
        let bytes = u64::try_from(bytes).context("readback length does not fit u64")?;
        let mut error = [0 as c_char; ERROR_CAPACITY];
        // SAFETY: The destination is valid for bytes and the native call
        // synchronizes before returning.
        let status = unsafe {
            (self.read_fn)(
                self.handle.as_ptr(),
                offset_bytes,
                output.as_mut_ptr().cast(),
                bytes,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ensure_success(status, &error)?;
        Ok(output)
    }
}

impl Drop for DeviceWeightBuffer {
    fn drop(&mut self) {
        // SAFETY: This object uniquely owns the handle and destroys it once.
        unsafe { (self.destroy_fn)(self.handle.as_ptr()) };
    }
}

fn ensure_success(status: i32, error: &[c_char; ERROR_CAPACITY]) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let message = if error[0] == 0 {
        format!("native CUDA operation failed with status {status}")
    } else {
        // SAFETY: The native ABI always NUL-terminates this fixed buffer.
        unsafe { CStr::from_ptr(error.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    };
    bail!("{message} (status {status})")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_metrics_layout_matches_c_abi_contract() {
        assert_eq!(std::mem::size_of::<RawUploadMetrics>(), 64);
        assert_eq!(std::mem::align_of::<RawUploadMetrics>(), 8);
    }

    #[test]
    fn native_error_without_text_is_still_actionable() {
        let error = [0 as c_char; ERROR_CAPACITY];
        let message = ensure_success(-7, &error).unwrap_err().to_string();
        assert!(message.contains("status -7"));
    }
}
