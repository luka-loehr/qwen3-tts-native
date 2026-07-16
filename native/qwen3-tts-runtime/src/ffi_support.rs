use std::ffi::c_char;
use std::mem::{align_of, size_of};
use std::ptr;
use std::slice;
use std::str;

use crate::{
    EngineConfig, GenerationConfig, Language, MAX_TEXT_BYTES, RequestInput, RuntimeStatus,
};

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RequestInputV1 {
    pub struct_size: u32,
    pub language: u32,
    pub text_utf8: *const u8,
    pub text_bytes: usize,
    pub instruct_utf8: *const u8,
    pub instruct_bytes: usize,
    pub generation: GenerationConfig,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FfiError {
    pub status: RuntimeStatus,
    pub message: &'static str,
}

impl FfiError {
    const fn invalid_argument(message: &'static str) -> Self {
        Self {
            status: RuntimeStatus::InvalidArgument,
            message,
        }
    }
}

pub(crate) unsafe fn copy_engine_config(
    pointer: *const EngineConfig,
) -> Result<EngineConfig, FfiError> {
    unsafe { copy_versioned_struct(pointer, "engine config pointer is null or misaligned") }
}

pub(crate) unsafe fn copy_request_input(
    pointer: *const RequestInputV1,
    config: &EngineConfig,
) -> Result<(RequestInput, GenerationConfig), FfiError> {
    let raw =
        unsafe { copy_versioned_struct(pointer, "request input pointer is null or misaligned")? };
    if raw.generation.struct_size != size_of::<GenerationConfig>() as u32 {
        return Err(FfiError::invalid_argument(
            "generation struct_size does not match ABI v1",
        ));
    }
    if raw.text_bytes > config.max_text_bytes as usize {
        return Err(FfiError::invalid_argument(
            "text exceeds configured byte limit",
        ));
    }
    if raw.instruct_bytes > config.max_instruct_bytes as usize {
        return Err(FfiError::invalid_argument(
            "instruction exceeds configured byte limit",
        ));
    }
    let language = language_from_raw(raw.language)?;
    let text = unsafe { copy_utf8(raw.text_utf8, raw.text_bytes, "text")? };
    if text.is_empty() {
        return Err(FfiError::invalid_argument("text must not be empty"));
    }
    let instruct = unsafe { copy_utf8(raw.instruct_utf8, raw.instruct_bytes, "instruction")? };
    Ok((
        RequestInput {
            text,
            instruct,
            language,
        },
        raw.generation,
    ))
}

pub(crate) unsafe fn copy_model_root(pointer: *const u8, bytes: usize) -> Result<String, FfiError> {
    if bytes > MAX_TEXT_BYTES as usize {
        return Err(FfiError::invalid_argument(
            "model root exceeds the supported byte limit",
        ));
    }
    let model_root = unsafe { copy_utf8(pointer, bytes, "model root")? };
    if model_root.is_empty() {
        return Err(FfiError::invalid_argument("model root must not be empty"));
    }
    if model_root.as_bytes().contains(&0) {
        return Err(FfiError::invalid_argument(
            "model root must not contain embedded NUL bytes",
        ));
    }
    Ok(model_root)
}

pub(crate) fn language_from_raw(value: u32) -> Result<Language, FfiError> {
    match value {
        0 => Ok(Language::Auto),
        1 => Ok(Language::Chinese),
        2 => Ok(Language::English),
        3 => Ok(Language::Japanese),
        4 => Ok(Language::Korean),
        5 => Ok(Language::German),
        6 => Ok(Language::French),
        7 => Ok(Language::Russian),
        8 => Ok(Language::Portuguese),
        9 => Ok(Language::Spanish),
        10 => Ok(Language::Italian),
        _ => Err(FfiError {
            status: RuntimeStatus::UnsupportedLanguage,
            message: "language is not supported by Qwen3-TTS VoiceDesign",
        }),
    }
}

pub(crate) unsafe fn clear_error_buffer(pointer: *mut c_char, capacity: usize) -> bool {
    if capacity == 0 {
        return pointer.is_null();
    }
    if pointer.is_null() {
        return false;
    }
    unsafe { pointer.write(0) };
    true
}

pub(crate) unsafe fn write_error_buffer(pointer: *mut c_char, capacity: usize, message: &str) {
    if pointer.is_null() || capacity == 0 {
        return;
    }
    let maximum = capacity - 1;
    let mut end = message.len().min(maximum);
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    unsafe {
        ptr::copy_nonoverlapping(message.as_ptr(), pointer.cast::<u8>(), end);
        pointer.add(end).write(0);
    }
}

unsafe fn copy_versioned_struct<T: Copy>(
    pointer: *const T,
    pointer_message: &'static str,
) -> Result<T, FfiError> {
    if pointer.is_null() || !(pointer as usize).is_multiple_of(align_of::<T>()) {
        return Err(FfiError::invalid_argument(pointer_message));
    }
    let struct_size = unsafe { pointer.cast::<u32>().read() };
    if struct_size != size_of::<T>() as u32 {
        return Err(FfiError::invalid_argument(
            "struct_size does not match ABI v1",
        ));
    }
    Ok(unsafe { pointer.read() })
}

unsafe fn copy_utf8(
    pointer: *const u8,
    bytes: usize,
    field: &'static str,
) -> Result<String, FfiError> {
    if bytes > isize::MAX as usize {
        return Err(FfiError::invalid_argument(
            "UTF-8 input length is too large",
        ));
    }
    if bytes == 0 {
        return Ok(String::new());
    }
    if pointer.is_null() {
        return Err(FfiError::invalid_argument(match field {
            "text" => "text pointer is null for a non-empty input",
            "instruction" => "instruction pointer is null for a non-empty input",
            _ => "model root pointer is null for a non-empty input",
        }));
    }
    let input = unsafe { slice::from_raw_parts(pointer, bytes) };
    let decoded = str::from_utf8(input).map_err(|_| FfiError {
        status: RuntimeStatus::InvalidUtf8,
        message: "input is not valid UTF-8",
    })?;
    Ok(decoded.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_request(text: &[u8], instruct: &[u8]) -> RequestInputV1 {
        RequestInputV1 {
            struct_size: size_of::<RequestInputV1>() as u32,
            language: Language::German as u32,
            text_utf8: text.as_ptr(),
            text_bytes: text.len(),
            instruct_utf8: instruct.as_ptr(),
            instruct_bytes: instruct.len(),
            generation: GenerationConfig::default(),
        }
    }

    #[test]
    fn request_input_v1_layout_matches_the_c_header() {
        assert_eq!(size_of::<RequestInputV1>(), 160);
        assert_eq!(align_of::<RequestInputV1>(), 8);
        assert_eq!(std::mem::offset_of!(RequestInputV1, text_utf8), 8);
        assert_eq!(std::mem::offset_of!(RequestInputV1, text_bytes), 16);
        assert_eq!(std::mem::offset_of!(RequestInputV1, instruct_utf8), 24);
        assert_eq!(std::mem::offset_of!(RequestInputV1, instruct_bytes), 32);
        assert_eq!(std::mem::offset_of!(RequestInputV1, generation), 40);
    }

    #[test]
    fn request_input_is_copied_and_raw_language_is_checked() {
        let text = "Guten Morgen".as_bytes();
        let instruct = "A calm voice".as_bytes();
        let raw = raw_request(text, instruct);
        let (input, generation) =
            unsafe { copy_request_input(&raw, &EngineConfig::default()) }.unwrap();
        assert_eq!(input.text, "Guten Morgen");
        assert_eq!(input.instruct, "A calm voice");
        assert_eq!(input.language, Language::German);
        assert_eq!(generation, GenerationConfig::default());

        let mut invalid = raw;
        invalid.language = 11;
        let error = unsafe { copy_request_input(&invalid, &EngineConfig::default()) }.unwrap_err();
        assert_eq!(error.status, RuntimeStatus::UnsupportedLanguage);
    }

    #[test]
    fn invalid_utf8_and_null_positive_length_are_rejected() {
        let invalid_utf8 = [0xff_u8];
        let mut raw = raw_request(&invalid_utf8, &[]);
        let error = unsafe { copy_request_input(&raw, &EngineConfig::default()) }.unwrap_err();
        assert_eq!(error.status, RuntimeStatus::InvalidUtf8);

        raw.text_utf8 = ptr::null();
        let error = unsafe { copy_request_input(&raw, &EngineConfig::default()) }.unwrap_err();
        assert_eq!(error.status, RuntimeStatus::InvalidArgument);
    }

    #[test]
    fn error_buffer_is_bounded_utf8_safe_and_nul_terminated() {
        let mut capacity_one = [9 as c_char; 3];
        unsafe { write_error_buffer(capacity_one.as_mut_ptr(), 1, "failure") };
        assert_eq!(capacity_one, [0 as c_char, 9 as c_char, 9 as c_char]);

        let mut truncated = [9 as c_char; 8];
        unsafe { write_error_buffer(truncated.as_mut_ptr(), 5, "échec") };
        assert_eq!(truncated[4], 0 as c_char);
        assert_eq!(truncated[5..], [9 as c_char, 9 as c_char, 9 as c_char]);
        let bytes = truncated[..4]
            .iter()
            .map(|value| *value as u8)
            .collect::<Vec<_>>();
        assert!(str::from_utf8(&bytes).is_ok());
    }

    #[test]
    fn zero_capacity_error_buffer_accepts_only_null() {
        assert!(unsafe { clear_error_buffer(ptr::null_mut(), 0) });
        let mut byte = 7 as c_char;
        assert!(!unsafe { clear_error_buffer(&mut byte, 0) });
        assert_eq!(byte, 7 as c_char);
    }
}
