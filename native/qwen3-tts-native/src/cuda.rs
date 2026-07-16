use std::ffi::{CStr, OsString};
use std::fs;
use std::os::raw::{c_char, c_int};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use libloading::{Library, Symbol};
use serde_json::{Value, json};

const ERROR_CAPACITY: usize = 512;
const DEFAULT_DEVICE: i32 = 0;
const DEFAULT_VOCABULARY: i32 = 3_072;
const DEFAULT_ITERATIONS: i32 = 10_000;
const DEFAULT_INPUT_FEATURES: i32 = 2_048;
const DEFAULT_OUTPUT_FEATURES: i32 = 6_144;

#[repr(C)]
struct DeviceInfo {
    device_index: c_int,
    compute_major: c_int,
    compute_minor: c_int,
    total_global_memory_bytes: u64,
    runtime_free_memory_bytes: u64,
    runtime_total_memory_bytes: u64,
    device_name: [c_char; 256],
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            device_index: 0,
            compute_major: 0,
            compute_minor: 0,
            total_global_memory_bytes: 0,
            runtime_free_memory_bytes: 0,
            runtime_total_memory_bytes: 0,
            device_name: [0; 256],
        }
    }
}

#[repr(C)]
#[derive(Default)]
struct ArgmaxBenchmark {
    vocabulary_size: c_int,
    iterations: c_int,
    selected_token: c_int,
    expected_token: c_int,
    cold_launch_microseconds: f32,
    mean_launch_microseconds: f32,
}

#[repr(C)]
#[derive(Default)]
struct GemvBenchmark {
    input_features: c_int,
    output_features: c_int,
    iterations: c_int,
    reserved: c_int,
    weight_bytes: u64,
    cold_launch_microseconds: f32,
    mean_launch_microseconds: f32,
    tera_operations_per_second: f32,
}

#[repr(C)]
#[derive(Default)]
struct PrimitiveParity {
    rms_norm_max_absolute_error: f32,
    rope_max_absolute_error: f32,
    attention_max_absolute_error: f32,
    silu_gate_max_absolute_error: f32,
}

type ProbeDevice = unsafe extern "C" fn(c_int, *mut DeviceInfo, *mut c_char, usize) -> c_int;

type BenchmarkArgmax =
    unsafe extern "C" fn(c_int, c_int, c_int, *mut ArgmaxBenchmark, *mut c_char, usize) -> c_int;

type BenchmarkGemv = unsafe extern "C" fn(
    c_int,
    c_int,
    c_int,
    c_int,
    *mut GemvBenchmark,
    *mut c_char,
    usize,
) -> c_int;

type ValidateTransformerPrimitives =
    unsafe extern "C" fn(c_int, *mut PrimitiveParity, *mut c_char, usize) -> c_int;

pub fn run_probe(mut arguments: impl Iterator<Item = OsString>) -> Result<()> {
    let library_path = PathBuf::from(
        arguments
            .next()
            .context("probe-cuda requires a shared-library path")?,
    );
    let mut device = DEFAULT_DEVICE;
    let mut output_path = None;

    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--device") => device = parse_i32(&mut arguments, "--device")?,
            Some("--output") => output_path = Some(next_path(&mut arguments, "--output")?),
            _ => bail!("unknown probe-cuda argument {argument:?}"),
        }
    }

    let report = unsafe { probe_device(&library_path, device)? };
    emit_report(report, output_path.as_deref())
}

pub fn run_argmax_benchmark(mut arguments: impl Iterator<Item = OsString>) -> Result<()> {
    let library_path = PathBuf::from(
        arguments
            .next()
            .context("benchmark-argmax requires a shared-library path")?,
    );
    let mut device = DEFAULT_DEVICE;
    let mut vocabulary = DEFAULT_VOCABULARY;
    let mut iterations = DEFAULT_ITERATIONS;
    let mut output_path = None;

    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--device") => device = parse_i32(&mut arguments, "--device")?,
            Some("--vocabulary") => {
                vocabulary = parse_i32(&mut arguments, "--vocabulary")?;
            }
            Some("--iterations") => {
                iterations = parse_i32(&mut arguments, "--iterations")?;
            }
            Some("--output") => output_path = Some(next_path(&mut arguments, "--output")?),
            _ => bail!("unknown benchmark-argmax argument {argument:?}"),
        }
    }

    let report = unsafe { benchmark_argmax(&library_path, device, vocabulary, iterations)? };
    emit_report(report, output_path.as_deref())
}

pub fn run_gemv_benchmark(mut arguments: impl Iterator<Item = OsString>) -> Result<()> {
    let library_path = PathBuf::from(
        arguments
            .next()
            .context("benchmark-gemv requires a shared-library path")?,
    );
    let mut device = DEFAULT_DEVICE;
    let mut input_features = DEFAULT_INPUT_FEATURES;
    let mut output_features = DEFAULT_OUTPUT_FEATURES;
    let mut iterations = DEFAULT_ITERATIONS;
    let mut output_path = None;

    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--device") => device = parse_i32(&mut arguments, "--device")?,
            Some("--input-features") => {
                input_features = parse_i32(&mut arguments, "--input-features")?;
            }
            Some("--output-features") => {
                output_features = parse_i32(&mut arguments, "--output-features")?;
            }
            Some("--iterations") => {
                iterations = parse_i32(&mut arguments, "--iterations")?;
            }
            Some("--output") => output_path = Some(next_path(&mut arguments, "--output")?),
            _ => bail!("unknown benchmark-gemv argument {argument:?}"),
        }
    }

    let report = unsafe {
        benchmark_gemv(
            &library_path,
            device,
            input_features,
            output_features,
            iterations,
        )?
    };
    emit_report(report, output_path.as_deref())
}

pub fn run_primitive_validation(mut arguments: impl Iterator<Item = OsString>) -> Result<()> {
    let library_path = PathBuf::from(
        arguments
            .next()
            .context("validate-transformer-primitives requires a shared-library path")?,
    );
    let mut device = DEFAULT_DEVICE;
    let mut output_path = None;

    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--device") => device = parse_i32(&mut arguments, "--device")?,
            Some("--output") => output_path = Some(next_path(&mut arguments, "--output")?),
            _ => bail!("unknown validate-transformer-primitives argument {argument:?}"),
        }
    }

    let report = unsafe { validate_transformer_primitives(&library_path, device)? };
    emit_report(report, output_path.as_deref())
}

unsafe fn probe_device(library_path: &Path, device: i32) -> Result<Value> {
    let library = unsafe { Library::new(library_path) }
        .with_context(|| format!("failed to load {}", library_path.display()))?;
    let probe: Symbol<'_, ProbeDevice> = unsafe { library.get(b"qwen3_tts_probe_device\0") }
        .context("missing qwen3_tts_probe_device symbol")?;

    let mut output = DeviceInfo::default();
    let mut error = [0 as c_char; ERROR_CAPACITY];
    let status = unsafe { probe(device, &mut output, error.as_mut_ptr(), error.len()) };
    ensure_success(status, &error)?;

    let device_name = unsafe { CStr::from_ptr(output.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    Ok(json!({
        "schema_version": 1,
        "operation": "probe-cuda",
        "device_index": output.device_index,
        "device_name": device_name,
        "compute_capability": format!(
            "{}.{}",
            output.compute_major,
            output.compute_minor
        ),
        "total_global_memory_bytes": output.total_global_memory_bytes,
        "runtime_free_memory_bytes": output.runtime_free_memory_bytes,
        "runtime_total_memory_bytes": output.runtime_total_memory_bytes,
    }))
}

unsafe fn benchmark_argmax(
    library_path: &Path,
    device: i32,
    vocabulary: i32,
    iterations: i32,
) -> Result<Value> {
    let library = unsafe { Library::new(library_path) }
        .with_context(|| format!("failed to load {}", library_path.display()))?;
    let benchmark: Symbol<'_, BenchmarkArgmax> =
        unsafe { library.get(b"qwen3_tts_benchmark_bf16_argmax\0") }
            .context("missing qwen3_tts_benchmark_bf16_argmax symbol")?;

    let mut output = ArgmaxBenchmark::default();
    let mut error = [0 as c_char; ERROR_CAPACITY];
    let status = unsafe {
        benchmark(
            device,
            vocabulary,
            iterations,
            &mut output,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    ensure_success(status, &error)?;

    Ok(json!({
        "schema_version": 1,
        "operation": "benchmark-bf16-argmax",
        "device_index": device,
        "vocabulary_size": output.vocabulary_size,
        "iterations": output.iterations,
        "selected_token": output.selected_token,
        "expected_token": output.expected_token,
        "cold_launch_microseconds": output.cold_launch_microseconds,
        "mean_launch_microseconds": output.mean_launch_microseconds,
    }))
}

unsafe fn benchmark_gemv(
    library_path: &Path,
    device: i32,
    input_features: i32,
    output_features: i32,
    iterations: i32,
) -> Result<Value> {
    let library = unsafe { Library::new(library_path) }
        .with_context(|| format!("failed to load {}", library_path.display()))?;
    let benchmark: Symbol<'_, BenchmarkGemv> =
        unsafe { library.get(b"qwen3_tts_benchmark_bf16_gemv\0") }
            .context("missing qwen3_tts_benchmark_bf16_gemv symbol")?;

    let mut output = GemvBenchmark::default();
    let mut error = [0 as c_char; ERROR_CAPACITY];
    let status = unsafe {
        benchmark(
            device,
            input_features,
            output_features,
            iterations,
            &mut output,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    ensure_success(status, &error)?;

    Ok(json!({
        "schema_version": 1,
        "operation": "benchmark-bf16-gemv",
        "device_index": device,
        "input_features": output.input_features,
        "output_features": output.output_features,
        "iterations": output.iterations,
        "weight_bytes": output.weight_bytes,
        "cold_launch_microseconds": output.cold_launch_microseconds,
        "mean_launch_microseconds": output.mean_launch_microseconds,
        "tera_operations_per_second": output.tera_operations_per_second,
    }))
}

unsafe fn validate_transformer_primitives(library_path: &Path, device: i32) -> Result<Value> {
    let library = unsafe { Library::new(library_path) }
        .with_context(|| format!("failed to load {}", library_path.display()))?;
    let validate: Symbol<'_, ValidateTransformerPrimitives> =
        unsafe { library.get(b"qwen3_tts_validate_transformer_primitives\0") }
            .context("missing qwen3_tts_validate_transformer_primitives symbol")?;

    let mut output = PrimitiveParity::default();
    let mut error = [0 as c_char; ERROR_CAPACITY];
    let status = unsafe { validate(device, &mut output, error.as_mut_ptr(), error.len()) };
    ensure_success(status, &error)?;

    Ok(json!({
        "schema_version": 1,
        "operation": "validate-transformer-primitives",
        "device_index": device,
        "rms_norm_max_absolute_error": output.rms_norm_max_absolute_error,
        "rope_max_absolute_error": output.rope_max_absolute_error,
        "attention_max_absolute_error": output.attention_max_absolute_error,
        "silu_gate_max_absolute_error": output.silu_gate_max_absolute_error,
    }))
}

fn ensure_success(status: i32, error: &[c_char]) -> Result<()> {
    if status == 0 {
        return Ok(());
    }

    let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
    bail!("native CUDA call failed with status {status}: {message}")
}

fn parse_i32(arguments: &mut impl Iterator<Item = OsString>, flag: &str) -> Result<i32> {
    let value = arguments
        .next()
        .with_context(|| format!("{flag} requires an integer"))?;
    value
        .to_str()
        .with_context(|| format!("{flag} is not valid UTF-8"))?
        .parse::<i32>()
        .with_context(|| format!("{flag} must be a signed 32-bit integer"))
}

fn next_path(arguments: &mut impl Iterator<Item = OsString>, flag: &str) -> Result<PathBuf> {
    arguments
        .next()
        .map(PathBuf::from)
        .with_context(|| format!("{flag} requires a path"))
}

fn emit_report(report: Value, output_path: Option<&Path>) -> Result<()> {
    let encoded = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output_path {
        fs::write(path, encoded)
            .with_context(|| format!("failed to write report to {}", path.display()))?;
    } else {
        println!("{encoded}");
    }
    Ok(())
}
