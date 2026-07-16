mod ffi;
mod model;
mod reference;

use ffi::{Api, CODEBOOKS, MAX_PACKET_FRAMES, MAX_PACKET_SAMPLES, STATUS_STATE};
use reference::ReferenceState;
use serde_json::{Value, json};
use std::error::Error;
use std::fs;
use std::io;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args().collect::<Vec<_>>();
    let command = arguments.get(1).map(String::as_str).unwrap_or("help");
    match command {
        "parity" => {
            let library = required_argument(&arguments, 2, "shared library path")?;
            parity(Path::new(library))
        }
        "benchmark" => {
            let library = required_argument(&arguments, 2, "shared library path")?;
            let iterations = arguments
                .get(3)
                .map(|value| value.parse::<usize>())
                .transpose()?
                .unwrap_or(200);
            benchmark(Path::new(library), iterations)
        }
        "inspect-model" => {
            let model = required_argument(&arguments, 2, "speech tokenizer safetensors path")?;
            inspect_model(Path::new(model))
        }
        "load-model" => {
            let library = required_argument(&arguments, 2, "shared library path")?;
            let model = required_argument(&arguments, 3, "speech tokenizer safetensors path")?;
            load_model(Path::new(library), Path::new(model))
        }
        "frontend-parity" => {
            let library = required_argument(&arguments, 2, "shared library path")?;
            let model = required_argument(&arguments, 3, "speech tokenizer safetensors path")?;
            let fixture = required_argument(&arguments, 4, "decoder fixture directory")?;
            frontend_parity(Path::new(library), Path::new(model), Path::new(fixture))
        }
        _ => {
            eprintln!(
                "usage: qwen3-tts-native-codec <parity|benchmark> <library> [iterations]\n\
                 or: qwen3-tts-native-codec inspect-model <speech-tokenizer.safetensors>\n\
                 or: qwen3-tts-native-codec load-model <library> <speech-tokenizer.safetensors>"
            );
            Ok(())
        }
    }
}

fn frontend_parity(
    library_path: &Path,
    model_path: &Path,
    fixture_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let codes = read_u16_le(&fixture_path.join("codes.u16le"))?;
    if codes.len() % ffi::CODEBOOKS != 0 {
        return Err(io::Error::other("fixture code count is not divisible by 16").into());
    }
    let frames = codes
        .chunks_exact(ffi::CODEBOOKS)
        .map(|values| values.try_into().expect("chunk contains exactly 16 codes"))
        .collect::<Vec<[u16; ffi::CODEBOOKS]>>();
    let expected_rvq = read_f32_le(&fixture_path.join("01-rvq.f32le"))?;
    let expected_preconv = read_f32_le(&fixture_path.join("02-pre-conv.f32le"))?;

    let api = Api::load(library_path)?;
    let model = model::SafetensorsFile::open(model_path)?;
    let mut codec = api.create_codec(0).map_err(io::Error::other)?;
    codec.load_model(&model).map_err(io::Error::other)?;
    let (actual_rvq, actual_preconv) = codec.debug_frontend(&frames).map_err(io::Error::other)?;
    let rvq = compare_f32(&actual_rvq, &expected_rvq)?;
    let preconv = compare_f32(&actual_preconv, &expected_preconv)?;
    let passed = rvq.maximum_absolute_error <= 1.0e-4 && preconv.maximum_absolute_error <= 1.0e-5;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "mode": "real_qwen_decoder_frontend",
            "passed": passed,
            "frames": frames.len(),
            "rvq": rvq.to_json(),
            "preconv": preconv.to_json(),
            "thresholds": {
                "rvq_maximum_absolute_error": 1.0e-4,
                "preconv_maximum_absolute_error": 1.0e-5
            }
        }))?
    );
    if !passed {
        return Err(io::Error::other("native decoder frontend parity failed").into());
    }
    Ok(())
}

struct Comparison {
    count: usize,
    maximum_absolute_error: f64,
    root_mean_squared_error: f64,
    maximum_error_index: usize,
    actual_at_maximum: f32,
    expected_at_maximum: f32,
}

impl Comparison {
    fn to_json(&self) -> Value {
        json!({
            "count": self.count,
            "maximum_absolute_error": self.maximum_absolute_error,
            "root_mean_squared_error": self.root_mean_squared_error
            ,"maximum_error_index": self.maximum_error_index
            ,"actual_at_maximum": self.actual_at_maximum
            ,"expected_at_maximum": self.expected_at_maximum
        })
    }
}

fn compare_f32(actual: &[f32], expected: &[f32]) -> Result<Comparison, Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err(io::Error::other(format!(
            "checkpoint lengths differ: {} != {}",
            actual.len(),
            expected.len()
        ))
        .into());
    }
    let mut maximum_absolute_error = 0.0_f64;
    let mut maximum_error_index = 0_usize;
    let mut squared_error = 0.0_f64;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = f64::from(*actual) - f64::from(*expected);
        if error.abs() > maximum_absolute_error {
            maximum_absolute_error = error.abs();
            maximum_error_index = index;
        }
        squared_error += error * error;
    }
    Ok(Comparison {
        count: actual.len(),
        maximum_absolute_error,
        root_mean_squared_error: (squared_error / actual.len() as f64).sqrt(),
        maximum_error_index,
        actual_at_maximum: actual[maximum_error_index],
        expected_at_maximum: expected[maximum_error_index],
    })
}

fn read_u16_le(path: &Path) -> Result<Vec<u16>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 2 != 0 {
        return Err(io::Error::other("u16 fixture has a partial scalar").into());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().expect("two bytes")))
        .collect())
}

fn read_f32_le(path: &Path) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(io::Error::other("f32 fixture has a partial scalar").into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect())
}

fn load_model(library_path: &Path, model_path: &Path) -> Result<(), Box<dyn Error>> {
    let api = Api::load(library_path)?;
    let model = model::SafetensorsFile::open(model_path)?;
    let mut codec = api.create_codec(0).map_err(io::Error::other)?;
    let loaded = codec.load_model(&model).map_err(io::Error::other)?;
    let queried = codec.model_info().map_err(io::Error::other)?;
    if loaded.tensor_count != 271
        || loaded.parameter_count != 114_323_137
        || loaded.loaded != 1
        || loaded.source_bytes != queried.source_bytes
        || loaded.device_bytes != queried.device_bytes
    {
        return Err(io::Error::other("native model info does not match decoder contract").into());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "loaded": loaded.loaded == 1,
            "tensor_count": loaded.tensor_count,
            "parameter_count": loaded.parameter_count,
            "source_bytes": loaded.source_bytes,
            "device_bytes": loaded.device_bytes,
            "source_dtype_counts": {
                "F32": loaded.source_dtype_f32_count,
                "BF16": loaded.source_dtype_bf16_count
            },
            "ownership": "all decoder tensors copied into native CUDA allocations",
            "runtime_dependency": "CUDA runtime and Rust host; no Python or Node.js"
        }))?
    );
    Ok(())
}

fn inspect_model(path: &Path) -> Result<(), Box<dyn Error>> {
    let model = model::SafetensorsFile::open(path)?;
    let (f32_tensors, bf16_tensors) = model.decoder_dtype_counts();
    let required = [
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum",
        "decoder.pre_transformer.layers.0.self_attn.q_proj.weight",
        "decoder.decoder.6.conv.weight",
    ];
    for name in required {
        if model.tensor(name).is_none() {
            return Err(io::Error::other(format!("missing required tensor {name}")).into());
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "format": "safetensors",
            "tensor_count": model.tensor_count(),
            "decoder_tensor_count": model.decoder_tensor_count(),
            "decoder_payload_bytes": model.decoder_payload_bytes(),
            "decoder_dtype_counts": { "F32": f32_tensors, "BF16": bf16_tensors },
            "accepted_dtypes": ["F32", "BF16"],
            "runtime_dependency": "Rust standard library and serde_json; no Python or Node.js"
        }))?
    );
    Ok(())
}

fn parity(library_path: &Path) -> Result<(), Box<dyn Error>> {
    let api = Api::load(library_path)?;
    if api.abi_version() != 1 {
        return Err(io::Error::other("unexpected codec ABI version").into());
    }
    let frames = deterministic_frames(83);
    let mut reference_state = ReferenceState::new();
    let reference = reference_state.process(&frames);
    let mut codec = api.create_codec(0).map_err(io::Error::other)?;
    let initial_state = codec.state_info().map_err(io::Error::other)?;

    let packet_pattern = [1_usize, 4, 2, 3, 4, 1, 3, 2];
    let mut incremental = Vec::with_capacity(reference.len());
    let mut boundaries = Vec::new();
    let mut packet_results = Vec::new();
    let mut position = 0;
    let mut pattern_position = 0;
    while position < frames.len() {
        let requested = packet_pattern[pattern_position % packet_pattern.len()];
        let count = requested.min(frames.len() - position);
        let is_final = position + count == frames.len();
        let (pcm, result) = codec
            .process(&frames[position..position + count], is_final)
            .map_err(|(status, message)| {
                io::Error::other(format!("process status {status}: {message}"))
            })?;
        incremental.extend_from_slice(&pcm);
        position += count;
        pattern_position += 1;
        packet_results.push(result);
        if !is_final {
            boundaries.push(incremental.len());
        }
    }

    let final_state = codec.state_info().map_err(io::Error::other)?;
    let post_final_rejection = match codec.process(&frames[..1], false) {
        Err((status, _)) => status == STATUS_STATE,
        Ok(_) => false,
    };
    codec.reset().map_err(io::Error::other)?;
    let reset_state = codec.state_info().map_err(io::Error::other)?;

    let sample_count_matches = incremental.len() == reference.len();
    let mut maximum_absolute_error = 0_i32;
    let mut squared_error_sum = 0_f64;
    let mut signal_power_sum = 0_f64;
    for (actual, expected) in incremental.iter().zip(&reference) {
        let difference = i32::from(*actual) - i32::from(*expected);
        maximum_absolute_error = maximum_absolute_error.max(difference.abs());
        squared_error_sum += f64::from(difference).powi(2);
        signal_power_sum += f64::from(*expected).powi(2);
    }
    let mean_squared_error = if reference.is_empty() {
        0.0
    } else {
        squared_error_sum / reference.len() as f64
    };
    let snr_db = if squared_error_sum == 0.0 {
        None
    } else {
        Some(10.0 * (signal_power_sum / squared_error_sum).log10())
    };
    let seam_maximum_absolute_error = boundaries
        .iter()
        .flat_map(|boundary| boundary.saturating_sub(1)..=(*boundary + 1).min(reference.len() - 1))
        .map(|index| (i32::from(incremental[index]) - i32::from(reference[index])).abs())
        .max()
        .unwrap_or(0);
    let boundary_delta_error = boundaries
        .iter()
        .map(|boundary| {
            let actual_delta =
                i32::from(incremental[*boundary]) - i32::from(incremental[*boundary - 1]);
            let expected_delta =
                i32::from(reference[*boundary]) - i32::from(reference[*boundary - 1]);
            (actual_delta - expected_delta).abs()
        })
        .max()
        .unwrap_or(0);

    let positions_match = packet_results.iter().enumerate().all(|(index, result)| {
        let expected_first_frame = packet_results[..index]
            .iter()
            .map(|prior| u64::from(prior.frame_count))
            .sum::<u64>();
        result.first_frame_position == expected_first_frame
            && result.first_sample_position == expected_first_frame * 1920
            && result.sample_count == result.frame_count * 1920
            && result.ring_slot == (index as u32 % 3)
    });
    let state_matches = final_state.frame_position == frames.len() as u64
        && final_state.emitted_samples == reference.len() as u64
        && final_state.kv_ring_head == (frames.len() % 72) as u32
        && final_state.next_ring_slot == (packet_results.len() % 3) as u32;
    let reset_matches = reset_state.frame_position == 0
        && reset_state.emitted_samples == 0
        && reset_state.kv_ring_head == 0
        && reset_state.next_ring_slot == 0;
    let snr_satisfies_minimum = snr_db.is_none_or(|value| value >= 50.0);
    let passed = sample_count_matches
        && maximum_absolute_error == 0
        && seam_maximum_absolute_error == 0
        && boundary_delta_error == 0
        && positions_match
        && state_matches
        && reset_matches
        && post_final_rejection
        && snr_satisfies_minimum;

    let report = json!({
        "schema_version": 1,
        "mode": "deterministic_fixture_not_neural_audio",
        "passed": passed,
        "abi_version": api.abi_version(),
        "input_frames": frames.len(),
        "packet_count": packet_results.len(),
        "packet_pattern": packet_pattern,
        "expected_samples": reference.len(),
        "actual_samples": incremental.len(),
        "sample_count_matches": sample_count_matches,
        "maximum_absolute_error": maximum_absolute_error,
        "mean_squared_error": mean_squared_error,
        "snr_db": snr_db.map_or(Value::String("infinite".to_owned()), Value::from),
        "snr_minimum_db": 50.0,
        "snr_minimum_satisfied": snr_satisfies_minimum,
        "seam_maximum_absolute_error": seam_maximum_absolute_error,
        "boundary_delta_error": boundary_delta_error,
        "positions_match": positions_match,
        "state_matches": state_matches,
        "post_final_packet_rejected": post_final_rejection,
        "reset_matches": reset_matches,
        "state_bytes": state_json(initial_state),
        "limitations": [
            "This fixture validates incremental state and packet-boundary invariance.",
            "It does not execute Qwen3-TTS tokenizer decoder weights and is not generated speech."
        ]
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !passed {
        return Err(io::Error::other("incremental parity validation failed").into());
    }
    Ok(())
}

fn benchmark(library_path: &Path, iterations: usize) -> Result<(), Box<dyn Error>> {
    if iterations < 200 {
        return Err(io::Error::other("benchmark requires at least 200 iterations").into());
    }
    let api = Api::load(library_path)?;
    let mut codec = api.create_codec(0).map_err(io::Error::other)?;
    let frames = deterministic_frames(MAX_PACKET_FRAMES);

    codec.reset().map_err(io::Error::other)?;
    for _ in 0..20 {
        codec.process(&frames, false).map_err(|(status, message)| {
            io::Error::other(format!("warmup status {status}: {message}"))
        })?;
    }

    codec.reset().map_err(io::Error::other)?;
    let mut gpu_microseconds = Vec::with_capacity(iterations);
    let mut end_to_end_microseconds = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        let (pcm, result) = codec
            .process(&frames, iteration + 1 == iterations)
            .map_err(|(status, message)| {
                io::Error::other(format!("benchmark status {status}: {message}"))
            })?;
        if pcm.len() != MAX_PACKET_SAMPLES {
            return Err(io::Error::other("unexpected benchmark sample count").into());
        }
        gpu_microseconds.push(f64::from(result.gpu_microseconds));
        end_to_end_microseconds.push(f64::from(result.end_to_end_microseconds));
    }
    gpu_microseconds.sort_by(f64::total_cmp);
    end_to_end_microseconds.sort_by(f64::total_cmp);
    let state = codec.state_info().map_err(io::Error::other)?;

    let report = json!({
        "schema_version": 1,
        "mode": "deterministic_fixture_not_neural_audio",
        "iterations": iterations,
        "warmup_iterations": 20,
        "continuous_measured_stream": true,
        "reset_between_measured_packets": false,
        "frames_per_packet": MAX_PACKET_FRAMES,
        "samples_per_packet": MAX_PACKET_SAMPLES,
        "audio_milliseconds_per_packet": 320.0,
        "gpu_microseconds": latency_json(&gpu_microseconds),
        "end_to_end_microseconds": latency_json(&end_to_end_microseconds),
        "state_bytes": state_json(state),
        "limitations": [
            "Latency covers the deterministic CUDA state/parity fixture only.",
            "It must not be reported as Qwen3-TTS neural decoder latency or audio quality."
        ]
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn deterministic_frames(count: usize) -> Vec<[u16; CODEBOOKS]> {
    let mut state = 0x6d2b_79f5_u32;
    (0..count)
        .map(|_| {
            let mut frame = [0_u16; CODEBOOKS];
            for code in &mut frame {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *code = ((state >> 8) & 2047) as u16;
            }
            frame
        })
        .collect()
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * percentile).ceil() as usize;
    sorted[index]
}

fn latency_json(sorted: &[f64]) -> Value {
    json!({
        "minimum": sorted[0],
        "p50": percentile(sorted, 0.50),
        "p95": percentile(sorted, 0.95),
        "p99": percentile(sorted, 0.99),
        "maximum": sorted[sorted.len() - 1]
    })
}

fn state_json(state: ffi::StateInfo) -> Value {
    json!({
        "device_total": state.device_bytes,
        "host_pinned": state.host_pinned_bytes,
        "transformer_kv": state.transformer_kv_bytes,
        "convolution_history": state.convolution_history_bytes,
        "codec_ring": state.codec_ring_bytes,
        "pcm_ring": state.pcm_ring_bytes,
        "ring_slots": state.ring_slots,
        "max_packet_frames": state.max_packet_frames
    })
}

fn required_argument<'a>(
    arguments: &'a [String],
    index: usize,
    label: &str,
) -> Result<&'a str, Box<dyn Error>> {
    arguments
        .get(index)
        .map(String::as_str)
        .ok_or_else(|| io::Error::other(format!("missing {label}")).into())
}
