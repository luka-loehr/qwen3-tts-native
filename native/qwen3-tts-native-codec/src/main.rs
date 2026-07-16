mod reference;

use ffi::{Api, CODEBOOKS, MAX_PACKET_FRAMES, MAX_PACKET_SAMPLES, STATUS_STATE};
use qwen3_tts_native_codec::{ffi, model};
use reference::ReferenceState;
use serde_json::{Value, json};
use std::error::Error;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

type BatchScheduleOutput = (Vec<Vec<i16>>, Vec<Vec<u32>>);

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
        "neural-benchmark" => {
            let library = required_argument(&arguments, 2, "shared library path")?;
            let model = required_argument(&arguments, 3, "speech tokenizer safetensors path")?;
            let iterations = arguments
                .get(4)
                .map(|value| value.parse::<usize>())
                .transpose()?
                .unwrap_or(200);
            neural_benchmark(Path::new(library), Path::new(model), iterations)
        }
        "neural-cold-start" => {
            let library = required_argument(&arguments, 2, "shared library path")?;
            let model = required_argument(&arguments, 3, "speech tokenizer safetensors path")?;
            neural_cold_start(Path::new(library), Path::new(model))
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
        "transformer-parity" => {
            let library = required_argument(&arguments, 2, "shared library path")?;
            let model = required_argument(&arguments, 3, "speech tokenizer safetensors path")?;
            let fixture = required_argument(&arguments, 4, "decoder fixture directory")?;
            transformer_parity(Path::new(library), Path::new(model), Path::new(fixture))
        }
        "latent-parity" => {
            let library = required_argument(&arguments, 2, "shared library path")?;
            let model = required_argument(&arguments, 3, "speech tokenizer safetensors path")?;
            let fixture = required_argument(&arguments, 4, "decoder fixture directory")?;
            latent_parity(Path::new(library), Path::new(model), Path::new(fixture))
        }
        "decoder-parity" => {
            let library = required_argument(&arguments, 2, "shared library path")?;
            let model = required_argument(&arguments, 3, "speech tokenizer safetensors path")?;
            let fixture = required_argument(&arguments, 4, "decoder fixture directory")?;
            decoder_parity(Path::new(library), Path::new(model), Path::new(fixture))
        }
        "neural-parity" => {
            let library = required_argument(&arguments, 2, "shared library path")?;
            let model = required_argument(&arguments, 3, "speech tokenizer safetensors path")?;
            let fixture = required_argument(&arguments, 4, "decoder fixture directory")?;
            neural_parity(Path::new(library), Path::new(model), Path::new(fixture))
        }
        "batch-parity" => {
            let library = required_argument(&arguments, 2, "shared library path")?;
            let model = required_argument(&arguments, 3, "speech tokenizer safetensors path")?;
            let fixture = required_argument(&arguments, 4, "decoder fixture directory")?;
            batch_parity(Path::new(library), Path::new(model), Path::new(fixture))
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

fn batch_parity(
    library_path: &Path,
    model_path: &Path,
    fixture_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let codes = read_u16_le(&fixture_path.join("codes.u16le"))?;
    if codes.len() % ffi::CODEBOOKS != 0 {
        return Err(io::Error::other("fixture code count is not divisible by 16").into());
    }
    let base_frames = codes
        .chunks_exact(ffi::CODEBOOKS)
        .map(|values| values.try_into().expect("chunk contains exactly 16 codes"))
        .collect::<Vec<[u16; ffi::CODEBOOKS]>>();
    if base_frames.len() != 4 {
        return Err(io::Error::other("batch fixture must contain four frames").into());
    }
    let streams = (0..6)
        .map(|stream| {
            base_frames
                .iter()
                .enumerate()
                .map(|(frame_index, frame)| {
                    let mut varied = *frame;
                    for (codebook, code) in varied.iter_mut().enumerate() {
                        let offset = stream * 71 + frame_index * 17 + codebook * 3;
                        *code = ((*code as usize + offset) & 2047) as u16;
                    }
                    varied
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let api = Api::load(library_path)?;
    let model = model::SafetensorsFile::open(model_path)?;

    let schedules_three = vec![vec![1_usize; 4]; 3];
    let expected_three = standalone_schedules(&api, &model, &streams[..3], &schedules_three)?;
    let mut codecs_three = (0..3)
        .map(|_| api.create_codec(0).map_err(io::Error::other))
        .collect::<Result<Vec<_>, _>>()?;
    for codec in &mut codecs_three {
        codec.load_model(&model).map_err(io::Error::other)?;
    }
    let (actual_three, slots_three) =
        run_batch_schedule(&api, &mut codecs_three, &streams[..3], &schedules_three)?;
    let states_three = codecs_three
        .iter()
        .map(|codec| codec.state_info().map_err(io::Error::other))
        .collect::<Result<Vec<_>, _>>()?;
    for codec in &mut codecs_three {
        codec.reset().map_err(io::Error::other)?;
    }
    let (replay_three, replay_slots_three) =
        run_batch_schedule(&api, &mut codecs_three, &streams[..3], &schedules_three)?;
    let b3_exact = actual_three == expected_three && replay_three == actual_three;
    let b3_slots_match =
        slots_three.iter().all(|slots| slots == &[0, 1, 2, 0]) && replay_slots_three == slots_three;
    let b3_states_match = states_three.iter().all(|state| {
        state.frame_position == 4
            && state.emitted_samples == 4 * ffi::SAMPLES_PER_FRAME as u64
            && state.next_ring_slot == 1
            && state.kv_ring_head == 4
    });
    drop(codecs_three);

    let first_counts = [1_usize, 2, 3, 1, 2, 3];
    let schedules_six = first_counts
        .iter()
        .map(|first| vec![*first, 4 - *first])
        .collect::<Vec<_>>();
    let expected_six = standalone_schedules(&api, &model, &streams, &schedules_six)?;
    let mut codecs_six = (0..6)
        .map(|_| api.create_codec(0).map_err(io::Error::other))
        .collect::<Result<Vec<_>, _>>()?;
    for codec in &mut codecs_six {
        codec.load_model(&model).map_err(io::Error::other)?;
    }
    let (actual_six, slots_six) =
        run_batch_schedule(&api, &mut codecs_six, &streams, &schedules_six)?;
    let states_six = codecs_six
        .iter()
        .map(|codec| codec.state_info().map_err(io::Error::other))
        .collect::<Result<Vec<_>, _>>()?;
    let b6_exact = actual_six == expected_six;
    let b6_slots_match = slots_six.iter().all(|slots| slots == &[0, 1]);
    let b6_states_match = states_six.iter().all(|state| {
        state.frame_position == 4
            && state.emitted_samples == 4 * ffi::SAMPLES_PER_FRAME as u64
            && state.next_ring_slot == 2
            && state.kv_ring_head == 4
    });
    let mixed_final_lengths = schedules_six
        .iter()
        .map(|schedule| schedule[1] * ffi::SAMPLES_PER_FRAME)
        .collect::<Vec<_>>();
    let passed = b3_exact
        && b3_slots_match
        && b3_states_match
        && b6_exact
        && b6_slots_match
        && b6_states_match;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "mode": "independent_native_decoder_state_handles",
            "passed": passed,
            "dispatch": "array_order_reference_not_fused",
            "batch_three": {
                "streams": 3,
                "packets_per_stream": 4,
                "frame_schedule": [1, 1, 1, 1],
                "ring_slots_per_stream": slots_three,
                "slot_zero_overwritten": true,
                "standalone_bit_exact": b3_exact,
                "reset_replay_bit_exact": replay_three == actual_three,
                "state_matches": b3_states_match
            },
            "batch_six": {
                "streams": 6,
                "first_packet_frames": first_counts,
                "final_packet_samples": mixed_final_lengths,
                "ring_slots_per_stream": slots_six,
                "standalone_bit_exact": b6_exact,
                "state_matches": b6_states_match,
                "cross_request_leakage_detected": !b6_exact
            }
        }))?
    );
    if !passed {
        return Err(io::Error::other("native decoder batch parity failed").into());
    }
    Ok(())
}

fn standalone_schedules(
    api: &Api,
    model: &model::SafetensorsFile,
    streams: &[Vec<[u16; ffi::CODEBOOKS]>],
    schedules: &[Vec<usize>],
) -> Result<Vec<Vec<i16>>, Box<dyn Error>> {
    let mut codec = api.create_codec(0).map_err(io::Error::other)?;
    codec.load_model(model).map_err(io::Error::other)?;
    let mut outputs = Vec::with_capacity(streams.len());
    for (stream, schedule) in streams.iter().zip(schedules) {
        codec.reset().map_err(io::Error::other)?;
        let mut position = 0_usize;
        let mut output = Vec::with_capacity(stream.len() * ffi::SAMPLES_PER_FRAME);
        for count in schedule {
            let end = position + count;
            let (packet, _) = codec
                .process(&stream[position..end], end == stream.len())
                .map_err(|(status, message)| {
                    io::Error::other(format!("standalone status {status}: {message}"))
                })?;
            output.extend_from_slice(&packet);
            position = end;
        }
        if position != stream.len() {
            return Err(io::Error::other("standalone schedule is incomplete").into());
        }
        outputs.push(output);
    }
    Ok(outputs)
}

fn run_batch_schedule(
    api: &Api,
    codecs: &mut [ffi::Codec<'_>],
    streams: &[Vec<[u16; ffi::CODEBOOKS]>],
    schedules: &[Vec<usize>],
) -> Result<BatchScheduleOutput, Box<dyn Error>> {
    if codecs.len() != streams.len()
        || codecs.len() != schedules.len()
        || schedules.is_empty()
        || schedules
            .iter()
            .any(|schedule| schedule.len() != schedules[0].len())
    {
        return Err(io::Error::other("invalid batch schedule shape").into());
    }
    let mut positions = vec![0_usize; codecs.len()];
    let mut outputs = streams
        .iter()
        .map(|stream| Vec::with_capacity(stream.len() * ffi::SAMPLES_PER_FRAME))
        .collect::<Vec<_>>();
    let mut slots = vec![Vec::new(); codecs.len()];
    for (step, _) in schedules[0].iter().enumerate() {
        let packets = streams
            .iter()
            .enumerate()
            .map(|(stream, frames)| {
                let start = positions[stream];
                let end = start + schedules[stream][step];
                &frames[start..end]
            })
            .collect::<Vec<_>>();
        let finals = packets
            .iter()
            .enumerate()
            .map(|(stream, packet)| positions[stream] + packet.len() == streams[stream].len())
            .collect::<Vec<_>>();
        let mut codec_refs = codecs.iter_mut().collect::<Vec<_>>();
        let batch = api
            .process_batch(&mut codec_refs, &packets, &finals)
            .map_err(|(status, message)| {
                io::Error::other(format!("batch status {status}: {message}"))
            })?;
        for (stream, (pcm, result)) in batch.into_iter().enumerate() {
            if result.sample_count as usize != pcm.len() {
                return Err(io::Error::other("batch packet length metadata mismatch").into());
            }
            positions[stream] += packets[stream].len();
            outputs[stream].extend_from_slice(&pcm);
            slots[stream].push(result.ring_slot);
        }
    }
    if positions
        .iter()
        .zip(streams)
        .any(|(position, stream)| *position != stream.len())
    {
        return Err(io::Error::other("batch schedule is incomplete").into());
    }
    Ok((outputs, slots))
}

fn neural_parity(
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
    let expected = read_i16_le(&fixture_path.join("reference-pcm.s16le"))?;
    if expected.len() != frames.len() * ffi::SAMPLES_PER_FRAME {
        return Err(io::Error::other("official PCM fixture has an invalid length").into());
    }

    let api = Api::load(library_path)?;
    let model = model::SafetensorsFile::open(model_path)?;
    let mut codec = api.create_codec(0).map_err(io::Error::other)?;
    codec.load_model(&model).map_err(io::Error::other)?;
    codec.warmup().map_err(io::Error::other)?;
    let state_after_warmup = codec.state_info().map_err(io::Error::other)?;
    let warmup_state_reset = state_after_warmup.frame_position == 0
        && state_after_warmup.emitted_samples == 0
        && state_after_warmup.kv_ring_head == 0
        && state_after_warmup.next_ring_slot == 0;

    let (single, single_result) = codec.process(&frames, true).map_err(|(status, message)| {
        io::Error::other(format!("single status {status}: {message}"))
    })?;
    let single_comparison = compare_pcm(&single, &expected)?;
    let post_final_rejected = match codec.process(&frames[..1], false) {
        Err((status, _)) => status == STATUS_STATE,
        Ok(_) => false,
    };
    let warmup_after_stream_rejected = codec.warmup().is_err();

    codec.reset().map_err(io::Error::other)?;
    let mut split = Vec::with_capacity(expected.len());
    let mut split_results = Vec::new();
    let mut frame_position = 0_usize;
    for count in [1_usize, 3] {
        let end = frame_position + count;
        let (pcm, result) = codec
            .process(&frames[frame_position..end], end == frames.len())
            .map_err(|(status, message)| {
                io::Error::other(format!("split status {status}: {message}"))
            })?;
        split.extend_from_slice(&pcm);
        split_results.push(result);
        frame_position = end;
    }
    let split_comparison = compare_pcm(&split, &expected)?;
    let split_vs_single = compare_pcm(&split, &single)?;

    codec.reset().map_err(io::Error::other)?;
    let mut wrapped = Vec::with_capacity(expected.len());
    let mut wrapped_slots = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        let (pcm, result) = codec
            .process(std::slice::from_ref(frame), index + 1 == frames.len())
            .map_err(|(status, message)| {
                io::Error::other(format!("slot-wrap status {status}: {message}"))
            })?;
        wrapped.extend_from_slice(&pcm);
        wrapped_slots.push(result.ring_slot);
    }
    let wrapped_comparison = compare_pcm(&wrapped, &expected)?;
    let wrapped_state = codec.state_info().map_err(io::Error::other)?;

    codec.reset().map_err(io::Error::other)?;
    let (replay, replay_result) = codec.process(&frames, true).map_err(|(status, message)| {
        io::Error::other(format!("replay status {status}: {message}"))
    })?;
    let replay_comparison = compare_pcm(&replay, &single)?;

    codec.reset().map_err(io::Error::other)?;
    const SENTINEL: i16 = 0x5a5a;
    let mut poisoned_output = vec![SENTINEL; MAX_PACKET_SAMPLES];
    let short_result = codec
        .process_into(&frames[..1], true, &mut poisoned_output)
        .map_err(|(status, message)| {
            io::Error::other(format!("short-final status {status}: {message}"))
        })?;
    let short_comparison = compare_pcm(
        &poisoned_output[..ffi::SAMPLES_PER_FRAME],
        &expected[..ffi::SAMPLES_PER_FRAME],
    )?;
    let stale_tail_unchanged = poisoned_output[ffi::SAMPLES_PER_FRAME..]
        .iter()
        .all(|sample| *sample == SENTINEL);

    let metadata_matches = single_result.first_frame_position == 0
        && single_result.first_sample_position == 0
        && single_result.frame_count == frames.len() as u32
        && single_result.sample_count == expected.len() as u32
        && single_result.ring_slot == 0
        && single_result.is_final == 1
        && split_results[0].sample_count == ffi::SAMPLES_PER_FRAME as u32
        && split_results[1].first_frame_position == 1
        && split_results[1].first_sample_position == ffi::SAMPLES_PER_FRAME as u64
        && replay_result.sample_count == expected.len() as u32
        && short_result.sample_count == ffi::SAMPLES_PER_FRAME as u32
        && short_result.is_final == 1;
    let wrapped_state_matches = wrapped_slots == [0, 1, 2, 0]
        && wrapped_state.frame_position == frames.len() as u64
        && wrapped_state.emitted_samples == expected.len() as u64
        && wrapped_state.kv_ring_head == frames.len() as u32
        && wrapped_state.next_ring_slot == 1;
    let passed = single_comparison.maximum_absolute_error <= 1
        && split_comparison.maximum_absolute_error <= 1
        && split_vs_single.maximum_absolute_error <= 1
        && wrapped_comparison.maximum_absolute_error <= 1
        && replay_comparison.maximum_absolute_error == 0
        && short_comparison.maximum_absolute_error <= 1
        && stale_tail_unchanged
        && post_final_rejected
        && warmup_state_reset
        && warmup_after_stream_rejected
        && metadata_matches
        && wrapped_state_matches;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "mode": "real_qwen_speech_tokenizer_decoder",
            "passed": passed,
            "frames": frames.len(),
            "expected_samples": expected.len(),
            "single_packet_vs_official": single_comparison.to_json(),
            "split_1_3_vs_official": split_comparison.to_json(),
            "split_1_3_vs_single": split_vs_single.to_json(),
            "single_frame_packets_vs_official": wrapped_comparison.to_json(),
            "reset_replay_vs_single": replay_comparison.to_json(),
            "short_final_vs_official_prefix": short_comparison.to_json(),
            "ring_slots": wrapped_slots,
            "ring_slot_reuse_verified": wrapped_state_matches,
            "short_final_samples": short_result.sample_count,
            "stale_tail_sentinel_unchanged": stale_tail_unchanged,
            "post_final_packet_rejected": post_final_rejected,
            "warmup_state_reset": warmup_state_reset,
            "warmup_after_stream_rejected": warmup_after_stream_rejected,
            "packet_metadata_matches": metadata_matches,
            "pcm_maximum_absolute_error_threshold": 1
        }))?
    );
    if !passed {
        return Err(io::Error::other("real neural decoder parity failed").into());
    }
    Ok(())
}

fn decoder_parity(
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
    let specifications = [
        (6_u32, "06-decoder-pre-conv.f32le", 5.0e-5_f64, 4, 1536),
        (7, "07-decoder-block-1.f32le", 5.0e-5, 32, 768),
        (8, "08-decoder-block-2.f32le", 1.0e-4, 160, 384),
        (9, "09-decoder-block-3.f32le", 2.0e-4, 640, 192),
        (10, "10-decoder-block-4.f32le", 1.0e-3, 1920, 96),
        (11, "11-final-snake.f32le", 2.0e-3, 1920, 96),
        (12, "12-final-pre-clamp.f32le", 5.0e-6, 1920, 1),
        (13, "13-final-clamp.f32le", 5.0e-6, 1920, 1),
    ];
    let api = Api::load(library_path)?;
    let model = model::SafetensorsFile::open(model_path)?;
    let mut codec = api.create_codec(0).map_err(io::Error::other)?;
    codec.load_model(&model).map_err(io::Error::other)?;
    let mut results = serde_json::Map::new();
    let mut passed = true;
    let split_pattern = [1_usize, 3];
    for (checkpoint, filename, threshold, positions_per_frame, channels) in specifications {
        codec.reset().map_err(io::Error::other)?;
        let expected = read_f32_le(&fixture_path.join(filename))?;
        let actual = codec
            .debug_decoder_checkpoint(&frames, checkpoint, expected.len())
            .map_err(io::Error::other)?;
        let comparison = compare_f32(&actual, &expected)?;
        codec.reset().map_err(io::Error::other)?;
        let stream_positions = frames.len() * positions_per_frame;
        let mut split = vec![0.0_f32; expected.len()];
        let mut frame_position = 0_usize;
        let mut output_position = 0_usize;
        for requested in split_pattern {
            let count = requested.min(frames.len() - frame_position);
            let packet_positions = count * positions_per_frame;
            let packet = codec
                .debug_decoder_checkpoint(
                    &frames[frame_position..frame_position + count],
                    checkpoint,
                    packet_positions * channels,
                )
                .map_err(io::Error::other)?;
            copy_channel_major_packet(
                &packet,
                packet_positions,
                &mut split,
                stream_positions,
                output_position,
                channels,
            );
            frame_position += count;
            output_position += packet_positions;
            if frame_position == frames.len() {
                break;
            }
        }
        let split_comparison = compare_f32(&split, &expected)?;
        let checkpoint_passed = comparison.maximum_absolute_error <= threshold
            && split_comparison.maximum_absolute_error <= threshold;
        passed &= checkpoint_passed;
        results.insert(
            checkpoint.to_string(),
            json!({
                "file": filename,
                "passed": checkpoint_passed,
                "maximum_absolute_error_threshold": threshold,
                "comparison": comparison.to_json(),
                "split_packet_comparison": split_comparison.to_json(),
                "split_packet_pattern": split_pattern
            }),
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "mode": "real_qwen_waveform_decoder",
            "passed": passed,
            "frames": frames.len(),
            "checkpoints": results,
            "prefix_recomputed": false
        }))?
    );
    if !passed {
        return Err(io::Error::other("native waveform decoder parity failed").into());
    }
    Ok(())
}

fn latent_parity(
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
    let expected_one = read_f32_le(&fixture_path.join("04-latent-upsample-1.f32le"))?;
    let expected_two = read_f32_le(&fixture_path.join("05-latent-upsample-2.f32le"))?;
    let api = Api::load(library_path)?;
    let model = model::SafetensorsFile::open(model_path)?;
    let mut codec = api.create_codec(0).map_err(io::Error::other)?;
    codec.load_model(&model).map_err(io::Error::other)?;
    let (actual_one, actual_two) = codec.debug_latent(&frames).map_err(io::Error::other)?;
    let stage_one = compare_f32(&actual_one, &expected_one)?;
    let stage_two = compare_f32(&actual_two, &expected_two)?;
    codec.reset().map_err(io::Error::other)?;
    let split_pattern = [1_usize, 3];
    let mut split_one = vec![0.0_f32; expected_one.len()];
    let mut split_two = vec![0.0_f32; expected_two.len()];
    let mut frame_position = 0_usize;
    let mut stage_one_position = 0_usize;
    let mut stage_two_position = 0_usize;
    for requested in split_pattern {
        let count = requested.min(frames.len() - frame_position);
        let (packet_one, packet_two) = codec
            .debug_latent(&frames[frame_position..frame_position + count])
            .map_err(io::Error::other)?;
        copy_channel_major_packet(
            &packet_one,
            count * 2,
            &mut split_one,
            frames.len() * 2,
            stage_one_position,
            1024,
        );
        copy_channel_major_packet(
            &packet_two,
            count * 4,
            &mut split_two,
            frames.len() * 4,
            stage_two_position,
            1024,
        );
        frame_position += count;
        stage_one_position += count * 2;
        stage_two_position += count * 4;
        if frame_position == frames.len() {
            break;
        }
    }
    let split_stage_one = compare_f32(&split_one, &expected_one)?;
    let split_stage_two = compare_f32(&split_two, &expected_two)?;
    let passed = stage_one.maximum_absolute_error <= 1.0e-5
        && stage_two.maximum_absolute_error <= 1.0e-4
        && split_stage_one.maximum_absolute_error <= 1.0e-5
        && split_stage_two.maximum_absolute_error <= 1.0e-4;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "mode": "real_qwen_incremental_latent_upsampler",
            "passed": passed,
            "frames": frames.len(),
            "stage_one": stage_one.to_json(),
            "stage_two": stage_two.to_json(),
            "split_packet_stage_one": split_stage_one.to_json(),
            "split_packet_stage_two": split_stage_two.to_json(),
            "split_packet_pattern": split_pattern,
            "thresholds": {
                "stage_one_maximum_absolute_error": 1.0e-5,
                "stage_two_maximum_absolute_error": 1.0e-4
            },
            "prefix_recomputed": false
        }))?
    );
    if !passed {
        return Err(io::Error::other("native latent upsampler parity failed").into());
    }
    Ok(())
}

fn copy_channel_major_packet(
    packet: &[f32],
    packet_positions: usize,
    stream: &mut [f32],
    stream_positions: usize,
    stream_position: usize,
    channels: usize,
) {
    for channel in 0..channels {
        let source = &packet[channel * packet_positions..(channel + 1) * packet_positions];
        let destination_start = channel * stream_positions + stream_position;
        stream[destination_start..destination_start + packet_positions].copy_from_slice(source);
    }
}

fn transformer_parity(
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
    let expected = read_f32_le(&fixture_path.join("03-transformer.f32le"))?;
    let api = Api::load(library_path)?;
    let model = model::SafetensorsFile::open(model_path)?;
    let mut codec = api.create_codec(0).map_err(io::Error::other)?;
    codec.load_model(&model).map_err(io::Error::other)?;
    let packet_pattern = [4_usize, 1, 3, 2];
    let mut actual = Vec::with_capacity(expected.len());
    let mut position = 0_usize;
    let mut packet_count = 0_usize;
    while position < frames.len() {
        let count =
            packet_pattern[packet_count % packet_pattern.len()].min(frames.len() - position);
        actual.extend(
            codec
                .debug_transformer(&frames[position..position + count])
                .map_err(io::Error::other)?,
        );
        position += count;
        packet_count += 1;
    }
    let comparison = compare_f32(&actual, &expected)?;
    let passed = comparison.maximum_absolute_error <= 1.0e-6;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "mode": "real_qwen_incremental_transformer",
            "passed": passed,
            "frames": frames.len(),
            "packet_count": packet_count,
            "packet_pattern": packet_pattern,
            "comparison": comparison.to_json(),
            "maximum_absolute_error_threshold": 1.0e-6,
            "kv_window_frames": 72,
            "prefix_recomputed": false
        }))?
    );
    if !passed {
        return Err(io::Error::other("native decoder transformer parity failed").into());
    }
    Ok(())
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

struct PcmComparison {
    count: usize,
    maximum_absolute_error: i32,
    mean_squared_error: f64,
    exact_samples: usize,
}

impl PcmComparison {
    fn to_json(&self) -> Value {
        json!({
            "count": self.count,
            "maximum_absolute_error": self.maximum_absolute_error,
            "mean_squared_error": self.mean_squared_error,
            "exact_samples": self.exact_samples
        })
    }
}

fn compare_pcm(actual: &[i16], expected: &[i16]) -> Result<PcmComparison, Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err(io::Error::other(format!(
            "PCM lengths differ: {} != {}",
            actual.len(),
            expected.len()
        ))
        .into());
    }
    let mut maximum_absolute_error = 0_i32;
    let mut squared_error = 0_f64;
    let mut exact_samples = 0_usize;
    for (actual, expected) in actual.iter().zip(expected) {
        let difference = i32::from(*actual) - i32::from(*expected);
        maximum_absolute_error = maximum_absolute_error.max(difference.abs());
        squared_error += f64::from(difference).powi(2);
        exact_samples += usize::from(difference == 0);
    }
    Ok(PcmComparison {
        count: actual.len(),
        maximum_absolute_error,
        mean_squared_error: squared_error / actual.len() as f64,
        exact_samples,
    })
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

fn read_i16_le(path: &Path) -> Result<Vec<i16>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 2 != 0 {
        return Err(io::Error::other("i16 fixture has a partial scalar").into());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes(chunk.try_into().expect("two bytes")))
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
            .process_fixture(&frames[position..position + count], is_final)
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
    let post_final_rejection = match codec.process_fixture(&frames[..1], false) {
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

fn neural_cold_start(library_path: &Path, model_path: &Path) -> Result<(), Box<dyn Error>> {
    let api = Api::load(library_path)?;
    let model = model::SafetensorsFile::open(model_path)?;
    let mut codec = api.create_codec(0).map_err(io::Error::other)?;
    let load_start = Instant::now();
    let model_info = codec.load_model(&model).map_err(io::Error::other)?;
    let model_load_milliseconds = load_start.elapsed().as_secs_f64() * 1000.0;
    let frame = deterministic_frames(1);
    let (_, result) = codec.process(&frame, true).map_err(|(status, message)| {
        io::Error::other(format!("cold-start status {status}: {message}"))
    })?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "mode": "real_qwen_decoder_cold_start_without_warmup",
            "model_load_milliseconds": model_load_milliseconds,
            "source_bytes": model_info.source_bytes,
            "device_bytes": model_info.device_bytes,
            "first_80ms_chunk": {
                "gpu_microseconds": result.gpu_microseconds,
                "end_to_end_microseconds": result.end_to_end_microseconds
            }
        }))?
    );
    Ok(())
}

fn neural_benchmark(
    library_path: &Path,
    model_path: &Path,
    iterations: usize,
) -> Result<(), Box<dyn Error>> {
    if iterations < 200 {
        return Err(io::Error::other("neural benchmark requires at least 200 iterations").into());
    }
    let api = Api::load(library_path)?;
    let model = model::SafetensorsFile::open(model_path)?;
    let mut codec = api.create_codec(0).map_err(io::Error::other)?;
    let load_start = Instant::now();
    let model_info = codec.load_model(&model).map_err(io::Error::other)?;
    let model_load_milliseconds = load_start.elapsed().as_secs_f64() * 1000.0;

    let warmup_start = Instant::now();
    codec.warmup().map_err(io::Error::other)?;
    let warmup_milliseconds = warmup_start.elapsed().as_secs_f64() * 1000.0;
    let warmup_state = codec.state_info().map_err(io::Error::other)?;
    let warmup_state_reset = warmup_state.frame_position == 0
        && warmup_state.emitted_samples == 0
        && warmup_state.kv_ring_head == 0
        && warmup_state.next_ring_slot == 0;
    let first_frame = deterministic_frames(1);
    let (_, first_result) = codec
        .process(&first_frame, false)
        .map_err(|(status, message)| {
            io::Error::other(format!(
                "post-warmup first-chunk status {status}: {message}"
            ))
        })?;
    codec.reset().map_err(io::Error::other)?;
    let one_frame = neural_benchmark_case(&mut codec, 1, iterations)?;
    codec.reset().map_err(io::Error::other)?;
    let four_frames = neural_benchmark_case(&mut codec, 4, iterations)?;
    let state = codec.state_info().map_err(io::Error::other)?;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "mode": "real_qwen_speech_tokenizer_decoder",
            "model_load_milliseconds": model_load_milliseconds,
            "explicit_startup_warmup_milliseconds": warmup_milliseconds,
            "warmup_state_reset": warmup_state_reset,
            "model": {
                "tensor_count": model_info.tensor_count,
                "parameter_count": model_info.parameter_count,
                "source_bytes": model_info.source_bytes,
                "device_bytes": model_info.device_bytes,
                "source_f32_tensors": model_info.source_dtype_f32_count,
                "source_bf16_tensors": model_info.source_dtype_bf16_count
            },
            "first_80ms_chunk_after_explicit_warmup": {
                "gpu_microseconds": first_result.gpu_microseconds,
                "end_to_end_microseconds": first_result.end_to_end_microseconds
            },
            "one_frame_80ms_packets": one_frame,
            "four_frame_320ms_packets": four_frames,
            "runtime_state_bytes": state_json(state),
            "weights_loaded_before_timing": true,
            "pcm_delivered_after_each_packet": true,
            "python_or_node_runtime": false
        }))?
    );
    Ok(())
}

fn neural_benchmark_case(
    codec: &mut ffi::Codec<'_>,
    frames_per_packet: usize,
    iterations: usize,
) -> Result<Value, Box<dyn Error>> {
    let frames = deterministic_frames(frames_per_packet);
    for _ in 0..20 {
        codec.process(&frames, false).map_err(|(status, message)| {
            io::Error::other(format!("neural warmup status {status}: {message}"))
        })?;
    }
    codec.reset().map_err(io::Error::other)?;
    let mut gpu_microseconds = Vec::with_capacity(iterations);
    let mut end_to_end_microseconds = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        let (pcm, result) = codec
            .process(&frames, iteration + 1 == iterations)
            .map_err(|(status, message)| {
                io::Error::other(format!("neural benchmark status {status}: {message}"))
            })?;
        if pcm.len() != frames_per_packet * ffi::SAMPLES_PER_FRAME {
            return Err(io::Error::other("unexpected neural benchmark sample count").into());
        }
        gpu_microseconds.push(f64::from(result.gpu_microseconds));
        end_to_end_microseconds.push(f64::from(result.end_to_end_microseconds));
    }
    gpu_microseconds.sort_by(f64::total_cmp);
    end_to_end_microseconds.sort_by(f64::total_cmp);
    let audio_microseconds = frames_per_packet as f64 * 80_000.0;
    let end_to_end_p50 = percentile(&end_to_end_microseconds, 0.50);
    Ok(json!({
        "iterations": iterations,
        "warmup_iterations": 20,
        "continuous_stream": true,
        "reset_between_measured_packets": false,
        "frames_per_packet": frames_per_packet,
        "samples_per_packet": frames_per_packet * ffi::SAMPLES_PER_FRAME,
        "audio_milliseconds_per_packet": audio_microseconds / 1000.0,
        "gpu_microseconds": latency_json(&gpu_microseconds),
        "end_to_end_microseconds": latency_json(&end_to_end_microseconds),
        "p50_real_time_factor": end_to_end_p50 / audio_microseconds,
        "p50_times_faster_than_realtime": audio_microseconds / end_to_end_p50
    }))
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
        codec
            .process_fixture(&frames, false)
            .map_err(|(status, message)| {
                io::Error::other(format!("warmup status {status}: {message}"))
            })?;
    }

    codec.reset().map_err(io::Error::other)?;
    let mut gpu_microseconds = Vec::with_capacity(iterations);
    let mut end_to_end_microseconds = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        let (pcm, result) = codec
            .process_fixture(&frames, iteration + 1 == iterations)
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
