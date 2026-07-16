use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use qwen3_tts_native::native_talker::{NativeTalker, SamplingConfig, VoiceDesignRequest};
use qwen3_tts_native_codec::{CODEBOOKS, DecoderWeights, NativeCodecLibrary, SAMPLES_PER_FRAME};
use serde_json::json;

const SAMPLE_RATE: u32 = 24_000;
const CHANNELS: u16 = 1;
const BITS_PER_SAMPLE: u16 = 16;

#[derive(Debug)]
struct Arguments {
    talker_library: PathBuf,
    codec_library: PathBuf,
    model_root: PathBuf,
    wav_output: PathBuf,
    report_output: PathBuf,
    text: String,
    instruction: String,
    language: String,
    max_frames: usize,
    max_sequence: usize,
    packet_frames: usize,
    seed: u64,
    greedy: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let arguments = parse_arguments()?;
    let talker_load_started = Instant::now();
    let mut talker = NativeTalker::load(
        &arguments.talker_library,
        &arguments.model_root,
        0,
        arguments.max_sequence,
        arguments.seed,
    )?;
    let talker_load_milliseconds = talker_load_started.elapsed().as_secs_f64() * 1_000.0;

    let codec_load_started = Instant::now();
    let codec_library = NativeCodecLibrary::load(&arguments.codec_library)?;
    let decoder_weights = DecoderWeights::open(
        &arguments
            .model_root
            .join("speech_tokenizer/model.safetensors"),
    )?;
    let mut codec = codec_library.create_codec(0).map_err(io::Error::other)?;
    let codec_model = codec
        .load_model(&decoder_weights)
        .map_err(io::Error::other)?;
    codec.warmup().map_err(io::Error::other)?;
    let codec_load_and_warmup_milliseconds = codec_load_started.elapsed().as_secs_f64() * 1_000.0;

    let mut request = VoiceDesignRequest::new(
        arguments.text.clone(),
        arguments.instruction.clone(),
        arguments.language.clone(),
    );
    request.max_frames = arguments.max_frames;
    request.random_seed = arguments.seed;
    if arguments.greedy {
        request.talker_sampling = SamplingConfig::greedy();
        request.predictor_sampling = SamplingConfig::greedy();
    }

    let pipeline_started = Instant::now();
    let talker_started = Instant::now();
    let generated = talker.generate(request)?;
    let talker_wall_milliseconds = talker_started.elapsed().as_secs_f64() * 1_000.0;
    if generated.codec_codes.len() != generated.frame_count * CODEBOOKS {
        return Err(io::Error::other("talker returned a non-frame-aligned code stream").into());
    }
    if generated.frame_count == 0 {
        return Err(io::Error::other("talker returned no codec frames").into());
    }

    let frames = generated
        .codec_codes
        .chunks_exact(CODEBOOKS)
        .map(|codes| {
            let mut frame = [0_u16; CODEBOOKS];
            frame.copy_from_slice(codes);
            frame
        })
        .collect::<Vec<_>>();
    let mut pcm = Vec::with_capacity(frames.len() * SAMPLES_PER_FRAME);
    let mut packet_reports = Vec::new();
    let mut expected_frame = 0_u64;
    let mut expected_sample = 0_u64;
    let mut first_audio_milliseconds = None;
    let decoder_started = Instant::now();
    let packet_count = frames.len().div_ceil(arguments.packet_frames);
    for (packet_index, packet_frames) in frames.chunks(arguments.packet_frames).enumerate() {
        let is_final = packet_index + 1 == packet_count;
        let (samples, packet) =
            codec
                .process(packet_frames, is_final)
                .map_err(|(status, message)| {
                    io::Error::other(format!("decoder status {status}: {message}"))
                })?;
        if packet.first_frame_position != expected_frame {
            return Err(io::Error::other("decoder frame positions are not contiguous").into());
        }
        if packet.first_sample_position != expected_sample {
            return Err(io::Error::other("decoder sample positions are not contiguous").into());
        }
        let expected_samples = packet_frames.len() * SAMPLES_PER_FRAME;
        if samples.len() != expected_samples || packet.sample_count as usize != expected_samples {
            return Err(io::Error::other("decoder returned an invalid sample count").into());
        }
        if (packet.is_final != 0) != is_final {
            return Err(io::Error::other("decoder final-packet flag is inconsistent").into());
        }
        if first_audio_milliseconds.is_none() {
            first_audio_milliseconds = Some(pipeline_started.elapsed().as_secs_f64() * 1_000.0);
        }
        packet_reports.push(json!({
            "sequence": packet_index,
            "first_codec_frame": packet.first_frame_position,
            "first_sample": packet.first_sample_position,
            "codec_frames": packet.frame_count,
            "samples": packet.sample_count,
            "is_final": packet.is_final != 0,
            "gpu_microseconds": packet.gpu_microseconds,
            "end_to_end_microseconds": packet.end_to_end_microseconds,
        }));
        expected_frame += u64::from(packet.frame_count);
        expected_sample += u64::from(packet.sample_count);
        pcm.extend_from_slice(&samples);
    }
    let decoder_wall_milliseconds = decoder_started.elapsed().as_secs_f64() * 1_000.0;
    let pipeline_wall_milliseconds = pipeline_started.elapsed().as_secs_f64() * 1_000.0;

    let expected_samples = generated.frame_count * SAMPLES_PER_FRAME;
    if pcm.len() != expected_samples {
        return Err(io::Error::other("E2E PCM length does not match generated frame count").into());
    }
    write_wav(&arguments.wav_output, &pcm)?;

    let audio_milliseconds = pcm.len() as f64 * 1_000.0 / f64::from(SAMPLE_RATE);
    let report = json!({
        "schema_version": 1,
        "operation": "native-qwen3-tts-e2e-smoke",
        "qualifying_run": false,
        "streaming_mode": "whole_sequence_talker_then_packet_decoder",
        "limitation": "This smoke proves real text-to-code-to-PCM execution. It is not the incremental runtime qualification because the current public talker API returns the complete code sequence before decoder polling.",
        "model_root": arguments.model_root,
        "language": arguments.language,
        "text": arguments.text,
        "instruction": arguments.instruction,
        "greedy": arguments.greedy,
        "seed": arguments.seed,
        "packet_frames": arguments.packet_frames,
        "wav_output": arguments.wav_output,
        "talker": {
            "load_milliseconds": talker_load_milliseconds,
            "wall_milliseconds": talker_wall_milliseconds,
            "prompt_tokens": generated.prompt_tokens,
            "frame_count": generated.frame_count,
            "ended_by_eos": generated.ended_by_eos,
            "first_semantic_token": generated.first_semantic_token,
            "final_semantic_token": generated.final_semantic_token,
            "prefill_gpu_milliseconds": generated.prefill_talker_gpu_milliseconds,
            "frame_timings": generated.frame_timings,
            "memory": generated.memory,
        },
        "decoder": {
            "load_and_warmup_milliseconds": codec_load_and_warmup_milliseconds,
            "wall_milliseconds": decoder_wall_milliseconds,
            "source_bytes": codec_model.source_bytes,
            "device_bytes": codec_model.device_bytes,
            "tensor_count": codec_model.tensor_count,
            "packets": packet_reports,
        },
        "audio": {
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "samples": pcm.len(),
            "milliseconds": audio_milliseconds,
        },
        "observed_whole_sequence_pipeline": {
            "first_audio_milliseconds": first_audio_milliseconds,
            "wall_milliseconds": pipeline_wall_milliseconds,
            "real_time_factor": pipeline_wall_milliseconds / audio_milliseconds,
        },
    });
    let encoded = serde_json::to_string_pretty(&report)?;
    fs::write(&arguments.report_output, format!("{encoded}\n"))?;
    println!("{encoded}");
    Ok(())
}

fn parse_arguments() -> Result<Arguments, Box<dyn Error>> {
    let mut values = env::args().skip(1);
    let talker_library = required_path(&mut values, "talker shared-library path")?;
    let codec_library = required_path(&mut values, "codec shared-library path")?;
    let model_root = required_path(&mut values, "model root")?;
    let wav_output = required_path(&mut values, "WAV output path")?;
    let mut report_output = wav_output.with_extension("json");
    let mut text = None;
    let mut instruction = None;
    let mut language = "German".to_owned();
    let mut max_frames = 64_usize;
    let mut max_sequence = 1_024_usize;
    let mut packet_frames = 4_usize;
    let mut seed = 0_u64;
    let mut greedy = false;

    while let Some(flag) = values.next() {
        match flag.as_str() {
            "--text" => text = Some(required_string(&mut values, "--text")?),
            "--instruction" => instruction = Some(required_string(&mut values, "--instruction")?),
            "--language" => language = required_string(&mut values, "--language")?,
            "--max-frames" => max_frames = required_usize(&mut values, "--max-frames")?,
            "--max-sequence" => max_sequence = required_usize(&mut values, "--max-sequence")?,
            "--packet-frames" => packet_frames = required_usize(&mut values, "--packet-frames")?,
            "--seed" => seed = required_string(&mut values, "--seed")?.parse()?,
            "--report" => report_output = required_path(&mut values, "--report path")?,
            "--greedy" => greedy = true,
            _ => return Err(io::Error::other(format!("unknown argument {flag:?}")).into()),
        }
    }
    if max_frames == 0 || max_sequence == 0 {
        return Err(io::Error::other("frame and sequence limits must be positive").into());
    }
    if !(1..=4).contains(&packet_frames) {
        return Err(io::Error::other("--packet-frames must be between 1 and 4").into());
    }

    Ok(Arguments {
        talker_library,
        codec_library,
        model_root,
        wav_output,
        report_output,
        text: text.ok_or_else(|| io::Error::other("--text is required"))?,
        instruction: instruction.ok_or_else(|| io::Error::other("--instruction is required"))?,
        language,
        max_frames,
        max_sequence,
        packet_frames,
        seed,
        greedy,
    })
}

fn required_path(
    values: &mut impl Iterator<Item = String>,
    label: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    Ok(PathBuf::from(values.next().ok_or_else(|| {
        io::Error::other(format!("missing {label}"))
    })?))
}

fn required_string(
    values: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    values
        .next()
        .ok_or_else(|| io::Error::other(format!("{flag} requires a value")).into())
}

fn required_usize(
    values: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<usize, Box<dyn Error>> {
    Ok(required_string(values, flag)?.parse()?)
}

fn write_wav(path: &Path, samples: &[i16]) -> Result<(), Box<dyn Error>> {
    let data_bytes = samples
        .len()
        .checked_mul(size_of::<i16>())
        .and_then(|bytes| u32::try_from(bytes).ok())
        .ok_or_else(|| io::Error::other("WAV payload is too large"))?;
    let riff_bytes = 36_u32
        .checked_add(data_bytes)
        .ok_or_else(|| io::Error::other("WAV RIFF size overflow"))?;
    let byte_rate = SAMPLE_RATE * u32::from(CHANNELS) * u32::from(BITS_PER_SAMPLE) / 8;
    let block_align = CHANNELS * BITS_PER_SAMPLE / 8;

    let mut file = fs::File::create(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&riff_bytes.to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16_u32.to_le_bytes())?;
    file.write_all(&1_u16.to_le_bytes())?;
    file.write_all(&CHANNELS.to_le_bytes())?;
    file.write_all(&SAMPLE_RATE.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&BITS_PER_SAMPLE.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_bytes.to_le_bytes())?;
    for sample in samples {
        file.write_all(&sample.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}
