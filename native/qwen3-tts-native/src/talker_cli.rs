use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use crate::native_talker::{NativeTalker, SamplingConfig, VoiceDesignRequest};

pub fn run_generate_codes(mut arguments: impl Iterator<Item = OsString>) -> Result<()> {
    let library = next_path(
        &mut arguments,
        "generate-codes requires a shared-library path",
    )?;
    let model = next_path(&mut arguments, "generate-codes requires a model directory")?;
    let mut text = None;
    let mut instruction = None;
    let mut language = "German".to_owned();
    let mut output = None;
    let mut max_frames = 64_usize;
    let mut max_sequence = 1_024_usize;
    let mut seed = 0_u64;
    let mut greedy = false;

    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--text") => text = Some(next_string(&mut arguments, "--text")?),
            Some("--instruction") => {
                instruction = Some(next_string(&mut arguments, "--instruction")?);
            }
            Some("--language") => language = next_string(&mut arguments, "--language")?,
            Some("--output") => {
                output = Some(next_path(&mut arguments, "--output requires a path")?)
            }
            Some("--max-frames") => max_frames = next_usize(&mut arguments, "--max-frames")?,
            Some("--max-sequence") => max_sequence = next_usize(&mut arguments, "--max-sequence")?,
            Some("--seed") => seed = next_u64(&mut arguments, "--seed")?,
            Some("--greedy") => greedy = true,
            _ => bail!("unknown generate-codes argument {argument:?}"),
        }
    }

    let text = text.context("generate-codes requires --text")?;
    let instruction = instruction.context("generate-codes requires --instruction")?;
    let engine = NativeTalker::load(&library, &model, 0)?;
    let mut request = VoiceDesignRequest::new(text, instruction, language);
    request.max_frames = max_frames;
    request.max_sequence_length = max_sequence;
    request.random_seed = seed;
    if greedy {
        request.talker_sampling = SamplingConfig::greedy();
        request.predictor_sampling = SamplingConfig::greedy();
    }
    let generated = engine.generate(request)?;
    let encoded = serde_json::to_string_pretty(&generated)?;
    if let Some(path) = output {
        fs::write(&path, encoded).with_context(|| format!("failed to write {}", path.display()))?;
    } else {
        println!("{encoded}");
    }
    Ok(())
}

fn next_path(arguments: &mut impl Iterator<Item = OsString>, message: &str) -> Result<PathBuf> {
    arguments
        .next()
        .map(PathBuf::from)
        .context(message.to_owned())
}

fn next_string(arguments: &mut impl Iterator<Item = OsString>, flag: &str) -> Result<String> {
    arguments
        .next()
        .with_context(|| format!("{flag} requires a value"))?
        .into_string()
        .map_err(|_| anyhow::anyhow!("{flag} is not valid UTF-8"))
}

fn next_usize(arguments: &mut impl Iterator<Item = OsString>, flag: &str) -> Result<usize> {
    next_string(arguments, flag)?
        .parse::<usize>()
        .with_context(|| format!("{flag} must be an unsigned integer"))
}

fn next_u64(arguments: &mut impl Iterator<Item = OsString>, flag: &str) -> Result<u64> {
    next_string(arguments, flag)?
        .parse::<u64>()
        .with_context(|| format!("{flag} must be an unsigned integer"))
}
