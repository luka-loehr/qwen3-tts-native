use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use qwen3_tts_native::artifact::{PackOptions, pack_artifact};
use qwen3_tts_native::contract::CodecWeightDtype;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut arguments = env::args_os();
    let program = arguments
        .next()
        .unwrap_or_else(|| OsString::from("pack_artifact"));
    let source = arguments.next().with_context(|| usage(&program))?;
    let output = arguments.next().with_context(|| usage(&program))?;
    let mut options = PackOptions::new(PathBuf::from(source), PathBuf::from(output));
    let mut report_path = None;

    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--codec-dtype") => {
                let value = arguments
                    .next()
                    .context("--codec-dtype requires bf16 or f32")?;
                options.codec_dtype = CodecWeightDtype::parse(
                    value
                        .to_str()
                        .context("--codec-dtype must be valid UTF-8")?,
                )?;
            }
            Some("--buffer-mib") => {
                let value = arguments
                    .next()
                    .context("--buffer-mib requires an integer")?;
                let mib = value
                    .to_str()
                    .context("--buffer-mib must be valid UTF-8")?
                    .parse::<usize>()
                    .context("--buffer-mib must be an integer")?;
                if !(1..=64).contains(&mib) {
                    bail!("--buffer-mib must be between 1 and 64");
                }
                options.copy_buffer_bytes = mib
                    .checked_mul(1024 * 1024)
                    .context("--buffer-mib overflows usize")?;
            }
            Some("--report") => {
                report_path = Some(PathBuf::from(
                    arguments.next().context("--report requires a path")?,
                ));
            }
            _ => bail!("unknown argument {argument:?}\n{}", usage(&program)),
        }
    }

    let report = pack_artifact(&options)?;
    let encoded = serde_json::to_string_pretty(&report)?;
    if let Some(path) = report_path {
        fs::write(&path, format!("{encoded}\n"))
            .with_context(|| format!("failed to write {}", path.display()))?;
    } else {
        println!("{encoded}");
    }
    Ok(())
}

fn usage(program: &OsString) -> String {
    format!(
        "usage: {} <source-snapshot> <output-directory> [--codec-dtype bf16|f32] [--buffer-mib 1..64] [--report path]",
        Path::new(program).display()
    )
}
