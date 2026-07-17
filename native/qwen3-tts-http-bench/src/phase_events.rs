use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::{BenchError, Result};

const PHASE_EVENTS_SCHEMA_VERSION: &str = "qwen3-tts-http-bench-phase-events/v1";

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PhaseEvent {
    WarmupStart,
    WarmupEnd,
    MeasuredStart,
    MeasuredEnd,
}

impl PhaseEvent {
    const fn sequence(self) -> usize {
        match self {
            Self::WarmupStart => 0,
            Self::WarmupEnd => 1,
            Self::MeasuredStart => 2,
            Self::MeasuredEnd => 3,
        }
    }
}

#[derive(Debug, Serialize)]
struct PhaseEventRecord {
    schema_version: &'static str,
    sequence: usize,
    event: PhaseEvent,
    wall_time_unix_ns: u128,
    monotonic_elapsed_ns: u128,
}

/// Reserves an optional evidence file and records phase boundaries without I/O
/// on the measured path.
pub(crate) struct PhaseEventLog {
    file: Option<File>,
    origin: Instant,
    records: Vec<PhaseEventRecord>,
}

impl PhaseEventLog {
    pub(crate) fn create(path: Option<&Path>) -> Result<Self> {
        let file = path.map(create_new_file).transpose()?;
        Ok(Self {
            file,
            origin: Instant::now(),
            records: Vec::with_capacity(4),
        })
    }

    pub(crate) fn mark(&mut self, event: PhaseEvent) -> Result<Instant> {
        if event.sequence() != self.records.len() {
            return Err(BenchError::Timing(format!(
                "phase event {event:?} is out of order"
            )));
        }

        let instant = Instant::now();
        let wall_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| {
                BenchError::Timing(format!("system clock predates Unix epoch: {error}"))
            })?;
        self.records.push(PhaseEventRecord {
            schema_version: PHASE_EVENTS_SCHEMA_VERSION,
            sequence: event.sequence(),
            event,
            wall_time_unix_ns: wall_time.as_nanos(),
            monotonic_elapsed_ns: instant.duration_since(self.origin).as_nanos(),
        });
        Ok(instant)
    }

    /// Writes and syncs either a complete successful sequence or the useful
    /// prefix retained after a failed warmup.
    pub(crate) fn finish(mut self, require_complete: bool) -> Result<()> {
        if require_complete && self.records.len() != 4 {
            return Err(BenchError::Timing(format!(
                "successful benchmark has {} phase events instead of four",
                self.records.len()
            )));
        }

        let Some(mut file) = self.file.take() else {
            return Ok(());
        };
        let mut payload = Vec::new();
        for record in &self.records {
            serde_json::to_writer(&mut payload, record)?;
            payload.push(b'\n');
        }
        file.write_all(&payload)?;
        file.flush()?;
        file.sync_data()?;
        Ok(())
    }
}

pub(crate) fn validate_path(path: Option<&Path>, output_dir: &Path) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    if path.as_os_str().is_empty() {
        return Err(BenchError::Configuration(
            "--phase-events must not be an empty path".to_owned(),
        ));
    }

    let phase_path = lexical_absolute(path)?;
    if phase_path == lexical_absolute(output_dir)? {
        return Err(BenchError::Configuration(
            "--phase-events must not replace --output-dir".to_owned(),
        ));
    }
    for report_name in ["requests.jsonl", "packets.jsonl", "summary.json"] {
        if phase_path == lexical_absolute(&output_dir.join(report_name))? {
            return Err(BenchError::Configuration(format!(
                "--phase-events must not replace canonical report file {report_name}"
            )));
        }
    }
    Ok(())
}

fn create_new_file(path: &Path) -> Result<File> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    Ok(OpenOptions::new().write(true).create_new(true).open(path)?)
}

fn lexical_absolute(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(Path::new(std::path::MAIN_SEPARATOR_STR)),
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(segment) => normalized.push(segment),
        }
    }
    Ok(normalized)
}
