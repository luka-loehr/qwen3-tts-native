mod ffi;
mod report;
mod wav;

use std::collections::BTreeSet;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use ffi::{
    Api, Engine, EngineConfig, GenerationConfig, PollResult, Request, SAMPLE_RATE,
    SAMPLES_PER_FRAME, StartResult,
};
use report::{
    Distribution, QualificationGates, QualificationReport, RequestReport, RuntimeMetricsReport,
    ScenarioReport,
};
use serde::Deserialize;

const DEFAULT_REQUESTS_PER_CONCURRENCY: usize = 200;
const DEFAULT_CONCURRENCIES: &[usize] = &[1, 3, 6];
const DEFAULT_PACKET_FRAMES: u32 = 4;
const DEFAULT_RING_SLOTS: u32 = 3;
const DEFAULT_MAX_CODEC_FRAMES: u32 = 4_096;
const DEFAULT_REQUEST_TIMEOUT_SECONDS: u64 = 180;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let config = CliConfig::parse(env::args().skip(1))?;
    let corpus = load_corpus(&config.corpus)?;
    let api = Api::load(&config.library)?;
    if api.abi_version() != 1 {
        bail!("runtime ABI version must be 1, found {}", api.abi_version());
    }

    let maximum_concurrency = *config
        .concurrencies
        .iter()
        .max()
        .context("at least one concurrency is required")?;
    let engine_config = EngineConfig::new(
        maximum_concurrency as u32,
        config.packet_frames,
        config.ring_slots,
    );
    let mut engine = api.create_engine(&config.model_root, &engine_config)?;

    if config.warmup_requests > 0 {
        let warmup = ExecuteConfig {
            count: config.warmup_requests,
            concurrency: 1,
            packet_frames: config.packet_frames,
            max_codec_frames: config.max_codec_frames,
            poll_timeout_milliseconds: config.poll_timeout_milliseconds,
            request_timeout: Duration::from_secs(config.request_timeout_seconds),
            save_audio_count: 0,
            audio_dir: None,
        };
        execute_scenario(&api, &engine, &corpus, &warmup)?;
    }

    let mut scenarios = Vec::new();
    for concurrency in &config.concurrencies {
        let execute = ExecuteConfig {
            count: config.requests_per_concurrency,
            concurrency: *concurrency,
            packet_frames: config.packet_frames,
            max_codec_frames: config.max_codec_frames,
            poll_timeout_milliseconds: config.poll_timeout_milliseconds,
            request_timeout: Duration::from_secs(config.request_timeout_seconds),
            save_audio_count: config.save_audio_count,
            audio_dir: config.audio_dir.as_deref(),
        };
        scenarios.push(execute_scenario(&api, &engine, &corpus, &execute)?);
    }

    engine.destroy()?;
    let gates = evaluate_gates(
        config.qualifying_run,
        config.requests_per_concurrency,
        &scenarios,
    );
    let report = QualificationReport {
        schema_version: 1,
        qualifying_run: config.qualifying_run,
        runtime_abi_version: api.abi_version(),
        model_root: config.model_root.display().to_string(),
        library: config.library.display().to_string(),
        packet_frames: config.packet_frames,
        samples_per_packet_capacity: config.packet_frames as usize * SAMPLES_PER_FRAME as usize,
        corpus_entries: corpus.len(),
        warmup_requests: config.warmup_requests,
        requests_per_concurrency: config.requests_per_concurrency,
        scenarios,
        gates,
    };

    let encoded = serde_json::to_string_pretty(&report)?;
    if let Some(parent) = config.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&config.output, &encoded)
        .with_context(|| format!("failed to write {}", config.output.display()))?;
    println!("{encoded}");
    if config.qualifying_run && !report.gates.passed {
        bail!("native Qwen3-TTS qualification gates failed");
    }
    Ok(())
}

#[derive(Debug)]
struct CliConfig {
    qualifying_run: bool,
    library: PathBuf,
    model_root: PathBuf,
    corpus: PathBuf,
    output: PathBuf,
    audio_dir: Option<PathBuf>,
    save_audio_count: usize,
    requests_per_concurrency: usize,
    concurrencies: Vec<usize>,
    warmup_requests: usize,
    packet_frames: u32,
    ring_slots: u32,
    max_codec_frames: u32,
    poll_timeout_milliseconds: u32,
    request_timeout_seconds: u64,
}

impl CliConfig {
    fn parse(arguments: impl Iterator<Item = String>) -> Result<Self> {
        let mut arguments = arguments.peekable();
        let mode = arguments.next().context(usage())?;
        let qualifying_run = match mode.as_str() {
            "suite" => true,
            "smoke" => false,
            _ => bail!("unknown command {mode:?}\n{}", usage()),
        };
        let mut library = None;
        let mut model_root = None;
        let mut corpus = None;
        let mut output = None;
        let mut audio_dir = None;
        let mut save_audio_count = 3;
        let mut requests_per_concurrency = if qualifying_run {
            DEFAULT_REQUESTS_PER_CONCURRENCY
        } else {
            3
        };
        let mut concurrencies = if qualifying_run {
            DEFAULT_CONCURRENCIES.to_vec()
        } else {
            vec![1]
        };
        let mut warmup_requests = 3;
        let mut packet_frames = DEFAULT_PACKET_FRAMES;
        let mut ring_slots = DEFAULT_RING_SLOTS;
        let mut max_codec_frames = DEFAULT_MAX_CODEC_FRAMES;
        let mut poll_timeout_milliseconds = 0;
        let mut request_timeout_seconds = DEFAULT_REQUEST_TIMEOUT_SECONDS;

        while let Some(flag) = arguments.next() {
            match flag.as_str() {
                "--library" => library = Some(next_path(&mut arguments, &flag)?),
                "--model-root" => model_root = Some(next_path(&mut arguments, &flag)?),
                "--corpus" => corpus = Some(next_path(&mut arguments, &flag)?),
                "--output" => output = Some(next_path(&mut arguments, &flag)?),
                "--audio-dir" => audio_dir = Some(next_path(&mut arguments, &flag)?),
                "--save-audio-count" => save_audio_count = next_number(&mut arguments, &flag)?,
                "--requests-per-concurrency" => {
                    requests_per_concurrency = next_number(&mut arguments, &flag)?;
                }
                "--concurrency" => {
                    let value = next_value(&mut arguments, &flag)?;
                    concurrencies = parse_concurrencies(&value)?;
                }
                "--warmup-requests" => warmup_requests = next_number(&mut arguments, &flag)?,
                "--packet-frames" => packet_frames = next_number(&mut arguments, &flag)?,
                "--ring-slots" => ring_slots = next_number(&mut arguments, &flag)?,
                "--max-codec-frames" => {
                    max_codec_frames = next_number(&mut arguments, &flag)?;
                }
                "--poll-timeout-ms" => {
                    poll_timeout_milliseconds = next_number(&mut arguments, &flag)?;
                }
                "--request-timeout-seconds" => {
                    request_timeout_seconds = next_number(&mut arguments, &flag)?;
                }
                _ => bail!("unknown option {flag:?}\n{}", usage()),
            }
        }

        if qualifying_run && requests_per_concurrency < 200 {
            bail!("suite requires at least 200 completed requests per concurrency");
        }
        if requests_per_concurrency == 0 || packet_frames == 0 || ring_slots == 0 {
            bail!("request count, packet frames, and ring slots must be non-zero");
        }
        if concurrencies.contains(&0) {
            bail!("concurrency values must be non-zero");
        }
        if audio_dir.is_none() {
            save_audio_count = 0;
        }

        Ok(Self {
            qualifying_run,
            library: library.context("--library is required")?,
            model_root: model_root.context("--model-root is required")?,
            corpus: corpus.context("--corpus is required")?,
            output: output.context("--output is required")?,
            audio_dir,
            save_audio_count,
            requests_per_concurrency,
            concurrencies,
            warmup_requests,
            packet_frames,
            ring_slots,
            max_codec_frames,
            poll_timeout_milliseconds,
            request_timeout_seconds,
        })
    }
}

fn usage() -> &'static str {
    "usage: qwen3-tts-bench <suite|smoke> --library LIB --model-root DIR \\\n+     --corpus FILE.jsonl --output REPORT.json [--concurrency 1,3,6] \\\n+     [--requests-per-concurrency 200] [--audio-dir DIR --save-audio-count 3]"
}

fn next_value(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    arguments
        .next()
        .with_context(|| format!("{flag} requires a value"))
}

fn next_path(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<PathBuf> {
    next_value(arguments, flag).map(PathBuf::from)
}

fn next_number<T>(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    next_value(arguments, flag)?
        .parse::<T>()
        .with_context(|| format!("{flag} requires a valid number"))
}

fn parse_concurrencies(value: &str) -> Result<Vec<usize>> {
    let values = value
        .split(',')
        .map(str::parse::<usize>)
        .collect::<Result<BTreeSet<_>, _>>()?;
    if values.is_empty() || values.contains(&0) {
        bail!("--concurrency requires non-zero comma-separated integers");
    }
    Ok(values.into_iter().collect())
}

#[derive(Clone, Debug, Deserialize)]
struct CorpusEntry {
    id: String,
    language: String,
    text: String,
    instruct: String,
}

impl CorpusEntry {
    fn language_id(&self) -> Result<u32> {
        let id = match self.language.as_str() {
            "Auto" => 0,
            "Chinese" => 1,
            "English" => 2,
            "Japanese" => 3,
            "Korean" => 4,
            "German" => 5,
            "French" => 6,
            "Russian" => 7,
            "Portuguese" => 8,
            "Spanish" => 9,
            "Italian" => 10,
            unsupported => bail!("unsupported corpus language {unsupported:?}"),
        };
        Ok(id)
    }
}

fn load_corpus(path: &Path) -> Result<Vec<CorpusEntry>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut entries = Vec::new();
    let mut identifiers = BTreeSet::new();
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let entry = serde_json::from_str::<CorpusEntry>(&line)
            .with_context(|| format!("invalid corpus JSON on line {}", line_index + 1))?;
        if entry.id.is_empty() || entry.text.is_empty() || entry.instruct.is_empty() {
            bail!(
                "corpus line {} contains an empty required field",
                line_index + 1
            );
        }
        entry.language_id()?;
        if !identifiers.insert(entry.id.clone()) {
            bail!("duplicate corpus id {:?}", entry.id);
        }
        entries.push(entry);
    }
    if entries.is_empty() {
        bail!("corpus must contain at least one request");
    }
    Ok(entries)
}

struct ExecuteConfig<'path> {
    count: usize,
    concurrency: usize,
    packet_frames: u32,
    max_codec_frames: u32,
    poll_timeout_milliseconds: u32,
    request_timeout: Duration,
    save_audio_count: usize,
    audio_dir: Option<&'path Path>,
}

struct ActiveRequest<'api> {
    request: Request<'api>,
    ordinal: usize,
    corpus: CorpusEntry,
    started: Instant,
    caller_ttfa: Option<Duration>,
    request_id: Option<u64>,
    next_sequence: u64,
    next_frame: u64,
    next_sample: u64,
    saw_final: bool,
    exact_copy_bounds: bool,
    captured_audio: Option<Vec<i16>>,
    pcm: Vec<i16>,
}

impl ActiveRequest<'_> {
    fn accept_packet(&mut self, descriptor: ffi::AudioPacket, tail_untouched: bool) -> Result<()> {
        if descriptor.codec_frames == 0 {
            bail!("request {} emitted an empty packet", self.ordinal);
        }
        if descriptor.sample_rate != SAMPLE_RATE || descriptor.channels != 1 {
            bail!("request {} emitted non-24-kHz-mono audio", self.ordinal);
        }
        if descriptor.sample_count != descriptor.codec_frames * SAMPLES_PER_FRAME {
            bail!("request {} emitted a frame/sample mismatch", self.ordinal);
        }
        if descriptor.sequence != self.next_sequence
            || descriptor.first_codec_frame != self.next_frame
            || descriptor.first_sample != self.next_sample
        {
            bail!(
                "request {} emitted non-contiguous packet positions",
                self.ordinal
            );
        }
        match self.request_id {
            Some(request_id) if request_id != descriptor.request_id => {
                bail!("request {} changed runtime request id", self.ordinal);
            }
            None => self.request_id = Some(descriptor.request_id),
            _ => {}
        }
        if self.saw_final {
            bail!(
                "request {} emitted audio after its final packet",
                self.ordinal
            );
        }
        if self.caller_ttfa.is_none() {
            self.caller_ttfa = Some(self.started.elapsed());
        }
        self.exact_copy_bounds &= tail_untouched;
        let sample_count = descriptor.sample_count as usize;
        if let Some(audio) = &mut self.captured_audio {
            audio.extend_from_slice(&self.pcm[..sample_count]);
        }
        self.next_sequence += 1;
        self.next_frame += u64::from(descriptor.codec_frames);
        self.next_sample += u64::from(descriptor.sample_count);
        self.saw_final = descriptor.is_final != 0;
        Ok(())
    }
}

fn execute_scenario<'api>(
    api: &'api Api,
    engine: &Engine<'api>,
    corpus: &[CorpusEntry],
    config: &ExecuteConfig<'_>,
) -> Result<ScenarioReport> {
    if config.count == 0 || config.concurrency == 0 {
        bail!("scenario count and concurrency must be non-zero");
    }
    if let Some(directory) = config.audio_dir {
        fs::create_dir_all(directory)?;
    }

    let scenario_started = Instant::now();
    let rss_start = current_rss_bytes();
    let mut rss_peak = rss_start;
    let mut next_ordinal = 0_usize;
    let mut active = Vec::<ActiveRequest<'_>>::new();
    let mut completed = Vec::<RequestReport>::with_capacity(config.count);
    let mut loop_count = 0_u64;

    while completed.len() < config.count {
        while active.len() < config.concurrency && next_ordinal < config.count {
            let entry = corpus[next_ordinal % corpus.len()].clone();
            let generation = GenerationConfig::official_defaults(
                0x5157_454e_0000_0000_u64.wrapping_add(next_ordinal as u64),
                config.max_codec_frames,
            );
            match api.start_request(
                engine,
                entry.language_id()?,
                &entry.text,
                &entry.instruct,
                generation,
            )? {
                StartResult::Started(request) => {
                    let capture = next_ordinal < config.save_audio_count;
                    active.push(ActiveRequest {
                        request,
                        ordinal: next_ordinal,
                        corpus: entry,
                        started: Instant::now(),
                        caller_ttfa: None,
                        request_id: None,
                        next_sequence: 0,
                        next_frame: 0,
                        next_sample: 0,
                        saw_final: false,
                        exact_copy_bounds: true,
                        captured_audio: capture.then(Vec::new),
                        pcm: vec![0; config.packet_frames as usize * SAMPLES_PER_FRAME as usize],
                    });
                    next_ordinal += 1;
                }
                StartResult::WouldBlock => break,
            }
        }

        let mut made_progress = false;
        let mut index = 0_usize;
        while index < active.len() {
            if active[index].started.elapsed() > config.request_timeout {
                bail!(
                    "request {} exceeded the {:?} qualification timeout",
                    active[index].ordinal,
                    config.request_timeout
                );
            }
            let outcome = {
                let active_request = &mut active[index];
                active_request
                    .request
                    .poll(config.poll_timeout_milliseconds, &mut active_request.pcm)?
            };
            match outcome {
                PollResult::Packet {
                    descriptor,
                    tail_untouched,
                } => {
                    active[index].accept_packet(descriptor, tail_untouched)?;
                    made_progress = true;
                    index += 1;
                }
                PollResult::WouldBlock => index += 1,
                PollResult::EndOfStream => {
                    let mut finished = active.swap_remove(index);
                    if !finished.saw_final {
                        bail!(
                            "request {} reached EOS without a final packet",
                            finished.ordinal
                        );
                    }
                    let caller_wall = finished.started.elapsed();
                    let caller_ttfa = finished
                        .caller_ttfa
                        .context("request reached EOS without emitting audio")?;
                    let runtime_metrics = finished.request.metrics()?;
                    if runtime_metrics.emitted_packets != finished.next_sequence
                        || runtime_metrics.generated_codec_frames != finished.next_frame
                        || runtime_metrics.emitted_samples != finished.next_sample
                    {
                        bail!(
                            "request {} runtime metrics disagree with observed packets",
                            finished.ordinal
                        );
                    }
                    finished.request.destroy()?;
                    if let (Some(directory), Some(audio)) =
                        (config.audio_dir, finished.captured_audio.as_deref())
                    {
                        let filename = format!(
                            "c{}-{:04}-{}.wav",
                            config.concurrency,
                            finished.ordinal,
                            safe_filename(&finished.corpus.id)
                        );
                        wav::write_pcm16_mono(&directory.join(filename), SAMPLE_RATE, audio)?;
                    }
                    let audio_seconds = finished.next_sample as f64 / SAMPLE_RATE as f64;
                    let wall_seconds = caller_wall.as_secs_f64();
                    let progressive_streaming =
                        finished.next_sequence > 1 && caller_ttfa < caller_wall;
                    completed.push(RequestReport {
                        ordinal: finished.ordinal,
                        corpus_id: finished.corpus.id,
                        language: finished.corpus.language,
                        packets: finished.next_sequence,
                        codec_frames: finished.next_frame,
                        samples: finished.next_sample,
                        audio_seconds,
                        caller_ttfa_ms: caller_ttfa.as_secs_f64() * 1_000.0,
                        caller_wall_ms: wall_seconds * 1_000.0,
                        rtf: wall_seconds / audio_seconds,
                        progressive_streaming,
                        exact_copy_bounds: finished.exact_copy_bounds,
                        runtime: RuntimeMetricsReport::from(runtime_metrics),
                    });
                    made_progress = true;
                }
            }
        }

        loop_count += 1;
        if loop_count.is_multiple_of(32) {
            rss_peak = maximum_optional(rss_peak, current_rss_bytes());
        }
        if !made_progress {
            thread::sleep(Duration::from_micros(200));
        }
    }

    completed.sort_by_key(|request| request.ordinal);
    let wall_seconds = scenario_started.elapsed().as_secs_f64();
    let synthesized_audio_seconds = completed
        .iter()
        .map(|request| request.audio_seconds)
        .sum::<f64>();
    let caller_ttfa = completed
        .iter()
        .map(|request| request.caller_ttfa_ms)
        .collect::<Vec<_>>();
    let request_wall = completed
        .iter()
        .map(|request| request.caller_wall_ms)
        .collect::<Vec<_>>();
    let request_rtf = completed
        .iter()
        .map(|request| request.rtf)
        .collect::<Vec<_>>();
    let runtime_first_audio = completed
        .iter()
        .map(|request| request.runtime.first_audio_ms)
        .collect::<Vec<_>>();
    let runtime_prefill = completed
        .iter()
        .map(|request| request.runtime.prefill_ms)
        .collect::<Vec<_>>();
    let peak_request_device_bytes = completed
        .iter()
        .map(|request| request.runtime.peak_request_device_bytes)
        .max()
        .unwrap_or(0);
    let peak_request_host_bytes = completed
        .iter()
        .map(|request| request.runtime.peak_request_host_bytes)
        .max()
        .unwrap_or(0);
    let progressive_streaming_requests = completed
        .iter()
        .filter(|request| request.progressive_streaming)
        .count();
    let exact_copy_bound_requests = completed
        .iter()
        .filter(|request| request.exact_copy_bounds)
        .count();
    let rss_end = current_rss_bytes();
    rss_peak = maximum_optional(maximum_optional(rss_peak, rss_end), rss_start);

    Ok(ScenarioReport {
        concurrency: config.concurrency,
        requested: config.count,
        completed: completed.len(),
        failed: config.count - completed.len(),
        wall_seconds,
        synthesized_audio_seconds,
        aggregate_rtf: wall_seconds / synthesized_audio_seconds,
        requests_per_second: completed.len() as f64 / wall_seconds,
        progressive_streaming_requests,
        exact_copy_bound_requests,
        host_rss_start_bytes: rss_start,
        host_rss_peak_bytes: rss_peak,
        host_rss_end_bytes: rss_end,
        caller_ttfa_ms: Distribution::from_values(&caller_ttfa),
        request_wall_ms: Distribution::from_values(&request_wall),
        request_rtf: Distribution::from_values(&request_rtf),
        runtime_first_audio_ms: Distribution::from_values(&runtime_first_audio),
        runtime_prefill_ms: Distribution::from_values(&runtime_prefill),
        peak_request_device_bytes,
        peak_request_host_bytes,
        requests: completed,
    })
}

fn evaluate_gates(
    qualifying_run: bool,
    requests_per_concurrency: usize,
    scenarios: &[ScenarioReport],
) -> QualificationGates {
    let all_requests_completed = scenarios
        .iter()
        .all(|scenario| scenario.completed == scenario.requested && scenario.failed == 0);
    let at_least_200_requests_per_scenario = requests_per_concurrency >= 200
        && scenarios.iter().all(|scenario| scenario.completed >= 200);
    let progressive_streaming_observed = scenarios
        .iter()
        .all(|scenario| scenario.progressive_streaming_requests * 100 >= scenario.completed * 95);
    let packet_positions_contiguous = all_requests_completed;
    let exact_pcm_copy_bounds = scenarios
        .iter()
        .all(|scenario| scenario.exact_copy_bound_requests == scenario.completed);
    let rtf_below_one_all_scenarios = scenarios
        .iter()
        .all(|scenario| scenario.aggregate_rtf < 1.0 && scenario.request_rtf.p95 < 1.0);
    let first_audio_p95_below_200_ms_all_scenarios = scenarios
        .iter()
        .all(|scenario| scenario.caller_ttfa_ms.p95 < 200.0);
    let passed = qualifying_run
        && all_requests_completed
        && at_least_200_requests_per_scenario
        && progressive_streaming_observed
        && packet_positions_contiguous
        && exact_pcm_copy_bounds
        && rtf_below_one_all_scenarios
        && first_audio_p95_below_200_ms_all_scenarios;
    QualificationGates {
        all_requests_completed,
        at_least_200_requests_per_scenario,
        progressive_streaming_observed,
        packet_positions_contiguous,
        exact_pcm_copy_bounds,
        rtf_below_one_all_scenarios,
        first_audio_p95_below_200_ms_all_scenarios,
        passed,
    }
}

fn current_rss_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
    let kilobytes = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    kilobytes.checked_mul(1_024)
}

fn maximum_optional(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn safe_filename(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '-'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{evaluate_gates, parse_concurrencies, safe_filename};

    #[test]
    fn concurrency_parser_deduplicates_and_sorts() {
        assert_eq!(parse_concurrencies("6,1,3,3").unwrap(), vec![1, 3, 6]);
        assert!(parse_concurrencies("0,1").is_err());
    }

    #[test]
    fn filenames_cannot_escape_the_audio_directory() {
        assert_eq!(safe_filename("../de test"), "---de-test");
    }

    #[test]
    fn empty_smoke_result_never_claims_qualification() {
        let gates = evaluate_gates(false, 3, &[]);
        assert!(!gates.passed);
        assert!(!gates.at_least_200_requests_per_scenario);
    }
}
