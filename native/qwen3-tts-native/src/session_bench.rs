use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;

use crate::native_talker::{
    NativeTalkerModel, NativeTalkerSession, SamplingConfig, SessionEndReason, SessionMemory,
    SessionRuntimeState, SessionStartTiming, SharedModelMemory, VoiceDesignRequest,
};

const CODEBOOKS: usize = 16;
const AUDIO_SECONDS_PER_FRAME: f64 = 0.08;
const DEFAULT_WARM_REQUESTS: usize = 20;
const DEFAULT_BATCH_ROUNDS: usize = 2;
const DEFAULT_MAX_FRAMES: usize = 256;
const DEFAULT_MAX_SEQUENCE_LENGTH: usize = 1_024;
const TTFA_GATE_MILLISECONDS: f64 = 200.0;
const SAMPLE_SEEDS: [u64; 6] = [11, 22, 11, 33, 44, 55];

#[derive(Clone)]
struct RequestSpec {
    text: String,
    instruction: String,
    language: String,
    max_frames: usize,
    max_sequence_length: usize,
    seed: u64,
    greedy: bool,
}

impl RequestSpec {
    fn request(&self) -> VoiceDesignRequest {
        let mut request = VoiceDesignRequest::new(
            self.text.clone(),
            self.instruction.clone(),
            self.language.clone(),
        );
        request.max_frames = self.max_frames;
        request.max_sequence_length = self.max_sequence_length;
        request.random_seed = self.seed;
        if self.greedy {
            request.talker_sampling = SamplingConfig::greedy();
            request.predictor_sampling = SamplingConfig::greedy();
        }
        request
    }
}

struct RequestResult {
    codes: Vec<u16>,
    frame_count: usize,
    end_reason: Option<SessionEndReason>,
    ttfa_wall_milliseconds: f64,
    request_wall_milliseconds: f64,
    start_timing: SessionStartTiming,
    memory: SessionMemory,
    runtime_state: SessionRuntimeState,
}

#[derive(Serialize)]
struct RequestSummary {
    seed: u64,
    frame_count: usize,
    ended_by_eos: bool,
    codec_hash_fnv1a64: String,
    ttfa_wall_milliseconds: f64,
    request_wall_milliseconds: f64,
    start_timing: SessionStartTiming,
    session_memory: SessionMemory,
    runtime_state: SessionRuntimeState,
}

#[derive(Serialize)]
struct BatchRoundReport {
    wall_milliseconds: f64,
    aggregate_audio_seconds: f64,
    aggregate_realtime_factor: f64,
    all_pool_hits: bool,
    requests: Vec<RequestSummary>,
}

#[derive(Serialize)]
struct BatchReport {
    concurrency: usize,
    rounds: usize,
    greedy: bool,
    exact_reference_parity: bool,
    aggregate_session_bytes: u64,
    rounds_report: Vec<BatchRoundReport>,
}

#[derive(Serialize)]
struct WarmLatencyReport {
    request_count: usize,
    pool_hit_count: usize,
    all_pool_hits: bool,
    exact_reference_parity: bool,
    ttfa_p50_milliseconds: f64,
    ttfa_p95_milliseconds: f64,
    ttfa_p99_milliseconds: f64,
    ttfa_max_milliseconds: f64,
    gate_milliseconds: f64,
    gate_passed: bool,
    samples_milliseconds: Vec<f64>,
    start_timing_samples: Vec<SessionStartTiming>,
    tokenize_p95_milliseconds: f64,
    prompt_plan_p95_milliseconds: f64,
    session_acquire_p95_milliseconds: f64,
    session_create_p95_milliseconds: f64,
    session_reset_p95_milliseconds: f64,
    prefill_p95_milliseconds: f64,
}

#[derive(Serialize)]
struct QualificationReport {
    at_least_twenty_warm_requests: bool,
    warm_ttfa_p95_under_200ms: bool,
    sampled_seed_isolation: bool,
    greedy_b3_exact: bool,
    concurrent_b1_b3_b6_exact: bool,
    corpus_completed_without_max_frame_truncation: bool,
    cancel_drop_isolated: bool,
    round_robin_b3_exact: bool,
    session_thread_move_exact: bool,
}

#[derive(Serialize)]
struct LifecycleReport {
    cancel_drop_isolated: bool,
    round_robin_b3_exact: bool,
    session_thread_move_exact: bool,
    caller_arc_dropped_before_thread_completion: bool,
}

#[derive(Serialize)]
struct SessionBenchmarkReport {
    model_load_wall_milliseconds: f64,
    shared_model_memory: SharedModelMemory,
    requested_max_frames: usize,
    maximum_sequence_length: usize,
    warm_latency: WarmLatencyReport,
    sampled_batches: Vec<BatchReport>,
    greedy_b3: BatchReport,
    lifecycle: LifecycleReport,
    qualification: QualificationReport,
}

pub fn run_session_benchmark(mut arguments: impl Iterator<Item = OsString>) -> Result<()> {
    let library = next_path(
        &mut arguments,
        "benchmark-sessions requires a shared-library path",
    )?;
    let model_directory = next_path(
        &mut arguments,
        "benchmark-sessions requires a model directory",
    )?;
    let mut text = "Hallo, dies ist ein nativer Parallelitaetstest.".to_owned();
    let mut instruction = "A calm male voice, clear, relaxed, and natural.".to_owned();
    let mut language = "German".to_owned();
    let mut output = None;
    let mut warm_requests = DEFAULT_WARM_REQUESTS;
    let mut rounds = DEFAULT_BATCH_ROUNDS;
    let mut max_frames = DEFAULT_MAX_FRAMES;
    let mut max_sequence_length = DEFAULT_MAX_SEQUENCE_LENGTH;

    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--text") => text = next_string(&mut arguments, "--text")?,
            Some("--instruction") => {
                instruction = next_string(&mut arguments, "--instruction")?;
            }
            Some("--language") => language = next_string(&mut arguments, "--language")?,
            Some("--output") => {
                output = Some(next_path(&mut arguments, "--output requires a path")?)
            }
            Some("--warm-requests") => {
                warm_requests = next_usize(&mut arguments, "--warm-requests")?;
            }
            Some("--rounds") => rounds = next_usize(&mut arguments, "--rounds")?,
            Some("--max-frames") => {
                max_frames = next_usize(&mut arguments, "--max-frames")?;
            }
            Some("--max-sequence") => {
                max_sequence_length = next_usize(&mut arguments, "--max-sequence")?;
            }
            _ => bail!("unknown benchmark-sessions argument {argument:?}"),
        }
    }
    ensure!(warm_requests > 0, "--warm-requests must be positive");
    ensure!(rounds > 0, "--rounds must be positive");
    ensure!(max_frames > 0, "--max-frames must be positive");

    let base = RequestSpec {
        text,
        instruction,
        language,
        max_frames,
        max_sequence_length,
        seed: SAMPLE_SEEDS[0],
        greedy: false,
    };

    let model_load_started = Instant::now();
    let model = NativeTalkerModel::load(&library, &model_directory, 0)?;
    let model_load_wall_milliseconds = model_load_started.elapsed().as_secs_f64() * 1_000.0;

    let reference = run_request(Arc::clone(&model), base.clone())?;
    ensure_completed(&reference, &base)?;
    let warm_latency = measure_warm_latency(
        &model,
        &base,
        &reference,
        warm_requests,
        TTFA_GATE_MILLISECONDS,
    )?;

    let seed_22 = RequestSpec {
        seed: SAMPLE_SEEDS[1],
        ..base.clone()
    };
    let seed_22_reference = run_request(Arc::clone(&model), seed_22.clone())?;
    ensure_completed(&seed_22_reference, &seed_22)?;
    let sampled_seed_isolation = reference.codes != seed_22_reference.codes;
    ensure!(
        sampled_seed_isolation,
        "sampled seed isolation failed: seeds 11 and 22 returned identical codec sequences"
    );

    let mut sampled_batches = Vec::new();
    for concurrency in [1_usize, 3, 6] {
        sampled_batches.push(benchmark_batch(&model, &base, concurrency, rounds, false)?);
    }
    let greedy_b3 = benchmark_batch(&model, &base, 3, rounds, true)?;
    let lifecycle = verify_lifecycle_isolation(&model, &base)?;

    let corpus_completed_without_max_frame_truncation = sampled_batches
        .iter()
        .flat_map(|batch| &batch.rounds_report)
        .flat_map(|round| &round.requests)
        .chain(
            greedy_b3
                .rounds_report
                .iter()
                .flat_map(|round| &round.requests),
        )
        .all(|request| request.ended_by_eos);
    let report = SessionBenchmarkReport {
        model_load_wall_milliseconds,
        shared_model_memory: model.shared_memory(),
        requested_max_frames: max_frames,
        maximum_sequence_length: max_sequence_length,
        qualification: QualificationReport {
            at_least_twenty_warm_requests: warm_requests >= 20,
            warm_ttfa_p95_under_200ms: warm_latency.gate_passed,
            sampled_seed_isolation,
            greedy_b3_exact: greedy_b3.exact_reference_parity,
            concurrent_b1_b3_b6_exact: sampled_batches
                .iter()
                .all(|batch| batch.exact_reference_parity),
            corpus_completed_without_max_frame_truncation,
            cancel_drop_isolated: lifecycle.cancel_drop_isolated,
            round_robin_b3_exact: lifecycle.round_robin_b3_exact,
            session_thread_move_exact: lifecycle.session_thread_move_exact,
        },
        warm_latency,
        sampled_batches,
        greedy_b3,
        lifecycle,
    };
    let encoded = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output {
        fs::write(&path, encoded).with_context(|| format!("failed to write {}", path.display()))?;
    } else {
        println!("{encoded}");
    }
    Ok(())
}

struct InterleavedSession {
    reference_index: usize,
    session: NativeTalkerSession,
    codes: Vec<u16>,
    completed: bool,
}

fn verify_lifecycle_isolation(
    model: &Arc<NativeTalkerModel>,
    base: &RequestSpec,
) -> Result<LifecycleReport> {
    let specs: Vec<_> = [11_u64, 22, 33, 44]
        .into_iter()
        .map(|seed| RequestSpec {
            seed,
            ..base.clone()
        })
        .collect();
    let mut references = Vec::with_capacity(specs.len());
    for spec in &specs {
        let result = run_request(Arc::clone(model), spec.clone())?;
        ensure_completed(&result, spec)?;
        references.push(result);
    }

    let mut sessions: Vec<_> = specs
        .iter()
        .enumerate()
        .map(|(reference_index, spec)| {
            Ok(InterleavedSession {
                reference_index,
                session: model.start(spec.request())?,
                codes: Vec::with_capacity(spec.max_frames * CODEBOOKS),
                completed: false,
            })
        })
        .collect::<Result<_>>()?;
    let mut cancelled = sessions.remove(1);
    cancelled.session.cancel();
    ensure!(
        cancelled.session.end_reason() == Some(SessionEndReason::Cancelled),
        "cancelled session did not expose the cancelled end reason"
    );
    drop(cancelled);

    while sessions.iter().any(|entry| !entry.completed) {
        for entry in &mut sessions {
            if entry.completed {
                continue;
            }
            match entry.session.next_frame()? {
                Some(frame) => entry.codes.extend_from_slice(&frame.codes),
                None => entry.completed = true,
            }
        }
    }
    for entry in &sessions {
        ensure!(
            entry.session.end_reason() == Some(SessionEndReason::CodecEos),
            "round-robin session {} did not complete with EOS",
            entry.reference_index
        );
        ensure!(
            entry.codes == references[entry.reference_index].codes,
            "round-robin session {} diverged after cancelling a sibling",
            entry.reference_index
        );
    }

    let move_spec = RequestSpec {
        seed: 55,
        ..base.clone()
    };
    let move_reference = run_request(Arc::clone(model), move_spec.clone())?;
    ensure_completed(&move_reference, &move_spec)?;
    let caller_model = Arc::clone(model);
    let moved_session = caller_model.start(move_spec.request())?;
    drop(caller_model);
    let moved = thread::spawn(move || finish_existing_session(moved_session))
        .join()
        .map_err(|_| anyhow::anyhow!("moved native session worker panicked"))??;
    ensure!(
        moved.0 == move_reference.codes,
        "a session moved to another host thread diverged from its reference"
    );
    ensure!(
        moved.1 == Some(SessionEndReason::CodecEos),
        "a session moved to another host thread did not finish with EOS"
    );

    Ok(LifecycleReport {
        cancel_drop_isolated: true,
        round_robin_b3_exact: true,
        session_thread_move_exact: true,
        caller_arc_dropped_before_thread_completion: true,
    })
}

fn finish_existing_session(
    mut session: NativeTalkerSession,
) -> Result<(Vec<u16>, Option<SessionEndReason>)> {
    let mut codes = Vec::new();
    while let Some(frame) = session.next_frame()? {
        codes.extend_from_slice(&frame.codes);
    }
    Ok((codes, session.end_reason()))
}

fn measure_warm_latency(
    model: &Arc<NativeTalkerModel>,
    spec: &RequestSpec,
    reference: &RequestResult,
    request_count: usize,
    gate_milliseconds: f64,
) -> Result<WarmLatencyReport> {
    let mut samples = Vec::with_capacity(request_count);
    let mut start_timing_samples = Vec::with_capacity(request_count);
    let mut all_pool_hits = true;
    for index in 0..request_count {
        let result = run_request(Arc::clone(model), spec.clone())?;
        ensure_completed(&result, spec)?;
        ensure_same_output(reference, &result, &format!("warm request {index}"))?;
        all_pool_hits &= result.start_timing.session_pool_hit;
        samples.push(result.ttfa_wall_milliseconds);
        start_timing_samples.push(result.start_timing);
    }
    let p50 = percentile(&samples, 0.50);
    let p95 = percentile(&samples, 0.95);
    let p99 = percentile(&samples, 0.99);
    let maximum = samples.iter().copied().fold(0.0_f64, f64::max);
    let timing_percentile = |value: fn(&SessionStartTiming) -> f64| {
        percentile(
            &start_timing_samples.iter().map(value).collect::<Vec<_>>(),
            0.95,
        )
    };
    Ok(WarmLatencyReport {
        request_count,
        pool_hit_count: start_timing_samples
            .iter()
            .filter(|timing| timing.session_pool_hit)
            .count(),
        all_pool_hits,
        exact_reference_parity: true,
        ttfa_p50_milliseconds: p50,
        ttfa_p95_milliseconds: p95,
        ttfa_p99_milliseconds: p99,
        ttfa_max_milliseconds: maximum,
        gate_milliseconds,
        gate_passed: request_count >= 20 && all_pool_hits && p95 < gate_milliseconds,
        samples_milliseconds: samples,
        tokenize_p95_milliseconds: timing_percentile(|timing| timing.tokenize_wall_milliseconds),
        prompt_plan_p95_milliseconds: timing_percentile(|timing| {
            timing.prompt_plan_wall_milliseconds
        }),
        session_acquire_p95_milliseconds: timing_percentile(|timing| {
            timing.session_acquire_wall_milliseconds
        }),
        session_create_p95_milliseconds: timing_percentile(|timing| {
            timing.session_create_wall_milliseconds
        }),
        session_reset_p95_milliseconds: timing_percentile(|timing| {
            timing.session_reset_wall_milliseconds
        }),
        prefill_p95_milliseconds: timing_percentile(|timing| timing.prefill_wall_milliseconds),
        start_timing_samples,
    })
}

fn benchmark_batch(
    model: &Arc<NativeTalkerModel>,
    base: &RequestSpec,
    concurrency: usize,
    rounds: usize,
    greedy: bool,
) -> Result<BatchReport> {
    let specs: Vec<_> = (0..concurrency)
        .map(|index| RequestSpec {
            seed: SAMPLE_SEEDS[index],
            greedy,
            ..base.clone()
        })
        .collect();
    let mut references = Vec::with_capacity(concurrency);
    for spec in &specs {
        let reference = run_request(Arc::clone(model), spec.clone())?;
        ensure_completed(&reference, spec)?;
        references.push(reference);
    }

    let (warmup, _) = run_concurrent(model, &specs)?;
    compare_batch(&references, &warmup, "concurrent warmup")?;

    let mut rounds_report = Vec::with_capacity(rounds);
    let mut aggregate_session_bytes = 0_u64;
    for round_index in 0..rounds {
        let (results, wall_milliseconds) = run_concurrent(model, &specs)?;
        compare_batch(
            &references,
            &results,
            &format!("concurrent measured round {round_index}"),
        )?;
        if round_index == 0 {
            aggregate_session_bytes = results
                .iter()
                .map(|result| session_bytes(result.memory))
                .sum();
        }
        let aggregate_audio_seconds = results
            .iter()
            .map(|result| result.frame_count as f64 * AUDIO_SECONDS_PER_FRAME)
            .sum::<f64>();
        let aggregate_realtime_factor =
            (wall_milliseconds / 1_000.0) / aggregate_audio_seconds.max(f64::EPSILON);
        let requests = results
            .into_iter()
            .zip(&specs)
            .map(|(result, spec)| summarize_request(spec.seed, result))
            .collect();
        rounds_report.push(BatchRoundReport {
            wall_milliseconds,
            aggregate_audio_seconds,
            aggregate_realtime_factor,
            all_pool_hits: true,
            requests,
        });
    }
    let all_pool_hits = rounds_report
        .iter()
        .flat_map(|round| &round.requests)
        .all(|request| request.start_timing.session_pool_hit);
    ensure!(
        all_pool_hits,
        "measured B{concurrency} run missed the session pool"
    );
    for round in &mut rounds_report {
        round.all_pool_hits = all_pool_hits;
    }
    Ok(BatchReport {
        concurrency,
        rounds,
        greedy,
        exact_reference_parity: true,
        aggregate_session_bytes,
        rounds_report,
    })
}

fn run_concurrent(
    model: &Arc<NativeTalkerModel>,
    specs: &[RequestSpec],
) -> Result<(Vec<RequestResult>, f64)> {
    let barrier = Arc::new(Barrier::new(specs.len() + 1));
    let mut workers = Vec::with_capacity(specs.len());
    for spec in specs.iter().cloned() {
        let worker_model = Arc::clone(model);
        let worker_barrier = Arc::clone(&barrier);
        workers.push(thread::spawn(move || {
            worker_barrier.wait();
            run_request(worker_model, spec)
        }));
    }
    let batch_started = Instant::now();
    barrier.wait();
    let mut results = Vec::with_capacity(workers.len());
    for worker in workers {
        results.push(
            worker
                .join()
                .map_err(|_| anyhow::anyhow!("native session worker panicked"))??,
        );
    }
    Ok((results, batch_started.elapsed().as_secs_f64() * 1_000.0))
}

fn run_request(model: Arc<NativeTalkerModel>, spec: RequestSpec) -> Result<RequestResult> {
    let request_started = Instant::now();
    let mut session = model.start(spec.request())?;
    let start_timing = session.start_timing();
    let memory = session.memory_usage();
    let mut codes = Vec::with_capacity(spec.max_frames * CODEBOOKS);
    let first = session
        .next_frame()?
        .context("session ended during prefill without producing an audio frame")?;
    let ttfa_wall_milliseconds = request_started.elapsed().as_secs_f64() * 1_000.0;
    codes.extend_from_slice(&first.codes);
    while let Some(frame) = session.next_frame()? {
        codes.extend_from_slice(&frame.codes);
    }
    let runtime_state = session.runtime_state()?;
    let end_reason = session.end_reason();
    Ok(RequestResult {
        frame_count: codes.len() / CODEBOOKS,
        codes,
        end_reason,
        ttfa_wall_milliseconds,
        request_wall_milliseconds: request_started.elapsed().as_secs_f64() * 1_000.0,
        start_timing,
        memory,
        runtime_state,
    })
}

fn ensure_completed(result: &RequestResult, spec: &RequestSpec) -> Result<()> {
    ensure!(
        result.end_reason == Some(SessionEndReason::CodecEos),
        "seed {} hit {:?} after {} frames; increase --max-frames above {} to avoid truncation",
        spec.seed,
        result.end_reason,
        result.frame_count,
        spec.max_frames
    );
    Ok(())
}

fn compare_batch(
    references: &[RequestResult],
    concurrent: &[RequestResult],
    label: &str,
) -> Result<()> {
    ensure!(
        references.len() == concurrent.len(),
        "{label} returned the wrong number of sessions"
    );
    for (index, (reference, candidate)) in references.iter().zip(concurrent).enumerate() {
        ensure_same_output(reference, candidate, &format!("{label}, session {index}"))?;
    }
    Ok(())
}

fn ensure_same_output(
    reference: &RequestResult,
    candidate: &RequestResult,
    label: &str,
) -> Result<()> {
    ensure!(
        reference.codes == candidate.codes,
        "{label} codec mismatch: reference hash {}, candidate hash {}",
        codec_hash(&reference.codes),
        codec_hash(&candidate.codes)
    );
    ensure!(
        reference.end_reason == candidate.end_reason,
        "{label} ended differently"
    );
    Ok(())
}

fn summarize_request(seed: u64, result: RequestResult) -> RequestSummary {
    RequestSummary {
        seed,
        frame_count: result.frame_count,
        ended_by_eos: result.end_reason == Some(SessionEndReason::CodecEos),
        codec_hash_fnv1a64: codec_hash(&result.codes),
        ttfa_wall_milliseconds: result.ttfa_wall_milliseconds,
        request_wall_milliseconds: result.request_wall_milliseconds,
        start_timing: result.start_timing,
        session_memory: result.memory,
        runtime_state: result.runtime_state,
    }
}

fn session_bytes(memory: SessionMemory) -> u64 {
    memory.talker_kv_bytes + memory.predictor_kv_bytes + memory.workspace_bytes
}

fn codec_hash(codes: &[u16]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for code in codes {
        for byte in code.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    format!("{hash:016x}")
}

fn percentile(samples: &[f64], quantile: f64) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = ((sorted.len() as f64 * quantile).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len().saturating_sub(1));
    sorted[rank]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_uses_nearest_rank() {
        let values: Vec<_> = (1..=20).map(f64::from).collect();
        assert_eq!(percentile(&values, 0.50), 10.0);
        assert_eq!(percentile(&values, 0.95), 19.0);
        assert_eq!(percentile(&values, 0.99), 20.0);
    }

    #[test]
    fn codec_hash_is_stable_and_order_sensitive() {
        assert_eq!(codec_hash(&[1, 2, 3]), "3b408ad7e81440fd");
        assert_ne!(codec_hash(&[1, 2, 3]), codec_hash(&[3, 2, 1]));
    }
}
