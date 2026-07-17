use std::path::PathBuf;

use clap::Parser;
use qwen3_tts_http_bench::{BackendProfile, BenchmarkConfig, Concurrency, run_benchmark};

#[derive(Debug, Parser)]
#[command(
    name = "qwen3-tts-http-bench",
    version,
    about = "Reproducible external HTTP benchmark for native Qwen3-TTS and SGLang-Omni"
)]
struct Cli {
    /// Loopback HTTP endpoint, including the request path and explicit port.
    #[arg(long)]
    endpoint: String,

    /// Target API contract: native or sglang-omni.
    #[arg(long, default_value = "native")]
    profile: BackendProfile,

    /// Served Qwen3-TTS `VoiceDesign` model identifier (required for sglang-omni).
    #[arg(long)]
    sglang_model: Option<String>,

    /// Deterministic UTF-8 JSONL workload.
    #[arg(long)]
    workload: PathBuf,

    /// New directory (or empty directory) receiving requests.jsonl, packets.jsonl, and summary.json.
    #[arg(long)]
    output_dir: PathBuf,

    /// Optional create-new JSONL file for externally alignable phase boundaries.
    #[arg(long)]
    phase_events: Option<PathBuf>,

    /// Number of measured requests; workload entries repeat in file order.
    #[arg(long)]
    requests: usize,

    /// Number of unreported warmup requests.
    #[arg(long, default_value_t = 0)]
    warmups: usize,

    /// Synchronized request batch width: B1, B3, or B6.
    #[arg(long, default_value = "B1")]
    concurrency: Concurrency,

    /// Per-connection and per-request timeout in seconds.
    #[arg(long, default_value_t = 600)]
    timeout_seconds: u64,

    /// Include raw prompt and voice-description text in requests.jsonl.
    #[arg(long, default_value_t = false)]
    log_prompt_text: bool,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let config = BenchmarkConfig {
        endpoint: cli.endpoint,
        profile: cli.profile,
        sglang_model: cli.sglang_model,
        workload_path: cli.workload,
        output_dir: cli.output_dir,
        phase_events_path: cli.phase_events,
        requests: cli.requests,
        warmups: cli.warmups,
        concurrency: cli.concurrency,
        timeout_seconds: cli.timeout_seconds,
        log_prompt_text: cli.log_prompt_text,
    };
    if let Err(error) = run_benchmark(config).await {
        eprintln!("benchmark failed: {error}");
        std::process::exit(1);
    }
}
