use std::env;
use std::error::Error;
use std::future::IntoFuture;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use qwen3_tts_runtime::EngineConfig;
use qwen3_tts_server::{
    NativeEngineConfig, NativeRuntimeEngine, ServerConfig, ShutdownController,
    build_router_with_shutdown,
};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let bind: SocketAddr = env_or("QWEN3_TTS_BIND", "127.0.0.1:8080").parse()?;
    let model_root = required_path("QWEN3_TTS_MODEL_ROOT")?;
    let talker_library = required_path("QWEN3_TTS_TALKER_LIBRARY")?;
    let codec_library = required_path("QWEN3_TTS_CODEC_LIBRARY")?;

    let mut runtime = EngineConfig::default();
    runtime.device_index = parse_env("QWEN3_TTS_DEVICE_INDEX", runtime.device_index)?;
    runtime.max_concurrent_requests = parse_env(
        "QWEN3_TTS_MAX_CONCURRENT_REQUESTS",
        runtime.max_concurrent_requests,
    )?;

    let mut server = ServerConfig::default();
    server.max_text_bytes = parse_env("QWEN3_TTS_MAX_TEXT_BYTES", server.max_text_bytes)?;
    server.max_voice_description_bytes = parse_env(
        "QWEN3_TTS_MAX_VOICE_DESCRIPTION_BYTES",
        server.max_voice_description_bytes,
    )?;
    server.max_duration_seconds = parse_env(
        "QWEN3_TTS_MAX_DURATION_SECONDS",
        server.max_duration_seconds,
    )?;
    server.max_concurrent_requests = runtime.max_concurrent_requests;
    server.default_duration_seconds = server
        .default_duration_seconds
        .min(server.max_duration_seconds);
    server.validate().map_err(invalid_config)?;
    runtime.max_text_bytes = u32::try_from(server.max_text_bytes)?;
    runtime.max_instruct_bytes = u32::try_from(server.max_voice_description_bytes)?;

    let native_config = NativeEngineConfig {
        talker_library,
        codec_library,
        model_root,
        runtime,
    };
    let engine =
        tokio::task::spawn_blocking(move || NativeRuntimeEngine::load(&native_config)).await??;
    let shutdown_timeout = server.shutdown_timeout;
    let shutdown = ShutdownController::new();
    let app = build_router_with_shutdown(Arc::new(engine), server, shutdown.clone())
        .map_err(invalid_config)?;
    let listener = tokio::net::TcpListener::bind(bind).await?;
    info!(%bind, model = qwen3_tts_server::MODEL_ID, "native VoiceDesign server ready");
    let signal_shutdown = shutdown.clone();
    tokio::spawn(async move {
        shutdown_signal().await;
        install_hard_shutdown_watchdog(shutdown_timeout);
        signal_shutdown.cancel();
    });
    let graceful_shutdown = {
        let shutdown = shutdown.clone();
        async move { shutdown.cancelled().await }
    };
    let server = axum::serve(listener, app)
        .with_graceful_shutdown(graceful_shutdown)
        .into_future();
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => result?,
        () = shutdown.cancelled() => {
            tokio::time::timeout(shutdown_timeout, &mut server)
                .await
                .map_err(|_| std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "HTTP server exceeded its bounded shutdown deadline",
                ))??;
        }
    }
    Ok(())
}

fn install_hard_shutdown_watchdog(timeout: Duration) {
    if std::thread::Builder::new()
        .name("qwen3-tts-shutdown-watchdog".to_owned())
        .spawn(move || {
            std::thread::sleep(timeout);
            eprintln!(
                "native VoiceDesign server exceeded its {timeout:?} shutdown deadline; forcing process exit"
            );
            std::process::exit(124);
        })
        .is_err()
    {
        eprintln!("failed to start the hard shutdown watchdog; forcing process exit");
        std::process::exit(125);
    }
}

async fn shutdown_signal() {
    let interrupt = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = interrupt => {},
        () = terminate => {},
    }
}

fn required_path(name: &str) -> Result<PathBuf, Box<dyn Error>> {
    let value = env::var_os(name).ok_or_else(|| format!("{name} is required"))?;
    if value.is_empty() {
        return Err(format!("{name} must not be empty").into());
    }
    Ok(PathBuf::from(value))
}

fn env_or(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_owned())
}

fn parse_env<T>(name: &str, default: T) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    match env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn invalid_config(message: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message)
}
