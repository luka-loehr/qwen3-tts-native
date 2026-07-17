use std::mem::size_of;
use std::time::Duration;

use qwen3_tts_runtime::{
    EngineConfig, MAX_CODEC_FRAMES, MAX_CONCURRENT_REQUESTS, MAX_INSTRUCT_BYTES, MAX_TEXT_BYTES,
};

pub const DEFAULT_MAX_TEXT_BYTES: usize = 32 * 1024;
pub const DEFAULT_MAX_VOICE_DESCRIPTION_BYTES: usize = 4 * 1024;
pub const DEFAULT_MAX_DURATION_SECONDS: f64 = 120.0;
pub const INTRINSIC_MAX_DURATION_SECONDS: f64 = 655.36;

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub max_concurrent_requests: u32,
    pub max_text_bytes: usize,
    pub max_voice_description_bytes: usize,
    pub max_duration_seconds: f64,
    pub default_duration_seconds: f64,
    pub poll_timeout: Duration,
    pub slow_client_timeout: Duration,
    pub retirement_timeout: Duration,
    pub shutdown_timeout: Duration,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: EngineConfig::default().max_concurrent_requests,
            max_text_bytes: DEFAULT_MAX_TEXT_BYTES,
            max_voice_description_bytes: DEFAULT_MAX_VOICE_DESCRIPTION_BYTES,
            max_duration_seconds: DEFAULT_MAX_DURATION_SECONDS,
            default_duration_seconds: DEFAULT_MAX_DURATION_SECONDS,
            poll_timeout: Duration::from_millis(100),
            slow_client_timeout: Duration::from_secs(5),
            retirement_timeout: Duration::from_secs(25),
            shutdown_timeout: Duration::from_secs(35),
        }
    }
}

impl ServerConfig {
    /// Checks deployment limits against intrinsic native runtime limits.
    ///
    /// # Errors
    ///
    /// Returns a configuration error for zero, non-finite, or out-of-range values.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_concurrent_requests == 0
            || self.max_concurrent_requests > MAX_CONCURRENT_REQUESTS
        {
            return Err(format!(
                "max_concurrent_requests must be in 1..={MAX_CONCURRENT_REQUESTS}"
            ));
        }
        if self.max_text_bytes == 0 || self.max_text_bytes > MAX_TEXT_BYTES as usize {
            return Err(format!("max_text_bytes must be in 1..={MAX_TEXT_BYTES}"));
        }
        if self.max_voice_description_bytes == 0
            || self.max_voice_description_bytes > MAX_INSTRUCT_BYTES as usize
        {
            return Err(format!(
                "max_voice_description_bytes must be in 1..={MAX_INSTRUCT_BYTES}"
            ));
        }
        if !self.max_duration_seconds.is_finite()
            || self.max_duration_seconds < 0.08
            || self.max_duration_seconds > INTRINSIC_MAX_DURATION_SECONDS
        {
            return Err(format!(
                "max_duration_seconds must be in 0.08..={INTRINSIC_MAX_DURATION_SECONDS}"
            ));
        }
        if !self.default_duration_seconds.is_finite()
            || self.default_duration_seconds < 0.08
            || self.default_duration_seconds > self.max_duration_seconds
        {
            return Err(
                "default_duration_seconds must be finite and inside the configured duration limit"
                    .to_owned(),
            );
        }
        if self.poll_timeout.is_zero()
            || self.slow_client_timeout.is_zero()
            || self.retirement_timeout.is_zero()
            || self.shutdown_timeout.is_zero()
        {
            return Err("server timeouts must be non-zero".to_owned());
        }
        let cleanup_budget = self
            .slow_client_timeout
            .checked_add(self.retirement_timeout)
            .ok_or_else(|| "server cleanup timeout budget overflowed".to_owned())?;
        if cleanup_budget >= self.shutdown_timeout {
            return Err(
                "slow_client_timeout + retirement_timeout must be below shutdown_timeout"
                    .to_owned(),
            );
        }
        let maximum_frames = duration_to_frames(self.max_duration_seconds);
        if maximum_frames == 0 || maximum_frames > MAX_CODEC_FRAMES {
            return Err("duration limit exceeds the native frame ceiling".to_owned());
        }
        Ok(())
    }

    #[must_use]
    pub fn max_body_bytes(&self) -> usize {
        self.max_text_bytes
            .saturating_add(self.max_voice_description_bytes)
            .saturating_add(16 * 1024)
    }

    #[must_use]
    pub fn max_buffered_pcm_bytes(&self) -> usize {
        usize::try_from(duration_to_frames(self.max_duration_seconds))
            .unwrap_or(usize::MAX)
            .saturating_mul(qwen3_tts_runtime::SAMPLES_PER_CODEC_FRAME as usize)
            .saturating_mul(size_of::<i16>())
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn duration_to_frames(seconds: f64) -> u32 {
    let frames = (seconds * 12.5).ceil();
    if !frames.is_finite() || frames <= 0.0 {
        return 0;
    }
    if frames >= f64::from(u32::MAX) {
        return u32::MAX;
    }
    frames as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_conversion_uses_the_native_twelve_point_five_hertz_cadence() {
        assert_eq!(duration_to_frames(0.08), 1);
        assert_eq!(duration_to_frames(0.081), 2);
        assert_eq!(duration_to_frames(120.0), 1_500);
        assert_eq!(duration_to_frames(INTRINSIC_MAX_DURATION_SECONDS), 8_192);
    }

    #[test]
    fn default_limits_are_inside_the_native_contract() {
        let config = ServerConfig::default();
        config.validate().unwrap();
        assert_eq!(config.max_buffered_pcm_bytes(), 5_760_000);
    }

    #[test]
    fn cleanup_budget_must_fit_inside_hard_shutdown_deadline() {
        let config = ServerConfig {
            slow_client_timeout: Duration::from_secs(10),
            retirement_timeout: Duration::from_secs(25),
            shutdown_timeout: Duration::from_secs(35),
            ..ServerConfig::default()
        };
        assert_eq!(
            config.validate().unwrap_err(),
            "slow_client_timeout + retirement_timeout must be below shutdown_timeout"
        );
    }
}
