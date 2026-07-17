use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
pub struct ServiceMetrics {
    requests_total: AtomicU64,
    active_requests: AtomicU64,
    streaming_requests: AtomicU64,
    buffered_requests: AtomicU64,
    completed_requests: AtomicU64,
    failed_requests: AtomicU64,
    cancelled_requests: AtomicU64,
    rejected_requests: AtomicU64,
    retirement_timeouts: AtomicU64,
    emitted_samples: AtomicU64,
}

impl ServiceMetrics {
    pub fn request_started(&self, streaming: bool) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.active_requests.fetch_add(1, Ordering::Relaxed);
        if streaming {
            self.streaming_requests.fetch_add(1, Ordering::Relaxed);
        } else {
            self.buffered_requests.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn request_completed(&self, emitted_samples: u64) {
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
        self.completed_requests.fetch_add(1, Ordering::Relaxed);
        self.emitted_samples
            .fetch_add(emitted_samples, Ordering::Relaxed);
    }

    pub fn request_failed(&self) {
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn request_cancelled(&self) {
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
        self.cancelled_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn request_rejected(&self) {
        self.rejected_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn request_retirement_timed_out(&self) {
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
        self.retirement_timeouts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn render_prometheus(&self) -> String {
        format!(
            concat!(
                "# HELP qwen3_tts_http_requests_total Accepted synthesis requests.\n",
                "# TYPE qwen3_tts_http_requests_total counter\n",
                "qwen3_tts_http_requests_total {}\n",
                "# HELP qwen3_tts_active_requests Currently active native requests.\n",
                "# TYPE qwen3_tts_active_requests gauge\n",
                "qwen3_tts_active_requests {}\n",
                "# TYPE qwen3_tts_streaming_requests_total counter\n",
                "qwen3_tts_streaming_requests_total {}\n",
                "# TYPE qwen3_tts_buffered_requests_total counter\n",
                "qwen3_tts_buffered_requests_total {}\n",
                "# TYPE qwen3_tts_completed_requests_total counter\n",
                "qwen3_tts_completed_requests_total {}\n",
                "# TYPE qwen3_tts_failed_requests_total counter\n",
                "qwen3_tts_failed_requests_total {}\n",
                "# TYPE qwen3_tts_cancelled_requests_total counter\n",
                "qwen3_tts_cancelled_requests_total {}\n",
                "# TYPE qwen3_tts_rejected_requests_total counter\n",
                "qwen3_tts_rejected_requests_total {}\n",
                "# HELP qwen3_tts_retirement_timeouts_total Native requests that did not retire before the safety deadline.\n",
                "# TYPE qwen3_tts_retirement_timeouts_total counter\n",
                "qwen3_tts_retirement_timeouts_total {}\n",
                "# TYPE qwen3_tts_emitted_samples_total counter\n",
                "qwen3_tts_emitted_samples_total {}\n"
            ),
            self.requests_total.load(Ordering::Relaxed),
            self.active_requests.load(Ordering::Relaxed),
            self.streaming_requests.load(Ordering::Relaxed),
            self.buffered_requests.load(Ordering::Relaxed),
            self.completed_requests.load(Ordering::Relaxed),
            self.failed_requests.load(Ordering::Relaxed),
            self.cancelled_requests.load(Ordering::Relaxed),
            self.rejected_requests.load(Ordering::Relaxed),
            self.retirement_timeouts.load(Ordering::Relaxed),
            self.emitted_samples.load(Ordering::Relaxed),
        )
    }
}
