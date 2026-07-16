use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TryRecvError, sync_channel};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::{
    AudioPacketDescriptor, EngineConfig, GenerationConfig, MAX_CODEC_FRAMES,
    MAX_CONCURRENT_REQUESTS, MAX_INSTRUCT_BYTES, MAX_PACKET_FRAMES, MAX_PCM_RING_SLOTS,
    MAX_TEXT_BYTES, RequestInput, RequestInputError, RequestMetrics, RequestPhase, RequestRecord,
    RuntimeStatus, SAMPLE_RATE, SAMPLES_PER_CODEC_FRAME,
};

const WORKER_IDLE_WAIT: Duration = Duration::from_millis(2);
const PCM_SENTINEL: i16 = i16::MIN;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendError {
    status: RuntimeStatus,
    message: String,
}

impl BackendError {
    pub fn new(message: impl Into<String>) -> Self {
        Self::with_status(RuntimeStatus::Internal, message)
    }

    pub fn with_status(status: RuntimeStatus, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    pub fn status(&self) -> RuntimeStatus {
        self.status
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl From<&str> for BackendError {
    fn from(message: &str) -> Self {
        Self::new(message)
    }
}

impl From<String> for BackendError {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

impl fmt::Display for BackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for BackendError {}

#[derive(Clone, Debug)]
pub struct BackendRequest {
    pub id: u64,
    pub input: RequestInput,
    pub generation: GenerationConfig,
}

#[derive(Debug)]
pub struct BackendStarted<Session> {
    pub session: Session,
    pub prefill_microseconds: u64,
    pub peak_request_device_bytes: u64,
    pub peak_request_host_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BackendPacket {
    pub codec_frames: u32,
    pub is_final: bool,
    pub talker_gpu_microseconds: f32,
    pub codec_gpu_microseconds: f32,
    pub peak_request_device_bytes: u64,
    pub peak_request_host_bytes: u64,
}

pub struct BackendStepInput<'session, Session> {
    pub session: &'session mut Session,
    pub pcm: &'session mut [i16],
}

pub trait StreamingBackend: Send + 'static {
    type Session: Send + 'static;

    fn start(
        &mut self,
        request: BackendRequest,
    ) -> Result<BackendStarted<Self::Session>, BackendError>;

    fn start_batch(
        &mut self,
        requests: Vec<BackendRequest>,
    ) -> Vec<Result<BackendStarted<Self::Session>, BackendError>> {
        requests
            .into_iter()
            .map(|request| self.start(request))
            .collect()
    }

    fn step(
        &mut self,
        session: &mut Self::Session,
        packet_frames: u32,
        pcm: &mut [i16],
    ) -> Result<BackendPacket, BackendError>;

    fn step_batch(
        &mut self,
        requests: &mut [BackendStepInput<'_, Self::Session>],
        packet_frames: u32,
    ) -> Vec<Result<BackendPacket, BackendError>> {
        let mut outputs = Vec::with_capacity(requests.len());
        for request in requests {
            outputs.push(self.step(request.session, packet_frames, request.pcm));
        }
        outputs
    }

    fn cancel(&mut self, _session: &mut Self::Session) -> Result<(), BackendError> {
        Ok(())
    }
}

pub struct OwnedAudioPacket {
    pub descriptor: AudioPacketDescriptor,
    pcm: Option<Vec<i16>>,
    recycler: Weak<RequestShared>,
}

impl OwnedAudioPacket {
    pub fn pcm(&self) -> &[i16] {
        let samples = self.descriptor.sample_count as usize;
        &self.pcm.as_deref().unwrap_or_default()[..samples]
    }
}

impl fmt::Debug for OwnedAudioPacket {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OwnedAudioPacket")
            .field("descriptor", &self.descriptor)
            .field("pcm_samples", &self.pcm().len())
            .finish()
    }
}

impl PartialEq for OwnedAudioPacket {
    fn eq(&self, other: &Self) -> bool {
        self.descriptor == other.descriptor && self.pcm() == other.pcm()
    }
}

impl Drop for OwnedAudioPacket {
    fn drop(&mut self) {
        let Some(pcm) = self.pcm.take() else {
            return;
        };
        if let Some(shared) = self.recycler.upgrade() {
            shared.recycle_pcm(pcm);
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PollError {
    Cancelled,
    Failed(BackendError),
}

impl fmt::Display for PollError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("request was cancelled"),
            Self::Failed(error) => write!(formatter, "request failed: {error}"),
        }
    }
}

impl std::error::Error for PollError {}

#[derive(Debug, PartialEq)]
pub enum PollOutcome {
    Packet(OwnedAudioPacket),
    WouldBlock,
    EndOfStream,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SchedulerError {
    InvalidConfiguration(&'static str),
    InvalidGeneration(&'static str),
    InvalidInput(RequestInputError),
    Full,
    Closed,
    Worker(String),
}

impl fmt::Display for SchedulerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => {
                write!(formatter, "invalid scheduler configuration: {message}")
            }
            Self::InvalidGeneration(message) => {
                write!(formatter, "invalid generation configuration: {message}")
            }
            Self::InvalidInput(error) => write!(formatter, "invalid request input: {error}"),
            Self::Full => formatter.write_str("request capacity is full"),
            Self::Closed => formatter.write_str("scheduler is closed"),
            Self::Worker(message) => write!(formatter, "scheduler worker failed: {message}"),
        }
    }
}

impl std::error::Error for SchedulerError {}

struct SlotPool {
    capacity: usize,
    used: Mutex<usize>,
}

impl SlotPool {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            used: Mutex::new(0),
        }
    }

    fn try_acquire(self: &Arc<Self>) -> Option<SlotPermit> {
        let mut used = lock_unpoisoned(&self.used);
        if *used == self.capacity {
            return None;
        }
        *used += 1;
        Some(SlotPermit {
            pool: Arc::clone(self),
        })
    }

    fn release(&self) {
        let mut used = lock_unpoisoned(&self.used);
        *used = used.saturating_sub(1);
    }
}

struct SlotPermit {
    pool: Arc<SlotPool>,
}

impl Drop for SlotPermit {
    fn drop(&mut self) {
        self.pool.release();
    }
}

struct RequestState {
    phase: RequestPhase,
    packets: VecDeque<OwnedAudioPacket>,
    packet_capacity: usize,
    producer_finished: bool,
    retired: bool,
    cancel_requested: bool,
    failure: Option<BackendError>,
    metrics: RequestMetrics,
}

struct RequestShared {
    id: u64,
    started_at: Instant,
    state: Mutex<RequestState>,
    free_pcm: Mutex<Vec<Vec<i16>>>,
    changed: Condvar,
    commands: SyncSender<Command>,
}

impl RequestShared {
    fn new(
        id: u64,
        packet_capacity: usize,
        packet_samples: usize,
        commands: SyncSender<Command>,
        started_at: Instant,
    ) -> Self {
        let free_pcm = (0..packet_capacity)
            .map(|_| vec![0_i16; packet_samples])
            .collect();
        Self {
            id,
            started_at,
            state: Mutex::new(RequestState {
                phase: RequestPhase::Queued,
                packets: VecDeque::with_capacity(packet_capacity),
                packet_capacity,
                producer_finished: false,
                retired: false,
                cancel_requested: false,
                failure: None,
                metrics: RequestMetrics::default(),
            }),
            free_pcm: Mutex::new(free_pcm),
            changed: Condvar::new(),
            commands,
        }
    }

    fn is_cancel_requested(&self) -> bool {
        lock_unpoisoned(&self.state).cancel_requested
    }

    fn can_accept_packet(&self) -> bool {
        let state = lock_unpoisoned(&self.state);
        let state_accepts = !state.cancel_requested
            && state.failure.is_none()
            && state.packets.len() < state.packet_capacity;
        drop(state);
        state_accepts && !lock_unpoisoned(&self.free_pcm).is_empty()
    }

    fn take_pcm(&self) -> Option<Vec<i16>> {
        lock_unpoisoned(&self.free_pcm).pop()
    }

    fn recycle_pcm(&self, mut pcm: Vec<i16>) {
        pcm.fill(0);
        lock_unpoisoned(&self.free_pcm).push(pcm);
        let _ = self.commands.try_send(Command::Wake);
        self.changed.notify_all();
    }

    fn mark_prefilling(&self, queue_microseconds: u64) {
        let mut state = lock_unpoisoned(&self.state);
        state.phase = RequestPhase::Prefilling;
        state.metrics.queue_microseconds = queue_microseconds;
        self.changed.notify_all();
    }

    fn mark_generating(&self, started: &BackendStarted<impl Send>) {
        let mut state = lock_unpoisoned(&self.state);
        state.phase = RequestPhase::Generating;
        state.metrics.prefill_microseconds = started.prefill_microseconds;
        state.metrics.peak_request_device_bytes = started.peak_request_device_bytes;
        state.metrics.peak_request_host_bytes = started.peak_request_host_bytes;
        self.changed.notify_all();
    }

    fn push_packet(&self, packet: OwnedAudioPacket, metrics: RequestMetrics) -> Result<(), ()> {
        let mut state = lock_unpoisoned(&self.state);
        if state.cancel_requested
            || state.failure.is_some()
            || state.packets.len() == state.packet_capacity
        {
            return Err(());
        }
        let first_audio_microseconds = state.metrics.first_audio_microseconds;
        let emitted_samples = state.metrics.emitted_samples;
        let emitted_packets = state.metrics.emitted_packets;
        state.metrics = metrics;
        state.metrics.first_audio_microseconds = first_audio_microseconds;
        state.metrics.emitted_samples = emitted_samples;
        state.metrics.emitted_packets = emitted_packets;
        if packet.descriptor.is_final != 0 {
            state.phase = RequestPhase::Draining;
            state.producer_finished = true;
        }
        state.packets.push_back(packet);
        self.changed.notify_all();
        Ok(())
    }

    fn mark_cancelled(&self) {
        let mut state = lock_unpoisoned(&self.state);
        state.cancel_requested = true;
        state.phase = RequestPhase::Cancelled;
        state.packets.clear();
        state.producer_finished = true;
        self.changed.notify_all();
    }

    fn mark_failed(&self, error: impl Into<BackendError>) {
        let mut state = lock_unpoisoned(&self.state);
        state.phase = RequestPhase::Failed;
        state.packets.clear();
        state.producer_finished = true;
        state.failure = Some(error.into());
        self.changed.notify_all();
    }

    fn mark_retired(&self) {
        let mut state = lock_unpoisoned(&self.state);
        state.retired = true;
        self.changed.notify_all();
    }

    fn wait_retired(&self, timeout: Duration) -> bool {
        let deadline = Instant::now().checked_add(timeout);
        let mut state = lock_unpoisoned(&self.state);
        while !state.retired {
            let Some(deadline) = deadline else {
                return false;
            };
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let waited = self
                .changed
                .wait_timeout(state, deadline.saturating_duration_since(now));
            let (next_state, result) = match waited {
                Ok(value) => value,
                Err(poisoned) => poisoned.into_inner(),
            };
            state = next_state;
            if result.timed_out() && !state.retired {
                return false;
            }
        }
        true
    }
}

enum Command {
    Start(Box<PendingStart>),
    Cancel(u64),
    Wake,
    Shutdown,
}

struct PendingStart {
    request: BackendRequest,
    shared: Arc<RequestShared>,
    enqueued_at: Instant,
    _permit: SlotPermit,
}

struct ActiveRequest<Session> {
    session: Option<Session>,
    shared: Arc<RequestShared>,
    record: RequestRecord,
    started_at: Instant,
    max_codec_frames: u32,
    _permit: SlotPermit,
}

fn retire_pending(request: PendingStart) {
    let shared = Arc::clone(&request.shared);
    drop(request);
    shared.mark_retired();
}

fn retire_active<Session>(request: ActiveRequest<Session>) {
    let shared = Arc::clone(&request.shared);
    drop(request);
    shared.mark_retired();
}

pub struct RequestHandle {
    shared: Arc<RequestShared>,
    commands: SyncSender<Command>,
    cancel_sent: AtomicBool,
}

impl fmt::Debug for RequestHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RequestHandle")
            .field("id", &self.id())
            .field("phase", &self.phase())
            .finish_non_exhaustive()
    }
}

impl RequestHandle {
    pub fn id(&self) -> u64 {
        self.shared.id
    }

    pub fn phase(&self) -> RequestPhase {
        lock_unpoisoned(&self.shared.state).phase
    }

    pub fn metrics(&self) -> RequestMetrics {
        lock_unpoisoned(&self.shared.state).metrics
    }

    pub fn poll(&self, timeout: Duration) -> Result<PollOutcome, PollError> {
        let deadline = Instant::now().checked_add(timeout);
        let mut state = lock_unpoisoned(&self.shared.state);
        loop {
            if let Some(error) = &state.failure {
                return Err(PollError::Failed(error.clone()));
            }
            if state.cancel_requested || state.phase == RequestPhase::Cancelled {
                return Err(PollError::Cancelled);
            }
            if let Some(packet) = state.packets.pop_front() {
                let final_packet = packet.descriptor.is_final != 0;
                state.metrics.emitted_samples += u64::from(packet.descriptor.sample_count);
                state.metrics.emitted_packets += 1;
                if state.metrics.first_audio_microseconds == 0 {
                    state.metrics.first_audio_microseconds =
                        duration_microseconds(self.shared.started_at.elapsed());
                }
                if final_packet
                    && state.packets.is_empty()
                    && state.failure.is_none()
                    && !state.cancel_requested
                {
                    state.phase = RequestPhase::Completed;
                }
                drop(state);
                let _ = self.commands.try_send(Command::Wake);
                return Ok(PollOutcome::Packet(packet));
            }
            if state.producer_finished {
                state.phase = RequestPhase::Completed;
                return Ok(PollOutcome::EndOfStream);
            }
            if timeout.is_zero() {
                return Ok(PollOutcome::WouldBlock);
            }
            let Some(deadline) = deadline else {
                return Ok(PollOutcome::WouldBlock);
            };
            let now = Instant::now();
            if now >= deadline {
                return Ok(PollOutcome::WouldBlock);
            }
            let remaining = deadline.saturating_duration_since(now);
            let waited = self.shared.changed.wait_timeout(state, remaining);
            let (next_state, result) = match waited {
                Ok(value) => value,
                Err(poisoned) => poisoned.into_inner(),
            };
            state = next_state;
            if result.timed_out() {
                return Ok(PollOutcome::WouldBlock);
            }
        }
    }

    pub fn cancel(&self) -> Result<(), SchedulerError> {
        if self.cancel_sent.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        {
            let mut state = lock_unpoisoned(&self.shared.state);
            if state.phase.is_terminal() {
                return Ok(());
            }
            state.cancel_requested = true;
            state.packets.clear();
            self.shared.changed.notify_all();
        }
        self.commands
            .send(Command::Cancel(self.shared.id))
            .map_err(|_| SchedulerError::Closed)
    }

    pub fn wait_retired(&self, timeout: Duration) -> bool {
        self.shared.wait_retired(timeout)
    }

    pub fn cancel_and_wait(&self, timeout: Duration) -> Result<bool, SchedulerError> {
        self.cancel()?;
        Ok(self.wait_retired(timeout))
    }
}

impl Drop for RequestHandle {
    fn drop(&mut self) {
        let _ = self.cancel();
    }
}

pub struct Scheduler<B: StreamingBackend> {
    config: EngineConfig,
    slots: Arc<SlotPool>,
    commands: SyncSender<Command>,
    next_request_id: AtomicU64,
    worker: Option<JoinHandle<()>>,
    _backend: std::marker::PhantomData<B>,
}

impl<B: StreamingBackend> Scheduler<B> {
    pub fn new(config: EngineConfig, backend: B) -> Result<Self, SchedulerError> {
        validate_config(&config)?;
        let capacity = config.max_concurrent_requests as usize;
        let command_capacity = capacity
            .checked_mul(4)
            .and_then(|value| value.checked_add(8))
            .ok_or(SchedulerError::InvalidConfiguration(
                "command queue capacity overflow",
            ))?;
        let (commands, receiver) = sync_channel(command_capacity);
        let worker_config = config;
        let worker = thread::Builder::new()
            .name("qwen3-tts-native-worker".to_owned())
            .spawn(move || worker_loop(backend, worker_config, receiver))
            .map_err(|error| SchedulerError::Worker(error.to_string()))?;
        Ok(Self {
            config,
            slots: Arc::new(SlotPool::new(capacity)),
            commands,
            next_request_id: AtomicU64::new(1),
            worker: Some(worker),
            _backend: std::marker::PhantomData,
        })
    }

    pub fn start(
        &self,
        input: RequestInput,
        generation: GenerationConfig,
    ) -> Result<RequestHandle, SchedulerError> {
        input
            .validate(&self.config)
            .map_err(SchedulerError::InvalidInput)?;
        validate_generation(&generation)?;
        let permit = self.slots.try_acquire().ok_or(SchedulerError::Full)?;
        let id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let enqueued_at = Instant::now();
        let packet_samples = self.config.packet_frames as usize * SAMPLES_PER_CODEC_FRAME as usize;
        let shared = Arc::new(RequestShared::new(
            id,
            self.config.pcm_ring_slots as usize,
            packet_samples,
            self.commands.clone(),
            enqueued_at,
        ));
        let pending = PendingStart {
            request: BackendRequest {
                id,
                input,
                generation,
            },
            shared: Arc::clone(&shared),
            enqueued_at,
            _permit: permit,
        };
        self.commands
            .send(Command::Start(Box::new(pending)))
            .map_err(|_| SchedulerError::Closed)?;
        Ok(RequestHandle {
            shared,
            commands: self.commands.clone(),
            cancel_sent: AtomicBool::new(false),
        })
    }
}

impl<B: StreamingBackend> Drop for Scheduler<B> {
    fn drop(&mut self) {
        let _ = self.commands.send(Command::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn validate_config(config: &EngineConfig) -> Result<(), SchedulerError> {
    if config.struct_size != size_of::<EngineConfig>() as u32 {
        return Err(SchedulerError::InvalidConfiguration(
            "struct_size does not match EngineConfig",
        ));
    }
    if config.flags != 0 || config.reserved.iter().any(|value| *value != 0) {
        return Err(SchedulerError::InvalidConfiguration(
            "flags and reserved fields must be zero",
        ));
    }
    if config.max_concurrent_requests == 0 {
        return Err(SchedulerError::InvalidConfiguration(
            "max_concurrent_requests must be non-zero",
        ));
    }
    if config.max_concurrent_requests > MAX_CONCURRENT_REQUESTS {
        return Err(SchedulerError::InvalidConfiguration(
            "max_concurrent_requests exceeds the native batch limit",
        ));
    }
    if config.packet_frames == 0 {
        return Err(SchedulerError::InvalidConfiguration(
            "packet_frames must be non-zero",
        ));
    }
    if config.packet_frames > MAX_PACKET_FRAMES {
        return Err(SchedulerError::InvalidConfiguration(
            "packet_frames exceeds the neural decoder limit",
        ));
    }
    if config.pcm_ring_slots == 0 {
        return Err(SchedulerError::InvalidConfiguration(
            "pcm_ring_slots must be non-zero",
        ));
    }
    if config.pcm_ring_slots > MAX_PCM_RING_SLOTS {
        return Err(SchedulerError::InvalidConfiguration(
            "pcm_ring_slots exceeds the bounded runtime limit",
        ));
    }
    if config.max_text_bytes == 0 || config.max_text_bytes > MAX_TEXT_BYTES {
        return Err(SchedulerError::InvalidConfiguration(
            "max_text_bytes is outside the supported range",
        ));
    }
    if config.max_instruct_bytes > MAX_INSTRUCT_BYTES {
        return Err(SchedulerError::InvalidConfiguration(
            "max_instruct_bytes exceeds the supported range",
        ));
    }
    Ok(())
}

fn validate_generation(config: &GenerationConfig) -> Result<(), SchedulerError> {
    if config.struct_size != size_of::<GenerationConfig>() as u32 {
        return Err(SchedulerError::InvalidGeneration(
            "struct_size does not match GenerationConfig",
        ));
    }
    if config.reserved.iter().any(|value| *value != 0) {
        return Err(SchedulerError::InvalidGeneration(
            "reserved fields must be zero",
        ));
    }
    if config.max_codec_frames == 0 {
        return Err(SchedulerError::InvalidGeneration(
            "max_codec_frames must be non-zero",
        ));
    }
    if config.max_codec_frames > MAX_CODEC_FRAMES {
        return Err(SchedulerError::InvalidGeneration(
            "max_codec_frames exceeds the native sequence limit",
        ));
    }
    if !config.temperature.is_finite() || config.temperature <= 0.0 {
        return Err(SchedulerError::InvalidGeneration(
            "temperature must be finite and positive",
        ));
    }
    if !config.predictor_temperature.is_finite() || config.predictor_temperature <= 0.0 {
        return Err(SchedulerError::InvalidGeneration(
            "predictor_temperature must be finite and positive",
        ));
    }
    if !config.top_p.is_finite() || !(0.0..=1.0).contains(&config.top_p) || config.top_p == 0.0 {
        return Err(SchedulerError::InvalidGeneration(
            "top_p must be finite and in (0, 1]",
        ));
    }
    if !config.predictor_top_p.is_finite()
        || !(0.0..=1.0).contains(&config.predictor_top_p)
        || config.predictor_top_p == 0.0
    {
        return Err(SchedulerError::InvalidGeneration(
            "predictor_top_p must be finite and in (0, 1]",
        ));
    }
    if !config.repetition_penalty.is_finite() || config.repetition_penalty <= 0.0 {
        return Err(SchedulerError::InvalidGeneration(
            "repetition_penalty must be finite and positive",
        ));
    }
    if config.do_sample > 1 || config.predictor_do_sample > 1 {
        return Err(SchedulerError::InvalidGeneration(
            "sampling flags must be zero or one",
        ));
    }
    if config.top_k > 3_072 || config.predictor_top_k > 2_048 {
        return Err(SchedulerError::InvalidGeneration(
            "top_k exceeds the corresponding model vocabulary",
        ));
    }
    Ok(())
}

fn worker_loop<B: StreamingBackend>(
    mut backend: B,
    config: EngineConfig,
    receiver: Receiver<Command>,
) {
    let mut active = Vec::<ActiveRequest<B::Session>>::new();
    let mut pending = Vec::<PendingStart>::new();
    let mut shutdown = false;
    while !shutdown {
        if active.is_empty() && pending.is_empty() {
            match receiver.recv() {
                Ok(command) => shutdown = handle_command(command, &mut active, &mut pending),
                Err(_) => shutdown = true,
            }
        }
        drain_commands(&receiver, &mut active, &mut pending, &mut shutdown);
        if shutdown {
            break;
        }
        start_pending(&mut backend, &mut pending, &mut active);
        remove_cancelled(&mut backend, &mut active);
        let eligible = active
            .iter()
            .enumerate()
            .filter_map(|(index, request)| request.shared.can_accept_packet().then_some(index))
            .collect::<Vec<_>>();
        if eligible.is_empty() {
            match receiver.recv_timeout(WORKER_IDLE_WAIT) {
                Ok(command) => {
                    shutdown = handle_command(command, &mut active, &mut pending);
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => shutdown = true,
            }
            continue;
        }
        step_eligible(&mut backend, config.packet_frames, &mut active, &eligible);
        remove_finished(&mut backend, &mut active);
    }

    for mut request in active {
        if let Some(session) = &mut request.session {
            let _ = backend.cancel(session);
        }
        request.shared.mark_cancelled();
        retire_active(request);
    }
    for request in pending {
        request.shared.mark_cancelled();
        retire_pending(request);
    }
}

fn drain_commands<Session>(
    receiver: &Receiver<Command>,
    active: &mut [ActiveRequest<Session>],
    pending: &mut Vec<PendingStart>,
    shutdown: &mut bool,
) {
    loop {
        match receiver.try_recv() {
            Ok(command) => {
                if handle_command(command, active, pending) {
                    *shutdown = true;
                    return;
                }
            }
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => {
                *shutdown = true;
                return;
            }
        }
    }
}

fn handle_command<Session>(
    command: Command,
    active: &mut [ActiveRequest<Session>],
    pending: &mut Vec<PendingStart>,
) -> bool {
    match command {
        Command::Start(request) => pending.push(*request),
        Command::Cancel(id) => {
            if let Some(index) = pending.iter().position(|request| request.request.id == id) {
                let request = pending.swap_remove(index);
                request.shared.mark_cancelled();
                retire_pending(request);
            }
            if let Some(request) = active.iter().find(|request| request.record.id == id) {
                request.shared.mark_cancelled();
            }
        }
        Command::Wake => {}
        Command::Shutdown => return true,
    }
    false
}

fn start_pending<B: StreamingBackend>(
    backend: &mut B,
    pending: &mut Vec<PendingStart>,
    active: &mut Vec<ActiveRequest<B::Session>>,
) {
    if pending.is_empty() {
        return;
    }
    let mut ready = Vec::with_capacity(pending.len());
    for request in pending.drain(..) {
        if request.shared.is_cancel_requested() {
            request.shared.mark_cancelled();
            retire_pending(request);
        } else {
            ready.push(request);
        }
    }
    if ready.is_empty() {
        return;
    }
    let prefill_started = Instant::now();
    for request in &ready {
        request.shared.mark_prefilling(duration_microseconds(
            prefill_started.saturating_duration_since(request.enqueued_at),
        ));
    }
    let backend_requests = ready
        .iter()
        .map(|request| request.request.clone())
        .collect::<Vec<_>>();
    let results = backend.start_batch(backend_requests);
    if results.len() != ready.len() {
        for request in ready {
            request
                .shared
                .mark_failed("backend returned the wrong number of prefill results");
            retire_pending(request);
        }
        return;
    }
    for (request, result) in ready.into_iter().zip(results) {
        match result {
            Ok(started) => {
                request.shared.mark_generating(&started);
                let mut record = RequestRecord::new(request.request.id);
                let _ = record.transition(RequestPhase::Prefilling);
                let _ = record.transition(RequestPhase::Generating);
                record.metrics = lock_unpoisoned(&request.shared.state).metrics;
                active.push(ActiveRequest {
                    session: Some(started.session),
                    shared: request.shared,
                    record,
                    started_at: request.enqueued_at,
                    max_codec_frames: request.request.generation.max_codec_frames,
                    _permit: request._permit,
                });
            }
            Err(error) => {
                request.shared.mark_failed(error);
                retire_pending(request);
            }
        }
    }
}

fn remove_cancelled<B: StreamingBackend>(
    backend: &mut B,
    active: &mut Vec<ActiveRequest<B::Session>>,
) {
    let mut index = 0;
    while index < active.len() {
        if active[index].shared.is_cancel_requested() {
            let mut request = active.swap_remove(index);
            if let Some(session) = &mut request.session {
                let _ = backend.cancel(session);
            }
            request.shared.mark_cancelled();
            retire_active(request);
        } else {
            index += 1;
        }
    }
}

fn step_eligible<B: StreamingBackend>(
    backend: &mut B,
    packet_frames: u32,
    active: &mut [ActiveRequest<B::Session>],
    eligible: &[usize],
) {
    let step_started = Instant::now();
    let mut selected = Vec::with_capacity(eligible.len());
    for index in eligible {
        let session = active[*index]
            .session
            .take()
            .expect("eligible request must own a backend session");
        let Some(mut pcm) = active[*index].shared.take_pcm() else {
            active[*index].session = Some(session);
            active[*index]
                .shared
                .mark_failed("scheduler PCM pool was unexpectedly empty");
            continue;
        };
        pcm.fill(PCM_SENTINEL);
        selected.push((*index, session, pcm));
    }
    let results = {
        let mut requests = selected
            .iter_mut()
            .map(|(_, session, pcm)| BackendStepInput {
                session,
                pcm: pcm.as_mut_slice(),
            })
            .collect::<Vec<_>>();
        backend.step_batch(&mut requests, packet_frames)
    };
    if results.len() != selected.len() {
        for (index, session, pcm) in selected {
            active[index].session = Some(session);
            active[index].shared.recycle_pcm(pcm);
            active[index]
                .shared
                .mark_failed("backend returned the wrong number of generation results");
        }
        return;
    }
    let step_microseconds = duration_microseconds(step_started.elapsed()) as f32;
    for ((index, session, pcm), result) in selected.into_iter().zip(results) {
        let request = &mut active[index];
        request.session = Some(session);
        match result {
            Ok(packet) => {
                if let Err(message) = validate_backend_packet(&packet, packet_frames, &pcm) {
                    request.shared.recycle_pcm(pcm);
                    request.shared.mark_failed(message);
                    continue;
                }
                let next_frames = request
                    .record
                    .next_codec_frame
                    .saturating_add(u64::from(packet.codec_frames));
                if next_frames > u64::from(request.max_codec_frames)
                    || (next_frames == u64::from(request.max_codec_frames) && !packet.is_final)
                {
                    request.shared.recycle_pcm(pcm);
                    request
                        .shared
                        .mark_failed("backend violated max_codec_frames termination");
                    continue;
                }
                let descriptor = AudioPacketDescriptor {
                    request_id: request.record.id,
                    sequence: request.record.next_packet_sequence,
                    first_codec_frame: request.record.next_codec_frame,
                    first_sample: request.record.next_sample,
                    codec_frames: packet.codec_frames,
                    sample_count: packet.codec_frames * SAMPLES_PER_CODEC_FRAME,
                    sample_rate: SAMPLE_RATE,
                    channels: 1,
                    is_final: u32::from(packet.is_final),
                    reserved: 0,
                    talker_gpu_microseconds: packet.talker_gpu_microseconds,
                    codec_gpu_microseconds: packet.codec_gpu_microseconds,
                    end_to_end_microseconds: step_microseconds,
                };
                if request
                    .record
                    .record_packet(&descriptor, packet_frames)
                    .is_err()
                {
                    request.shared.recycle_pcm(pcm);
                    request
                        .shared
                        .mark_failed("scheduler packet invariant failed");
                    continue;
                }
                let elapsed = duration_microseconds(request.started_at.elapsed());
                if request.record.metrics.first_codec_frame_microseconds == 0 {
                    request.record.metrics.first_codec_frame_microseconds = elapsed;
                }
                request.record.metrics.wall_microseconds = elapsed;
                request.record.metrics.peak_request_device_bytes = request
                    .record
                    .metrics
                    .peak_request_device_bytes
                    .max(packet.peak_request_device_bytes);
                request.record.metrics.peak_request_host_bytes = request
                    .record
                    .metrics
                    .peak_request_host_bytes
                    .max(packet.peak_request_host_bytes);
                if packet.is_final {
                    let _ = request.record.transition(RequestPhase::Draining);
                }
                let owned = OwnedAudioPacket {
                    descriptor,
                    pcm: Some(pcm),
                    recycler: Arc::downgrade(&request.shared),
                };
                if request
                    .shared
                    .push_packet(owned, request.record.metrics)
                    .is_err()
                    && !request.shared.is_cancel_requested()
                {
                    request
                        .shared
                        .mark_failed("scheduler packet queue overflowed");
                }
            }
            Err(error) => {
                request.shared.recycle_pcm(pcm);
                request.shared.mark_failed(error);
            }
        }
    }
}

fn remove_finished<B: StreamingBackend>(
    backend: &mut B,
    active: &mut Vec<ActiveRequest<B::Session>>,
) {
    let mut index = 0;
    while index < active.len() {
        let (finished, failed_or_cancelled) = {
            let state = lock_unpoisoned(&active[index].shared.state);
            let failed_or_cancelled = state.failure.is_some() || state.cancel_requested;
            (
                state.producer_finished || failed_or_cancelled,
                failed_or_cancelled,
            )
        };
        if finished {
            let mut request = active.swap_remove(index);
            if failed_or_cancelled && let Some(session) = &mut request.session {
                let _ = backend.cancel(session);
            }
            retire_active(request);
        } else {
            index += 1;
        }
    }
}

fn validate_backend_packet(
    packet: &BackendPacket,
    packet_frames: u32,
    pcm: &[i16],
) -> Result<(), &'static str> {
    if packet.codec_frames == 0 || packet.codec_frames > packet_frames {
        return Err("backend emitted an invalid codec frame count");
    }
    let expected = packet.codec_frames as usize * SAMPLES_PER_CODEC_FRAME as usize;
    if expected > pcm.len() {
        return Err("backend PCM output exceeds its caller-owned buffer");
    }
    if pcm[expected..].iter().any(|sample| *sample != PCM_SENTINEL) {
        return Err("backend wrote beyond the exact PCM sample count");
    }
    Ok(())
}

fn duration_microseconds(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::Language;

    struct ScriptedBackend {
        steps: usize,
        maximum_batch: Arc<AtomicUsize>,
    }

    struct ScriptedSession {
        id: u64,
        step: usize,
    }

    impl StreamingBackend for ScriptedBackend {
        type Session = ScriptedSession;

        fn start(
            &mut self,
            request: BackendRequest,
        ) -> Result<BackendStarted<Self::Session>, BackendError> {
            Ok(BackendStarted {
                session: ScriptedSession {
                    id: request.id,
                    step: 0,
                },
                prefill_microseconds: 17,
                peak_request_device_bytes: 1_024,
                peak_request_host_bytes: 512,
            })
        }

        fn step(
            &mut self,
            session: &mut Self::Session,
            _packet_frames: u32,
            pcm: &mut [i16],
        ) -> Result<BackendPacket, BackendError> {
            session.step += 1;
            let frames = 1;
            pcm[..SAMPLES_PER_CODEC_FRAME as usize].fill(session.id as i16);
            Ok(BackendPacket {
                codec_frames: frames,
                is_final: session.step == self.steps,
                talker_gpu_microseconds: 10.0,
                codec_gpu_microseconds: 5.0,
                peak_request_device_bytes: 2_048,
                peak_request_host_bytes: 1_024,
            })
        }

        fn step_batch(
            &mut self,
            requests: &mut [BackendStepInput<'_, Self::Session>],
            packet_frames: u32,
        ) -> Vec<Result<BackendPacket, BackendError>> {
            self.maximum_batch
                .fetch_max(requests.len(), Ordering::Relaxed);
            let mut outputs = Vec::with_capacity(requests.len());
            for request in requests {
                outputs.push(self.step(request.session, packet_frames, request.pcm));
            }
            outputs
        }
    }

    fn scheduler(capacity: u32, ring_slots: u32, steps: usize) -> Scheduler<ScriptedBackend> {
        let config = EngineConfig {
            max_concurrent_requests: capacity,
            pcm_ring_slots: ring_slots,
            ..EngineConfig::default()
        };
        Scheduler::new(
            config,
            ScriptedBackend {
                steps,
                maximum_batch: Arc::new(AtomicUsize::new(0)),
            },
        )
        .unwrap()
    }

    fn input() -> RequestInput {
        RequestInput {
            text: "A bounded native streaming test".to_owned(),
            instruct: "A calm technical voice".to_owned(),
            language: Language::English,
        }
    }

    #[test]
    fn packets_are_progressive_contiguous_and_terminal() {
        let scheduler = scheduler(1, 1, 3);
        let request = scheduler
            .start(input(), GenerationConfig::default())
            .unwrap();
        let mut packets = Vec::new();
        loop {
            match request.poll(Duration::from_secs(1)).unwrap() {
                PollOutcome::Packet(packet) => {
                    assert_eq!(packet.pcm().len(), SAMPLES_PER_CODEC_FRAME as usize);
                    packets.push(packet.descriptor);
                }
                PollOutcome::EndOfStream => break,
                PollOutcome::WouldBlock => panic!("worker did not make progress"),
            }
        }
        assert_eq!(packets.len(), 3);
        assert_eq!(packets[0].sequence, 0);
        assert_eq!(packets[1].first_codec_frame, 1);
        assert_eq!(
            packets[2].first_sample,
            2 * u64::from(SAMPLES_PER_CODEC_FRAME)
        );
        assert_eq!(packets[2].is_final, 1);
        assert_eq!(request.phase(), RequestPhase::Completed);
        let metrics = request.metrics();
        assert_eq!(metrics.emitted_packets, 3);
        assert_eq!(
            metrics.emitted_samples,
            3 * u64::from(SAMPLES_PER_CODEC_FRAME)
        );
        assert_eq!(metrics.peak_request_device_bytes, 2_048);
        assert!(metrics.first_audio_microseconds >= metrics.first_codec_frame_microseconds);
        assert!(request.wait_retired(Duration::from_secs(1)));
    }

    #[test]
    fn ring_capacity_pauses_generation_until_poll_frees_a_slot() {
        let scheduler = scheduler(1, 1, 3);
        let request = scheduler
            .start(input(), GenerationConfig::default())
            .unwrap();
        wait_for_phase(&request, RequestPhase::Generating);
        thread::sleep(Duration::from_millis(20));
        assert_eq!(request.metrics().generated_codec_frames, 1);
        assert_eq!(request.metrics().emitted_packets, 0);
        assert!(matches!(
            request.poll(Duration::from_secs(1)).unwrap(),
            PollOutcome::Packet(_)
        ));
        assert_eq!(request.metrics().emitted_packets, 1);
        let second = request.poll(Duration::from_secs(1)).unwrap();
        assert!(matches!(second, PollOutcome::Packet(_)));
    }

    #[test]
    fn capacity_is_released_after_cancellation() {
        let scheduler = scheduler(1, 1, 100);
        let request = scheduler
            .start(input(), GenerationConfig::default())
            .unwrap();
        assert_eq!(
            scheduler
                .start(input(), GenerationConfig::default())
                .unwrap_err(),
            SchedulerError::Full
        );
        assert!(request.cancel_and_wait(Duration::from_secs(1)).unwrap());
        let replacement = scheduler
            .start(input(), GenerationConfig::default())
            .expect("retired request must release capacity synchronously");
        replacement.cancel().unwrap();
        assert_eq!(request.poll(Duration::ZERO), Err(PollError::Cancelled));
    }

    #[test]
    fn invalid_backend_pcm_fails_without_emitting_a_packet() {
        struct InvalidBackend;
        impl StreamingBackend for InvalidBackend {
            type Session = ();

            fn start(
                &mut self,
                _request: BackendRequest,
            ) -> Result<BackendStarted<Self::Session>, BackendError> {
                Ok(BackendStarted {
                    session: (),
                    prefill_microseconds: 0,
                    peak_request_device_bytes: 0,
                    peak_request_host_bytes: 0,
                })
            }

            fn step(
                &mut self,
                _session: &mut Self::Session,
                _packet_frames: u32,
                pcm: &mut [i16],
            ) -> Result<BackendPacket, BackendError> {
                pcm[SAMPLES_PER_CODEC_FRAME as usize] = 0;
                Ok(BackendPacket {
                    codec_frames: 1,
                    is_final: true,
                    talker_gpu_microseconds: 0.0,
                    codec_gpu_microseconds: 0.0,
                    peak_request_device_bytes: 0,
                    peak_request_host_bytes: 0,
                })
            }
        }

        let scheduler = Scheduler::new(EngineConfig::default(), InvalidBackend).unwrap();
        let request = scheduler
            .start(input(), GenerationConfig::default())
            .unwrap();
        let error = request.poll(Duration::from_secs(1)).unwrap_err();
        assert!(matches!(error, PollError::Failed(_)));
        assert_eq!(request.metrics().emitted_packets, 0);
    }

    #[test]
    fn backend_failure_preserves_runtime_status_through_poll() {
        struct FailingBackend;
        impl StreamingBackend for FailingBackend {
            type Session = ();

            fn start(
                &mut self,
                _request: BackendRequest,
            ) -> Result<BackendStarted<Self::Session>, BackendError> {
                Ok(BackendStarted {
                    session: (),
                    prefill_microseconds: 0,
                    peak_request_device_bytes: 0,
                    peak_request_host_bytes: 0,
                })
            }

            fn step(
                &mut self,
                _session: &mut Self::Session,
                _packet_frames: u32,
                _pcm: &mut [i16],
            ) -> Result<BackendPacket, BackendError> {
                Err(BackendError::with_status(
                    RuntimeStatus::Cuda,
                    "CUDA kernel launch failed",
                ))
            }
        }

        let scheduler = Scheduler::new(EngineConfig::default(), FailingBackend).unwrap();
        let request = scheduler
            .start(input(), GenerationConfig::default())
            .unwrap();
        let PollError::Failed(error) = request.poll(Duration::from_secs(1)).unwrap_err() else {
            panic!("request did not preserve its backend failure");
        };
        assert_eq!(error.status(), RuntimeStatus::Cuda);
        assert_eq!(error.message(), "CUDA kernel launch failed");
    }

    #[test]
    fn max_codec_frames_is_a_hard_backend_boundary() {
        let scheduler = scheduler(1, 1, 3);
        let generation = GenerationConfig {
            max_codec_frames: 2,
            ..GenerationConfig::default()
        };
        let request = scheduler.start(input(), generation).unwrap();
        assert!(matches!(
            request.poll(Duration::from_secs(1)).unwrap(),
            PollOutcome::Packet(_)
        ));
        let error = request.poll(Duration::from_secs(1)).unwrap_err();
        assert!(matches!(error, PollError::Failed(_)));
        assert_eq!(request.metrics().emitted_packets, 1);
    }

    #[test]
    fn invalid_sampling_values_never_reach_the_worker() {
        let scheduler = scheduler(1, 1, 1);
        let generation = GenerationConfig {
            temperature: f32::NAN,
            ..GenerationConfig::default()
        };
        assert!(matches!(
            scheduler.start(input(), generation),
            Err(SchedulerError::InvalidGeneration(_))
        ));
    }

    #[test]
    fn unbounded_or_abi_incompatible_engine_configs_are_rejected() {
        let backend = || ScriptedBackend {
            steps: 1,
            maximum_batch: Arc::new(AtomicUsize::new(0)),
        };
        let invalid = [
            EngineConfig {
                struct_size: 0,
                ..EngineConfig::default()
            },
            EngineConfig {
                flags: 1,
                ..EngineConfig::default()
            },
            EngineConfig {
                reserved: [1; 8],
                ..EngineConfig::default()
            },
            EngineConfig {
                max_concurrent_requests: MAX_CONCURRENT_REQUESTS + 1,
                ..EngineConfig::default()
            },
            EngineConfig {
                packet_frames: MAX_PACKET_FRAMES + 1,
                ..EngineConfig::default()
            },
            EngineConfig {
                pcm_ring_slots: MAX_PCM_RING_SLOTS + 1,
                ..EngineConfig::default()
            },
            EngineConfig {
                max_text_bytes: MAX_TEXT_BYTES + 1,
                ..EngineConfig::default()
            },
            EngineConfig {
                max_instruct_bytes: MAX_INSTRUCT_BYTES + 1,
                ..EngineConfig::default()
            },
        ];

        for config in invalid {
            assert!(matches!(
                Scheduler::new(config, backend()),
                Err(SchedulerError::InvalidConfiguration(_))
            ));
        }
    }

    #[test]
    fn generation_abi_and_model_limits_are_enforced() {
        let scheduler = scheduler(1, 1, 1);
        let invalid = [
            GenerationConfig {
                struct_size: 0,
                ..GenerationConfig::default()
            },
            GenerationConfig {
                reserved: [1; 8],
                ..GenerationConfig::default()
            },
            GenerationConfig {
                max_codec_frames: MAX_CODEC_FRAMES + 1,
                ..GenerationConfig::default()
            },
            GenerationConfig {
                top_k: 3_073,
                ..GenerationConfig::default()
            },
            GenerationConfig {
                predictor_top_k: 2_049,
                ..GenerationConfig::default()
            },
        ];

        for generation in invalid {
            assert!(matches!(
                scheduler.start(input(), generation),
                Err(SchedulerError::InvalidGeneration(_))
            ));
        }
    }

    fn wait_for_phase(request: &RequestHandle, phase: RequestPhase) {
        let deadline = Instant::now() + Duration::from_secs(1);
        while request.phase() != phase {
            assert!(Instant::now() < deadline, "phase did not become {phase:?}");
            thread::sleep(Duration::from_millis(1));
        }
    }
}
