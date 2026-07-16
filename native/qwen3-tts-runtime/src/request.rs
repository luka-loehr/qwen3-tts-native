use std::collections::VecDeque;
use std::fmt;

use crate::{AudioPacketDescriptor, RequestMetrics, RequestPhase};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransitionError {
    pub from: RequestPhase,
    pub to: RequestPhase,
}

impl fmt::Display for TransitionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid request transition from {:?} to {:?}",
            self.from, self.to
        )
    }
}

impl std::error::Error for TransitionError {}

#[derive(Clone, Debug, PartialEq)]
pub struct RequestRecord {
    pub id: u64,
    pub phase: RequestPhase,
    pub next_packet_sequence: u64,
    pub next_codec_frame: u64,
    pub next_sample: u64,
    pub metrics: RequestMetrics,
}

impl RequestRecord {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            phase: RequestPhase::Queued,
            next_packet_sequence: 0,
            next_codec_frame: 0,
            next_sample: 0,
            metrics: RequestMetrics::default(),
        }
    }

    pub fn transition(&mut self, next: RequestPhase) -> Result<(), TransitionError> {
        let allowed = matches!(
            (self.phase, next),
            (RequestPhase::Queued, RequestPhase::Prefilling)
                | (RequestPhase::Queued, RequestPhase::Cancelled)
                | (RequestPhase::Queued, RequestPhase::Failed)
                | (RequestPhase::Prefilling, RequestPhase::Generating)
                | (RequestPhase::Prefilling, RequestPhase::Cancelled)
                | (RequestPhase::Prefilling, RequestPhase::Failed)
                | (RequestPhase::Generating, RequestPhase::Draining)
                | (RequestPhase::Generating, RequestPhase::Completed)
                | (RequestPhase::Generating, RequestPhase::Cancelled)
                | (RequestPhase::Generating, RequestPhase::Failed)
                | (RequestPhase::Draining, RequestPhase::Completed)
                | (RequestPhase::Draining, RequestPhase::Cancelled)
                | (RequestPhase::Draining, RequestPhase::Failed)
        );
        if !allowed {
            return Err(TransitionError {
                from: self.phase,
                to: next,
            });
        }
        self.phase = next;
        Ok(())
    }

    pub fn record_packet(
        &mut self,
        packet: &AudioPacketDescriptor,
        configured_packet_frames: u32,
    ) -> Result<(), &'static str> {
        if !matches!(
            self.phase,
            RequestPhase::Generating | RequestPhase::Draining
        ) {
            return Err("request is not accepting audio packets");
        }
        packet.validate(configured_packet_frames)?;
        if packet.request_id != self.id {
            return Err("audio packet request id does not match request");
        }
        if packet.sequence != self.next_packet_sequence {
            return Err("audio packet sequence is not contiguous");
        }
        if packet.first_codec_frame != self.next_codec_frame {
            return Err("audio packet codec frame position is not contiguous");
        }
        if packet.first_sample != self.next_sample {
            return Err("audio packet sample position is not contiguous");
        }

        self.next_packet_sequence += 1;
        self.next_codec_frame += u64::from(packet.codec_frames);
        self.next_sample += u64::from(packet.sample_count);
        self.metrics.generated_codec_frames += u64::from(packet.codec_frames);
        self.metrics.emitted_samples += u64::from(packet.sample_count);
        self.metrics.emitted_packets += 1;
        self.metrics.talker_gpu_microseconds += f64::from(packet.talker_gpu_microseconds);
        self.metrics.codec_gpu_microseconds += f64::from(packet.codec_gpu_microseconds);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PacketQueueError {
    Full,
}

#[derive(Clone, Debug)]
pub struct PacketQueue<T> {
    capacity: usize,
    values: VecDeque<T>,
}

impl<T> PacketQueue<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "packet queue capacity must be non-zero");
        Self {
            capacity,
            values: VecDeque::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, value: T) -> Result<(), PacketQueueError> {
        if self.values.len() == self.capacity {
            return Err(PacketQueueError::Full);
        }
        self.values.push_back(value);
        Ok(())
    }

    pub fn pop(&mut self) -> Option<T> {
        self.values.pop_front()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}
