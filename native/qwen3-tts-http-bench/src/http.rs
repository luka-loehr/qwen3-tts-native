use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;

use tokio::io::AsyncReadExt;
use tokio::net::TcpStream;

use crate::{BenchError, Result};

const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_CHUNK_LINE_BYTES: usize = 8 * 1024;
const READ_BUFFER_BYTES: usize = 16 * 1024;

#[derive(Clone, Debug)]
pub(crate) struct TimedBytes {
    pub bytes: Vec<u8>,
    pub arrived: Instant,
}

#[derive(Debug)]
struct Segment {
    bytes: Vec<u8>,
    offset: usize,
    arrived: Instant,
}

#[derive(Debug, Default)]
pub(crate) struct TimedBuffer {
    segments: VecDeque<Segment>,
    len: usize,
}

impl TimedBuffer {
    pub fn push(&mut self, bytes: Vec<u8>, arrived: Instant) {
        if !bytes.is_empty() {
            self.len += bytes.len();
            self.segments.push_back(Segment {
                bytes,
                offset: 0,
                arrived,
            });
        }
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn find(&self, needle: &[u8]) -> Option<usize> {
        if needle.is_empty() {
            return Some(0);
        }
        let mut matched = 0;
        let mut position = 0;
        for segment in &self.segments {
            for &byte in &segment.bytes[segment.offset..] {
                if byte == needle[matched] {
                    matched += 1;
                    if matched == needle.len() {
                        return Some(position + 1 - needle.len());
                    }
                } else {
                    matched = usize::from(byte == needle[0]);
                }
                position += 1;
            }
        }
        None
    }

    pub fn drain_bytes(&mut self, mut count: usize) -> Result<Vec<u8>> {
        if count > self.len {
            return Err(BenchError::Http(
                "internal buffer underflow while draining bytes".to_owned(),
            ));
        }
        let mut output = Vec::with_capacity(count);
        while count > 0 {
            let front = self
                .segments
                .front_mut()
                .expect("non-empty byte count requires a segment");
            let available = front.bytes.len() - front.offset;
            let take = available.min(count);
            output.extend_from_slice(&front.bytes[front.offset..front.offset + take]);
            front.offset += take;
            self.len -= take;
            count -= take;
            if front.offset == front.bytes.len() {
                self.segments.pop_front();
            }
        }
        Ok(output)
    }

    pub fn pop_segment(&mut self, maximum: usize) -> Option<TimedBytes> {
        let front = self.segments.front_mut()?;
        let take = (front.bytes.len() - front.offset).min(maximum);
        let bytes = front.bytes[front.offset..front.offset + take].to_vec();
        let arrived = front.arrived;
        front.offset += take;
        self.len -= take;
        if front.offset == front.bytes.len() {
            self.segments.pop_front();
        }
        Some(TimedBytes { bytes, arrived })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ResponseHead {
    pub status: u16,
    pub headers: BTreeMap<String, String>,
}

impl ResponseHead {
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }
}

#[derive(Debug)]
enum TransferMode {
    ContentLength(u64),
    Chunked(ChunkState),
    CloseDelimited,
}

#[derive(Debug)]
enum ChunkState {
    Size,
    Data(u64),
    DataCrlf,
    Trailers,
    End,
}

pub(crate) struct HttpBody {
    stream: TcpStream,
    raw: TimedBuffer,
    transfer: TransferMode,
    response_bytes: u64,
    eof: bool,
}

impl HttpBody {
    pub(crate) async fn read_head(mut stream: TcpStream) -> Result<(ResponseHead, Self)> {
        let mut raw = TimedBuffer::default();
        let mut response_bytes = 0_u64;
        loop {
            if let Some(position) = raw.find(b"\r\n\r\n") {
                let head_bytes = raw.drain_bytes(position + 4)?;
                let head = parse_head(&head_bytes)?;
                let transfer = select_transfer(&head)?;
                return Ok((
                    head,
                    Self {
                        stream,
                        raw,
                        transfer,
                        response_bytes,
                        eof: false,
                    },
                ));
            }
            if raw.len() >= MAX_HEADER_BYTES {
                return Err(BenchError::Http(format!(
                    "response headers exceed {MAX_HEADER_BYTES} bytes"
                )));
            }
            let count = read_into(&mut stream, &mut raw, &mut response_bytes).await?;
            if count == 0 {
                return Err(BenchError::Http(
                    "connection closed before response headers completed".to_owned(),
                ));
            }
        }
    }

    #[must_use]
    pub const fn response_bytes(&self) -> u64 {
        self.response_bytes
    }

    pub(crate) async fn next_segment(&mut self) -> Result<Option<TimedBytes>> {
        let transfer = std::mem::replace(&mut self.transfer, TransferMode::ContentLength(0));
        match transfer {
            TransferMode::ContentLength(remaining) => self.next_content_length(remaining).await,
            TransferMode::CloseDelimited => self.next_close_delimited().await,
            TransferMode::Chunked(state) => self.next_chunked(state).await,
        }
    }

    async fn next_content_length(&mut self, mut remaining: u64) -> Result<Option<TimedBytes>> {
        if remaining == 0 {
            if !self.raw.is_empty() {
                return Err(BenchError::Http(
                    "bytes remain after the declared Content-Length".to_owned(),
                ));
            }
            self.transfer = TransferMode::ContentLength(0);
            return Ok(None);
        }
        if self.raw.is_empty() {
            self.read_more().await?;
            if self.eof {
                return Err(BenchError::Http(
                    "connection closed before Content-Length bytes arrived".to_owned(),
                ));
            }
        }
        let maximum = usize::try_from(remaining).unwrap_or(usize::MAX);
        let segment = self
            .raw
            .pop_segment(maximum)
            .expect("a non-empty raw buffer yields a segment");
        remaining -= u64::try_from(segment.bytes.len()).expect("usize fits u64");
        self.transfer = TransferMode::ContentLength(remaining);
        Ok(Some(segment))
    }

    async fn next_close_delimited(&mut self) -> Result<Option<TimedBytes>> {
        if self.raw.is_empty() {
            self.read_more().await?;
            if self.eof {
                self.transfer = TransferMode::CloseDelimited;
                return Ok(None);
            }
        }
        let segment = self.raw.pop_segment(usize::MAX);
        self.transfer = TransferMode::CloseDelimited;
        Ok(segment)
    }

    async fn next_chunked(&mut self, mut state: ChunkState) -> Result<Option<TimedBytes>> {
        loop {
            match state {
                ChunkState::Size => {
                    let line = self.read_raw_line(MAX_CHUNK_LINE_BYTES).await?;
                    let token = line.split(|byte| *byte == b';').next().unwrap_or_default();
                    let token = std::str::from_utf8(token)
                        .map_err(|_| BenchError::Http("chunk size is not ASCII".to_owned()))?;
                    let size = u64::from_str_radix(token.trim(), 16).map_err(|_| {
                        BenchError::Http("chunk size is not valid hexadecimal".to_owned())
                    })?;
                    state = if size == 0 {
                        ChunkState::Trailers
                    } else {
                        ChunkState::Data(size)
                    };
                }
                ChunkState::Data(mut remaining) => {
                    if remaining == 0 {
                        state = ChunkState::DataCrlf;
                        continue;
                    }
                    if self.raw.is_empty() {
                        self.read_more().await?;
                        if self.eof {
                            return Err(BenchError::Http(
                                "connection closed inside a chunk".to_owned(),
                            ));
                        }
                    }
                    let maximum = usize::try_from(remaining).unwrap_or(usize::MAX);
                    let segment = self
                        .raw
                        .pop_segment(maximum)
                        .expect("a non-empty raw buffer yields a segment");
                    remaining -= u64::try_from(segment.bytes.len()).expect("usize fits u64");
                    self.transfer = TransferMode::Chunked(ChunkState::Data(remaining));
                    return Ok(Some(segment));
                }
                ChunkState::DataCrlf => {
                    if self.read_raw_exact(2).await? != b"\r\n" {
                        return Err(BenchError::Http(
                            "chunk data is not followed by CRLF".to_owned(),
                        ));
                    }
                    state = ChunkState::Size;
                }
                ChunkState::Trailers => {
                    let line = self.read_raw_line(MAX_HEADER_BYTES).await?;
                    if line.is_empty() {
                        state = ChunkState::End;
                    } else if !line.contains(&b':') {
                        return Err(BenchError::Http("malformed chunk trailer field".to_owned()));
                    }
                }
                ChunkState::End => {
                    if !self.raw.is_empty() {
                        return Err(BenchError::Http(
                            "bytes remain after the chunked message terminator".to_owned(),
                        ));
                    }
                    self.transfer = TransferMode::Chunked(ChunkState::End);
                    return Ok(None);
                }
            }
        }
    }

    async fn read_more(&mut self) -> Result<()> {
        if self.eof {
            return Ok(());
        }
        let count = read_into(&mut self.stream, &mut self.raw, &mut self.response_bytes).await?;
        self.eof = count == 0;
        Ok(())
    }

    async fn read_raw_exact(&mut self, count: usize) -> Result<Vec<u8>> {
        while self.raw.len() < count {
            self.read_more().await?;
            if self.eof {
                return Err(BenchError::Http(
                    "connection closed inside transfer framing".to_owned(),
                ));
            }
        }
        self.raw.drain_bytes(count)
    }

    async fn read_raw_line(&mut self, maximum: usize) -> Result<Vec<u8>> {
        loop {
            if let Some(position) = self.raw.find(b"\r\n") {
                if position > maximum {
                    return Err(BenchError::Http(format!(
                        "transfer framing line exceeds {maximum} bytes"
                    )));
                }
                let mut line = self.raw.drain_bytes(position + 2)?;
                line.truncate(position);
                return Ok(line);
            }
            if self.raw.len() > maximum {
                return Err(BenchError::Http(format!(
                    "transfer framing line exceeds {maximum} bytes"
                )));
            }
            self.read_more().await?;
            if self.eof {
                return Err(BenchError::Http(
                    "connection closed inside transfer framing line".to_owned(),
                ));
            }
        }
    }
}

async fn read_into(
    stream: &mut TcpStream,
    target: &mut TimedBuffer,
    response_bytes: &mut u64,
) -> Result<usize> {
    let mut buffer = vec![0_u8; READ_BUFFER_BYTES];
    let count = stream.read(&mut buffer).await?;
    let arrived = Instant::now();
    buffer.truncate(count);
    *response_bytes += u64::try_from(count).expect("usize fits u64");
    target.push(buffer, arrived);
    Ok(count)
}

fn parse_head(bytes: &[u8]) -> Result<ResponseHead> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| BenchError::Http("response headers are not valid ASCII/UTF-8".to_owned()))?;
    let mut lines = text[..text.len() - 4].split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| BenchError::Http("response status line is missing".to_owned()))?;
    let mut status_fields = status_line.split_whitespace();
    let version = status_fields.next().unwrap_or_default();
    if !matches!(version, "HTTP/1.0" | "HTTP/1.1") {
        return Err(BenchError::Http(
            "response is not HTTP/1.0 or HTTP/1.1".to_owned(),
        ));
    }
    let status = status_fields
        .next()
        .ok_or_else(|| BenchError::Http("response status is missing".to_owned()))?
        .parse::<u16>()
        .map_err(|_| BenchError::Http("response status is invalid".to_owned()))?;
    let mut headers = BTreeMap::new();
    for line in lines {
        if line.starts_with([' ', '\t']) {
            return Err(BenchError::Http(
                "obsolete folded response headers are rejected".to_owned(),
            ));
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| BenchError::Http("malformed response header".to_owned()))?;
        if name.is_empty() || !name.bytes().all(is_header_name_byte) {
            return Err(BenchError::Http("invalid response header name".to_owned()));
        }
        let name = name.to_ascii_lowercase();
        let value = value.trim().to_owned();
        headers
            .entry(name)
            .and_modify(|existing: &mut String| {
                existing.push_str(", ");
                existing.push_str(&value);
            })
            .or_insert(value);
    }
    Ok(ResponseHead { status, headers })
}

const fn is_header_name_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}

fn select_transfer(head: &ResponseHead) -> Result<TransferMode> {
    let transfer_encoding = head.header("transfer-encoding");
    let content_length = head.header("content-length");
    if transfer_encoding.is_some() && content_length.is_some() {
        return Err(BenchError::Http(
            "ambiguous response has both Transfer-Encoding and Content-Length".to_owned(),
        ));
    }
    if let Some(value) = transfer_encoding {
        let codings: Vec<_> = value
            .split(',')
            .map(|item| item.trim().to_ascii_lowercase())
            .collect();
        if codings.last().map(String::as_str) != Some("chunked")
            || codings.iter().any(|coding| coding != "chunked")
        {
            return Err(BenchError::Http(
                "only chunked Transfer-Encoding is supported".to_owned(),
            ));
        }
        return Ok(TransferMode::Chunked(ChunkState::Size));
    }
    if let Some(value) = content_length {
        let length = value
            .parse::<u64>()
            .map_err(|_| BenchError::Http("invalid Content-Length".to_owned()))?;
        return Ok(TransferMode::ContentLength(length));
    }
    Ok(TransferMode::CloseDelimited)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timed_buffer_finds_delimiter_across_segments() {
        let now = Instant::now();
        let mut buffer = TimedBuffer::default();
        buffer.push(b"abc\r".to_vec(), now);
        buffer.push(b"\ndef".to_vec(), now);
        assert_eq!(buffer.find(b"\r\n"), Some(3));
        assert_eq!(buffer.drain_bytes(5).unwrap(), b"abc\r\n");
        assert_eq!(buffer.drain_bytes(3).unwrap(), b"def");
    }

    #[test]
    fn rejects_transfer_length_ambiguity() {
        let head = parse_head(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 1\r\n\r\n",
        )
        .unwrap();
        assert!(select_transfer(&head).is_err());
    }
}
