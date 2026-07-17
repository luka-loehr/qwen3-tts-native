use std::collections::BTreeMap;
use std::time::Instant;

use crate::http::{HttpBody, TimedBuffer};
use crate::{BenchError, Result};

const MAX_PART_HEADER_BYTES: usize = 64 * 1024;
const MAX_PART_PAYLOAD_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug)]
pub(crate) struct Part {
    pub headers: BTreeMap<String, String>,
    pub payload: Vec<u8>,
    pub first_payload_at: Option<Instant>,
}

impl Part {
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }
}

pub(crate) struct MultipartReader {
    body: HttpBody,
    buffer: TimedBuffer,
    boundary_line: Vec<u8>,
    closing_boundary_line: Vec<u8>,
    started: bool,
    finished: bool,
}

impl MultipartReader {
    pub(crate) fn new(body: HttpBody, boundary: &str) -> Self {
        Self {
            body,
            buffer: TimedBuffer::default(),
            boundary_line: format!("--{boundary}").into_bytes(),
            closing_boundary_line: format!("--{boundary}--").into_bytes(),
            started: false,
            finished: false,
        }
    }

    #[must_use]
    pub const fn response_bytes(&self) -> u64 {
        self.body.response_bytes()
    }

    pub(crate) async fn next_part(&mut self) -> Result<Option<Part>> {
        if self.finished {
            return Ok(None);
        }
        if !self.started {
            let opening = self.read_line(MAX_PART_HEADER_BYTES).await?;
            if opening != self.boundary_line {
                return Err(BenchError::Multipart(
                    "body does not begin with the declared boundary".to_owned(),
                ));
            }
            self.started = true;
        }

        let header_bytes = self.read_until(b"\r\n\r\n", MAX_PART_HEADER_BYTES).await?;
        let headers = parse_part_headers(&header_bytes)?;
        let length = headers
            .get("content-length")
            .ok_or_else(|| {
                BenchError::Multipart("every part must declare Content-Length".to_owned())
            })?
            .parse::<u64>()
            .map_err(|_| BenchError::Multipart("invalid part Content-Length".to_owned()))?;
        if length > MAX_PART_PAYLOAD_BYTES {
            return Err(BenchError::Multipart(format!(
                "part payload exceeds {MAX_PART_PAYLOAD_BYTES} bytes"
            )));
        }

        let capacity = usize::try_from(length).map_err(|_| {
            BenchError::Multipart("part Content-Length does not fit memory".to_owned())
        })?;
        let mut payload = Vec::with_capacity(capacity);
        let mut remaining = capacity;
        let mut first_payload_at = None;
        while remaining > 0 {
            self.fill_if_empty().await?;
            let segment = self
                .buffer
                .pop_segment(remaining)
                .ok_or_else(|| BenchError::Multipart("part payload is truncated".to_owned()))?;
            first_payload_at.get_or_insert(segment.arrived);
            remaining -= segment.bytes.len();
            payload.extend_from_slice(&segment.bytes);
        }

        let payload_suffix = self.read_exact(2).await?;
        if payload_suffix != b"\r\n" {
            return Err(BenchError::Multipart(
                "part payload is not followed by CRLF".to_owned(),
            ));
        }
        let next_boundary = self.read_line(MAX_PART_HEADER_BYTES).await?;
        if next_boundary == self.closing_boundary_line {
            self.finish_after_closing_boundary().await?;
            self.finished = true;
        } else if next_boundary != self.boundary_line {
            return Err(BenchError::Multipart(
                "part is followed by an invalid boundary".to_owned(),
            ));
        }

        Ok(Some(Part {
            headers,
            payload,
            first_payload_at,
        }))
    }

    async fn finish_after_closing_boundary(&mut self) -> Result<()> {
        if !self.buffer.is_empty() {
            return Err(BenchError::Multipart(
                "multipart closing boundary is followed by epilogue bytes".to_owned(),
            ));
        }
        while let Some(segment) = self.body.next_segment().await? {
            if !segment.bytes.is_empty() {
                return Err(BenchError::Multipart(
                    "multipart closing boundary is followed by epilogue bytes".to_owned(),
                ));
            }
        }
        Ok(())
    }

    async fn fill_if_empty(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            let segment = self.body.next_segment().await?.ok_or_else(|| {
                BenchError::Multipart("multipart body ended unexpectedly".to_owned())
            })?;
            self.buffer.push(segment.bytes, segment.arrived);
        }
        Ok(())
    }

    async fn read_exact(&mut self, count: usize) -> Result<Vec<u8>> {
        while self.buffer.len() < count {
            let segment = self.body.next_segment().await?.ok_or_else(|| {
                BenchError::Multipart("multipart body ended unexpectedly".to_owned())
            })?;
            self.buffer.push(segment.bytes, segment.arrived);
        }
        self.buffer.drain_bytes(count).map_err(|error| {
            BenchError::Multipart(format!("could not consume multipart framing: {error}"))
        })
    }

    async fn read_line(&mut self, maximum: usize) -> Result<Vec<u8>> {
        self.read_until(b"\r\n", maximum).await
    }

    async fn read_until(&mut self, delimiter: &[u8], maximum: usize) -> Result<Vec<u8>> {
        loop {
            if let Some(position) = self.buffer.find(delimiter) {
                if position > maximum {
                    return Err(BenchError::Multipart(format!(
                        "multipart framing exceeds {maximum} bytes"
                    )));
                }
                let mut bytes = self.buffer.drain_bytes(position + delimiter.len())?;
                bytes.truncate(position);
                return Ok(bytes);
            }
            if self.buffer.len() > maximum {
                return Err(BenchError::Multipart(format!(
                    "multipart framing exceeds {maximum} bytes"
                )));
            }
            let segment = self.body.next_segment().await?.ok_or_else(|| {
                BenchError::Multipart("multipart body ended before a delimiter".to_owned())
            })?;
            self.buffer.push(segment.bytes, segment.arrived);
        }
    }
}

pub(crate) fn boundary_from_content_type(content_type: &str) -> Result<String> {
    let parameters = split_parameters(content_type)?;
    let media_type = parameters
        .first()
        .ok_or_else(|| BenchError::Multipart("Content-Type is empty".to_owned()))?;
    if !media_type.trim().eq_ignore_ascii_case("multipart/mixed") {
        return Err(BenchError::Multipart(
            "streaming response must use multipart/mixed".to_owned(),
        ));
    }
    let mut boundary = None;
    for parameter in parameters.iter().skip(1) {
        let (name, value) = parameter
            .split_once('=')
            .ok_or_else(|| BenchError::Multipart("malformed Content-Type parameter".to_owned()))?;
        if name.trim().eq_ignore_ascii_case("boundary") {
            if boundary.is_some() {
                return Err(BenchError::Multipart(
                    "Content-Type contains multiple boundary parameters".to_owned(),
                ));
            }
            boundary = Some(unquote_parameter(value.trim())?);
        }
    }
    let boundary = boundary.ok_or_else(|| {
        BenchError::Multipart("multipart Content-Type has no boundary".to_owned())
    })?;
    validate_boundary(&boundary)?;
    Ok(boundary)
}

fn split_parameters(value: &str) -> Result<Vec<String>> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    let mut escaped = false;
    for character in value.chars() {
        if escaped {
            current.push(character);
            escaped = false;
            continue;
        }
        match character {
            '\\' if quoted => {
                current.push(character);
                escaped = true;
            }
            '"' => {
                quoted = !quoted;
                current.push(character);
            }
            ';' if !quoted => {
                fields.push(current.trim().to_owned());
                current.clear();
            }
            _ => current.push(character),
        }
    }
    if quoted || escaped {
        return Err(BenchError::Multipart(
            "unterminated quoted Content-Type parameter".to_owned(),
        ));
    }
    fields.push(current.trim().to_owned());
    Ok(fields)
}

fn unquote_parameter(value: &str) -> Result<String> {
    if !value.starts_with('"') {
        return Ok(value.to_owned());
    }
    if !value.ends_with('"') || value.len() < 2 {
        return Err(BenchError::Multipart(
            "unterminated quoted boundary".to_owned(),
        ));
    }
    let mut output = String::new();
    let mut escaped = false;
    for character in value[1..value.len() - 1].chars() {
        if escaped {
            output.push(character);
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        } else {
            output.push(character);
        }
    }
    if escaped {
        return Err(BenchError::Multipart(
            "quoted boundary ends with an escape".to_owned(),
        ));
    }
    Ok(output)
}

fn validate_boundary(boundary: &str) -> Result<()> {
    if boundary.is_empty() || boundary.len() > 70 {
        return Err(BenchError::Multipart(
            "boundary length must be in 1..=70 bytes".to_owned(),
        ));
    }
    if boundary.ends_with(' ')
        || !boundary.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(
                    byte,
                    b'\''
                        | b'('
                        | b')'
                        | b'+'
                        | b'_'
                        | b','
                        | b'-'
                        | b'.'
                        | b'/'
                        | b':'
                        | b'='
                        | b'?'
                        | b' '
                )
        })
    {
        return Err(BenchError::Multipart(
            "boundary contains invalid characters".to_owned(),
        ));
    }
    Ok(())
}

fn parse_part_headers(bytes: &[u8]) -> Result<BTreeMap<String, String>> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| BenchError::Multipart("part headers are not valid ASCII/UTF-8".to_owned()))?;
    let mut headers = BTreeMap::new();
    for line in text.split("\r\n") {
        if line.starts_with([' ', '\t']) {
            return Err(BenchError::Multipart(
                "folded part headers are rejected".to_owned(),
            ));
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| BenchError::Multipart("malformed part header".to_owned()))?;
        let name = name.to_ascii_lowercase();
        if headers.insert(name, value.trim().to_owned()).is_some() {
            return Err(BenchError::Multipart(
                "duplicate part headers are rejected".to_owned(),
            ));
        }
    }
    Ok(headers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_quoted_boundary_case_insensitively() {
        assert_eq!(
            boundary_from_content_type("Multipart/Mixed; charset=utf-8; boundary=\"qwen-123\"")
                .unwrap(),
            "qwen-123"
        );
    }

    #[test]
    fn rejects_header_injection_boundary() {
        assert!(boundary_from_content_type("multipart/mixed; boundary=bad\r\nvalue").is_err());
    }

    #[test]
    fn rejects_duplicate_boundary() {
        assert!(boundary_from_content_type("multipart/mixed; boundary=a; boundary=b").is_err());
    }
}
