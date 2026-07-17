use std::env;
use std::error::Error;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(2);
const MAX_RESPONSE_BYTES: u64 = 64 * 1024;

#[derive(Debug)]
struct Arguments {
    endpoint: Endpoint,
    expect_ready: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Endpoint {
    authority: String,
    path: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = parse_arguments(env::args().skip(1))?;
    let response = request(&arguments.endpoint, DEFAULT_TIMEOUT)?;
    validate_response(&response, arguments.expect_ready)?;
    Ok(())
}

fn parse_arguments(mut values: impl Iterator<Item = String>) -> Result<Arguments, String> {
    let mut endpoint = None;
    let mut expect_ready = false;
    while let Some(argument) = values.next() {
        match argument.as_str() {
            "--url" => {
                let value = values
                    .next()
                    .ok_or_else(|| "--url requires a value".to_owned())?;
                endpoint = Some(parse_endpoint(&value)?);
            }
            "--expect-ready" => expect_ready = true,
            _ => return Err(format!("unknown argument {argument:?}")),
        }
    }
    Ok(Arguments {
        endpoint: endpoint.ok_or_else(|| "--url is required".to_owned())?,
        expect_ready,
    })
}

fn parse_endpoint(value: &str) -> Result<Endpoint, String> {
    let remainder = value
        .strip_prefix("http://")
        .ok_or_else(|| "healthcheck URL must use http://".to_owned())?;
    let (authority, path) = match remainder.split_once('/') {
        Some((authority, path)) => (authority, format!("/{path}")),
        None => (remainder, "/".to_owned()),
    };
    if authority.is_empty() || authority.contains('@') {
        return Err("healthcheck URL has an invalid authority".to_owned());
    }
    if path.contains('\r') || path.contains('\n') || path.contains('#') {
        return Err("healthcheck URL has an invalid path".to_owned());
    }
    let addresses = resolve_loopback(authority)?;
    if addresses.is_empty() {
        return Err("healthcheck URL must resolve only to a loopback address".to_owned());
    }
    Ok(Endpoint {
        authority: authority.to_owned(),
        path,
    })
}

fn resolve_loopback(authority: &str) -> Result<Vec<SocketAddr>, String> {
    let addresses = authority
        .to_socket_addrs()
        .map_err(|error| format!("failed to resolve healthcheck authority: {error}"))?
        .collect::<Vec<_>>();
    if addresses.is_empty() || addresses.iter().any(|address| !address.ip().is_loopback()) {
        return Err("healthcheck authority must resolve exclusively to loopback".to_owned());
    }
    Ok(addresses)
}

fn request(endpoint: &Endpoint, timeout: Duration) -> Result<Vec<u8>, String> {
    let addresses = resolve_loopback(&endpoint.authority)?;
    let mut last_error = None;
    let mut stream = None;
    for address in addresses {
        match TcpStream::connect_timeout(&address, timeout) {
            Ok(connected) => {
                stream = Some(connected);
                break;
            }
            Err(error) => last_error = Some(error),
        }
    }
    let mut stream = stream.ok_or_else(|| {
        format!(
            "failed to connect to health endpoint: {}",
            last_error.map_or_else(
                || "no loopback address".to_owned(),
                |error| error.to_string()
            )
        )
    })?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|error| format!("failed to set healthcheck read timeout: {error}"))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|error| format!("failed to set healthcheck write timeout: {error}"))?;
    let request = format!(
        "GET {} HTTP/1.1\r\nHost: {}\r\nAccept: application/json\r\nConnection: close\r\n\r\n",
        endpoint.path, endpoint.authority
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|error| format!("failed to send healthcheck request: {error}"))?;
    let mut response = Vec::new();
    stream
        .take(MAX_RESPONSE_BYTES)
        .read_to_end(&mut response)
        .map_err(|error| format!("failed to read healthcheck response: {error}"))?;
    Ok(response)
}

fn validate_response(response: &[u8], expect_ready: bool) -> Result<(), String> {
    let separator = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| "healthcheck response has no HTTP header terminator".to_owned())?;
    let headers = std::str::from_utf8(&response[..separator])
        .map_err(|_| "healthcheck response headers are not UTF-8".to_owned())?;
    let status = headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(|| "healthcheck response has an invalid status line".to_owned())?;
    if status != 200 {
        return Err(format!("health endpoint returned HTTP {status}"));
    }
    if !expect_ready {
        return Ok(());
    }
    let body = &response[separator + 4..];
    let payload: serde_json::Value = serde_json::from_slice(body)
        .map_err(|error| format!("healthcheck response is not valid JSON: {error}"))?;
    if payload.get("status").and_then(serde_json::Value::as_str) != Some("ready")
        || payload
            .get("engine_loaded")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
    {
        return Err("health endpoint is live but not ready".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_is_restricted_to_loopback_http() {
        assert_eq!(
            parse_endpoint("http://127.0.0.1:8080/health/ready").unwrap(),
            Endpoint {
                authority: "127.0.0.1:8080".to_owned(),
                path: "/health/ready".to_owned(),
            }
        );
        assert!(parse_endpoint("https://127.0.0.1:8080/health/ready").is_err());
        assert!(parse_endpoint("http://example.com:8080/health/ready").is_err());
        assert!(parse_endpoint("http://127.0.0.1:8080/a\r\nX-Evil: yes").is_err());
    }

    #[test]
    fn readiness_requires_status_and_engine_loaded() {
        let ready = b"HTTP/1.1 200 OK\r\nContent-Length: 39\r\n\r\n{\"status\":\"ready\",\"engine_loaded\":true}";
        validate_response(ready, true).unwrap();
        let not_ready = b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 50\r\n\r\n{\"status\":\"not_ready\",\"engine_loaded\":false}";
        assert!(validate_response(not_ready, true).is_err());
    }
}
