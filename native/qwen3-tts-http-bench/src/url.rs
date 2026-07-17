use std::net::{IpAddr, SocketAddr};

use tokio::net::lookup_host;

use crate::{BenchError, Result};

#[derive(Clone, Debug)]
pub(crate) struct Endpoint {
    pub authority: String,
    pub path_and_query: String,
    pub addresses: Vec<SocketAddr>,
}

impl Endpoint {
    pub(crate) async fn parse_loopback(raw: &str) -> Result<Self> {
        let remainder = raw.strip_prefix("http://").ok_or_else(|| {
            BenchError::Configuration(
                "endpoint must use plain http:// on a loopback interface".to_owned(),
            )
        })?;
        if remainder.contains('#') || remainder.contains('@') {
            return Err(BenchError::Configuration(
                "endpoint must not contain fragments or user information".to_owned(),
            ));
        }
        let (authority, path) = remainder
            .split_once('/')
            .map_or((remainder, "/"), |(authority, path)| (authority, path));
        if authority.is_empty() {
            return Err(BenchError::Configuration(
                "endpoint authority is empty".to_owned(),
            ));
        }
        let path_and_query = if path == "/" {
            "/".to_owned()
        } else {
            format!("/{path}")
        };
        let (host, port) = split_authority(authority)?;
        let addresses: Vec<SocketAddr> = lookup_host((host.as_str(), port)).await?.collect();
        if addresses.is_empty() {
            return Err(BenchError::Configuration(
                "endpoint did not resolve to an address".to_owned(),
            ));
        }
        if addresses.iter().any(|address| !address.ip().is_loopback()) {
            return Err(BenchError::Configuration(
                "every resolved endpoint address must be loopback".to_owned(),
            ));
        }
        Ok(Self {
            authority: authority.to_owned(),
            path_and_query,
            addresses,
        })
    }
}

fn split_authority(authority: &str) -> Result<(String, u16)> {
    if let Some(bracketed) = authority.strip_prefix('[') {
        let (host, suffix) = bracketed.split_once(']').ok_or_else(|| {
            BenchError::Configuration("invalid bracketed IPv6 endpoint".to_owned())
        })?;
        let port = suffix
            .strip_prefix(':')
            .ok_or_else(|| BenchError::Configuration("endpoint port is required".to_owned()))?
            .parse::<u16>()
            .map_err(|_| BenchError::Configuration("endpoint port is invalid".to_owned()))?;
        host.parse::<IpAddr>()
            .map_err(|_| BenchError::Configuration("invalid IPv6 endpoint".to_owned()))?;
        return Ok((host.to_owned(), port));
    }
    let (host, port) = authority.rsplit_once(':').ok_or_else(|| {
        BenchError::Configuration("endpoint must include an explicit port".to_owned())
    })?;
    if host.is_empty() || host.contains(':') {
        return Err(BenchError::Configuration(
            "IPv6 endpoints must use brackets".to_owned(),
        ));
    }
    let port = port
        .parse::<u16>()
        .map_err(|_| BenchError::Configuration("endpoint port is invalid".to_owned()))?;
    Ok((host.to_owned(), port))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn accepts_ipv4_loopback_with_path() {
        let endpoint = Endpoint::parse_loopback("http://127.0.0.1:8080/v1/speech")
            .await
            .unwrap();
        assert_eq!(endpoint.path_and_query, "/v1/speech");
        assert!(
            endpoint
                .addresses
                .iter()
                .all(|item| item.ip().is_loopback())
        );
    }

    #[tokio::test]
    async fn rejects_non_loopback_literal() {
        let error = Endpoint::parse_loopback("http://192.0.2.10:8080/test")
            .await
            .unwrap_err();
        assert!(error.to_string().contains("loopback"));
    }
}
