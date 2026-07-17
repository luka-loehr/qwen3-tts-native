use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use uuid::Uuid;

use crate::api::ValidationError;
use crate::engine::{EngineError, EngineErrorKind};

#[derive(Clone, Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub title: &'static str,
    pub detail: String,
    pub request_id: Option<Uuid>,
}

#[derive(Serialize)]
struct ProblemDetails<'a> {
    #[serde(rename = "type")]
    problem_type: String,
    title: &'a str,
    status: u16,
    code: &'a str,
    detail: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<Uuid>,
}

impl ApiError {
    pub fn new(
        status: StatusCode,
        code: &'static str,
        title: &'static str,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            status,
            code,
            title,
            detail: detail.into(),
            request_id: None,
        }
    }

    pub fn with_request_id(mut self, request_id: Uuid) -> Self {
        self.request_id = Some(request_id);
        self
    }

    pub fn validation(error: ValidationError) -> Self {
        Self::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            error.code,
            "Request validation failed",
            error.detail,
        )
    }

    pub fn engine(error: EngineError, request_id: Option<Uuid>) -> Self {
        let (status, code, title, detail) = match error.kind {
            EngineErrorKind::InvalidRequest => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_request",
                "Native request was rejected",
                error.message,
            ),
            EngineErrorKind::Capacity => (
                StatusCode::TOO_MANY_REQUESTS,
                "capacity_exhausted",
                "Native request capacity is exhausted",
                "all native request slots are busy".to_owned(),
            ),
            EngineErrorKind::Cancelled => (
                StatusCode::CONFLICT,
                "request_cancelled",
                "Request was cancelled",
                "the native request was cancelled".to_owned(),
            ),
            EngineErrorKind::BackendUnavailable => (
                StatusCode::SERVICE_UNAVAILABLE,
                "backend_unavailable",
                "Native backend is unavailable",
                "the native VoiceDesign runtime is unavailable".to_owned(),
            ),
            EngineErrorKind::Internal => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "Internal server error",
                "the native request violated an internal invariant".to_owned(),
            ),
        };
        Self {
            status,
            code,
            title,
            detail,
            request_id,
        }
    }

    pub fn malformed_json(status: StatusCode) -> Self {
        match status {
            StatusCode::UNSUPPORTED_MEDIA_TYPE => Self::new(
                status,
                "unsupported_media_type",
                "Unsupported media type",
                "Content-Type must be application/json",
            ),
            StatusCode::PAYLOAD_TOO_LARGE => Self::new(
                status,
                "request_too_large",
                "Request body is too large",
                "the JSON request exceeds the configured byte limit",
            ),
            _ => Self::new(
                StatusCode::BAD_REQUEST,
                "malformed_json",
                "Malformed JSON request",
                "the request body must be valid JSON and contain no unknown fields",
            ),
        }
    }

    pub fn stream_payload(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "error",
            "request_id": self.request_id,
            "error": {
                "code": self.code,
                "detail": self.detail,
            }
        })
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut response = (
            self.status,
            Json(ProblemDetails {
                problem_type: format!("urn:qwen3-tts-native:problem:{}", self.code),
                title: self.title,
                status: self.status.as_u16(),
                code: self.code,
                detail: &self.detail,
                request_id: self.request_id,
            }),
        )
            .into_response();
        response.headers_mut().insert(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/problem+json"),
        );
        if self.status == StatusCode::TOO_MANY_REQUESTS {
            response.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                axum::http::HeaderValue::from_static("1"),
            );
        }
        if let Some(request_id) = self.request_id
            && let Ok(value) = axum::http::HeaderValue::from_str(&request_id.to_string())
        {
            response.headers_mut().insert("x-request-id", value);
        }
        response
    }
}
