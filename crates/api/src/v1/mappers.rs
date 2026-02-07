//! API v1 DTO mapping helpers.

use crate::v1::{ApiV1ErrorCode, ApiV1ErrorDto, ApiV1ErrorKind, ApiV1ErrorMeta, ApiV1Result};
use semantic_code_shared::{ErrorEnvelope, ErrorKind};

const API_V1_REDACTED: &str = "[REDACTED]";
const API_V1_REDACTED_PREFIX: &str = "[REDACTED,len=";

/// Convert a shared `ErrorCode` into an API v1 error code string.
#[must_use]
pub fn error_code_to_api_v1(code: &semantic_code_shared::ErrorCode) -> ApiV1ErrorCode {
    let namespace = sanitize_code_segment(code.namespace());
    let detail = sanitize_code_segment(code.code());
    format!("ERR_{namespace}_{detail}")
}

/// Map an `ErrorEnvelope` into an API v1 error DTO.
#[must_use]
pub fn error_envelope_to_api_v1_error(
    envelope: &ErrorEnvelope,
    extra_meta: Option<ApiV1ErrorMeta>,
) -> ApiV1ErrorDto {
    let mut merged = ApiV1ErrorMeta::new();
    for (key, value) in &envelope.metadata {
        merged.insert(key.clone(), value.clone());
    }
    if let Some(extra) = extra_meta {
        for (key, value) in extra {
            merged.insert(key, value);
        }
    }
    let meta = if merged.is_empty() {
        None
    } else {
        Some(redact_api_v1_meta(&merged))
    };

    ApiV1ErrorDto {
        code: error_code_to_api_v1(&envelope.code),
        message: envelope.message.clone(),
        kind: map_error_kind(envelope.kind),
        meta,
    }
}

/// Map a shared result into an API v1 result wrapper.
#[must_use]
pub fn result_to_api_v1_result<T>(
    result: Result<T, ErrorEnvelope>,
    extra_meta: Option<ApiV1ErrorMeta>,
) -> ApiV1Result<T> {
    match result {
        Ok(data) => ApiV1Result::ok(data),
        Err(error) => ApiV1Result::err(error_envelope_to_api_v1_error(&error, extra_meta)),
    }
}

const fn map_error_kind(kind: ErrorKind) -> ApiV1ErrorKind {
    match kind {
        ErrorKind::Expected | ErrorKind::Unexpected => ApiV1ErrorKind::Expected,
        ErrorKind::Invariant => ApiV1ErrorKind::Invariant,
    }
}

fn sanitize_code_segment(segment: &str) -> String {
    segment
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn redact_api_v1_meta(meta: &ApiV1ErrorMeta) -> ApiV1ErrorMeta {
    let mut redacted = ApiV1ErrorMeta::new();
    for (key, value) in meta {
        let redacted_value = if is_secret_key(key) {
            API_V1_REDACTED.to_string()
        } else if is_query_key(key) {
            format!("{API_V1_REDACTED_PREFIX}{}]", value.len())
        } else {
            value.clone()
        };
        redacted.insert(key.clone(), redacted_value);
    }
    redacted
}

fn is_secret_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    key.contains("api_key")
        || key.contains("apikey")
        || key.contains("token")
        || key.contains("password")
        || key.contains("secret")
        || key.contains("authorization")
        || key.contains("bearer")
}

fn is_query_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    key == "query" || key.ends_with("query") || key == "content"
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope};
    use std::collections::BTreeMap;
    use std::error::Error;

    #[test]
    fn mapping_redacts_sensitive_metadata() -> Result<(), Box<dyn Error>> {
        let envelope = ErrorEnvelope::expected(
            ErrorCode::new("domain", "invalid_collection_name"),
            "bad name",
        )
        .with_metadata("token", "secret-token")
        .with_metadata("apiKey", "sk-123")
        .with_metadata("query", "hello world")
        .with_metadata("path", "src/lib.rs");

        let dto = error_envelope_to_api_v1_error(&envelope, None);
        let meta = dto.meta.ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::Other, "meta should be present")
        })?;
        assert_eq!(dto.code, "ERR_DOMAIN_INVALID_COLLECTION_NAME");
        assert_eq!(dto.kind, ApiV1ErrorKind::Expected);
        assert_eq!(meta.get("token").map(String::as_str), Some("[REDACTED]"));
        assert_eq!(meta.get("apiKey").map(String::as_str), Some("[REDACTED]"));
        assert_eq!(
            meta.get("query").map(String::as_str),
            Some("[REDACTED,len=11]")
        );
        assert_eq!(meta.get("path").map(String::as_str), Some("src/lib.rs"));
        Ok(())
    }

    #[test]
    fn unexpected_errors_map_to_expected_kind() {
        let envelope = ErrorEnvelope::unexpected(ErrorCode::io(), "io", ErrorClass::Retriable);
        let dto = error_envelope_to_api_v1_error(&envelope, None);
        assert_eq!(dto.kind, ApiV1ErrorKind::Expected);
    }

    #[test]
    fn result_mapping_preserves_ok_and_err() {
        let ok_result: Result<u32, ErrorEnvelope> = Ok(10);
        let mapped = result_to_api_v1_result(ok_result, None);
        assert!(matches!(mapped, ApiV1Result::Ok { ok: true, .. }));

        let mut extra = BTreeMap::new();
        extra.insert("requestId".to_string(), "abc".to_string());
        let err_result: Result<u32, ErrorEnvelope> = Err(ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            "bad input",
        ));
        let mapped = result_to_api_v1_result(err_result, Some(extra));
        assert!(matches!(mapped, ApiV1Result::Err { ok: false, .. }));
    }
}
