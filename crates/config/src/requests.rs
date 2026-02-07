//! Request DTOs and validation helpers.
//!
//! Requests are boundary inputs (CLI/API) and must be validated before being
//! passed into use-cases. Validation here is limited to:
//! - shape (required fields, trimming)
//! - bounds (topK/threshold)
//! - provider-facing allowlists (filterExpr)
//!
//! Domain invariants (e.g. `CollectionName` pattern) are delegated to domain
//! constructors and not duplicated here.

use schemars::JsonSchema;
use semantic_code_domain::CollectionName;
use semantic_code_shared::{ErrorCode, ErrorEnvelope, Validate, Validated, ValidationError};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;

/// Index request payload (boundary DTO).
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    JsonSchema,
    semantic_code_validate_derive::Validate,
)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[validate(error = "RequestValidationError")]
pub struct IndexRequestDto {
    /// Root path of the codebase to index.
    #[validate(field = "codebaseRoot", non_empty)]
    pub codebase_root: String,
    /// Optional override collection name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[validate(field = "collectionName", non_empty)]
    pub collection_name: Option<String>,
    /// Force a full reindex.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force_reindex: Option<bool>,
}

/// Search request payload (boundary DTO).
#[derive(
    Debug,
    Clone,
    PartialEq,
    Serialize,
    Deserialize,
    JsonSchema,
    semantic_code_validate_derive::Validate,
)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[validate(error = "RequestValidationError")]
pub struct SearchRequestDto {
    /// Root path of the codebase to search.
    #[validate(field = "codebaseRoot", non_empty)]
    pub codebase_root: String,
    /// User query string.
    #[validate(non_empty)]
    pub query: String,
    /// Optional top-k limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[validate(field = "topK", range(min = 1, max = 50))]
    pub top_k: Option<u32>,
    /// Optional similarity threshold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 0.0, max = 1.0))]
    pub threshold: Option<f64>,
    /// Optional filter expression (provider-specific).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter_expr: Option<String>,
    /// Optional hint to include content payloads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_content: Option<bool>,
}

/// Reindex-by-change request payload (boundary DTO).
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    JsonSchema,
    semantic_code_validate_derive::Validate,
)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[validate(error = "RequestValidationError")]
pub struct ReindexByChangeRequestDto {
    /// Root path of the codebase to reindex.
    #[validate(field = "codebaseRoot", non_empty)]
    pub codebase_root: String,
}

/// Clear-index request payload (boundary DTO).
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    JsonSchema,
    semantic_code_validate_derive::Validate,
)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[validate(error = "RequestValidationError")]
pub struct ClearIndexRequestDto {
    /// Root path of the codebase to clear.
    #[validate(field = "codebaseRoot", non_empty)]
    pub codebase_root: String,
}

/// Validated index request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexRequest {
    /// Normalized codebase root.
    pub codebase_root: PathBuf,
    /// Optional validated collection name override.
    pub collection_name: Option<CollectionName>,
    /// Force a full reindex.
    pub force_reindex: bool,
}

/// Validated index request proof.
pub type ValidatedIndexRequest = Validated<IndexRequest>;

/// Validated search request.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchRequest {
    /// Normalized codebase root.
    pub codebase_root: PathBuf,
    /// Trimmed query.
    pub query: Box<str>,
    /// Optional top-k limit.
    pub top_k: Option<u32>,
    /// Optional similarity threshold.
    pub threshold: Option<f64>,
    /// Optional validated filter expression (trimmed).
    pub filter_expr: Option<Box<str>>,
    /// Optional include-content hint.
    pub include_content: Option<bool>,
}

/// Validated search request proof.
pub type ValidatedSearchRequest = Validated<SearchRequest>;

/// Validated reindex-by-change request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReindexByChangeRequest {
    /// Normalized codebase root.
    pub codebase_root: PathBuf,
}

/// Validated reindex-by-change request proof.
pub type ValidatedReindexByChangeRequest = Validated<ReindexByChangeRequest>;

/// Validated clear-index request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClearIndexRequest {
    /// Normalized codebase root.
    pub codebase_root: PathBuf,
}

/// Validated clear-index request proof.
pub type ValidatedClearIndexRequest = Validated<ClearIndexRequest>;

/// Request validation errors mapped to `ErrorEnvelope`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestValidationError {
    /// A required string field is empty after trimming.
    EmptyField {
        /// Field name that failed validation.
        field: &'static str,
    },
    /// A string field contains invalid content.
    InvalidField {
        /// Field name that failed validation.
        field: &'static str,
        /// Short reason describing why validation failed.
        reason: &'static str,
    },
    /// Codebase root path is invalid.
    InvalidCodebaseRoot {
        /// Raw (trimmed) value, or a placeholder for non-printable input.
        value: String,
        /// Short reason describing why validation failed.
        reason: &'static str,
    },
    /// A numeric field is out of bounds.
    OutOfRange {
        /// Field name that failed validation.
        field: &'static str,
        /// Value provided (stringified).
        value: String,
        /// Inclusive minimum bound (stringified).
        min: String,
        /// Inclusive maximum bound (stringified).
        max: String,
    },
    /// Filter expression is not supported by the allowlist grammar.
    UnsupportedFilterExpr {
        /// Raw filter expression input.
        expr: String,
    },
}

impl RequestValidationError {
    fn error_code(&self) -> ErrorCode {
        match self {
            Self::EmptyField { .. } => ErrorCode::new("config", "empty_field"),
            Self::InvalidField { .. } => ErrorCode::new("config", "invalid_field"),
            Self::InvalidCodebaseRoot { .. } => ErrorCode::new("config", "invalid_codebase_root"),
            Self::OutOfRange { .. } => ErrorCode::new("config", "out_of_range"),
            Self::UnsupportedFilterExpr { .. } => ErrorCode::new("config", "invalid_filter_expr"),
        }
    }
}

impl fmt::Display for RequestValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField { field } => write!(formatter, "{field} must be non-empty"),
            Self::InvalidField { field, reason } => {
                write!(formatter, "{field} is invalid: {reason}")
            },
            Self::InvalidCodebaseRoot { reason, .. } => {
                write!(formatter, "codebaseRoot is invalid: {reason}")
            },
            Self::OutOfRange {
                field, min, max, ..
            } => {
                write!(formatter, "{field} must be between {min} and {max}")
            },
            Self::UnsupportedFilterExpr { .. } => {
                formatter.write_str("filterExpr is not supported")
            },
        }
    }
}

impl std::error::Error for RequestValidationError {}

impl ValidationError for RequestValidationError {
    fn empty(field: &'static str) -> Self {
        Self::EmptyField { field }
    }

    fn invalid(field: &'static str, reason: &'static str) -> Self {
        Self::InvalidField { field, reason }
    }

    fn out_of_range(field: &'static str, value: String, min: String, max: String) -> Self {
        Self::OutOfRange {
            field,
            value,
            min,
            max,
        }
    }
}

impl From<RequestValidationError> for ErrorEnvelope {
    fn from(error: RequestValidationError) -> Self {
        let code = error.error_code();
        let message = error.to_string();
        let mut envelope = Self::expected(code, message);

        match error {
            RequestValidationError::EmptyField { field } => {
                envelope = envelope.with_metadata("field", field);
            },
            RequestValidationError::InvalidField { field, reason } => {
                envelope = envelope
                    .with_metadata("field", field)
                    .with_metadata("reason", reason);
            },
            RequestValidationError::InvalidCodebaseRoot { value, reason } => {
                envelope = envelope
                    .with_metadata("field", "codebaseRoot")
                    .with_metadata("reason", reason)
                    .with_metadata("value", value);
            },
            RequestValidationError::OutOfRange {
                field,
                value,
                min,
                max,
            } => {
                envelope = envelope
                    .with_metadata("field", field)
                    .with_metadata("value", value)
                    .with_metadata("min", min)
                    .with_metadata("max", max);
            },
            RequestValidationError::UnsupportedFilterExpr { expr } => {
                envelope = envelope
                    .with_metadata("field", "filterExpr")
                    .with_metadata("expr", expr);
            },
        }

        envelope
    }
}

/// Validate and normalize an index request.
pub fn validate_index_request(
    dto: &IndexRequestDto,
) -> Result<ValidatedIndexRequest, ErrorEnvelope> {
    dto.validate().map_err(ErrorEnvelope::from)?;
    let codebase_root = validate_codebase_root(&dto.codebase_root)?;

    let collection_name = match dto.collection_name.as_deref() {
        None => None,
        Some(raw) => {
            let trimmed = raw.trim();
            Some(CollectionName::parse(trimmed).map_err(ErrorEnvelope::from)?)
        },
    };

    Ok(Validated::new(IndexRequest {
        codebase_root,
        collection_name,
        force_reindex: dto.force_reindex.unwrap_or(false),
    }))
}

/// Validate and normalize a search request.
pub fn validate_search_request(
    dto: &SearchRequestDto,
) -> Result<ValidatedSearchRequest, ErrorEnvelope> {
    dto.validate().map_err(ErrorEnvelope::from)?;
    let codebase_root = validate_codebase_root(&dto.codebase_root)?;
    let query = require_trimmed("query", &dto.query)?;

    let filter_expr = match dto.filter_expr.as_deref() {
        None => None,
        Some(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                None
            } else {
                validate_filter_expr_allowlist(trimmed)?;
                Some(trimmed.to_owned().into_boxed_str())
            }
        },
    };

    Ok(Validated::new(SearchRequest {
        codebase_root,
        query,
        top_k: dto.top_k,
        threshold: dto.threshold,
        filter_expr,
        include_content: dto.include_content,
    }))
}

/// Validate a reindex-by-change request.
pub fn validate_reindex_by_change_request(
    dto: &ReindexByChangeRequestDto,
) -> Result<ValidatedReindexByChangeRequest, ErrorEnvelope> {
    dto.validate().map_err(ErrorEnvelope::from)?;
    let codebase_root = validate_codebase_root(&dto.codebase_root)?;
    Ok(Validated::new(ReindexByChangeRequest { codebase_root }))
}

/// Validate a clear-index request.
pub fn validate_clear_index_request(
    dto: &ClearIndexRequestDto,
) -> Result<ValidatedClearIndexRequest, ErrorEnvelope> {
    dto.validate().map_err(ErrorEnvelope::from)?;
    let codebase_root = validate_codebase_root(&dto.codebase_root)?;
    Ok(Validated::new(ClearIndexRequest { codebase_root }))
}

/// Parse and validate an index request from JSON.
pub fn parse_index_request_json(input: &str) -> Result<ValidatedIndexRequest, ErrorEnvelope> {
    let dto: IndexRequestDto = parse_request_json("index", input)?;
    validate_index_request(&dto)
}

/// Parse and validate a search request from JSON.
pub fn parse_search_request_json(input: &str) -> Result<ValidatedSearchRequest, ErrorEnvelope> {
    let dto: SearchRequestDto = parse_request_json("search", input)?;
    validate_search_request(&dto)
}

/// Parse and validate a reindex-by-change request from JSON.
pub fn parse_reindex_by_change_request_json(
    input: &str,
) -> Result<ValidatedReindexByChangeRequest, ErrorEnvelope> {
    let dto: ReindexByChangeRequestDto = parse_request_json("reindexByChange", input)?;
    validate_reindex_by_change_request(&dto)
}

/// Parse and validate a clear-index request from JSON.
pub fn parse_clear_index_request_json(
    input: &str,
) -> Result<ValidatedClearIndexRequest, ErrorEnvelope> {
    let dto: ClearIndexRequestDto = parse_request_json("clearIndex", input)?;
    validate_clear_index_request(&dto)
}

fn parse_request_json<T: DeserializeOwned>(
    kind: &'static str,
    input: &str,
) -> Result<T, ErrorEnvelope> {
    serde_json::from_str(input).map_err(|error| {
        ErrorEnvelope::expected(
            ErrorCode::new("config", "invalid_json"),
            format!("invalid {kind} request JSON: {error}"),
        )
        .with_metadata("request_kind", kind)
    })
}

fn require_trimmed(field: &'static str, value: &str) -> Result<Box<str>, ErrorEnvelope> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(RequestValidationError::EmptyField { field }.into());
    }
    if trimmed.contains('\0') {
        return Err(RequestValidationError::InvalidField {
            field,
            reason: "contains NUL byte",
        }
        .into());
    }
    Ok(trimmed.to_owned().into_boxed_str())
}

fn validate_codebase_root(raw: &str) -> Result<PathBuf, ErrorEnvelope> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(RequestValidationError::EmptyField {
            field: "codebaseRoot",
        }
        .into());
    }

    if trimmed.contains('\0') {
        return Err(RequestValidationError::InvalidCodebaseRoot {
            value: "<nul>".to_string(),
            reason: "path contains NUL",
        }
        .into());
    }

    if trimmed.contains("://") {
        return Err(RequestValidationError::InvalidCodebaseRoot {
            value: trimmed.to_owned(),
            reason: "path must be a filesystem path (not a URL)",
        }
        .into());
    }

    Ok(PathBuf::from(trimmed))
}

/// Allowlist grammar for filter expressions.
///
/// Currently supported:
/// - `relativePath == '<value>'`
/// - `relativePath != '<value>'`
/// - `language == '<value>'`
/// - `fileExtension == '<value>'`
///
/// Where `<value>` is a single-quoted or double-quoted string with no newlines.
pub fn validate_filter_expr_allowlist(expr: &str) -> Result<(), ErrorEnvelope> {
    if expr.contains('\n') || expr.contains('\r') {
        return Err(RequestValidationError::UnsupportedFilterExpr {
            expr: expr.to_owned(),
        }
        .into());
    }

    let (field, op, value) = parse_simple_comparison(expr).ok_or_else(|| {
        ErrorEnvelope::from(RequestValidationError::UnsupportedFilterExpr {
            expr: expr.to_owned(),
        })
    })?;

    let allowed_field = matches!(field, "relativePath" | "language" | "fileExtension");
    let allowed_op = matches!(op, "==" | "!=");
    let non_empty_value = !value.is_empty();

    if allowed_field && allowed_op && non_empty_value {
        Ok(())
    } else {
        Err(RequestValidationError::UnsupportedFilterExpr {
            expr: expr.to_owned(),
        }
        .into())
    }
}

fn parse_simple_comparison(input: &str) -> Option<(&str, &str, &str)> {
    let input = input.trim();
    let (field, rest) = split_once_ws(input)?;
    let rest = rest.trim_start();

    let (op, rest) = if let Some(rest) = rest.strip_prefix("==") {
        ("==", rest)
    } else if let Some(rest) = rest.strip_prefix("!=") {
        ("!=", rest)
    } else {
        return None;
    };

    let value = rest.trim_start();
    let unquoted = strip_quotes(value)?;
    Some((field, op, unquoted))
}

fn split_once_ws(input: &str) -> Option<(&str, &str)> {
    for (idx, ch) in input.char_indices() {
        if ch.is_whitespace() {
            let (left, right) = input.split_at(idx);
            return Some((left, right));
        }
    }
    None
}

fn strip_quotes(input: &str) -> Option<&str> {
    let input = input.trim();
    if input.len() < 2 {
        return None;
    }
    let bytes = input.as_bytes();
    let first = *bytes.first()?;
    let last = *bytes.last()?;
    if (first == b'\'' && last == b'\'') || (first == b'"' && last == b'"') {
        Some(&input[1..input.len() - 1])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn accepts_valid_search_request() -> Result<(), Box<dyn Error>> {
        let dto = SearchRequestDto {
            codebase_root: "/tmp/repo".to_string(),
            query: " hello ".to_string(),
            top_k: Some(10),
            threshold: Some(0.5),
            filter_expr: None,
            include_content: Some(true),
        };

        let validated = validate_search_request(&dto)?;
        assert_eq!(validated.query.as_ref(), "hello");
        assert_eq!(validated.top_k, Some(10));
        assert_eq!(validated.threshold, Some(0.5));
        Ok(())
    }

    #[test]
    fn rejects_invalid_top_k() {
        let dto = SearchRequestDto {
            codebase_root: "/tmp/repo".to_string(),
            query: "hello".to_string(),
            top_k: Some(0),
            threshold: None,
            filter_expr: None,
            include_content: None,
        };

        let error = validate_search_request(&dto).err();
        assert!(
            matches!(error, Some(envelope) if envelope.code == ErrorCode::new("config", "out_of_range"))
        );
    }

    #[test]
    fn rejects_invalid_threshold() {
        let dto = SearchRequestDto {
            codebase_root: "/tmp/repo".to_string(),
            query: "hello".to_string(),
            top_k: None,
            threshold: Some(1.5),
            filter_expr: None,
            include_content: None,
        };

        let error = validate_search_request(&dto).err();
        assert!(
            matches!(error, Some(envelope) if envelope.code == ErrorCode::new("config", "out_of_range"))
        );
    }

    #[test]
    fn rejects_filter_expr_outside_allowlist() {
        let dto = SearchRequestDto {
            codebase_root: "/tmp/repo".to_string(),
            query: "hello".to_string(),
            top_k: None,
            threshold: None,
            filter_expr: Some("score > 0.5".to_string()),
            include_content: None,
        };

        let error = validate_search_request(&dto).err();
        assert!(
            matches!(error, Some(envelope) if envelope.code == ErrorCode::new("config", "invalid_filter_expr"))
        );
    }

    #[test]
    fn rejects_filter_expr_with_newlines() {
        let dto = SearchRequestDto {
            codebase_root: "/tmp/repo".to_string(),
            query: "hello".to_string(),
            top_k: None,
            threshold: None,
            filter_expr: Some("relativePath\n== 'a'".to_string()),
            include_content: None,
        };

        let error = validate_search_request(&dto).err();
        assert!(
            matches!(error, Some(envelope) if envelope.code == ErrorCode::new("config", "invalid_filter_expr"))
        );
    }

    #[test]
    fn rejects_codebase_root_with_url_scheme() {
        let dto = ClearIndexRequestDto {
            codebase_root: "https://example.com/repo".to_string(),
        };

        let error = validate_clear_index_request(&dto).err();
        assert!(
            matches!(error, Some(envelope) if envelope.code == ErrorCode::new("config", "invalid_codebase_root"))
        );
    }

    #[test]
    fn rejects_codebase_root_with_nul() {
        let dto = ReindexByChangeRequestDto {
            codebase_root: "repo\0x".to_string(),
        };

        let error = validate_reindex_by_change_request(&dto).err();
        assert!(
            matches!(error, Some(envelope) if envelope.code == ErrorCode::new("config", "invalid_codebase_root"))
        );
    }

    #[test]
    fn rejects_empty_query() {
        let dto = SearchRequestDto {
            codebase_root: "/tmp/repo".to_string(),
            query: "   ".to_string(),
            top_k: None,
            threshold: None,
            filter_expr: None,
            include_content: None,
        };

        let error = validate_search_request(&dto).err();
        assert!(
            matches!(error, Some(envelope) if envelope.code == ErrorCode::new("config", "empty_field"))
        );
    }

    #[test]
    fn rejects_query_with_nul() {
        let dto = SearchRequestDto {
            codebase_root: "/tmp/repo".to_string(),
            query: "hello\0world".to_string(),
            top_k: None,
            threshold: None,
            filter_expr: None,
            include_content: None,
        };

        let error = validate_search_request(&dto).err();
        assert!(
            matches!(error, Some(envelope) if envelope.code == ErrorCode::new("config", "invalid_field"))
        );
    }

    #[test]
    fn invalid_collection_name_is_delegated_to_domain() {
        let dto = IndexRequestDto {
            codebase_root: "/tmp/repo".to_string(),
            collection_name: Some("bad-name".to_string()),
            force_reindex: None,
        };

        let error = validate_index_request(&dto).err();
        assert!(
            matches!(error, Some(envelope) if envelope.code == ErrorCode::new("domain", "invalid_collection_name"))
        );
    }

    #[test]
    fn filter_expr_allowlist_accepts_simple_comparisons() -> Result<(), Box<dyn Error>> {
        validate_filter_expr_allowlist("relativePath == 'src/main.rs'")?;
        validate_filter_expr_allowlist("language != \"rust\"")?;
        validate_filter_expr_allowlist("fileExtension == 'rs'")?;
        Ok(())
    }
}
