//! API v1 DTO validation helpers (shape and limit checks only).

use crate::v1::{
    ApiV1ClearIndexRequestDto, ApiV1IndexRequestDto, ApiV1ReindexByChangeRequestDto,
    ApiV1SearchRequestDto,
};
use semantic_code_shared::{Validate, ValidationError};
use std::fmt;

/// Validation failure details for API v1 DTOs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiV1ValidationIssue {
    /// Field name that failed validation.
    pub field: &'static str,
    /// Human-readable validation error message.
    pub message: Box<str>,
}

impl ApiV1ValidationIssue {
    fn new(field: &'static str, message: impl Into<Box<str>>) -> Self {
        Self {
            field,
            message: message.into(),
        }
    }
}

impl fmt::Display for ApiV1ValidationIssue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.field, self.message)
    }
}

impl std::error::Error for ApiV1ValidationIssue {}

impl ValidationError for ApiV1ValidationIssue {
    fn empty(field: &'static str) -> Self {
        Self::new(field, "value must be non-empty")
    }

    fn invalid(field: &'static str, reason: &'static str) -> Self {
        Self::new(field, reason)
    }

    fn out_of_range(field: &'static str, _value: String, min: String, max: String) -> Self {
        Self::new(field, format!("value must be between {min} and {max}"))
    }
}

/// Validate an index request DTO.
pub fn validate_index_request(dto: &ApiV1IndexRequestDto) -> Result<(), ApiV1ValidationIssue> {
    dto.validate()
}

/// Validate a search request DTO.
pub fn validate_search_request(dto: &ApiV1SearchRequestDto) -> Result<(), ApiV1ValidationIssue> {
    dto.validate()
}

/// Validate a reindex-by-change request DTO.
pub fn validate_reindex_by_change_request(
    dto: &ApiV1ReindexByChangeRequestDto,
) -> Result<(), ApiV1ValidationIssue> {
    dto.validate()
}

/// Validate a clear-index request DTO.
pub fn validate_clear_index_request(
    dto: &ApiV1ClearIndexRequestDto,
) -> Result<(), ApiV1ValidationIssue> {
    dto.validate()
}

pub fn validate_filter_expr_disabled(value: Option<&String>) -> Result<(), ApiV1ValidationIssue> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.trim().is_empty() {
        return Ok(());
    }
    Err(ApiV1ValidationIssue::new(
        "filterExpr",
        "filterExpr is disabled by default",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_empty_filter_expr() {
        let dto = ApiV1SearchRequestDto {
            codebase_root: "/tmp/repo".to_string(),
            query: "hello".to_string(),
            top_k: None,
            threshold: None,
            filter_expr: Some("relativePath == 'x.ts'".to_string()),
        };

        let result = validate_search_request(&dto);
        assert!(result.is_err());
    }
}
