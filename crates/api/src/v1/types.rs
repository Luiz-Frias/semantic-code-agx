//! API v1 DTO types.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Error kind exposed in API v1 responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ApiV1ErrorKind {
    /// Expected, user-facing errors (validation, cancellation).
    Expected,
    /// Invariant violations that indicate a bug.
    Invariant,
}

/// API v1 error code string (stable contract value).
pub type ApiV1ErrorCode = String;

/// Metadata map attached to API v1 errors.
pub type ApiV1ErrorMeta = BTreeMap<String, String>;

/// API v1 error payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiV1ErrorDto {
    /// Stable error code (e.g. `ERR_DOMAIN_INVALID_COLLECTION_NAME`).
    pub code: ApiV1ErrorCode,
    /// Human-readable message for the caller.
    pub message: String,
    /// Error category.
    pub kind: ApiV1ErrorKind,
    /// Optional metadata for debugging and correlation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<ApiV1ErrorMeta>,
}

/// API v1 result wrapper for success or failure payloads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ApiV1Result<T> {
    /// Success response.
    Ok {
        /// Indicates success.
        ok: bool,
        /// Success payload.
        data: T,
    },
    /// Error response.
    Err {
        /// Indicates failure.
        ok: bool,
        /// Error payload.
        error: ApiV1ErrorDto,
    },
}

impl<T> ApiV1Result<T> {
    /// Build a success response wrapper.
    #[must_use]
    pub const fn ok(data: T) -> Self {
        Self::Ok { ok: true, data }
    }

    /// Build an error response wrapper.
    #[must_use]
    pub const fn err(error: ApiV1ErrorDto) -> Self {
        Self::Err { ok: false, error }
    }
}

/// API v1 index request payload.
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
#[serde(rename_all = "camelCase")]
#[validate(error = "crate::v1::validation::ApiV1ValidationIssue")]
pub struct ApiV1IndexRequestDto {
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

/// Indexing status result for API responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiV1IndexStatus {
    /// Index completed successfully.
    Completed,
    /// Index stopped due to limits.
    LimitReached,
}

/// API v1 index response payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiV1IndexResponseDto {
    /// Number of files indexed.
    pub indexed_files: u64,
    /// Total chunks created.
    pub total_chunks: u64,
    /// Completion status.
    pub status: ApiV1IndexStatus,
}

/// API v1 search request payload.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Serialize,
    Deserialize,
    JsonSchema,
    semantic_code_validate_derive::Validate,
)]
#[serde(rename_all = "camelCase")]
#[validate(error = "crate::v1::validation::ApiV1ValidationIssue")]
pub struct ApiV1SearchRequestDto {
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
    #[validate(
        field = "filterExpr",
        custom = "crate::v1::validation::validate_filter_expr_disabled"
    )]
    pub filter_expr: Option<String>,
}

/// API v1 search result payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiV1SearchResultDto {
    /// Matching chunk content.
    pub content: String,
    /// Relative path of the matching file.
    pub relative_path: String,
    /// Starting line of the match (1-indexed).
    pub start_line: u32,
    /// Ending line of the match (1-indexed).
    pub end_line: u32,
    /// Language identifier.
    pub language: String,
    /// Similarity score.
    pub score: f64,
}

/// API v1 search response payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiV1SearchResponseDto {
    /// Search results ordered by relevance.
    pub results: Vec<ApiV1SearchResultDto>,
}

/// API v1 reindex-by-change request payload.
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
#[serde(rename_all = "camelCase")]
#[validate(error = "crate::v1::validation::ApiV1ValidationIssue")]
pub struct ApiV1ReindexByChangeRequestDto {
    /// Root path of the codebase to reindex.
    #[validate(field = "codebaseRoot", non_empty)]
    pub codebase_root: String,
}

/// API v1 reindex-by-change response payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiV1ReindexByChangeResponseDto {
    /// Count of added files.
    pub added: u64,
    /// Count of removed files.
    pub removed: u64,
    /// Count of modified files.
    pub modified: u64,
}

/// API v1 clear-index request payload.
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
#[serde(rename_all = "camelCase")]
#[validate(error = "crate::v1::validation::ApiV1ValidationIssue")]
pub struct ApiV1ClearIndexRequestDto {
    /// Root path of the codebase to clear.
    #[validate(field = "codebaseRoot", non_empty)]
    pub codebase_root: String,
}

/// API v1 clear-index response payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiV1ClearIndexResponseDto {
    /// Whether the clear operation succeeded.
    pub cleared: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn dto_roundtrip_json() -> Result<(), Box<dyn Error>> {
        let error = ApiV1ErrorDto {
            code: "ERR_DOMAIN_INVALID_COLLECTION_NAME".to_string(),
            message: "CollectionName must match /^[a-zA-Z][a-zA-Z0-9_]*$/".to_string(),
            kind: ApiV1ErrorKind::Expected,
            meta: Some(BTreeMap::from([(
                "input".to_string(),
                "bad-name".to_string(),
            )])),
        };
        let error_json = serde_json::to_string(&error)?;
        let parsed: ApiV1ErrorDto = serde_json::from_str(&error_json)?;
        assert_eq!(parsed, error);

        let index_request = ApiV1IndexRequestDto {
            codebase_root: "/tmp/repo".to_string(),
            collection_name: Some("code_chunks_123".to_string()),
            force_reindex: Some(true),
        };
        let index_json = serde_json::to_string(&index_request)?;
        let parsed: ApiV1IndexRequestDto = serde_json::from_str(&index_json)?;
        assert_eq!(parsed, index_request);

        let search_request = ApiV1SearchRequestDto {
            codebase_root: "/tmp/repo".to_string(),
            query: "hello".to_string(),
            top_k: Some(5),
            threshold: Some(0.42),
            filter_expr: Some("".to_string()),
        };
        let search_json = serde_json::to_string(&search_request)?;
        let parsed: ApiV1SearchRequestDto = serde_json::from_str(&search_json)?;
        assert_eq!(parsed, search_request);

        let search_result = ApiV1SearchResultDto {
            content: "fn main() {}".to_string(),
            relative_path: "src/main.rs".to_string(),
            start_line: 1,
            end_line: 1,
            language: "rust".to_string(),
            score: 0.88,
        };
        let search_response = ApiV1SearchResponseDto {
            results: vec![search_result],
        };
        let search_json = serde_json::to_string(&search_response)?;
        let parsed: ApiV1SearchResponseDto = serde_json::from_str(&search_json)?;
        assert_eq!(parsed, search_response);

        let reindex_request = ApiV1ReindexByChangeRequestDto {
            codebase_root: "/tmp/repo".to_string(),
        };
        let reindex_json = serde_json::to_string(&reindex_request)?;
        let parsed: ApiV1ReindexByChangeRequestDto = serde_json::from_str(&reindex_json)?;
        assert_eq!(parsed, reindex_request);

        let reindex_response = ApiV1ReindexByChangeResponseDto {
            added: 1,
            removed: 2,
            modified: 3,
        };
        let reindex_json = serde_json::to_string(&reindex_response)?;
        let parsed: ApiV1ReindexByChangeResponseDto = serde_json::from_str(&reindex_json)?;
        assert_eq!(parsed, reindex_response);

        let clear_request = ApiV1ClearIndexRequestDto {
            codebase_root: "/tmp/repo".to_string(),
        };
        let clear_json = serde_json::to_string(&clear_request)?;
        let parsed: ApiV1ClearIndexRequestDto = serde_json::from_str(&clear_json)?;
        assert_eq!(parsed, clear_request);

        let clear_response = ApiV1ClearIndexResponseDto { cleared: true };
        let clear_json = serde_json::to_string(&clear_response)?;
        let parsed: ApiV1ClearIndexResponseDto = serde_json::from_str(&clear_json)?;
        assert_eq!(parsed, clear_response);

        let ok_result = ApiV1Result::ok(ApiV1IndexResponseDto {
            indexed_files: 3,
            total_chunks: 7,
            status: ApiV1IndexStatus::Completed,
        });
        let ok_json = serde_json::to_string(&ok_result)?;
        let parsed: ApiV1Result<ApiV1IndexResponseDto> = serde_json::from_str(&ok_json)?;
        assert_eq!(parsed, ok_result);

        let err_result = ApiV1Result::err(error);
        let err_json = serde_json::to_string(&err_result)?;
        let parsed: ApiV1Result<ApiV1IndexResponseDto> = serde_json::from_str(&err_json)?;
        assert_eq!(parsed, err_result);

        Ok(())
    }
}
