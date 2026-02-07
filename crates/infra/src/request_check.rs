//! Request validation helpers for CLI surfaces.

use crate::InfraResult;
use semantic_code_config::{
    ValidatedClearIndexRequest, ValidatedIndexRequest, ValidatedReindexByChangeRequest,
    ValidatedSearchRequest, parse_clear_index_request_json, parse_index_request_json,
    parse_reindex_by_change_request_json, parse_search_request_json,
};
use std::fmt;

/// Supported request kinds for validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestKind {
    /// Index request.
    Index,
    /// Search request.
    Search,
    /// Reindex-by-change request.
    ReindexByChange,
    /// Clear-index request.
    ClearIndex,
}

/// Validated request payloads by kind.
#[derive(Debug)]
pub enum ValidatedRequest {
    /// Validated index request.
    Index(ValidatedIndexRequest),
    /// Validated search request.
    Search(ValidatedSearchRequest),
    /// Validated reindex-by-change request.
    ReindexByChange(ValidatedReindexByChangeRequest),
    /// Validated clear-index request.
    ClearIndex(ValidatedClearIndexRequest),
}

impl RequestKind {
    /// Canonical string representation (for CLI/UI).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Index => "index",
            Self::Search => "search",
            Self::ReindexByChange => "reindexByChange",
            Self::ClearIndex => "clearIndex",
        }
    }
}

impl fmt::Display for RequestKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Validate a request payload provided as JSON for the given kind.
pub fn validate_request_json(kind: RequestKind, input_json: &str) -> InfraResult<ValidatedRequest> {
    match kind {
        RequestKind::Index => {
            let request = parse_index_request_json(input_json)?;
            Ok(ValidatedRequest::Index(request))
        },
        RequestKind::Search => {
            let request = parse_search_request_json(input_json)?;
            Ok(ValidatedRequest::Search(request))
        },
        RequestKind::ReindexByChange => {
            let request = parse_reindex_by_change_request_json(input_json)?;
            Ok(ValidatedRequest::ReindexByChange(request))
        },
        RequestKind::ClearIndex => {
            let request = parse_clear_index_request_json(input_json)?;
            Ok(ValidatedRequest::ClearIndex(request))
        },
    }
}
