//! Request validation helpers for CLI surfaces.

use crate::InfraResult;
use semantic_code_config::{
    ClearIndexRequestDto, IndexRequestDto, ReindexByChangeRequestDto, SearchRequestDto,
    validate_clear_index_request, validate_index_request, validate_reindex_by_change_request,
    validate_search_request,
};
use semantic_code_shared::{ErrorCode, ErrorEnvelope};
use serde::de::DeserializeOwned;
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
pub fn validate_request_json(kind: RequestKind, input_json: &str) -> InfraResult<()> {
    match kind {
        RequestKind::Index => {
            let dto: IndexRequestDto = parse_request_json("index", input_json)?;
            let _ = validate_index_request(&dto)?;
        },
        RequestKind::Search => {
            let dto: SearchRequestDto = parse_request_json("search", input_json)?;
            let _ = validate_search_request(&dto)?;
        },
        RequestKind::ReindexByChange => {
            let dto: ReindexByChangeRequestDto = parse_request_json("reindexByChange", input_json)?;
            let _ = validate_reindex_by_change_request(&dto)?;
        },
        RequestKind::ClearIndex => {
            let dto: ClearIndexRequestDto = parse_request_json("clearIndex", input_json)?;
            let _ = validate_clear_index_request(&dto)?;
        },
    }

    Ok(())
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
