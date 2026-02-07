//! JSON Schema exports for request DTOs.

use crate::{ClearIndexRequestDto, IndexRequestDto, ReindexByChangeRequestDto, SearchRequestDto};
use schemars::schema::RootSchema;
use schemars::schema_for;

/// JSON Schema for `IndexRequestDto`.
#[must_use]
pub fn index_request_schema() -> RootSchema {
    schema_for!(IndexRequestDto)
}

/// JSON Schema for `SearchRequestDto`.
#[must_use]
pub fn search_request_schema() -> RootSchema {
    schema_for!(SearchRequestDto)
}

/// JSON Schema for `ReindexByChangeRequestDto`.
#[must_use]
pub fn reindex_by_change_request_schema() -> RootSchema {
    schema_for!(ReindexByChangeRequestDto)
}

/// JSON Schema for `ClearIndexRequestDto`.
#[must_use]
pub fn clear_index_request_schema() -> RootSchema {
    schema_for!(ClearIndexRequestDto)
}
