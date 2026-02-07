//! JSON Schema exports for API v1 request DTOs.

use crate::v1::{
    ApiV1ClearIndexRequestDto, ApiV1IndexRequestDto, ApiV1ReindexByChangeRequestDto,
    ApiV1SearchRequestDto,
};
use schemars::schema::RootSchema;
use schemars::schema_for;

/// JSON Schema for `ApiV1IndexRequestDto`.
#[must_use]
pub fn api_v1_index_request_schema() -> RootSchema {
    schema_for!(ApiV1IndexRequestDto)
}

/// JSON Schema for `ApiV1SearchRequestDto`.
#[must_use]
pub fn api_v1_search_request_schema() -> RootSchema {
    schema_for!(ApiV1SearchRequestDto)
}

/// JSON Schema for `ApiV1ReindexByChangeRequestDto`.
#[must_use]
pub fn api_v1_reindex_by_change_request_schema() -> RootSchema {
    schema_for!(ApiV1ReindexByChangeRequestDto)
}

/// JSON Schema for `ApiV1ClearIndexRequestDto`.
#[must_use]
pub fn api_v1_clear_index_request_schema() -> RootSchema {
    schema_for!(ApiV1ClearIndexRequestDto)
}
