//! API v1 DTOs and helpers.

mod mappers;
mod schema;
mod types;
mod validation;

pub use mappers::{error_code_to_api_v1, error_envelope_to_api_v1_error, result_to_api_v1_result};
pub use schema::{
    api_v1_clear_index_request_schema, api_v1_index_request_schema,
    api_v1_reindex_by_change_request_schema, api_v1_search_request_schema,
};
pub use types::*;
pub use validation::{
    ApiV1ValidationIssue, validate_clear_index_request, validate_index_request,
    validate_reindex_by_change_request, validate_search_request,
};
