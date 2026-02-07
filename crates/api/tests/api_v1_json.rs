//! API v1 fixture parity tests.

use semantic_code_api::v1::{
    ApiV1ClearIndexRequestDto, ApiV1ClearIndexResponseDto, ApiV1ErrorDto, ApiV1IndexRequestDto,
    ApiV1IndexResponseDto, ApiV1IndexStatus, ApiV1ReindexByChangeRequestDto,
    ApiV1ReindexByChangeResponseDto, ApiV1Result, ApiV1SearchRequestDto, ApiV1SearchResponseDto,
    ApiV1SearchResultDto, error_envelope_to_api_v1_error,
};
use semantic_code_shared::{ErrorCode, ErrorEnvelope, Validate};
use semantic_code_testkit::parity::api_v1_json_fixtures;
use std::error::Error;

#[test]
fn api_v1_json_matches_fixtures() -> Result<(), Box<dyn Error>> {
    let fixtures = api_v1_json_fixtures()?;

    let error_envelope = ErrorEnvelope::expected(
        ErrorCode::new("domain", "invalid_collection_name"),
        "CollectionName must match /^[a-zA-Z][a-zA-Z0-9_]*$/",
    )
    .with_metadata("input", "bad-name")
    .with_metadata("token", "secret-token")
    .with_metadata("query", "hello world");
    let error_dto: ApiV1ErrorDto = error_envelope_to_api_v1_error(&error_envelope, None);
    assert_eq!(serde_json::to_value(&error_dto)?, fixtures.error_dto);

    let ok_result = ApiV1Result::ok(ApiV1IndexResponseDto {
        indexed_files: 3,
        total_chunks: 7,
        status: ApiV1IndexStatus::Completed,
    });
    assert_eq!(serde_json::to_value(&ok_result)?, fixtures.ok_result);

    let err_result: ApiV1Result<ApiV1IndexResponseDto> = ApiV1Result::err(error_dto.clone());
    assert_eq!(serde_json::to_value(&err_result)?, fixtures.error_result);

    let index_request = ApiV1IndexRequestDto {
        codebase_root: "/tmp/repo".to_string(),
        collection_name: Some("code_chunks_123".to_string()),
        force_reindex: Some(true),
    };
    assert!(index_request.validate().is_ok());
    assert_eq!(
        serde_json::to_value(&index_request)?,
        fixtures.index_request
    );

    let index_response = ApiV1IndexResponseDto {
        indexed_files: 3,
        total_chunks: 7,
        status: ApiV1IndexStatus::Completed,
    };
    assert_eq!(
        serde_json::to_value(&index_response)?,
        fixtures.index_response
    );

    let search_request = ApiV1SearchRequestDto {
        codebase_root: "/tmp/repo".to_string(),
        query: "hello".to_string(),
        top_k: Some(5),
        threshold: Some(0.42),
        filter_expr: Some("".to_string()),
    };
    assert_eq!(
        serde_json::to_value(&search_request)?,
        fixtures.search_request
    );

    let search_result = ApiV1SearchResultDto {
        content: "fn main() {}".to_string(),
        relative_path: "src/main.rs".to_string(),
        start_line: 1,
        end_line: 1,
        language: "rust".to_string(),
        score: 0.88,
    };
    assert_eq!(
        serde_json::to_value(&search_result)?,
        fixtures.search_result
    );

    let search_response = ApiV1SearchResponseDto {
        results: vec![search_result],
    };
    assert_eq!(
        serde_json::to_value(&search_response)?,
        fixtures.search_response
    );

    let reindex_request = ApiV1ReindexByChangeRequestDto {
        codebase_root: "/tmp/repo".to_string(),
    };
    assert_eq!(
        serde_json::to_value(&reindex_request)?,
        fixtures.reindex_by_change_request
    );

    let reindex_response = ApiV1ReindexByChangeResponseDto {
        added: 1,
        removed: 2,
        modified: 3,
    };
    assert_eq!(
        serde_json::to_value(&reindex_response)?,
        fixtures.reindex_by_change_response
    );

    let clear_request = ApiV1ClearIndexRequestDto {
        codebase_root: "/tmp/repo".to_string(),
    };
    assert_eq!(
        serde_json::to_value(&clear_request)?,
        fixtures.clear_index_request
    );

    let clear_response = ApiV1ClearIndexResponseDto { cleared: true };
    assert_eq!(
        serde_json::to_value(&clear_response)?,
        fixtures.clear_index_response
    );

    Ok(())
}
