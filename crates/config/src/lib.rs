//! # semantic-code-config
//!
//! Configuration schema, validation, and normalization logic for the CLI.
//! This crate depends on `domain` and `shared` only.

/// Environment variable parsing and merging.
pub mod env;
/// Config loading helpers (env + file + overrides).
pub mod load;
/// Request DTOs and validation.
pub mod requests;
/// JSON Schema exports for request DTOs.
pub mod requests_schema;
/// Configuration schema types and helpers.
pub mod schema;
/// Storage and persistence configuration.
pub mod storage;

pub use schema::{
    BackendConfig, CURRENT_CONFIG_VERSION, ConfigLimits, ConfigSchemaError,
    EmbeddingCacheDiskProvider, EmbeddingConfig, EmbeddingJobsConfig, EmbeddingRoutingConfig,
    EmbeddingRoutingMode, EmbeddingSplitConfig, OnnxEmbeddingConfig, ValidatedBackendConfig,
    VectorDbIndexConfig, VectorDbIndexSpec, parse_backend_config_json, parse_backend_config_toml,
};

pub use env::{BackendEnv, EnvParseError, apply_env_overrides};
pub use load::{
    load_backend_config_from_path, load_backend_config_from_sources, load_backend_config_std_env,
    to_pretty_json, to_pretty_toml,
};
pub use requests::{
    ClearIndexRequest, ClearIndexRequestDto, IndexRequest, IndexRequestDto, ReindexByChangeRequest,
    ReindexByChangeRequestDto, RequestValidationError, SearchRequest, SearchRequestDto,
    ValidatedClearIndexRequest, ValidatedIndexRequest, ValidatedReindexByChangeRequest,
    ValidatedSearchRequest, parse_clear_index_request_json, parse_index_request_json,
    parse_reindex_by_change_request_json, parse_search_request_json, validate_clear_index_request,
    validate_filter_expr_allowlist, validate_index_request, validate_reindex_by_change_request,
    validate_search_request,
};
pub use requests_schema::{
    clear_index_request_schema, index_request_schema, reindex_by_change_request_schema,
    search_request_schema,
};
pub use storage::SnapshotStorageMode;

/// Returns the config crate version.
#[must_use]
pub const fn config_crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_domain::domain_crate_version;
    use semantic_code_shared::shared_crate_version;

    #[test]
    fn config_crate_compiles() {
        let version = config_crate_version();
        assert!(!version.is_empty());
    }

    #[test]
    fn config_can_use_domain_and_shared() {
        let domain_version = domain_crate_version();
        let shared_version = shared_crate_version();

        assert!(!domain_version.is_empty());
        assert!(!shared_version.is_empty());
    }
}
