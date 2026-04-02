//! # semantic-code-config
//!
//! Configuration schema, validation, and normalization logic for the CLI.
//! This crate depends on `domain` and `shared` only.

/// Environment variable parsing and merging.
mod env;
/// Config loading helpers (env + file + overrides).
mod load;
/// Request DTOs and validation.
mod requests;
/// Runtime env projection used by infra/facade composition.
mod runtime;
/// Configuration schema types and helpers.
mod schema;
/// Storage and persistence configuration.
mod storage;

pub use schema::{
    BackendConfig, ConfigSchemaError, DfrrBq1Threshold, DfrrBq1ThresholdMode, DfrrQueryStrategy,
    DfrrSearchConfig, EmbeddingCacheDiskProvider, EmbeddingConfig, EmbeddingRoutingMode,
    HnswSearchConfig, ValidatedBackendConfig, VectorKernelKind, VectorSearchStrategy,
};

pub use load::{
    load_backend_config_from_path, load_backend_config_from_sources, load_backend_config_std_env,
    to_pretty_json, to_pretty_toml,
};
pub use requests::{
    ClearIndexRequestDto, IndexRequestDto, ReindexByChangeRequestDto, SearchRequestDto,
    ValidatedClearIndexRequest, ValidatedIndexRequest, ValidatedReindexByChangeRequest,
    ValidatedSearchRequest, validate_clear_index_request, validate_index_request,
    validate_reindex_by_change_request, validate_search_request,
};
pub use runtime::{RuntimeEnv, load_runtime_env_from_map, load_runtime_env_std_env};
pub use storage::{SnapshotStorageMode, VectorSnapshotFormat};

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
