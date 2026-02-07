//! # semantic-code-facade
//!
//! Facade API for consumers (CLI and future services).
//! This crate depends on `infra`, `api`, and `app`.

pub use semantic_code_app::{IndexCodebaseOutput, IndexCodebaseStatus, ReindexByChangeOutput};
pub use semantic_code_domain::SearchResult;
use std::collections::BTreeMap;
use std::path::Path;

/// Placeholder module for the facade layer.
pub mod placeholder {
    /// Placeholder function to verify the crate compiles.
    #[must_use]
    pub const fn facade_crate_version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}

pub use placeholder::facade_crate_version;

/// Validate that the provided env overrides can be parsed and merged into a config.
pub fn validate_env_parsing(env: &BTreeMap<String, String>) -> Result<(), InfraError> {
    semantic_code_infra::validate_env_parsing(env)
}

/// Request kind used for validation.
pub use semantic_code_infra::{RequestKind, ValidatedRequest};

/// Infra error type (shared error envelope).
pub use semantic_code_infra::InfraError;

/// Local CLI config summary.
pub use semantic_code_infra::CliConfigSummary;
/// Local CLI init status summary.
pub use semantic_code_infra::CliInitStatus;
/// Local CLI status summary.
pub use semantic_code_infra::CliStatus;
/// Local CLI snapshot status.
pub use semantic_code_infra::SnapshotStatus;
/// Background job types and helpers.
pub use semantic_code_infra::{
    JobError, JobKind, JobProgress, JobRequest, JobResult, JobState, JobStatus, cancel_job,
    create_job, read_job_status, run_job,
};
/// Re-export redaction utilities for CLI boundary sanitization.
pub use semantic_code_infra::{is_secret_key, redact_if_secret};

/// Validate a request payload (JSON) for the given request kind.
pub fn validate_request_json(
    kind: RequestKind,
    input_json: &str,
) -> Result<ValidatedRequest, InfraError> {
    semantic_code_infra::validate_request_json(kind, input_json)
}

/// Load and validate the effective config, returning deterministic pretty JSON.
pub fn load_effective_config_json(
    env: &BTreeMap<String, String>,
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
) -> Result<String, InfraError> {
    semantic_code_infra::load_effective_config_json(env, config_path, overrides_json)
}

/// Run the in-memory index smoke test.
pub fn run_index_smoke() -> Result<(), InfraError> {
    semantic_code_infra::run_index_smoke()
}

/// Run the in-memory search smoke test.
pub fn run_search_smoke() -> Result<(), InfraError> {
    semantic_code_infra::run_search_smoke()
}

/// Run the in-memory clear smoke test.
pub fn run_clear_smoke() -> Result<(), InfraError> {
    semantic_code_infra::run_clear_smoke()
}

/// Run a local index operation.
pub fn run_index_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    request: &semantic_code_config::ValidatedIndexRequest,
    init_if_missing: bool,
) -> Result<IndexCodebaseOutput, InfraError> {
    semantic_code_infra::run_index_local(config_path, overrides_json, request, init_if_missing)
}

/// Run a local semantic search.
pub fn run_search_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    request: &semantic_code_config::ValidatedSearchRequest,
) -> Result<Vec<SearchResult>, InfraError> {
    semantic_code_infra::run_search_local(config_path, overrides_json, request)
}

/// Clear the local index and snapshot.
pub fn run_clear_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    request: &semantic_code_config::ValidatedClearIndexRequest,
) -> Result<(), InfraError> {
    semantic_code_infra::run_clear_local(config_path, overrides_json, request)
}

/// Run a local reindex-by-change operation.
pub fn run_reindex_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    request: &semantic_code_config::ValidatedReindexByChangeRequest,
) -> Result<ReindexByChangeOutput, InfraError> {
    semantic_code_infra::run_reindex_local(config_path, overrides_json, request)
}

/// Initialize config and manifest for a codebase.
pub fn run_init_local(
    config_path: Option<&Path>,
    codebase_root: &Path,
    storage_mode: Option<semantic_code_config::SnapshotStorageMode>,
    force: bool,
) -> Result<CliInitStatus, InfraError> {
    semantic_code_infra::run_init_local(config_path, codebase_root, storage_mode, force)
}

/// Read local CLI status information.
pub fn read_status_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    codebase_root: &Path,
) -> Result<CliStatus, InfraError> {
    semantic_code_infra::read_status_local(config_path, overrides_json, codebase_root)
}

/// API v1 error payload type (re-exported for CLI formatting).
pub use semantic_code_api::v1::{ApiV1ErrorDto, ApiV1ErrorKind};

/// Convert an infra error into an API v1 error payload (stable code + meta).
pub fn infra_error_to_api_v1(error: &InfraError) -> ApiV1ErrorDto {
    semantic_code_api::v1::error_envelope_to_api_v1_error(error, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_api::api_crate_version;
    use semantic_code_app::app_crate_version;
    use semantic_code_infra::infra_crate_version;

    #[test]
    fn facade_crate_compiles() {
        let version = facade_crate_version();
        assert!(!version.is_empty());
    }

    #[test]
    fn facade_can_use_infra_api_app() {
        let infra_version = infra_crate_version();
        let api_version = api_crate_version();
        let app_version_value = app_crate_version();

        assert!(!infra_version.is_empty());
        assert!(!api_version.is_empty());
        assert!(!app_version_value.is_empty());
    }
}
