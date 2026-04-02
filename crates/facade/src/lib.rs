//! # semantic-code-facade
//!
//! Facade API for consumers (CLI and future services).
//! This crate depends on `infra`, `api`, and `app`.

use std::collections::BTreeMap;
use std::path::Path;
use tracing::instrument;

mod types;
pub use types::{
    ApiV1ErrorDto, ApiV1ErrorKind, BuildInfo, ClearIndexRequest, CliConfigSummary, CliInitStatus,
    CliManifestStatus, CliStatus, CliStorageEstimate, IndexCodebaseOutput, IndexCodebaseStatus,
    IndexEmbedStats, IndexInsertStats, IndexRequest, IndexScanStats, IndexSplitStats,
    IndexStageStats, InfraError, JobEmbedStats, JobError, JobInsertStats, JobKind, JobProgress,
    JobRequest, JobResult, JobScanStats, JobSplitStats, JobStageStats, JobState, JobStatus,
    ReindexByChangeOutput, ReindexByChangeRequest, RequestKind, SearchOutput, SearchRequest,
    SearchResult, SearchStats, SnapshotStatus, SnapshotStorageMode, StorageThresholdStatus,
};

/// Placeholder module for the facade layer.
mod placeholder {
    /// Placeholder function to verify the crate compiles.
    #[must_use]
    pub const fn facade_crate_version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}

pub use placeholder::facade_crate_version;
pub use semantic_code_domain::{
    CalibrationParamError, CalibrationParams, CalibrationPrecision, CalibrationQueryCount,
    CalibrationState, CalibrationTopK, TargetRecall,
};

/// Returns build metadata for the current binary.
#[must_use]
pub const fn build_info() -> BuildInfo {
    let build = semantic_code_core::build_info();
    BuildInfo {
        name: build.name,
        version: build.version,
        rustc_version: build.rustc_version,
        target: build.target,
        profile: build.profile,
        git_hash: build.git_hash,
        git_dirty: build.git_dirty,
    }
}

/// Validate that the provided env overrides can be parsed and merged into a config.
#[instrument(name = "facade.validate_env_parsing", skip_all, fields(env_size = env.len()))]
pub fn validate_env_parsing(env: &BTreeMap<String, String>) -> Result<(), InfraError> {
    semantic_code_infra::validate_env_parsing(env).map_err(Into::into)
}

/// Validate an index request from CLI primitives.
#[instrument(name = "facade.validate_index_request_for_root", skip_all)]
pub fn validate_index_request_for_root(codebase_root: &Path) -> Result<IndexRequest, InfraError> {
    let request = semantic_code_config::IndexRequestDto {
        codebase_root: codebase_root.to_string_lossy().to_string(),
        collection_name: None,
        force_reindex: None,
    };
    semantic_code_config::validate_index_request(&request)
        .map(Into::into)
        .map_err(Into::into)
}

/// Validate a search request from CLI primitives.
#[instrument(name = "facade.validate_search_request_for_query", skip_all)]
pub fn validate_search_request_for_query(
    codebase_root: &Path,
    query: &str,
    top_k: Option<u32>,
    threshold: Option<f32>,
    filter_expr: Option<&str>,
    include_content: bool,
) -> Result<SearchRequest, InfraError> {
    let request = semantic_code_config::SearchRequestDto {
        codebase_root: codebase_root.to_string_lossy().to_string(),
        query: query.to_string(),
        top_k,
        threshold: threshold.map(f64::from),
        filter_expr: filter_expr.map(str::to_owned),
        include_content: include_content.then_some(true),
    };
    semantic_code_config::validate_search_request(&request)
        .map(Into::into)
        .map_err(Into::into)
}

/// Validate a reindex-by-change request from CLI primitives.
#[instrument(name = "facade.validate_reindex_request_for_root", skip_all)]
pub fn validate_reindex_request_for_root(
    codebase_root: &Path,
) -> Result<ReindexByChangeRequest, InfraError> {
    let request = semantic_code_config::ReindexByChangeRequestDto {
        codebase_root: codebase_root.to_string_lossy().to_string(),
    };
    semantic_code_config::validate_reindex_by_change_request(&request)
        .map(Into::into)
        .map_err(Into::into)
}

/// Validate a clear-index request from CLI primitives.
#[instrument(name = "facade.validate_clear_request_for_root", skip_all)]
pub fn validate_clear_request_for_root(
    codebase_root: &Path,
) -> Result<ClearIndexRequest, InfraError> {
    let request = semantic_code_config::ClearIndexRequestDto {
        codebase_root: codebase_root.to_string_lossy().to_string(),
    };
    semantic_code_config::validate_clear_index_request(&request)
        .map(Into::into)
        .map_err(Into::into)
}

/// Effective vector-kernel kind used by CLI output payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliVectorKernelKind {
    /// Built-in HNSW kernel.
    HnswRs,
    /// Experimental DFRR kernel.
    Dfrr,
    /// Brute-force exact nearest-neighbor scan for benchmark ground truth.
    FlatScan,
}

impl CliVectorKernelKind {
    /// Effective kernel label (`hnsw-rs`, `dfrr`, `flat-scan`).
    #[must_use]
    pub const fn effective_label(self) -> &'static str {
        match self {
            Self::HnswRs => "hnsw-rs",
            Self::Dfrr => "dfrr",
            Self::FlatScan => "flat-scan",
        }
    }

    /// Returns `true` when the effective kernel is experimental.
    #[must_use]
    pub const fn is_experimental(self) -> bool {
        matches!(self, Self::Dfrr | Self::FlatScan)
    }
}

/// Resolve effective kernel using process environment.
#[instrument(
    name = "facade.resolve_vector_kernel_kind_std_env",
    skip_all,
    fields(
        has_config_path = config_path.is_some(),
        has_overrides_json = overrides_json.is_some()
    )
)]
pub fn resolve_vector_kernel_kind_std_env(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
) -> Result<CliVectorKernelKind, InfraError> {
    let config = semantic_code_config::load_backend_config_std_env(config_path, overrides_json)?;
    Ok(map_vector_kernel_kind(
        config.vector_db.effective_vector_kernel(),
    ))
}

/// Resolve effective kernel using explicit environment map.
#[instrument(
    name = "facade.resolve_vector_kernel_kind_from_env",
    skip_all,
    fields(env_size = env.len())
)]
pub fn resolve_vector_kernel_kind_from_env(
    env: &BTreeMap<String, String>,
) -> Result<CliVectorKernelKind, InfraError> {
    let config = semantic_code_config::load_backend_config_from_path(None, None, env)?;
    Ok(map_vector_kernel_kind(
        config.vector_db.effective_vector_kernel(),
    ))
}

const fn map_vector_kernel_kind(
    kind: semantic_code_config::VectorKernelKind,
) -> CliVectorKernelKind {
    match kind {
        semantic_code_config::VectorKernelKind::HnswRs => CliVectorKernelKind::HnswRs,
        semantic_code_config::VectorKernelKind::Dfrr => CliVectorKernelKind::Dfrr,
        semantic_code_config::VectorKernelKind::FlatScan => CliVectorKernelKind::FlatScan,
    }
}

/// Validate a request payload (JSON) for the given request kind.
#[instrument(
    name = "facade.validate_request_json",
    skip_all,
    fields(input_bytes = input_json.len())
)]
pub fn validate_request_json(kind: RequestKind, input_json: &str) -> Result<(), InfraError> {
    semantic_code_infra::validate_request_json(kind.into(), input_json).map_err(Into::into)
}

/// Load and validate the effective config, returning deterministic pretty JSON.
#[instrument(
    name = "facade.load_effective_config_json",
    skip_all,
    fields(
        env_size = env.len(),
        has_config_path = config_path.is_some(),
        has_overrides_json = overrides_json.is_some()
    )
)]
pub fn load_effective_config_json(
    env: &BTreeMap<String, String>,
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
) -> Result<String, InfraError> {
    semantic_code_infra::load_effective_config_json(env, config_path, overrides_json)
        .map_err(Into::into)
}

/// Run the in-memory index smoke test.
#[instrument(name = "facade.run_index_smoke", skip_all)]
pub fn run_index_smoke() -> Result<(), InfraError> {
    semantic_code_infra::run_index_smoke().map_err(Into::into)
}

/// Run the in-memory search smoke test.
#[instrument(name = "facade.run_search_smoke", skip_all)]
pub fn run_search_smoke() -> Result<(), InfraError> {
    semantic_code_infra::run_search_smoke().map_err(Into::into)
}

/// Run the in-memory clear smoke test.
#[instrument(name = "facade.run_clear_smoke", skip_all)]
pub fn run_clear_smoke() -> Result<(), InfraError> {
    semantic_code_infra::run_clear_smoke().map_err(Into::into)
}

/// Run a local index operation.
#[instrument(
    name = "facade.run_index_local",
    skip_all,
    fields(
        has_config_path = config_path.is_some(),
        has_overrides_json = overrides_json.is_some(),
        init_if_missing
    )
)]
pub fn run_index_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    request: &IndexRequest,
    init_if_missing: bool,
) -> Result<IndexCodebaseOutput, InfraError> {
    semantic_code_infra::run_index_local(
        config_path,
        overrides_json,
        request.as_validated(),
        init_if_missing,
    )
    .map(Into::into)
    .map_err(Into::into)
}

/// Run a local semantic search.
#[instrument(
    name = "facade.run_search_local",
    skip_all,
    fields(
        has_config_path = config_path.is_some(),
        has_overrides_json = overrides_json.is_some()
    )
)]
pub fn run_search_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    request: &SearchRequest,
) -> Result<SearchOutput, InfraError> {
    semantic_code_infra::run_search_local(config_path, overrides_json, request.as_validated())
        .map(Into::into)
        .map_err(Into::into)
}

/// A pre-warmed search session for running multiple queries without re-loading
/// the index. Created via [`open_search_session`].
pub struct SearchSession(semantic_code_infra::LocalSearchSession);

impl SearchSession {
    /// Execute a single search against the warm session.
    pub fn search(
        &self,
        query: &str,
        top_k: Option<u32>,
        threshold: Option<f32>,
    ) -> Result<SearchOutput, InfraError> {
        self.0
            .search(query, top_k, threshold)
            .map(Into::into)
            .map_err(Into::into)
    }

    /// Embed a query string and return the raw vector as `Vec<f32>`.
    pub fn embed(&self, query: &str) -> Result<Vec<f32>, InfraError> {
        self.0
            .embed(query)
            .map(|v| v.as_slice().to_vec())
            .map_err(Into::into)
    }

    /// Execute a search using a pre-computed query embedding vector.
    pub fn search_with_vector(
        &self,
        query_label: &str,
        vector: Vec<f32>,
        top_k: Option<u32>,
        threshold: Option<f32>,
    ) -> Result<SearchOutput, InfraError> {
        let ev = semantic_code_infra::EmbeddingVector::from_vec(vector);
        self.0
            .search_with_vector(query_label, ev, top_k, threshold)
            .map(Into::into)
            .map_err(Into::into)
    }
}

/// Open a warm search session that pre-loads config, embedding, and vectordb.
///
/// Use this for `--stdin-batch` mode where a single process handles many queries.
pub fn open_search_session(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    codebase_root: &Path,
) -> Result<SearchSession, InfraError> {
    open_search_session_with_options(config_path, overrides_json, codebase_root, false)
}

/// Open a warm search session with optional deferred embedding initialization.
///
/// When `query_vectors_only` is `true`, the session skips embedding provider
/// initialization and requires callers to use pre-computed query vectors.
pub fn open_search_session_with_options(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    codebase_root: &Path,
    query_vectors_only: bool,
) -> Result<SearchSession, InfraError> {
    semantic_code_infra::open_search_session_with_options(
        config_path,
        overrides_json,
        codebase_root,
        query_vectors_only,
    )
    .map(SearchSession)
    .map_err(Into::into)
}

/// Clear the local index and snapshot.
#[instrument(
    name = "facade.run_clear_local",
    skip_all,
    fields(
        has_config_path = config_path.is_some(),
        has_overrides_json = overrides_json.is_some()
    )
)]
pub fn run_clear_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    request: &ClearIndexRequest,
) -> Result<(), InfraError> {
    semantic_code_infra::run_clear_local(config_path, overrides_json, request.as_validated())
        .map_err(Into::into)
}

/// Run a local reindex-by-change operation.
#[instrument(
    name = "facade.run_reindex_local",
    skip_all,
    fields(
        has_config_path = config_path.is_some(),
        has_overrides_json = overrides_json.is_some()
    )
)]
pub fn run_reindex_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    request: &ReindexByChangeRequest,
) -> Result<ReindexByChangeOutput, InfraError> {
    semantic_code_infra::run_reindex_local(config_path, overrides_json, request.as_validated())
        .map(Into::into)
        .map_err(Into::into)
}

/// Create a new background job request and persist initial status.
#[instrument(name = "facade.create_job", skip_all, fields(job_id = request.id.as_ref()))]
pub fn create_job(request: &JobRequest) -> Result<JobStatus, InfraError> {
    let request: semantic_code_infra::JobRequest = request.clone().into();
    semantic_code_infra::create_job(&request)
        .map(Into::into)
        .map_err(Into::into)
}

/// Read job status from disk.
#[instrument(name = "facade.read_job_status", skip_all, fields(job_id = job_id))]
pub fn read_job_status(root: &Path, job_id: &str) -> Result<JobStatus, InfraError> {
    semantic_code_infra::read_job_status(root, job_id)
        .map(Into::into)
        .map_err(Into::into)
}

/// Mark a background job as cancelled.
#[instrument(name = "facade.cancel_job", skip_all, fields(job_id = job_id))]
pub fn cancel_job(root: &Path, job_id: &str) -> Result<JobStatus, InfraError> {
    semantic_code_infra::cancel_job(root, job_id)
        .map(Into::into)
        .map_err(Into::into)
}

/// Run a queued background job synchronously in the worker process.
#[instrument(name = "facade.run_job", skip_all, fields(job_id = job_id))]
pub fn run_job(root: &Path, job_id: &str) -> Result<JobStatus, InfraError> {
    semantic_code_infra::run_job(root, job_id)
        .map(Into::into)
        .map_err(Into::into)
}

/// Initialize config and manifest for a codebase.
#[instrument(
    name = "facade.run_init_local",
    skip_all,
    fields(
        has_config_path = config_path.is_some(),
        has_storage_mode = storage_mode.is_some(),
        force
    )
)]
pub fn run_init_local(
    config_path: Option<&Path>,
    codebase_root: &Path,
    storage_mode: Option<SnapshotStorageMode>,
    force: bool,
) -> Result<CliInitStatus, InfraError> {
    semantic_code_infra::run_init_local(
        config_path,
        codebase_root,
        storage_mode.map(Into::into),
        force,
    )
    .map(Into::into)
    .map_err(Into::into)
}

/// Read local CLI status information.
#[instrument(
    name = "facade.read_status_local",
    skip_all,
    fields(
        has_config_path = config_path.is_some(),
        has_overrides_json = overrides_json.is_some()
    )
)]
pub fn read_status_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    codebase_root: &Path,
) -> Result<CliStatus, InfraError> {
    semantic_code_infra::read_status_local(config_path, overrides_json, codebase_root)
        .map(Into::into)
        .map_err(Into::into)
}

/// Estimate local storage requirements for indexing.
#[instrument(
    name = "facade.estimate_storage_local",
    skip_all,
    fields(
        has_config_path = config_path.is_some(),
        has_overrides_json = overrides_json.is_some(),
        danger_close_storage
    )
)]
pub fn estimate_storage_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    codebase_root: &Path,
    danger_close_storage: bool,
) -> Result<CliStorageEstimate, InfraError> {
    semantic_code_infra::estimate_storage_local(
        config_path,
        overrides_json,
        codebase_root,
        danger_close_storage,
    )
    .map(Into::into)
    .map_err(Into::into)
}

/// Enforce local storage headroom before indexing.
#[instrument(
    name = "facade.ensure_storage_headroom_local",
    skip_all,
    fields(
        has_config_path = config_path.is_some(),
        has_overrides_json = overrides_json.is_some(),
        danger_close_storage
    )
)]
pub fn ensure_storage_headroom_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    codebase_root: &Path,
    danger_close_storage: bool,
) -> Result<CliStorageEstimate, InfraError> {
    semantic_code_infra::ensure_storage_headroom_local(
        config_path,
        overrides_json,
        codebase_root,
        danger_close_storage,
    )
    .map(Into::into)
    .map_err(Into::into)
}

/// Run BQ1 threshold calibration against the local vector index.
#[instrument(
    name = "facade.run_calibrate_local",
    skip_all,
    fields(
        has_config_path = config_path.is_some(),
        has_overrides_json = overrides_json.is_some()
    )
)]
pub fn run_calibrate_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    codebase_root: &Path,
    params: &CalibrationParams,
) -> Result<CalibrationState, InfraError> {
    semantic_code_infra::run_calibrate_local(config_path, overrides_json, codebase_root, params)
        .map_err(Into::into)
}

/// Convert an infra error into an API v1 error payload (stable code + meta).
#[instrument(name = "facade.infra_error_to_api_v1", skip_all)]
pub fn infra_error_to_api_v1(error: &InfraError) -> ApiV1ErrorDto {
    semantic_code_api::v1::error_envelope_to_api_v1_error(error.as_envelope(), None).into()
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
