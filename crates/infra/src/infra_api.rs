//! Curated public API boundary for `semantic-code-infra`.

pub use crate::cli_calibration::{
    calibration_path, delete_calibration, read_calibration, write_calibration,
};
pub use crate::cli_local::{
    CliConfigSummary, CliInitStatus, CliStatus, LocalSearchSession, SnapshotStatus,
    open_search_session, open_search_session_with_options, read_status_local, run_calibrate_local,
    run_clear_local, run_index_local, run_init_local, run_reindex_local, run_search_local,
};
pub use crate::config_check::load_effective_config_json;
pub use crate::env_check::{InfraError, InfraResult, validate_env_parsing};
pub use crate::index_smoke::{run_clear_smoke, run_index_smoke, run_search_smoke};
pub use crate::jobs::{
    JobError, JobKind, JobProgress, JobRequest, JobResult, JobState, JobStatus, cancel_job,
    create_job, read_job_status, run_job,
};
pub use crate::request_check::{RequestKind, validate_request_json};
pub use crate::storage_estimate::{
    CliStorageEstimate, StorageThresholdStatus, ensure_storage_headroom_local,
    estimate_storage_local,
};
pub use semantic_code_ports::EmbeddingVector;

/// Crate version from Cargo metadata.
#[must_use]
pub const fn infra_crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
