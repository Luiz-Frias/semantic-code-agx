//! # semantic-code-infra
//!
//! Infrastructure wiring and runtime composition.
//! This crate depends on `app`, `adapters`, `config`, and `shared`.

/// Local CLI orchestration helpers.
pub mod cli_local;
/// CLI manifest helpers for local commands.
pub mod cli_manifest;
/// Config loading helpers used by CLI surfaces.
pub mod config_check;
/// Embedding adapter selection helpers.
mod embedding_factory;
/// Embedding routing helpers.
mod embedding_router;
/// Environment validation helpers used by CLI surfaces.
pub mod env_check;
/// In-memory index smoke helper for CLI self-check.
pub mod index_smoke;
/// Background job helpers.
pub mod jobs;
/// Request validation helpers used by CLI surfaces.
pub mod request_check;
/// Vector DB adapter selection helpers.
mod vectordb_factory;

pub use cli_local::{
    CliConfigSummary, CliInitStatus, CliStatus, SnapshotStatus, read_status_local, run_clear_local,
    run_index_local, run_init_local, run_reindex_local, run_search_local,
};
pub use cli_manifest::{
    CliManifest, append_context_gitignore, config_path, ensure_default_config, manifest_path,
    read_manifest, touch_manifest, write_manifest,
};
pub use config_check::load_effective_config_json;
pub use embedding_factory::{build_embedding_port, build_embedding_port_with_telemetry};
pub use env_check::{InfraError, InfraResult, validate_env_parsing};
pub use index_smoke::{run_clear_smoke, run_index_smoke, run_search_smoke};
pub use jobs::{
    JobError, JobKind, JobProgress, JobRequest, JobResult, JobState, JobStatus, cancel_job,
    create_job, read_job_status, run_job,
};
pub use request_check::{RequestKind, ValidatedRequest, validate_request_json};
pub use vectordb_factory::build_vectordb_port;

// Re-export redaction utilities for CLI boundary sanitization
pub use semantic_code_shared::{is_secret_key, redact_if_secret};

/// Placeholder module for infrastructure wiring.
pub mod placeholder {
    /// Placeholder function to verify the crate compiles.
    #[must_use]
    pub const fn infra_crate_version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}

pub use placeholder::infra_crate_version;

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_adapters::adapters_crate_version;
    use semantic_code_app::app_crate_version;
    use semantic_code_config::config_crate_version;
    use semantic_code_shared::shared_crate_version;

    fn workspace_deps() -> Vec<String> {
        let cargo_toml = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"));
        let mut deps = Vec::new();
        let mut in_deps = false;
        let mut in_dev_deps = false;

        for raw_line in cargo_toml.lines() {
            let line = raw_line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('[') {
                in_deps = line == "[dependencies]";
                in_dev_deps = line == "[dev-dependencies]";
                continue;
            }
            if !(in_deps || in_dev_deps) {
                continue;
            }
            if line.starts_with("semantic-code-") {
                let key = line.split('=').next().unwrap_or("").trim();
                let name = key.split('.').next().unwrap_or("").trim();
                deps.push(name.to_string());
            }
        }

        deps
    }

    /// P01.M2.13: infra compiles with app + adapters + config
    #[test]
    fn infra_depends_on_app_adapters_config() {
        let deps = workspace_deps();
        let required = [
            "semantic-code-app",
            "semantic-code-adapters",
            "semantic-code-config",
        ];

        for expected in required {
            assert!(
                deps.iter().any(|dep| dep == expected),
                "missing dependency: {expected}"
            );
        }
    }

    #[test]
    fn infra_crate_compiles() {
        let version = infra_crate_version();
        assert!(!version.is_empty());
    }

    #[test]
    fn infra_can_use_app_adapters_config_shared() {
        let app_version = app_crate_version();
        let adapters_version = adapters_crate_version();
        let config_version = config_crate_version();
        let shared_version = shared_crate_version();

        assert!(!app_version.is_empty());
        assert!(!adapters_version.is_empty());
        assert!(!config_version.is_empty());
        assert!(!shared_version.is_empty());
    }
}
