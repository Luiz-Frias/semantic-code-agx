//! # semantic-code-infra
//!
//! Infrastructure wiring and runtime composition.
//! This crate depends on `app`, `adapters`, `config`, and `shared`.

/// EMA persistence strategies for calibrated BQ1 observations.
mod calibration_persist;
/// BQ1 calibration persistence helpers for local CLI commands.
mod cli_calibration;
/// Local CLI orchestration helpers.
mod cli_local;
/// CLI manifest helpers for local commands.
mod cli_manifest;
/// Config loading helpers used by CLI surfaces.
mod config_check;
/// Embedding adapter selection helpers.
mod embedding_factory;
/// Embedding routing helpers.
mod embedding_router;
/// Environment validation helpers used by CLI surfaces.
mod env_check;
/// In-memory index smoke helper for CLI self-check.
mod index_smoke;
/// Curated public API boundary.
mod infra_api;
/// Background job helpers.
mod jobs;
/// Request validation helpers used by CLI surfaces.
mod request_check;
/// Storage estimation and headroom preflight helpers.
mod storage_estimate;
/// Vector DB adapter selection helpers.
mod vectordb_factory;

pub use infra_api::*;

#[cfg(test)]
mod factory_selection_tests;

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
