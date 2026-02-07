//! # semantic-code-adapters
//!
//! Adapter implementations for ports (filesystem, embedding, vectordb, etc.).
//! This crate depends on `ports`, `shared`, and `vector`.

/// External embedding adapters.
pub mod embedding;

pub mod cache;
/// Deterministic embedding adapter for tests.
pub mod embedding_test;
pub mod file_sync;
pub mod fs;
pub mod ignore;
pub mod log_sink;
pub mod logger;
pub mod self_check;
pub mod splitter;
pub mod telemetry;
pub mod vectordb;
pub mod vectordb_local;

/// Placeholder module for adapters.
pub mod placeholder {
    /// Placeholder function to verify the crate compiles.
    #[must_use]
    pub const fn adapters_crate_version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}

pub use placeholder::adapters_crate_version;

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_ports::ports_crate_version;
    use semantic_code_shared::shared_crate_version;
    use semantic_code_vector::vector_crate_version;

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

    /// P01.M2.12: adapters compile without importing app or infra
    #[test]
    fn adapters_do_not_depend_on_app_or_infra() {
        let deps = workspace_deps();
        let forbidden = ["semantic-code-app", "semantic-code-infra"];

        for dep in &deps {
            assert!(
                !forbidden.contains(&dep.as_str()),
                "forbidden dependency found: {dep}"
            );
        }
    }

    #[test]
    fn adapters_crate_compiles() {
        let version = adapters_crate_version();
        assert!(!version.is_empty());
    }

    #[test]
    fn adapters_can_use_ports_shared_vector() {
        let ports_version = ports_crate_version();
        let shared_version = shared_crate_version();
        let vector_version = vector_crate_version();

        assert!(!ports_version.is_empty());
        assert!(!shared_version.is_empty());
        assert!(!vector_version.is_empty());
    }
}
