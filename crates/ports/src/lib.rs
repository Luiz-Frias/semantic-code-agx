//! # semantic-code-ports
//!
//! Port traits for the semantic-code-agents hexagonal architecture.
//!
//! This crate defines the interfaces between the domain and infrastructure
//! layers. It depends only on `domain` and `shared`.

use std::future::Future;
use std::pin::Pin;

/// Boxed future used by port traits.
///
/// We deliberately use boxed futures for boundary traits (I/O-bound work), and
/// prefer batch APIs for hot paths where allocation overhead matters.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Returns the ports crate version.
#[must_use]
pub const fn ports_crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

pub mod embedding;
pub mod filesystem;
pub mod ignore;
pub mod logger;
pub mod splitter;
pub mod sync;
pub mod telemetry;
pub mod vectordb;

pub use embedding::*;
pub use filesystem::*;
pub use ignore::*;
pub use logger::*;
pub use splitter::*;
pub use sync::*;
pub use telemetry::*;
pub use vectordb::*;

// Re-export selected domain types used in port signatures, so adapter crates
// can implement ports without directly depending on `semantic-code-domain`.
pub use semantic_code_domain::{
    CollectionName, EmbeddingProviderId, Language, LineSpan, VectorDbProviderId,
    VectorDocumentMetadata,
};

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_domain::domain_crate_version;
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

    /// P01.M2.11: ports depends only on domain + shared
    #[test]
    fn ports_depends_only_on_domain_and_shared() {
        let deps = workspace_deps();
        let allowed = ["semantic-code-domain", "semantic-code-shared"];

        for dep in &deps {
            assert!(
                allowed.contains(&dep.as_str()),
                "unexpected dependency found: {dep}"
            );
        }

        for expected in allowed {
            assert!(
                deps.iter().any(|dep| dep == expected),
                "missing dependency: {expected}"
            );
        }
    }

    #[test]
    fn ports_crate_compiles() {
        let version = ports_crate_version();
        assert!(!version.is_empty());
    }

    #[test]
    fn ports_can_use_domain_and_shared() {
        let domain_version = domain_crate_version();
        let shared_version = shared_crate_version();

        assert!(!domain_version.is_empty());
        assert!(!shared_version.is_empty());
    }
}
