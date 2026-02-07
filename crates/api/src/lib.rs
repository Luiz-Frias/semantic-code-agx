//! # semantic-code-api
//!
//! API data transfer objects and wire formats.
//! This crate depends only on `domain` and `shared`.

/// API v1 DTOs.
pub mod v1;

/// Placeholder module for API helpers.
pub mod placeholder {
    /// Placeholder function to verify the crate compiles.
    #[must_use]
    pub const fn api_crate_version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}

pub use placeholder::api_crate_version;

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_domain::domain_crate_version;
    use semantic_code_shared::shared_crate_version;

    #[test]
    fn api_crate_compiles() {
        let version = api_crate_version();
        assert!(!version.is_empty());
    }

    #[test]
    fn api_can_use_domain_and_shared() {
        let domain_version = domain_crate_version();
        let shared_version = shared_crate_version();

        assert!(!domain_version.is_empty());
        assert!(!shared_version.is_empty());
    }
}
