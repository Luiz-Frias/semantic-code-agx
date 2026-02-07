//! # semantic-code-testkit
//!
//! Test helpers and in-memory adapters.
//! This crate depends on `ports` and `shared`.

pub mod errors;
pub mod in_memory;
pub mod parity;

/// Returns the testkit crate version.
#[must_use]
pub const fn testkit_crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_ports::ports_crate_version;
    use semantic_code_shared::shared_crate_version;

    #[test]
    fn testkit_crate_compiles() {
        let version = testkit_crate_version();
        assert!(!version.is_empty());
    }

    #[test]
    fn testkit_can_use_ports_and_shared() {
        let ports_version = ports_crate_version();
        let shared_version = shared_crate_version();

        assert!(!ports_version.is_empty());
        assert!(!shared_version.is_empty());
    }

    #[test]
    fn error_fixtures_are_available() {
        let codes = errors::common_error_codes();
        assert!(!codes.is_empty());
    }

    #[test]
    fn in_memory_adapters_are_available() {
        let _ = in_memory::NoopLogger::default();
    }
}
