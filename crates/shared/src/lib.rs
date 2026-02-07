//! # semantic-code-shared
//!
//! Shared utilities, result types, and error handling for the semantic-code-agents workspace.
//!
//! This crate provides foundational types that are used across all other crates:
//!
//! - Result and error envelope types (Phase 02)
//! - Concurrency primitives (Phase 04)
//! - Common utilities
//!
//! ## Design Principles
//!
//! 1. **No workspace dependencies** - This crate only depends on external crates
//! 2. **Zero-cost abstractions** - Types should compile away to efficient code
//! 3. **Serde-compatible** - All public types support serialization

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]

// =============================================================================
// PHASE 02: RESULT + ERROR ENVELOPE
// =============================================================================

pub mod concurrency;
pub mod errors;
pub mod invariants;
pub mod merkle;
pub mod redaction;
pub mod result;
pub mod retry;
pub mod timeout;
pub mod validation;

// =============================================================================
// PHASE 04 PLACEHOLDERS
// =============================================================================
// These modules will be implemented in Phase 04: Ports and Use-cases

pub use concurrency::{
    BoundedQueue, BoundedQueueClosedError, CancellationToken, CorrelationId, RequestContext,
    WorkerPool, WorkerPoolOptions,
};
pub use errors::{
    ErrorClass, ErrorCode, ErrorEnvelope, ErrorKind, ErrorMetadata, REDACTED_VALUE,
    UnexpectedError, abort, check_cancelled, is_abort, normalize_unexpected_error, redact_metadata,
};
pub use invariants::{
    BoundedU32, BoundedU64, BoundedUsize, BoundsError, Unvalidated, Validated, ValidatedState,
};
pub use redaction::{REDACTED, Redacted, SecretString, is_secret_key, redact_if_secret};
pub use result::{Result, ResultExt};
pub use retry::{RetryPolicy, retry_async, retry_async_with_observer};
pub use timeout::timeout_with_context;
pub use validation::{Validate, ValidationError};

/// Returns the shared crate version.
#[must_use]
pub const fn shared_crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::errors::{ErrorClass, ErrorCode, ErrorEnvelope};
    use super::result::{Result, ResultExt};

    #[test]
    fn shared_error_types_are_available() {
        let error = ErrorEnvelope::expected(ErrorCode::invalid_input(), "invalid");
        assert_eq!(error.kind, super::errors::ErrorKind::Expected);
        assert_eq!(error.class, ErrorClass::NonRetriable);
    }

    #[test]
    fn shared_result_type_is_available() {
        let value: Result<i32> = Ok(5);
        let mapped = value.map_ok(|value| value + 1);
        assert!(matches!(mapped, Ok(6)));
    }
}
