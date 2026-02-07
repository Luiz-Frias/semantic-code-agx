//! Test fixtures for shared error codes and envelopes.

use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope};

/// Return a list of common error codes used in tests.
pub fn common_error_codes() -> Vec<ErrorCode> {
    vec![
        ErrorCode::cancelled(),
        ErrorCode::invalid_input(),
        ErrorCode::not_found(),
        ErrorCode::timeout(),
        ErrorCode::io(),
        ErrorCode::internal(),
    ]
}

/// A cancellation error fixture.
pub fn cancelled_error() -> ErrorEnvelope {
    ErrorEnvelope::cancelled("cancelled")
}

/// An invalid input error fixture.
pub fn invalid_input_error() -> ErrorEnvelope {
    ErrorEnvelope::expected(ErrorCode::invalid_input(), "invalid input")
}

/// A retriable timeout error fixture.
pub fn timeout_error() -> ErrorEnvelope {
    ErrorEnvelope::unexpected(ErrorCode::timeout(), "timeout", ErrorClass::Retriable)
}
