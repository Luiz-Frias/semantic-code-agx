//! Integration tests for shared error propagation.

use semantic_code_shared::{ErrorCode, ErrorEnvelope, UnexpectedError, normalize_unexpected_error};
use semantic_code_testkit::errors::{cancelled_error, timeout_error};

#[test]
fn error_envelope_crosses_crates() {
    let timeout = timeout_error();
    assert_eq!(timeout.code, ErrorCode::timeout());

    let boxed: Box<dyn std::error::Error> = Box::new(timeout);
    let description = boxed.to_string();
    assert!(description.contains("timeout"));

    let cancelled = cancelled_error();
    assert!(cancelled.is_cancelled());
}

#[test]
fn normalize_unexpected_error_is_available() {
    let envelope = normalize_unexpected_error(UnexpectedError::message("boom"));
    assert_eq!(envelope.code, ErrorCode::internal());
    assert_eq!(envelope.kind, semantic_code_shared::ErrorKind::Unexpected);

    let io_error = std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout");
    let envelope = normalize_unexpected_error(UnexpectedError::error(io_error));
    assert_eq!(envelope.code, ErrorCode::timeout());
}

#[test]
fn error_envelope_constructors_work() {
    let expected = ErrorEnvelope::expected(ErrorCode::invalid_input(), "bad input");
    assert_eq!(expected.kind, semantic_code_shared::ErrorKind::Expected);
}
