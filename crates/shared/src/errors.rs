//! Error envelope types and helpers.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::result::Result as StdResult;
use std::{fmt, io};

/// Metadata attached to errors for diagnostics.
pub type ErrorMetadata = BTreeMap<String, String>;

/// Redacted placeholder value for sensitive metadata.
pub const REDACTED_VALUE: &str = "<redacted>";

/// High-level classification of error origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorKind {
    /// Expected failures (validation, user input, cancellation).
    Expected,
    /// Invariant violations in domain logic.
    Invariant,
    /// Unexpected failures (I/O, external dependencies).
    Unexpected,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Expected => formatter.write_str("expected"),
            Self::Invariant => formatter.write_str("invariant"),
            Self::Unexpected => formatter.write_str("unexpected"),
        }
    }
}

/// Retry classification for failure handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorClass {
    /// The operation can be retried safely.
    Retriable,
    /// The operation should not be retried.
    NonRetriable,
}

impl ErrorClass {
    /// Returns true when the error is considered retriable.
    #[must_use]
    pub const fn is_retriable(self) -> bool {
        matches!(self, Self::Retriable)
    }
}

impl fmt::Display for ErrorClass {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Retriable => formatter.write_str("retriable"),
            Self::NonRetriable => formatter.write_str("non-retriable"),
        }
    }
}

/// Stable error code with namespace and identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ErrorCode {
    namespace: String,
    code: String,
}

impl ErrorCode {
    /// Create a new error code with a namespace and code.
    pub fn new(namespace: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            code: code.into(),
        }
    }

    /// Core cancellation code.
    pub fn cancelled() -> Self {
        Self::new("core", "cancelled")
    }

    /// Invalid input code.
    pub fn invalid_input() -> Self {
        Self::new("core", "invalid_input")
    }

    /// Not found code.
    pub fn not_found() -> Self {
        Self::new("core", "not_found")
    }

    /// Permission denied code.
    pub fn permission_denied() -> Self {
        Self::new("core", "permission_denied")
    }

    /// Timeout code.
    pub fn timeout() -> Self {
        Self::new("core", "timeout")
    }

    /// I/O error code.
    pub fn io() -> Self {
        Self::new("core", "io")
    }

    /// Internal failure code.
    pub fn internal() -> Self {
        Self::new("core", "internal")
    }

    /// Returns the namespace portion.
    #[must_use]
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Returns the code identifier.
    #[must_use]
    pub fn code(&self) -> &str {
        &self.code
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}", self.namespace, self.code)
    }
}

/// Structured error envelope shared across crates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorEnvelope {
    /// Error kind describing the origin category.
    pub kind: ErrorKind,
    /// Retry classification.
    pub class: ErrorClass,
    /// Stable error code.
    pub code: ErrorCode,
    /// Human-readable error message.
    pub message: String,
    /// Additional diagnostic metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: ErrorMetadata,
}

impl ErrorEnvelope {
    /// Create an expected error with non-retriable classification.
    pub fn expected(code: ErrorCode, message: impl Into<String>) -> Self {
        Self::expected_with_class(code, message, ErrorClass::NonRetriable)
    }

    /// Create an expected error with an explicit retry classification.
    pub fn expected_with_class(
        code: ErrorCode,
        message: impl Into<String>,
        class: ErrorClass,
    ) -> Self {
        Self {
            kind: ErrorKind::Expected,
            class,
            code,
            message: message.into(),
            metadata: BTreeMap::new(),
        }
    }

    /// Create an invariant error (always non-retriable).
    pub fn invariant(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Invariant,
            class: ErrorClass::NonRetriable,
            code,
            message: message.into(),
            metadata: BTreeMap::new(),
        }
    }

    /// Create an unexpected error with the provided retry classification.
    pub fn unexpected(code: ErrorCode, message: impl Into<String>, class: ErrorClass) -> Self {
        Self {
            kind: ErrorKind::Unexpected,
            class,
            code,
            message: message.into(),
            metadata: BTreeMap::new(),
        }
    }

    /// Create a cancellation error.
    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::expected_with_class(ErrorCode::cancelled(), message, ErrorClass::NonRetriable)
    }

    /// Returns true if the error represents a cancellation.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.code == ErrorCode::cancelled()
    }

    /// Attach a single metadata entry.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Replace metadata with a redacted copy for the provided keys.
    #[must_use]
    pub fn redact_metadata(self, keys: &[&str]) -> Self {
        Self {
            metadata: redact_metadata(self.metadata, keys),
            ..self
        }
    }
}

impl fmt::Display for ErrorEnvelope {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} {} {}: {}",
            self.kind, self.class, self.code, self.message
        )
    }
}

impl std::error::Error for ErrorEnvelope {}

impl From<io::Error> for ErrorEnvelope {
    fn from(error: io::Error) -> Self {
        normalize_unexpected_error(UnexpectedError::error(error))
    }
}

/// Check for cancellation and return a cancellation error when flagged.
pub fn check_cancelled(cancelled: bool) -> StdResult<(), ErrorEnvelope> {
    if cancelled {
        Err(ErrorEnvelope::cancelled("operation cancelled"))
    } else {
        Ok(())
    }
}

/// Create an abort error (alias of cancellation).
pub fn abort(message: impl Into<String>) -> ErrorEnvelope {
    ErrorEnvelope::cancelled(message)
}

/// Returns true when the error represents an abort/cancellation.
#[must_use]
pub fn is_abort(error: &ErrorEnvelope) -> bool {
    error.is_cancelled()
}

/// Redact sensitive metadata values for the provided keys.
#[must_use]
pub fn redact_metadata(mut metadata: ErrorMetadata, keys: &[&str]) -> ErrorMetadata {
    for key in keys {
        if metadata.contains_key(*key) {
            metadata.insert((*key).to_string(), REDACTED_VALUE.to_string());
        }
    }

    metadata
}

/// Normalize unexpected errors into a structured envelope.
pub fn normalize_unexpected_error(error: UnexpectedError) -> ErrorEnvelope {
    match error {
        UnexpectedError::Message(message) => {
            ErrorEnvelope::unexpected(ErrorCode::internal(), message, ErrorClass::NonRetriable)
        },
        UnexpectedError::Error(error) => {
            let (code, class) = classify_error(&*error);
            ErrorEnvelope::unexpected(code, error.to_string(), class)
        },
    }
}

/// Normalized wrapper for unexpected errors and messages.
#[derive(Debug)]
pub enum UnexpectedError {
    /// Unexpected error message.
    Message(String),
    /// Unexpected error payload.
    Error(Box<dyn std::error::Error + Send + Sync>),
}

impl UnexpectedError {
    /// Wrap an unexpected message.
    pub fn message(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }

    /// Wrap an unexpected error value.
    pub fn error<E>(error: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Error(Box::new(error))
    }
}

fn classify_error(error: &(dyn std::error::Error + 'static)) -> (ErrorCode, ErrorClass) {
    if let Some(io_error) = find_io_error(error) {
        let kind = io_error.kind();
        let code = error_code_from_io_kind(kind);
        let class = if is_retriable_io(kind) {
            ErrorClass::Retriable
        } else {
            ErrorClass::NonRetriable
        };
        return (code, class);
    }

    (ErrorCode::internal(), ErrorClass::NonRetriable)
}

fn find_io_error<'a>(error: &'a (dyn std::error::Error + 'static)) -> Option<&'a io::Error> {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(error);

    while let Some(candidate) = current {
        if let Some(io_error) = candidate.downcast_ref::<io::Error>() {
            return Some(io_error);
        }
        current = candidate.source();
    }

    None
}

fn error_code_from_io_kind(kind: io::ErrorKind) -> ErrorCode {
    match kind {
        io::ErrorKind::NotFound => ErrorCode::not_found(),
        io::ErrorKind::PermissionDenied => ErrorCode::permission_denied(),
        io::ErrorKind::TimedOut => ErrorCode::timeout(),
        io::ErrorKind::Interrupted => ErrorCode::cancelled(),
        _ => ErrorCode::io(),
    }
}

const fn is_retriable_io(kind: io::ErrorKind) -> bool {
    matches!(
        kind,
        io::ErrorKind::WouldBlock
            | io::ErrorKind::TimedOut
            | io::ErrorKind::Interrupted
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::NotConnected
            | io::ErrorKind::AddrInUse
            | io::ErrorKind::AddrNotAvailable
            | io::ErrorKind::BrokenPipe
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_envelope_constructors() {
        let expected = ErrorEnvelope::expected(ErrorCode::invalid_input(), "invalid");
        assert_eq!(expected.kind, ErrorKind::Expected);
        assert_eq!(expected.class, ErrorClass::NonRetriable);
        assert_eq!(expected.code, ErrorCode::invalid_input());

        let invariant = ErrorEnvelope::invariant(ErrorCode::internal(), "boom");
        assert_eq!(invariant.kind, ErrorKind::Invariant);
        assert_eq!(invariant.class, ErrorClass::NonRetriable);

        let unexpected =
            ErrorEnvelope::unexpected(ErrorCode::timeout(), "timeout", ErrorClass::Retriable);
        assert_eq!(unexpected.kind, ErrorKind::Unexpected);
        assert!(unexpected.class.is_retriable());
    }

    #[test]
    fn normalize_unexpected_error_handles_error_and_message() {
        let io_error = io::Error::new(io::ErrorKind::TimedOut, "timeout");
        let envelope = normalize_unexpected_error(UnexpectedError::error(io_error));

        assert_eq!(envelope.kind, ErrorKind::Unexpected);
        assert!(envelope.class.is_retriable());
        assert_eq!(envelope.code, ErrorCode::timeout());

        let envelope = normalize_unexpected_error(UnexpectedError::message("boom"));
        assert_eq!(envelope.kind, ErrorKind::Unexpected);
        assert_eq!(envelope.class, ErrorClass::NonRetriable);
        assert_eq!(envelope.code, ErrorCode::internal());
    }

    #[test]
    fn cancellation_helpers_detect_abort() {
        let cancelled = abort("stopped");
        assert!(is_abort(&cancelled));

        let result = check_cancelled(true);
        assert!(result.is_err());
    }

    #[test]
    fn metadata_redaction() {
        let error = ErrorEnvelope::expected(ErrorCode::invalid_input(), "bad")
            .with_metadata("token", "secret")
            .with_metadata("path", "value");
        let redacted = error.redact_metadata(&["token"]);

        assert_eq!(
            redacted.metadata.get("token").map(String::as_str),
            Some(REDACTED_VALUE)
        );
        assert_eq!(
            redacted.metadata.get("path").map(String::as_str),
            Some("value")
        );
    }
}
