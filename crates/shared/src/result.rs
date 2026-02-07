//! Result helpers for shared error handling.

use crate::errors::ErrorEnvelope;

/// Shared result type used across the workspace.
pub type Result<T, E = ErrorEnvelope> = std::result::Result<T, E>;

/// Extension helpers mirroring common `Result` combinators.
pub trait ResultExt<T, E> {
    /// Map the success value, preserving the error.
    fn map_ok<U, F>(self, op: F) -> Result<U, E>
    where
        F: FnOnce(T) -> U;

    /// Map the error value, preserving the success.
    fn map_err_with<F, E2>(self, op: F) -> Result<T, E2>
    where
        F: FnOnce(E) -> E2;

    /// Chain fallible operations using the shared `Result` type.
    fn and_then_with<U, F>(self, op: F) -> Result<U, E>
    where
        F: FnOnce(T) -> Result<U, E>;
}

impl<T, E> ResultExt<T, E> for Result<T, E> {
    fn map_ok<U, F>(self, op: F) -> Result<U, E>
    where
        F: FnOnce(T) -> U,
    {
        self.map(op)
    }

    fn map_err_with<F, E2>(self, op: F) -> Result<T, E2>
    where
        F: FnOnce(E) -> E2,
    {
        self.map_err(op)
    }

    fn and_then_with<U, F>(self, op: F) -> Result<U, E>
    where
        F: FnOnce(T) -> Result<U, E>,
    {
        self.and_then(op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{ErrorCode, ErrorEnvelope};

    #[test]
    fn result_ext_maps_ok() {
        let value: Result<i32> = Ok(1);
        let mapped = value.map_ok(|value| value + 2);

        assert!(matches!(mapped, Ok(3)));
    }

    #[test]
    fn result_ext_maps_err() {
        let error = ErrorEnvelope::expected(ErrorCode::invalid_input(), "bad input");
        let value: Result<i32> = Err(error);
        let mapped = value.map_err_with(|error| error.with_metadata("field", "name"));

        assert!(mapped.is_err());
        if let Err(error) = mapped {
            assert_eq!(
                error.metadata.get("field").map(String::as_str),
                Some("name")
            );
        }
    }

    #[test]
    fn result_ext_and_then() {
        let value: Result<i32> = Ok(2);
        let chained = value.and_then_with(|value| Ok(value * 3));

        assert!(matches!(chained, Ok(6)));
    }
}
