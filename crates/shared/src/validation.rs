//! Validation traits for request DTOs.

/// Trait for validation errors used by `Validate`.
pub trait ValidationError: Sized {
    /// A required field was empty.
    fn empty(field: &'static str) -> Self;

    /// A field value is invalid for a specific reason.
    fn invalid(field: &'static str, reason: &'static str) -> Self;

    /// A numeric field is outside the allowed range.
    fn out_of_range(field: &'static str, value: String, min: String, max: String) -> Self;
}

/// Validate a DTO using compile-time derived rules.
pub trait Validate {
    /// Error type returned by validation.
    type Error: ValidationError;

    /// Validate the DTO.
    fn validate(&self) -> Result<(), Self::Error>;
}
