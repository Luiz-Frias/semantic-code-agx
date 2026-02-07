//! Chunk content with compile-time max length.

use semantic_code_shared::{ErrorCode, ErrorEnvelope};
use std::fmt;

/// Hard upper bound for chunk content length.
pub const MAX_CHUNK_CHARS: usize = 20_000;

/// Error when chunk content exceeds the maximum size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkError {
    /// Observed length.
    pub length: usize,
    /// Maximum allowed length.
    pub max: usize,
}

impl fmt::Display for ChunkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "chunk length {} exceeds max {}",
            self.length, self.max
        )
    }
}

impl std::error::Error for ChunkError {}

impl From<ChunkError> for ErrorEnvelope {
    fn from(error: ChunkError) -> Self {
        Self::expected(
            ErrorCode::new("domain", "chunk_too_large"),
            error.to_string(),
        )
        .with_metadata("length", error.length.to_string())
        .with_metadata("max", error.max.to_string())
    }
}

/// Chunk content capped by a compile-time max length.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Chunk<const MAX: usize>(Box<str>);

impl<const MAX: usize> Chunk<MAX> {
    /// Validate and build a chunk from content.
    pub fn new(content: impl Into<Box<str>>) -> Result<Self, ChunkError> {
        let content = content.into();
        let length = content.len();
        if length > MAX {
            return Err(ChunkError { length, max: MAX });
        }
        Ok(Self(content))
    }

    /// Borrow the chunk content.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume and return the chunk content.
    #[must_use]
    pub fn into_inner(self) -> Box<str> {
        self.0
    }
}

impl<const MAX: usize> AsRef<str> for Chunk<MAX> {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}
