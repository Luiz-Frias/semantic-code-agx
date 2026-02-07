//! Domain primitives with validated constructors.

use crate::LineSpan;
use semantic_code_shared::{ErrorCode, ErrorEnvelope};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::path::{Path, PathBuf};

/// Validation failures for domain primitives and spans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrimitiveError {
    /// `CodebaseId` is empty after trimming.
    InvalidCodebaseId {
        /// Length of the raw input before trimming.
        input_length: usize,
    },
    /// `CollectionName` is empty after trimming.
    EmptyCollectionName {
        /// Length of the raw input before trimming.
        input_length: usize,
    },
    /// `CollectionName` violates the allowed pattern.
    InvalidCollectionName {
        /// Trimmed collection name that failed validation.
        input: String,
    },
    /// `DocumentId` is empty after trimming.
    InvalidDocumentId {
        /// Length of the raw input before trimming.
        input_length: usize,
    },
    /// `ChunkId` is empty after trimming.
    InvalidChunkId {
        /// Length of the raw input before trimming.
        input_length: usize,
    },
    /// `EmbeddingProviderId` is empty after trimming.
    InvalidEmbeddingProviderId {
        /// Length of the raw input before trimming.
        input_length: usize,
    },
    /// `VectorDbProviderId` is empty after trimming.
    InvalidVectorDbProviderId {
        /// Length of the raw input before trimming.
        input_length: usize,
    },
    /// `LineSpan` start/end must be >= 1.
    LineSpanNonPositive {
        /// Starting line (1-indexed).
        start_line: u32,
        /// Ending line (1-indexed).
        end_line: u32,
    },
    /// `LineSpan` start must be <= end.
    LineSpanStartAfterEnd {
        /// Starting line (1-indexed).
        start_line: u32,
        /// Ending line (1-indexed).
        end_line: u32,
    },
    /// Derived codebase id is invalid (invariant violation).
    DerivedCodebaseIdInvalid {
        /// Candidate codebase id that failed validation.
        candidate: String,
    },
    /// Derived chunk id is invalid (invariant violation).
    DerivedChunkIdInvalid {
        /// Candidate chunk id that failed validation.
        candidate: String,
    },
    /// Derived collection name is invalid (invariant violation).
    DerivedCollectionNameInvalid {
        /// Candidate collection name that failed validation.
        candidate: String,
    },
}

impl PrimitiveError {
    fn error_code(&self) -> ErrorCode {
        match self {
            Self::InvalidCodebaseId { .. } | Self::DerivedCodebaseIdInvalid { .. } => {
                ErrorCode::new("domain", "invalid_codebase_id")
            },
            Self::EmptyCollectionName { .. }
            | Self::InvalidCollectionName { .. }
            | Self::DerivedCollectionNameInvalid { .. } => {
                ErrorCode::new("domain", "invalid_collection_name")
            },
            Self::InvalidDocumentId { .. } => ErrorCode::new("domain", "invalid_document_id"),
            Self::InvalidChunkId { .. } | Self::DerivedChunkIdInvalid { .. } => {
                ErrorCode::new("domain", "invalid_chunk_id")
            },
            Self::InvalidEmbeddingProviderId { .. } => {
                ErrorCode::new("domain", "invalid_embedding_provider_id")
            },
            Self::InvalidVectorDbProviderId { .. } => {
                ErrorCode::new("domain", "invalid_vectordb_provider_id")
            },
            Self::LineSpanNonPositive { .. } | Self::LineSpanStartAfterEnd { .. } => {
                ErrorCode::new("domain", "invalid_line_span")
            },
        }
    }

    const fn is_invariant(&self) -> bool {
        matches!(
            self,
            Self::DerivedCodebaseIdInvalid { .. }
                | Self::DerivedChunkIdInvalid { .. }
                | Self::DerivedCollectionNameInvalid { .. }
        )
    }
}

impl fmt::Display for PrimitiveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCodebaseId { .. } => formatter.write_str("CodebaseId must be non-empty"),
            Self::EmptyCollectionName { .. } => {
                formatter.write_str("CollectionName must be non-empty")
            },
            Self::InvalidCollectionName { .. } => {
                formatter.write_str("CollectionName must match /^[a-zA-Z][a-zA-Z0-9_]*$/")
            },
            Self::InvalidDocumentId { .. } => formatter.write_str("DocumentId must be non-empty"),
            Self::InvalidChunkId { .. } => formatter.write_str("ChunkId must be non-empty"),
            Self::InvalidEmbeddingProviderId { .. } => {
                formatter.write_str("EmbeddingProviderId must be non-empty")
            },
            Self::InvalidVectorDbProviderId { .. } => {
                formatter.write_str("VectorDbProviderId must be non-empty")
            },
            Self::LineSpanNonPositive { .. } => {
                formatter.write_str("LineSpan start_line/end_line must be >= 1")
            },
            Self::LineSpanStartAfterEnd { .. } => {
                formatter.write_str("LineSpan start_line must be <= end_line")
            },
            Self::DerivedCodebaseIdInvalid { .. } => {
                formatter.write_str("Derived codebase id is invalid (this is a bug).")
            },
            Self::DerivedChunkIdInvalid { .. } => {
                formatter.write_str("Derived chunk id is invalid (this is a bug).")
            },
            Self::DerivedCollectionNameInvalid { .. } => {
                formatter.write_str("Derived collection name is invalid (this is a bug).")
            },
        }
    }
}

impl std::error::Error for PrimitiveError {}

impl From<PrimitiveError> for ErrorEnvelope {
    fn from(error: PrimitiveError) -> Self {
        let mut envelope = if error.is_invariant() {
            Self::invariant(error.error_code(), error.to_string())
        } else {
            Self::expected(error.error_code(), error.to_string())
        };

        match error {
            PrimitiveError::InvalidCodebaseId { input_length }
            | PrimitiveError::EmptyCollectionName { input_length }
            | PrimitiveError::InvalidDocumentId { input_length }
            | PrimitiveError::InvalidChunkId { input_length }
            | PrimitiveError::InvalidEmbeddingProviderId { input_length }
            | PrimitiveError::InvalidVectorDbProviderId { input_length } => {
                envelope = envelope.with_metadata("input_length", input_length.to_string());
            },
            PrimitiveError::InvalidCollectionName { input } => {
                envelope = envelope.with_metadata("input", input);
            },
            PrimitiveError::LineSpanNonPositive {
                start_line,
                end_line,
            }
            | PrimitiveError::LineSpanStartAfterEnd {
                start_line,
                end_line,
            } => {
                envelope = envelope
                    .with_metadata("start_line", start_line.to_string())
                    .with_metadata("end_line", end_line.to_string());
            },
            PrimitiveError::DerivedCodebaseIdInvalid { candidate }
            | PrimitiveError::DerivedChunkIdInvalid { candidate }
            | PrimitiveError::DerivedCollectionNameInvalid { candidate } => {
                envelope = envelope.with_metadata("candidate", candidate);
            },
        }

        envelope
    }
}

/// Identifier for a codebase in the domain model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CodebaseId(Box<str>);

impl CodebaseId {
    /// Parse a `CodebaseId` from user input.
    pub fn parse(input: impl AsRef<str>) -> Result<Self, PrimitiveError> {
        let raw = input.as_ref();
        let Some(trimmed) = trimmed_non_empty(raw) else {
            return Err(PrimitiveError::InvalidCodebaseId {
                input_length: raw.len(),
            });
        };

        Ok(Self(trimmed.to_owned().into_boxed_str()))
    }

    /// Access the underlying string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume and return the underlying string.
    #[must_use]
    pub fn into_inner(self) -> Box<str> {
        self.0
    }
}

impl AsRef<str> for CodebaseId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for CodebaseId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Derive a deterministic codebase identifier from a path.
pub fn derive_codebase_id(codebase_root: impl AsRef<Path>) -> Result<CodebaseId, PrimitiveError> {
    let normalized = normalize_root_path(codebase_root.as_ref());
    let normalized = normalized.to_string_lossy();
    let digest = md5::compute(normalized.as_bytes());
    let hash = format!("{digest:x}");
    let hash_prefix: String = hash.chars().take(12).collect();
    let candidate = format!("codebase_{hash_prefix}");

    CodebaseId::parse(candidate.as_str())
        .map_err(|_| PrimitiveError::DerivedCodebaseIdInvalid { candidate })
}

/// Identifier for a vector collection.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CollectionName(Box<str>);

impl CollectionName {
    /// Parse a collection name that satisfies the allowlist pattern.
    pub fn parse(input: impl AsRef<str>) -> Result<Self, PrimitiveError> {
        let raw = input.as_ref();
        let Some(trimmed) = trimmed_non_empty(raw) else {
            return Err(PrimitiveError::EmptyCollectionName {
                input_length: raw.len(),
            });
        };

        if !is_valid_collection_name(trimmed) {
            return Err(PrimitiveError::InvalidCollectionName {
                input: trimmed.to_owned(),
            });
        }

        Ok(Self(trimmed.to_owned().into_boxed_str()))
    }

    /// Access the underlying string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume and return the underlying string.
    #[must_use]
    pub fn into_inner(self) -> Box<str> {
        self.0
    }
}

impl AsRef<str> for CollectionName {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for CollectionName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Identifier for a source document.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DocumentId(Box<str>);

impl DocumentId {
    /// Parse a `DocumentId` from user input.
    pub fn parse(input: impl AsRef<str>) -> Result<Self, PrimitiveError> {
        let raw = input.as_ref();
        let Some(trimmed) = trimmed_non_empty(raw) else {
            return Err(PrimitiveError::InvalidDocumentId {
                input_length: raw.len(),
            });
        };

        Ok(Self(trimmed.to_owned().into_boxed_str()))
    }

    /// Access the underlying string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume and return the underlying string.
    #[must_use]
    pub fn into_inner(self) -> Box<str> {
        self.0
    }
}

impl AsRef<str> for DocumentId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for DocumentId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Identifier for a content chunk.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ChunkId(Box<str>);

impl ChunkId {
    /// Parse a `ChunkId` from user input.
    pub fn parse(input: impl AsRef<str>) -> Result<Self, PrimitiveError> {
        let raw = input.as_ref();
        let Some(trimmed) = trimmed_non_empty(raw) else {
            return Err(PrimitiveError::InvalidChunkId {
                input_length: raw.len(),
            });
        };

        Ok(Self(trimmed.to_owned().into_boxed_str()))
    }

    /// Access the underlying string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume and return the underlying string.
    #[must_use]
    pub fn into_inner(self) -> Box<str> {
        self.0
    }
}

impl AsRef<str> for ChunkId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for ChunkId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Inputs required to derive a deterministic chunk id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkIdInput {
    /// Stable path identifier for the chunk.
    pub relative_path: Box<str>,
    /// Line span for the chunk (validated).
    pub span: LineSpan,
    /// Chunk content.
    pub content: Box<str>,
}

impl ChunkIdInput {
    /// Construct inputs for deriving a chunk id.
    pub fn new(
        relative_path: impl Into<Box<str>>,
        span: LineSpan,
        content: impl Into<Box<str>>,
    ) -> Self {
        Self {
            relative_path: relative_path.into(),
            span,
            content: content.into(),
        }
    }
}

/// Derive a deterministic chunk identifier from content and location.
pub fn derive_chunk_id(input: &ChunkIdInput) -> Result<ChunkId, PrimitiveError> {
    let mut hasher = Sha256::new();
    hasher.update(input.relative_path.as_bytes());
    hasher.update(b":");
    hasher.update(input.span.start_line().to_string().as_bytes());
    hasher.update(b":");
    hasher.update(input.span.end_line().to_string().as_bytes());
    hasher.update(b":");
    hasher.update(input.content.as_bytes());
    let hash = format!("{:x}", hasher.finalize());
    let hash_prefix: String = hash.chars().take(16).collect();
    let candidate = format!("chunk_{hash_prefix}");

    ChunkId::parse(candidate.as_str())
        .map_err(|_| PrimitiveError::DerivedChunkIdInvalid { candidate })
}

/// Identifier for an embedding provider implementation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EmbeddingProviderId(Box<str>);

impl EmbeddingProviderId {
    /// Parse an `EmbeddingProviderId` from user input.
    pub fn parse(input: impl AsRef<str>) -> Result<Self, PrimitiveError> {
        let raw = input.as_ref();
        let Some(trimmed) = trimmed_non_empty(raw) else {
            return Err(PrimitiveError::InvalidEmbeddingProviderId {
                input_length: raw.len(),
            });
        };

        Ok(Self(trimmed.to_owned().into_boxed_str()))
    }

    /// Access the underlying string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume and return the underlying string.
    #[must_use]
    pub fn into_inner(self) -> Box<str> {
        self.0
    }
}

impl AsRef<str> for EmbeddingProviderId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for EmbeddingProviderId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Identifier for a vector database provider implementation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct VectorDbProviderId(Box<str>);

impl VectorDbProviderId {
    /// Parse a `VectorDbProviderId` from user input.
    pub fn parse(input: impl AsRef<str>) -> Result<Self, PrimitiveError> {
        let raw = input.as_ref();
        let Some(trimmed) = trimmed_non_empty(raw) else {
            return Err(PrimitiveError::InvalidVectorDbProviderId {
                input_length: raw.len(),
            });
        };

        Ok(Self(trimmed.to_owned().into_boxed_str()))
    }

    /// Access the underlying string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume and return the underlying string.
    #[must_use]
    pub fn into_inner(self) -> Box<str> {
        self.0
    }
}

impl AsRef<str> for VectorDbProviderId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for VectorDbProviderId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Indexing mode used when deriving collection names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IndexMode {
    /// Dense-only vector indexing.
    Dense,
    /// Hybrid (dense + sparse) vector indexing.
    Hybrid,
}

impl IndexMode {
    /// Returns the canonical string representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Hybrid => "hybrid",
        }
    }
}

impl fmt::Display for IndexMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Inputs required to derive a deterministic collection name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionNamingInput {
    /// Root path of the codebase.
    pub codebase_root: PathBuf,
    /// Indexing mode that determines the name prefix.
    pub index_mode: IndexMode,
}

impl CollectionNamingInput {
    /// Construct collection naming inputs from a path and index mode.
    pub fn new(codebase_root: impl Into<PathBuf>, index_mode: IndexMode) -> Self {
        Self {
            codebase_root: codebase_root.into(),
            index_mode,
        }
    }
}

/// Derive a deterministic collection name for a codebase and index mode.
pub fn derive_collection_name(
    input: &CollectionNamingInput,
) -> Result<CollectionName, PrimitiveError> {
    let normalized = normalize_root_path(&input.codebase_root);
    let normalized = normalized.to_string_lossy();
    let digest = md5::compute(normalized.as_bytes());
    let hash = format!("{digest:x}");
    let prefix = match input.index_mode {
        IndexMode::Hybrid => "hybrid_code_chunks",
        IndexMode::Dense => "code_chunks",
    };
    let hash_prefix: String = hash.chars().take(8).collect();
    let candidate = format!("{prefix}_{hash_prefix}");

    CollectionName::parse(candidate.as_str())
        .map_err(|_| PrimitiveError::DerivedCollectionNameInvalid { candidate })
}

fn trimmed_non_empty(input: &str) -> Option<&str> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn is_valid_collection_name(candidate: &str) -> bool {
    let mut chars = candidate.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_alphabetic() {
        return false;
    }

    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn normalize_root_path(path: &Path) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn codebase_id_requires_non_empty_input() {
        let error = CodebaseId::parse("   ").err();
        assert!(matches!(
            error,
            Some(PrimitiveError::InvalidCodebaseId { .. })
        ));
    }

    #[test]
    fn collection_name_rejects_invalid_pattern() {
        let error = CollectionName::parse("bad-name").err();
        assert!(matches!(
            error,
            Some(PrimitiveError::InvalidCollectionName { .. })
        ));
    }

    #[test]
    fn collection_name_rejects_empty_input() {
        let error = CollectionName::parse("   ").err();
        assert!(matches!(
            error,
            Some(PrimitiveError::EmptyCollectionName { .. })
        ));
    }

    #[test]
    fn derive_collection_name_is_deterministic() -> Result<(), PrimitiveError> {
        let input = CollectionNamingInput::new("repo", IndexMode::Dense);
        let first = derive_collection_name(&input)?;
        let second = derive_collection_name(&input)?;

        assert_eq!(first, second);
        assert!(first.as_str().starts_with("code_chunks_"));
        Ok(())
    }

    #[test]
    fn derive_collection_name_uses_hybrid_prefix() -> Result<(), PrimitiveError> {
        let input = CollectionNamingInput::new("repo", IndexMode::Hybrid);
        let derived = derive_collection_name(&input)?;
        assert!(derived.as_str().starts_with("hybrid_code_chunks_"));
        Ok(())
    }

    proptest! {
        #[test]
        fn collection_name_accepts_valid_inputs(name in valid_collection_name()) {
            let parsed = CollectionName::parse(&name);
            prop_assert!(parsed.is_ok());
        }
    }

    fn valid_collection_name() -> impl Strategy<Value = String> {
        let start_chars: Vec<char> = ('a'..='z').chain('A'..='Z').collect();
        let mut rest_chars: Vec<char> = ('a'..='z').chain('A'..='Z').chain('0'..='9').collect();
        rest_chars.push('_');

        let start = prop::sample::select(start_chars);
        let rest = prop::collection::vec(prop::sample::select(rest_chars), 0..24);

        (start, rest).prop_map(|(start, rest)| {
            let mut name = String::new();
            name.push(start);
            for ch in rest {
                name.push(ch);
            }
            name
        })
    }
}
