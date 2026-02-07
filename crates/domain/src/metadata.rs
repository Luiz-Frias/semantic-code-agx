//! Domain metadata types for documents and chunks.

use crate::{ChunkId, DocumentId, Language, LineSpan};
use semantic_code_shared::{ErrorCode, ErrorEnvelope};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Validation failures for domain metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetadataError {
    /// Relative paths must be non-empty after trimming.
    EmptyRelativePath {
        /// Length of the raw input before trimming.
        input_length: usize,
    },
    /// Line spans must be valid (1-indexed, start <= end).
    InvalidLineSpan {
        /// Starting line (1-indexed).
        start_line: u32,
        /// Ending line (1-indexed).
        end_line: u32,
    },
}

impl MetadataError {
    fn error_code(&self) -> ErrorCode {
        match self {
            Self::EmptyRelativePath { .. } => ErrorCode::new("domain", "invalid_relative_path"),
            Self::InvalidLineSpan { .. } => ErrorCode::new("domain", "invalid_line_span"),
        }
    }
}

impl fmt::Display for MetadataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRelativePath { .. } => formatter.write_str("relativePath must be non-empty"),
            Self::InvalidLineSpan { .. } => {
                formatter.write_str("span startLine/endLine must be valid")
            },
        }
    }
}

impl std::error::Error for MetadataError {}

impl From<MetadataError> for ErrorEnvelope {
    fn from(error: MetadataError) -> Self {
        let mut envelope = Self::expected(error.error_code(), error.to_string());

        match error {
            MetadataError::EmptyRelativePath { input_length } => {
                envelope = envelope.with_metadata("input_length", input_length.to_string());
            },
            MetadataError::InvalidLineSpan {
                start_line,
                end_line,
            } => {
                envelope = envelope
                    .with_metadata("start_line", start_line.to_string())
                    .with_metadata("end_line", end_line.to_string());
            },
        }

        envelope
    }
}

/// Metadata describing a source document in the domain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentMetadata {
    /// Document identifier.
    pub document_id: DocumentId,
    /// Stable logical path identifier (not necessarily an absolute filesystem path).
    pub relative_path: Box<str>,
    /// Optional language hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<Language>,
    /// Optional file extension hint (without leading dot).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_extension: Option<Box<str>>,
}

impl DocumentMetadata {
    /// Build metadata from required fields.
    pub fn builder(
        document_id: DocumentId,
        relative_path: impl AsRef<str>,
    ) -> Result<DocumentMetadataBuilder, MetadataError> {
        let relative_path = parse_relative_path(relative_path.as_ref())?;
        Ok(DocumentMetadataBuilder {
            document_id,
            relative_path,
            language: None,
            file_extension: None,
        })
    }

    /// Validate metadata invariants.
    pub fn validate(&self) -> Result<(), MetadataError> {
        ensure_non_empty(self.relative_path.as_ref())?;
        Ok(())
    }
}

/// Builder for `DocumentMetadata`.
#[derive(Debug, Clone)]
pub struct DocumentMetadataBuilder {
    document_id: DocumentId,
    relative_path: Box<str>,
    language: Option<Language>,
    file_extension: Option<Box<str>>,
}

impl DocumentMetadataBuilder {
    /// Set the language hint.
    #[must_use]
    pub const fn language(mut self, language: Language) -> Self {
        self.language = Some(language);
        self
    }

    /// Set the file extension hint (without leading dot).
    #[must_use]
    pub fn file_extension(mut self, extension: impl AsRef<str>) -> Self {
        self.file_extension = normalize_optional_string(Some(extension.as_ref()));
        self
    }

    /// Build a validated `DocumentMetadata`.
    #[must_use]
    pub fn build(self) -> DocumentMetadata {
        DocumentMetadata {
            document_id: self.document_id,
            relative_path: self.relative_path,
            language: self.language,
            file_extension: self.file_extension,
        }
    }
}

/// Metadata describing a chunk of a document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChunkMetadata {
    /// Chunk identifier.
    pub chunk_id: ChunkId,
    /// Document identifier.
    pub document_id: DocumentId,
    /// Stable logical path identifier (not necessarily an absolute filesystem path).
    pub relative_path: Box<str>,
    /// Optional language hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<Language>,
    /// Optional file extension hint (without leading dot).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_extension: Option<Box<str>>,
    /// Line span for the chunk.
    pub span: LineSpan,
    /// Optional structural hint (e.g. AST node kind).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_kind: Option<Box<str>>,
}

impl ChunkMetadata {
    /// Build metadata from required fields.
    pub fn builder(
        chunk_id: ChunkId,
        document_id: DocumentId,
        relative_path: impl AsRef<str>,
        span: LineSpan,
    ) -> Result<ChunkMetadataBuilder, MetadataError> {
        let relative_path = parse_relative_path(relative_path.as_ref())?;
        validate_span(span)?;
        Ok(ChunkMetadataBuilder {
            chunk_id,
            document_id,
            relative_path,
            language: None,
            file_extension: None,
            span,
            node_kind: None,
        })
    }

    /// Validate metadata invariants.
    pub fn validate(&self) -> Result<(), MetadataError> {
        ensure_non_empty(self.relative_path.as_ref())?;
        validate_span(self.span)?;
        Ok(())
    }
}

/// Builder for `ChunkMetadata`.
#[derive(Debug, Clone)]
pub struct ChunkMetadataBuilder {
    chunk_id: ChunkId,
    document_id: DocumentId,
    relative_path: Box<str>,
    language: Option<Language>,
    file_extension: Option<Box<str>>,
    span: LineSpan,
    node_kind: Option<Box<str>>,
}

impl ChunkMetadataBuilder {
    /// Set the language hint.
    #[must_use]
    pub const fn language(mut self, language: Language) -> Self {
        self.language = Some(language);
        self
    }

    /// Set the file extension hint (without leading dot).
    #[must_use]
    pub fn file_extension(mut self, extension: impl AsRef<str>) -> Self {
        self.file_extension = normalize_optional_string(Some(extension.as_ref()));
        self
    }

    /// Set the node kind hint.
    #[must_use]
    pub fn node_kind(mut self, node_kind: impl AsRef<str>) -> Self {
        self.node_kind = normalize_optional_string(Some(node_kind.as_ref()));
        self
    }

    /// Build a validated `ChunkMetadata`.
    #[must_use]
    pub fn build(self) -> ChunkMetadata {
        ChunkMetadata {
            chunk_id: self.chunk_id,
            document_id: self.document_id,
            relative_path: self.relative_path,
            language: self.language,
            file_extension: self.file_extension,
            span: self.span,
            node_kind: self.node_kind,
        }
    }
}

/// Metadata stored alongside vector documents in the vector DB.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorDocumentMetadata {
    /// Stable logical path identifier (not necessarily an absolute filesystem path).
    pub relative_path: Box<str>,
    /// Optional language hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<Language>,
    /// Optional file extension hint (without leading dot).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_extension: Option<Box<str>>,
    /// Line span for the chunk.
    pub span: LineSpan,
    /// Optional structural hint (e.g. AST node kind).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_kind: Option<Box<str>>,
}

impl VectorDocumentMetadata {
    /// Validate metadata invariants.
    pub fn validate(&self) -> Result<(), MetadataError> {
        ensure_non_empty(self.relative_path.as_ref())?;
        validate_span(self.span)?;
        Ok(())
    }
}

impl From<&ChunkMetadata> for VectorDocumentMetadata {
    fn from(metadata: &ChunkMetadata) -> Self {
        Self {
            relative_path: metadata.relative_path.clone(),
            language: metadata.language,
            file_extension: metadata.file_extension.clone(),
            span: metadata.span,
            node_kind: metadata.node_kind.clone(),
        }
    }
}

fn parse_relative_path(path: &str) -> Result<Box<str>, MetadataError> {
    let trimmed = ensure_non_empty(path)?;
    Ok(trimmed.to_owned().into_boxed_str())
}

fn ensure_non_empty(input: &str) -> Result<&str, MetadataError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(MetadataError::EmptyRelativePath {
            input_length: input.len(),
        });
    }
    Ok(trimmed)
}

fn normalize_optional_string<S: AsRef<str>>(value: Option<S>) -> Option<Box<str>> {
    value.and_then(|raw| {
        let trimmed = raw.as_ref().trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_owned().into_boxed_str())
        }
    })
}

const fn validate_span(span: LineSpan) -> Result<(), MetadataError> {
    let start_line = span.start_line();
    let end_line = span.end_line();
    if start_line == 0 || end_line == 0 || start_line > end_line {
        return Err(MetadataError::InvalidLineSpan {
            start_line,
            end_line,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChunkId, DocumentId, Language, LineSpan};
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseError;
    use std::error::Error;

    #[test]
    fn metadata_builder_validates_relative_path() -> Result<(), Box<dyn Error>> {
        let document_id = DocumentId::parse("doc-1")?;
        let metadata = DocumentMetadata::builder(document_id, "src/lib.rs")?
            .language(Language::Rust)
            .file_extension("rs")
            .build();

        metadata.validate()?;
        Ok(())
    }

    #[test]
    fn metadata_builder_rejects_empty_paths() -> Result<(), Box<dyn Error>> {
        let document_id = DocumentId::parse("doc-1")?;
        let error = DocumentMetadata::builder(document_id, "   ").err();
        assert!(matches!(
            error,
            Some(MetadataError::EmptyRelativePath { .. })
        ));
        Ok(())
    }

    #[test]
    fn metadata_validation_rejects_invalid_span() -> Result<(), Box<dyn Error>> {
        let payload = serde_json::json!({
            "chunkId": "chunk-1",
            "documentId": "doc-1",
            "relativePath": "src/lib.rs",
            "span": { "startLine": 0, "endLine": 2 }
        });

        let metadata: ChunkMetadata = serde_json::from_value(payload)?;
        let error = metadata.validate().err();
        assert!(matches!(error, Some(MetadataError::InvalidLineSpan { .. })));
        Ok(())
    }

    #[test]
    fn metadata_serializes_with_camel_case() -> Result<(), Box<dyn Error>> {
        let document_id = DocumentId::parse("doc-1")?;
        let chunk_id = ChunkId::parse("chunk-1")?;
        let span = LineSpan::new(1, 2)?;
        let chunk = ChunkMetadata::builder(chunk_id, document_id, "src/lib.rs", span)?
            .language(Language::Rust)
            .file_extension("rs")
            .node_kind("function")
            .build();

        let value = serde_json::to_value(&chunk)?;
        let expected = serde_json::json!({
            "chunkId": "chunk-1",
            "documentId": "doc-1",
            "relativePath": "src/lib.rs",
            "language": "rust",
            "fileExtension": "rs",
            "span": { "startLine": 1, "endLine": 2 },
            "nodeKind": "function"
        });
        assert_eq!(value, expected);
        Ok(())
    }

    proptest! {
        #[test]
        fn document_metadata_accepts_valid_paths(path in valid_relative_path()) {
            let document_id = DocumentId::parse("doc-1")
                .map_err(|_| TestCaseError::fail("doc id parse failed"))?;
            let metadata = DocumentMetadata::builder(document_id, path.as_str())
                .map_err(|_| TestCaseError::fail("metadata builder failed"))?
                .build();
            prop_assert!(metadata.validate().is_ok());
        }
    }

    fn valid_relative_path() -> impl Strategy<Value = String> {
        "[a-zA-Z0-9_./-]{1,64}".prop_map(|path| path.trim().to_string())
    }
}
