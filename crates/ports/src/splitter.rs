//! Splitter / chunking boundary contract.

use crate::BoxFuture;
use semantic_code_domain::{Language, LineSpan};
use semantic_code_shared::{RequestContext, Result};

/// A code chunk produced by a splitter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeChunk {
    /// Chunk content.
    pub content: Box<str>,
    /// Line span (1-indexed).
    pub span: LineSpan,
    /// Optional language hint.
    pub language: Option<Language>,
    /// Optional file path hint.
    pub file_path: Option<Box<str>>,
}

/// Options for splitting.
#[derive(Debug, Clone, Default)]
pub struct SplitOptions {
    /// Optional file path hint.
    pub file_path: Option<Box<str>>,
}

/// Boundary contract for chunking/splitting content for indexing.
pub trait SplitterPort: Send + Sync {
    /// Split code into chunks using the provided language hint.
    fn split(
        &self,
        ctx: &RequestContext,
        code: Box<str>,
        language: Language,
        options: SplitOptions,
    ) -> BoxFuture<'_, Result<Vec<CodeChunk>>>;

    /// Configure the target chunk size for this splitter instance.
    fn set_chunk_size(&self, chunk_size: usize);

    /// Configure the target overlap between adjacent chunks.
    fn set_chunk_overlap(&self, chunk_overlap: usize);
}
