//! # semantic-code-domain
//!
//! Domain entities, primitives, and value objects for semantic code search.
//!
//! This crate contains the core domain model with no infrastructure dependencies:
//!
//! - **Primitives** - `CodebaseId`, `DocumentId`, `ChunkId`, etc.
//! - **Spans** - `LineSpan`, `Language`
//! - **Metadata** - `DocumentMetadata`, `ChunkMetadata` (Phase 02)
//! - **Search** - `SearchQuery`, `SearchResult`, `SearchOptions` (Phase 02)
//! - **State** - `IndexingState`, `ProgressEvent` (Phase 02)
//!
//! ## Dependency Rules
//!
//! - Depends only on `shared` crate
//! - No infrastructure or adapter dependencies
//! - Pure domain logic with no I/O

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]

// Re-export shared types for convenience
pub use semantic_code_shared::shared_crate_version;

// =============================================================================
// DOMAIN MODULES
// =============================================================================

pub mod chunk;
pub mod metadata;
pub mod primitives;
pub mod search;
pub mod spans;
pub mod states;

pub use chunk::{Chunk, ChunkError, MAX_CHUNK_CHARS};
pub use metadata::{ChunkMetadata, DocumentMetadata, MetadataError, VectorDocumentMetadata};
pub use primitives::{
    ChunkId, ChunkIdInput, CodebaseId, CollectionName, CollectionNamingInput, DocumentId,
    EmbeddingProviderId, IndexMode, PrimitiveError, VectorDbProviderId, derive_chunk_id,
    derive_codebase_id, derive_collection_name,
};
pub use search::{
    SearchFilter, SearchOptions, SearchQuery, SearchResult, SearchResultKey, compare_search_results,
};
pub use spans::{Language, LineSpan};
pub use states::{IndexStatus, IndexingState, ProgressEvent};

/// Returns the domain crate version.
#[must_use]
pub const fn domain_crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_crate_compiles() {
        let version = domain_crate_version();
        assert!(!version.is_empty());
    }

    #[test]
    fn domain_depends_on_shared() {
        // Verify we can access shared crate
        let shared_version = shared_crate_version();
        assert!(!shared_version.is_empty());
    }
}
