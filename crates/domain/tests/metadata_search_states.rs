//! Integration coverage for metadata, search, and state domain types.

use semantic_code_domain::{
    ChunkId, ChunkMetadata, DocumentId, DocumentMetadata, IndexStatus, Language, LineSpan,
    ProgressEvent, SearchFilter, SearchOptions, SearchQuery, SearchResult, SearchResultKey,
    VectorDocumentMetadata,
};
use std::error::Error;

#[test]
fn metadata_search_and_state_types_compose() -> Result<(), Box<dyn Error>> {
    let document_id = DocumentId::parse("doc-1")?;
    let chunk_id = ChunkId::parse("chunk-1")?;
    let span = LineSpan::new(1, 2)?;

    let document = DocumentMetadata::builder(document_id.clone(), "src/lib.rs")?
        .language(Language::Rust)
        .file_extension("rs")
        .build();
    document.validate()?;

    let chunk =
        ChunkMetadata::builder(chunk_id, document_id, document.relative_path.as_ref(), span)?
            .language(Language::Rust)
            .node_kind("function")
            .build();
    chunk.validate()?;

    let vector_metadata = VectorDocumentMetadata::from(&chunk);
    vector_metadata.validate()?;

    let query = SearchQuery {
        query: "semantic search".into(),
    };
    let options = SearchOptions {
        top_k: 5,
        threshold: Some(0.5),
        filter: Some(SearchFilter {
            filter_expr: Some("language == 'rust'".into()),
        }),
        include_content: Some(true),
    };

    let result = SearchResult {
        key: SearchResultKey {
            relative_path: chunk.relative_path.clone(),
            span,
        },
        content: Some("fn main() {}".into()),
        language: Some(Language::Rust),
        score: 0.9,
    };

    let status = IndexStatus::Indexed;
    let event = ProgressEvent::status(status);

    assert_eq!(query.query.as_ref(), "semantic search");
    assert_eq!(options.top_k, 5);
    assert_eq!(result.key.relative_path.as_ref(), "src/lib.rs");
    assert_eq!(event, ProgressEvent::status(IndexStatus::Indexed));
    Ok(())
}
