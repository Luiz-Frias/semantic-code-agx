//! Phase 02 domain E2E tests.

use semantic_code_domain::{
    ChunkIdInput, ChunkMetadata, CollectionNamingInput, DocumentId, DocumentMetadata, IndexMode,
    Language, LineSpan, SearchOptions, SearchQuery, SearchResult, SearchResultKey,
    VectorDocumentMetadata, compare_search_results, derive_chunk_id, derive_codebase_id,
    derive_collection_name,
};
use std::error::Error;

#[test]
fn phase2_domain_end_to_end() -> Result<(), Box<dyn Error>> {
    let codebase_root = "/tmp/example-codebase";
    let codebase_id = derive_codebase_id(codebase_root)?;
    let codebase_id_again = derive_codebase_id(codebase_root)?;
    assert_eq!(codebase_id, codebase_id_again);
    assert!(codebase_id.as_str().starts_with("codebase_"));

    let collection_input = CollectionNamingInput::new(codebase_root, IndexMode::Dense);
    let collection_name = derive_collection_name(&collection_input)?;
    let collection_name_again = derive_collection_name(&collection_input)?;
    assert_eq!(collection_name, collection_name_again);
    assert!(collection_name.as_str().starts_with("code_chunks_"));

    let document_id = DocumentId::parse("doc-1")?;
    let span = LineSpan::new(1, 3)?;
    let chunk_id_input = ChunkIdInput::new(
        "src/lib.rs",
        span.start_line(),
        span.end_line(),
        "fn main() {}\n",
    );
    let chunk_id = derive_chunk_id(&chunk_id_input)?;
    let chunk_id_again = derive_chunk_id(&chunk_id_input)?;
    assert_eq!(chunk_id, chunk_id_again);

    let document_metadata = DocumentMetadata::builder(document_id.clone(), "src/lib.rs")?
        .language(Language::Rust)
        .file_extension("rs")
        .build();
    document_metadata.validate()?;

    let chunk_metadata = ChunkMetadata::builder(
        chunk_id,
        document_id,
        "src/lib.rs",
        span,
    )?
    .language(Language::Rust)
    .file_extension("rs")
    .node_kind("function")
    .build();
    chunk_metadata.validate()?;

    let vector_metadata = VectorDocumentMetadata::from(&chunk_metadata);
    vector_metadata.validate()?;

    let query = SearchQuery {
        query: "search".into(),
    };
    let options = SearchOptions {
        top_k: 3,
        threshold: Some(0.5_f32),
        filter: None,
        include_content: Some(true),
    };
    assert!(!query.query.is_empty());
    assert_eq!(options.top_k, 3);

    let mut results = vec![
        SearchResult {
            key: SearchResultKey {
                relative_path: "b.rs".into(),
                span: LineSpan::new(5, 7)?,
            },
            content: Some("b".into()),
            language: Some(Language::Rust),
            score: 0.9_f32,
        },
        SearchResult {
            key: SearchResultKey {
                relative_path: "a.rs".into(),
                span: LineSpan::new(10, 12)?,
            },
            content: Some("a2".into()),
            language: Some(Language::Rust),
            score: 0.9_f32,
        },
        SearchResult {
            key: SearchResultKey {
                relative_path: "a.rs".into(),
                span: LineSpan::new(2, 3)?,
            },
            content: Some("a1".into()),
            language: Some(Language::Rust),
            score: 0.9_f32,
        },
    ];

    results.sort_by(compare_search_results);

    let ordered: Vec<(&str, u32)> = results
        .iter()
        .map(|result| (result.key.relative_path.as_ref(), result.key.span.start_line()))
        .collect();
    assert_eq!(ordered, vec![("a.rs", 2), ("a.rs", 10), ("b.rs", 5)]);

    Ok(())
}
