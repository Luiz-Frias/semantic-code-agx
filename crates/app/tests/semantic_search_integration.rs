//! Integration tests for the semantic search use case.

use semantic_code_app::{SemanticSearchDeps, SemanticSearchInput, semantic_search};
use semantic_code_domain::{
    ChunkIdInput, CollectionName, EmbeddingProviderId, IndexMode, Language, LineSpan,
    VectorDbProviderId, VectorDocumentMetadata, derive_chunk_id,
};
use semantic_code_ports::{
    DetectDimensionOptions, EmbeddingPort, EmbeddingProviderInfo, VectorDbPort,
    VectorDbProviderInfo, VectorDocumentForInsert,
};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use semantic_code_testkit::in_memory::{InMemoryEmbeddingFixed, InMemoryVectorDbFixed};
use std::path::PathBuf;
use std::sync::Arc;

#[tokio::test]
async fn search_fixture_repo_with_in_memory_adapters() -> Result<()> {
    let fixture_root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testkit/fixtures/search/basic");
    let relative_path = "src/lib.rs";
    let content =
        std::fs::read_to_string(fixture_root.join(relative_path)).map_err(ErrorEnvelope::from)?;

    let embedding = Arc::new(InMemoryEmbeddingFixed::<8>::new(EmbeddingProviderInfo {
        id: EmbeddingProviderId::parse("openai").map_err(ErrorEnvelope::from)?,
        name: "in-memory".into(),
    }));
    let vectordb = Arc::new(InMemoryVectorDbFixed::<8>::new(VectorDbProviderInfo {
        id: VectorDbProviderId::parse("milvus_grpc").map_err(ErrorEnvelope::from)?,
        name: "in-memory".into(),
    }));

    let ctx = RequestContext::new_request();
    let dimension = embedding
        .detect_dimension(&ctx, DetectDimensionOptions::default().into())
        .await?;
    let collection = CollectionName::parse("code_chunks_search").map_err(ErrorEnvelope::from)?;
    vectordb
        .create_collection(&ctx, collection.clone(), dimension, None)
        .await?;

    let span = span_for(&content)?;
    let chunk_id = derive_chunk_id(&ChunkIdInput::new(relative_path, span, content.as_str()))
        .map_err(ErrorEnvelope::from)?;
    let embedding_vector = embedding
        .embed(&ctx, content.clone().into_boxed_str().into())
        .await?;

    vectordb
        .insert(
            &ctx,
            collection.clone(),
            vec![VectorDocumentForInsert {
                id: chunk_id.into_inner(),
                vector: embedding_vector.into_vector(),
                content: content.clone().into_boxed_str(),
                metadata: VectorDocumentMetadata {
                    relative_path: relative_path.into(),
                    language: Some(Language::Rust),
                    file_extension: Some("rs".into()),
                    span,
                    node_kind: None,
                },
            }],
        )
        .await?;

    let deps = SemanticSearchDeps {
        embedding,
        vectordb,
        logger: None,
        telemetry: None,
    };

    let input = SemanticSearchInput {
        codebase_root: fixture_root.to_string_lossy().to_string().into_boxed_str(),
        collection_name: collection,
        index_mode: IndexMode::Dense,
        query: "needle".into(),
        top_k: Some(5),
        threshold: Some(0.0),
    };

    let results = semantic_search(&ctx, &deps, input).await?;
    assert!(
        results
            .iter()
            .any(|result| result.key.relative_path.as_ref() == relative_path),
        "expected results to include fixture path"
    );

    Ok(())
}

fn span_for(content: &str) -> Result<LineSpan> {
    let line_count = content.lines().count().max(1);
    let line_count = u32::try_from(line_count).map_err(|_| {
        ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            "line count overflow",
            ErrorClass::NonRetriable,
        )
    })?;
    LineSpan::new(1, line_count).map_err(ErrorEnvelope::from)
}
