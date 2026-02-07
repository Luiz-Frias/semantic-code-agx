//! Contract-style tests for port traits using in-memory adapters.

use semantic_code_domain::{
    CollectionName, EmbeddingProviderId, Language, LineSpan, VectorDbProviderId,
    VectorDocumentMetadata,
};
use semantic_code_ports::{
    DetectDimensionOptions, EmbedBatchRequest, EmbeddingPort, EmbeddingProviderInfo, LoggerPort,
    TelemetryPort, VectorDbPort, VectorDbProviderInfo, VectorDocumentForInsert,
    VectorSearchOptions, VectorSearchRequest,
};
use semantic_code_shared::{ErrorEnvelope, RequestContext, Result};
use semantic_code_testkit::in_memory::{
    InMemoryEmbeddingFixed, InMemoryVectorDbFixed, NoopLogger, NoopTelemetry,
};
use std::sync::Arc;

#[tokio::test]
async fn embedding_port_contract_smoke() -> Result<()> {
    let ctx = RequestContext::new_request();
    let provider = EmbeddingProviderInfo {
        id: EmbeddingProviderId::parse("test").map_err(ErrorEnvelope::from)?,
        name: "test".into(),
    };
    let port = InMemoryEmbeddingFixed::<8>::new(provider);

    let dim = port
        .detect_dimension(&ctx, DetectDimensionOptions::default().into())
        .await?;
    assert_eq!(dim, 8);

    let vector = port.embed(&ctx, "hello".into()).await?;
    assert_eq!(vector.as_slice().len(), 8);

    let vectors = port
        .embed_batch(
            &ctx,
            EmbedBatchRequest::from(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        )
        .await?;
    assert_eq!(vectors.len(), 3);

    Ok(())
}

#[tokio::test]
async fn vectordb_port_contract_smoke() -> Result<()> {
    let ctx = RequestContext::new_request();
    let provider = VectorDbProviderInfo {
        id: VectorDbProviderId::parse("local").map_err(ErrorEnvelope::from)?,
        name: "local".into(),
    };
    let port = InMemoryVectorDbFixed::<4>::new(provider);

    let collection = CollectionName::parse("code_chunks_contract").map_err(ErrorEnvelope::from)?;
    port.create_collection(&ctx, collection.clone(), 4, None)
        .await?;

    let meta = VectorDocumentMetadata {
        relative_path: "src/main.rs".into(),
        language: Some(Language::Rust),
        file_extension: Some("rs".into()),
        span: LineSpan::new(1, 3).map_err(ErrorEnvelope::from)?,
        node_kind: None,
    };

    port.insert(
        &ctx,
        collection.clone(),
        vec![
            VectorDocumentForInsert {
                id: "chunk_a".into(),
                vector: Arc::from(vec![1.0, 0.0, 0.0, 0.0]),
                content: "fn a() {}".into(),
                metadata: meta.clone(),
            },
            VectorDocumentForInsert {
                id: "chunk_b".into(),
                vector: Arc::from(vec![0.0, 1.0, 0.0, 0.0]),
                content: "fn b() {}".into(),
                metadata: meta,
            },
        ],
    )
    .await?;

    let results = port
        .search(
            &ctx,
            VectorSearchRequest {
                collection_name: collection,
                query_vector: Arc::from(vec![1.0, 0.0, 0.0, 0.0]),
                options: VectorSearchOptions {
                    top_k: Some(1),
                    filter_expr: None,
                    threshold: None,
                },
            },
        )
        .await?;

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].document.id.as_ref(), "chunk_a");
    Ok(())
}

#[test]
fn noop_observability_ports_do_not_panic() {
    let logger = NoopLogger::default();
    let _child = logger.child(Default::default());

    let telemetry = NoopTelemetry::default();
    telemetry.increment_counter("counter", 1, None);
    telemetry.record_timer_ms("timer", 10, None);
    telemetry.start_timer("timer2", None).stop();
}
