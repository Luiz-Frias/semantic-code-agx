//! Semantic search use-case (dense + hybrid).

use semantic_code_domain::{
    CollectionName, IndexMode, SearchResult, SearchResultKey, compare_search_results,
};
use semantic_code_ports::{
    EmbeddingPort, HybridSearchBatchRequest, HybridSearchData, HybridSearchOptions,
    HybridSearchRequest, LogFields, LoggerPort, RerankStrategy, RerankStrategyKind, TelemetryPort,
    TelemetryTags, VectorDbPort, VectorSearchOptions, VectorSearchRequest,
};
use semantic_code_shared::{ErrorEnvelope, RequestContext, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

/// Input payload for semantic search.
#[derive(Debug, Clone)]
pub struct SemanticSearchInput {
    /// Root identifier for the codebase (for logging only).
    pub codebase_root: Box<str>,
    /// Target collection name.
    pub collection_name: CollectionName,
    /// Index mode for the target collection.
    pub index_mode: IndexMode,
    /// Query text to embed.
    pub query: Box<str>,
    /// Optional top-k override (defaults to 5).
    pub top_k: Option<u32>,
    /// Optional score threshold (defaults to 0.5 for dense).
    pub threshold: Option<f32>,
}

/// Dependencies required by semantic search.
#[derive(Clone)]
pub struct SemanticSearchDeps {
    /// Embedding adapter.
    pub embedding: Arc<dyn EmbeddingPort>,
    /// Vector database adapter.
    pub vectordb: Arc<dyn VectorDbPort>,
    /// Optional logger.
    pub logger: Option<Arc<dyn LoggerPort>>,
    /// Optional telemetry sink.
    pub telemetry: Option<Arc<dyn TelemetryPort>>,
}

/// Execute semantic search for the given input.
pub async fn semantic_search(
    ctx: &RequestContext,
    deps: &SemanticSearchDeps,
    input: SemanticSearchInput,
) -> Result<Vec<SearchResult>> {
    let started_at = Instant::now();
    let total_tags = tags_index_mode(input.index_mode);
    let total_timer = deps
        .telemetry
        .as_ref()
        .map(|telemetry| telemetry.start_timer("backend.search.total", Some(&total_tags)));

    let top_k = input.top_k.unwrap_or(5).max(1);
    let threshold = input.threshold.unwrap_or(0.5);

    if let Some(logger) = deps.logger.as_ref() {
        logger.info(
            "backend.search.start",
            "Semantic search started",
            Some(log_fields_start(&input, top_k, threshold)),
        );
    }

    let result = run_search(ctx, deps, &input, top_k, threshold, started_at).await;

    if let Some(timer) = total_timer.as_ref() {
        timer.stop();
    }

    match result {
        Ok(results) => Ok(results),
        Err(error) => {
            let duration_ms = duration_ms(started_at);
            if error.is_cancelled() {
                if let Some(telemetry) = deps.telemetry.as_ref() {
                    telemetry.increment_counter(
                        "backend.search.aborted",
                        1,
                        Some(&tags_index_mode(input.index_mode)),
                    );
                }
                if let Some(logger) = deps.logger.as_ref() {
                    logger.info(
                        "backend.search.aborted",
                        "Semantic search aborted",
                        Some(log_fields_abort(&input, duration_ms)),
                    );
                }
            } else {
                if let Some(telemetry) = deps.telemetry.as_ref() {
                    telemetry.increment_counter(
                        "backend.search.failed",
                        1,
                        Some(&tags_index_mode(input.index_mode)),
                    );
                }
                if let Some(logger) = deps.logger.as_ref() {
                    logger.error(
                        "backend.search.failed",
                        "Semantic search failed",
                        Some(log_fields_error(&input, duration_ms, &error)),
                    );
                }
            }
            Err(error)
        },
    }
}

async fn run_search(
    ctx: &RequestContext,
    deps: &SemanticSearchDeps,
    input: &SemanticSearchInput,
    top_k: u32,
    threshold: f32,
    started_at: Instant,
) -> Result<Vec<SearchResult>> {
    ctx.ensure_not_cancelled("semantic_search.start")?;

    let has_collection = deps
        .vectordb
        .has_collection(ctx, input.collection_name.clone())
        .await?;
    if !has_collection {
        log_completed(deps, input, top_k, threshold, 0, started_at);
        return Ok(Vec::new());
    }

    let embedding = embed_query(ctx, deps, input).await?;
    let results = search_vectordb(ctx, deps, input, embedding, top_k, threshold).await?;
    let ordered = rerank_results(deps, input, results);

    if let Some(telemetry) = deps.telemetry.as_ref() {
        telemetry.increment_counter(
            "backend.search.executed",
            1,
            Some(&tags_index_mode(input.index_mode)),
        );
    }

    log_completed(deps, input, top_k, threshold, ordered.len(), started_at);

    Ok(ordered)
}

async fn embed_query(
    ctx: &RequestContext,
    deps: &SemanticSearchDeps,
    input: &SemanticSearchInput,
) -> Result<semantic_code_ports::EmbeddingVector> {
    ctx.ensure_not_cancelled("semantic_search.embed")?;

    let embed_tags = tags_with(
        input.index_mode,
        "providerId",
        deps.embedding.provider().id.as_str(),
    );
    let embed_timer = deps
        .telemetry
        .as_ref()
        .map(|telemetry| telemetry.start_timer("backend.search.embed", Some(&embed_tags)));
    let embedding = deps
        .embedding
        .embed(ctx, input.query.clone().into())
        .await?;
    if let Some(timer) = embed_timer.as_ref() {
        timer.stop();
    }

    Ok(embedding)
}

async fn search_vectordb(
    ctx: &RequestContext,
    deps: &SemanticSearchDeps,
    input: &SemanticSearchInput,
    embedding: semantic_code_ports::EmbeddingVector,
    top_k: u32,
    threshold: f32,
) -> Result<Vec<SearchResult>> {
    ctx.ensure_not_cancelled("semantic_search.vectordb")?;

    let method = match input.index_mode {
        IndexMode::Hybrid => "hybrid_search",
        IndexMode::Dense => "search",
    };
    let vectordb_tags = tags_with_method(
        input.index_mode,
        method,
        deps.vectordb.provider().id.as_str(),
    );
    let vectordb_timer = deps
        .telemetry
        .as_ref()
        .map(|telemetry| telemetry.start_timer("backend.search.vectordb", Some(&vectordb_tags)));

    let vector = embedding.into_vector();
    let results = match input.index_mode {
        IndexMode::Hybrid => {
            let requests = hybrid_requests(&vector, input.query.clone(), top_k);
            deps.vectordb
                .hybrid_search(
                    ctx,
                    HybridSearchBatchRequest {
                        collection_name: input.collection_name.clone(),
                        search_requests: requests,
                        options: hybrid_options(top_k),
                    },
                )
                .await?
                .into_iter()
                .map(map_hybrid_result)
                .collect::<Vec<_>>()
        },
        IndexMode::Dense => deps
            .vectordb
            .search(
                ctx,
                VectorSearchRequest {
                    collection_name: input.collection_name.clone(),
                    query_vector: vector,
                    options: VectorSearchOptions {
                        top_k: Some(top_k),
                        threshold: Some(threshold),
                        filter_expr: None,
                    },
                },
            )
            .await?
            .into_iter()
            .map(map_vector_result)
            .collect::<Vec<_>>(),
    };

    if let Some(timer) = vectordb_timer.as_ref() {
        timer.stop();
    }

    Ok(results)
}

fn rerank_results(
    deps: &SemanticSearchDeps,
    input: &SemanticSearchInput,
    mut results: Vec<SearchResult>,
) -> Vec<SearchResult> {
    let rerank_tags = tags_index_mode(input.index_mode);
    let rerank_timer = deps
        .telemetry
        .as_ref()
        .map(|telemetry| telemetry.start_timer("backend.search.rerank", Some(&rerank_tags)));

    results.sort_by(compare_search_results);

    if let Some(timer) = rerank_timer.as_ref() {
        timer.stop();
    }

    results
}

fn log_completed(
    deps: &SemanticSearchDeps,
    input: &SemanticSearchInput,
    top_k: u32,
    threshold: f32,
    results: usize,
    started_at: Instant,
) {
    if let Some(logger) = deps.logger.as_ref() {
        logger.info(
            "backend.search.completed",
            "Semantic search completed",
            Some(log_fields_completed(
                input, top_k, threshold, results, started_at,
            )),
        );
    }
}

fn duration_ms(started_at: Instant) -> u64 {
    u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn tags_index_mode(index_mode: IndexMode) -> TelemetryTags {
    let mut tags = TelemetryTags::new();
    tags.insert(
        "indexMode".to_owned().into_boxed_str(),
        index_mode.as_str().to_owned().into_boxed_str(),
    );
    tags
}

fn tags_with(index_mode: IndexMode, key: &str, value: &str) -> TelemetryTags {
    let mut tags = tags_index_mode(index_mode);
    tags.insert(
        key.to_owned().into_boxed_str(),
        value.to_owned().into_boxed_str(),
    );
    tags
}

fn tags_with_method(index_mode: IndexMode, method: &str, provider: &str) -> TelemetryTags {
    let mut tags = tags_index_mode(index_mode);
    tags.insert(
        "method".to_owned().into_boxed_str(),
        method.to_owned().into_boxed_str(),
    );
    tags.insert(
        "providerId".to_owned().into_boxed_str(),
        provider.to_owned().into_boxed_str(),
    );
    tags
}

fn log_fields_start(input: &SemanticSearchInput, top_k: u32, threshold: f32) -> LogFields {
    let mut fields = LogFields::new();
    fields.insert(
        "codebaseRoot".to_owned().into_boxed_str(),
        Value::String(input.codebase_root.as_ref().to_owned()),
    );
    fields.insert(
        "collectionName".to_owned().into_boxed_str(),
        Value::String(input.collection_name.as_str().to_owned()),
    );
    fields.insert(
        "indexMode".to_owned().into_boxed_str(),
        Value::String(input.index_mode.as_str().to_owned()),
    );
    fields.insert("topK".to_owned().into_boxed_str(), Value::from(top_k));
    fields.insert(
        "threshold".to_owned().into_boxed_str(),
        Value::from(f64::from(threshold)),
    );
    fields.insert(
        "queryLength".to_owned().into_boxed_str(),
        Value::from(input.query.len()),
    );
    fields
}

fn log_fields_completed(
    input: &SemanticSearchInput,
    top_k: u32,
    threshold: f32,
    results: usize,
    started_at: Instant,
) -> LogFields {
    let mut fields = log_fields_start(input, top_k, threshold);
    fields.insert(
        "durationMs".to_owned().into_boxed_str(),
        Value::from(duration_ms(started_at)),
    );
    fields.insert("results".to_owned().into_boxed_str(), Value::from(results));
    fields
}

fn log_fields_abort(input: &SemanticSearchInput, duration_ms: u64) -> LogFields {
    let mut fields = LogFields::new();
    fields.insert(
        "indexMode".to_owned().into_boxed_str(),
        Value::String(input.index_mode.as_str().to_owned()),
    );
    fields.insert(
        "durationMs".to_owned().into_boxed_str(),
        Value::from(duration_ms),
    );
    fields
}

fn log_fields_error(
    input: &SemanticSearchInput,
    duration_ms: u64,
    error: &ErrorEnvelope,
) -> LogFields {
    let mut fields = log_fields_abort(input, duration_ms);
    fields.insert(
        "error".to_owned().into_boxed_str(),
        Value::String(error.to_string()),
    );
    fields
}

fn hybrid_requests(dense: &Arc<[f32]>, query: Box<str>, top_k: u32) -> Vec<HybridSearchRequest> {
    let mut dense_params = BTreeMap::new();
    dense_params.insert("nprobe".to_owned().into_boxed_str(), Value::from(10));
    let mut sparse_params = BTreeMap::new();
    sparse_params.insert(
        "drop_ratio_search".to_owned().into_boxed_str(),
        Value::from(0.2),
    );

    vec![
        HybridSearchRequest {
            data: HybridSearchData::DenseVector(Arc::clone(dense)),
            anns_field: "vector".to_owned().into_boxed_str(),
            params: dense_params,
            limit: top_k,
        },
        HybridSearchRequest {
            data: HybridSearchData::SparseQuery(query),
            anns_field: "sparse_vector".to_owned().into_boxed_str(),
            params: sparse_params,
            limit: top_k,
        },
    ]
}

fn hybrid_options(top_k: u32) -> HybridSearchOptions {
    let mut params = BTreeMap::new();
    params.insert("k".to_owned().into_boxed_str(), Value::from(100));
    HybridSearchOptions {
        rerank: Some(RerankStrategy {
            strategy: RerankStrategyKind::Rrf,
            params: Some(params),
        }),
        limit: Some(top_k),
        filter_expr: None,
    }
}

fn map_vector_result(result: semantic_code_ports::VectorSearchResult) -> SearchResult {
    let metadata = result.document.metadata;
    SearchResult {
        key: SearchResultKey {
            relative_path: metadata.relative_path,
            span: metadata.span,
        },
        content: Some(result.document.content),
        language: metadata.language,
        score: result.score,
    }
}

fn map_hybrid_result(result: semantic_code_ports::HybridSearchResult) -> SearchResult {
    let metadata = result.document.metadata;
    SearchResult {
        key: SearchResultKey {
            relative_path: metadata.relative_path,
            span: metadata.span,
        },
        content: Some(result.document.content),
        language: metadata.language,
        score: result.score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_domain::{EmbeddingProviderId, Language, LineSpan, VectorDbProviderId};
    use semantic_code_ports::{
        DetectDimensionRequest, EmbedBatchRequest, EmbedRequest, EmbeddingProviderInfo,
        EmbeddingVector, VectorDbProviderInfo, VectorDocument, VectorDocumentMetadata,
        VectorSearchResult,
    };
    use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, Result as SharedResult};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct TestEmbedding {
        provider: EmbeddingProviderInfo,
        vector: Arc<[f32]>,
        calls: Arc<AtomicUsize>,
    }

    impl TestEmbedding {
        fn new(vector: Vec<f32>) -> SharedResult<Self> {
            let provider = EmbeddingProviderInfo {
                id: EmbeddingProviderId::parse("openai").map_err(ErrorEnvelope::from)?,
                name: "test".into(),
            };
            Ok(Self {
                provider,
                vector: Arc::from(vector),
                calls: Arc::new(AtomicUsize::new(0)),
            })
        }
    }

    impl EmbeddingPort for TestEmbedding {
        fn provider(&self) -> &EmbeddingProviderInfo {
            &self.provider
        }

        fn detect_dimension(
            &self,
            _ctx: &RequestContext,
            _request: DetectDimensionRequest,
        ) -> semantic_code_ports::BoxFuture<'_, SharedResult<u32>> {
            let dim = u32::try_from(self.vector.len()).map_err(|_| {
                ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    "embedding dimension overflow",
                    ErrorClass::NonRetriable,
                )
            });
            Box::pin(async move { dim })
        }

        fn embed(
            &self,
            _ctx: &RequestContext,
            _request: EmbedRequest,
        ) -> semantic_code_ports::BoxFuture<'_, SharedResult<EmbeddingVector>> {
            let vector = Arc::clone(&self.vector);
            let calls = self.calls.clone();
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(EmbeddingVector::new(vector))
            })
        }

        fn embed_batch(
            &self,
            _ctx: &RequestContext,
            _request: EmbedBatchRequest,
        ) -> semantic_code_ports::BoxFuture<'_, SharedResult<Vec<EmbeddingVector>>> {
            Box::pin(async move {
                Err(ErrorEnvelope::expected(
                    ErrorCode::invalid_input(),
                    "embed_batch not used in tests",
                ))
            })
        }
    }

    #[derive(Clone)]
    struct TestVectorDb {
        provider: VectorDbProviderInfo,
        has_collection: bool,
        search_results: Vec<VectorSearchResult>,
        hybrid_requests: Arc<Mutex<Vec<HybridSearchRequest>>>,
        hybrid_options: Arc<Mutex<Option<HybridSearchOptions>>>,
        search_calls: Arc<AtomicUsize>,
        last_search_options: Arc<Mutex<Option<VectorSearchOptions>>>,
    }

    impl TestVectorDb {
        fn new(search_results: Vec<VectorSearchResult>) -> SharedResult<Self> {
            let provider = VectorDbProviderInfo {
                id: VectorDbProviderId::parse("milvus_grpc").map_err(ErrorEnvelope::from)?,
                name: "test".into(),
            };
            Ok(Self {
                provider,
                has_collection: true,
                search_results,
                hybrid_requests: Arc::new(Mutex::new(Vec::new())),
                hybrid_options: Arc::new(Mutex::new(None)),
                search_calls: Arc::new(AtomicUsize::new(0)),
                last_search_options: Arc::new(Mutex::new(None)),
            })
        }

        fn take_hybrid_requests(&self) -> SharedResult<Vec<HybridSearchRequest>> {
            let mut guard = self.hybrid_requests.lock().map_err(|_| {
                ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    "hybrid request lock poisoned",
                    ErrorClass::NonRetriable,
                )
            })?;
            Ok(std::mem::take(&mut *guard))
        }

        fn last_search_options(&self) -> SharedResult<Option<VectorSearchOptions>> {
            let guard = self.last_search_options.lock().map_err(|_| {
                ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    "search options lock poisoned",
                    ErrorClass::NonRetriable,
                )
            })?;
            Ok(guard.clone())
        }
    }

    impl VectorDbPort for TestVectorDb {
        fn provider(&self) -> &VectorDbProviderInfo {
            &self.provider
        }

        fn create_collection(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _dimension: u32,
            _description: Option<Box<str>>,
        ) -> semantic_code_ports::BoxFuture<'_, SharedResult<()>> {
            Box::pin(async move { Ok(()) })
        }

        fn create_hybrid_collection(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _dimension: u32,
            _description: Option<Box<str>>,
        ) -> semantic_code_ports::BoxFuture<'_, SharedResult<()>> {
            Box::pin(async move { Ok(()) })
        }

        fn drop_collection(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
        ) -> semantic_code_ports::BoxFuture<'_, SharedResult<()>> {
            Box::pin(async move { Ok(()) })
        }

        fn has_collection(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
        ) -> semantic_code_ports::BoxFuture<'_, SharedResult<bool>> {
            let has_collection = self.has_collection;
            Box::pin(async move { Ok(has_collection) })
        }

        fn list_collections(
            &self,
            _ctx: &RequestContext,
        ) -> semantic_code_ports::BoxFuture<'_, SharedResult<Vec<CollectionName>>> {
            Box::pin(async move { Ok(Vec::new()) })
        }

        fn insert(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _documents: Vec<semantic_code_ports::VectorDocumentForInsert>,
        ) -> semantic_code_ports::BoxFuture<'_, SharedResult<()>> {
            Box::pin(async move { Ok(()) })
        }

        fn insert_hybrid(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _documents: Vec<semantic_code_ports::VectorDocumentForInsert>,
        ) -> semantic_code_ports::BoxFuture<'_, SharedResult<()>> {
            Box::pin(async move { Ok(()) })
        }

        fn search(
            &self,
            _ctx: &RequestContext,
            request: VectorSearchRequest,
        ) -> semantic_code_ports::BoxFuture<'_, SharedResult<Vec<VectorSearchResult>>> {
            let results = self.search_results.clone();
            let last_search_options = self.last_search_options.clone();
            let calls = self.search_calls.clone();
            let VectorSearchRequest { options, .. } = request;
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                let mut guard = last_search_options.lock().map_err(|_| {
                    ErrorEnvelope::unexpected(
                        ErrorCode::internal(),
                        "search options lock poisoned",
                        ErrorClass::NonRetriable,
                    )
                })?;
                *guard = Some(options);
                Ok(results)
            })
        }

        fn hybrid_search(
            &self,
            _ctx: &RequestContext,
            request: HybridSearchBatchRequest,
        ) -> semantic_code_ports::BoxFuture<
            '_,
            SharedResult<Vec<semantic_code_ports::HybridSearchResult>>,
        > {
            let store = self.hybrid_requests.clone();
            let options_store = self.hybrid_options.clone();
            let HybridSearchBatchRequest {
                search_requests,
                options,
                ..
            } = request;
            Box::pin(async move {
                let mut guard = store.lock().map_err(|_| {
                    ErrorEnvelope::unexpected(
                        ErrorCode::internal(),
                        "hybrid request lock poisoned",
                        ErrorClass::NonRetriable,
                    )
                })?;
                guard.extend(search_requests);
                let mut options_guard = options_store.lock().map_err(|_| {
                    ErrorEnvelope::unexpected(
                        ErrorCode::internal(),
                        "hybrid options lock poisoned",
                        ErrorClass::NonRetriable,
                    )
                })?;
                *options_guard = Some(options);
                Ok(Vec::new())
            })
        }

        fn delete(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _ids: Vec<Box<str>>,
        ) -> semantic_code_ports::BoxFuture<'_, SharedResult<()>> {
            Box::pin(async move { Ok(()) })
        }

        fn query(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _filter: Box<str>,
            _output_fields: Vec<Box<str>>,
            _limit: Option<u32>,
        ) -> semantic_code_ports::BoxFuture<'_, SharedResult<Vec<semantic_code_ports::VectorDbRow>>>
        {
            Box::pin(async move { Ok(Vec::new()) })
        }
    }

    fn result_doc(
        relative_path: &str,
        start: u32,
        end: u32,
        score: f32,
    ) -> SharedResult<VectorSearchResult> {
        let span = LineSpan::new(start, end).map_err(ErrorEnvelope::from)?;
        Ok(VectorSearchResult {
            document: VectorDocument {
                id: format!("chunk_{relative_path}_{start}").into_boxed_str(),
                vector: None,
                content: "content".into(),
                metadata: VectorDocumentMetadata {
                    relative_path: relative_path.into(),
                    language: Some(Language::Rust),
                    file_extension: Some("rs".into()),
                    span,
                    node_kind: None,
                },
            },
            score,
        })
    }

    #[tokio::test]
    async fn ordering_and_tiebreakers_are_deterministic() -> SharedResult<()> {
        let results = vec![
            result_doc("b.rs", 1, 2, 0.9)?,
            result_doc("a.rs", 5, 6, 0.9)?,
            result_doc("a.rs", 1, 2, 0.9)?,
            result_doc("a.rs", 1, 2, 0.95)?,
        ];
        let vectordb = Arc::new(TestVectorDb::new(results)?);
        let embedding = Arc::new(TestEmbedding::new(vec![0.1, 0.2, 0.3])?);
        let deps = SemanticSearchDeps {
            embedding,
            vectordb,
            logger: None,
            telemetry: None,
        };

        let ctx = RequestContext::new_request();
        let input = SemanticSearchInput {
            codebase_root: "/tmp".into(),
            collection_name: CollectionName::parse("code_chunks_test")
                .map_err(ErrorEnvelope::from)?,
            index_mode: IndexMode::Dense,
            query: "hello".into(),
            top_k: Some(10),
            threshold: Some(0.0),
        };

        let results = semantic_search(&ctx, &deps, input).await?;
        let ordered_paths: Vec<&str> = results
            .iter()
            .map(|result| result.key.relative_path.as_ref())
            .collect();
        assert_eq!(ordered_paths, vec!["a.rs", "a.rs", "a.rs", "b.rs"]);
        Ok(())
    }

    #[tokio::test]
    async fn threshold_is_forwarded_to_vectordb() -> SharedResult<()> {
        let vectordb = Arc::new(TestVectorDb::new(Vec::new())?);
        let embedding = Arc::new(TestEmbedding::new(vec![0.1, 0.2, 0.3])?);
        let deps = SemanticSearchDeps {
            embedding,
            vectordb: vectordb.clone(),
            logger: None,
            telemetry: None,
        };

        let ctx = RequestContext::new_request();
        let input = SemanticSearchInput {
            codebase_root: "/tmp".into(),
            collection_name: CollectionName::parse("code_chunks_test")
                .map_err(ErrorEnvelope::from)?,
            index_mode: IndexMode::Dense,
            query: "hello".into(),
            top_k: None,
            threshold: Some(0.7),
        };

        let _ = semantic_search(&ctx, &deps, input).await?;
        let options = vectordb.last_search_options()?.ok_or_else(|| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "missing search options",
                ErrorClass::NonRetriable,
            )
        })?;
        assert_eq!(options.threshold, Some(0.7));
        Ok(())
    }

    #[tokio::test]
    async fn abort_stops_before_vectordb_call() -> SharedResult<()> {
        let vectordb = Arc::new(TestVectorDb::new(Vec::new())?);
        let embedding = Arc::new(TestEmbedding::new(vec![0.1, 0.2, 0.3])?);
        let deps = SemanticSearchDeps {
            embedding: embedding.clone(),
            vectordb: vectordb.clone(),
            logger: None,
            telemetry: None,
        };

        let ctx = RequestContext::new_request();
        ctx.cancel();

        let input = SemanticSearchInput {
            codebase_root: "/tmp".into(),
            collection_name: CollectionName::parse("code_chunks_test")
                .map_err(ErrorEnvelope::from)?,
            index_mode: IndexMode::Dense,
            query: "hello".into(),
            top_k: None,
            threshold: None,
        };

        let result = semantic_search(&ctx, &deps, input).await;
        assert!(matches!(result, Err(error) if error.is_cancelled()));
        assert_eq!(embedding.calls.load(Ordering::SeqCst), 0);
        assert_eq!(vectordb.search_calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[tokio::test]
    async fn hybrid_request_shape_is_correct() -> SharedResult<()> {
        let vectordb = Arc::new(TestVectorDb::new(Vec::new())?);
        let embedding = Arc::new(TestEmbedding::new(vec![0.1, 0.2, 0.3])?);
        let deps = SemanticSearchDeps {
            embedding,
            vectordb: vectordb.clone(),
            logger: None,
            telemetry: None,
        };

        let ctx = RequestContext::new_request();
        let input = SemanticSearchInput {
            codebase_root: "/tmp".into(),
            collection_name: CollectionName::parse("code_chunks_test")
                .map_err(ErrorEnvelope::from)?,
            index_mode: IndexMode::Hybrid,
            query: "hello".into(),
            top_k: Some(3),
            threshold: None,
        };

        let _ = semantic_search(&ctx, &deps, input).await?;
        let requests = vectordb.take_hybrid_requests()?;
        assert_eq!(requests.len(), 2);
        let first = requests.get(0).ok_or_else(|| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "missing first request",
                ErrorClass::NonRetriable,
            )
        })?;
        let second = requests.get(1).ok_or_else(|| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "missing second request",
                ErrorClass::NonRetriable,
            )
        })?;
        assert!(matches!(first.data, HybridSearchData::DenseVector(_)));
        assert!(matches!(second.data, HybridSearchData::SparseQuery(_)));
        assert_eq!(first.anns_field.as_ref(), "vector");
        assert_eq!(second.anns_field.as_ref(), "sparse_vector");
        Ok(())
    }
}
