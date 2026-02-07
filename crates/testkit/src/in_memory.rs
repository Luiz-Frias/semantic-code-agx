//! In-memory adapter implementations for port contracts.
//!
//! These implementations are intended for:
//! - Unit/integration tests
//! - Deterministic contract tests for the ports layer
//! - Local experimentation without external dependencies

use semantic_code_ports::{
    BoxFuture, DetectDimensionRequest, EmbedBatchRequest, EmbedRequest, EmbeddingPort,
    EmbeddingProviderInfo, EmbeddingVector, EmbeddingVectorFixed, HybridSearchBatchRequest,
    HybridSearchData, HybridSearchResult, LogEvent, LogFields, LogLevel, LoggerPort,
    RerankStrategy, RerankStrategyKind, TelemetryPort, TelemetryTags, TelemetryTimer, VectorDbPort,
    VectorDbProviderInfo, VectorDocument, VectorDocumentForInsert, VectorSearchOptions,
    VectorSearchRequest, VectorSearchResult,
};
use semantic_code_shared::{ErrorCode, ErrorEnvelope, RequestContext, Result};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio::sync::RwLock;

/// A no-op logger implementation.
#[derive(Debug, Default)]
pub struct NoopLogger;

impl LoggerPort for NoopLogger {
    fn log(&self, _event: LogEvent) {}

    fn child(&self, _fields: LogFields) -> Box<dyn LoggerPort> {
        Box::new(Self)
    }
}

/// A no-op telemetry timer.
#[derive(Debug, Default)]
pub struct NoopTimer;

impl TelemetryTimer for NoopTimer {
    fn stop(&self) {}
}

/// A no-op telemetry implementation.
#[derive(Debug, Default)]
pub struct NoopTelemetry;

impl TelemetryPort for NoopTelemetry {
    fn increment_counter(&self, _name: &str, _value: u64, _tags: Option<&TelemetryTags>) {}

    fn record_timer_ms(&self, _name: &str, _duration_ms: u64, _tags: Option<&TelemetryTags>) {}

    fn start_timer(&self, _name: &str, _tags: Option<&TelemetryTags>) -> Box<dyn TelemetryTimer> {
        Box::new(NoopTimer)
    }
}

/// Deterministic in-memory embedding provider.
#[derive(Debug, Clone)]
pub struct InMemoryEmbedding {
    provider: EmbeddingProviderInfo,
    dimension: u32,
}

fn embed_text_with_dimension(text: &str, dimension: u32) -> Vec<f32> {
    let dim = dimension.max(1) as usize;
    let mut buckets = vec![0u32; dim];
    for (idx, byte) in text.as_bytes().iter().enumerate() {
        let slot = idx % dim;
        buckets[slot] = buckets[slot].wrapping_add(u32::from(*byte));
    }

    buckets
        .into_iter()
        .map(|value| (value as f32) / 255.0)
        .collect()
}

impl InMemoryEmbedding {
    /// Create a deterministic embedder.
    #[must_use]
    pub fn new(provider: EmbeddingProviderInfo, dimension: u32) -> Self {
        Self {
            provider,
            dimension,
        }
    }

    fn embed_text(&self, text: &str) -> Vec<f32> {
        embed_text_with_dimension(text, self.dimension)
    }
}

/// Deterministic in-memory embedding provider with a fixed dimension.
#[derive(Debug, Clone)]
pub struct InMemoryEmbeddingFixed<const D: usize> {
    provider: EmbeddingProviderInfo,
}

impl<const D: usize> InMemoryEmbeddingFixed<D> {
    /// Create a deterministic embedder with fixed dimension `D`.
    #[must_use]
    pub fn new(provider: EmbeddingProviderInfo) -> Self {
        Self { provider }
    }

    fn dimension() -> Result<u32> {
        u32::try_from(D).map_err(|_| {
            ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "embedding dimension out of range",
            )
        })
    }
}

impl EmbeddingPort for InMemoryEmbedding {
    fn provider(&self) -> &EmbeddingProviderInfo {
        &self.provider
    }

    fn detect_dimension(
        &self,
        ctx: &RequestContext,
        _request: DetectDimensionRequest,
    ) -> BoxFuture<'_, Result<u32>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            ctx.ensure_not_cancelled("in_memory_embedding.detect_dimension")?;
            Ok(self.dimension)
        })
    }

    fn embed(
        &self,
        ctx: &RequestContext,
        request: EmbedRequest,
    ) -> BoxFuture<'_, Result<EmbeddingVector>> {
        let ctx = ctx.clone();
        let text = request.text;
        Box::pin(async move {
            ctx.ensure_not_cancelled("in_memory_embedding.embed")?;
            let vector = self.embed_text(text.as_ref());
            let _ = self.dimension;
            Ok(EmbeddingVector::from_vec(vector))
        })
    }

    fn embed_batch(
        &self,
        ctx: &RequestContext,
        request: EmbedBatchRequest,
    ) -> BoxFuture<'_, Result<Vec<EmbeddingVector>>> {
        let ctx = ctx.clone();
        let texts = request.texts;
        Box::pin(async move {
            ctx.ensure_not_cancelled("in_memory_embedding.embed_batch")?;
            let out = texts
                .into_iter()
                .map(|text| {
                    let _ = self.dimension;
                    EmbeddingVector::from_vec(self.embed_text(text.as_ref()))
                })
                .collect();
            Ok(out)
        })
    }
}

impl<const D: usize> EmbeddingPort for InMemoryEmbeddingFixed<D> {
    fn provider(&self) -> &EmbeddingProviderInfo {
        &self.provider
    }

    fn detect_dimension(
        &self,
        ctx: &RequestContext,
        _request: DetectDimensionRequest,
    ) -> BoxFuture<'_, Result<u32>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            ctx.ensure_not_cancelled("in_memory_embedding_fixed.detect_dimension")?;
            Self::dimension()
        })
    }

    fn embed(
        &self,
        ctx: &RequestContext,
        request: EmbedRequest,
    ) -> BoxFuture<'_, Result<EmbeddingVector>> {
        let ctx = ctx.clone();
        let text = request.text;
        Box::pin(async move {
            ctx.ensure_not_cancelled("in_memory_embedding_fixed.embed")?;
            let dimension = Self::dimension()?;
            let vector = embed_text_with_dimension(text.as_ref(), dimension);
            let fixed = EmbeddingVectorFixed::<D>::new(Arc::from(vector))?;
            Ok(EmbeddingVector::from(fixed))
        })
    }

    fn embed_batch(
        &self,
        ctx: &RequestContext,
        request: EmbedBatchRequest,
    ) -> BoxFuture<'_, Result<Vec<EmbeddingVector>>> {
        let ctx = ctx.clone();
        let texts = request.texts;
        Box::pin(async move {
            ctx.ensure_not_cancelled("in_memory_embedding_fixed.embed_batch")?;
            let dimension = Self::dimension()?;
            let out = texts
                .into_iter()
                .map(|text| {
                    let vector = embed_text_with_dimension(text.as_ref(), dimension);
                    let fixed = EmbeddingVectorFixed::<D>::new(Arc::from(vector))?;
                    Ok(EmbeddingVector::from(fixed))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(out)
        })
    }
}

#[derive(Debug, Clone)]
struct StoredDocument {
    vector: Vec<f32>,
    content: Box<str>,
    metadata: semantic_code_ports::VectorDocumentMetadata,
}

#[derive(Debug)]
struct CollectionState {
    dimension: u32,
    documents: HashMap<Box<str>, StoredDocument>,
}

/// In-memory vector DB implementation with naive dense search.
#[derive(Debug)]
pub struct InMemoryVectorDb {
    provider: VectorDbProviderInfo,
    collections: RwLock<HashMap<semantic_code_ports::CollectionName, CollectionState>>,
}

impl InMemoryVectorDb {
    /// Create a new empty in-memory vector DB.
    #[must_use]
    pub fn new(provider: VectorDbProviderInfo) -> Self {
        Self {
            provider,
            collections: RwLock::new(HashMap::new()),
        }
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn ensure_dimension(expected: u32, vector: &[f32]) -> Result<()> {
        if vector.len() != expected as usize {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "vector dimension mismatch",
            )
            .with_metadata("expected", expected.to_string())
            .with_metadata("actual", vector.len().to_string()));
        }
        Ok(())
    }
}

/// In-memory vector DB with a fixed dimension.
#[derive(Debug)]
pub struct InMemoryVectorDbFixed<const D: usize> {
    inner: InMemoryVectorDb,
}

impl<const D: usize> InMemoryVectorDbFixed<D> {
    /// Create a new fixed-dimension vector DB.
    #[must_use]
    pub fn new(provider: VectorDbProviderInfo) -> Self {
        Self {
            inner: InMemoryVectorDb::new(provider),
        }
    }

    fn expected_dimension() -> Result<u32> {
        u32::try_from(D).map_err(|_| {
            ErrorEnvelope::expected(ErrorCode::invalid_input(), "vector dimension out of range")
        })
    }

    fn ensure_dimension_matches(dimension: u32) -> Result<u32> {
        let expected = Self::expected_dimension()?;
        if dimension != expected {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "vector dimension mismatch",
            )
            .with_metadata("expected", expected.to_string())
            .with_metadata("actual", dimension.to_string()));
        }
        Ok(expected)
    }
}

impl VectorDbPort for InMemoryVectorDb {
    fn provider(&self) -> &VectorDbProviderInfo {
        &self.provider
    }

    fn create_collection(
        &self,
        ctx: &RequestContext,
        collection_name: semantic_code_ports::CollectionName,
        dimension: u32,
        _description: Option<Box<str>>,
    ) -> BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            ctx.ensure_not_cancelled("in_memory_vectordb.create_collection")?;
            let mut state = self.collections.write().await;
            state
                .entry(collection_name)
                .or_insert_with(|| CollectionState {
                    dimension,
                    documents: HashMap::new(),
                });
            Ok(())
        })
    }

    fn create_hybrid_collection(
        &self,
        ctx: &RequestContext,
        collection_name: semantic_code_ports::CollectionName,
        dimension: u32,
        description: Option<Box<str>>,
    ) -> BoxFuture<'_, Result<()>> {
        self.create_collection(ctx, collection_name, dimension, description)
    }

    fn drop_collection(
        &self,
        ctx: &RequestContext,
        collection_name: semantic_code_ports::CollectionName,
    ) -> BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            ctx.ensure_not_cancelled("in_memory_vectordb.drop_collection")?;
            let mut state = self.collections.write().await;
            state.remove(&collection_name);
            Ok(())
        })
    }

    fn has_collection(
        &self,
        ctx: &RequestContext,
        collection_name: semantic_code_ports::CollectionName,
    ) -> BoxFuture<'_, Result<bool>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            ctx.ensure_not_cancelled("in_memory_vectordb.has_collection")?;
            let state = self.collections.read().await;
            Ok(state.contains_key(&collection_name))
        })
    }

    fn list_collections(
        &self,
        ctx: &RequestContext,
    ) -> BoxFuture<'_, Result<Vec<semantic_code_ports::CollectionName>>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            ctx.ensure_not_cancelled("in_memory_vectordb.list_collections")?;
            let state = self.collections.read().await;
            let mut out: Vec<_> = state.keys().cloned().collect();
            out.sort_by(|a, b| a.as_str().cmp(b.as_str()));
            Ok(out)
        })
    }

    fn insert(
        &self,
        ctx: &RequestContext,
        collection_name: semantic_code_ports::CollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            ctx.ensure_not_cancelled("in_memory_vectordb.insert")?;
            let mut state = self.collections.write().await;
            let Some(collection) = state.get_mut(&collection_name) else {
                return Err(ErrorEnvelope::expected(
                    ErrorCode::not_found(),
                    "collection not found",
                ));
            };

            for doc in documents {
                Self::ensure_dimension(collection.dimension, doc.vector.as_ref())?;
                collection.documents.insert(
                    doc.id,
                    StoredDocument {
                        vector: doc.vector.as_ref().to_vec(),
                        content: doc.content,
                        metadata: doc.metadata,
                    },
                );
            }

            Ok(())
        })
    }

    fn insert_hybrid(
        &self,
        ctx: &RequestContext,
        collection_name: semantic_code_ports::CollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> BoxFuture<'_, Result<()>> {
        self.insert(ctx, collection_name, documents)
    }

    fn search(
        &self,
        ctx: &RequestContext,
        request: VectorSearchRequest,
    ) -> BoxFuture<'_, Result<Vec<VectorSearchResult>>> {
        let ctx = ctx.clone();
        let VectorSearchRequest {
            collection_name,
            query_vector,
            options,
        } = request;
        Box::pin(async move {
            ctx.ensure_not_cancelled("in_memory_vectordb.search")?;
            let state = self.collections.read().await;
            let Some(collection) = state.get(&collection_name) else {
                return Err(ErrorEnvelope::expected(
                    ErrorCode::not_found(),
                    "collection not found",
                ));
            };

            Self::ensure_dimension(collection.dimension, query_vector.as_ref())?;

            let top_k = options.top_k.unwrap_or(10).max(1) as usize;
            let threshold = options.threshold;
            let _filter_expr = options.filter_expr;

            let mut scored: Vec<VectorSearchResult> = collection
                .documents
                .iter()
                .map(|(id, doc)| {
                    let score = Self::dot(query_vector.as_ref(), &doc.vector);
                    VectorSearchResult {
                        document: VectorDocument {
                            id: id.clone(),
                            vector: None,
                            content: doc.content.clone(),
                            metadata: doc.metadata.clone(),
                        },
                        score,
                    }
                })
                .filter(|result| threshold.map_or(true, |t| result.score >= t))
                .collect();

            scored.sort_by(|a, b| {
                let score = b.score.total_cmp(&a.score);
                if score != std::cmp::Ordering::Equal {
                    return score;
                }
                a.document.id.cmp(&b.document.id)
            });
            scored.truncate(top_k);

            Ok(scored)
        })
    }

    fn hybrid_search(
        &self,
        ctx: &RequestContext,
        request: HybridSearchBatchRequest,
    ) -> BoxFuture<'_, Result<Vec<HybridSearchResult>>> {
        let ctx = ctx.clone();
        let HybridSearchBatchRequest {
            collection_name,
            search_requests,
            options,
        } = request;
        Box::pin(async move {
            ctx.ensure_not_cancelled("in_memory_vectordb.hybrid_search")?;

            let mut merged: HashMap<Box<str>, HybridSearchResult> = HashMap::new();
            let global_limit = options.limit;
            let _filter_expr = options.filter_expr;
            let _rerank = options.rerank;

            for req in search_requests {
                let limit = req.limit.max(1);
                let query = match req.data {
                    HybridSearchData::DenseVector(vector) => vector,
                    HybridSearchData::SparseQuery(_) => {
                        return Err(ErrorEnvelope::expected(
                            ErrorCode::new("core", "not_supported"),
                            "sparse hybrid search not supported by in-memory adapter",
                        ));
                    },
                };

                let results = self
                    .search(
                        &ctx,
                        VectorSearchRequest {
                            collection_name: collection_name.clone(),
                            query_vector: query,
                            options: VectorSearchOptions {
                                top_k: Some(limit),
                                filter_expr: None,
                                threshold: None,
                            },
                        },
                    )
                    .await?;

                for result in results {
                    let id = result.document.id.clone();
                    let entry = merged.entry(id).or_insert(HybridSearchResult {
                        document: result.document,
                        score: result.score,
                    });
                    if result.score > entry.score {
                        entry.score = result.score;
                    }
                }
            }

            let mut out: Vec<_> = merged.into_values().collect();
            out.sort_by(|a, b| {
                let score = b.score.total_cmp(&a.score);
                if score != std::cmp::Ordering::Equal {
                    return score;
                }
                a.document.id.cmp(&b.document.id)
            });

            if let Some(limit) = global_limit {
                out.truncate(limit.max(1) as usize);
            }

            Ok(out)
        })
    }

    fn delete(
        &self,
        ctx: &RequestContext,
        collection_name: semantic_code_ports::CollectionName,
        ids: Vec<Box<str>>,
    ) -> BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            ctx.ensure_not_cancelled("in_memory_vectordb.delete")?;
            let mut state = self.collections.write().await;
            let Some(collection) = state.get_mut(&collection_name) else {
                return Ok(());
            };

            for id in ids {
                collection.documents.remove(&id);
            }
            Ok(())
        })
    }

    fn query(
        &self,
        ctx: &RequestContext,
        _collection_name: semantic_code_ports::CollectionName,
        _filter: Box<str>,
        _output_fields: Vec<Box<str>>,
        _limit: Option<u32>,
    ) -> BoxFuture<'_, Result<Vec<BTreeMap<Box<str>, Value>>>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            ctx.ensure_not_cancelled("in_memory_vectordb.query")?;
            Ok(Vec::new())
        })
    }
}

impl<const D: usize> VectorDbPort for InMemoryVectorDbFixed<D> {
    fn provider(&self) -> &VectorDbProviderInfo {
        VectorDbPort::provider(&self.inner)
    }

    fn create_collection(
        &self,
        ctx: &RequestContext,
        collection_name: semantic_code_ports::CollectionName,
        dimension: u32,
        description: Option<Box<str>>,
    ) -> BoxFuture<'_, Result<()>> {
        let expected = match Self::ensure_dimension_matches(dimension) {
            Ok(value) => value,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        self.inner
            .create_collection(ctx, collection_name, expected, description)
    }

    fn create_hybrid_collection(
        &self,
        ctx: &RequestContext,
        collection_name: semantic_code_ports::CollectionName,
        dimension: u32,
        description: Option<Box<str>>,
    ) -> BoxFuture<'_, Result<()>> {
        let expected = match Self::ensure_dimension_matches(dimension) {
            Ok(value) => value,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        self.inner
            .create_hybrid_collection(ctx, collection_name, expected, description)
    }

    fn drop_collection(
        &self,
        ctx: &RequestContext,
        collection_name: semantic_code_ports::CollectionName,
    ) -> BoxFuture<'_, Result<()>> {
        self.inner.drop_collection(ctx, collection_name)
    }

    fn has_collection(
        &self,
        ctx: &RequestContext,
        collection_name: semantic_code_ports::CollectionName,
    ) -> BoxFuture<'_, Result<bool>> {
        self.inner.has_collection(ctx, collection_name)
    }

    fn list_collections(
        &self,
        ctx: &RequestContext,
    ) -> BoxFuture<'_, Result<Vec<semantic_code_ports::CollectionName>>> {
        self.inner.list_collections(ctx)
    }

    fn insert(
        &self,
        ctx: &RequestContext,
        collection_name: semantic_code_ports::CollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> BoxFuture<'_, Result<()>> {
        self.inner.insert(ctx, collection_name, documents)
    }

    fn insert_hybrid(
        &self,
        ctx: &RequestContext,
        collection_name: semantic_code_ports::CollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> BoxFuture<'_, Result<()>> {
        self.inner.insert_hybrid(ctx, collection_name, documents)
    }

    fn search(
        &self,
        ctx: &RequestContext,
        request: VectorSearchRequest,
    ) -> BoxFuture<'_, Result<Vec<VectorSearchResult>>> {
        self.inner.search(ctx, request)
    }

    fn hybrid_search(
        &self,
        ctx: &RequestContext,
        request: HybridSearchBatchRequest,
    ) -> BoxFuture<'_, Result<Vec<HybridSearchResult>>> {
        self.inner.hybrid_search(ctx, request)
    }

    fn delete(
        &self,
        ctx: &RequestContext,
        collection_name: semantic_code_ports::CollectionName,
        ids: Vec<Box<str>>,
    ) -> BoxFuture<'_, Result<()>> {
        self.inner.delete(ctx, collection_name, ids)
    }

    fn query(
        &self,
        ctx: &RequestContext,
        collection_name: semantic_code_ports::CollectionName,
        filter: Box<str>,
        output_fields: Vec<Box<str>>,
        limit: Option<u32>,
    ) -> BoxFuture<'_, Result<Vec<BTreeMap<Box<str>, Value>>>> {
        self.inner
            .query(ctx, collection_name, filter, output_fields, limit)
    }
}

/// Build a `LogEvent` helper for tests.
#[must_use]
pub fn log_event(level: LogLevel, event: &str, message: &str) -> LogEvent {
    LogEvent {
        event: event.to_owned().into_boxed_str(),
        level,
        message: message.to_owned().into_boxed_str(),
        fields: None,
        error: None,
    }
}

/// Build a single-tag map for tests.
#[must_use]
pub fn tags_1(key: &str, value: &str) -> TelemetryTags {
    let mut tags = TelemetryTags::new();
    tags.insert(
        key.to_owned().into_boxed_str(),
        value.to_owned().into_boxed_str(),
    );
    tags
}

/// Helper to create a simple rerank strategy placeholder.
#[must_use]
pub fn rerank_rrf() -> RerankStrategy {
    RerankStrategy {
        strategy: RerankStrategyKind::Rrf,
        params: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_domain::{
        CollectionName, EmbeddingProviderId, Language, LineSpan, VectorDbProviderId,
        VectorDocumentMetadata,
    };
    use semantic_code_ports::DetectDimensionOptions;

    #[tokio::test]
    async fn in_memory_embedding_is_deterministic() -> Result<()> {
        let ctx = RequestContext::new_request();
        let provider = EmbeddingProviderInfo {
            id: EmbeddingProviderId::parse("test").map_err(ErrorEnvelope::from)?,
            name: "test".into(),
        };
        let embedder = InMemoryEmbedding::new(provider, 8);

        let d = embedder
            .detect_dimension(&ctx, DetectDimensionOptions::default().into())
            .await?;
        assert_eq!(d, 8);

        let one = embedder.embed(&ctx, "hello".into()).await?;
        let two = embedder.embed(&ctx, "hello".into()).await?;

        assert_eq!(one.dimension(), 8);
        assert_eq!(one.as_slice().len(), 8);
        assert_eq!(one, two, "same input should embed deterministically");

        let batch = embedder
            .embed_batch(
                &ctx,
                EmbedBatchRequest::from(vec!["a".to_string(), "b".to_string()]),
            )
            .await?;
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].dimension(), 8);

        Ok(())
    }

    #[tokio::test]
    async fn in_memory_vectordb_supports_insert_and_search() -> Result<()> {
        let ctx = RequestContext::new_request();
        let provider = VectorDbProviderInfo {
            id: VectorDbProviderId::parse("local").map_err(ErrorEnvelope::from)?,
            name: "local".into(),
        };
        let db = InMemoryVectorDb::new(provider);

        let collection = CollectionName::parse("code_chunks_test").map_err(ErrorEnvelope::from)?;
        db.create_collection(&ctx, collection.clone(), 4, None)
            .await?;

        let span = LineSpan::new(1, 2).map_err(ErrorEnvelope::from)?;
        let meta = VectorDocumentMetadata {
            relative_path: "src/lib.rs".into(),
            language: Some(Language::Rust),
            file_extension: Some("rs".into()),
            span,
            node_kind: None,
        };

        db.insert(
            &ctx,
            collection.clone(),
            vec![
                VectorDocumentForInsert {
                    id: "doc_a".into(),
                    vector: Arc::from(vec![1.0, 0.0, 0.0, 0.0]),
                    content: "a".into(),
                    metadata: meta.clone(),
                },
                VectorDocumentForInsert {
                    id: "doc_b".into(),
                    vector: Arc::from(vec![0.0, 1.0, 0.0, 0.0]),
                    content: "b".into(),
                    metadata: meta,
                },
            ],
        )
        .await?;

        let results = db
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection,
                    query_vector: Arc::from(vec![1.0, 0.0, 0.0, 0.0]),
                    options: VectorSearchOptions {
                        top_k: Some(2),
                        filter_expr: None,
                        threshold: None,
                    },
                },
            )
            .await?;

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].document.id.as_ref(), "doc_a");
        Ok(())
    }

    #[test]
    fn noop_logger_and_telemetry_are_safe() {
        let logger = NoopLogger::default();
        let child = logger.child(LogFields::new());
        child.log(log_event(LogLevel::Info, "event", "message"));

        let telemetry = NoopTelemetry::default();
        telemetry.increment_counter("counter", 1, None);
        telemetry.record_timer_ms("timer", 10, None);
        let timer = telemetry.start_timer("timer2", None);
        timer.stop();
    }
}
