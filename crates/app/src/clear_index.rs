//! Clear an index collection and associated sync snapshot.

use semantic_code_domain::CollectionName;
use semantic_code_ports::{FileSyncPort, LoggerPort, TelemetryPort, VectorDbPort};
use semantic_code_shared::{
    ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result, RetryPolicy, retry_async,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

const DROP_COLLECTION_RETRY_POLICY: RetryPolicy = RetryPolicy {
    max_attempts: 5,
    base_delay_ms: 150,
    max_delay_ms: 1_500,
    jitter_ratio_pct: 10,
};

/// Input payload for clearing an index.
#[derive(Debug, Clone)]
pub struct ClearIndexInput {
    /// Codebase root (absolute path).
    pub codebase_root: PathBuf,
    /// Target collection name.
    pub collection_name: CollectionName,
}

/// Dependencies required by clear-index.
#[derive(Clone)]
pub struct ClearIndexDeps {
    /// Vector DB adapter.
    pub vectordb: Arc<dyn VectorDbPort>,
    /// File sync adapter (snapshot delete).
    pub file_sync: Arc<dyn FileSyncPort>,
    /// Optional logger.
    pub logger: Option<Arc<dyn LoggerPort>>,
    /// Optional telemetry sink.
    pub telemetry: Option<Arc<dyn TelemetryPort>>,
}

/// Clear the collection and delete any sync snapshot.
#[tracing::instrument(
    name = "app.clear_index",
    skip_all,
    fields(collection = %input.collection_name.as_str())
)]
pub async fn clear_index(
    ctx: &RequestContext,
    deps: &ClearIndexDeps,
    input: ClearIndexInput,
) -> Result<()> {
    let started_at = Instant::now();
    let total_timer = deps
        .telemetry
        .as_ref()
        .map(|telemetry| telemetry.start_timer("backend.clearIndex.total", None));

    if let Some(logger) = deps.logger.as_ref() {
        logger.info(
            "backend.clearIndex.start",
            "Clear index started",
            Some(log_fields_start(&input)),
        );
    }

    let result: Result<()> = (async {
        ctx.ensure_not_cancelled("clear_index.start")?;

        let provider_tags = tags_provider(deps.vectordb.provider().id.as_str());
        let has_collection_timer = deps.telemetry.as_ref().map(|telemetry| {
            telemetry.start_timer("backend.clearIndex.hasCollection", Some(&provider_tags))
        });
        let exists = deps
            .vectordb
            .has_collection(ctx, input.collection_name.clone())
            .await?;
        tracing::debug!(collection_exists = exists, "checked collection existence");
        if let Some(timer) = has_collection_timer.as_ref() {
            timer.stop();
        }

        if exists {
            ctx.ensure_not_cancelled("clear_index.drop_collection")?;
            let drop_tags = tags_provider(deps.vectordb.provider().id.as_str());
            let drop_timer = deps.telemetry.as_ref().map(|telemetry| {
                telemetry.start_timer("backend.clearIndex.dropCollection", Some(&drop_tags))
            });
            drop_collection_with_retry(ctx, deps, input.collection_name.clone()).await?;
            if let Some(timer) = drop_timer.as_ref() {
                timer.stop();
            }
        }

        ctx.ensure_not_cancelled("clear_index.delete_snapshot")?;
        let snapshot_timer = deps
            .telemetry
            .as_ref()
            .map(|telemetry| telemetry.start_timer("backend.clearIndex.deleteSnapshot", None));
        deps.file_sync
            .delete_snapshot(ctx, input.codebase_root.clone())
            .await?;
        if let Some(timer) = snapshot_timer.as_ref() {
            timer.stop();
        }

        if let Some(telemetry) = deps.telemetry.as_ref() {
            telemetry.increment_counter("backend.clearIndex.executed", 1, None);
        }
        tracing::debug!("clear index operation completed");

        if let Some(logger) = deps.logger.as_ref() {
            logger.info(
                "backend.clearIndex.completed",
                "Clear index completed",
                Some(log_fields_completed(&input, started_at)),
            );
        }

        Ok(())
    })
    .await;

    if let Some(timer) = total_timer.as_ref() {
        timer.stop();
    }

    match result {
        Ok(()) => Ok(()),
        Err(error) => {
            log_clear_index_failure(deps, &input, started_at, &error);
            Err(error)
        },
    }
}

fn log_clear_index_failure(
    deps: &ClearIndexDeps,
    input: &ClearIndexInput,
    started_at: Instant,
    error: &ErrorEnvelope,
) {
    let duration_ms = duration_ms(started_at);
    if error.is_cancelled() {
        if let Some(telemetry) = deps.telemetry.as_ref() {
            telemetry.increment_counter("backend.clearIndex.aborted", 1, None);
        }
        if let Some(logger) = deps.logger.as_ref() {
            logger.info(
                "backend.clearIndex.aborted",
                "Clear index aborted",
                Some(log_fields_abort(duration_ms)),
            );
        }
    } else {
        if let Some(telemetry) = deps.telemetry.as_ref() {
            telemetry.increment_counter("backend.clearIndex.failed", 1, None);
        }
        if let Some(logger) = deps.logger.as_ref() {
            logger.error(
                "backend.clearIndex.failed",
                "Clear index failed",
                Some(log_fields_error(input, duration_ms, error)),
            );
        }
    }
    tracing::warn!(
        error = %error,
        cancelled = error.is_cancelled(),
        "clear index failed"
    );
}

#[tracing::instrument(
    name = "app.clear_index.drop_collection_retry",
    skip_all,
    fields(
        collection = %collection_name.as_str(),
        provider = deps.vectordb.provider().id.as_str(),
    )
)]
async fn drop_collection_with_retry(
    ctx: &RequestContext,
    deps: &ClearIndexDeps,
    collection_name: CollectionName,
) -> Result<()> {
    let provider_id = deps.vectordb.provider().id.as_str().to_owned();
    let mut operation = || {
        let collection_name = collection_name.clone();
        let provider_id = provider_id.clone();
        async move {
            match deps
                .vectordb
                .drop_collection(ctx, collection_name.clone())
                .await
            {
                Ok(()) => Ok(()),
                Err(error) => {
                    let exists = deps
                        .vectordb
                        .has_collection(ctx, collection_name.clone())
                        .await
                        .unwrap_or(true);
                    if !exists {
                        return Ok(());
                    }
                    if should_retry_drop_collection_error(provider_id.as_str(), &error) {
                        let mut retryable = error;
                        retryable.class = ErrorClass::Retriable;
                        return Err(retryable);
                    }
                    Err(error)
                },
            }
        }
    };

    retry_async(
        ctx,
        DROP_COLLECTION_RETRY_POLICY,
        "clear_index.drop_collection",
        &mut operation,
    )
    .await
}

fn should_retry_drop_collection_error(provider_id: &str, error: &ErrorEnvelope) -> bool {
    if error.class.is_retriable() {
        return true;
    }
    if provider_id != "milvus_grpc" && provider_id != "milvus_rest" {
        return false;
    }
    if error.code != ErrorCode::new("vector", "vdb_unknown") {
        return false;
    }

    let operation = error.metadata.get("operation").map(String::as_str);
    matches!(
        operation,
        Some("milvus_grpc.drop_collection" | "milvus_rest.drop_collection")
    )
}

fn duration_ms(started_at: Instant) -> u64 {
    u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn tags_provider(provider_id: &str) -> BTreeMap<Box<str>, Box<str>> {
    let mut tags = BTreeMap::new();
    tags.insert(
        "providerId".to_owned().into_boxed_str(),
        provider_id.to_owned().into_boxed_str(),
    );
    tags
}

fn log_fields_start(input: &ClearIndexInput) -> BTreeMap<Box<str>, Value> {
    let mut fields = BTreeMap::new();
    fields.insert(
        "codebaseRoot".to_owned().into_boxed_str(),
        Value::String(input.codebase_root.to_string_lossy().to_string()),
    );
    fields.insert(
        "collectionName".to_owned().into_boxed_str(),
        Value::String(input.collection_name.as_str().to_owned()),
    );
    fields
}

fn log_fields_completed(input: &ClearIndexInput, started_at: Instant) -> BTreeMap<Box<str>, Value> {
    let mut fields = log_fields_start(input);
    fields.insert(
        "durationMs".to_owned().into_boxed_str(),
        Value::from(duration_ms(started_at)),
    );
    fields
}

fn log_fields_abort(duration_ms: u64) -> BTreeMap<Box<str>, Value> {
    let mut fields = BTreeMap::new();
    fields.insert(
        "durationMs".to_owned().into_boxed_str(),
        Value::from(duration_ms),
    );
    fields
}

fn log_fields_error(
    input: &ClearIndexInput,
    duration_ms: u64,
    error: &ErrorEnvelope,
) -> BTreeMap<Box<str>, Value> {
    let mut fields = log_fields_abort(duration_ms);
    fields.insert(
        "collectionName".to_owned().into_boxed_str(),
        Value::String(input.collection_name.as_str().to_owned()),
    );
    fields.insert(
        "error".to_owned().into_boxed_str(),
        Value::String(error.to_string()),
    );
    fields
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_domain::VectorDbProviderId;
    use semantic_code_ports::{
        FileSyncOptions, FileSyncPort, VectorDbProviderInfo, VectorDocumentForInsert,
        VectorSearchResponse,
    };
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    #[derive(Clone)]
    struct NoopVectorDb {
        provider: VectorDbProviderInfo,
        has_collection: bool,
        drop_called: Arc<AtomicBool>,
    }

    #[derive(Clone)]
    struct FlakyMilvusDropVectorDb {
        provider: VectorDbProviderInfo,
        attempts: Arc<AtomicU32>,
        dropped: Arc<AtomicBool>,
        fail_attempts: u32,
    }

    impl FlakyMilvusDropVectorDb {
        fn new(fail_attempts: u32) -> Result<Self> {
            Ok(Self {
                provider: VectorDbProviderInfo {
                    id: VectorDbProviderId::parse("milvus_grpc").map_err(ErrorEnvelope::from)?,
                    name: "flaky-milvus".into(),
                },
                attempts: Arc::new(AtomicU32::new(0)),
                dropped: Arc::new(AtomicBool::new(false)),
                fail_attempts,
            })
        }
    }

    impl VectorDbPort for FlakyMilvusDropVectorDb {
        fn provider(&self) -> &VectorDbProviderInfo {
            &self.provider
        }

        fn create_collection(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _dimension: u32,
            _description: Option<Box<str>>,
        ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
            Box::pin(async move { Ok(()) })
        }

        fn create_hybrid_collection(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _dimension: u32,
            _description: Option<Box<str>>,
        ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
            Box::pin(async move { Ok(()) })
        }

        fn drop_collection(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
        ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
            let attempts = Arc::clone(&self.attempts);
            let dropped = Arc::clone(&self.dropped);
            let fail_attempts = self.fail_attempts;
            Box::pin(async move {
                let attempt = attempts.fetch_add(1, Ordering::SeqCst) + 1;
                if attempt <= fail_attempts {
                    return Err(ErrorEnvelope::unexpected(
                        ErrorCode::new("vector", "vdb_unknown"),
                        "transport error",
                        ErrorClass::NonRetriable,
                    )
                    .with_metadata("provider", "milvus_grpc")
                    .with_metadata("operation", "milvus_grpc.drop_collection")
                    .with_metadata("grpc_code", "Unknown error"));
                }
                dropped.store(true, Ordering::SeqCst);
                Ok(())
            })
        }

        fn has_collection(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
        ) -> semantic_code_ports::BoxFuture<'_, Result<bool>> {
            let dropped = Arc::clone(&self.dropped);
            Box::pin(async move { Ok(!dropped.load(Ordering::SeqCst)) })
        }

        fn list_collections(
            &self,
            _ctx: &RequestContext,
        ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<CollectionName>>> {
            Box::pin(async move { Ok(Vec::new()) })
        }

        fn insert(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _documents: Vec<VectorDocumentForInsert>,
        ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
            Box::pin(async move { Ok(()) })
        }

        fn insert_hybrid(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _documents: Vec<VectorDocumentForInsert>,
        ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
            Box::pin(async move { Ok(()) })
        }

        fn search(
            &self,
            _ctx: &RequestContext,
            _request: semantic_code_ports::VectorSearchRequest,
        ) -> semantic_code_ports::BoxFuture<'_, Result<VectorSearchResponse>> {
            Box::pin(async move {
                Ok(VectorSearchResponse {
                    results: Vec::new(),
                    stats: None,
                })
            })
        }

        fn hybrid_search(
            &self,
            _ctx: &RequestContext,
            _request: semantic_code_ports::HybridSearchBatchRequest,
        ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<semantic_code_ports::HybridSearchResult>>>
        {
            Box::pin(async move { Ok(Vec::new()) })
        }

        fn delete(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _ids: Vec<Box<str>>,
        ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
            Box::pin(async move { Ok(()) })
        }

        fn query(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _filter: Box<str>,
            _output_fields: Vec<Box<str>>,
            _limit: Option<u32>,
        ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<semantic_code_ports::VectorDbRow>>>
        {
            Box::pin(async move { Ok(Vec::new()) })
        }
    }

    impl NoopVectorDb {
        fn new(has_collection: bool) -> Result<Self> {
            Ok(Self {
                provider: VectorDbProviderInfo {
                    id: VectorDbProviderId::parse("milvus_grpc").map_err(ErrorEnvelope::from)?,
                    name: "noop".into(),
                },
                has_collection,
                drop_called: Arc::new(AtomicBool::new(false)),
            })
        }
    }

    impl VectorDbPort for NoopVectorDb {
        fn provider(&self) -> &VectorDbProviderInfo {
            &self.provider
        }

        fn create_collection(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _dimension: u32,
            _description: Option<Box<str>>,
        ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
            Box::pin(async move { Ok(()) })
        }

        fn create_hybrid_collection(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _dimension: u32,
            _description: Option<Box<str>>,
        ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
            Box::pin(async move { Ok(()) })
        }

        fn drop_collection(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
        ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
            let called = self.drop_called.clone();
            Box::pin(async move {
                called.store(true, Ordering::SeqCst);
                Ok(())
            })
        }

        fn has_collection(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
        ) -> semantic_code_ports::BoxFuture<'_, Result<bool>> {
            let value = self.has_collection;
            Box::pin(async move { Ok(value) })
        }

        fn list_collections(
            &self,
            _ctx: &RequestContext,
        ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<CollectionName>>> {
            Box::pin(async move { Ok(Vec::new()) })
        }

        fn insert(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _documents: Vec<VectorDocumentForInsert>,
        ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
            Box::pin(async move { Ok(()) })
        }

        fn insert_hybrid(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _documents: Vec<VectorDocumentForInsert>,
        ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
            Box::pin(async move { Ok(()) })
        }

        fn search(
            &self,
            _ctx: &RequestContext,
            _request: semantic_code_ports::VectorSearchRequest,
        ) -> semantic_code_ports::BoxFuture<'_, Result<VectorSearchResponse>> {
            Box::pin(async move {
                Ok(VectorSearchResponse {
                    results: Vec::new(),
                    stats: None,
                })
            })
        }

        fn hybrid_search(
            &self,
            _ctx: &RequestContext,
            _request: semantic_code_ports::HybridSearchBatchRequest,
        ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<semantic_code_ports::HybridSearchResult>>>
        {
            Box::pin(async move { Ok(Vec::new()) })
        }

        fn delete(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _ids: Vec<Box<str>>,
        ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
            Box::pin(async move { Ok(()) })
        }

        fn query(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _filter: Box<str>,
            _output_fields: Vec<Box<str>>,
            _limit: Option<u32>,
        ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<semantic_code_ports::VectorDbRow>>>
        {
            Box::pin(async move { Ok(Vec::new()) })
        }
    }

    #[derive(Clone, Default)]
    struct TestFileSync {
        deleted: Arc<AtomicBool>,
    }

    impl FileSyncPort for TestFileSync {
        fn initialize(
            &self,
            _ctx: &RequestContext,
            _options: semantic_code_ports::FileSyncInitOptions,
        ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
            Box::pin(async move { Ok(()) })
        }

        fn check_for_changes(
            &self,
            _ctx: &RequestContext,
            _options: FileSyncOptions,
        ) -> semantic_code_ports::BoxFuture<'_, Result<semantic_code_ports::FileChangeSet>>
        {
            Box::pin(async move { Ok(semantic_code_ports::FileChangeSet::default()) })
        }

        fn delete_snapshot(
            &self,
            _ctx: &RequestContext,
            _codebase_root: PathBuf,
        ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
            let deleted = self.deleted.clone();
            Box::pin(async move {
                deleted.store(true, Ordering::SeqCst);
                Ok(())
            })
        }
    }

    #[tokio::test]
    async fn clear_index_respects_cancellation() -> Result<()> {
        let ctx = RequestContext::new_request();
        ctx.cancel();

        let vectordb = Arc::new(NoopVectorDb::new(true)?);
        let file_sync = Arc::new(TestFileSync::default());
        let deps = ClearIndexDeps {
            vectordb,
            file_sync,
            logger: None,
            telemetry: None,
        };
        let input = ClearIndexInput {
            codebase_root: PathBuf::from("/tmp/repo"),
            collection_name: CollectionName::parse("code_chunks_test")
                .map_err(ErrorEnvelope::from)?,
        };

        let result = clear_index(&ctx, &deps, input).await;
        assert!(matches!(result, Err(error) if error.is_cancelled()));
        Ok(())
    }

    #[tokio::test]
    async fn clear_index_skips_drop_when_missing() -> Result<()> {
        let ctx = RequestContext::new_request();
        let vectordb = Arc::new(NoopVectorDb::new(false)?);
        let file_sync = Arc::new(TestFileSync::default());
        let deps = ClearIndexDeps {
            vectordb: vectordb.clone(),
            file_sync: file_sync.clone(),
            logger: None,
            telemetry: None,
        };

        let input = ClearIndexInput {
            codebase_root: PathBuf::from("/tmp/repo"),
            collection_name: CollectionName::parse("code_chunks_test")
                .map_err(ErrorEnvelope::from)?,
        };

        clear_index(&ctx, &deps, input).await?;
        assert!(!vectordb.drop_called.load(Ordering::SeqCst));
        assert!(file_sync.deleted.load(Ordering::SeqCst));
        Ok(())
    }

    #[tokio::test]
    async fn clear_index_retries_transient_milvus_drop_collection_error() -> Result<()> {
        let ctx = RequestContext::new_request();
        let vectordb = Arc::new(FlakyMilvusDropVectorDb::new(2)?);
        let file_sync = Arc::new(TestFileSync::default());
        let deps = ClearIndexDeps {
            vectordb: vectordb.clone(),
            file_sync: file_sync.clone(),
            logger: None,
            telemetry: None,
        };
        let input = ClearIndexInput {
            codebase_root: PathBuf::from("/tmp/repo"),
            collection_name: CollectionName::parse("code_chunks_retry")
                .map_err(ErrorEnvelope::from)?,
        };

        clear_index(&ctx, &deps, input).await?;
        assert_eq!(vectordb.attempts.load(Ordering::SeqCst), 3);
        assert!(file_sync.deleted.load(Ordering::SeqCst));
        Ok(())
    }
}
