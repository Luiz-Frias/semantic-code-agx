//! Clear an index collection and associated sync snapshot.

use semantic_code_domain::CollectionName;
use semantic_code_ports::{FileSyncPort, LoggerPort, TelemetryPort, VectorDbPort};
use semantic_code_shared::{ErrorEnvelope, RequestContext, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

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
        if let Some(timer) = has_collection_timer.as_ref() {
            timer.stop();
        }

        if exists {
            ctx.ensure_not_cancelled("clear_index.drop_collection")?;
            let drop_tags = tags_provider(deps.vectordb.provider().id.as_str());
            let drop_timer = deps.telemetry.as_ref().map(|telemetry| {
                telemetry.start_timer("backend.clearIndex.dropCollection", Some(&drop_tags))
            });
            deps.vectordb
                .drop_collection(ctx, input.collection_name.clone())
                .await?;
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
                        Some(log_fields_error(&input, duration_ms, &error)),
                    );
                }
            }
            Err(error)
        },
    }
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
    };
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Clone)]
    struct NoopVectorDb {
        provider: VectorDbProviderInfo,
        has_collection: bool,
        drop_called: Arc<AtomicBool>,
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
        ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<semantic_code_ports::VectorSearchResult>>>
        {
            Box::pin(async move { Ok(Vec::new()) })
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
}
