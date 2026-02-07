//! Integration tests for clear index use case.

use semantic_code_app::{ClearIndexDeps, ClearIndexInput, clear_index};
use semantic_code_domain::{CollectionName, VectorDbProviderId};
use semantic_code_ports::{FileSyncPort, VectorDbPort, VectorDbProviderInfo};
use semantic_code_shared::{ErrorEnvelope, RequestContext, Result};
use semantic_code_testkit::in_memory::InMemoryVectorDbFixed;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[tokio::test]
async fn clear_index_drops_collection_and_snapshot() -> Result<()> {
    let ctx = RequestContext::new_request();
    let vectordb = Arc::new(InMemoryVectorDbFixed::<8>::new(VectorDbProviderInfo {
        id: VectorDbProviderId::parse("milvus_grpc").map_err(ErrorEnvelope::from)?,
        name: "in-memory".into(),
    }));
    let collection = CollectionName::parse("code_chunks_clear").map_err(ErrorEnvelope::from)?;
    vectordb
        .create_collection(&ctx, collection.clone(), 8, None)
        .await?;

    let file_sync = Arc::new(TrackingFileSync::default());
    let deps = ClearIndexDeps {
        vectordb: vectordb.clone(),
        file_sync: file_sync.clone(),
        logger: None,
        telemetry: None,
    };
    let input = ClearIndexInput {
        codebase_root: PathBuf::from("/tmp/repo"),
        collection_name: collection.clone(),
    };

    clear_index(&ctx, &deps, input).await?;
    let exists = vectordb.has_collection(&ctx, collection).await?;
    assert!(!exists);
    assert!(file_sync.deleted.load(Ordering::SeqCst));
    Ok(())
}

#[derive(Clone, Default)]
struct TrackingFileSync {
    deleted: Arc<AtomicBool>,
}

impl FileSyncPort for TrackingFileSync {
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
        _options: semantic_code_ports::FileSyncOptions,
    ) -> semantic_code_ports::BoxFuture<'_, Result<semantic_code_ports::FileChangeSet>> {
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
