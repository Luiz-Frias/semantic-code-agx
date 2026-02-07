//! Integration test for local vector DB adapter.

use semantic_code_adapters::fs::LocalFileSystem;
use semantic_code_adapters::ignore::IgnoreMatcher;
use semantic_code_adapters::splitter::TreeSitterSplitter;
use semantic_code_adapters::vectordb_local::LocalVectorDb;
use semantic_code_app::{
    IndexCodebaseDeps, IndexCodebaseInput, IndexCodebaseStatus, SemanticSearchDeps,
    SemanticSearchInput, index_codebase, semantic_search,
};
use semantic_code_config::SnapshotStorageMode;
use semantic_code_domain::{CollectionName, EmbeddingProviderId, IndexMode};
use semantic_code_ports::{EmbeddingPort, EmbeddingProviderInfo, VectorDbPort};
use semantic_code_shared::{ErrorEnvelope, RequestContext, Result};
use semantic_code_testkit::in_memory::{InMemoryEmbeddingFixed, NoopLogger, NoopTelemetry};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn workspace_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_dir.to_path_buf())
}

#[tokio::test]
async fn index_and_search_with_local_vectordb() -> Result<()> {
    let codebase_root = workspace_root()
        .join("crates")
        .join("testkit")
        .join("fixtures")
        .join("local-index")
        .join("basic");

    let embedding: Arc<dyn EmbeddingPort> =
        Arc::new(InMemoryEmbeddingFixed::<8>::new(EmbeddingProviderInfo {
            id: EmbeddingProviderId::parse("fixture").map_err(ErrorEnvelope::from)?,
            name: "fixture".into(),
        }));
    let vectordb: Arc<dyn VectorDbPort> = Arc::new(LocalVectorDb::new(
        codebase_root.clone(),
        SnapshotStorageMode::Disabled,
    )?);

    let deps = IndexCodebaseDeps {
        embedding: Arc::clone(&embedding),
        vectordb: Arc::clone(&vectordb),
        splitter: Arc::new(TreeSitterSplitter::default()),
        filesystem: Arc::new(LocalFileSystem::new(None)),
        path_policy: Arc::new(semantic_code_adapters::fs::LocalPathPolicy::new()),
        ignore: Arc::new(IgnoreMatcher::new()),
        logger: Some(Arc::new(NoopLogger::default())),
        telemetry: Some(Arc::new(NoopTelemetry::default())),
    };

    let input = IndexCodebaseInput {
        codebase_root: codebase_root.clone(),
        collection_name: CollectionName::parse("code_chunks_local").map_err(ErrorEnvelope::from)?,
        index_mode: IndexMode::Dense,
        supported_extensions: Some(vec![".rs".into()]),
        ignore_patterns: None,
        file_list: None,
        force_reindex: true,
        on_progress: None,
        embedding_batch_size: NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN),
        chunk_limit: NonZeroUsize::new(100).unwrap_or(NonZeroUsize::MIN),
        max_files: None,
        max_file_size_bytes: None,
        max_buffered_chunks: None,
        max_buffered_embeddings: None,
        max_in_flight_files: Some(NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN)),
        max_in_flight_embedding_batches: Some(NonZeroUsize::MIN),
        max_in_flight_inserts: Some(NonZeroUsize::MIN),
    };

    let ctx = RequestContext::new_request();
    let output = index_codebase(&ctx, &deps, input).await?;
    assert_eq!(output.status, IndexCodebaseStatus::Completed);

    let search_deps = SemanticSearchDeps {
        embedding,
        vectordb,
        logger: None,
        telemetry: None,
    };
    let results = semantic_search(
        &ctx,
        &search_deps,
        SemanticSearchInput {
            codebase_root: codebase_root.display().to_string().into_boxed_str(),
            collection_name: CollectionName::parse("code_chunks_local")
                .map_err(ErrorEnvelope::from)?,
            index_mode: IndexMode::Dense,
            query: "local-index".into(),
            top_k: Some(3),
            threshold: Some(0.1),
        },
    )
    .await?;

    assert!(!results.is_empty());
    Ok(())
}
