//! In-memory index smoke test used by CLI self-check.

use crate::InfraResult;
use semantic_code_adapters::self_check::{
    SelfCheckEmbedding, SelfCheckFileSync, SelfCheckFileSystem, SelfCheckIgnore,
    SelfCheckPathPolicy, SelfCheckSplitter, SelfCheckVectorDb,
};
use semantic_code_app::{
    ClearIndexDeps, ClearIndexInput, IndexCodebaseDeps, IndexCodebaseInput, IndexCodebaseStatus,
    SemanticSearchDeps, SemanticSearchInput, clear_index, index_codebase, semantic_search,
};
use semantic_code_domain::{CollectionName, IndexMode};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;

/// Run a minimal in-memory index smoke test.
pub fn run_index_smoke() -> InfraResult<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(ErrorEnvelope::from)?;

    runtime.block_on(async {
        let embedding = Arc::new(SelfCheckEmbedding::new()?);
        let vectordb = Arc::new(SelfCheckVectorDb::new()?);

        let deps = IndexCodebaseDeps {
            embedding,
            vectordb,
            splitter: Arc::new(SelfCheckSplitter),
            filesystem: Arc::new(SelfCheckFileSystem::new()),
            path_policy: Arc::new(SelfCheckPathPolicy),
            ignore: Arc::new(SelfCheckIgnore),
            logger: None,
            telemetry: None,
        };

        let input = IndexCodebaseInput {
            codebase_root: PathBuf::from("/tmp/self-check"),
            collection_name: CollectionName::parse("code_chunks_self_check")
                .map_err(ErrorEnvelope::from)?,
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
            max_in_flight_files: Some(NonZeroUsize::MIN),
            max_in_flight_embedding_batches: Some(NonZeroUsize::MIN),
            max_in_flight_inserts: Some(NonZeroUsize::MIN),
        };

        let ctx = RequestContext::new_request();
        let output = index_codebase(&ctx, &deps, input).await?;

        if output.status != IndexCodebaseStatus::Completed {
            return Err(ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "index smoke did not complete",
                ErrorClass::NonRetriable,
            ));
        }

        Ok(())
    })
}

/// Run a minimal in-memory semantic search smoke test.
pub fn run_search_smoke() -> InfraResult<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(ErrorEnvelope::from)?;

    runtime.block_on(async {
        let embedding = Arc::new(SelfCheckEmbedding::new()?);
        let vectordb = Arc::new(SelfCheckVectorDb::new()?);

        let collection =
            CollectionName::parse("code_chunks_self_check").map_err(ErrorEnvelope::from)?;

        let index_deps = IndexCodebaseDeps {
            embedding: embedding.clone(),
            vectordb: vectordb.clone(),
            splitter: Arc::new(SelfCheckSplitter),
            filesystem: Arc::new(SelfCheckFileSystem::new()),
            path_policy: Arc::new(SelfCheckPathPolicy),
            ignore: Arc::new(SelfCheckIgnore),
            logger: None,
            telemetry: None,
        };

        let index_input = IndexCodebaseInput {
            codebase_root: PathBuf::from("/tmp/self-check"),
            collection_name: collection.clone(),
            index_mode: IndexMode::Dense,
            supported_extensions: Some(vec![".rs".into()]),
            ignore_patterns: None,
            file_list: None,
            force_reindex: true,
            on_progress: None,
            embedding_batch_size: NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN),
            chunk_limit: NonZeroUsize::new(50).unwrap_or(NonZeroUsize::MIN),
            max_files: None,
            max_file_size_bytes: None,
            max_buffered_chunks: None,
            max_buffered_embeddings: None,
            max_in_flight_files: Some(NonZeroUsize::MIN),
            max_in_flight_embedding_batches: Some(NonZeroUsize::MIN),
            max_in_flight_inserts: Some(NonZeroUsize::MIN),
        };

        let ctx = RequestContext::new_request();
        let output = index_codebase(&ctx, &index_deps, index_input).await?;
        if output.status != IndexCodebaseStatus::Completed {
            return Err(ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "index smoke did not complete",
                ErrorClass::NonRetriable,
            ));
        }

        let deps = SemanticSearchDeps {
            embedding,
            vectordb,
            logger: None,
            telemetry: None,
        };

        let input = SemanticSearchInput {
            codebase_root: "/tmp/self-check".into(),
            collection_name: collection,
            index_mode: IndexMode::Dense,
            query: "ok".into(),
            top_k: Some(3),
            threshold: Some(0.0),
        };

        let results = semantic_search(&ctx, &deps, input).await?;
        if results.is_empty() {
            return Err(ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "search smoke returned no results",
                ErrorClass::NonRetriable,
            ));
        }

        Ok(())
    })
}

/// Run a minimal in-memory clear-index smoke test.
pub fn run_clear_smoke() -> InfraResult<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(ErrorEnvelope::from)?;

    runtime.block_on(async {
        let embedding = Arc::new(SelfCheckEmbedding::new()?);
        let vectordb = Arc::new(SelfCheckVectorDb::new()?);
        let file_sync = Arc::new(SelfCheckFileSync::new());

        let collection =
            CollectionName::parse("code_chunks_self_check").map_err(ErrorEnvelope::from)?;

        let index_deps = IndexCodebaseDeps {
            embedding,
            vectordb: vectordb.clone(),
            splitter: Arc::new(SelfCheckSplitter),
            filesystem: Arc::new(SelfCheckFileSystem::new()),
            path_policy: Arc::new(SelfCheckPathPolicy),
            ignore: Arc::new(SelfCheckIgnore),
            logger: None,
            telemetry: None,
        };

        let index_input = IndexCodebaseInput {
            codebase_root: PathBuf::from("/tmp/self-check"),
            collection_name: collection.clone(),
            index_mode: IndexMode::Dense,
            supported_extensions: Some(vec![".rs".into()]),
            ignore_patterns: None,
            file_list: None,
            force_reindex: true,
            on_progress: None,
            embedding_batch_size: NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN),
            chunk_limit: NonZeroUsize::new(50).unwrap_or(NonZeroUsize::MIN),
            max_files: None,
            max_file_size_bytes: None,
            max_buffered_chunks: None,
            max_buffered_embeddings: None,
            max_in_flight_files: Some(NonZeroUsize::MIN),
            max_in_flight_embedding_batches: Some(NonZeroUsize::MIN),
            max_in_flight_inserts: Some(NonZeroUsize::MIN),
        };

        let ctx = RequestContext::new_request();
        let output = index_codebase(&ctx, &index_deps, index_input).await?;
        if output.status != IndexCodebaseStatus::Completed {
            return Err(ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "index smoke did not complete",
                ErrorClass::NonRetriable,
            ));
        }

        let deps = ClearIndexDeps {
            vectordb,
            file_sync,
            logger: None,
            telemetry: None,
        };

        let input = ClearIndexInput {
            codebase_root: PathBuf::from("/tmp/self-check"),
            collection_name: collection,
        };

        clear_index(&ctx, &deps, input).await?;
        Ok(())
    })
}
