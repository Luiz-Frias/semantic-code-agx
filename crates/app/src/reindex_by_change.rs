//! Reindex changed files by diffing snapshots.

use crate::index_codebase::{
    IndexCodebaseInput, IndexProgress, delete_modified_files, delete_removed_files, detect_changes,
    emit_progress, index_codebase, total_changes,
};
use semantic_code_domain::{CollectionName, IndexMode};
use semantic_code_ports::{
    EmbeddingPort, FileChangeSet, FileSyncPort, FileSystemPort, IgnorePort, LoggerPort,
    PathPolicyPort, SplitterPort, TelemetryPort, VectorDbPort,
};
use semantic_code_shared::{ErrorEnvelope, RequestContext, Result};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

// TODO: refactor repeated optional logger/telemetry checks with a helper mapper.
/// Input payload for reindex-by-change.
#[derive(Clone)]
pub struct ReindexByChangeInput {
    /// Codebase root directory (absolute path).
    pub codebase_root: PathBuf,
    /// Target collection name.
    pub collection_name: CollectionName,
    /// Index mode (dense or hybrid).
    pub index_mode: IndexMode,
    /// Allowed file extensions (normalized, optional).
    pub supported_extensions: Option<Vec<Box<str>>>,
    /// Ignore patterns (normalized, optional).
    pub ignore_patterns: Option<Vec<Box<str>>>,
    /// Embedding batch size (chunks per batch).
    pub embedding_batch_size: NonZeroUsize,
    /// Maximum number of chunks to index.
    pub chunk_limit: NonZeroUsize,
    /// Maximum number of files to scan.
    pub max_files: Option<NonZeroUsize>,
    /// Skip files larger than this size.
    pub max_file_size_bytes: Option<u64>,
    /// Maximum buffered chunks (best-effort).
    pub max_buffered_chunks: Option<NonZeroUsize>,
    /// Maximum buffered embeddings (best-effort).
    pub max_buffered_embeddings: Option<NonZeroUsize>,
    /// Max in-flight file tasks (default 1).
    pub max_in_flight_files: Option<NonZeroUsize>,
    /// Max in-flight embedding batches (default 1).
    pub max_in_flight_embedding_batches: Option<NonZeroUsize>,
    /// Max in-flight insert batches (default 1).
    pub max_in_flight_inserts: Option<NonZeroUsize>,
    /// Optional progress callback.
    pub on_progress: Option<Arc<dyn Fn(IndexProgress) + Send + Sync>>,
}

/// Output returned by reindex-by-change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReindexByChangeOutput {
    /// Added files count.
    pub added: usize,
    /// Removed files count.
    pub removed: usize,
    /// Modified files count.
    pub modified: usize,
}

/// Dependencies required by reindex-by-change.
#[derive(Clone)]
pub struct ReindexByChangeDeps {
    /// File sync adapter.
    pub file_sync: Arc<dyn FileSyncPort>,
    /// Vector DB adapter.
    pub vectordb: Arc<dyn VectorDbPort>,
    /// Embedding adapter.
    pub embedding: Arc<dyn EmbeddingPort>,
    /// Splitter adapter.
    pub splitter: Arc<dyn SplitterPort>,
    /// Filesystem adapter.
    pub filesystem: Arc<dyn FileSystemPort>,
    /// Path policy adapter.
    pub path_policy: Arc<dyn PathPolicyPort>,
    /// Ignore matcher.
    pub ignore: Arc<dyn IgnorePort>,
    /// Optional logger.
    pub logger: Option<Arc<dyn LoggerPort>>,
    /// Optional telemetry sink.
    pub telemetry: Option<Arc<dyn TelemetryPort>>,
}

/// Reindex files based on snapshot changes.
pub async fn reindex_by_change(
    ctx: &RequestContext,
    deps: &ReindexByChangeDeps,
    input: ReindexByChangeInput,
) -> Result<ReindexByChangeOutput> {
    let started_at = Instant::now();
    let total_tags = tags_index_mode(input.index_mode);
    let total_timer = deps
        .telemetry
        .as_ref()
        .map(|telemetry| telemetry.start_timer("backend.reindex.total", Some(&total_tags)));

    if let Some(logger) = deps.logger.as_ref() {
        logger.info(
            "backend.reindex.start",
            "Reindex-by-change started",
            Some(log_fields_start(&input)),
        );
    }

    let result = run_reindex(ctx, deps, &input, started_at).await;

    if let Some(timer) = total_timer.as_ref() {
        timer.stop();
    }

    match result {
        Ok(output) => Ok(output),
        Err(error) => {
            let duration_ms = duration_ms(started_at);
            if error.is_cancelled() {
                if let Some(telemetry) = deps.telemetry.as_ref() {
                    telemetry.increment_counter(
                        "backend.reindex.aborted",
                        1,
                        Some(&tags_index_mode(input.index_mode)),
                    );
                }
                if let Some(logger) = deps.logger.as_ref() {
                    logger.info(
                        "backend.reindex.aborted",
                        "Reindex-by-change aborted",
                        Some(log_fields_abort(input.index_mode, duration_ms)),
                    );
                }
            } else {
                if let Some(telemetry) = deps.telemetry.as_ref() {
                    telemetry.increment_counter(
                        "backend.reindex.failed",
                        1,
                        Some(&tags_index_mode(input.index_mode)),
                    );
                }
                if let Some(logger) = deps.logger.as_ref() {
                    logger.error(
                        "backend.reindex.failed",
                        "Reindex-by-change failed",
                        Some(log_fields_error(&input, duration_ms, &error)),
                    );
                }
            }
            Err(error)
        },
    }
}

struct ReindexPipeline<'a> {
    ctx: &'a RequestContext,
    deps: &'a ReindexByChangeDeps,
    input: &'a ReindexByChangeInput,
}

impl<'a> ReindexPipeline<'a> {
    const fn new(
        ctx: &'a RequestContext,
        deps: &'a ReindexByChangeDeps,
        input: &'a ReindexByChangeInput,
    ) -> Self {
        Self { ctx, deps, input }
    }

    async fn detect(self) -> Result<ReindexDetected<'a>> {
        let changes = detect_changes(self.ctx, self.deps, self.input).await?;
        let total = total_changes(&changes);
        Ok(ReindexDetected {
            ctx: self.ctx,
            deps: self.deps,
            input: self.input,
            changes,
            total,
            processed: 0,
        })
    }
}

struct ReindexDetected<'a> {
    ctx: &'a RequestContext,
    deps: &'a ReindexByChangeDeps,
    input: &'a ReindexByChangeInput,
    changes: FileChangeSet,
    total: usize,
    processed: usize,
}

impl<'a> ReindexDetected<'a> {
    async fn delete_removed(mut self) -> Result<ReindexRemoved<'a>> {
        delete_removed_files(
            self.ctx,
            self.deps,
            self.input,
            &self.changes,
            self.total,
            &mut self.processed,
        )
        .await?;
        Ok(ReindexRemoved {
            ctx: self.ctx,
            deps: self.deps,
            input: self.input,
            changes: self.changes,
            total: self.total,
            processed: self.processed,
        })
    }
}

struct ReindexRemoved<'a> {
    ctx: &'a RequestContext,
    deps: &'a ReindexByChangeDeps,
    input: &'a ReindexByChangeInput,
    changes: FileChangeSet,
    total: usize,
    processed: usize,
}

impl<'a> ReindexRemoved<'a> {
    async fn delete_modified(mut self) -> Result<ReindexModified<'a>> {
        delete_modified_files(
            self.ctx,
            self.deps,
            self.input,
            &self.changes,
            self.total,
            &mut self.processed,
        )
        .await?;
        Ok(ReindexModified {
            ctx: self.ctx,
            deps: self.deps,
            input: self.input,
            changes: self.changes,
        })
    }
}

struct ReindexModified<'a> {
    ctx: &'a RequestContext,
    deps: &'a ReindexByChangeDeps,
    input: &'a ReindexByChangeInput,
    changes: FileChangeSet,
}

impl ReindexModified<'_> {
    async fn reindex_changed(self) -> Result<ReindexCompleted> {
        reindex_changed_files(self.ctx, self.deps, self.input, &self.changes).await?;
        Ok(ReindexCompleted {
            changes: self.changes,
        })
    }
}

struct ReindexCompleted {
    changes: FileChangeSet,
}

impl ReindexCompleted {
    const fn output(&self) -> ReindexByChangeOutput {
        ReindexByChangeOutput {
            added: self.changes.added.len(),
            removed: self.changes.removed.len(),
            modified: self.changes.modified.len(),
        }
    }
}

async fn run_reindex(
    ctx: &RequestContext,
    deps: &ReindexByChangeDeps,
    input: &ReindexByChangeInput,
    started_at: Instant,
) -> Result<ReindexByChangeOutput> {
    ctx.ensure_not_cancelled("reindex_by_change.start")?;

    emit_progress(
        input.on_progress.as_ref(),
        "Checking for file changes...",
        0,
        100,
        Some(0),
    );

    let detected = ReindexPipeline::new(ctx, deps, input).detect().await?;
    if detected.total == 0 {
        emit_progress(
            input.on_progress.as_ref(),
            "No changes detected",
            100,
            100,
            Some(100),
        );
        if let Some(logger) = deps.logger.as_ref() {
            logger.info(
                "backend.reindex.completed",
                "Reindex-by-change completed",
                Some(log_fields_completed(input, started_at, 0, 0, 0)),
            );
        }
        return Ok(ReindexByChangeOutput {
            added: 0,
            removed: 0,
            modified: 0,
        });
    }

    let total = detected.total;
    let removed = detected.delete_removed().await?;
    let modified = removed.delete_modified().await?;
    let completed = modified.reindex_changed().await?;
    let changes = &completed.changes;

    emit_progress(
        input.on_progress.as_ref(),
        "Re-indexing complete!",
        total as u64,
        total as u64,
        Some(100),
    );

    if let Some(telemetry) = deps.telemetry.as_ref() {
        let count_tags = tags_counts(
            changes.added.len(),
            changes.removed.len(),
            changes.modified.len(),
        );
        telemetry.increment_counter("backend.reindexByChange.executed", 1, Some(&count_tags));
    }

    if let Some(logger) = deps.logger.as_ref() {
        logger.info(
            "backend.reindex.completed",
            "Reindex-by-change completed",
            Some(log_fields_completed(
                input,
                started_at,
                changes.added.len(),
                changes.removed.len(),
                changes.modified.len(),
            )),
        );
    }

    Ok(completed.output())
}

async fn reindex_changed_files(
    ctx: &RequestContext,
    deps: &ReindexByChangeDeps,
    input: &ReindexByChangeInput,
    changes: &FileChangeSet,
) -> Result<()> {
    let files_to_index = files_to_index(&changes.added, &changes.modified);
    if files_to_index.is_empty() {
        return Ok(());
    }

    let file_count_tags = tags_file_count(input.index_mode, files_to_index.len());
    let index_timer = deps.telemetry.as_ref().map(|telemetry| {
        telemetry.start_timer("backend.reindex.indexChangedFiles", Some(&file_count_tags))
    });

    let index_input = IndexCodebaseInput {
        codebase_root: input.codebase_root.clone(),
        collection_name: input.collection_name.clone(),
        index_mode: input.index_mode,
        supported_extensions: input.supported_extensions.clone(),
        ignore_patterns: input.ignore_patterns.clone(),
        file_list: Some(files_to_index),
        force_reindex: false,
        on_progress: None,
        embedding_batch_size: input.embedding_batch_size,
        chunk_limit: input.chunk_limit,
        max_files: input.max_files,
        max_file_size_bytes: input.max_file_size_bytes,
        max_buffered_chunks: input.max_buffered_chunks,
        max_buffered_embeddings: input.max_buffered_embeddings,
        max_in_flight_files: input.max_in_flight_files,
        max_in_flight_embedding_batches: input.max_in_flight_embedding_batches,
        max_in_flight_inserts: input.max_in_flight_inserts,
    };

    let index_deps = crate::index_codebase::IndexCodebaseDeps {
        embedding: deps.embedding.clone(),
        vectordb: deps.vectordb.clone(),
        splitter: deps.splitter.clone(),
        filesystem: deps.filesystem.clone(),
        path_policy: deps.path_policy.clone(),
        ignore: deps.ignore.clone(),
        logger: deps.logger.clone(),
        telemetry: deps.telemetry.clone(),
    };

    let _ = index_codebase(ctx, &index_deps, index_input).await?;
    if let Some(timer) = index_timer.as_ref() {
        timer.stop();
    }

    Ok(())
}

fn files_to_index(added: &[Box<str>], modified: &[Box<str>]) -> Vec<Box<str>> {
    let mut set = HashSet::new();
    for path in added.iter().chain(modified.iter()) {
        set.insert(path.clone());
    }
    let mut files = set.into_iter().collect::<Vec<_>>();
    files.sort();
    files
}

fn duration_ms(started_at: Instant) -> u64 {
    u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn tags_index_mode(index_mode: IndexMode) -> BTreeMap<Box<str>, Box<str>> {
    let mut tags = BTreeMap::new();
    tags.insert(
        "indexMode".to_owned().into_boxed_str(),
        index_mode.as_str().to_owned().into_boxed_str(),
    );
    tags
}

fn tags_file_count(index_mode: IndexMode, count: usize) -> BTreeMap<Box<str>, Box<str>> {
    let mut tags = tags_index_mode(index_mode);
    tags.insert(
        "fileCount".to_owned().into_boxed_str(),
        count.to_string().into_boxed_str(),
    );
    tags
}

fn tags_counts(added: usize, removed: usize, modified: usize) -> BTreeMap<Box<str>, Box<str>> {
    let mut tags = BTreeMap::new();
    tags.insert(
        "added".to_owned().into_boxed_str(),
        added.to_string().into_boxed_str(),
    );
    tags.insert(
        "removed".to_owned().into_boxed_str(),
        removed.to_string().into_boxed_str(),
    );
    tags.insert(
        "modified".to_owned().into_boxed_str(),
        modified.to_string().into_boxed_str(),
    );
    tags
}

fn log_fields_start(input: &ReindexByChangeInput) -> BTreeMap<Box<str>, Value> {
    let mut fields = BTreeMap::new();
    fields.insert(
        "codebaseRoot".to_owned().into_boxed_str(),
        Value::String(input.codebase_root.to_string_lossy().to_string()),
    );
    fields.insert(
        "collectionName".to_owned().into_boxed_str(),
        Value::String(input.collection_name.as_str().to_owned()),
    );
    fields.insert(
        "indexMode".to_owned().into_boxed_str(),
        Value::String(input.index_mode.as_str().to_owned()),
    );
    fields
}

fn log_fields_completed(
    input: &ReindexByChangeInput,
    started_at: Instant,
    added: usize,
    removed: usize,
    modified: usize,
) -> BTreeMap<Box<str>, Value> {
    let mut fields = log_fields_start(input);
    fields.insert(
        "durationMs".to_owned().into_boxed_str(),
        Value::from(duration_ms(started_at)),
    );
    fields.insert("added".to_owned().into_boxed_str(), Value::from(added));
    fields.insert("removed".to_owned().into_boxed_str(), Value::from(removed));
    fields.insert(
        "modified".to_owned().into_boxed_str(),
        Value::from(modified),
    );
    fields
}

fn log_fields_abort(index_mode: IndexMode, duration_ms: u64) -> BTreeMap<Box<str>, Value> {
    let mut fields = BTreeMap::new();
    fields.insert(
        "indexMode".to_owned().into_boxed_str(),
        Value::String(index_mode.as_str().to_owned()),
    );
    fields.insert(
        "durationMs".to_owned().into_boxed_str(),
        Value::from(duration_ms),
    );
    fields
}

fn log_fields_error(
    input: &ReindexByChangeInput,
    duration_ms: u64,
    error: &ErrorEnvelope,
) -> BTreeMap<Box<str>, Value> {
    let mut fields = log_fields_abort(input.index_mode, duration_ms);
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
    use crate::index_codebase::{delete_file_chunks_by_relative_path, normalize_change_set};
    use semantic_code_domain::{EmbeddingProviderId, VectorDbProviderId};
    use semantic_code_ports::{
        CollectionName, DetectDimensionRequest, EmbedBatchRequest, EmbedRequest,
        EmbeddingProviderInfo, EmbeddingVector, FileChangeSet, FileSyncInitOptions,
        FileSyncOptions, HybridSearchBatchRequest, VectorDbProviderInfo, VectorDbRow,
        VectorDocumentForInsert, VectorSearchRequest, VectorSearchResult,
    };
    use semantic_code_shared::{ErrorClass, ErrorCode};
    use std::num::NonZeroUsize;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    #[test]
    fn change_set_is_sorted_and_deduped() {
        let changes = FileChangeSet {
            added: vec!["b.rs".into(), "a.rs".into(), "a.rs".into()],
            removed: vec!["z.rs".into(), "m.rs".into()],
            modified: vec!["c.rs".into(), "b.rs".into()],
        };
        let normalized = normalize_change_set(changes);
        assert_eq!(normalized.added, vec!["a.rs".into(), "b.rs".into()]);
        assert_eq!(normalized.modified, vec!["b.rs".into(), "c.rs".into()]);
    }

    #[tokio::test]
    async fn delete_by_relative_path_queries_and_deletes() -> Result<()> {
        let vectordb = Arc::new(SpyVectorDb::new()?);
        let deps = ReindexByChangeDeps {
            file_sync: Arc::new(NoopFileSync),
            vectordb: vectordb.clone(),
            embedding: Arc::new(NoopEmbedding::new()?),
            splitter: Arc::new(NoopSplitter),
            filesystem: Arc::new(NoopFileSystem),
            path_policy: Arc::new(NoopPathPolicy),
            ignore: Arc::new(NoopIgnore),
            logger: None,
            telemetry: None,
        };
        let ctx = RequestContext::new_request();
        delete_file_chunks_by_relative_path(
            &ctx,
            &deps,
            CollectionName::parse("code_chunks_test").map_err(ErrorEnvelope::from)?,
            "src/lib.rs",
        )
        .await?;

        let state = vectordb.state.lock().map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "spy vectordb lock poisoned",
                ErrorClass::NonRetriable,
            )
        })?;
        assert_eq!(
            state.last_filter.as_deref(),
            Some("relativePath == \"src/lib.rs\"")
        );
        assert_eq!(state.deleted_ids, vec!["chunk_a".into(), "chunk_b".into()]);
        Ok(())
    }

    #[tokio::test]
    async fn progress_updates_complete() -> Result<()> {
        let file_sync = Arc::new(StaticFileSync::new(FileChangeSet {
            added: Vec::new(),
            removed: vec!["old.rs".into()],
            modified: Vec::new(),
        }));
        let vectordb = Arc::new(SpyVectorDb::new()?);
        let deps = ReindexByChangeDeps {
            file_sync,
            vectordb,
            embedding: Arc::new(NoopEmbedding::new()?),
            splitter: Arc::new(NoopSplitter),
            filesystem: Arc::new(NoopFileSystem),
            path_policy: Arc::new(NoopPathPolicy),
            ignore: Arc::new(NoopIgnore),
            logger: None,
            telemetry: None,
        };

        let progress = Arc::new(Mutex::new(Vec::new()));
        let progress_handle = progress.clone();
        let ctx = RequestContext::new_request();
        let output = reindex_by_change(
            &ctx,
            &deps,
            ReindexByChangeInput {
                codebase_root: PathBuf::from("/tmp/repo"),
                collection_name: CollectionName::parse("code_chunks_test")
                    .map_err(ErrorEnvelope::from)?,
                index_mode: IndexMode::Dense,
                supported_extensions: None,
                ignore_patterns: None,
                embedding_batch_size: NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN),
                chunk_limit: NonZeroUsize::new(100).unwrap_or(NonZeroUsize::MIN),
                max_files: None,
                max_file_size_bytes: None,
                max_buffered_chunks: None,
                max_buffered_embeddings: None,
                max_in_flight_files: None,
                max_in_flight_embedding_batches: None,
                max_in_flight_inserts: None,
                on_progress: Some(Arc::new(move |event| {
                    let mut guard = progress_handle
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner());
                    guard.push(event);
                })),
            },
        )
        .await?;

        assert_eq!(output.removed, 1);
        let guard = progress.lock().map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "progress lock poisoned",
                ErrorClass::NonRetriable,
            )
        })?;
        assert!(guard.last().is_some_and(|event| event.percentage == 100));
        Ok(())
    }

    #[derive(Clone)]
    struct SpyVectorDb {
        provider: VectorDbProviderInfo,
        state: Arc<Mutex<SpyVectorDbState>>,
    }

    #[derive(Debug, Default)]
    struct SpyVectorDbState {
        last_filter: Option<Box<str>>,
        deleted_ids: Vec<Box<str>>,
    }

    impl SpyVectorDb {
        fn new() -> Result<Self> {
            Ok(Self {
                provider: VectorDbProviderInfo {
                    id: VectorDbProviderId::parse("milvus_grpc").map_err(ErrorEnvelope::from)?,
                    name: "spy".into(),
                },
                state: Arc::new(Mutex::new(SpyVectorDbState::default())),
            })
        }
    }

    impl VectorDbPort for SpyVectorDb {
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
            Box::pin(async move { Ok(()) })
        }

        fn has_collection(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
        ) -> semantic_code_ports::BoxFuture<'_, Result<bool>> {
            Box::pin(async move { Ok(true) })
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
            _request: VectorSearchRequest,
        ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<VectorSearchResult>>> {
            Box::pin(async move { Ok(Vec::new()) })
        }

        fn hybrid_search(
            &self,
            _ctx: &RequestContext,
            _request: HybridSearchBatchRequest,
        ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<semantic_code_ports::HybridSearchResult>>>
        {
            Box::pin(async move { Ok(Vec::new()) })
        }

        fn delete(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            ids: Vec<Box<str>>,
        ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
            let state = self.state.clone();
            Box::pin(async move {
                let mut guard = state.lock().map_err(|_| {
                    ErrorEnvelope::unexpected(
                        ErrorCode::internal(),
                        "spy vectordb lock poisoned",
                        ErrorClass::NonRetriable,
                    )
                })?;
                guard.deleted_ids = ids;
                Ok(())
            })
        }

        fn query(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            filter: Box<str>,
            _output_fields: Vec<Box<str>>,
            _limit: Option<u32>,
        ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<VectorDbRow>>> {
            let state = self.state.clone();
            Box::pin(async move {
                let mut guard = state.lock().map_err(|_| {
                    ErrorEnvelope::unexpected(
                        ErrorCode::internal(),
                        "spy vectordb lock poisoned",
                        ErrorClass::NonRetriable,
                    )
                })?;
                guard.last_filter = Some(filter);
                Ok(vec![
                    row_with_id("chunk_a"),
                    row_with_id(""),
                    row_with_id("chunk_b"),
                ])
            })
        }
    }

    fn row_with_id(id: &str) -> VectorDbRow {
        let mut row = VectorDbRow::new();
        row.insert("id".into(), Value::String(id.to_owned()));
        row
    }

    #[derive(Clone)]
    struct NoopEmbedding {
        provider: EmbeddingProviderInfo,
    }

    impl NoopEmbedding {
        fn new() -> Result<Self> {
            Ok(Self {
                provider: EmbeddingProviderInfo {
                    id: EmbeddingProviderId::parse("openai").map_err(ErrorEnvelope::from)?,
                    name: "noop".into(),
                },
            })
        }
    }

    impl EmbeddingPort for NoopEmbedding {
        fn provider(&self) -> &EmbeddingProviderInfo {
            &self.provider
        }

        fn detect_dimension(
            &self,
            _ctx: &RequestContext,
            _request: DetectDimensionRequest,
        ) -> semantic_code_ports::BoxFuture<'_, Result<u32>> {
            Box::pin(async move { Ok(8) })
        }

        fn embed(
            &self,
            _ctx: &RequestContext,
            _request: EmbedRequest,
        ) -> semantic_code_ports::BoxFuture<'_, Result<EmbeddingVector>> {
            Box::pin(async move { Ok(EmbeddingVector::from_vec(vec![0.0; 8])) })
        }

        fn embed_batch(
            &self,
            _ctx: &RequestContext,
            _request: EmbedBatchRequest,
        ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<EmbeddingVector>>> {
            Box::pin(async move { Ok(Vec::new()) })
        }
    }

    #[derive(Clone)]
    struct NoopFileSync;

    impl FileSyncPort for NoopFileSync {
        fn initialize(
            &self,
            _ctx: &RequestContext,
            _options: FileSyncInitOptions,
        ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
            Box::pin(async move { Ok(()) })
        }

        fn check_for_changes(
            &self,
            _ctx: &RequestContext,
            _options: FileSyncOptions,
        ) -> semantic_code_ports::BoxFuture<'_, Result<FileChangeSet>> {
            Box::pin(async move { Ok(FileChangeSet::default()) })
        }

        fn delete_snapshot(
            &self,
            _ctx: &RequestContext,
            _codebase_root: PathBuf,
        ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
            Box::pin(async move { Ok(()) })
        }
    }

    #[derive(Clone)]
    struct StaticFileSync {
        changes: FileChangeSet,
    }

    impl StaticFileSync {
        fn new(changes: FileChangeSet) -> Self {
            Self { changes }
        }
    }

    impl FileSyncPort for StaticFileSync {
        fn initialize(
            &self,
            _ctx: &RequestContext,
            _options: FileSyncInitOptions,
        ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
            Box::pin(async move { Ok(()) })
        }

        fn check_for_changes(
            &self,
            _ctx: &RequestContext,
            _options: FileSyncOptions,
        ) -> semantic_code_ports::BoxFuture<'_, Result<FileChangeSet>> {
            let changes = self.changes.clone();
            Box::pin(async move { Ok(changes) })
        }

        fn delete_snapshot(
            &self,
            _ctx: &RequestContext,
            _codebase_root: PathBuf,
        ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
            Box::pin(async move { Ok(()) })
        }
    }

    #[derive(Clone)]
    struct NoopFileSystem;

    impl FileSystemPort for NoopFileSystem {
        fn read_dir(
            &self,
            _ctx: &RequestContext,
            _codebase_root: PathBuf,
            _dir: semantic_code_ports::SafeRelativePath,
        ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<semantic_code_ports::FileSystemDirEntry>>>
        {
            Box::pin(async move { Ok(Vec::new()) })
        }

        fn read_file_text(
            &self,
            _ctx: &RequestContext,
            _codebase_root: PathBuf,
            _file: semantic_code_ports::SafeRelativePath,
        ) -> semantic_code_ports::BoxFuture<'_, Result<Box<str>>> {
            Box::pin(async move { Ok("".into()) })
        }

        fn stat(
            &self,
            _ctx: &RequestContext,
            _codebase_root: PathBuf,
            _path: semantic_code_ports::SafeRelativePath,
        ) -> semantic_code_ports::BoxFuture<'_, Result<semantic_code_ports::FileSystemStat>>
        {
            Box::pin(async move {
                Ok(semantic_code_ports::FileSystemStat {
                    kind: semantic_code_ports::FileSystemEntryKind::File,
                    size_bytes: 0,
                    mtime_ms: 0,
                })
            })
        }
    }

    #[derive(Clone)]
    struct NoopPathPolicy;

    impl PathPolicyPort for NoopPathPolicy {
        fn to_safe_relative_path(
            &self,
            input: &str,
        ) -> Result<semantic_code_ports::SafeRelativePath> {
            semantic_code_ports::SafeRelativePath::new(input)
        }
    }

    #[derive(Clone)]
    struct NoopIgnore;

    impl IgnorePort for NoopIgnore {
        fn is_ignored(&self, _input: &semantic_code_ports::IgnoreMatchInput) -> bool {
            false
        }
    }

    #[derive(Clone)]
    struct NoopSplitter;

    impl SplitterPort for NoopSplitter {
        fn split(
            &self,
            _ctx: &RequestContext,
            _code: Box<str>,
            _language: semantic_code_ports::Language,
            _options: semantic_code_ports::SplitOptions,
        ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<semantic_code_ports::CodeChunk>>>
        {
            Box::pin(async move { Ok(Vec::new()) })
        }

        fn set_chunk_size(&self, _chunk_size: usize) {}

        fn set_chunk_overlap(&self, _chunk_overlap: usize) {}
    }
}
