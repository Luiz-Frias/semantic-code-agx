//! Shared types for `index_codebase` pipeline.

use semantic_code_domain::{Chunk, CollectionName, IndexMode, Language, LineSpan, MAX_CHUNK_CHARS};
use semantic_code_ports::{
    CodeChunk, EmbeddingPort, FileSystemPort, IgnorePort, LoggerPort, PathPolicyPort, SplitterPort,
    TelemetryPort, VectorDbPort, VectorDocumentForInsert,
};
use semantic_code_shared::{RequestContext, Result, WorkerPool, WorkerPoolOptions};
use std::future::Future;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

type BoxedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub type BoxFuture<'a, T> = BoxedFuture<'a, T>;

/// Progress update emitted by the index use-case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexProgress {
    /// Current phase description.
    pub phase: Box<str>,
    /// Current item count.
    pub current: u64,
    /// Total item count.
    pub total: u64,
    /// Completion percentage (0-100).
    pub percentage: u8,
}

impl IndexProgress {
    pub(crate) fn new(phase: impl AsRef<str>, current: u64, total: u64, percentage: u8) -> Self {
        Self {
            phase: phase.as_ref().to_owned().into_boxed_str(),
            current,
            total,
            percentage,
        }
    }
}

/// Scan stage stats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanStageStats {
    /// Files discovered for indexing.
    pub files: u64,
    /// Elapsed time in milliseconds.
    pub duration_ms: u64,
}

/// Split stage stats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitStageStats {
    /// Files processed by the splitter.
    pub files: u64,
    /// Chunks produced by the splitter.
    pub chunks: u64,
    /// Elapsed time in milliseconds.
    pub duration_ms: u64,
}

/// Embedding stage stats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedStageStats {
    /// Embedding batches executed.
    pub batches: u64,
    /// Chunks embedded.
    pub chunks: u64,
    /// Elapsed time in milliseconds.
    pub duration_ms: u64,
}

/// Insert stage stats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsertStageStats {
    /// Insert batches executed.
    pub batches: u64,
    /// Chunks inserted.
    pub chunks: u64,
    /// Elapsed time in milliseconds.
    pub duration_ms: u64,
}

/// Aggregated ingestion stage stats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexStageStats {
    /// Scan stage stats.
    pub scan: ScanStageStats,
    /// Split stage stats.
    pub split: SplitStageStats,
    /// Embedding stage stats.
    pub embed: EmbedStageStats,
    /// Insert stage stats.
    pub insert: InsertStageStats,
}

/// Completion status for indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexCodebaseStatus {
    /// Completed successfully.
    Completed,
    /// Stopped because the chunk limit was reached.
    LimitReached,
}

/// Output returned by the index use-case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexCodebaseOutput {
    /// Number of files indexed.
    pub indexed_files: usize,
    /// Number of chunks indexed.
    pub total_chunks: usize,
    /// Completion status.
    pub status: IndexCodebaseStatus,
    /// Stage-level ingestion stats.
    pub stage_stats: IndexStageStats,
}

/// Input configuration for indexing.
#[derive(Clone)]
pub struct IndexCodebaseInput {
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
    /// Optional explicit file list (relative paths) to index.
    pub file_list: Option<Vec<Box<str>>>,
    /// Force reindex (drop collection if it exists).
    pub force_reindex: bool,
    /// Optional progress callback.
    pub on_progress: Option<Arc<dyn Fn(IndexProgress) + Send + Sync>>,
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
}

/// Dependencies required by the index use-case.
#[derive(Clone)]
pub struct IndexCodebaseDeps {
    /// Embedding adapter.
    pub embedding: Arc<dyn EmbeddingPort>,
    /// Vector database adapter.
    pub vectordb: Arc<dyn VectorDbPort>,
    /// Splitter adapter.
    pub splitter: Arc<dyn SplitterPort>,
    /// Filesystem adapter.
    pub filesystem: Arc<dyn FileSystemPort>,
    /// Path policy adapter.
    pub path_policy: Arc<dyn PathPolicyPort>,
    /// Ignore matcher adapter.
    pub ignore: Arc<dyn IgnorePort>,
    /// Optional logger.
    pub logger: Option<Arc<dyn LoggerPort>>,
    /// Optional telemetry.
    pub telemetry: Option<Arc<dyn TelemetryPort>>,
}

#[derive(Debug, Clone, Copy)]
pub struct IndexingLimits {
    pub(crate) embedding_batch_size: NonZeroUsize,
    pub(crate) chunk_limit: NonZeroUsize,
    pub(crate) max_in_flight_files: NonZeroUsize,
    pub(crate) max_in_flight_embedding_batches: NonZeroUsize,
    pub(crate) max_in_flight_inserts: NonZeroUsize,
    pub(crate) max_pending_embedding_batches: usize,
    pub(crate) max_pending_insert_batches: usize,
    pub(crate) prefetch_limit: usize,
}

impl IndexingLimits {
    pub(crate) fn from_input(input: &IndexCodebaseInput) -> Self {
        let embedding_batch_size = input.embedding_batch_size;
        let chunk_limit = input.chunk_limit;
        let max_in_flight_files = input.max_in_flight_files.unwrap_or(NonZeroUsize::MIN);
        let max_in_flight_embedding_batches = input
            .max_in_flight_embedding_batches
            .unwrap_or(NonZeroUsize::MIN);
        let max_in_flight_inserts = input.max_in_flight_inserts.unwrap_or(NonZeroUsize::MIN);
        let max_pending_embedding_batches = max_pending_batches(
            max_in_flight_embedding_batches.get(),
            input.max_buffered_chunks.map(NonZeroUsize::get),
            embedding_batch_size.get(),
        );
        let max_pending_insert_batches = max_pending_batches(
            max_in_flight_inserts.get(),
            input.max_buffered_embeddings.map(NonZeroUsize::get),
            embedding_batch_size.get(),
        );
        let prefetch_limit = max_in_flight_files.get().saturating_mul(2).max(1);

        Self {
            embedding_batch_size,
            chunk_limit,
            max_in_flight_files,
            max_in_flight_embedding_batches,
            max_in_flight_inserts,
            max_pending_embedding_batches,
            max_pending_insert_batches,
            prefetch_limit,
        }
    }
}

pub struct IndexWorkerPools {
    pub(crate) embedding: WorkerPool,
    pub(crate) insert: WorkerPool,
    pub(crate) files: WorkerPool,
}

impl IndexWorkerPools {
    pub(crate) fn new(ctx: &RequestContext, limits: &IndexingLimits) -> Result<Self> {
        let embedding_pool = WorkerPool::new(
            ctx.clone(),
            WorkerPoolOptions {
                concurrency: limits.max_in_flight_embedding_batches.get(),
                queue_capacity: Some(
                    limits
                        .max_in_flight_embedding_batches
                        .get()
                        .saturating_mul(2)
                        .max(1),
                ),
            },
        )?;
        let insert_pool = WorkerPool::new(
            ctx.clone(),
            WorkerPoolOptions {
                concurrency: limits.max_in_flight_inserts.get(),
                queue_capacity: Some(limits.max_in_flight_inserts.get().saturating_mul(2).max(1)),
            },
        )?;
        let file_pool = WorkerPool::new(
            ctx.clone(),
            WorkerPoolOptions {
                concurrency: limits.max_in_flight_files.get(),
                queue_capacity: Some(limits.max_in_flight_files.get().saturating_mul(2).max(1)),
            },
        )?;

        Ok(Self {
            embedding: embedding_pool,
            insert: insert_pool,
            files: file_pool,
        })
    }

    pub(crate) async fn stop(&self) {
        () = self.files.stop().await;
        () = self.embedding.stop().await;
        () = self.insert.stop().await;
    }
}

pub struct ProgressTracker {
    on_progress: Option<Arc<dyn Fn(IndexProgress) + Send + Sync>>,
    last_percentage: u8,
}

impl ProgressTracker {
    pub(crate) fn new(on_progress: Option<Arc<dyn Fn(IndexProgress) + Send + Sync>>) -> Self {
        Self {
            on_progress,
            last_percentage: 0,
        }
    }

    pub(crate) fn emit(
        &mut self,
        phase: &str,
        current: u64,
        total: u64,
        percentage_override: Option<u8>,
    ) {
        emit_progress(
            self.on_progress.as_ref(),
            phase,
            current,
            total,
            percentage_override,
            &mut self.last_percentage,
        );
    }
}

#[derive(Debug)]
pub struct IndexStageStatsCollector {
    scan_files: AtomicU64,
    scan_duration_ms: AtomicU64,
    split_files: AtomicU64,
    split_chunks: AtomicU64,
    split_duration_ms: AtomicU64,
    embed_batches: AtomicU64,
    embed_chunks: AtomicU64,
    embed_duration_ms: AtomicU64,
    insert_batches: AtomicU64,
    insert_chunks: AtomicU64,
    insert_duration_ms: AtomicU64,
}

impl IndexStageStatsCollector {
    pub(crate) const fn new() -> Self {
        Self {
            scan_files: AtomicU64::new(0),
            scan_duration_ms: AtomicU64::new(0),
            split_files: AtomicU64::new(0),
            split_chunks: AtomicU64::new(0),
            split_duration_ms: AtomicU64::new(0),
            embed_batches: AtomicU64::new(0),
            embed_chunks: AtomicU64::new(0),
            embed_duration_ms: AtomicU64::new(0),
            insert_batches: AtomicU64::new(0),
            insert_chunks: AtomicU64::new(0),
            insert_duration_ms: AtomicU64::new(0),
        }
    }

    pub(crate) fn record_scan(&self, files: u64, duration: Duration) {
        self.scan_files.store(files, Ordering::Release);
        self.scan_duration_ms
            .store(duration_ms(duration), Ordering::Release);
    }

    pub(crate) fn record_split(&self, files: u64, chunks: u64, duration: Duration) {
        self.split_files.fetch_add(files, Ordering::AcqRel);
        self.split_chunks.fetch_add(chunks, Ordering::AcqRel);
        self.split_duration_ms
            .fetch_add(duration_ms(duration), Ordering::AcqRel);
    }

    pub(crate) fn record_embed(&self, chunks: u64, duration: Duration) {
        self.embed_batches.fetch_add(1, Ordering::AcqRel);
        self.embed_chunks.fetch_add(chunks, Ordering::AcqRel);
        self.embed_duration_ms
            .fetch_add(duration_ms(duration), Ordering::AcqRel);
    }

    pub(crate) fn record_insert(&self, chunks: u64, duration: Duration) {
        self.insert_batches.fetch_add(1, Ordering::AcqRel);
        self.insert_chunks.fetch_add(chunks, Ordering::AcqRel);
        self.insert_duration_ms
            .fetch_add(duration_ms(duration), Ordering::AcqRel);
    }

    pub(crate) fn snapshot(&self) -> IndexStageStats {
        IndexStageStats {
            scan: ScanStageStats {
                files: self.scan_files.load(Ordering::Acquire),
                duration_ms: self.scan_duration_ms.load(Ordering::Acquire),
            },
            split: SplitStageStats {
                files: self.split_files.load(Ordering::Acquire),
                chunks: self.split_chunks.load(Ordering::Acquire),
                duration_ms: self.split_duration_ms.load(Ordering::Acquire),
            },
            embed: EmbedStageStats {
                batches: self.embed_batches.load(Ordering::Acquire),
                chunks: self.embed_chunks.load(Ordering::Acquire),
                duration_ms: self.embed_duration_ms.load(Ordering::Acquire),
            },
            insert: InsertStageStats {
                batches: self.insert_batches.load(Ordering::Acquire),
                chunks: self.insert_chunks.load(Ordering::Acquire),
                duration_ms: self.insert_duration_ms.load(Ordering::Acquire),
            },
        }
    }
}

#[derive(Clone)]
pub struct FileTaskContext<'a> {
    pub(crate) file_pool: &'a WorkerPool,
    pub(crate) request_ctx: &'a RequestContext,
    pub(crate) deps: &'a IndexCodebaseDeps,
    pub(crate) files: &'a [Box<str>],
    pub(crate) codebase_root: PathBuf,
    pub(crate) max_file_size_bytes: Option<u64>,
    pub(crate) stats: Arc<IndexStageStatsCollector>,
}

impl<'a> FileTaskContext<'a> {
    pub(crate) const fn new(
        file_pool: &'a WorkerPool,
        request_ctx: &'a RequestContext,
        deps: &'a IndexCodebaseDeps,
        files: &'a [Box<str>],
        codebase_root: PathBuf,
        max_file_size_bytes: Option<u64>,
        stats: Arc<IndexStageStatsCollector>,
    ) -> Self {
        Self {
            file_pool,
            request_ctx,
            deps,
            files,
            codebase_root,
            max_file_size_bytes,
            stats,
        }
    }
}

pub struct BatchContext<'a> {
    pub(crate) ctx: &'a RequestContext,
    pub(crate) deps: &'a IndexCodebaseDeps,
    pub(crate) input: &'a IndexCodebaseInput,
    pub(crate) embedding_pool: &'a WorkerPool,
    pub(crate) insert_pool: &'a WorkerPool,
    pub(crate) embedding_batch_size: NonZeroUsize,
    pub(crate) max_pending_embedding_batches: usize,
    pub(crate) max_pending_insert_batches: usize,
    pub(crate) stats: Arc<IndexStageStatsCollector>,
}

impl<'a> BatchContext<'a> {
    pub(crate) const fn new(
        ctx: &'a RequestContext,
        deps: &'a IndexCodebaseDeps,
        input: &'a IndexCodebaseInput,
        embedding_pool: &'a WorkerPool,
        insert_pool: &'a WorkerPool,
        limits: &IndexingLimits,
        stats: Arc<IndexStageStatsCollector>,
    ) -> Self {
        Self {
            ctx,
            deps,
            input,
            embedding_pool,
            insert_pool,
            embedding_batch_size: limits.embedding_batch_size,
            max_pending_embedding_batches: limits.max_pending_embedding_batches,
            max_pending_insert_batches: limits.max_pending_insert_batches,
            stats,
        }
    }
}

pub struct BatchState<'a> {
    pub(crate) pending: Vec<PendingChunk>,
    pub(crate) embedding_tasks: Vec<BoxFuture<'a, Result<EmbeddedBatch>>>,
    pub(crate) insert_tasks: Vec<InsertTask<'a>>,
    pub(crate) next_batch_to_insert: usize,
    pub(crate) next_insert_to_await: usize,
}

impl BatchState<'_> {
    pub(crate) fn new() -> Self {
        Self {
            pending: Vec::new(),
            embedding_tasks: Vec::new(),
            insert_tasks: Vec::new(),
            next_batch_to_insert: 0,
            next_insert_to_await: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PendingChunk {
    pub(crate) relative_path: Box<str>,
    pub(crate) span: LineSpan,
    pub(crate) language: Language,
    pub(crate) content: Chunk<MAX_CHUNK_CHARS>,
    pub(crate) file_extension: Option<Box<str>>,
}

#[derive(Debug)]
pub struct EmbeddedBatch {
    pub(crate) documents: Vec<VectorDocumentForInsert>,
}

pub struct InsertTask<'a> {
    pub(crate) promise: Option<BoxFuture<'a, Result<()>>>,
}

#[derive(Debug)]
pub enum FileResult {
    Skipped,
    Ok {
        relative_path: Box<str>,
        language: Language,
        chunks: Vec<CodeChunk>,
    },
}

pub fn duration_ms(duration: Duration) -> u64 {
    let millis = duration.as_millis();
    u64::try_from(millis).unwrap_or(u64::MAX)
}

pub fn max_pending_batches(
    concurrency: usize,
    max_buffered: Option<usize>,
    batch_size: usize,
) -> usize {
    let by_concurrency = concurrency.saturating_mul(2).max(1);
    let Some(max_buffered) = max_buffered else {
        return by_concurrency;
    };

    let by_memory = (max_buffered.max(1) / batch_size.max(1)).max(1);
    by_concurrency.min(by_memory).max(1)
}

fn emit_progress(
    on_progress: Option<&Arc<dyn Fn(IndexProgress) + Send + Sync>>,
    phase: &str,
    current: u64,
    total: u64,
    percentage_override: Option<u8>,
    last_percentage: &mut u8,
) {
    let Some(callback) = on_progress else {
        return;
    };

    let percentage = percentage_override.unwrap_or_else(|| progress_percentage(current, total));
    let percentage = percentage.max(*last_percentage);
    *last_percentage = percentage;

    callback(IndexProgress::new(phase, current, total, percentage));
}

fn progress_percentage(current: u64, total: u64) -> u8 {
    if total == 0 {
        return 0;
    }
    let capped = current.min(total);
    let percent = (capped.saturating_mul(100)) / total;
    u8::try_from(percent).unwrap_or(u8::MAX)
}

pub struct IndexRunContext<'a> {
    pub(crate) ctx: &'a RequestContext,
    pub(crate) files: &'a [Box<str>],
    pub(crate) limits: &'a IndexingLimits,
    pub(crate) file_tasks: FileTaskContext<'a>,
    pub(crate) batch: BatchContext<'a>,
    pub(crate) stats: Arc<IndexStageStatsCollector>,
}

impl<'a> IndexRunContext<'a> {
    pub(crate) fn new(
        ctx: &'a RequestContext,
        deps: &'a IndexCodebaseDeps,
        input: &'a IndexCodebaseInput,
        files: &'a [Box<str>],
        limits: &'a IndexingLimits,
        pools: &'a IndexWorkerPools,
        stats: Arc<IndexStageStatsCollector>,
    ) -> Self {
        let file_tasks = FileTaskContext::new(
            &pools.files,
            ctx,
            deps,
            files,
            input.codebase_root.clone(),
            input.max_file_size_bytes,
            Arc::clone(&stats),
        );
        let batch = BatchContext::new(
            ctx,
            deps,
            input,
            &pools.embedding,
            &pools.insert,
            limits,
            Arc::clone(&stats),
        );

        Self {
            ctx,
            files,
            limits,
            file_tasks,
            batch,
            stats,
        }
    }
}

pub struct IndexState<'a> {
    pub(crate) indexed_files: usize,
    pub(crate) total_chunks: usize,
    pub(crate) status: IndexCodebaseStatus,
    pub(crate) inflight: std::collections::HashMap<usize, BoxFuture<'a, Result<FileResult>>>,
    pub(crate) next_to_submit: usize,
    pub(crate) batch: BatchState<'a>,
}

impl IndexState<'_> {
    pub(crate) fn new() -> Self {
        Self {
            indexed_files: 0,
            total_chunks: 0,
            status: IndexCodebaseStatus::Completed,
            inflight: std::collections::HashMap::new(),
            next_to_submit: 0,
            batch: BatchState::new(),
        }
    }
}

pub struct IndexPipeline<S> {
    pub(crate) fsm: IndexPipelineFsm,
    pub(crate) _state: PhantomData<S>,
}

#[derive(Debug)]
pub struct IndexPipelineFsm {
    pub(crate) state: crate::generated::IndexPipelineState,
}

impl IndexPipelineFsm {
    pub(crate) const fn new() -> Self {
        Self {
            state: crate::generated::IndexPipelineState::Prepared,
        }
    }
}
