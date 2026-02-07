//! Index a codebase by scanning, splitting, embedding, and inserting chunks.

mod change_detector;
mod embedder;
mod inserter;
mod scanner;
mod splitter;
mod types;

#[cfg(test)]
pub(crate) use change_detector::{delete_file_chunks_by_relative_path, normalize_change_set};
pub(crate) use change_detector::{
    delete_modified_files, delete_removed_files, detect_changes, emit_progress, total_changes,
};
pub use types::{
    EmbedStageStats, IndexCodebaseDeps, IndexCodebaseInput, IndexCodebaseOutput,
    IndexCodebaseStatus, IndexProgress, IndexStageStats, InsertStageStats, ScanStageStats,
    SplitStageStats,
};

use crate::generated::{INDEX_PIPELINE_TRANSITIONS, IndexPipelineState};
use embedder::{drain_one_embedding_batch, flush_pending_batches, schedule_embedding_batch};
use inserter::drain_one_insert_batch;
use scanner::file_extension_of;
use semantic_code_domain::{Chunk, IndexMode, MAX_CHUNK_CHARS};
use semantic_code_ports::DetectDimensionOptions;
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;
use types::{
    FileResult, IndexPipeline, IndexPipelineFsm, IndexRunContext, IndexStageStatsCollector,
    IndexState, IndexWorkerPools, IndexingLimits, PendingChunk, ProgressTracker,
};

struct Prepared;
struct Scanned;
struct Embedded;
struct Inserted;
struct Completed;

impl IndexPipeline<Prepared> {
    const fn new() -> Self {
        Self {
            fsm: IndexPipelineFsm::new(),
            _state: PhantomData,
        }
    }

    fn scanned(self) -> Result<IndexPipeline<Scanned>> {
        self.transition(IndexPipelineState::Scanned)
    }
}

impl IndexPipeline<Scanned> {
    fn embedded(self) -> Result<IndexPipeline<Embedded>> {
        self.transition(IndexPipelineState::Embedded)
    }
}

impl IndexPipeline<Embedded> {
    fn inserted(self) -> Result<IndexPipeline<Inserted>> {
        self.transition(IndexPipelineState::Inserted)
    }
}

impl IndexPipeline<Inserted> {
    fn completed(self) -> Result<IndexPipeline<Completed>> {
        self.transition(IndexPipelineState::Completed)
    }
}

impl<S> IndexPipeline<S> {
    fn transition<T>(self, next: IndexPipelineState) -> Result<IndexPipeline<T>> {
        let mut fsm = self.fsm;
        fsm.transition(next)?;
        Ok(IndexPipeline {
            fsm,
            _state: PhantomData,
        })
    }
}

impl IndexPipelineFsm {
    fn transition(&mut self, next: IndexPipelineState) -> Result<()> {
        if is_allowed_transition(self.state, next) {
            self.state = next;
            return Ok(());
        }
        Err(ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            format!(
                "invalid index pipeline transition: {} -> {}",
                self.state.as_str(),
                next.as_str()
            ),
            ErrorClass::NonRetriable,
        ))
    }
}

fn is_allowed_transition(from: IndexPipelineState, to: IndexPipelineState) -> bool {
    INDEX_PIPELINE_TRANSITIONS
        .iter()
        .any(|(source, target)| *source == from && *target == to)
}

/// Index a codebase using the provided dependencies and input.
pub async fn index_codebase(
    ctx: &RequestContext,
    deps: &IndexCodebaseDeps,
    input: IndexCodebaseInput,
) -> Result<IndexCodebaseOutput> {
    ctx.ensure_not_cancelled("index_codebase")?;

    let pipeline = IndexPipeline::<Prepared>::new();
    let mut progress = ProgressTracker::new(input.on_progress.clone());
    let stats = Arc::new(IndexStageStatsCollector::new());

    progress.emit("Preparing collection...", 0, 100, Some(0));
    ensure_collection(ctx, deps, &input).await?;

    progress.emit("Scanning files...", 0, 100, Some(5));
    let scan_started = Instant::now();
    let files = scanner::load_index_files(ctx, deps, &input).await?;
    stats.record_scan(
        u64::try_from(files.len()).unwrap_or(u64::MAX),
        scan_started.elapsed(),
    );
    let pipeline = pipeline.scanned()?;

    if files.is_empty() {
        progress.emit("No files to index", 100, 100, Some(100));
        let _pipeline = pipeline.embedded()?.inserted()?.completed()?;
        return Ok(IndexCodebaseOutput {
            indexed_files: 0,
            total_chunks: 0,
            status: IndexCodebaseStatus::Completed,
            stage_stats: stats.snapshot(),
        });
    }

    let limits = IndexingLimits::from_input(&input);
    let pools = IndexWorkerPools::new(ctx, &limits)?;
    let run_ctx = IndexRunContext::new(ctx, deps, &input, &files, &limits, &pools, stats);

    let output = run_indexing(&run_ctx, &mut progress, pipeline).await;

    pools.stop().await;

    let (pipeline, output) = output?;
    let _pipeline = pipeline.completed()?;
    Ok(output)
}

async fn run_indexing(
    ctx: &IndexRunContext<'_>,
    progress: &mut ProgressTracker,
    pipeline: IndexPipeline<Scanned>,
) -> Result<(IndexPipeline<Inserted>, IndexCodebaseOutput)> {
    let mut state = IndexState::new();

    process_files(ctx, &mut state, progress).await?;
    let pipeline = pipeline.embedded()?;
    finalize_batches(ctx, &mut state).await?;
    let pipeline = pipeline.inserted()?;

    Ok((
        pipeline,
        IndexCodebaseOutput {
            indexed_files: state.indexed_files,
            total_chunks: state.total_chunks,
            status: state.status,
            stage_stats: ctx.stats.snapshot(),
        },
    ))
}

async fn process_files<'a>(
    ctx: &IndexRunContext<'a>,
    state: &mut IndexState<'a>,
    progress: &mut ProgressTracker,
) -> Result<()> {
    for file_index in 0..ctx.files.len() {
        ctx.ctx
            .ensure_not_cancelled("index_codebase.process_file")?;

        while state.next_to_submit < ctx.files.len()
            && state.inflight.len() < ctx.limits.prefetch_limit
        {
            splitter::submit_file_task(&ctx.file_tasks, &mut state.inflight, state.next_to_submit)?;
            state.next_to_submit += 1;
        }

        splitter::submit_file_task(&ctx.file_tasks, &mut state.inflight, file_index)?;

        let task = state.inflight.remove(&file_index).ok_or_else(|| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "missing file task",
                ErrorClass::NonRetriable,
            )
        })?;
        let result = task.await?;

        if state.total_chunks >= ctx.limits.chunk_limit.get() {
            state.status = IndexCodebaseStatus::LimitReached;
            break;
        }

        let FileResult::Ok {
            relative_path,
            language,
            chunks,
        } = result
        else {
            continue;
        };

        for chunk in chunks {
            ctx.ctx.ensure_not_cancelled("index_codebase.chunk_loop")?;

            if state.total_chunks >= ctx.limits.chunk_limit.get() {
                state.status = IndexCodebaseStatus::LimitReached;
                break;
            }

            let content =
                Chunk::<MAX_CHUNK_CHARS>::new(chunk.content).map_err(ErrorEnvelope::from)?;
            state.batch.pending.push(PendingChunk {
                relative_path: relative_path.clone(),
                span: chunk.span,
                language: chunk.language.unwrap_or(language),
                content,
                file_extension: file_extension_of(relative_path.as_ref()),
            });
            state.total_chunks += 1;

            if state.batch.pending.len() >= ctx.limits.embedding_batch_size.get() {
                flush_pending_batches(&ctx.batch, &mut state.batch).await?;
            }
        }

        if state.status == IndexCodebaseStatus::LimitReached {
            break;
        }

        state.indexed_files += 1;
        progress.emit(
            &format!(
                "Processing files ({}/{})...",
                file_index + 1,
                ctx.files.len()
            ),
            (file_index + 1) as u64,
            ctx.files.len() as u64,
            None,
        );
    }

    Ok(())
}

async fn finalize_batches<'a>(ctx: &IndexRunContext<'a>, state: &mut IndexState<'a>) -> Result<()> {
    flush_pending_batches(&ctx.batch, &mut state.batch).await?;

    if !state.batch.pending.is_empty() {
        let tail = std::mem::take(&mut state.batch.pending);
        schedule_embedding_batch(&ctx.batch, &mut state.batch, tail);
    }

    while state.batch.next_batch_to_insert < state.batch.embedding_tasks.len() {
        drain_one_embedding_batch(&ctx.batch, &mut state.batch).await?;
    }

    while state.batch.next_insert_to_await < state.batch.insert_tasks.len() {
        drain_one_insert_batch(&ctx.batch, &mut state.batch).await?;
    }

    Ok(())
}

async fn ensure_collection(
    ctx: &RequestContext,
    deps: &IndexCodebaseDeps,
    input: &IndexCodebaseInput,
) -> Result<()> {
    let exists = deps
        .vectordb
        .has_collection(ctx, input.collection_name.clone())
        .await?;

    if exists && input.force_reindex {
        deps.vectordb
            .drop_collection(ctx, input.collection_name.clone())
            .await?;
    }

    if exists && !input.force_reindex {
        return Ok(());
    }

    let dimension = deps
        .embedding
        .detect_dimension(ctx, DetectDimensionOptions::default().into())
        .await?;

    match input.index_mode {
        IndexMode::Hybrid => {
            deps.vectordb
                .create_hybrid_collection(ctx, input.collection_name.clone(), dimension, None)
                .await?;
        },
        IndexMode::Dense => {
            deps.vectordb
                .create_collection(ctx, input.collection_name.clone(), dimension, None)
                .await?;
        },
    }

    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_domain::{EmbeddingProviderId, VectorDbProviderId};
    use semantic_code_ports::{
        BoxFuture, CodeChunk, CollectionName, DetectDimensionRequest, EmbedBatchRequest,
        EmbedRequest, EmbeddingPort, EmbeddingProviderInfo, EmbeddingVector, FileSystemDirEntry,
        FileSystemEntryKind, FileSystemPort, FileSystemStat, HybridSearchBatchRequest,
        HybridSearchResult, IgnoreMatchInput, IgnorePort, Language, LineSpan, PathPolicyPort,
        SplitOptions, SplitterPort, VectorDbPort, VectorDbProviderInfo, VectorDocumentForInsert,
        VectorSearchRequest, VectorSearchResult,
    };
    use std::collections::HashMap;
    use std::num::NonZeroUsize;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    fn embedding_provider(id: &str) -> EmbeddingProviderId {
        EmbeddingProviderId::parse(id).expect("embedding provider id")
    }

    fn vectordb_provider(id: &str) -> VectorDbProviderId {
        VectorDbProviderId::parse(id).expect("vectordb provider id")
    }

    #[test]
    fn max_pending_batches_respects_buffer_cap() {
        let pending = types::max_pending_batches(4, Some(4), 4);
        assert_eq!(pending, 1);

        let pending = types::max_pending_batches(4, Some(32), 4);
        assert_eq!(pending, 8.min(4 * 2));

        let pending = types::max_pending_batches(2, None, 4);
        assert_eq!(pending, 4);
    }

    #[derive(Default)]
    struct TestPathPolicy;

    impl PathPolicyPort for TestPathPolicy {
        fn to_safe_relative_path(
            &self,
            input: &str,
        ) -> Result<semantic_code_ports::SafeRelativePath> {
            semantic_code_ports::SafeRelativePath::new(input)
        }
    }

    #[derive(Default)]
    struct TestIgnore;

    impl IgnorePort for TestIgnore {
        fn is_ignored(&self, input: &IgnoreMatchInput) -> bool {
            input.ignore_patterns.iter().any(|pattern| {
                !pattern.is_empty() && input.relative_path.contains(pattern.as_ref())
            })
        }
    }

    #[derive(Clone)]
    struct TestFileSystem {
        state: Arc<Mutex<TestFileSystemState>>,
    }

    impl Default for TestFileSystem {
        fn default() -> Self {
            Self {
                state: Arc::new(Mutex::new(TestFileSystemState::default())),
            }
        }
    }

    #[derive(Default)]
    struct TestFileSystemState {
        files: HashMap<String, String>,
        dirs: HashMap<String, Vec<FileSystemDirEntry>>,
    }

    impl TestFileSystem {
        fn add_file(&self, path: &str, content: &str) {
            let normalized = path.replace('\\', "/");
            let mut state = self.state.lock().expect("test file system state lock");
            state.files.insert(normalized.clone(), content.to_string());

            let (dir, name) = normalized
                .rsplit_once('/')
                .map(|(dir, name)| (dir, name))
                .unwrap_or((".", normalized.as_str()));

            state.add_dir_entry(dir, name, FileSystemEntryKind::File);
            state.ensure_dirs(dir);
        }
    }

    impl TestFileSystemState {
        fn ensure_dirs(&mut self, dir: &str) {
            if dir == "." || dir.is_empty() {
                return;
            }
            let mut current = String::new();
            for segment in dir.split('/') {
                let parent = if current.is_empty() {
                    "."
                } else {
                    current.as_str()
                };
                let next = if current.is_empty() {
                    segment.to_string()
                } else {
                    format!("{current}/{segment}")
                };
                self.add_dir_entry(parent, segment, FileSystemEntryKind::Directory);
                current = next;
            }
        }

        fn add_dir_entry(&mut self, dir: &str, name: &str, kind: FileSystemEntryKind) {
            let entries = self.dirs.entry(dir.to_string()).or_default();
            if entries.iter().any(|entry| entry.name.as_ref() == name) {
                return;
            }
            entries.push(FileSystemDirEntry {
                name: name.to_string().into_boxed_str(),
                kind,
            });
        }
    }

    impl FileSystemPort for TestFileSystem {
        fn read_dir(
            &self,
            _ctx: &RequestContext,
            _codebase_root: PathBuf,
            dir: semantic_code_ports::SafeRelativePath,
        ) -> BoxFuture<'_, Result<Vec<FileSystemDirEntry>>> {
            Box::pin(async move {
                let state = self.state.lock().expect("test file system state lock");
                Ok(state.dirs.get(dir.as_str()).cloned().unwrap_or_default())
            })
        }

        fn read_file_text(
            &self,
            _ctx: &RequestContext,
            _codebase_root: PathBuf,
            file: semantic_code_ports::SafeRelativePath,
        ) -> BoxFuture<'_, Result<Box<str>>> {
            Box::pin(async move {
                let state = self.state.lock().expect("test file system state lock");
                state
                    .files
                    .get(file.as_str())
                    .map(|value| value.to_string().into_boxed_str())
                    .ok_or_else(|| ErrorEnvelope::expected(ErrorCode::not_found(), "missing file"))
            })
        }

        fn stat(
            &self,
            _ctx: &RequestContext,
            _codebase_root: PathBuf,
            path: semantic_code_ports::SafeRelativePath,
        ) -> BoxFuture<'_, Result<FileSystemStat>> {
            Box::pin(async move {
                let state = self.state.lock().expect("test file system state lock");
                if path.as_str() == "." || state.dirs.contains_key(path.as_str()) {
                    return Ok(FileSystemStat {
                        kind: FileSystemEntryKind::Directory,
                        size_bytes: 0,
                        mtime_ms: 0,
                    });
                }
                if let Some(contents) = state.files.get(path.as_str()) {
                    return Ok(FileSystemStat {
                        kind: FileSystemEntryKind::File,
                        size_bytes: contents.len() as u64,
                        mtime_ms: 0,
                    });
                }
                Ok(FileSystemStat {
                    kind: FileSystemEntryKind::Other,
                    size_bytes: 0,
                    mtime_ms: 0,
                })
            })
        }
    }

    #[derive(Clone)]
    struct TestSplitter {
        chunks_per_file: usize,
    }

    impl TestSplitter {
        fn new(chunks_per_file: usize) -> Self {
            Self { chunks_per_file }
        }
    }

    impl SplitterPort for TestSplitter {
        fn split(
            &self,
            _ctx: &RequestContext,
            code: Box<str>,
            language: Language,
            options: SplitOptions,
        ) -> BoxFuture<'_, Result<Vec<CodeChunk>>> {
            Box::pin(async move {
                let lines = code.lines().count().max(1) as u32;
                let span = LineSpan::new(1, lines).map_err(ErrorEnvelope::from)?;
                let mut out = Vec::new();
                for index in 0..self.chunks_per_file {
                    out.push(CodeChunk {
                        content: format!("{language}:{index}:{code}").into_boxed_str(),
                        span,
                        language: Some(language),
                        file_path: options.file_path.clone(),
                    });
                }
                Ok(out)
            })
        }

        fn set_chunk_size(&self, _chunk_size: usize) {}

        fn set_chunk_overlap(&self, _chunk_overlap: usize) {}
    }

    #[derive(Clone)]
    struct TestEmbedding {
        provider: EmbeddingProviderInfo,
        vector: Arc<[f32]>,
    }

    impl TestEmbedding {
        fn new() -> Self {
            Self {
                provider: EmbeddingProviderInfo {
                    id: embedding_provider("openai"),
                    name: "test".into(),
                },
                vector: Arc::from(vec![0.0, 0.1, 0.2]),
            }
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
        ) -> BoxFuture<'_, Result<u32>> {
            let dimension = self.vector.len() as u32;
            Box::pin(async move { Ok(dimension) })
        }

        fn embed(
            &self,
            _ctx: &RequestContext,
            _request: EmbedRequest,
        ) -> BoxFuture<'_, Result<EmbeddingVector>> {
            let vector = Arc::clone(&self.vector);
            Box::pin(async move { Ok(EmbeddingVector::new(vector)) })
        }

        fn embed_batch(
            &self,
            _ctx: &RequestContext,
            request: EmbedBatchRequest,
        ) -> BoxFuture<'_, Result<Vec<EmbeddingVector>>> {
            let vector = Arc::clone(&self.vector);
            Box::pin(async move {
                let texts = request.texts;
                Ok(texts
                    .into_iter()
                    .map(|_| EmbeddingVector::new(Arc::clone(&vector)))
                    .collect())
            })
        }
    }

    #[derive(Clone)]
    struct SlowEmbedding {
        provider: EmbeddingProviderInfo,
        delay: Duration,
    }

    impl SlowEmbedding {
        fn new(delay: Duration) -> Self {
            Self {
                provider: EmbeddingProviderInfo {
                    id: embedding_provider("openai"),
                    name: "slow".into(),
                },
                delay,
            }
        }
    }

    impl EmbeddingPort for SlowEmbedding {
        fn provider(&self) -> &EmbeddingProviderInfo {
            &self.provider
        }

        fn detect_dimension(
            &self,
            _ctx: &RequestContext,
            _request: DetectDimensionRequest,
        ) -> BoxFuture<'_, Result<u32>> {
            Box::pin(async move { Ok(3) })
        }

        fn embed(
            &self,
            ctx: &RequestContext,
            _request: EmbedRequest,
        ) -> BoxFuture<'_, Result<EmbeddingVector>> {
            let ctx = ctx.clone();
            let delay = self.delay;
            Box::pin(async move {
                tokio::time::sleep(delay).await;
                ctx.ensure_not_cancelled("test.embed")?;
                Ok(EmbeddingVector::from_vec(vec![0.0, 0.0, 0.0]))
            })
        }

        fn embed_batch(
            &self,
            ctx: &RequestContext,
            request: EmbedBatchRequest,
        ) -> BoxFuture<'_, Result<Vec<EmbeddingVector>>> {
            let ctx = ctx.clone();
            let delay = self.delay;
            Box::pin(async move {
                tokio::time::sleep(delay).await;
                ctx.ensure_not_cancelled("test.embed_batch")?;
                let texts = request.texts;
                Ok(texts
                    .into_iter()
                    .map(|_| EmbeddingVector::from_vec(vec![0.0, 0.0, 0.0]))
                    .collect())
            })
        }
    }

    #[derive(Clone)]
    struct SpyVectorDb {
        provider: VectorDbProviderInfo,
        inserted: Arc<Mutex<Vec<VectorDocumentForInsert>>>,
        exists: Arc<Mutex<bool>>,
    }

    impl SpyVectorDb {
        fn new() -> Self {
            Self {
                provider: VectorDbProviderInfo {
                    id: vectordb_provider("milvus_grpc"),
                    name: "spy".into(),
                },
                inserted: Arc::new(Mutex::new(Vec::new())),
                exists: Arc::new(Mutex::new(false)),
            }
        }

        fn inserted_paths(&self) -> Vec<String> {
            let guard = self.inserted.lock().expect("inserted lock");
            guard
                .iter()
                .map(|doc| doc.metadata.relative_path.as_ref().to_string())
                .collect()
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
        ) -> BoxFuture<'_, Result<()>> {
            let exists = self.exists.clone();
            Box::pin(async move {
                let mut guard = exists.lock().expect("exists lock");
                *guard = true;
                Ok(())
            })
        }

        fn create_hybrid_collection(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _dimension: u32,
            _description: Option<Box<str>>,
        ) -> BoxFuture<'_, Result<()>> {
            let exists = self.exists.clone();
            Box::pin(async move {
                let mut guard = exists.lock().expect("exists lock");
                *guard = true;
                Ok(())
            })
        }

        fn drop_collection(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
        ) -> BoxFuture<'_, Result<()>> {
            let exists = self.exists.clone();
            Box::pin(async move {
                let mut guard = exists.lock().expect("exists lock");
                *guard = false;
                Ok(())
            })
        }

        fn has_collection(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
        ) -> BoxFuture<'_, Result<bool>> {
            let exists = self.exists.clone();
            Box::pin(async move { Ok(*exists.lock().expect("exists lock")) })
        }

        fn list_collections(
            &self,
            _ctx: &RequestContext,
        ) -> BoxFuture<'_, Result<Vec<CollectionName>>> {
            Box::pin(async move { Ok(Vec::new()) })
        }

        fn insert(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            documents: Vec<VectorDocumentForInsert>,
        ) -> BoxFuture<'_, Result<()>> {
            let inserted = self.inserted.clone();
            Box::pin(async move {
                let mut guard = inserted.lock().expect("inserted lock");
                guard.extend(documents);
                Ok(())
            })
        }

        fn insert_hybrid(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            documents: Vec<VectorDocumentForInsert>,
        ) -> BoxFuture<'_, Result<()>> {
            let inserted = self.inserted.clone();
            Box::pin(async move {
                let mut guard = inserted.lock().expect("inserted lock");
                guard.extend(documents);
                Ok(())
            })
        }

        fn search(
            &self,
            _ctx: &RequestContext,
            _request: VectorSearchRequest,
        ) -> BoxFuture<'_, Result<Vec<VectorSearchResult>>> {
            Box::pin(async move { Ok(Vec::new()) })
        }

        fn hybrid_search(
            &self,
            _ctx: &RequestContext,
            _request: HybridSearchBatchRequest,
        ) -> BoxFuture<'_, Result<Vec<HybridSearchResult>>> {
            Box::pin(async move { Ok(Vec::new()) })
        }

        fn delete(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _ids: Vec<Box<str>>,
        ) -> BoxFuture<'_, Result<()>> {
            Box::pin(async move { Ok(()) })
        }

        fn query(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _filter: Box<str>,
            _output_fields: Vec<Box<str>>,
            _limit: Option<u32>,
        ) -> BoxFuture<'_, Result<Vec<semantic_code_ports::VectorDbRow>>> {
            Box::pin(async move { Ok(Vec::new()) })
        }
    }

    fn test_deps(
        fs: TestFileSystem,
        embedding: Arc<dyn EmbeddingPort>,
        vectordb: Arc<SpyVectorDb>,
        splitter: Arc<dyn SplitterPort>,
    ) -> IndexCodebaseDeps {
        IndexCodebaseDeps {
            embedding,
            vectordb,
            splitter,
            filesystem: Arc::new(fs),
            path_policy: Arc::new(TestPathPolicy),
            ignore: Arc::new(TestIgnore),
            logger: None,
            telemetry: None,
        }
    }

    fn default_input(collection_name: CollectionName) -> IndexCodebaseInput {
        IndexCodebaseInput {
            codebase_root: PathBuf::from("/tmp"),
            collection_name,
            index_mode: IndexMode::Dense,
            supported_extensions: None,
            ignore_patterns: None,
            file_list: None,
            force_reindex: false,
            on_progress: None,
            embedding_batch_size: NonZeroUsize::new(4).unwrap_or(NonZeroUsize::MIN),
            chunk_limit: NonZeroUsize::new(100).unwrap_or(NonZeroUsize::MIN),
            max_files: None,
            max_file_size_bytes: None,
            max_buffered_chunks: None,
            max_buffered_embeddings: None,
            max_in_flight_files: Some(NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN)),
            max_in_flight_embedding_batches: Some(
                NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN),
            ),
            max_in_flight_inserts: Some(NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN)),
        }
    }

    #[tokio::test]
    async fn extension_filtering_is_deterministic() -> Result<()> {
        let fs = TestFileSystem::default();
        fs.add_file("src/b.ts", "export const b = 2;\n");
        fs.add_file("src/a.rs", "fn a() {}\n");
        fs.add_file("src/c.rs", "fn c() {}\n");

        let vectordb = Arc::new(SpyVectorDb::new());
        let deps = test_deps(
            fs,
            Arc::new(TestEmbedding::new()),
            vectordb.clone(),
            Arc::new(TestSplitter::new(1)),
        );

        let mut input =
            default_input(CollectionName::parse("code_chunks_test").map_err(ErrorEnvelope::from)?);
        input.supported_extensions = Some(vec![".rs".into()]);

        let ctx = RequestContext::new_request();
        let output = index_codebase(&ctx, &deps, input).await?;

        let mut paths = vectordb.inserted_paths();
        paths.sort();
        assert_eq!(paths, vec!["src/a.rs", "src/c.rs"]);
        assert_eq!(output.indexed_files, 2);
        assert_eq!(output.total_chunks, 2);
        Ok(())
    }

    #[tokio::test]
    async fn chunk_limit_stops_indexing() -> Result<()> {
        let fs = TestFileSystem::default();
        fs.add_file("src/a.rs", "fn a() {}\n");

        let vectordb = Arc::new(SpyVectorDb::new());
        let deps = test_deps(
            fs,
            Arc::new(TestEmbedding::new()),
            vectordb,
            Arc::new(TestSplitter::new(3)),
        );

        let mut input =
            default_input(CollectionName::parse("code_chunks_test").map_err(ErrorEnvelope::from)?);
        input.chunk_limit = NonZeroUsize::new(1).unwrap_or(NonZeroUsize::MIN);

        let ctx = RequestContext::new_request();
        let output = index_codebase(&ctx, &deps, input).await?;

        assert_eq!(output.status, IndexCodebaseStatus::LimitReached);
        assert_eq!(output.total_chunks, 1);
        Ok(())
    }

    #[tokio::test]
    async fn abort_cancels_run() -> Result<()> {
        let fs = TestFileSystem::default();
        for index in 0..20 {
            fs.add_file(&format!("src/{index}.rs"), "fn a() {}\n");
        }

        let vectordb = Arc::new(SpyVectorDb::new());
        let deps = test_deps(
            fs,
            Arc::new(SlowEmbedding::new(Duration::from_millis(50))),
            vectordb,
            Arc::new(TestSplitter::new(1)),
        );

        let mut input =
            default_input(CollectionName::parse("code_chunks_test").map_err(ErrorEnvelope::from)?);
        input.embedding_batch_size = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);

        let ctx = RequestContext::new_request();
        let ctx_clone = ctx.clone();
        let deps_clone = deps.clone();

        let handle =
            tokio::spawn(async move { index_codebase(&ctx_clone, &deps_clone, input).await });

        tokio::time::sleep(Duration::from_millis(20)).await;
        ctx.cancel();

        let result = handle.await.expect("join");
        assert!(matches!(result, Err(error) if error.is_cancelled()));
        Ok(())
    }

    #[tokio::test]
    async fn progress_is_monotonic() -> Result<()> {
        let fs = TestFileSystem::default();
        fs.add_file("src/a.rs", "fn a() {}\n");
        fs.add_file("src/b.rs", "fn b() {}\n");

        let vectordb = Arc::new(SpyVectorDb::new());
        let deps = test_deps(
            fs,
            Arc::new(TestEmbedding::new()),
            vectordb,
            Arc::new(TestSplitter::new(1)),
        );

        let percentages = Arc::new(Mutex::new(Vec::new()));
        let percentages_clone = percentages.clone();

        let mut input =
            default_input(CollectionName::parse("code_chunks_test").map_err(ErrorEnvelope::from)?);
        input.on_progress = Some(Arc::new(move |progress| {
            let mut guard = percentages_clone.lock().expect("progress lock");
            guard.push(progress.percentage);
        }));

        let ctx = RequestContext::new_request();
        let output = index_codebase(&ctx, &deps, input).await?;
        assert_eq!(output.status, IndexCodebaseStatus::Completed);

        let guard = percentages.lock().expect("progress lock");
        for window in guard.windows(2) {
            assert!(window[0] <= window[1], "progress should be monotonic");
        }
        assert_eq!(guard.last().copied(), Some(100));
        Ok(())
    }
}
