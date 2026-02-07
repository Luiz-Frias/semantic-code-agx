//! Integration tests for reindex-by-change use case.

use semantic_code_app::{ReindexByChangeDeps, ReindexByChangeInput, reindex_by_change};
use semantic_code_domain::{CollectionName, EmbeddingProviderId, IndexMode, VectorDbProviderId};
use semantic_code_ports::{
    EmbeddingPort, FileSyncPort, FileSystemDirEntry, FileSystemEntryKind, FileSystemPort,
    IgnorePort, PathPolicyPort, SplitOptions, SplitterPort, VectorDbPort, VectorDbProviderInfo,
    VectorSearchOptions, VectorSearchRequest,
};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use semantic_code_testkit::in_memory::{InMemoryEmbeddingFixed, InMemoryVectorDbFixed};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;

#[tokio::test]
async fn reindex_by_change_indexes_added_files() -> Result<()> {
    let ctx = RequestContext::new_request();
    let embedding = Arc::new(InMemoryEmbeddingFixed::<8>::new(
        semantic_code_ports::EmbeddingProviderInfo {
            id: EmbeddingProviderId::parse("openai").map_err(ErrorEnvelope::from)?,
            name: "in-memory".into(),
        },
    ));
    let vectordb = Arc::new(InMemoryVectorDbFixed::<8>::new(VectorDbProviderInfo {
        id: VectorDbProviderId::parse("milvus_grpc").map_err(ErrorEnvelope::from)?,
        name: "in-memory".into(),
    }));

    let file_sync = Arc::new(StaticFileSync {
        changes: semantic_code_ports::FileChangeSet {
            added: vec!["src/lib.rs".into()],
            removed: Vec::new(),
            modified: Vec::new(),
        },
    });
    let filesystem = Arc::new(FixtureFileSystem::new([(
        "src/lib.rs",
        "pub fn search_target() -> &'static str {\n    \"needle\"\n}\n",
    )]));

    let deps = ReindexByChangeDeps {
        file_sync,
        vectordb: vectordb.clone(),
        embedding: embedding.clone(),
        splitter: Arc::new(FixtureSplitter),
        filesystem,
        path_policy: Arc::new(AllowPathPolicy),
        ignore: Arc::new(NoopIgnore),
        logger: None,
        telemetry: None,
    };

    let input = ReindexByChangeInput {
        codebase_root: PathBuf::from("/tmp/repo"),
        collection_name: CollectionName::parse("code_chunks_reindex")
            .map_err(ErrorEnvelope::from)?,
        index_mode: IndexMode::Dense,
        supported_extensions: Some(vec![".rs".into()]),
        ignore_patterns: None,
        embedding_batch_size: NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN),
        chunk_limit: NonZeroUsize::new(50).unwrap_or(NonZeroUsize::MIN),
        max_files: None,
        max_file_size_bytes: None,
        max_buffered_chunks: None,
        max_buffered_embeddings: None,
        max_in_flight_files: Some(NonZeroUsize::MIN),
        max_in_flight_embedding_batches: Some(NonZeroUsize::MIN),
        max_in_flight_inserts: Some(NonZeroUsize::MIN),
        on_progress: None,
    };

    let output = reindex_by_change(&ctx, &deps, input).await?;
    assert_eq!(output.added, 1);
    assert_eq!(output.removed, 0);
    assert_eq!(output.modified, 0);

    let embedding_vector = embedding.embed(&ctx, "needle".into()).await?;
    let results = vectordb
        .search(
            &ctx,
            VectorSearchRequest {
                collection_name: CollectionName::parse("code_chunks_reindex")
                    .map_err(ErrorEnvelope::from)?,
                query_vector: embedding_vector.into_vector(),
                options: VectorSearchOptions {
                    top_k: Some(5),
                    threshold: None,
                    filter_expr: None,
                },
            },
        )
        .await?;
    assert!(!results.is_empty());
    Ok(())
}

#[derive(Clone)]
struct StaticFileSync {
    changes: semantic_code_ports::FileChangeSet,
}

impl FileSyncPort for StaticFileSync {
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
struct FixtureFileSystem {
    files: Arc<HashMap<Box<str>, Box<str>>>,
}

impl FixtureFileSystem {
    fn new<I, K, V>(entries: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<Box<str>>,
        V: Into<Box<str>>,
    {
        let files = entries
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect::<HashMap<_, _>>();
        Self {
            files: Arc::new(files),
        }
    }
}

impl FileSystemPort for FixtureFileSystem {
    fn read_dir(
        &self,
        _ctx: &RequestContext,
        _codebase_root: PathBuf,
        _dir: semantic_code_ports::SafeRelativePath,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<FileSystemDirEntry>>> {
        Box::pin(async move { Ok(Vec::new()) })
    }

    fn read_file_text(
        &self,
        _ctx: &RequestContext,
        _codebase_root: PathBuf,
        file: semantic_code_ports::SafeRelativePath,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Box<str>>> {
        let files = self.files.clone();
        Box::pin(async move {
            files
                .get(file.as_str())
                .cloned()
                .ok_or_else(|| ErrorEnvelope::expected(ErrorCode::not_found(), "file not found"))
        })
    }

    fn stat(
        &self,
        _ctx: &RequestContext,
        _codebase_root: PathBuf,
        path: semantic_code_ports::SafeRelativePath,
    ) -> semantic_code_ports::BoxFuture<'_, Result<semantic_code_ports::FileSystemStat>> {
        let files = self.files.clone();
        Box::pin(async move {
            let Some(contents) = files.get(path.as_str()) else {
                return Err(ErrorEnvelope::expected(
                    ErrorCode::not_found(),
                    "file not found",
                ));
            };
            Ok(semantic_code_ports::FileSystemStat {
                kind: FileSystemEntryKind::File,
                size_bytes: contents.len() as u64,
                mtime_ms: 0,
            })
        })
    }
}

#[derive(Clone)]
struct FixtureSplitter;

impl SplitterPort for FixtureSplitter {
    fn split(
        &self,
        _ctx: &RequestContext,
        code: Box<str>,
        language: semantic_code_ports::Language,
        options: SplitOptions,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<semantic_code_ports::CodeChunk>>> {
        Box::pin(async move {
            let lines = u32::try_from(code.lines().count().max(1)).map_err(|_| {
                ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    "line count overflow",
                    ErrorClass::NonRetriable,
                )
            })?;
            let span = semantic_code_ports::LineSpan::new(1, lines).map_err(ErrorEnvelope::from)?;
            Ok(vec![semantic_code_ports::CodeChunk {
                content: code,
                span,
                language: Some(language),
                file_path: options.file_path,
            }])
        })
    }

    fn set_chunk_size(&self, _chunk_size: usize) {}

    fn set_chunk_overlap(&self, _chunk_overlap: usize) {}
}

#[derive(Clone)]
struct AllowPathPolicy;

impl PathPolicyPort for AllowPathPolicy {
    fn to_safe_relative_path(&self, input: &str) -> Result<semantic_code_ports::SafeRelativePath> {
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
