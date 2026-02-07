//! Integration tests for the index codebase use case.

use semantic_code_app::{
    IndexCodebaseDeps, IndexCodebaseInput, IndexCodebaseStatus, index_codebase,
};
use semantic_code_domain::{
    CollectionName, EmbeddingProviderId, IndexMode, Language, LineSpan, VectorDbProviderId,
};
use semantic_code_ports::{
    CodeChunk, EmbeddingProviderInfo, FileSystemDirEntry, FileSystemEntryKind, FileSystemPort,
    FileSystemStat, IgnoreMatchInput, IgnorePort, PathPolicyPort, SplitOptions, SplitterPort,
    VectorDbProviderInfo,
};
use semantic_code_shared::{ErrorEnvelope, RequestContext, Result};
use semantic_code_testkit::in_memory::{InMemoryEmbeddingFixed, InMemoryVectorDbFixed};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

struct FixturePathPolicy;

impl PathPolicyPort for FixturePathPolicy {
    fn to_safe_relative_path(&self, input: &str) -> Result<semantic_code_ports::SafeRelativePath> {
        semantic_code_ports::SafeRelativePath::new(input)
    }
}

struct FixtureIgnore;

impl IgnorePort for FixtureIgnore {
    fn is_ignored(&self, _input: &IgnoreMatchInput) -> bool {
        false
    }
}

struct ExactMatchIgnore;

impl IgnorePort for ExactMatchIgnore {
    fn is_ignored(&self, input: &IgnoreMatchInput) -> bool {
        input
            .ignore_patterns
            .iter()
            .any(|pattern| pattern.as_ref() == input.relative_path.as_ref())
    }
}

struct FixtureSplitter;

impl SplitterPort for FixtureSplitter {
    fn split(
        &self,
        _ctx: &RequestContext,
        code: Box<str>,
        language: Language,
        options: SplitOptions,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<CodeChunk>>> {
        Box::pin(async move {
            let lines = code.lines().count().max(1) as u32;
            let span = LineSpan::new(1, lines).map_err(ErrorEnvelope::from)?;
            Ok(vec![CodeChunk {
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

struct FixtureFileSystem;

impl FixtureFileSystem {
    fn resolve_path(root: &Path, relative: &semantic_code_ports::SafeRelativePath) -> PathBuf {
        if relative.as_str() == "." {
            root.to_path_buf()
        } else {
            root.join(relative.as_str())
        }
    }

    fn to_mtime_ms(time: SystemTime) -> u64 {
        time.duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or(0)
    }
}

impl FileSystemPort for FixtureFileSystem {
    fn read_dir(
        &self,
        _ctx: &RequestContext,
        codebase_root: PathBuf,
        dir: semantic_code_ports::SafeRelativePath,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<FileSystemDirEntry>>> {
        Box::pin(async move {
            let path = Self::resolve_path(&codebase_root, &dir);
            let mut entries = tokio::fs::read_dir(&path)
                .await
                .map_err(ErrorEnvelope::from)?;
            let mut out = Vec::new();
            while let Some(entry) = entries.next_entry().await.map_err(ErrorEnvelope::from)? {
                let file_type = entry.file_type().await.map_err(ErrorEnvelope::from)?;
                let kind = if file_type.is_dir() {
                    FileSystemEntryKind::Directory
                } else if file_type.is_file() {
                    FileSystemEntryKind::File
                } else {
                    FileSystemEntryKind::Other
                };
                let name = entry
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
                    .into_boxed_str();
                out.push(FileSystemDirEntry { name, kind });
            }
            Ok(out)
        })
    }

    fn read_file_text(
        &self,
        _ctx: &RequestContext,
        codebase_root: PathBuf,
        file: semantic_code_ports::SafeRelativePath,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Box<str>>> {
        Box::pin(async move {
            let path = Self::resolve_path(&codebase_root, &file);
            let contents = tokio::fs::read_to_string(path)
                .await
                .map_err(ErrorEnvelope::from)?;
            Ok(contents.into_boxed_str())
        })
    }

    fn stat(
        &self,
        _ctx: &RequestContext,
        codebase_root: PathBuf,
        path: semantic_code_ports::SafeRelativePath,
    ) -> semantic_code_ports::BoxFuture<'_, Result<FileSystemStat>> {
        Box::pin(async move {
            let resolved = Self::resolve_path(&codebase_root, &path);
            let metadata = tokio::fs::metadata(resolved)
                .await
                .map_err(ErrorEnvelope::from)?;
            let kind = if metadata.is_dir() {
                FileSystemEntryKind::Directory
            } else if metadata.is_file() {
                FileSystemEntryKind::File
            } else {
                FileSystemEntryKind::Other
            };
            let mtime_ms = metadata.modified().map(Self::to_mtime_ms).unwrap_or(0);
            Ok(FileSystemStat {
                kind,
                size_bytes: metadata.len(),
                mtime_ms,
            })
        })
    }
}

fn embedding_provider() -> EmbeddingProviderId {
    EmbeddingProviderId::parse("openai").expect("embedding provider")
}

fn vectordb_provider() -> VectorDbProviderId {
    VectorDbProviderId::parse("milvus_grpc").expect("vectordb provider")
}

fn create_contextignore_fixture() -> Result<PathBuf> {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let root = std::env::temp_dir().join(format!("sca-contextignore-{unique}"));
    let src_dir = root.join("src");

    std::fs::create_dir_all(&src_dir).map_err(ErrorEnvelope::from)?;
    std::fs::write(src_dir.join("main.rs"), "fn main() {}\n").map_err(ErrorEnvelope::from)?;
    std::fs::write(src_dir.join("lib.rs"), "pub fn lib() {}\n").map_err(ErrorEnvelope::from)?;
    std::fs::write(root.join(".contextignore"), "src/main.rs\n").map_err(ErrorEnvelope::from)?;

    Ok(root)
}

#[tokio::test]
async fn index_fixture_repo_with_in_memory_adapters() -> Result<()> {
    let codebase_root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testkit/fixtures/index/basic");

    let embedding = Arc::new(InMemoryEmbeddingFixed::<3>::new(EmbeddingProviderInfo {
        id: embedding_provider(),
        name: "fixture".into(),
    }));
    let vectordb = Arc::new(InMemoryVectorDbFixed::<3>::new(VectorDbProviderInfo {
        id: vectordb_provider(),
        name: "fixture".into(),
    }));

    let deps = IndexCodebaseDeps {
        embedding,
        vectordb,
        splitter: Arc::new(FixtureSplitter),
        filesystem: Arc::new(FixtureFileSystem),
        path_policy: Arc::new(FixturePathPolicy),
        ignore: Arc::new(FixtureIgnore),
        logger: None,
        telemetry: None,
    };

    let input = IndexCodebaseInput {
        codebase_root,
        collection_name: CollectionName::parse("code_chunks_fixture")
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
        max_in_flight_files: Some(NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN)),
        max_in_flight_embedding_batches: Some(NonZeroUsize::MIN),
        max_in_flight_inserts: Some(NonZeroUsize::MIN),
    };

    let ctx = RequestContext::new_request();
    let output = index_codebase(&ctx, &deps, input).await?;

    assert_eq!(output.status, IndexCodebaseStatus::Completed);
    assert_eq!(output.indexed_files, 2);
    assert_eq!(output.total_chunks, 2);

    Ok(())
}

#[tokio::test]
async fn index_respects_contextignore_file() -> Result<()> {
    let codebase_root = create_contextignore_fixture()?;

    let embedding = Arc::new(InMemoryEmbeddingFixed::<3>::new(EmbeddingProviderInfo {
        id: embedding_provider(),
        name: "fixture".into(),
    }));
    let vectordb = Arc::new(InMemoryVectorDbFixed::<3>::new(VectorDbProviderInfo {
        id: vectordb_provider(),
        name: "fixture".into(),
    }));

    let deps = IndexCodebaseDeps {
        embedding,
        vectordb,
        splitter: Arc::new(FixtureSplitter),
        filesystem: Arc::new(FixtureFileSystem),
        path_policy: Arc::new(FixturePathPolicy),
        ignore: Arc::new(ExactMatchIgnore),
        logger: None,
        telemetry: None,
    };

    let input = IndexCodebaseInput {
        codebase_root: codebase_root.clone(),
        collection_name: CollectionName::parse("code_chunks_contextignore")
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
        max_in_flight_files: Some(NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN)),
        max_in_flight_embedding_batches: Some(NonZeroUsize::MIN),
        max_in_flight_inserts: Some(NonZeroUsize::MIN),
    };

    let ctx = RequestContext::new_request();
    let output = index_codebase(&ctx, &deps, input).await?;

    std::fs::remove_dir_all(&codebase_root).map_err(ErrorEnvelope::from)?;

    assert_eq!(output.status, IndexCodebaseStatus::Completed);
    assert_eq!(output.indexed_files, 1);
    assert_eq!(output.total_chunks, 1);

    Ok(())
}
