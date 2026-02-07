//! In-memory adapters for CLI self-check smoke tests.

use semantic_code_ports::{
    CodeChunk, CollectionName, DetectDimensionRequest, EmbedBatchRequest, EmbedRequest,
    EmbeddingPort, EmbeddingProviderInfo, EmbeddingVector, FileChangeSet, FileSyncInitOptions,
    FileSyncOptions, FileSyncPort, FileSystemDirEntry, FileSystemEntryKind, FileSystemPort,
    FileSystemStat, HybridSearchBatchRequest, HybridSearchData, HybridSearchOptions,
    HybridSearchRequest, HybridSearchResult, IgnoreMatchInput, IgnorePort, PathPolicyPort,
    SafeRelativePath, SplitOptions, SplitterPort, VectorDbPort, VectorDbProviderInfo, VectorDbRow,
    VectorDocument, VectorDocumentForInsert, VectorDocumentMetadata, VectorSearchOptions,
    VectorSearchRequest, VectorSearchResult,
};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// In-memory embedding adapter for self-check.
#[derive(Clone)]
pub struct SelfCheckEmbedding {
    provider: EmbeddingProviderInfo,
    vector: Arc<[f32]>,
}

impl SelfCheckEmbedding {
    /// Build a default embedding adapter.
    pub fn new() -> Result<Self> {
        let provider = EmbeddingProviderInfo {
            id: semantic_code_ports::EmbeddingProviderId::parse("openai")
                .map_err(ErrorEnvelope::from)?,
            name: "self-check".into(),
        };
        Ok(Self {
            provider,
            vector: Arc::from(vec![0.0, 0.0, 0.0]),
        })
    }

    fn dimension(&self) -> Result<u32> {
        u32::try_from(self.vector.len()).map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "embedding dimension overflow",
                ErrorClass::NonRetriable,
            )
        })
    }
}

impl EmbeddingPort for SelfCheckEmbedding {
    fn provider(&self) -> &EmbeddingProviderInfo {
        &self.provider
    }

    fn detect_dimension(
        &self,
        _ctx: &RequestContext,
        _request: DetectDimensionRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<u32>> {
        let dimension = match self.dimension() {
            Ok(value) => value,
            Err(error) => {
                return Box::pin(async move { Err(error) });
            },
        };
        Box::pin(async move { Ok(dimension) })
    }

    fn embed(
        &self,
        _ctx: &RequestContext,
        _request: EmbedRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<EmbeddingVector>> {
        let vector = Arc::clone(&self.vector);
        let _dimension = match self.dimension() {
            Ok(value) => value,
            Err(error) => {
                return Box::pin(async move { Err(error) });
            },
        };
        Box::pin(async move { Ok(EmbeddingVector::new(vector)) })
    }

    fn embed_batch(
        &self,
        _ctx: &RequestContext,
        request: EmbedBatchRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<EmbeddingVector>>> {
        let vector = Arc::clone(&self.vector);
        let _dimension = match self.dimension() {
            Ok(value) => value,
            Err(error) => {
                return Box::pin(async move { Err(error) });
            },
        };
        Box::pin(async move {
            let texts = request.texts;
            Ok(texts
                .into_iter()
                .map(|_| EmbeddingVector::new(Arc::clone(&vector)))
                .collect())
        })
    }
}

/// In-memory vector DB adapter for self-check.
#[derive(Clone)]
pub struct SelfCheckVectorDb {
    provider: VectorDbProviderInfo,
    collections: Arc<RwLock<HashMap<CollectionName, CollectionState>>>,
}

#[derive(Clone)]
struct CollectionState {
    dimension: u32,
    documents: HashMap<Box<str>, StoredDocument>,
}

#[derive(Clone)]
struct StoredDocument {
    vector: Vec<f32>,
    content: Box<str>,
    metadata: VectorDocumentMetadata,
}

impl SelfCheckVectorDb {
    /// Build a default vector DB adapter.
    pub fn new() -> Result<Self> {
        let provider = VectorDbProviderInfo {
            id: semantic_code_ports::VectorDbProviderId::parse("milvus_grpc")
                .map_err(ErrorEnvelope::from)?,
            name: "self-check".into(),
        };
        Ok(Self {
            provider,
            collections: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    async fn lock_collections(
        &self,
    ) -> tokio::sync::RwLockReadGuard<'_, HashMap<CollectionName, CollectionState>> {
        self.collections.read().await
    }

    async fn lock_collections_mut(
        &self,
    ) -> tokio::sync::RwLockWriteGuard<'_, HashMap<CollectionName, CollectionState>> {
        self.collections.write().await
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(lhs, rhs)| lhs * rhs).sum()
    }

    async fn search_internal(
        &self,
        collection_name: &CollectionName,
        query_vector: &[f32],
        options: &VectorSearchOptions,
    ) -> Result<Vec<VectorSearchResult>> {
        let (dimension, documents) = {
            let collections = self.lock_collections().await;
            let collection = collections.get(collection_name).ok_or_else(|| {
                ErrorEnvelope::expected(ErrorCode::not_found(), "collection not found")
            })?;
            let dimension = collection.dimension;
            let documents = collection.documents.clone();
            drop(collections);
            (dimension, documents)
        };

        let query_dimension = u32::try_from(query_vector.len()).map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "vector dimension overflow",
                ErrorClass::NonRetriable,
            )
        })?;
        if query_dimension != dimension {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "vector dimension mismatch",
            ));
        }

        let top_k = usize::try_from(options.top_k.unwrap_or(10).max(1)).map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "top_k overflow",
                ErrorClass::NonRetriable,
            )
        })?;
        let threshold = options.threshold;
        let _filter_expr = options.filter_expr.as_ref();

        let mut scored = documents
            .iter()
            .map(|(id, doc)| {
                let score = Self::dot(query_vector, &doc.vector);
                VectorSearchResult {
                    document: VectorDocument {
                        id: id.clone(),
                        vector: None,
                        content: doc.content.clone(),
                        metadata: doc.metadata.clone(),
                    },
                    score,
                }
            })
            .filter(|result| threshold.is_none_or(|value| result.score >= value))
            .collect::<Vec<_>>();
        drop(documents);

        scored.sort_by(|a, b| {
            let score = b.score.total_cmp(&a.score);
            if score != std::cmp::Ordering::Equal {
                return score;
            }
            a.document.id.cmp(&b.document.id)
        });
        scored.truncate(top_k);

        Ok(scored)
    }

    async fn hybrid_search_internal(
        &self,
        collection_name: &CollectionName,
        search_requests: Vec<HybridSearchRequest>,
        options: &HybridSearchOptions,
    ) -> Result<Vec<HybridSearchResult>> {
        let mut merged: HashMap<Box<str>, HybridSearchResult> = HashMap::new();
        let global_limit = options.limit;

        for req in search_requests {
            let limit = req.limit.max(1);
            let query = match req.data {
                HybridSearchData::DenseVector(vector) => vector,
                HybridSearchData::SparseQuery(_) => {
                    return Err(ErrorEnvelope::expected(
                        ErrorCode::new("core", "not_supported"),
                        "sparse hybrid search not supported by self-check adapter",
                    ));
                },
            };

            let results = self
                .search_internal(
                    collection_name,
                    query.as_ref(),
                    &VectorSearchOptions {
                        top_k: Some(limit),
                        threshold: None,
                        filter_expr: None,
                    },
                )
                .await?;

            for result in results {
                let id = result.document.id.clone();
                let entry = merged.entry(id).or_insert(HybridSearchResult {
                    document: result.document,
                    score: result.score,
                });
                if result.score > entry.score {
                    entry.score = result.score;
                }
            }
        }

        let mut merged = merged.into_values().collect::<Vec<_>>();
        merged.sort_by(|a, b| {
            let score = b.score.total_cmp(&a.score);
            if score != std::cmp::Ordering::Equal {
                return score;
            }
            a.document.id.cmp(&b.document.id)
        });
        if let Some(limit) = global_limit {
            let limit = usize::try_from(limit.max(1)).map_err(|_| {
                ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    "limit overflow",
                    ErrorClass::NonRetriable,
                )
            })?;
            merged.truncate(limit);
        }

        Ok(merged)
    }

    async fn insert_internal(
        &self,
        collection_name: &CollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> Result<()> {
        let mut collections = self.lock_collections_mut().await;
        let collection = collections.get_mut(collection_name).ok_or_else(|| {
            ErrorEnvelope::expected(ErrorCode::not_found(), "collection not found")
        })?;
        let result = Self::insert_documents(collection, documents);
        drop(collections);
        result
    }

    fn insert_documents(
        collection: &mut CollectionState,
        documents: Vec<VectorDocumentForInsert>,
    ) -> Result<()> {
        for doc in documents {
            let dimension = u32::try_from(doc.vector.len()).map_err(|_| {
                ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    "vector dimension overflow",
                    ErrorClass::NonRetriable,
                )
            })?;
            if dimension != collection.dimension {
                return Err(ErrorEnvelope::expected(
                    ErrorCode::invalid_input(),
                    "vector dimension mismatch",
                ));
            }
            collection.documents.insert(
                doc.id.clone(),
                StoredDocument {
                    vector: doc.vector.as_ref().to_vec(),
                    content: doc.content,
                    metadata: doc.metadata,
                },
            );
        }
        Ok(())
    }
}

impl VectorDbPort for SelfCheckVectorDb {
    fn provider(&self) -> &VectorDbProviderInfo {
        &self.provider
    }

    fn create_collection(
        &self,
        _ctx: &RequestContext,
        collection_name: CollectionName,
        dimension: u32,
        _description: Option<Box<str>>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let this = self.clone();
        Box::pin(async move {
            {
                let mut collections = this.lock_collections_mut().await;
                collections
                    .entry(collection_name)
                    .or_insert_with(|| CollectionState {
                        dimension,
                        documents: HashMap::new(),
                    });
            }
            Ok(())
        })
    }

    fn create_hybrid_collection(
        &self,
        _ctx: &RequestContext,
        collection_name: CollectionName,
        dimension: u32,
        _description: Option<Box<str>>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let this = self.clone();
        Box::pin(async move {
            {
                let mut collections = this.lock_collections_mut().await;
                collections
                    .entry(collection_name)
                    .or_insert_with(|| CollectionState {
                        dimension,
                        documents: HashMap::new(),
                    });
            }
            Ok(())
        })
    }

    fn drop_collection(
        &self,
        _ctx: &RequestContext,
        collection_name: CollectionName,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let this = self.clone();
        Box::pin(async move {
            {
                let mut collections = this.lock_collections_mut().await;
                collections.remove(&collection_name);
            }
            Ok(())
        })
    }

    fn has_collection(
        &self,
        _ctx: &RequestContext,
        collection_name: CollectionName,
    ) -> semantic_code_ports::BoxFuture<'_, Result<bool>> {
        let this = self.clone();
        Box::pin(async move {
            let collections = this.lock_collections().await;
            Ok(collections.contains_key(&collection_name))
        })
    }

    fn list_collections(
        &self,
        _ctx: &RequestContext,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<CollectionName>>> {
        let this = self.clone();
        Box::pin(async move {
            let collections = this.lock_collections().await;
            Ok(collections.keys().cloned().collect())
        })
    }

    fn insert(
        &self,
        _ctx: &RequestContext,
        collection_name: CollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let this = self.clone();
        Box::pin(async move { this.insert_internal(&collection_name, documents).await })
    }

    fn insert_hybrid(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        self.insert(ctx, collection_name, documents)
    }

    fn search(
        &self,
        _ctx: &RequestContext,
        request: VectorSearchRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<VectorSearchResult>>> {
        let this = self.clone();
        let VectorSearchRequest {
            collection_name,
            query_vector,
            options,
        } = request;
        Box::pin(async move {
            this.search_internal(&collection_name, query_vector.as_ref(), &options)
                .await
        })
    }

    fn hybrid_search(
        &self,
        _ctx: &RequestContext,
        request: HybridSearchBatchRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<HybridSearchResult>>> {
        let this = self.clone();
        let HybridSearchBatchRequest {
            collection_name,
            search_requests,
            options,
        } = request;
        Box::pin(async move {
            this.hybrid_search_internal(&collection_name, search_requests, &options)
                .await
        })
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
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<VectorDbRow>>> {
        Box::pin(async move { Ok(Vec::new()) })
    }
}

/// In-memory filesystem for self-check.
#[derive(Clone)]
pub struct SelfCheckFileSystem {
    state: Arc<RwLock<SelfCheckFileSystemState>>,
}

#[derive(Default)]
struct SelfCheckFileSystemState {
    files: HashMap<String, String>,
    dirs: HashMap<String, Vec<FileSystemDirEntry>>,
}

impl SelfCheckFileSystem {
    /// Build a filesystem with a small Rust fixture.
    #[must_use]
    pub fn new() -> Self {
        let mut state = SelfCheckFileSystemState::default();
        state.add_file("src/main.rs", "fn main() { println!(\"ok\"); }\n");
        state.add_file("src/lib.rs", "pub fn meaning() -> i32 { 42 }\n");
        Self {
            state: Arc::new(RwLock::new(state)),
        }
    }
}

impl Default for SelfCheckFileSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl SelfCheckFileSystemState {
    fn add_file(&mut self, path: &str, content: &str) {
        let normalized = path.replace('\\', "/");
        self.files.insert(normalized.clone(), content.to_string());

        let (dir, name) = normalized
            .rsplit_once('/')
            .map_or((".", normalized.as_str()), |(dir, name)| (dir, name));

        self.add_dir_entry(dir, name, FileSystemEntryKind::File);
        self.ensure_dirs(dir);
    }

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

impl FileSystemPort for SelfCheckFileSystem {
    fn read_dir(
        &self,
        _ctx: &RequestContext,
        _codebase_root: PathBuf,
        dir: SafeRelativePath,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<FileSystemDirEntry>>> {
        let state = Arc::clone(&self.state);
        Box::pin(async move {
            let state = state.read().await;
            Ok(state.dirs.get(dir.as_str()).cloned().unwrap_or_default())
        })
    }

    fn read_file_text(
        &self,
        _ctx: &RequestContext,
        _codebase_root: PathBuf,
        file: SafeRelativePath,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Box<str>>> {
        let state = Arc::clone(&self.state);
        Box::pin(async move {
            let state = state.read().await;
            state
                .files
                .get(file.as_str())
                .map(|value| value.clone().into_boxed_str())
                .ok_or_else(|| ErrorEnvelope::expected(ErrorCode::not_found(), "missing file"))
        })
    }

    fn stat(
        &self,
        _ctx: &RequestContext,
        _codebase_root: PathBuf,
        path: SafeRelativePath,
    ) -> semantic_code_ports::BoxFuture<'_, Result<FileSystemStat>> {
        let state = Arc::clone(&self.state);
        Box::pin(async move {
            let state = state.read().await;
            let stat = if path.as_str() == "." || state.dirs.contains_key(path.as_str()) {
                FileSystemStat {
                    kind: FileSystemEntryKind::Directory,
                    size_bytes: 0,
                    mtime_ms: 0,
                }
            } else if let Some(contents) = state.files.get(path.as_str()) {
                FileSystemStat {
                    kind: FileSystemEntryKind::File,
                    size_bytes: contents.len() as u64,
                    mtime_ms: 0,
                }
            } else {
                FileSystemStat {
                    kind: FileSystemEntryKind::Other,
                    size_bytes: 0,
                    mtime_ms: 0,
                }
            };
            drop(state);
            Ok(stat)
        })
    }
}

/// In-memory file sync adapter for self-check.
#[derive(Clone, Default)]
pub struct SelfCheckFileSync;

impl SelfCheckFileSync {
    /// Build a default file sync adapter.
    pub const fn new() -> Self {
        Self
    }
}

impl FileSyncPort for SelfCheckFileSync {
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

/// Path policy used for self-check.
pub struct SelfCheckPathPolicy;

impl PathPolicyPort for SelfCheckPathPolicy {
    fn to_safe_relative_path(&self, input: &str) -> Result<SafeRelativePath> {
        SafeRelativePath::new(input)
    }
}

/// Ignore matcher used for self-check.
pub struct SelfCheckIgnore;

impl IgnorePort for SelfCheckIgnore {
    fn is_ignored(&self, _input: &IgnoreMatchInput) -> bool {
        false
    }
}

/// Splitter used for self-check (single chunk).
pub struct SelfCheckSplitter;

impl SplitterPort for SelfCheckSplitter {
    fn split(
        &self,
        _ctx: &RequestContext,
        code: Box<str>,
        language: semantic_code_ports::Language,
        options: SplitOptions,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<CodeChunk>>> {
        Box::pin(async move {
            let lines = u32::try_from(code.lines().count().max(1)).map_err(|_| {
                ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    "line count overflow",
                    ErrorClass::NonRetriable,
                )
            })?;
            let span = semantic_code_ports::LineSpan::new(1, lines).map_err(ErrorEnvelope::from)?;
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
