//! Local vector database adapter backed by HNSW.

use semantic_code_config::SnapshotStorageMode;
use semantic_code_domain::{IndexMode, Language};
use semantic_code_ports::{
    CollectionName, HybridSearchBatchRequest, HybridSearchData, HybridSearchResult, VectorDbPort,
    VectorDbProviderId, VectorDbProviderInfo, VectorDbRow, VectorDocument, VectorDocumentForInsert,
    VectorDocumentMetadata, VectorSearchRequest, VectorSearchResult,
};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use semantic_code_vector::{HnswParams, VectorIndex, VectorRecord};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

const LOCAL_SNAPSHOT_VERSION: u32 = 1;
const LOCAL_SNAPSHOT_DIR: &str = "vector";
const LOCAL_COLLECTIONS_DIR: &str = "collections";

/// Local vector DB backed by an HNSW index.
pub struct LocalVectorDb {
    provider: VectorDbProviderInfo,
    codebase_root: PathBuf,
    storage_mode: SnapshotStorageMode,
    collections: Arc<RwLock<HashMap<CollectionName, LocalCollection>>>,
}

impl LocalVectorDb {
    /// Create a local vector DB adapter scoped to a codebase root.
    pub fn new(codebase_root: PathBuf, storage_mode: SnapshotStorageMode) -> Result<Self> {
        let provider = VectorDbProviderInfo {
            id: VectorDbProviderId::parse("local").map_err(ErrorEnvelope::from)?,
            name: "Local".into(),
        };
        Ok(Self {
            provider,
            codebase_root,
            storage_mode,
            collections: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    fn snapshot_root(&self) -> Option<PathBuf> {
        self.storage_mode
            .resolve_root(&self.codebase_root)
            .map(|root| root.join(LOCAL_SNAPSHOT_DIR).join(LOCAL_COLLECTIONS_DIR))
    }

    fn snapshot_path(&self, collection_name: &CollectionName) -> Option<PathBuf> {
        let root = self.snapshot_root()?;
        Some(root.join(format!("{}.json", collection_name.as_str())))
    }

    async fn ensure_loaded(&self, collection_name: &CollectionName) -> Result<()> {
        {
            let collections = self.collections.read().await;
            if collections.contains_key(collection_name) {
                return Ok(());
            }
        }

        let snapshot = self.read_snapshot(collection_name).await?;
        let Some(snapshot) = snapshot else {
            return Ok(());
        };
        let collection = LocalCollection::from_snapshot(snapshot)?;
        self.collections
            .write()
            .await
            .entry(collection_name.clone())
            .or_insert(collection);
        Ok(())
    }

    async fn read_snapshot(
        &self,
        collection_name: &CollectionName,
    ) -> Result<Option<CollectionSnapshot>> {
        let Some(path) = self.snapshot_path(collection_name) else {
            return Ok(None);
        };

        match tokio::fs::read(&path).await {
            Ok(payload) => {
                let snapshot = serde_json::from_slice(&payload).map_err(|error| {
                    snapshot_error("snapshot_parse_failed", "failed to parse snapshot", error)
                })?;
                Ok(Some(snapshot))
            },
            Err(error) => {
                if error.kind() == std::io::ErrorKind::NotFound {
                    Ok(None)
                } else {
                    Err(ErrorEnvelope::from(error))
                }
            },
        }
    }

    async fn write_snapshot(
        &self,
        collection_name: &CollectionName,
        snapshot: &CollectionSnapshot,
    ) -> Result<()> {
        let Some(path) = self.snapshot_path(collection_name) else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(ErrorEnvelope::from)?;
        }
        let payload = serde_json::to_vec_pretty(snapshot).map_err(|error| {
            snapshot_error(
                "snapshot_serialize_failed",
                "failed to serialize snapshot",
                error,
            )
        })?;
        tokio::fs::write(&path, payload)
            .await
            .map_err(ErrorEnvelope::from)?;
        Ok(())
    }
}

impl VectorDbPort for LocalVectorDb {
    fn provider(&self) -> &VectorDbProviderInfo {
        &self.provider
    }

    fn create_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        dimension: u32,
        _description: Option<Box<str>>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let db = self.clone();
        Box::pin(async move {
            ctx.ensure_not_cancelled("vectordb_local.create_collection")?;
            let collection = LocalCollection::new(dimension, IndexMode::Dense)?;
            let mut guard = db.collections.write().await;
            guard.insert(collection_name.clone(), collection);
            let snapshot = guard.get(&collection_name).map(LocalCollection::snapshot);
            drop(guard);
            let Some(snapshot) = snapshot else {
                return Ok(());
            };
            db.write_snapshot(&collection_name, &snapshot).await
        })
    }

    fn create_hybrid_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        dimension: u32,
        _description: Option<Box<str>>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let db = self.clone();
        Box::pin(async move {
            ctx.ensure_not_cancelled("vectordb_local.create_hybrid_collection")?;
            let collection = LocalCollection::new(dimension, IndexMode::Hybrid)?;
            let mut guard = db.collections.write().await;
            guard.insert(collection_name.clone(), collection);
            let snapshot = guard.get(&collection_name).map(LocalCollection::snapshot);
            drop(guard);
            let Some(snapshot) = snapshot else {
                return Ok(());
            };
            db.write_snapshot(&collection_name, &snapshot).await
        })
    }

    fn drop_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let collections = Arc::clone(&self.collections);
        let snapshot = self.snapshot_path(&collection_name);
        Box::pin(async move {
            ctx.ensure_not_cancelled("vectordb_local.drop_collection")?;
            let mut guard = collections.write().await;
            guard.remove(&collection_name);
            drop(guard);

            if let Some(path) = snapshot {
                match tokio::fs::remove_file(&path).await {
                    Ok(()) => (),
                    Err(error) => {
                        if error.kind() != std::io::ErrorKind::NotFound {
                            return Err(ErrorEnvelope::from(error));
                        }
                    },
                }
            }
            Ok(())
        })
    }

    fn has_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
    ) -> semantic_code_ports::BoxFuture<'_, Result<bool>> {
        let ctx = ctx.clone();
        let collections = Arc::clone(&self.collections);
        let snapshot = self.snapshot_path(&collection_name);
        Box::pin(async move {
            ctx.ensure_not_cancelled("vectordb_local.has_collection")?;
            let guard = collections.read().await;
            if guard.contains_key(&collection_name) {
                return Ok(true);
            }
            drop(guard);

            let Some(path) = snapshot else {
                return Ok(false);
            };

            match tokio::fs::metadata(&path).await {
                Ok(metadata) => Ok(metadata.is_file()),
                Err(error) => {
                    if error.kind() == std::io::ErrorKind::NotFound {
                        Ok(false)
                    } else {
                        Err(ErrorEnvelope::from(error))
                    }
                },
            }
        })
    }

    fn list_collections(
        &self,
        ctx: &RequestContext,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<CollectionName>>> {
        let ctx = ctx.clone();
        let collections = Arc::clone(&self.collections);
        let snapshot_root = self.snapshot_root();
        Box::pin(async move {
            ctx.ensure_not_cancelled("vectordb_local.list_collections")?;
            let guard = collections.read().await;
            let mut names: BTreeMap<Box<str>, CollectionName> = guard
                .keys()
                .map(|name| (name.as_str().into(), name.clone()))
                .collect();
            drop(guard);

            let Some(root) = snapshot_root else {
                return Ok(names.into_values().collect());
            };

            let mut dir = match tokio::fs::read_dir(&root).await {
                Ok(dir) => dir,
                Err(error) => {
                    if error.kind() == std::io::ErrorKind::NotFound {
                        return Ok(names.into_values().collect());
                    }
                    return Err(ErrorEnvelope::from(error));
                },
            };

            while let Some(entry) = dir.next_entry().await.map_err(ErrorEnvelope::from)? {
                let name = entry.file_name().to_string_lossy().to_string();
                if let Some(collection) = collection_name_from_filename(&name) {
                    names
                        .entry(collection.as_str().into())
                        .or_insert(collection);
                }
            }

            Ok(names.into_values().collect())
        })
    }

    fn insert(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let db = self.clone();
        Box::pin(async move {
            ctx.ensure_not_cancelled("vectordb_local.insert")?;
            db.ensure_loaded(&collection_name).await?;
            let mut guard = db.collections.write().await;
            let Some(collection) = guard.get_mut(&collection_name) else {
                return Err(ErrorEnvelope::expected(
                    ErrorCode::not_found(),
                    "collection not found",
                ));
            };

            collection.insert(documents)?;
            let snapshot = collection.snapshot();
            drop(guard);
            db.write_snapshot(&collection_name, &snapshot).await
        })
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
        ctx: &RequestContext,
        request: VectorSearchRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<VectorSearchResult>>> {
        let ctx = ctx.clone();
        let db = self.clone();
        let VectorSearchRequest {
            collection_name,
            query_vector,
            options,
        } = request;
        Box::pin(async move {
            ctx.ensure_not_cancelled("vectordb_local.search")?;
            db.ensure_loaded(&collection_name).await?;
            let top_k = options.top_k.unwrap_or(10).max(1) as usize;
            let threshold = options.threshold;
            let filter = parse_filter_expr(options.filter_expr.as_deref())?;

            let results = {
                let guard = db.collections.read().await;
                let Some(collection) = guard.get(&collection_name) else {
                    return Err(ErrorEnvelope::expected(
                        ErrorCode::not_found(),
                        "collection not found",
                    ));
                };

                let matches = collection
                    .index
                    .search(query_vector.as_ref(), top_k.saturating_mul(5))?;

                let mut results = Vec::new();
                for candidate in matches {
                    let Some(doc) = collection.documents.get(candidate.id.as_ref()) else {
                        continue;
                    };
                    if !filter_matches(filter.as_ref(), doc) {
                        continue;
                    }
                    let score = candidate.score;
                    if threshold.is_some_and(|value| score < value) {
                        continue;
                    }
                    results.push(VectorSearchResult {
                        document: VectorDocument {
                            id: candidate.id,
                            vector: None,
                            content: doc.content.clone(),
                            metadata: doc.metadata.clone(),
                        },
                        score,
                    });
                    if results.len() >= top_k {
                        break;
                    }
                }

                drop(guard);
                results
            };

            Ok(results)
        })
    }

    fn hybrid_search(
        &self,
        ctx: &RequestContext,
        request: HybridSearchBatchRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<HybridSearchResult>>> {
        let ctx = ctx.clone();
        let db = self.clone();
        let HybridSearchBatchRequest {
            collection_name,
            search_requests,
            options,
        } = request;
        Box::pin(async move {
            ctx.ensure_not_cancelled("vectordb_local.hybrid_search")?;
            db.ensure_loaded(&collection_name).await?;
            let mut merged: HashMap<Box<str>, HybridSearchResult> = HashMap::new();
            let global_limit = options.limit.map(|value| value.max(1) as usize);
            let filter = parse_filter_expr(options.filter_expr.as_deref())?;

            {
                let guard = db.collections.read().await;
                let Some(collection) = guard.get(&collection_name) else {
                    return Err(ErrorEnvelope::expected(
                        ErrorCode::not_found(),
                        "collection not found",
                    ));
                };

                for req in search_requests {
                    let limit = req.limit.max(1) as usize;
                    let query = match req.data {
                        HybridSearchData::DenseVector(vector) => vector,
                        HybridSearchData::SparseQuery(_) => {
                            continue;
                        },
                    };
                    let matches = collection
                        .index
                        .search(query.as_ref(), limit.saturating_mul(5))?;

                    for candidate in matches {
                        let Some(doc) = collection.documents.get(candidate.id.as_ref()) else {
                            continue;
                        };
                        if !filter_matches(filter.as_ref(), doc) {
                            continue;
                        }
                        let entry = merged.entry(candidate.id.clone()).or_insert_with(|| {
                            HybridSearchResult {
                                document: VectorDocument {
                                    id: candidate.id.clone(),
                                    vector: None,
                                    content: doc.content.clone(),
                                    metadata: doc.metadata.clone(),
                                },
                                score: candidate.score,
                            }
                        });
                        if candidate.score > entry.score {
                            entry.score = candidate.score;
                        }
                    }
                }
                drop(guard);
            }

            let mut out: Vec<HybridSearchResult> = merged.into_values().collect();
            out.sort_by(|a, b| {
                let score = b.score.total_cmp(&a.score);
                if score != std::cmp::Ordering::Equal {
                    return score;
                }
                a.document.id.cmp(&b.document.id)
            });

            if let Some(limit) = global_limit {
                out.truncate(limit);
            }

            Ok(out)
        })
    }

    fn delete(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        ids: Vec<Box<str>>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let db = self.clone();
        Box::pin(async move {
            ctx.ensure_not_cancelled("vectordb_local.delete")?;
            db.ensure_loaded(&collection_name).await?;
            let mut guard = db.collections.write().await;
            let Some(collection) = guard.get_mut(&collection_name) else {
                return Err(ErrorEnvelope::expected(
                    ErrorCode::not_found(),
                    "collection not found",
                ));
            };
            collection.delete(&ids)?;
            let snapshot = collection.snapshot();
            drop(guard);
            db.write_snapshot(&collection_name, &snapshot).await
        })
    }

    fn query(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        filter: Box<str>,
        output_fields: Vec<Box<str>>,
        limit: Option<u32>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<VectorDbRow>>> {
        let ctx = ctx.clone();
        let db = self.clone();
        Box::pin(async move {
            ctx.ensure_not_cancelled("vectordb_local.query")?;
            db.ensure_loaded(&collection_name).await?;
            let limit = limit.map(|value| value.max(1) as usize);
            let filter = parse_filter_expr(Some(filter.as_ref()))?;

            let rows = {
                let guard = db.collections.read().await;
                let Some(collection) = guard.get(&collection_name) else {
                    return Err(ErrorEnvelope::expected(
                        ErrorCode::not_found(),
                        "collection not found",
                    ));
                };
                let mut rows = Vec::new();
                for (id, doc) in &collection.documents {
                    if !filter_matches(filter.as_ref(), doc) {
                        continue;
                    }
                    rows.push(build_row(id, doc, &output_fields));
                    if limit.is_some_and(|value| rows.len() >= value) {
                        break;
                    }
                }
                drop(guard);
                rows
            };

            Ok(rows)
        })
    }
}

impl Clone for LocalVectorDb {
    fn clone(&self) -> Self {
        Self {
            provider: self.provider.clone(),
            codebase_root: self.codebase_root.clone(),
            storage_mode: self.storage_mode.clone(),
            collections: Arc::clone(&self.collections),
        }
    }
}

struct LocalCollection {
    dimension: u32,
    index_mode: IndexMode,
    index: VectorIndex,
    documents: BTreeMap<Box<str>, StoredDocument>,
}

impl LocalCollection {
    fn new(dimension: u32, index_mode: IndexMode) -> Result<Self> {
        let params = HnswParams::default();
        let index = VectorIndex::new(dimension, params)?;
        Ok(Self {
            dimension,
            index_mode,
            index,
            documents: BTreeMap::new(),
        })
    }

    fn insert(&mut self, documents: Vec<VectorDocumentForInsert>) -> Result<()> {
        let mut records = Vec::new();
        let mut docs = BTreeMap::new();
        for doc in documents {
            let id = doc.id.clone();
            records.push(VectorRecord {
                id: id.clone(),
                vector: doc.vector.as_ref().to_vec(),
            });
            docs.insert(
                id,
                StoredDocument {
                    content: doc.content,
                    metadata: doc.metadata,
                },
            );
        }

        self.index.insert(records)?;
        for (id, doc) in docs {
            self.documents.insert(id, doc);
        }
        Ok(())
    }

    fn delete(&mut self, ids: &[Box<str>]) -> Result<()> {
        self.index.delete(ids)?;
        for id in ids {
            self.documents.remove(id.as_ref());
        }
        Ok(())
    }

    fn snapshot(&self) -> CollectionSnapshot {
        let mut records = Vec::new();
        for (id, doc) in &self.documents {
            if let Some(record) = self.index.record_for_id(id.as_ref()) {
                records.push(CollectionRecord {
                    id: id.clone(),
                    vector: record.vector.clone(),
                    content: doc.content.clone(),
                    metadata: doc.metadata.clone(),
                });
            }
        }

        CollectionSnapshot {
            version: LOCAL_SNAPSHOT_VERSION,
            dimension: self.dimension,
            index_mode: self.index_mode,
            records,
        }
    }

    fn from_snapshot(snapshot: CollectionSnapshot) -> Result<Self> {
        if snapshot.version != LOCAL_SNAPSHOT_VERSION {
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("vector", "snapshot_version_mismatch"),
                "snapshot version mismatch",
            )
            .with_metadata("found", snapshot.version.to_string())
            .with_metadata("expected", LOCAL_SNAPSHOT_VERSION.to_string()));
        }
        let params = HnswParams::default();
        let mut index = VectorIndex::new(snapshot.dimension, params)?;
        let mut documents = BTreeMap::new();
        let mut records = Vec::new();
        for record in snapshot.records {
            records.push(VectorRecord {
                id: record.id.clone(),
                vector: record.vector.clone(),
            });
            documents.insert(
                record.id.clone(),
                StoredDocument {
                    content: record.content,
                    metadata: record.metadata,
                },
            );
        }
        index.insert(records)?;
        Ok(Self {
            dimension: snapshot.dimension,
            index_mode: snapshot.index_mode,
            index,
            documents,
        })
    }
}

#[derive(Debug, Clone)]
struct StoredDocument {
    content: Box<str>,
    metadata: VectorDocumentMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CollectionSnapshot {
    version: u32,
    dimension: u32,
    index_mode: IndexMode,
    records: Vec<CollectionRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CollectionRecord {
    id: Box<str>,
    vector: Vec<f32>,
    content: Box<str>,
    metadata: VectorDocumentMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilterOp {
    Eq,
    NotEq,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilterField {
    RelativePath,
    Language,
    FileExtension,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FilterCondition {
    field: FilterField,
    op: FilterOp,
    value: Box<str>,
}

fn parse_filter_expr(expr: Option<&str>) -> Result<Option<FilterCondition>> {
    let Some(expr) = expr else {
        return Ok(None);
    };
    let expr = expr.trim();
    if expr.is_empty() {
        return Ok(None);
    }
    if expr.contains('\n') || expr.contains('\r') {
        return Err(invalid_filter_expr(expr));
    }

    let (field, op, value) =
        parse_simple_comparison(expr).ok_or_else(|| invalid_filter_expr(expr))?;
    let field = match field {
        "relativePath" => FilterField::RelativePath,
        "language" => FilterField::Language,
        "fileExtension" => FilterField::FileExtension,
        _ => return Err(invalid_filter_expr(expr)),
    };
    let op = match op {
        "==" => FilterOp::Eq,
        "!=" => FilterOp::NotEq,
        _ => return Err(invalid_filter_expr(expr)),
    };
    if value.is_empty() {
        return Err(invalid_filter_expr(expr));
    }

    Ok(Some(FilterCondition {
        field,
        op,
        value: value.to_owned().into_boxed_str(),
    }))
}

fn parse_simple_comparison(input: &str) -> Option<(&str, &str, &str)> {
    let input = input.trim();
    let (field, rest) = split_once_ws(input)?;
    let rest = rest.trim_start();

    let (op, rest) = if let Some(rest) = rest.strip_prefix("==") {
        ("==", rest)
    } else if let Some(rest) = rest.strip_prefix("!=") {
        ("!=", rest)
    } else {
        return None;
    };

    let value = rest.trim_start();
    let unquoted = strip_quotes(value)?;
    Some((field, op, unquoted))
}

fn split_once_ws(input: &str) -> Option<(&str, &str)> {
    for (idx, ch) in input.char_indices() {
        if ch.is_whitespace() {
            let (left, right) = input.split_at(idx);
            return Some((left, right));
        }
    }
    None
}

fn strip_quotes(input: &str) -> Option<&str> {
    let input = input.trim();
    if input.len() < 2 {
        return None;
    }
    let bytes = input.as_bytes();
    let first = *bytes.first()?;
    let last = *bytes.last()?;
    if (first == b'\'' && last == b'\'') || (first == b'"' && last == b'"') {
        Some(&input[1..input.len() - 1])
    } else {
        None
    }
}

fn filter_matches(filter: Option<&FilterCondition>, doc: &StoredDocument) -> bool {
    let Some(filter) = filter else {
        return true;
    };

    let value = match filter.field {
        FilterField::RelativePath => Some(doc.metadata.relative_path.as_ref()),
        FilterField::Language => doc.metadata.language.map(Language::as_str),
        FilterField::FileExtension => doc.metadata.file_extension.as_deref(),
    };

    match filter.op {
        FilterOp::Eq => value.is_some_and(|v| v == filter.value.as_ref()),
        FilterOp::NotEq => value.is_none_or(|v| v != filter.value.as_ref()),
    }
}

fn build_row(id: &str, doc: &StoredDocument, output_fields: &[Box<str>]) -> VectorDbRow {
    let mut row = BTreeMap::new();
    for field in output_fields {
        match field.as_ref() {
            "id" => {
                row.insert(field.clone(), Value::String(id.to_owned()));
            },
            "relativePath" => {
                row.insert(
                    field.clone(),
                    Value::String(doc.metadata.relative_path.as_ref().to_owned()),
                );
            },
            "language" => {
                if let Some(language) = doc.metadata.language {
                    row.insert(field.clone(), Value::String(language.as_str().to_owned()));
                }
            },
            "fileExtension" => {
                if let Some(ext) = doc.metadata.file_extension.as_ref() {
                    row.insert(field.clone(), Value::String(ext.as_ref().to_owned()));
                }
            },
            "startLine" => {
                row.insert(field.clone(), Value::from(doc.metadata.span.start_line()));
            },
            "endLine" => {
                row.insert(field.clone(), Value::from(doc.metadata.span.end_line()));
            },
            "content" => {
                row.insert(
                    field.clone(),
                    Value::String(doc.content.as_ref().to_owned()),
                );
            },
            _ => {},
        }
    }
    row
}

fn invalid_filter_expr(expr: &str) -> ErrorEnvelope {
    ErrorEnvelope::expected(
        ErrorCode::new("vector", "invalid_filter_expr"),
        format!("filterExpr is not supported: {expr}"),
    )
}

fn snapshot_error(
    code: &'static str,
    message: &str,
    error: impl std::error::Error,
) -> ErrorEnvelope {
    ErrorEnvelope::unexpected(
        ErrorCode::new("vector", code),
        format!("{message}: {error}"),
        ErrorClass::NonRetriable,
    )
}

fn collection_name_from_filename(filename: &str) -> Option<CollectionName> {
    let trimmed = filename.strip_suffix(".json")?;
    CollectionName::parse(trimmed).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_domain::LineSpan;
    use semantic_code_ports::VectorSearchOptions;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn sample_metadata(path: &str) -> Result<VectorDocumentMetadata> {
        Ok(VectorDocumentMetadata {
            relative_path: path.into(),
            language: None,
            file_extension: Some("rs".into()),
            span: LineSpan::new(1, 1)?,
            node_kind: None,
        })
    }

    #[tokio::test]
    async fn filter_expr_allowlist_accepts_valid_inputs() -> Result<()> {
        let parsed = parse_filter_expr(Some("relativePath == 'src/lib.rs'"))?;
        assert!(parsed.is_some());
        Ok(())
    }

    #[tokio::test]
    async fn filter_expr_rejects_unknown_field() {
        let error = parse_filter_expr(Some("score > 0.5")).err();
        assert!(matches!(
            error,
            Some(envelope) if envelope.code == ErrorCode::new("vector", "invalid_filter_expr")
        ));
    }

    #[tokio::test]
    async fn snapshot_roundtrip_persists_records() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let db = LocalVectorDb::new(tmp.clone(), SnapshotStorageMode::Custom(tmp.clone()))?;
        let collection = CollectionName::parse("local_snapshot")?;
        let ctx = RequestContext::new_request();
        db.create_collection(&ctx, collection.clone(), 3, None)
            .await?;
        db.insert(
            &ctx,
            collection.clone(),
            vec![VectorDocumentForInsert {
                id: "doc1".into(),
                vector: Arc::from(vec![0.1, 0.2, 0.3]),
                content: "hello".into(),
                metadata: sample_metadata("src/lib.rs")?,
            }],
        )
        .await?;

        let restored = LocalVectorDb::new(tmp.clone(), SnapshotStorageMode::Custom(tmp.clone()))?;
        let results = restored
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection,
                    query_vector: Arc::from(vec![0.1, 0.2, 0.3]),
                    options: VectorSearchOptions {
                        top_k: Some(1),
                        filter_expr: None,
                        threshold: None,
                    },
                },
            )
            .await?;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.id, "doc1".into());
        Ok(())
    }
}
