//! Vector DB boundary contract.

use crate::BoxFuture;
use semantic_code_domain::{CollectionName, VectorDbProviderId, VectorDocumentMetadata};
use semantic_code_shared::{RequestContext, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;

/// Provider descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorDbProviderInfo {
    /// Stable provider identifier.
    pub id: VectorDbProviderId,
    /// Human-readable provider name.
    pub name: Box<str>,
}

/// A vector document stored in the vector DB.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorDocument {
    /// Stable document identifier (typically a chunk id).
    pub id: Box<str>,
    /// Dense vector (typically omitted in results).
    pub vector: Option<Arc<[f32]>>,
    /// Content payload.
    pub content: Box<str>,
    /// Typed metadata for search results and filters.
    pub metadata: VectorDocumentMetadata,
}

/// A vector document for insert (vector required).
#[derive(Debug, Clone, PartialEq)]
pub struct VectorDocumentForInsert {
    /// Stable document identifier (typically a chunk id).
    pub id: Box<str>,
    /// Dense embedding vector (required for inserts).
    pub vector: Arc<[f32]>,
    /// Content payload.
    pub content: Box<str>,
    /// Typed metadata for filters and result shaping.
    pub metadata: VectorDocumentMetadata,
}

/// Options for dense vector search.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VectorSearchOptions {
    /// Maximum number of results to return.
    pub top_k: Option<u32>,
    /// Optional provider-specific filter expression.
    pub filter_expr: Option<Box<str>>,
    /// Optional score threshold.
    pub threshold: Option<f32>,
}

/// Owned request for dense vector search.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorSearchRequest {
    /// Target collection name.
    pub collection_name: CollectionName,
    /// Dense query vector.
    pub query_vector: Arc<[f32]>,
    /// Search options.
    pub options: VectorSearchOptions,
}

/// Dense vector search result.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorSearchResult {
    /// Result document (without embeddings).
    pub document: VectorDocument,
    /// Similarity score.
    pub score: f32,
}

/// Hybrid search request payload.
#[derive(Debug, Clone, PartialEq)]
pub enum HybridSearchData {
    /// Dense embedding vector.
    DenseVector(Arc<[f32]>),
    /// Sparse query text (provider-specific).
    SparseQuery(Box<str>),
}

/// Provider-specific hybrid search request.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridSearchRequest {
    /// Dense vector or query text (for sparse search).
    pub data: HybridSearchData,
    /// Vector field name (e.g. `vector` or `sparse_vector`).
    pub anns_field: Box<str>,
    /// Provider-specific parameters (e.g. nprobe).
    pub params: BTreeMap<Box<str>, Value>,
    /// Result limit for this sub-query.
    pub limit: u32,
}

/// Rerank strategy kind for hybrid search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankStrategyKind {
    /// Reciprocal Rank Fusion.
    Rrf,
    /// Weighted merge.
    Weighted,
}

/// Rerank configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerankStrategy {
    /// Rerank strategy.
    pub strategy: RerankStrategyKind,
    /// Provider-specific params.
    pub params: Option<BTreeMap<Box<str>, Value>>,
}

/// Options for hybrid search.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HybridSearchOptions {
    /// Optional rerank strategy.
    pub rerank: Option<RerankStrategy>,
    /// Optional global limit.
    pub limit: Option<u32>,
    /// Optional provider-specific filter expression.
    pub filter_expr: Option<Box<str>>,
}

/// Owned hybrid search batch request.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridSearchBatchRequest {
    /// Target collection name.
    pub collection_name: CollectionName,
    /// Sub-queries for hybrid search.
    pub search_requests: Vec<HybridSearchRequest>,
    /// Hybrid search options.
    pub options: HybridSearchOptions,
}

/// Hybrid search result.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridSearchResult {
    /// Result document (without embeddings).
    pub document: VectorDocument,
    /// Similarity score.
    pub score: f32,
}

/// Boundary contract for vector storage + retrieval.
pub trait VectorDbPort: Send + Sync {
    /// Provider info for this implementation.
    fn provider(&self) -> &VectorDbProviderInfo;

    /// Create a dense-only collection.
    fn create_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        dimension: u32,
        description: Option<Box<str>>,
    ) -> BoxFuture<'_, Result<()>>;

    /// Create a hybrid-capable collection.
    fn create_hybrid_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        dimension: u32,
        description: Option<Box<str>>,
    ) -> BoxFuture<'_, Result<()>>;

    /// Drop a collection (best-effort).
    fn drop_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
    ) -> BoxFuture<'_, Result<()>>;

    /// Return true when the collection exists.
    fn has_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
    ) -> BoxFuture<'_, Result<bool>>;

    /// List available collections.
    fn list_collections(&self, ctx: &RequestContext) -> BoxFuture<'_, Result<Vec<CollectionName>>>;

    /// Insert documents into a dense collection.
    fn insert(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> BoxFuture<'_, Result<()>>;

    /// Insert documents into a hybrid collection.
    fn insert_hybrid(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> BoxFuture<'_, Result<()>>;

    /// Perform a dense vector search.
    fn search(
        &self,
        ctx: &RequestContext,
        request: VectorSearchRequest,
    ) -> BoxFuture<'_, Result<Vec<VectorSearchResult>>>;

    /// Perform a hybrid search (dense and/or sparse sub-queries).
    fn hybrid_search(
        &self,
        ctx: &RequestContext,
        request: HybridSearchBatchRequest,
    ) -> BoxFuture<'_, Result<Vec<HybridSearchResult>>>;

    /// Delete documents by id.
    fn delete(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        ids: Vec<Box<str>>,
    ) -> BoxFuture<'_, Result<()>>;

    /// Query documents using a provider-specific filter expression.
    fn query(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        filter: Box<str>,
        output_fields: Vec<Box<str>>,
        limit: Option<u32>,
    ) -> BoxFuture<'_, Result<Vec<VectorDbRow>>>;
}

/// Lending-style vector DB port using GAT futures.
pub trait VectorDbPortLend: Send + Sync {
    /// Future type returned by this port.
    type Future<'a, T>: Future<Output = Result<T>> + Send + 'a
    where
        Self: 'a,
        T: 'a;

    /// Provider info for this implementation.
    fn provider(&self) -> &VectorDbProviderInfo;

    /// Create a dense-only collection.
    fn create_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        dimension: u32,
        description: Option<Box<str>>,
    ) -> Self::Future<'_, ()>;

    /// Create a hybrid-capable collection.
    fn create_hybrid_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        dimension: u32,
        description: Option<Box<str>>,
    ) -> Self::Future<'_, ()>;

    /// Drop a collection (best-effort).
    fn drop_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
    ) -> Self::Future<'_, ()>;

    /// Return true when the collection exists.
    fn has_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
    ) -> Self::Future<'_, bool>;

    /// List available collections.
    fn list_collections(&self, ctx: &RequestContext) -> Self::Future<'_, Vec<CollectionName>>;

    /// Insert documents into a dense collection.
    fn insert(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> Self::Future<'_, ()>;

    /// Insert documents into a hybrid collection.
    fn insert_hybrid(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> Self::Future<'_, ()>;

    /// Perform a dense vector search.
    fn search(
        &self,
        ctx: &RequestContext,
        request: VectorSearchRequest,
    ) -> Self::Future<'_, Vec<VectorSearchResult>>;

    /// Perform a hybrid search (dense and/or sparse sub-queries).
    fn hybrid_search(
        &self,
        ctx: &RequestContext,
        request: HybridSearchBatchRequest,
    ) -> Self::Future<'_, Vec<HybridSearchResult>>;

    /// Delete documents by id.
    fn delete(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        ids: Vec<Box<str>>,
    ) -> Self::Future<'_, ()>;

    /// Query documents using a provider-specific filter expression.
    fn query(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        filter: Box<str>,
        output_fields: Vec<Box<str>>,
        limit: Option<u32>,
    ) -> Self::Future<'_, Vec<VectorDbRow>>;
}

impl<T> VectorDbPortLend for T
where
    T: VectorDbPort + ?Sized,
{
    type Future<'a, U>
        = BoxFuture<'a, Result<U>>
    where
        T: 'a,
        U: 'a;

    fn provider(&self) -> &VectorDbProviderInfo {
        VectorDbPort::provider(self)
    }

    fn create_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        dimension: u32,
        description: Option<Box<str>>,
    ) -> Self::Future<'_, ()> {
        VectorDbPort::create_collection(self, ctx, collection_name, dimension, description)
    }

    fn create_hybrid_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        dimension: u32,
        description: Option<Box<str>>,
    ) -> Self::Future<'_, ()> {
        VectorDbPort::create_hybrid_collection(self, ctx, collection_name, dimension, description)
    }

    fn drop_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
    ) -> Self::Future<'_, ()> {
        VectorDbPort::drop_collection(self, ctx, collection_name)
    }

    fn has_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
    ) -> Self::Future<'_, bool> {
        VectorDbPort::has_collection(self, ctx, collection_name)
    }

    fn list_collections(&self, ctx: &RequestContext) -> Self::Future<'_, Vec<CollectionName>> {
        VectorDbPort::list_collections(self, ctx)
    }

    fn insert(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> Self::Future<'_, ()> {
        VectorDbPort::insert(self, ctx, collection_name, documents)
    }

    fn insert_hybrid(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> Self::Future<'_, ()> {
        VectorDbPort::insert_hybrid(self, ctx, collection_name, documents)
    }

    fn search(
        &self,
        ctx: &RequestContext,
        request: VectorSearchRequest,
    ) -> Self::Future<'_, Vec<VectorSearchResult>> {
        VectorDbPort::search(self, ctx, request)
    }

    fn hybrid_search(
        &self,
        ctx: &RequestContext,
        request: HybridSearchBatchRequest,
    ) -> Self::Future<'_, Vec<HybridSearchResult>> {
        VectorDbPort::hybrid_search(self, ctx, request)
    }

    fn delete(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        ids: Vec<Box<str>>,
    ) -> Self::Future<'_, ()> {
        VectorDbPort::delete(self, ctx, collection_name, ids)
    }

    fn query(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        filter: Box<str>,
        output_fields: Vec<Box<str>>,
        limit: Option<u32>,
    ) -> Self::Future<'_, Vec<VectorDbRow>> {
        VectorDbPort::query(self, ctx, collection_name, filter, output_fields, limit)
    }
}

/// A row returned from a vector DB query.
pub type VectorDbRow = BTreeMap<Box<str>, Value>;
