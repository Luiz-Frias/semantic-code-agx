//! Compile-pass check: vector DB lending shim remains available.

use semantic_code_domain::CollectionName;
use semantic_code_ports::{
    BoxFuture, HybridSearchBatchRequest, HybridSearchResult, VectorDbPort, VectorDbPortLend,
    VectorDbProviderInfo, VectorDocumentForInsert, VectorDbRow, VectorSearchRequest,
    VectorSearchResponse,
};
use semantic_code_shared::{RequestContext, Result};

struct Adapter;

impl VectorDbPort for Adapter {
    fn provider(&self) -> &VectorDbProviderInfo {
        unreachable!("compile-only test")
    }

    fn create_collection(
        &self,
        _ctx: &RequestContext,
        _collection_name: CollectionName,
        _dimension: u32,
        _description: Option<Box<str>>,
    ) -> BoxFuture<'_, Result<()>> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn create_hybrid_collection(
        &self,
        _ctx: &RequestContext,
        _collection_name: CollectionName,
        _dimension: u32,
        _description: Option<Box<str>>,
    ) -> BoxFuture<'_, Result<()>> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn drop_collection(
        &self,
        _ctx: &RequestContext,
        _collection_name: CollectionName,
    ) -> BoxFuture<'_, Result<()>> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn has_collection(
        &self,
        _ctx: &RequestContext,
        _collection_name: CollectionName,
    ) -> BoxFuture<'_, Result<bool>> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn list_collections(&self, _ctx: &RequestContext) -> BoxFuture<'_, Result<Vec<CollectionName>>> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn insert(
        &self,
        _ctx: &RequestContext,
        _collection_name: CollectionName,
        _documents: Vec<VectorDocumentForInsert>,
    ) -> BoxFuture<'_, Result<()>> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn insert_hybrid(
        &self,
        _ctx: &RequestContext,
        _collection_name: CollectionName,
        _documents: Vec<VectorDocumentForInsert>,
    ) -> BoxFuture<'_, Result<()>> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn search(
        &self,
        _ctx: &RequestContext,
        _request: VectorSearchRequest,
    ) -> BoxFuture<'_, Result<VectorSearchResponse>> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn hybrid_search(
        &self,
        _ctx: &RequestContext,
        _request: HybridSearchBatchRequest,
    ) -> BoxFuture<'_, Result<Vec<HybridSearchResult>>> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn delete(
        &self,
        _ctx: &RequestContext,
        _collection_name: CollectionName,
        _ids: Vec<Box<str>>,
    ) -> BoxFuture<'_, Result<()>> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn query(
        &self,
        _ctx: &RequestContext,
        _collection_name: CollectionName,
        _filter: Box<str>,
        _output_fields: Vec<Box<str>>,
        _limit: Option<u32>,
    ) -> BoxFuture<'_, Result<Vec<VectorDbRow>>> {
        Box::pin(async { unreachable!("compile-only test") })
    }
}

fn accepts_vector_lending<P: VectorDbPortLend>(_port: &P) {}

fn main() {
    let adapter = Adapter;
    accepts_vector_lending(&adapter);
}
