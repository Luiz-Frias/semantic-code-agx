use semantic_code_domain::CollectionName;
use semantic_code_ports::{
    HybridSearchBatchRequest, HybridSearchResult, VectorDbPortLend, VectorDbProviderInfo,
    VectorDbRow, VectorDocumentForInsert, VectorSearchRequest, VectorSearchResponse,
};
use semantic_code_shared::{RequestContext, Result};

struct External;

impl VectorDbPortLend for External {
    type Future<'a, T>
        = semantic_code_ports::BoxFuture<'a, Result<T>>
    where
        Self: 'a,
        T: 'a;

    fn provider(&self) -> &VectorDbProviderInfo {
        unreachable!("compile-only test")
    }

    fn create_collection(
        &self,
        _ctx: &RequestContext,
        _collection_name: CollectionName,
        _dimension: u32,
        _description: Option<Box<str>>,
    ) -> Self::Future<'_, ()> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn create_hybrid_collection(
        &self,
        _ctx: &RequestContext,
        _collection_name: CollectionName,
        _dimension: u32,
        _description: Option<Box<str>>,
    ) -> Self::Future<'_, ()> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn drop_collection(
        &self,
        _ctx: &RequestContext,
        _collection_name: CollectionName,
    ) -> Self::Future<'_, ()> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn has_collection(
        &self,
        _ctx: &RequestContext,
        _collection_name: CollectionName,
    ) -> Self::Future<'_, bool> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn list_collections(&self, _ctx: &RequestContext) -> Self::Future<'_, Vec<CollectionName>> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn insert(
        &self,
        _ctx: &RequestContext,
        _collection_name: CollectionName,
        _documents: Vec<VectorDocumentForInsert>,
    ) -> Self::Future<'_, ()> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn insert_hybrid(
        &self,
        _ctx: &RequestContext,
        _collection_name: CollectionName,
        _documents: Vec<VectorDocumentForInsert>,
    ) -> Self::Future<'_, ()> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn flush(
        &self,
        _ctx: &RequestContext,
        _collection_name: CollectionName,
    ) -> Self::Future<'_, ()> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn search(&self, _ctx: &RequestContext, _request: VectorSearchRequest) -> Self::Future<'_, VectorSearchResponse> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn hybrid_search(
        &self,
        _ctx: &RequestContext,
        _request: HybridSearchBatchRequest,
    ) -> Self::Future<'_, Vec<HybridSearchResult>> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn delete(
        &self,
        _ctx: &RequestContext,
        _collection_name: CollectionName,
        _ids: Vec<Box<str>>,
    ) -> Self::Future<'_, ()> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn query(
        &self,
        _ctx: &RequestContext,
        _collection_name: CollectionName,
        _filter: Box<str>,
        _output_fields: Vec<Box<str>>,
        _limit: Option<u32>,
    ) -> Self::Future<'_, Vec<VectorDbRow>> {
        Box::pin(async { unreachable!("compile-only test") })
    }
}

fn main() {}
