//! Fixed-dimension vector DB wrappers.

use semantic_code_domain::CollectionName;
use semantic_code_ports::{
    BoxFuture, HybridSearchBatchRequest, HybridSearchData, HybridSearchRequest, HybridSearchResult,
    VectorDbPort, VectorDbProviderInfo, VectorDocumentForInsert, VectorSearchRequest,
    VectorSearchResponse,
};
use semantic_code_shared::{ErrorCode, ErrorEnvelope, RequestContext, Result};
use serde_json::Value;
use std::collections::BTreeMap;

/// Wrapper that enforces a compile-time vector dimension.
#[derive(Debug, Clone)]
pub struct FixedDimensionVectorDb<P, const D: usize> {
    inner: P,
}

impl<P, const D: usize> FixedDimensionVectorDb<P, D> {
    /// Wrap a vector DB port with fixed dimension enforcement.
    #[must_use]
    pub const fn new(inner: P) -> Self {
        Self { inner }
    }

    /// Consume the wrapper and return the inner port.
    #[must_use]
    pub fn into_inner(self) -> P {
        self.inner
    }

    fn expected_dimension() -> Result<u32> {
        if D == 0 {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "vector dimension must be greater than zero",
            ));
        }
        u32::try_from(D).map_err(|_| {
            ErrorEnvelope::expected(ErrorCode::invalid_input(), "vector dimension out of range")
        })
    }

    fn ensure_dimension_matches(dimension: u32) -> Result<u32> {
        let expected = Self::expected_dimension()?;
        if dimension != expected {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "vector dimension mismatch",
            )
            .with_metadata("expected", expected.to_string())
            .with_metadata("actual", dimension.to_string()));
        }
        Ok(expected)
    }

    fn ensure_vector_dimension(expected: u32, actual_len: usize) -> Result<()> {
        if actual_len != expected as usize {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "vector dimension mismatch",
            )
            .with_metadata("expected", expected.to_string())
            .with_metadata("actual", actual_len.to_string()));
        }
        Ok(())
    }

    fn ensure_documents(expected: u32, documents: &[VectorDocumentForInsert]) -> Result<()> {
        for doc in documents {
            Self::ensure_vector_dimension(expected, doc.vector.len())?;
        }
        Ok(())
    }

    fn ensure_query_vector(expected: u32, request: &VectorSearchRequest) -> Result<()> {
        Self::ensure_vector_dimension(expected, request.query_vector.len())
    }

    fn ensure_hybrid_vectors(expected: u32, requests: &[HybridSearchRequest]) -> Result<()> {
        for request in requests {
            if let HybridSearchData::DenseVector(vector) = &request.data {
                Self::ensure_vector_dimension(expected, vector.len())?;
            }
        }
        Ok(())
    }
}

impl<P, const D: usize> VectorDbPort for FixedDimensionVectorDb<P, D>
where
    P: VectorDbPort,
{
    fn provider(&self) -> &VectorDbProviderInfo {
        VectorDbPort::provider(&self.inner)
    }

    fn create_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        dimension: u32,
        description: Option<Box<str>>,
    ) -> BoxFuture<'_, Result<()>> {
        let expected = match Self::ensure_dimension_matches(dimension) {
            Ok(value) => value,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        self.inner
            .create_collection(ctx, collection_name, expected, description)
    }

    fn create_hybrid_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        dimension: u32,
        description: Option<Box<str>>,
    ) -> BoxFuture<'_, Result<()>> {
        let expected = match Self::ensure_dimension_matches(dimension) {
            Ok(value) => value,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        self.inner
            .create_hybrid_collection(ctx, collection_name, expected, description)
    }

    fn drop_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
    ) -> BoxFuture<'_, Result<()>> {
        self.inner.drop_collection(ctx, collection_name)
    }

    fn has_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
    ) -> BoxFuture<'_, Result<bool>> {
        self.inner.has_collection(ctx, collection_name)
    }

    fn list_collections(&self, ctx: &RequestContext) -> BoxFuture<'_, Result<Vec<CollectionName>>> {
        self.inner.list_collections(ctx)
    }

    fn insert(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> BoxFuture<'_, Result<()>> {
        let expected = match Self::expected_dimension() {
            Ok(value) => value,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        if let Err(error) = Self::ensure_documents(expected, &documents) {
            return Box::pin(async move { Err(error) });
        }
        self.inner.insert(ctx, collection_name, documents)
    }

    fn insert_hybrid(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> BoxFuture<'_, Result<()>> {
        let expected = match Self::expected_dimension() {
            Ok(value) => value,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        if let Err(error) = Self::ensure_documents(expected, &documents) {
            return Box::pin(async move { Err(error) });
        }
        self.inner.insert_hybrid(ctx, collection_name, documents)
    }

    fn flush(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
    ) -> BoxFuture<'_, Result<()>> {
        self.inner.flush(ctx, collection_name)
    }

    fn search(
        &self,
        ctx: &RequestContext,
        request: VectorSearchRequest,
    ) -> BoxFuture<'_, Result<VectorSearchResponse>> {
        let expected = match Self::expected_dimension() {
            Ok(value) => value,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        if let Err(error) = Self::ensure_query_vector(expected, &request) {
            return Box::pin(async move { Err(error) });
        }
        self.inner.search(ctx, request)
    }

    fn hybrid_search(
        &self,
        ctx: &RequestContext,
        request: HybridSearchBatchRequest,
    ) -> BoxFuture<'_, Result<Vec<HybridSearchResult>>> {
        let expected = match Self::expected_dimension() {
            Ok(value) => value,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        if let Err(error) = Self::ensure_hybrid_vectors(expected, &request.search_requests) {
            return Box::pin(async move { Err(error) });
        }
        self.inner.hybrid_search(ctx, request)
    }

    fn delete(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        ids: Vec<Box<str>>,
    ) -> BoxFuture<'_, Result<()>> {
        self.inner.delete(ctx, collection_name, ids)
    }

    fn query(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        filter: Box<str>,
        output_fields: Vec<Box<str>>,
        limit: Option<u32>,
    ) -> BoxFuture<'_, Result<Vec<BTreeMap<Box<str>, Value>>>> {
        self.inner
            .query(ctx, collection_name, filter, output_fields, limit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_domain::{CollectionName, VectorDbProviderId};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug, Clone)]
    struct FlushSpyVectorDb {
        provider: VectorDbProviderInfo,
        flush_calls: Arc<AtomicUsize>,
    }

    impl FlushSpyVectorDb {
        fn new() -> Result<Self> {
            Ok(Self {
                provider: VectorDbProviderInfo {
                    id: VectorDbProviderId::parse("flush_spy").map_err(ErrorEnvelope::from)?,
                    name: "flush-spy".into(),
                },
                flush_calls: Arc::new(AtomicUsize::new(0)),
            })
        }
    }

    impl VectorDbPort for FlushSpyVectorDb {
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
            Box::pin(async { Ok(()) })
        }

        fn create_hybrid_collection(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _dimension: u32,
            _description: Option<Box<str>>,
        ) -> BoxFuture<'_, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn drop_collection(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
        ) -> BoxFuture<'_, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn has_collection(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
        ) -> BoxFuture<'_, Result<bool>> {
            Box::pin(async { Ok(true) })
        }

        fn list_collections(
            &self,
            _ctx: &RequestContext,
        ) -> BoxFuture<'_, Result<Vec<CollectionName>>> {
            Box::pin(async { Ok(Vec::new()) })
        }

        fn insert(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _documents: Vec<VectorDocumentForInsert>,
        ) -> BoxFuture<'_, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn insert_hybrid(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _documents: Vec<VectorDocumentForInsert>,
        ) -> BoxFuture<'_, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn flush(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
        ) -> BoxFuture<'_, Result<()>> {
            let flush_calls = Arc::clone(&self.flush_calls);
            Box::pin(async move {
                flush_calls.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })
        }

        fn search(
            &self,
            _ctx: &RequestContext,
            _request: VectorSearchRequest,
        ) -> BoxFuture<'_, Result<VectorSearchResponse>> {
            Box::pin(async {
                Ok(VectorSearchResponse {
                    results: Vec::new(),
                    stats: None,
                })
            })
        }

        fn hybrid_search(
            &self,
            _ctx: &RequestContext,
            _request: HybridSearchBatchRequest,
        ) -> BoxFuture<'_, Result<Vec<HybridSearchResult>>> {
            Box::pin(async { Ok(Vec::new()) })
        }

        fn delete(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _ids: Vec<Box<str>>,
        ) -> BoxFuture<'_, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn query(
            &self,
            _ctx: &RequestContext,
            _collection_name: CollectionName,
            _filter: Box<str>,
            _output_fields: Vec<Box<str>>,
            _limit: Option<u32>,
        ) -> BoxFuture<'_, Result<Vec<BTreeMap<Box<str>, Value>>>> {
            Box::pin(async { Ok(Vec::new()) })
        }
    }

    #[tokio::test]
    async fn fixed_dimension_wrapper_forwards_flush() -> Result<()> {
        let inner = FlushSpyVectorDb::new()?;
        let flush_calls = Arc::clone(&inner.flush_calls);
        let wrapper = FixedDimensionVectorDb::<FlushSpyVectorDb, 384>::new(inner);
        let ctx = RequestContext::new_request();
        let collection = CollectionName::parse("flush_forwarding").map_err(ErrorEnvelope::from)?;

        wrapper.flush(&ctx, collection).await?;

        assert_eq!(flush_calls.load(Ordering::Relaxed), 1);
        Ok(())
    }
}
