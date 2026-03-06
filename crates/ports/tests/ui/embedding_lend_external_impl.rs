use semantic_code_ports::{
    DetectDimensionRequest, EmbedBatchRequest, EmbedRequest, EmbeddingPortLend,
    EmbeddingProviderInfo, EmbeddingVector,
};
use semantic_code_shared::{RequestContext, Result};

struct External;

impl EmbeddingPortLend for External {
    type Future<'a, T>
        = semantic_code_ports::BoxFuture<'a, Result<T>>
    where
        Self: 'a,
        T: 'a;

    fn provider(&self) -> &EmbeddingProviderInfo {
        unreachable!("compile-only test")
    }

    fn detect_dimension(
        &self,
        _ctx: &RequestContext,
        _request: DetectDimensionRequest,
    ) -> Self::Future<'_, u32> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn embed(&self, _ctx: &RequestContext, _request: EmbedRequest) -> Self::Future<'_, EmbeddingVector> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn embed_batch(
        &self,
        _ctx: &RequestContext,
        _request: EmbedBatchRequest,
    ) -> Self::Future<'_, Vec<EmbeddingVector>> {
        Box::pin(async { unreachable!("compile-only test") })
    }
}

fn main() {}
