//! Compile-pass check: blanket lending bridge remains available.

use semantic_code_ports::{
    BoxFuture, DetectDimensionRequest, EmbedBatchRequest, EmbedRequest, EmbeddingPort,
    EmbeddingPortLend, EmbeddingProviderInfo, EmbeddingVector,
};
use semantic_code_shared::{RequestContext, Result};

struct Adapter;

impl EmbeddingPort for Adapter {
    fn provider(&self) -> &EmbeddingProviderInfo {
        unreachable!("compile-only test")
    }

    fn detect_dimension(
        &self,
        _ctx: &RequestContext,
        _request: DetectDimensionRequest,
    ) -> BoxFuture<'_, Result<u32>> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn embed(&self, _ctx: &RequestContext, _request: EmbedRequest) -> BoxFuture<'_, Result<EmbeddingVector>> {
        Box::pin(async { unreachable!("compile-only test") })
    }

    fn embed_batch(
        &self,
        _ctx: &RequestContext,
        _request: EmbedBatchRequest,
    ) -> BoxFuture<'_, Result<Vec<EmbeddingVector>>> {
        Box::pin(async { unreachable!("compile-only test") })
    }
}

fn accepts_lending<P: EmbeddingPortLend>(_port: &P) {}

fn main() {
    let adapter = Adapter;
    accepts_lending(&adapter);
}
