//! Fixed-dimension embedding wrappers.

use semantic_code_ports::{
    BoxFuture, DetectDimensionRequest, EmbedBatchRequest, EmbedRequest, EmbeddingPort,
    EmbeddingProviderInfo, EmbeddingVector, EmbeddingVectorFixed,
};
use semantic_code_shared::{ErrorCode, ErrorEnvelope, RequestContext, Result};

/// Wrapper that enforces a compile-time embedding dimension.
#[derive(Debug, Clone)]
pub struct FixedDimensionEmbedding<P, const D: usize> {
    inner: P,
}

impl<P, const D: usize> FixedDimensionEmbedding<P, D> {
    /// Wrap an embedding port with fixed dimension enforcement.
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
                "embedding dimension must be greater than zero",
            ));
        }
        u32::try_from(D).map_err(|_| {
            ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "embedding dimension out of range",
            )
        })
    }
}

impl<P, const D: usize> EmbeddingPort for FixedDimensionEmbedding<P, D>
where
    P: EmbeddingPort,
{
    fn provider(&self) -> &EmbeddingProviderInfo {
        EmbeddingPort::provider(&self.inner)
    }

    fn detect_dimension(
        &self,
        ctx: &RequestContext,
        _request: DetectDimensionRequest,
    ) -> BoxFuture<'_, Result<u32>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            ctx.ensure_not_cancelled("embedding_fixed.detect_dimension")?;
            Self::expected_dimension()
        })
    }

    fn embed(
        &self,
        ctx: &RequestContext,
        request: EmbedRequest,
    ) -> BoxFuture<'_, Result<EmbeddingVector>> {
        let ctx = ctx.clone();
        let future = self.inner.embed(&ctx, request);
        Box::pin(async move {
            ctx.ensure_not_cancelled("embedding_fixed.embed")?;
            let vector = future.await?;
            let fixed = EmbeddingVectorFixed::<D>::try_from(vector)?;
            Ok(EmbeddingVector::from(fixed))
        })
    }

    fn embed_batch(
        &self,
        ctx: &RequestContext,
        request: EmbedBatchRequest,
    ) -> BoxFuture<'_, Result<Vec<EmbeddingVector>>> {
        let ctx = ctx.clone();
        let future = self.inner.embed_batch(&ctx, request);
        Box::pin(async move {
            ctx.ensure_not_cancelled("embedding_fixed.embed_batch")?;
            let vectors = future.await?;
            let fixed = vectors
                .into_iter()
                .map(EmbeddingVectorFixed::<D>::try_from)
                .collect::<Result<Vec<_>>>()?;
            Ok(fixed.into_iter().map(EmbeddingVector::from).collect())
        })
    }
}
