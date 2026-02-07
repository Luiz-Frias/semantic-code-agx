//! Embedding routing helpers for hybrid/local selection.

use semantic_code_domain::EmbeddingProviderId;
use semantic_code_ports::{
    BoxFuture, DetectDimensionRequest, EmbedBatchRequest, EmbedRequest, EmbeddingPort,
    EmbeddingProviderInfo, EmbeddingVector,
};
use semantic_code_shared::{ErrorCode, ErrorEnvelope, Result};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// Routing wrapper that splits embedding batches between remote and local providers.
pub struct SplitEmbeddingRouter {
    local: Arc<dyn EmbeddingPort>,
    remote: Arc<dyn EmbeddingPort>,
    remaining_remote_batches: AtomicU32,
    provider: EmbeddingProviderInfo,
}

impl SplitEmbeddingRouter {
    /// Create a new split router.
    pub fn new(
        local: Arc<dyn EmbeddingPort>,
        remote: Arc<dyn EmbeddingPort>,
        max_remote_batches: u32,
    ) -> Result<Self> {
        if max_remote_batches == 0 {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "split routing requires at least one remote batch",
            ));
        }
        let provider = EmbeddingProviderInfo {
            id: EmbeddingProviderId::parse("hybrid").map_err(ErrorEnvelope::from)?,
            name: "Hybrid (local + remote)".to_string().into_boxed_str(),
        };
        Ok(Self {
            local,
            remote,
            remaining_remote_batches: AtomicU32::new(max_remote_batches),
            provider,
        })
    }

    fn take_remote_batch(&self) -> bool {
        self.remaining_remote_batches
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_sub(1)
            })
            .is_ok()
    }

    async fn fallback_embed_batch(
        local: &Arc<dyn EmbeddingPort>,
        remote: &Arc<dyn EmbeddingPort>,
        ctx: &semantic_code_shared::RequestContext,
        request: EmbedBatchRequest,
        use_remote: bool,
    ) -> Result<Vec<EmbeddingVector>> {
        if use_remote {
            let primary = remote.embed_batch(ctx, request.clone()).await;
            return match primary {
                Ok(result) => Ok(result),
                Err(primary_error) => remote_fallback(local, ctx, request, primary_error).await,
            };
        }

        let primary = local.embed_batch(ctx, request.clone()).await;
        match primary {
            Ok(result) => Ok(result),
            Err(primary_error) => local_fallback(remote, ctx, request, primary_error).await,
        }
    }
}

async fn local_fallback(
    remote: &Arc<dyn EmbeddingPort>,
    ctx: &semantic_code_shared::RequestContext,
    request: EmbedBatchRequest,
    primary_error: ErrorEnvelope,
) -> Result<Vec<EmbeddingVector>> {
    remote
        .embed_batch(ctx, request)
        .await
        .map_or(Err(primary_error), Ok)
}

async fn remote_fallback(
    local: &Arc<dyn EmbeddingPort>,
    ctx: &semantic_code_shared::RequestContext,
    request: EmbedBatchRequest,
    primary_error: ErrorEnvelope,
) -> Result<Vec<EmbeddingVector>> {
    local
        .embed_batch(ctx, request)
        .await
        .map_or(Err(primary_error), Ok)
}

impl EmbeddingPort for SplitEmbeddingRouter {
    fn provider(&self) -> &EmbeddingProviderInfo {
        &self.provider
    }

    fn detect_dimension(
        &self,
        ctx: &semantic_code_shared::RequestContext,
        request: DetectDimensionRequest,
    ) -> BoxFuture<'_, Result<u32>> {
        let local = Arc::clone(&self.local);
        let remote = Arc::clone(&self.remote);
        // Clone to own the context across the boxed future boundary.
        let ctx = ctx.clone();
        Box::pin(async move {
            match local.detect_dimension(&ctx, request.clone()).await {
                Ok(result) => Ok(result),
                Err(primary_error) => remote
                    .detect_dimension(&ctx, request)
                    .await
                    .map_or(Err(primary_error), Ok),
            }
        })
    }

    fn embed(
        &self,
        ctx: &semantic_code_shared::RequestContext,
        request: EmbedRequest,
    ) -> BoxFuture<'_, Result<EmbeddingVector>> {
        let local = Arc::clone(&self.local);
        let remote = Arc::clone(&self.remote);
        // Clone to own the context across the boxed future boundary.
        let ctx = ctx.clone();
        Box::pin(async move {
            match local.embed(&ctx, request.clone()).await {
                Ok(result) => Ok(result),
                Err(primary_error) => remote
                    .embed(&ctx, request)
                    .await
                    .map_or(Err(primary_error), Ok),
            }
        })
    }

    fn embed_batch(
        &self,
        ctx: &semantic_code_shared::RequestContext,
        request: EmbedBatchRequest,
    ) -> BoxFuture<'_, Result<Vec<EmbeddingVector>>> {
        let local = Arc::clone(&self.local);
        let remote = Arc::clone(&self.remote);
        let use_remote = self.take_remote_batch();
        // Clone to own the context across the boxed future boundary.
        let ctx = ctx.clone();
        Box::pin(async move {
            Self::fallback_embed_batch(&local, &remote, &ctx, request, use_remote).await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_ports::{EmbedBatchRequest, EmbeddingProviderInfo};
    use semantic_code_shared::RequestContext;
    use std::sync::Arc;

    struct TestEmbedding {
        provider: EmbeddingProviderInfo,
        id: &'static str,
    }

    impl TestEmbedding {
        fn new(id: &'static str) -> Self {
            Self {
                provider: EmbeddingProviderInfo {
                    id: EmbeddingProviderId::parse(id).expect("provider id"),
                    name: id.to_string().into_boxed_str(),
                },
                id,
            }
        }
    }

    impl EmbeddingPort for TestEmbedding {
        fn provider(&self) -> &EmbeddingProviderInfo {
            &self.provider
        }

        fn detect_dimension(
            &self,
            _ctx: &RequestContext,
            _request: DetectDimensionRequest,
        ) -> BoxFuture<'_, Result<u32>> {
            Box::pin(async move { Ok(8) })
        }

        fn embed(
            &self,
            _ctx: &RequestContext,
            _request: EmbedRequest,
        ) -> BoxFuture<'_, Result<EmbeddingVector>> {
            let id = self.id;
            Box::pin(async move { Ok(EmbeddingVector::from_vec(vec![id.len() as f32])) })
        }

        fn embed_batch(
            &self,
            _ctx: &RequestContext,
            request: EmbedBatchRequest,
        ) -> BoxFuture<'_, Result<Vec<EmbeddingVector>>> {
            let id = self.id;
            let count = request.texts.len();
            Box::pin(async move {
                Ok((0..count)
                    .map(|_| EmbeddingVector::from_vec(vec![id.len() as f32]))
                    .collect())
            })
        }
    }

    #[tokio::test]
    async fn split_router_consumes_remote_budget() -> Result<()> {
        let local = Arc::new(TestEmbedding::new("local"));
        let remote = Arc::new(TestEmbedding::new("remote"));
        let router = SplitEmbeddingRouter::new(local, remote, 1)?;
        let ctx = RequestContext::new_request();
        let request = EmbedBatchRequest::from(vec!["hello".to_string()]);

        let first = router.embed_batch(&ctx, request.clone()).await?;
        let second = router.embed_batch(&ctx, request).await?;

        assert_eq!(first[0].as_slice()[0], "remote".len() as f32);
        assert_eq!(second[0].as_slice()[0], "local".len() as f32);
        Ok(())
    }
}
