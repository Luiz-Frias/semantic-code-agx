// Allow missing docs in integration test.
#![allow(missing_docs)]

use semantic_code_adapters::cache::{CachingEmbedding, EmbeddingCache, EmbeddingCacheConfig};
use semantic_code_domain::EmbeddingProviderId;
use semantic_code_ports::EmbeddingPort;
use semantic_code_ports::embedding::{
    DetectDimensionRequest, EmbedBatchRequest, EmbedRequest, EmbeddingProviderInfo, EmbeddingVector,
};
use semantic_code_shared::{
    ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result, RetryPolicy,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

struct FlakyEmbedding {
    provider: EmbeddingProviderInfo,
    attempts: AtomicU32,
}

impl FlakyEmbedding {
    fn new() -> Self {
        Self {
            provider: EmbeddingProviderInfo {
                id: EmbeddingProviderId::parse("test").expect("provider"),
                name: "test".into(),
            },
            attempts: AtomicU32::new(0),
        }
    }
}

impl EmbeddingPort for FlakyEmbedding {
    fn provider(&self) -> &EmbeddingProviderInfo {
        &self.provider
    }

    fn detect_dimension(
        &self,
        _ctx: &RequestContext,
        _request: DetectDimensionRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<u32>> {
        Box::pin(async { Ok(3) })
    }

    fn embed(
        &self,
        _ctx: &RequestContext,
        _request: EmbedRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<EmbeddingVector>> {
        Box::pin(async move {
            let attempt = self.attempts.fetch_add(1, Ordering::Relaxed) + 1;
            if attempt < 3 {
                return Err(ErrorEnvelope::unexpected(
                    ErrorCode::timeout(),
                    "temporary",
                    ErrorClass::Retriable,
                ));
            }
            Ok(EmbeddingVector::from_vec(vec![0.1, 0.2, 0.3]))
        })
    }

    fn embed_batch(
        &self,
        _ctx: &RequestContext,
        request: EmbedBatchRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<EmbeddingVector>>> {
        Box::pin(async move {
            let texts = request.texts;
            Ok(texts
                .into_iter()
                .map(|_| EmbeddingVector::from_vec(vec![0.1, 0.2, 0.3]))
                .collect())
        })
    }
}

#[tokio::test]
async fn resilience_stack_retries_and_caches() -> Result<()> {
    let cache_config = EmbeddingCacheConfig {
        enabled: true,
        max_entries: 16,
        max_bytes: 1024,
        disk_enabled: false,
        disk_provider: semantic_code_adapters::cache::DiskCacheProvider::Sqlite,
        disk_path: None,
        disk_connection: None,
        disk_table: None,
        disk_max_bytes: None,
    };
    let cache = EmbeddingCache::new(&cache_config)?;

    let retry_policy = RetryPolicy {
        max_attempts: 3,
        base_delay_ms: 10,
        max_delay_ms: 50,
        jitter_ratio_pct: 0,
    };

    let inner = Arc::new(FlakyEmbedding::new());
    let wrapped =
        CachingEmbedding::new(inner, cache, "test".into(), retry_policy, 1_000, None, None);

    let ctx = RequestContext::new_request();

    let first = wrapped.embed(&ctx, "hello".into()).await?;
    let second = wrapped.embed(&ctx, "hello".into()).await?;

    assert_eq!(first, second);
    Ok(())
}
