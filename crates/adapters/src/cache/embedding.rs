use super::{CacheSource, EmbeddingCache};
use semantic_code_ports::embedding::{EmbeddingPort, EmbeddingVector};
use semantic_code_ports::{EmbeddingProviderInfo, TelemetryPort, TelemetryTags};
use semantic_code_shared::{
    ErrorClass, ErrorCode, ErrorEnvelope, Result, RetryPolicy, retry_async_with_observer,
    timeout_with_context,
};
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Embedding port wrapper that adds caching, retries, and timeouts.
pub struct CachingEmbedding {
    inner: Arc<dyn EmbeddingPort>,
    cache: EmbeddingCache,
    cache_namespace: Box<str>,
    retry_policy: RetryPolicy,
    timeout_ms: u64,
    in_flight: Option<Arc<Semaphore>>,
    telemetry: Option<Arc<dyn TelemetryPort>>,
}

impl CachingEmbedding {
    /// Create a new caching wrapper.
    pub fn new(
        inner: Arc<dyn EmbeddingPort>,
        cache: EmbeddingCache,
        cache_namespace: Box<str>,
        retry_policy: RetryPolicy,
        timeout_ms: u64,
        max_in_flight: Option<usize>,
        telemetry: Option<Arc<dyn TelemetryPort>>,
    ) -> Self {
        Self {
            inner,
            cache,
            cache_namespace,
            retry_policy,
            timeout_ms,
            in_flight: max_in_flight.map(|value| Arc::new(Semaphore::new(value.max(1)))),
            telemetry,
        }
    }

    fn provider_info(&self) -> &EmbeddingProviderInfo {
        self.inner.provider()
    }

    fn cache_key(&self, text: &str) -> Box<str> {
        EmbeddingCache::make_key(&self.cache_namespace, text)
    }

    fn cache_tags(&self, source: Option<&str>) -> TelemetryTags {
        let mut tags = BTreeMap::new();
        tags.insert(
            "provider".into(),
            self.provider_info().id.as_str().to_owned().into_boxed_str(),
        );
        if let Some(source) = source {
            tags.insert("source".into(), source.to_owned().into_boxed_str());
        }
        tags
    }

    fn record_cache_hit(&self, source: CacheSource) {
        if let Some(telemetry) = self.telemetry.as_ref() {
            let label = match source {
                CacheSource::Memory => "memory",
                CacheSource::Disk => "disk",
            };
            let tags = self.cache_tags(Some(label));
            telemetry.increment_counter("embedding.cache.hit", 1, Some(&tags));
        }
    }

    fn record_cache_miss(&self) {
        if let Some(telemetry) = self.telemetry.as_ref() {
            let tags = self.cache_tags(None);
            telemetry.increment_counter("embedding.cache.miss", 1, Some(&tags));
        }
    }

    fn record_retry_attempt(&self) {
        if let Some(telemetry) = self.telemetry.as_ref() {
            let tags = self.cache_tags(None);
            telemetry.increment_counter("retry.attempt", 1, Some(&tags));
        }
    }

    fn record_retry_exhausted(&self) {
        if let Some(telemetry) = self.telemetry.as_ref() {
            let tags = self.cache_tags(None);
            telemetry.increment_counter("retry.exhausted", 1, Some(&tags));
        }
    }

    fn record_timeout(&self) {
        if let Some(telemetry) = self.telemetry.as_ref() {
            let tags = self.cache_tags(None);
            telemetry.increment_counter("timeout.triggered", 1, Some(&tags));
        }
    }

    async fn with_in_flight<F, Fut, T>(
        &self,
        ctx: &semantic_code_shared::RequestContext,
        operation: &'static str,
        op: F,
    ) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        if let Some(semaphore) = self.in_flight.as_ref() {
            let _permit = tokio::select! {
                () = ctx.cancelled() => {
                    return Err(ErrorEnvelope::cancelled("operation cancelled")
                        .with_metadata("operation", operation));
                }
                permit = semaphore.clone().acquire_owned() => permit,
            }
            .map_err(|_| {
                ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    "embedding concurrency limiter closed",
                    ErrorClass::NonRetriable,
                )
            })?;
            op().await
        } else {
            op().await
        }
    }

    async fn run_with_resilience<F, Fut, T>(
        &self,
        ctx: &semantic_code_shared::RequestContext,
        operation: &'static str,
        mut op: F,
    ) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        let mut retry_attempts = 0u32;
        let result =
            retry_async_with_observer(ctx, self.retry_policy, operation, &mut op, |_, _| {
                retry_attempts = retry_attempts.saturating_add(1);
                self.record_retry_attempt();
            })
            .await;

        match &result {
            Err(error) if error.code == ErrorCode::timeout() => {
                self.record_timeout();
            },
            Err(_) if retry_attempts > 0 => {
                self.record_retry_exhausted();
            },
            _ => {},
        }

        result
    }
}

impl EmbeddingPort for CachingEmbedding {
    fn provider(&self) -> &EmbeddingProviderInfo {
        self.provider_info()
    }

    fn detect_dimension(
        &self,
        ctx: &semantic_code_shared::RequestContext,
        request: semantic_code_ports::embedding::DetectDimensionRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<u32>> {
        let ctx = semantic_code_shared::RequestContext::with_cancellation(
            ctx.correlation_id().clone(),
            ctx.cancellation_token(),
        );
        Box::pin(async move {
            let ctx_ref = &ctx;
            let timeout = std::time::Duration::from_millis(self.timeout_ms);
            self.run_with_resilience(ctx_ref, "embedding.detect_dimension", || async {
                self.with_in_flight(ctx_ref, "embedding.detect_dimension", || async {
                    timeout_with_context(
                        ctx_ref,
                        timeout,
                        "embedding.detect_dimension",
                        self.inner.detect_dimension(ctx_ref, request.clone()),
                    )
                    .await
                })
                .await
            })
            .await
        })
    }

    fn embed(
        &self,
        ctx: &semantic_code_shared::RequestContext,
        request: semantic_code_ports::embedding::EmbedRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<EmbeddingVector>> {
        let ctx = semantic_code_shared::RequestContext::with_cancellation(
            ctx.correlation_id().clone(),
            ctx.cancellation_token(),
        );
        let text = request.text;
        Box::pin(async move {
            let ctx_ref = &ctx;
            let key = self.cache_key(&text);
            if let Some(hit) = self.cache.get(&key).await? {
                self.record_cache_hit(hit.source);
                return Ok(hit.value);
            }
            self.record_cache_miss();

            let timeout = std::time::Duration::from_millis(self.timeout_ms);
            let text_for_retry = text.clone();
            // Clone is required to allow retries without reallocating per attempt.
            let result = self
                .run_with_resilience(ctx_ref, "embedding.embed", || {
                    let text = text_for_retry.clone();
                    async move {
                        self.with_in_flight(ctx_ref, "embedding.embed", || async {
                            timeout_with_context(
                                ctx_ref,
                                timeout,
                                "embedding.embed",
                                self.inner.embed(ctx_ref, text.into()),
                            )
                            .await
                        })
                        .await
                    }
                })
                .await?;

            self.cache.insert(&key, result.clone()).await?;
            Ok(result)
        })
    }

    fn embed_batch(
        &self,
        ctx: &semantic_code_shared::RequestContext,
        request: semantic_code_ports::embedding::EmbedBatchRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<EmbeddingVector>>> {
        let ctx = semantic_code_shared::RequestContext::with_cancellation(
            ctx.correlation_id().clone(),
            ctx.cancellation_token(),
        );
        let texts = request.texts;
        Box::pin(async move {
            let ctx_ref = &ctx;
            let mut results: Vec<Option<EmbeddingVector>> = vec![None; texts.len()];
            let mut missing = Vec::new();
            let mut missing_indices = Vec::new();

            for (idx, text) in texts.iter().enumerate() {
                let key = self.cache_key(text);
                if let Some(hit) = self.cache.get(&key).await? {
                    self.record_cache_hit(hit.source);
                    if let Some(slot) = results.get_mut(idx) {
                        *slot = Some(hit.value);
                    } else {
                        return Err(ErrorEnvelope::unexpected(
                            ErrorCode::internal(),
                            "embedding cache index out of bounds",
                            ErrorClass::NonRetriable,
                        ));
                    }
                } else {
                    self.record_cache_miss();
                    missing.push(text.clone());
                    missing_indices.push((idx, key));
                }
            }

            if !missing.is_empty() {
                let timeout = std::time::Duration::from_millis(self.timeout_ms);
                let missing_for_retry = missing.clone();
                let batch_result = self
                    .run_with_resilience(ctx_ref, "embedding.embed_batch", || {
                        let batch = missing_for_retry.clone();
                        async move {
                            self.with_in_flight(ctx_ref, "embedding.embed_batch", || async {
                                timeout_with_context(
                                    ctx_ref,
                                    timeout,
                                    "embedding.embed_batch",
                                    self.inner.embed_batch(
                                        ctx_ref,
                                        semantic_code_ports::EmbedBatchRequest { texts: batch },
                                    ),
                                )
                                .await
                            })
                            .await
                        }
                    })
                    .await?;

                if batch_result.len() != missing_indices.len() {
                    return Err(ErrorEnvelope::unexpected(
                        ErrorCode::internal(),
                        "embedding batch result length mismatch",
                        ErrorClass::NonRetriable,
                    ));
                }

                for ((idx, key), vector) in missing_indices.into_iter().zip(batch_result) {
                    self.cache.insert(&key, vector.clone()).await?;
                    if let Some(slot) = results.get_mut(idx) {
                        *slot = Some(vector);
                    } else {
                        return Err(ErrorEnvelope::unexpected(
                            ErrorCode::internal(),
                            "embedding cache index out of bounds",
                            ErrorClass::NonRetriable,
                        ));
                    }
                }
            }

            let mut out = Vec::with_capacity(results.len());
            for item in results {
                out.push(item.ok_or_else(|| {
                    ErrorEnvelope::unexpected(
                        ErrorCode::internal(),
                        "embedding cache missing result",
                        ErrorClass::NonRetriable,
                    )
                })?);
            }
            Ok(out)
        })
    }
}
