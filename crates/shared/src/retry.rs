//! Retry helpers with exponential backoff and jitter.

use crate::{ErrorEnvelope, RequestContext, Result};
use std::future::Future;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Retry policy configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Maximum attempts (including the first try).
    pub max_attempts: u32,
    /// Base delay for backoff in milliseconds.
    pub base_delay_ms: u64,
    /// Maximum delay cap in milliseconds.
    pub max_delay_ms: u64,
    /// Jitter ratio as percentage (0..=100).
    pub jitter_ratio_pct: u32,
}

impl RetryPolicy {
    /// Convert jitter ratio to a unit interval (0.0..=1.0).
    #[must_use]
    pub fn jitter_ratio(self) -> f64 {
        f64::from(self.jitter_ratio_pct) / 100.0
    }
}

/// Retry a fallible async operation with backoff + jitter.
pub async fn retry_async<T, F, Fut>(
    ctx: &RequestContext,
    policy: RetryPolicy,
    operation: &'static str,
    mut op: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    retry_async_with_observer(ctx, policy, operation, &mut op, |_, _| {}).await
}

/// Retry with a callback invoked on each retryable failure.
pub async fn retry_async_with_observer<T, F, Fut, Obs>(
    ctx: &RequestContext,
    policy: RetryPolicy,
    operation: &'static str,
    op: &mut F,
    mut on_retry: Obs,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
    Obs: FnMut(u32, &ErrorEnvelope),
{
    let mut attempt = 0u32;

    loop {
        attempt = attempt.saturating_add(1);
        ctx.ensure_not_cancelled(operation)?;

        match op().await {
            Ok(value) => return Ok(value),
            Err(error) => {
                if !error.class.is_retriable() || attempt >= policy.max_attempts {
                    return Err(error);
                }

                on_retry(attempt, &error);
                let delay = backoff_delay(policy, attempt);
                sleep_with_cancellation(ctx, delay, operation).await?;
            },
        }
    }
}

fn backoff_delay(policy: RetryPolicy, attempt: u32) -> Duration {
    let pow = attempt.saturating_sub(1).min(30);
    let base = policy.base_delay_ms.saturating_mul(1u64 << pow);
    let capped = base.min(policy.max_delay_ms);
    let jitter_pct = u64::from(policy.jitter_ratio_pct.min(100));
    if jitter_pct == 0 {
        return Duration::from_millis(capped);
    }
    let jitter_range = (capped.saturating_mul(jitter_pct)) / 100;
    let seed = jitter_seed(attempt);
    let unit = i64::from(u32::try_from(seed % 1000).unwrap_or(0));
    let signed = unit - 500;
    let jitter_range_i64 = i64::try_from(jitter_range).unwrap_or(i64::MAX);
    let capped_i64 = i64::try_from(capped).unwrap_or(i64::MAX);
    let offset = jitter_range_i64.saturating_mul(signed) / 500;
    let max_i64 = i64::try_from(policy.max_delay_ms).unwrap_or(i64::MAX);
    let jittered = capped_i64.saturating_add(offset).clamp(0, max_i64);
    let jittered_u64 = u64::try_from(jittered).unwrap_or(0);
    Duration::from_millis(jittered_u64)
}

fn jitter_seed(attempt: u32) -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| u64::from(duration.subsec_nanos()));
    nanos ^ u64::from(attempt).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

async fn sleep_with_cancellation(
    ctx: &RequestContext,
    delay: Duration,
    operation: &'static str,
) -> Result<()> {
    tokio::select! {
        () = ctx.cancelled() => Err(cancelled_error(operation)),
        () = tokio::time::sleep(delay) => Ok(()),
    }
}

fn cancelled_error(operation: &'static str) -> ErrorEnvelope {
    ErrorEnvelope::cancelled("operation cancelled").with_metadata("operation", operation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ErrorClass, ErrorCode};

    #[tokio::test]
    async fn retry_backoff_obeys_attempts() -> Result<()> {
        let ctx = RequestContext::new_request();
        let policy = RetryPolicy {
            max_attempts: 3,
            base_delay_ms: 1,
            max_delay_ms: 5,
            jitter_ratio_pct: 0,
        };
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let calls_task = calls.clone();

        let result = retry_async(&ctx, policy, "test", || async {
            let attempt = calls_task.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if attempt < 3 {
                Err(ErrorEnvelope::unexpected(
                    ErrorCode::timeout(),
                    "timeout",
                    ErrorClass::Retriable,
                ))
            } else {
                Ok(attempt)
            }
        })
        .await?;

        assert_eq!(result, 3);
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 3);
        Ok(())
    }
}
