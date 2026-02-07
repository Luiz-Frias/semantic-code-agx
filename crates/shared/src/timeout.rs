//! Timeout helpers with cancellation awareness.

use crate::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use std::future::Future;
use std::time::Duration;

/// Apply a timeout to a future, honoring request cancellation.
pub async fn timeout_with_context<T, F>(
    ctx: &RequestContext,
    timeout: Duration,
    operation: &'static str,
    fut: F,
) -> Result<T>
where
    F: Future<Output = Result<T>>,
{
    ctx.ensure_not_cancelled(operation)?;

    tokio::select! {
        () = ctx.cancelled() => Err(cancelled_error(operation)),
        res = tokio::time::timeout(timeout, fut) => {
            res.unwrap_or_else(|_| Err(timeout_error(operation)))
        }
    }
}

fn timeout_error(operation: &'static str) -> ErrorEnvelope {
    ErrorEnvelope::unexpected(
        ErrorCode::timeout(),
        format!("operation timed out: {operation}"),
        ErrorClass::Retriable,
    )
    .with_metadata("operation", operation)
}

fn cancelled_error(operation: &'static str) -> ErrorEnvelope {
    ErrorEnvelope::cancelled("operation cancelled").with_metadata("operation", operation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::Duration;

    #[tokio::test]
    async fn timeout_triggers() {
        let ctx = RequestContext::new_request();
        let fut = async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok::<_, ErrorEnvelope>(())
        };

        let task = tokio::spawn(async move {
            timeout_with_context(&ctx, Duration::from_millis(10), "test", fut).await
        });

        tokio::time::sleep(Duration::from_millis(20)).await;
        let result = task.await.expect("join");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn cancellation_triggers() {
        let ctx = RequestContext::new_request();
        let token = ctx.cancellation_token();
        let fut = async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok::<_, ErrorEnvelope>(())
        };

        let task = tokio::spawn(async move {
            timeout_with_context(&ctx, Duration::from_millis(200), "test_cancel", fut).await
        });

        tokio::task::yield_now().await;
        token.cancel();
        let result = task.await.expect("join");
        assert!(result.is_err());
    }
}
