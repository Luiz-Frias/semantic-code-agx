//! Concurrency primitives and request-scoped context.
//!
//! Phase 04 introduces:
//! - Cancellation + correlation identifiers via `RequestContext`
//! - A bounded async queue (`BoundedQueue`) with backpressure
//! - A small worker pool executor (`WorkerPool`) built on top of the queue
//!
//! Notes:
//! - These primitives are intended for I/O-heavy orchestration (index/search pipelines),
//!   not CPU-bound workloads.
//! - Cancellation is "best-effort": work that has not started is cancelled; in-flight
//!   work may complete unless the task itself cooperates.

use crate::{ErrorCode, ErrorEnvelope, Result};
use std::collections::VecDeque;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tokio::sync::{Mutex, Notify, oneshot};

/// A correlation identifier used for logging/telemetry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CorrelationId(Arc<str>);

impl CorrelationId {
    /// Parse a correlation identifier from user input.
    ///
    /// The value is trimmed; empty values are rejected.
    pub fn parse(value: impl AsRef<str>) -> Result<Self> {
        let trimmed = value.as_ref().trim();
        if trimmed.is_empty() {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "correlationId must be non-empty",
            ));
        }
        Ok(Self(Arc::<str>::from(trimmed)))
    }

    /// Create a new request id, best-effort unique within this process.
    #[must_use]
    pub fn new_request_id() -> Self {
        next_scoped_id(&REQUEST_ID_COUNTER, "req_")
    }

    /// Create a new job id, best-effort unique within this process.
    #[must_use]
    pub fn new_job_id() -> Self {
        next_scoped_id(&JOB_ID_COUNTER, "job_")
    }

    /// Borrow the identifier as a string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CorrelationId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

static REQUEST_ID_COUNTER: AtomicU64 = AtomicU64::new(1);
static JOB_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_scoped_id(counter: &AtomicU64, prefix: &'static str) -> CorrelationId {
    let n = counter.fetch_add(1, Ordering::Relaxed);
    let id: Box<str> = format!("{prefix}{n}").into_boxed_str();
    CorrelationId(Arc::<str>::from(id))
}

/// A clonable cancellation token that can be awaited.
#[derive(Debug, Clone)]
pub struct CancellationToken {
    inner: Arc<CancellationState>,
}

#[derive(Debug)]
struct CancellationState {
    cancelled: AtomicBool,
    notify: Notify,
}

impl CancellationToken {
    /// Create a new token in the non-cancelled state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(CancellationState {
                cancelled: AtomicBool::new(false),
                notify: Notify::new(),
            }),
        }
    }

    /// Cancel the token and wake all current/future waiters.
    pub fn cancel(&self) {
        let was_cancelled = self.inner.cancelled.swap(true, Ordering::SeqCst);
        if !was_cancelled {
            self.inner.notify.notify_waiters();
        }
    }

    /// Returns true if the token has been cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::SeqCst)
    }

    /// Wait until the token is cancelled.
    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }

        loop {
            let notified = self.inner.notify.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
            if self.is_cancelled() {
                return;
            }
        }
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

/// Request-scoped context passed across boundaries.
#[derive(Debug, Clone)]
pub struct RequestContext {
    correlation_id: CorrelationId,
    cancellation: CancellationToken,
}

impl RequestContext {
    /// Create a new request context with a fresh cancellation token.
    #[must_use]
    pub fn new(correlation_id: CorrelationId) -> Self {
        Self {
            correlation_id,
            cancellation: CancellationToken::new(),
        }
    }

    /// Convenience constructor: create a context with an auto-generated `req_*` id.
    #[must_use]
    pub fn new_request() -> Self {
        Self::new(CorrelationId::new_request_id())
    }

    /// Create a context with an explicit cancellation token (for sharing cancellation).
    #[must_use]
    pub const fn with_cancellation(
        correlation_id: CorrelationId,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            correlation_id,
            cancellation,
        }
    }

    /// Return the correlation id.
    #[must_use]
    pub const fn correlation_id(&self) -> &CorrelationId {
        &self.correlation_id
    }

    /// Return a clone of the cancellation token.
    #[must_use]
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    /// Returns true if the request was cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    /// Cancel this request.
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    /// Await cancellation.
    pub async fn cancelled(&self) {
        self.cancellation.cancelled().await;
    }

    /// Return a cancellation error when cancelled, including operation metadata.
    pub fn ensure_not_cancelled(&self, operation: &'static str) -> Result<()> {
        if self.is_cancelled() {
            return Err(ErrorEnvelope::cancelled("operation cancelled")
                .with_metadata("operation", operation));
        }
        Ok(())
    }
}

/// Error returned when a `BoundedQueue` is closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundedQueueClosedError;

impl fmt::Display for BoundedQueueClosedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("BoundedQueue is closed")
    }
}

impl std::error::Error for BoundedQueueClosedError {}

impl From<BoundedQueueClosedError> for ErrorEnvelope {
    fn from(_: BoundedQueueClosedError) -> Self {
        Self::expected(
            ErrorCode::new("core", "bounded_queue_closed"),
            "BoundedQueue is closed",
        )
    }
}

/// A bounded async queue with explicit backpressure.
///
/// - `enqueue` waits when the queue is full
/// - `dequeue` waits when the queue is empty
/// - both are cancellation-aware via `RequestContext`
#[derive(Debug)]
pub struct BoundedQueue<T> {
    capacity: usize,
    state: Arc<Mutex<QueueState<T>>>,
}

impl<T> Clone for BoundedQueue<T> {
    fn clone(&self) -> Self {
        Self {
            capacity: self.capacity,
            state: Arc::clone(&self.state),
        }
    }
}

#[derive(Debug)]
struct QueueState<T> {
    items: VecDeque<T>,
    waiting_consumers: VecDeque<oneshot::Sender<T>>,
    waiting_producers: VecDeque<oneshot::Sender<()>>,
    closed: bool,
}

impl<T> BoundedQueue<T> {
    /// Create a new bounded queue.
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "capacity must be a positive number",
            ));
        }

        Ok(Self {
            capacity,
            state: Arc::new(Mutex::new(QueueState {
                items: VecDeque::new(),
                waiting_consumers: VecDeque::new(),
                waiting_producers: VecDeque::new(),
                closed: false,
            })),
        })
    }

    /// Return the configured capacity.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Return the current queue length.
    pub async fn len(&self) -> usize {
        let state = self.state.lock().await;
        state.items.len()
    }

    /// Return true when the queue is empty.
    pub async fn is_empty(&self) -> bool {
        let state = self.state.lock().await;
        state.items.is_empty()
    }

    /// Close the queue (rejecting any waiting producers/consumers).
    ///
    /// Items already in the queue are retained and may still be dequeued.
    pub async fn close(&self) {
        let mut state = self.state.lock().await;
        if state.closed {
            return;
        }

        state.closed = true;
        state.waiting_consumers.clear();
        state.waiting_producers.clear();
    }

    /// Close the queue and drop all queued items.
    ///
    /// This is intended for cancellation paths where queued work should not execute.
    pub async fn close_and_clear(&self) {
        let mut state = self.state.lock().await;
        if state.closed {
            state.items.clear();
            return;
        }

        state.closed = true;
        state.items.clear();
        state.waiting_consumers.clear();
        state.waiting_producers.clear();
    }

    /// Enqueue an item, waiting for capacity when the queue is full.
    pub async fn enqueue(&self, ctx: &RequestContext, mut item: T) -> Result<()> {
        ctx.ensure_not_cancelled("queue.enqueue")?;

        loop {
            let producer_gate = {
                let mut state = self.state.lock().await;
                if state.closed {
                    return Err(ErrorEnvelope::from(BoundedQueueClosedError));
                }

                // If a consumer is waiting, satisfy it immediately.
                while let Some(consumer) = state.waiting_consumers.pop_front() {
                    match consumer.send(item) {
                        Ok(()) => return Ok(()),
                        Err(returned) => item = returned,
                    }
                }

                if state.items.len() < self.capacity {
                    state.items.push_back(item);
                    return Ok(());
                }

                let (tx, rx) = oneshot::channel::<()>();
                state.waiting_producers.push_back(tx);
                rx
            };

            tokio::select! {
                () = ctx.cancelled() => {
                    return Err(ErrorEnvelope::cancelled("operation cancelled").with_metadata("operation", "queue.enqueue"));
                }
                res = producer_gate => {
                    if res.is_err() {
                        // Sender dropped due to close.
                        return Err(ErrorEnvelope::from(BoundedQueueClosedError));
                    }
                }
            }
        }
    }

    /// Dequeue an item, waiting for an item when the queue is empty.
    pub async fn dequeue(&self, ctx: &RequestContext) -> Result<T> {
        ctx.ensure_not_cancelled("queue.dequeue")?;

        let consumer_wait = {
            let mut state = self.state.lock().await;

            if let Some(item) = state.items.pop_front() {
                // Allow a blocked producer through when we just freed capacity.
                while let Some(producer) = state.waiting_producers.pop_front() {
                    if producer.send(()).is_ok() {
                        break;
                    }
                }
                return Ok(item);
            }

            if state.closed {
                return Err(ErrorEnvelope::from(BoundedQueueClosedError));
            }

            let (tx, rx) = oneshot::channel::<T>();
            state.waiting_consumers.push_back(tx);
            rx
        };

        tokio::select! {
            () = ctx.cancelled() => {
                Err(ErrorEnvelope::cancelled("operation cancelled").with_metadata("operation", "queue.dequeue"))
            }
            res = consumer_wait => {
                res.map_or_else(|_| Err(ErrorEnvelope::from(BoundedQueueClosedError)), Ok)
            }
        }
    }
}

type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send + 'static>>;
type Task = Box<dyn FnOnce() -> BoxFuture<()> + Send + 'static>;

/// Options for the worker pool.
#[derive(Debug, Clone, Copy)]
pub struct WorkerPoolOptions {
    /// Number of worker tasks (bounded concurrency).
    pub concurrency: usize,
    /// Backpressure: maximum queued tasks waiting for workers.
    ///
    /// Default: `concurrency * 2` (minimum 1).
    pub queue_capacity: Option<usize>,
}

/// A bounded worker pool executor.
///
/// - bounded concurrency
/// - bounded queue (backpressure)
/// - best-effort cancellation: queued tasks are dropped when cancelled
/// - deterministic result ordering for `map` (input index order)
pub struct WorkerPool {
    ctx: RequestContext,
    queue: BoundedQueue<Task>,
    workers: Vec<tokio::task::JoinHandle<()>>,
    cancel_watcher: tokio::task::JoinHandle<()>,
}

impl WorkerPool {
    /// Create a new worker pool bound to the provided `RequestContext`.
    pub fn new(ctx: RequestContext, options: WorkerPoolOptions) -> Result<Self> {
        if options.concurrency == 0 {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "concurrency must be a positive number",
            ));
        }

        let capacity = options
            .queue_capacity
            .unwrap_or_else(|| options.concurrency.saturating_mul(2))
            .max(1);

        let queue = BoundedQueue::new(capacity)?;

        let token = ctx.cancellation_token();
        let queue_for_cancel = queue.clone();
        let cancel_watcher = tokio::spawn(async move {
            token.cancelled().await;
            queue_for_cancel.close_and_clear().await;
        });

        let mut workers = Vec::with_capacity(options.concurrency);
        for _ in 0..options.concurrency {
            let queue = queue.clone();
            let ctx = ctx.clone();
            workers.push(tokio::spawn(async move {
                worker_loop(queue, ctx).await;
            }));
        }

        Ok(Self {
            ctx,
            queue,
            workers,
            cancel_watcher,
        })
    }

    /// Stop the pool by cancelling queued work and closing the queue.
    pub async fn stop(&self) {
        self.queue.close_and_clear().await;
    }

    /// Stop the pool and await worker termination.
    pub async fn shutdown(mut self) -> Result<()> {
        self.queue.close_and_clear().await;
        self.cancel_watcher.abort();
        match self.cancel_watcher.await {
            Ok(()) => {},
            Err(join_error) if join_error.is_cancelled() => {
                // Expected after abort; no action needed.
            },
            Err(join_error) if join_error.is_panic() => {
                eprintln!("worker_pool cancel watcher panicked: {join_error:?}");
                return Err(ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    format!("worker_pool cancel watcher panicked: {join_error}"),
                    crate::ErrorClass::NonRetriable,
                ));
            },
            Err(join_error) => {
                return Err(ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    format!("worker_pool cancel watcher failed: {join_error}"),
                    crate::ErrorClass::NonRetriable,
                ));
            },
        }

        for handle in self.workers.drain(..) {
            if let Err(join_error) = handle.await {
                return Err(ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    join_error.to_string(),
                    crate::ErrorClass::NonRetriable,
                ));
            }
        }

        Ok(())
    }

    /// Submit a task to the pool and await its result.
    pub async fn submit<T, Fut, F>(&self, task: F) -> Result<T>
    where
        T: Send + 'static,
        Fut: Future<Output = Result<T>> + Send + 'static,
        F: FnOnce() -> Fut + Send + 'static,
    {
        self.ctx.ensure_not_cancelled("worker_pool.submit")?;

        let (tx, rx) = oneshot::channel::<Result<T>>();
        let ctx = self.ctx.clone();
        let wrapped: Task = Box::new(move || {
            Box::pin(async move {
                // Best-effort cancellation: skip queued tasks once cancelled.
                if ctx.is_cancelled() {
                    if tx
                        .send(Err(ErrorEnvelope::cancelled("operation cancelled")))
                        .is_err()
                    {
                        // Receiver dropped; nothing to do.
                    }
                    return;
                }

                let result = task().await;
                if tx.send(result).is_err() {
                    // Receiver dropped; nothing to do.
                }
            })
        });

        if let Err(error) = self.queue.enqueue(&self.ctx, wrapped).await {
            if self.ctx.is_cancelled() {
                return Err(ErrorEnvelope::cancelled("operation cancelled")
                    .with_metadata("operation", "worker_pool.submit.enqueue"));
            }
            return Err(error);
        }

        rx.await.unwrap_or_else(|_| {
            Err(ErrorEnvelope::cancelled("operation cancelled")
                .with_metadata("operation", "worker_pool.submit.await"))
        })
    }

    /// Apply an async function over inputs with deterministic ordering.
    pub async fn map<TIn, TOut, Fut, F>(&self, inputs: Vec<TIn>, f: F) -> Result<Vec<TOut>>
    where
        TIn: Send + 'static,
        TOut: Send + 'static,
        Fut: Future<Output = Result<TOut>> + Send + 'static,
        F: Fn(TIn, usize) -> Fut + Send + Sync + 'static,
    {
        self.ctx.ensure_not_cancelled("worker_pool.map")?;

        let count = inputs.len();
        let mut receivers = Vec::with_capacity(count);
        let f = Arc::new(f);

        for (index, input) in inputs.into_iter().enumerate() {
            let (tx, rx) = oneshot::channel::<Result<TOut>>();
            let ctx = self.ctx.clone();
            let f = Arc::clone(&f);
            let task: Task = Box::new(move || {
                Box::pin(async move {
                    if ctx.is_cancelled() {
                        if tx
                            .send(Err(ErrorEnvelope::cancelled("operation cancelled")))
                            .is_err()
                        {
                            // Receiver dropped; nothing to do.
                        }
                        return;
                    }
                    let out = f(input, index).await;
                    if tx.send(out).is_err() {
                        // Receiver dropped; nothing to do.
                    }
                })
            });

            if let Err(error) = self.queue.enqueue(&self.ctx, task).await {
                if self.ctx.is_cancelled() {
                    return Err(ErrorEnvelope::cancelled("operation cancelled")
                        .with_metadata("operation", "worker_pool.map.enqueue"));
                }
                return Err(error);
            }
            receivers.push((index, rx));
        }

        let mut results: Vec<Option<TOut>> = (0..count).map(|_| None).collect();
        for (index, rx) in receivers {
            let value = tokio::select! {
                () = self.ctx.cancelled() => {
                    return Err(ErrorEnvelope::cancelled("operation cancelled").with_metadata("operation", "worker_pool.map.await"));
                }
                res = rx => res
            };

            let value = match value {
                Ok(value) => value?,
                Err(_) => {
                    return Err(ErrorEnvelope::cancelled("operation cancelled")
                        .with_metadata("operation", "worker_pool.map.await"));
                },
            };

            if let Some(slot) = results.get_mut(index) {
                *slot = Some(value);
            } else {
                return Err(ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    "worker_pool.map index out of bounds",
                    crate::ErrorClass::NonRetriable,
                ));
            }
        }

        let mut out = Vec::with_capacity(count);
        for value in results {
            match value {
                Some(value) => out.push(value),
                None => {
                    return Err(ErrorEnvelope::unexpected(
                        ErrorCode::internal(),
                        "worker_pool.map missing result",
                        crate::ErrorClass::NonRetriable,
                    ));
                },
            }
        }
        Ok(out)
    }
}

async fn worker_loop(queue: BoundedQueue<Task>, ctx: RequestContext) {
    loop {
        if ctx.is_cancelled() {
            return;
        }

        let task = match queue.dequeue(&ctx).await {
            Ok(task) => task,
            Err(error) => {
                // Closed queue is expected on stop/cancel.
                if error.code == ErrorCode::new("core", "bounded_queue_closed") {
                    return;
                }
                return;
            },
        };

        // Execute task.
        task().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn bounded_queue_applies_backpressure() -> Result<()> {
        let ctx = RequestContext::new_request();
        let queue = BoundedQueue::new(1)?;

        queue.enqueue(&ctx, 1u32).await?;

        let q2 = queue.clone();
        let ctx2 = ctx.clone();
        let mut blocked = tokio::spawn(async move { q2.enqueue(&ctx2, 2u32).await });

        // The second enqueue should block while the queue is full.
        let timed = tokio::time::timeout(Duration::from_millis(50), &mut blocked).await;
        assert!(timed.is_err(), "enqueue should be backpressured");

        let first = queue.dequeue(&ctx).await?;
        assert_eq!(first, 1);

        // Now that capacity is available, the blocked enqueue should complete.
        blocked.await.expect("join failed")?;

        let second = queue.dequeue(&ctx).await?;
        assert_eq!(second, 2);

        Ok(())
    }

    #[tokio::test]
    async fn worker_pool_map_is_deterministic() -> Result<()> {
        let ctx = RequestContext::new_request();
        let pool = WorkerPool::new(
            ctx,
            WorkerPoolOptions {
                concurrency: 2,
                queue_capacity: Some(4),
            },
        )?;

        let inputs = vec![1u32, 2u32, 3u32, 4u32];
        let out = pool
            .map(inputs, |value, index| async move {
                // Force out-of-order completion.
                let delay_ms = (4 - index) as u64 * 10;
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                Ok(value * 2)
            })
            .await?;

        assert_eq!(out, vec![2, 4, 6, 8]);
        Ok(())
    }

    #[tokio::test]
    async fn worker_pool_cancels_queued_work() -> Result<()> {
        let ctx = RequestContext::new_request();
        let pool = WorkerPool::new(
            ctx.clone(),
            WorkerPoolOptions {
                concurrency: 1,
                queue_capacity: Some(1),
            },
        )?;

        // First task blocks, occupying the only worker.
        let (gate_tx, gate_rx) = oneshot::channel::<()>();
        let first = pool.submit(move || async move {
            let _ = gate_rx.await;
            Ok::<_, ErrorEnvelope>(())
        });

        // Second task is queued and should be cancelled once we cancel the context.
        let second = pool.submit(|| async { Ok::<_, ErrorEnvelope>(123u32) });

        let cancel_and_release = async move {
            // Ensure the second task is enqueued (best-effort).
            tokio::time::sleep(Duration::from_millis(20)).await;
            ctx.cancel();
            let _ = gate_tx.send(());
        };

        let (first_result, second_result, ()) = tokio::join!(first, second, cancel_and_release);

        assert!(first_result.is_ok(), "in-flight task should complete");
        assert!(
            matches!(second_result, Err(ref e) if e.is_cancelled()),
            "queued task should be cancelled"
        );

        Ok(())
    }
}
