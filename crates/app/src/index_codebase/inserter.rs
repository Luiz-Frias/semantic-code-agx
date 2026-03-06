//! Vector DB insertion and backpressure handling.

use super::types::{BatchContext, BatchState, EmbeddedBatch, InsertTask};
use semantic_code_domain::IndexMode;
use semantic_code_ports::{TelemetryPort, VectorDbPort};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use std::sync::Arc;
use std::time::Instant;

struct InsertBatchTask {
    request_ctx: RequestContext,
    vectordb: Arc<dyn VectorDbPort>,
    collection_name: semantic_code_domain::CollectionName,
    index_mode: IndexMode,
    telemetry: Option<Arc<dyn TelemetryPort>>,
    documents: Vec<semantic_code_ports::VectorDocumentForInsert>,
    stats: Arc<super::types::IndexStageStatsCollector>,
}

impl InsertBatchTask {
    pub(super) fn new(ctx: &BatchContext<'_>, embedded: EmbeddedBatch) -> Self {
        Self {
            request_ctx: ctx.ctx.clone(),
            vectordb: Arc::clone(&ctx.deps.vectordb),
            collection_name: ctx.input.collection_name.clone(),
            index_mode: ctx.input.index_mode,
            telemetry: ctx.deps.telemetry.clone(),
            documents: embedded.documents,
            stats: Arc::clone(&ctx.stats),
        }
    }

    pub(super) async fn run(self) -> Result<()> {
        let Self {
            request_ctx,
            vectordb,
            collection_name,
            index_mode,
            telemetry,
            documents,
            stats,
        } = self;

        request_ctx.ensure_not_cancelled("index_codebase.insert_batch")?;

        let insert_started = Instant::now();
        let document_count = u64::try_from(documents.len()).unwrap_or(u64::MAX);
        let timer = telemetry
            .as_ref()
            .map(|telemetry| telemetry.start_timer("index.insert_batch", None));

        let result = match index_mode {
            IndexMode::Hybrid => {
                vectordb
                    .insert_hybrid(&request_ctx, collection_name, documents)
                    .await
            },
            IndexMode::Dense => {
                vectordb
                    .insert(&request_ctx, collection_name, documents)
                    .await
            },
        };

        if let Some(timer) = timer.as_ref() {
            timer.stop();
        }

        if result.is_ok() {
            stats.record_insert(document_count, insert_started.elapsed());
        }

        result
    }
}

pub(super) fn schedule_insert_batch<'a>(
    ctx: &BatchContext<'a>,
    state: &mut BatchState<'a>,
    embedded: EmbeddedBatch,
) {
    if embedded.documents.is_empty() {
        return;
    }

    let task_ctx = InsertBatchTask::new(ctx, embedded);
    let promise = ctx
        .insert_pool
        .submit(move || async move { task_ctx.run().await });

    state.insert_tasks.push(InsertTask {
        promise: Some(Box::pin(promise)),
    });
    tracing::debug!(
        queued_insert_tasks = state.insert_tasks.len(),
        next_insert_to_await = state.next_insert_to_await,
        max_pending_insert_batches = ctx.max_pending_insert_batches,
        "insert batch task queued"
    );
}

pub(super) async fn drain_insert_batches_for_backpressure<'a>(
    ctx: &BatchContext<'a>,
    state: &mut BatchState<'a>,
) -> Result<()> {
    while state
        .insert_tasks
        .len()
        .saturating_sub(state.next_insert_to_await)
        >= ctx.max_pending_insert_batches
    {
        tracing::debug!(
            queued_insert_tasks = state.insert_tasks.len(),
            next_insert_to_await = state.next_insert_to_await,
            max_pending_insert_batches = ctx.max_pending_insert_batches,
            "insert backpressure triggered drain"
        );
        drain_one_insert_batch(ctx, state).await?;
    }
    Ok(())
}

pub(super) async fn drain_one_insert_batch<'a>(
    ctx: &BatchContext<'a>,
    state: &mut BatchState<'a>,
) -> Result<()> {
    if state.next_insert_to_await >= state.insert_tasks.len() {
        return Ok(());
    }

    let queued_insert_tasks = state.insert_tasks.len();
    let task_index = state.next_insert_to_await;
    state.next_insert_to_await += 1;
    tracing::debug!(
        queued_insert_tasks = queued_insert_tasks,
        next_insert_to_await = state.next_insert_to_await,
        "draining insert batch"
    );

    let task = state.insert_tasks.get_mut(task_index).ok_or_else(|| {
        ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            "missing insert task",
            ErrorClass::NonRetriable,
        )
    })?;

    let Some(promise) = task.promise.take() else {
        return Ok(());
    };

    if let Err(error) = promise.await {
        if error.is_cancelled() {
            return Err(error);
        }
        if let Some(telemetry) = ctx.deps.telemetry.as_ref() {
            telemetry.increment_counter("index.insert_batch_failed", 1, None);
        }
        if let Some(logger) = ctx.deps.logger.as_ref() {
            logger.error(
                "index.insert_batch_failed",
                "Failed to insert chunk batch; aborting",
                None,
            );
        }
        return Err(with_insert_failure_metadata(
            error,
            task_index,
            queued_insert_tasks,
            state.next_insert_to_await,
        ));
    }

    Ok(())
}

fn with_insert_failure_metadata(
    error: ErrorEnvelope,
    task_index: usize,
    queued_insert_tasks: usize,
    next_insert_to_await: usize,
) -> ErrorEnvelope {
    error
        .with_metadata("stage", "insert")
        .with_metadata("operation", "index_codebase.insert_batch")
        .with_metadata("insertTaskIndex", task_index.to_string())
        .with_metadata("queuedInsertTasks", queued_insert_tasks.to_string())
        .with_metadata("nextInsertToAwait", next_insert_to_await.to_string())
}
