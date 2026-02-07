//! Embedding orchestration for `index_codebase`.

use super::inserter::{drain_insert_batches_for_backpressure, schedule_insert_batch};
use super::types::{BatchContext, BatchState, EmbeddedBatch, PendingChunk};
use semantic_code_domain::{ChunkIdInput, VectorDocumentMetadata, derive_chunk_id};
use semantic_code_ports::{EmbeddingPort, TelemetryPort, VectorDocumentForInsert};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use std::sync::Arc;
use std::time::Instant;

pub struct EmbedBatchTask {
    request_ctx: RequestContext,
    embedding: Arc<dyn EmbeddingPort>,
    telemetry: Option<Arc<dyn TelemetryPort>>,
    batch: Vec<PendingChunk>,
    queued_at: Instant,
    stats: Arc<super::types::IndexStageStatsCollector>,
}

impl EmbedBatchTask {
    pub(crate) fn new(ctx: &BatchContext<'_>, batch: Vec<PendingChunk>) -> Self {
        Self {
            request_ctx: ctx.ctx.clone(),
            embedding: Arc::clone(&ctx.deps.embedding),
            telemetry: ctx.deps.telemetry.clone(),
            batch,
            queued_at: Instant::now(),
            stats: Arc::clone(&ctx.stats),
        }
    }

    pub(crate) async fn run(self) -> Result<EmbeddedBatch> {
        let Self {
            request_ctx,
            embedding,
            telemetry,
            batch,
            queued_at,
            stats,
        } = self;

        request_ctx.ensure_not_cancelled("index_codebase.embed_batch")?;

        if let Some(telemetry) = telemetry.as_ref() {
            let latency_ms = u64::try_from(queued_at.elapsed().as_millis()).unwrap_or(u64::MAX);
            telemetry.record_timer_ms("index.embed_batch.queue_latency_ms", latency_ms, None);
        }

        let timer = telemetry
            .as_ref()
            .map(|telemetry| telemetry.start_timer("index.embed_batch", None));

        let batch_len = u64::try_from(batch.len()).unwrap_or(u64::MAX);
        let texts = batch
            .iter()
            .map(|chunk| chunk.content.clone().into_inner())
            .collect::<Vec<_>>();
        let embed_started = Instant::now();
        let vectors = embedding.embed_batch(&request_ctx, texts.into()).await?;

        if let Some(timer) = timer.as_ref() {
            timer.stop();
        }

        if vectors.len() != batch.len() {
            return Err(ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "embedding batch size mismatch",
                ErrorClass::NonRetriable,
            ));
        }

        stats.record_embed(batch_len, embed_started.elapsed());

        if let Some(telemetry) = telemetry.as_ref() {
            let count = u64::try_from(batch.len()).unwrap_or(u64::MAX);
            telemetry.increment_counter("index.embed_batch.items", count, None);
        }

        let mut documents = Vec::with_capacity(batch.len());
        for (chunk, vector) in batch.into_iter().zip(vectors.into_iter()) {
            let chunk_id = derive_chunk_id(&ChunkIdInput::new(
                chunk.relative_path.clone(),
                chunk.span,
                chunk.content.clone().into_inner(),
            ))
            .map_err(ErrorEnvelope::from)?;

            documents.push(VectorDocumentForInsert {
                id: chunk_id.into_inner(),
                vector: vector.into_vector(),
                content: chunk.content.into_inner(),
                metadata: VectorDocumentMetadata {
                    relative_path: chunk.relative_path,
                    language: Some(chunk.language),
                    file_extension: chunk.file_extension,
                    span: chunk.span,
                    node_kind: None,
                },
            });
        }

        Ok(EmbeddedBatch { documents })
    }
}

pub async fn flush_pending_batches<'a>(
    ctx: &BatchContext<'a>,
    state: &mut BatchState<'a>,
) -> Result<()> {
    ctx.ctx
        .ensure_not_cancelled("index_codebase.flush_pending")?;

    while state.pending.len() >= ctx.embedding_batch_size.get() {
        let batch = state
            .pending
            .drain(0..ctx.embedding_batch_size.get())
            .collect::<Vec<_>>();
        schedule_embedding_batch(ctx, state, batch);
    }

    drain_embedding_batches_for_backpressure(ctx, state).await
}

pub fn schedule_embedding_batch<'a>(
    ctx: &BatchContext<'a>,
    state: &mut BatchState<'a>,
    batch: Vec<PendingChunk>,
) {
    if batch.is_empty() {
        return;
    }

    let task_ctx = EmbedBatchTask::new(ctx, batch);
    let task = ctx
        .embedding_pool
        .submit(move || async move { task_ctx.run().await });

    state.embedding_tasks.push(Box::pin(task));
}

async fn drain_embedding_batches_for_backpressure<'a>(
    ctx: &BatchContext<'a>,
    state: &mut BatchState<'a>,
) -> Result<()> {
    while state
        .embedding_tasks
        .len()
        .saturating_sub(state.next_batch_to_insert)
        >= ctx.max_pending_embedding_batches
    {
        drain_one_embedding_batch(ctx, state).await?;
    }
    Ok(())
}

pub async fn drain_one_embedding_batch<'a>(
    ctx: &BatchContext<'a>,
    state: &mut BatchState<'a>,
) -> Result<()> {
    if state.next_batch_to_insert >= state.embedding_tasks.len() {
        return Ok(());
    }

    let batch_index = state.next_batch_to_insert;
    state.next_batch_to_insert += 1;

    let Some(task) = state.embedding_tasks.get_mut(batch_index) else {
        return Err(ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            "missing embedding task",
            ErrorClass::NonRetriable,
        ));
    };

    let embedded = match task.as_mut().await {
        Ok(embedded) => embedded,
        Err(error) => {
            if error.is_cancelled() {
                return Err(error);
            }
            if let Some(telemetry) = ctx.deps.telemetry.as_ref() {
                telemetry.increment_counter("index.embed_batch_failed", 1, None);
            }
            if let Some(logger) = ctx.deps.logger.as_ref() {
                logger.error(
                    "index.embed_batch_failed",
                    "Failed to embed chunk batch; continuing",
                    None,
                );
            }
            return Ok(());
        },
    };

    schedule_insert_batch(ctx, state, embedded);
    drain_insert_batches_for_backpressure(ctx, state).await?;

    Ok(())
}
