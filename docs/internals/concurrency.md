## Concurrency

Phase 04 introduces shared concurrency primitives in `semantic-code-shared`:

- `CancellationToken`
- `RequestContext` (cancellation + correlation id)
- `BoundedQueue<T>` (async queue with backpressure)
- `WorkerPool` (bounded executor built on the bounded queue)

### Cancellation and `RequestContext`

Cancellation is treated as a **boundary concern**:

- Boundary methods accept `&RequestContext`.
- Cancellation is best-effort:
  - queued work is cancelled/dropped
  - in-flight work may complete unless it cooperates

### Backpressure policy

`BoundedQueue<T>` enforces a hard capacity:

- `enqueue` awaits when full
- `dequeue` awaits when empty

This provides explicit backpressure for pipelines like:
scan → split → embed → insert.

### Worker pool semantics

`WorkerPool` provides:

- bounded concurrency (`concurrency`)
- bounded queue capacity (`queue_capacity`, default `concurrency * 2`)
- deterministic ordering for `map` results (input index order)

On cancellation, the pool closes the queue and drops queued tasks so callers
observing results can unblock with a cancellation error.

## Embedding performance knobs

Tune these settings together to balance throughput, memory, and latency:

- `embedding.batchSize`: number of chunks per embedding request.
- `embedding.onnx.sessionPoolSize`: number of ONNX sessions available for parallel inference.
- `core.maxInFlightEmbeddingBatches`: limits concurrent embedding batches (also caps in-flight
  remote embedding requests).
- `core.maxInFlightInserts`: limits concurrent vector DB insert batches.
- `core.maxBufferedChunks` / `core.maxBufferedEmbeddings`: backpressure limits for queued work.
- `core.maxChunkChars`: character cap for each chunk (upstream safety before embeddings).
- `vectorDb.batchSize`: number of documents per insert request.
- `embedding.cache.*`: cache size and disk settings to reduce remote embedding calls.
