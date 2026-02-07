# Environment Variables

Phase 03 Milestone 2 introduces **env parsing + deterministic normalization**
for configuration overrides.

Env values are parsed in `semantic-code-config` and can be merged into a
`BackendConfig` (defaults/file) with **env taking precedence**.

## Supported variables

### Core

- `SCA_CORE_TIMEOUT_MS` (u64): overrides `core.timeoutMs`
- `SCA_CORE_MAX_CONCURRENCY` (u32): overrides `core.maxConcurrency`
- `SCA_CORE_MAX_IN_FLIGHT_FILES` (u32): overrides `core.maxInFlightFiles`
- `SCA_CORE_MAX_IN_FLIGHT_EMBEDDING_BATCHES` (u32): overrides `core.maxInFlightEmbeddingBatches`
- `SCA_CORE_MAX_IN_FLIGHT_INSERTS` (u32): overrides `core.maxInFlightInserts`
- `SCA_CORE_MAX_BUFFERED_CHUNKS` (u32): overrides `core.maxBufferedChunks`
- `SCA_CORE_MAX_BUFFERED_EMBEDDINGS` (u32): overrides `core.maxBufferedEmbeddings`
- `SCA_CORE_MAX_CHUNK_CHARS` (u32): overrides `core.maxChunkChars`
- `SCA_CORE_RETRY_MAX_ATTEMPTS` (u32): overrides `core.retry.maxAttempts`
- `SCA_CORE_RETRY_BASE_DELAY_MS` (u64): overrides `core.retry.baseDelayMs`
- `SCA_CORE_RETRY_MAX_DELAY_MS` (u64): overrides `core.retry.maxDelayMs`
- `SCA_CORE_RETRY_JITTER_RATIO_PCT` (u32): overrides `core.retry.jitterRatioPct`

### Embedding

- `SCA_EMBEDDING_PROVIDER` (string): overrides `embedding.provider` (trimmed)
- `SCA_EMBEDDING_MODEL` (string): overrides `embedding.model` (trimmed)
- `SCA_EMBEDDING_TIMEOUT_MS` (u64): overrides `embedding.timeoutMs`
- `SCA_EMBEDDING_BATCH_SIZE` (u32): overrides `embedding.batchSize`
- `SCA_EMBEDDING_DIMENSION` (u32): overrides `embedding.dimension`
- `SCA_EMBEDDING_BASE_URL` (string URL): overrides `embedding.baseUrl` (`http`/`https`)
- `SCA_EMBEDDING_LOCAL_FIRST` (bool): overrides `embedding.localFirst`
- `SCA_EMBEDDING_LOCAL_ONLY` (bool): overrides `embedding.localOnly`
- `SCA_EMBEDDING_ROUTING_MODE` (string): overrides `embedding.routing.mode`
- `SCA_EMBEDDING_SPLIT_MAX_REMOTE_BATCHES` (u32): overrides `embedding.routing.split.maxRemoteBatches`
- `SCA_EMBEDDING_JOBS_PROGRESS_INTERVAL_MS` (u64): overrides `embedding.jobs.progressIntervalMs`
- `SCA_EMBEDDING_JOBS_CANCEL_POLL_INTERVAL_MS` (u64): overrides `embedding.jobs.cancelPollIntervalMs`
- `SCA_EMBEDDING_TEST_FALLBACK` (bool): allow fallback to test embeddings when ONNX assets are missing
- `SCA_EMBEDDING_ONNX_MODEL_DIR` (string): overrides `embedding.onnx.modelDir`
- `SCA_EMBEDDING_ONNX_MODEL_FILENAME` (string): overrides `embedding.onnx.modelFilename`
- `SCA_EMBEDDING_ONNX_TOKENIZER_FILENAME` (string): overrides `embedding.onnx.tokenizerFilename`
- `SCA_EMBEDDING_ONNX_REPO` (string): overrides `embedding.onnx.repo`
- `SCA_EMBEDDING_ONNX_DOWNLOAD` (bool): overrides `embedding.onnx.downloadOnMissing`
- `SCA_EMBEDDING_ONNX_SESSION_POOL_SIZE` (u32): overrides `embedding.onnx.sessionPoolSize`
- `SCA_EMBEDDING_CACHE_ENABLED` (bool): overrides `embedding.cache.enabled`
- `SCA_EMBEDDING_CACHE_MAX_ENTRIES` (u32): overrides `embedding.cache.maxEntries`
- `SCA_EMBEDDING_CACHE_MAX_BYTES` (u64): overrides `embedding.cache.maxBytes`
- `SCA_EMBEDDING_CACHE_DISK_ENABLED` (bool): overrides `embedding.cache.diskEnabled`
- `SCA_EMBEDDING_CACHE_DISK_PATH` (string): overrides `embedding.cache.diskPath`
- `SCA_EMBEDDING_CACHE_DISK_PROVIDER` (string): overrides `embedding.cache.diskProvider` (`sqlite` | `postgres` | `mysql` | `mssql`)
- `SCA_EMBEDDING_CACHE_DISK_CONNECTION` (string): overrides `embedding.cache.diskConnection`
- `SCA_EMBEDDING_CACHE_DISK_TABLE` (string): overrides `embedding.cache.diskTable`
- `SCA_EMBEDDING_CACHE_DISK_MAX_BYTES` (u64): overrides `embedding.cache.diskMaxBytes`
- `SCA_EMBEDDING_API_KEY` (string, **secret**): read/validated from env only

Aliases (no `SCA_` prefix) are also supported for the embedding keys above:
`EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_TIMEOUT_MS`,
`EMBEDDING_BATCH_SIZE`, `EMBEDDING_DIMENSION`, `EMBEDDING_BASE_URL`,
`EMBEDDING_LOCAL_FIRST`, `EMBEDDING_LOCAL_ONLY`, `EMBEDDING_ROUTING_MODE`,
`EMBEDDING_SPLIT_MAX_REMOTE_BATCHES`, `EMBEDDING_JOBS_PROGRESS_INTERVAL_MS`,
`EMBEDDING_JOBS_CANCEL_POLL_INTERVAL_MS`, `EMBEDDING_TEST_FALLBACK`,
`EMBEDDING_ONNX_MODEL_DIR`,
`EMBEDDING_ONNX_MODEL_FILENAME`, `EMBEDDING_ONNX_TOKENIZER_FILENAME`,
`EMBEDDING_ONNX_REPO`, `EMBEDDING_ONNX_DOWNLOAD`,
`EMBEDDING_ONNX_SESSION_POOL_SIZE`, `EMBEDDING_API_KEY`.

Provider-specific overrides (used by the adapter factory, not persisted in config):

- `OPENAI_API_KEY` (string, **secret**)
- `OPENAI_BASE_URL` (string URL)
- `OPENAI_MODEL` (string)
- `GEMINI_API_KEY` (string, **secret**)
- `GEMINI_BASE_URL` (string URL)
- `GEMINI_MODEL` (string)
- `VOYAGEAI_API_KEY` (string, **secret**)
- `VOYAGEAI_BASE_URL` (string URL)
- `VOYAGEAI_MODEL` (string)
- `OLLAMA_MODEL` (string)
- `OLLAMA_HOST` (string URL)

### Vector DB

- `SCA_VECTOR_DB_PROVIDER` (string): overrides `vectorDb.provider` (trimmed)
- `SCA_VECTOR_DB_INDEX_MODE` (`dense` | `hybrid`): overrides `vectorDb.indexMode`
- `SCA_VECTOR_DB_TIMEOUT_MS` (u64): overrides `vectorDb.timeoutMs`
- `SCA_VECTOR_DB_BATCH_SIZE` (u32): overrides `vectorDb.batchSize`
- `SCA_VECTOR_DB_BASE_URL` (string URL): overrides `vectorDb.baseUrl` (`http`/`https`)

### Sync

- `SCA_SYNC_ALLOWED_EXTENSIONS` (CSV string): overrides `sync.allowedExtensions`
  - Normalization: trim → strip leading `.` / `*.` → lowercase → sort + dedupe
- `SCA_SYNC_IGNORE_PATTERNS` (CSV string): overrides `sync.ignorePatterns`
  - Normalization: trim → `\` → `/` → collapse repeated `/` → sort + dedupe
- `SCA_SYNC_MAX_FILES` (u32): overrides `sync.maxFiles`
- `SCA_SYNC_MAX_FILE_SIZE_BYTES` (u64): overrides `sync.maxFileSizeBytes`

### Observability

- `SCA_LOG_FORMAT` (`json`): enable structured JSON logs on stderr
- `SCA_TELEMETRY_FORMAT` (`json`): enable JSON telemetry on stderr (defaults to log format)
- `SCA_LOG_LEVEL` (`debug` | `info` | `warn` | `error`): minimum log level (default `info`)
- `SCA_TRACE_SAMPLE_RATE` (`0.0` - `1.0`): span sampling rate (default `1.0`)

## Secret redaction

When an env var is considered a **secret** (`*_KEY`, `*_TOKEN`, `*_SECRET`,
`*_PASSWORD`), validation errors **never echo the raw value**. Instead, errors
store `<redacted>` in metadata (or omit the value entirely for empty inputs).

## Tools

- `scripts/print-effective-config.sh`: prints the effective config as JSON
  (defaults + env overrides).
