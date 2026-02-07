# Resilience (Cache, Retry, Timeouts)

Phase 06 Milestone 3 adds a resilience layer across embedding requests and
indexing pipelines.

## Embedding Cache

- **In-memory LRU** cache keyed by content hash (provider + model + base URL + dimension + text).
- **Disk cache** (SQLite) stored under `.context/cache/embeddings/cache.db` by default.
- Cache is **disabled by default**; enable via config or env.

### Disk Cache Providers

- **SQLite** (default): single file stored under `.context/cache/embeddings/cache.db`.
- **Postgres / MySQL / MSSQL**: shared table storage via connection string and optional table name.
  - Requires adapter features: `cache-postgres`, `cache-mysql`, `cache-mssql`.
- All providers use a **versioned schema** and treat caches as disposable.
  - On schema version mismatch, SQLite files are rotated to `.legacy.<version>_<timestamp>`.
  - For server backends, tables are renamed to `{table}_legacy_<version>_<timestamp>`.
- Eviction is best-effort by `last_accessed_ms` when `diskMaxBytes` is exceeded.

## Retry Policy

Retries apply only to **retriable** errors (`ErrorClass::Retriable`).

- Exponential backoff with jitter.
- Retry policy is configurable under `core.retry`.

## Timeout Wrappers

All embedding calls are wrapped with a shared timeout helper that respects
request cancellation.

## Telemetry

When telemetry is provided, the following counters are emitted:

- `embedding.cache.hit` (tags: `provider`, `source` = `memory` | `disk`)
- `embedding.cache.miss` (tags: `provider`)
- `retry.attempt` (tags: `provider`)
- `retry.exhausted` (tags: `provider`)
- `timeout.triggered` (tags: `provider`)

## Config Examples

```json
{
  "core": {
    "retry": {
      "maxAttempts": 3,
      "baseDelayMs": 250,
      "maxDelayMs": 5000,
      "jitterRatioPct": 20
    },
    "maxInFlightEmbeddingBatches": 4,
    "maxBufferedChunks": 800
  },
  "embedding": {
    "cache": {
      "enabled": true,
      "maxEntries": 4096,
      "maxBytes": 134217728,
      "diskEnabled": true,
      "diskMaxBytes": 1073741824
    }
  }
}
```
