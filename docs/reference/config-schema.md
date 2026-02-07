# Config Schema

Phase 03 introduces a **deterministic, validated backend config schema**. The
schema is parsed with `serde` and then validated + normalized in
`semantic-code-config`.

## Format

- **File format**: JSON or TOML.
- **Top-level**: object with `version` and section objects (`core`, `embedding`,
  `vectorDb`, `sync`).
- **Unknown fields**: rejected (deny-by-default) to prevent silent typos.

## Versioning

- `version`: integer schema version.
- Current supported version: `1` (`CURRENT_CONFIG_VERSION`).

## Sections

### `core`

- `timeoutMs` (u64): boundary timeout in milliseconds.
  - Bounds: `1000..=600000`
- `maxConcurrency` (u32): max concurrent in-flight work.
  - Bounds: `1..=256`
- `maxInFlightFiles` (u32, optional): cap in-flight file tasks.
  - Bounds: `1..=256`
- `maxInFlightEmbeddingBatches` (u32, optional): cap in-flight embedding batches.
  - Bounds: `1..=256`
- `maxInFlightInserts` (u32, optional): cap in-flight insert batches.
  - Bounds: `1..=256`
- `maxBufferedChunks` (u32, optional): cap buffered chunks.
  - Bounds: `1..=1000000`
- `maxBufferedEmbeddings` (u32, optional): cap buffered embeddings.
  - Bounds: `1..=1000000`
- `maxChunkChars` (u32): max characters per chunk (best-effort).
  - Bounds: `1..=20000`
- `retry` (object): retry policy for transient failures.
  - `maxAttempts` (u32): total attempts including the first.
    - Bounds: `1..=10`
  - `baseDelayMs` (u64): base backoff delay.
    - Bounds: `1..=60000`
  - `maxDelayMs` (u64): maximum backoff delay.
    - Bounds: `1..=600000`
  - `jitterRatioPct` (u32): jitter ratio percent.
    - Bounds: `0..=100`

### `embedding`

- `provider` (string, optional): provider identifier (trimmed).
- `model` (string, optional): provider-specific model override.
- `baseUrl` (string, optional): provider base URL (`http`/`https`).
- `dimension` (u32, optional): embedding dimension override.
  - Bounds: `1..=65536`
- `timeoutMs` (u64): embedding call timeout.
  - Bounds: `1000..=1200000`
- `batchSize` (u32): embedding batch size.
  - Bounds: `1..=8192`
- `localFirst` (bool): prefer local ONNX embeddings if available.
- `localOnly` (bool): force local ONNX embeddings only.
- `onnx` (object): local ONNX configuration.
  - `modelDir` (string, optional): directory containing `tokenizer.json` + ONNX model.
  - `modelFilename` (string, optional): model file override (default checks `onnx/model.onnx`).
  - `tokenizerFilename` (string, optional): tokenizer file override.
  - `repo` (string, optional): Hugging Face repo ID used for downloads.
    - Default cache: `.context/models/onnx/<repo>` (fallback: `.context/onnx-cache/<repo>`).
  - `downloadOnMissing` (bool): download missing ONNX assets on first use.
  - `sessionPoolSize` (u32): number of ONNX sessions kept in the pool.
    - Bounds: `1..=64`
- `routing` (object): embedding routing configuration.
  - `mode` (string, optional): routing mode.
    - Allowed: `localFirst` | `remoteFirst` | `split`
  - `split` (object): split routing settings.
    - `maxRemoteBatches` (u32, optional): max remote batches when `mode = split`.
      - Bounds: `1..=1000000`
- `jobs` (object): background job tuning.
  - `progressIntervalMs` (u64): progress update interval for background jobs.
    - Bounds: `50..=60000`
  - `cancelPollIntervalMs` (u64): cancel polling interval for background jobs.
    - Bounds: `50..=60000`
- `cache` (object): embedding cache configuration.
  - `enabled` (bool): enable in-memory cache.
  - `maxEntries` (u32): max in-memory entries.
    - Bounds: `1..=100000`
  - `maxBytes` (u64): max in-memory bytes.
    - Bounds: `1..=10000000000`
  - `diskEnabled` (bool): enable disk cache.
  - `diskProvider` (string, optional): disk cache provider.
    - Allowed: `sqlite` | `postgres` | `mysql` | `mssql`
    - Default: `sqlite`
  - `diskPath` (string, optional): SQLite cache path (only when `diskProvider = sqlite`).
  - `diskConnection` (string, optional): connection string for Postgres/MySQL/MSSQL (required for non-sqlite).
  - `diskTable` (string, optional): table name override (alphanumeric + `_` only, default `embedding_cache`).
  - `diskMaxBytes` (u64): max disk bytes.
    - Bounds: `1..=100000000000`

### `vectorDb`

- `provider` (string, optional): provider identifier (trimmed).
- `baseUrl` (string, optional): provider base URL (`http`/`https`).
- `indexMode` (`dense` | `hybrid`): used for collection naming decisions.
- `timeoutMs` (u64): vectordb call timeout.
  - Bounds: `1000..=1200000`
- `batchSize` (u32): insert/delete batch size.
  - Bounds: `1..=16384`
- `snapshotStorage` (`disabled` | `project` | `{ custom: "<path>" }`):
  local snapshot persistence mode.

### `sync`

- `allowedExtensions` (string[]): allowlist of file extensions.
  - Normalization:
    - trims whitespace
    - strips leading `.` and `*.` (examples: `.rs`, `*.rs`)
    - lowercases
    - sorts + deduplicates
  - Validation:
    - entries must be non-empty and `[a-zA-Z0-9]+`
    - max entries: `128`
- `ignorePatterns` (string[]): ignore patterns applied during scan.
  - Normalization:
    - trims whitespace
    - converts `\` to `/` and collapses repeated `/`
    - sorts + deduplicates
  - Validation:
    - entries must be non-empty after trimming
    - max entries: `512`
- `maxFiles` (u32): max files considered during scan.
  - Bounds: `1..=10000000`
- `maxFileSizeBytes` (u64): max file size read into memory.
  - Bounds: `1..=100000000`

## Error mapping

Validation failures are mapped to `ErrorEnvelope` with a `config:*` `ErrorCode`
and helpful metadata (ex: `section`, `field`, bounds).

## Example

```json
{
  "version": 1,
  "core": { "timeoutMs": 30000, "maxConcurrency": 8 },
  "embedding": { "dimension": 1536, "timeoutMs": 60000, "batchSize": 32 },
  "vectorDb": {
    "indexMode": "dense",
    "timeoutMs": 60000,
    "batchSize": 128,
    "snapshotStorage": "project"
  },
  "sync": {
    "allowedExtensions": ["rs", "ts", "tsx"],
    "ignorePatterns": ["target/", "node_modules/"],
    "maxFiles": 250000,
    "maxFileSizeBytes": 2000000
  }
}
```
