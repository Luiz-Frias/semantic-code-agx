# Vector DB adapters (Milvus)

This repository supports local vector storage and external Milvus adapters.

## Providers

- `local`: in-process HNSW-backed storage with snapshots.
- `milvus_grpc`: Milvus gRPC adapter (preferred for external). Use when low-latency and full feature coverage is needed.
- `milvus_rest`: Milvus REST adapter (useful when gRPC is not available).
- `milvus`: alias for `milvus_grpc`.

## Config fields (`vectorDb`)

- `provider`: `local` | `milvus_grpc` | `milvus_rest` | `milvus`.
- `address`: host or URL for Milvus (e.g. `localhost:19530`, `https://milvus.example.com:443`).
- `baseUrl`: REST base URL override (e.g. `http://localhost:9091/v2/vectordb`).
- `database`: optional database name (default depends on Milvus configuration).
- `ssl`: enable TLS for gRPC (`true`/`false`).
- `token`: optional auth token (not serialized to disk).
- `username`: optional auth username.
- `password`: optional auth password (not serialized to disk).
- `indexMode`: `dense` | `hybrid` (controls collection naming and creation).
- `timeoutMs`: request timeout.
- `batchSize`: insert/delete batch size.
- `snapshotStorage`: applies to local snapshots only.

If both `address` and `baseUrl` are provided, `address` is used.

## CLI overrides

All index/search/clear/status commands accept vector DB overrides:

- `--vector-db-provider`
- `--vector-db-address`
- `--vector-db-base-url`
- `--vector-db-database`
- `--vector-db-ssl`
- `--vector-db-token`
- `--vector-db-username`
- `--vector-db-password`

These flags are merged as config overrides for the current command only.

## Environment variables

- `SCA_VECTOR_DB_PROVIDER`
- `SCA_VECTOR_DB_ADDRESS`
- `SCA_VECTOR_DB_BASE_URL`
- `SCA_VECTOR_DB_DATABASE`
- `SCA_VECTOR_DB_SSL`
- `SCA_VECTOR_DB_TOKEN`
- `SCA_VECTOR_DB_USERNAME`
- `SCA_VECTOR_DB_PASSWORD`
- `SCA_VECTOR_DB_INDEX_MODE`
- `SCA_VECTOR_DB_TIMEOUT_MS`
- `SCA_VECTOR_DB_BATCH_SIZE`

## Notes

- Hybrid search expects both dense and sparse sub-queries. The adapter returns an error when fewer than two sub-queries are provided.
- REST base URLs are normalized to `/v2/vectordb` when missing.
