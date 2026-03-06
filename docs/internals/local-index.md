# Local Vector Index

The local vector index stores and searches dense embeddings without an external
vector database. The default kernel is `hnsw-rs`.

## Kernel selection

- Config field: `vectorDb.vectorKernel`
- Env override: `SCA_VECTOR_DB_VECTOR_KERNEL`
- Effective default: `hnsw-rs` when unset

Additional kernel variants may be available via optional dependencies.
When snapshots are written in v2 format, kernel metadata is persisted and
validated during reload.

## Persistence

Persistence is controlled by `SnapshotStorageMode`:

- `Disabled`: in-memory only.
- `Project`: snapshots stored under `<codebase_root>/.context/`.
- `Custom(<path>)`: snapshots stored under the custom absolute path.

Local vector snapshots are also controlled by `vectorDb.snapshotFormat`:

- `v1` (default): legacy JSON-only local snapshot.
- `v2`: JSON snapshot plus vector companion bundle (`snapshot.meta` +
  `vectors.u8.bin`) loaded via mmap.

Optional local snapshot cap:

- `vectorDb.snapshotMaxBytes`: max bytes per snapshot write for the selected
  persistence format. Oversize writes return `vector:snapshot_oversize`.

Optional kernel-mismatch migration switch:

- `vectorDb.forceReindexOnKernelChange` (env:
  `SCA_VECTOR_DB_FORCE_REINDEX_ON_KERNEL_CHANGE`)
  - `false` (default): kernel mismatch fails with
    `vector:snapshot_kernel_mismatch`
  - `true`: adapter rebuilds from JSON snapshot state and rewrites v2 kernel
    metadata to match configured kernel

Per collection, local metadata is persisted to:

```
.context/vector/collections/<collection>.json
```

When `snapshotFormat = "v2"`, companions are written under:

```
.context/vector/collections/<collection>.v2/
  snapshot.meta
  vectors.u8.bin
  ids.json
```

Load behavior for `snapshotFormat = "v2"`:

- If both companion files exist, local DB loads vectors from v2 bundle and
  validates snapshot kernel metadata against configured kernel.
- If neither exists, local DB falls back to v1 JSON and migrates to v2 on load.
- If only one companion exists, load fails with
  `vector:snapshot_missing_companion`.

Kernel metadata mismatch behavior (`snapshot.meta` vs configured kernel):

- `forceReindexOnKernelChange = false`: return
  `vector:snapshot_kernel_mismatch` with metadata:
  `snapshotKernel`, `configuredKernel`.
- `forceReindexOnKernelChange = true`: rebuild index from local JSON snapshot
  and rewrite v2 bundle with configured kernel metadata.

When writing snapshots, the adapter emits deterministic snapshot stats
(version, dimension, count, bytes, sorted metadata) to stderr.

The JSON snapshot payload still carries document metadata for local search.

The vector crate now includes a safe read-only mmap wrapper
(`semantic_code_vector::mmap::MmapBytes`) for upcoming binary snapshot payloads.
Local index v2 loading uses this wrapper to map companion byte files after
validating expected file length and checksum.

Snapshot v2 metadata constants are now defined in
`semantic_code_vector::snapshot`:

- `snapshot.meta` (metadata envelope with `SCA-SNAPSHOT` magic header)
- `vectors.u8.bin` (quantized vector payload + CRC32 integrity checksum)
- `graph.u32.bin` (graph payload, reserved for upcoming milestones)
- metadata includes `kernel` (default: `hnsw-rs`; additional variants available via optional dependencies)

The vector crate also provides `upgrade_v1_to_v2` and optional auto-upgrade
loading (`ReadSnapshotV2Options { auto_upgrade_v1: true }`) for migrating
legacy `snapshot.v1.json` bundles in place.

## Search metrics

Config field: `vectorDb.enableSearchMetrics` (default: `false`)

When enabled, structured metrics are collected during kernel searches and
emitted via tracing spans. Available metrics depend on the active kernel
variant.

Snapshot load timing is recorded as `load_ms` on the
`vector.snapshot.read_v2_with_options` span.

After each local search, an `adapter.vectordb.local.search_completed` debug
event logs kernel kind, backend, top_k, and match count.

## Filter expressions

Local filtering supports a strict allowlist:

- `relativePath == '<value>'`
- `relativePath != '<value>'`
- `language == '<value>'`
- `fileExtension == '<value>'`

Any other expression returns `vector:invalid_filter_expr`.

## Schema versioning

Snapshots are versioned and validated on load. A version mismatch returns
`vector:snapshot_version_mismatch` and requires a reindex.
