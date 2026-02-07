# Local Vector Index

The local vector index uses the HNSW-backed kernel to store and search dense
embeddings without an external vector database.

## Persistence

Persistence is controlled by `SnapshotStorageMode`:

- `Disabled`: in-memory only.
- `Project`: snapshots stored under `<codebase_root>/.context/`.
- `Custom(<path>)`: snapshots stored under the custom absolute path.

Snapshots are written per collection to:

```
.context/vector/collections/<collection>.json
```

The snapshot payload contains both vectors and document metadata so local
search can restart without re-indexing.

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
