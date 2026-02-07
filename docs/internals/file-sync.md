# File Sync Adapter

The local file sync adapter tracks **file-level** changes using a deterministic
Merkle DAG built from SHA256 file hashes. It is used by `reindex_by_change` to
identify added, removed, and modified files.

## Snapshot storage

Snapshots are stored under `.context/sync/` by default and are keyed by an MD5
hash of the absolute codebase root:

```
.context/
  sync/
    <md5-of-absolute-root>.json
```

`SnapshotStorageMode` controls the base directory. When disabled, snapshots are
kept in memory only.

## Snapshot format

The snapshot JSON is versioned and uses deterministic ordering:

```json
{
  "version": 1,
  "fileHashes": [["src/main.rs", "<sha256>"]],
  "merkleDAG": {
    "nodes": [["<node-id>", {"id": "<node-id>", "hash": "<node-id>", "data": "...", "parents": [], "children": []}]],
    "rootIds": ["<root-id>"]
  }
}
```

## Merkle DAG rules

- Each **file** hash is SHA256 of file bytes.
- The **root** hash is computed from the concatenation of file hashes in
  lexicographic path order.
- Each file node stores `"<relativePath>:<fileHash>"` as its data payload.

## Ignore semantics

The adapter honors configured ignore patterns and also ignores `.context/` to
avoid hashing snapshot state.

## Notes

- Sync is **file-level**. Chunk hashing is used for chunk IDs and embedding
  caches, not for sync snapshots.
