# Vector Snapshot Format v2

Phase 01 defined the base v2 bundle structure for quantized payloads.
Later phases layered kernel metadata and kernel-private sidecars on top of that
base contract.

## Scope

- Rust module: `semantic_code_vector::snapshot`
- Metadata type: `VectorSnapshotMeta`
- Version enum: `VectorSnapshotVersion` (`V1`, `V2`)
- Bundle read/write: `write_snapshot_v2`, `read_snapshot_v2`
- Upgrade helper: `upgrade_v1_to_v2`

## Layout constants

`semantic_code_vector::snapshot` defines stable v2 constants:

- `SNAPSHOT_V2_META_FILE_NAME`: `snapshot.meta`
- `SNAPSHOT_V2_VECTORS_FILE_NAME`: `vectors.u8.bin`
- `SNAPSHOT_V2_GRAPH_FILE_NAME`: `graph.u32.bin`
- `SNAPSHOT_V2_MAGIC_HEADER`: `SCA-SNAPSHOT\n`
- `SNAPSHOT_V2_CHECKSUM_ALGORITHM`: `crc32`
- `SNAPSHOT_VERSION_V1`: `1`
- `SNAPSHOT_VERSION_V2`: `2`

## Bundle files

A v2 bundle is a directory containing at minimum:

1. `snapshot.meta`
2. `vectors.u8.bin`
3. (reserved) `graph.u32.bin`
4. `ids.json` when written through `VectorIndex` / local adapter flows
5. optional kernel-private companion directories written by higher-level integrations

## Metadata file format (`snapshot.meta`)

`snapshot.meta` bytes are:

1. Magic header bytes (`SNAPSHOT_V2_MAGIC_HEADER`)
2. UTF-8 JSON payload for `VectorSnapshotMeta`

`VectorSnapshotMeta` fields:

- `version`
- `dimension`
- `count`
- `params` (`HnswParams`)
- `kernel` (`VectorKernelKind`: `hnsw-rs`)
- `quantization` (`QuantizationParams`)
- `vectorsCrc32` (CRC32 checksum for `vectors.u8.bin`)

## Integrity and length validation

- `write_snapshot_v2` validates `count * dimension == vectors.len()`.
- `read_snapshot_v2` validates expected file length before mmap.
- `read_snapshot_v2` recomputes CRC32 and compares with metadata.
- Corrupt data returns typed `SnapshotError` variants (length mismatch, hash
  mismatch, parse/metadata errors).

## Kernel metadata compatibility

Kernel metadata in v2 snapshots supports future kernel variants via optional
dependencies. The `kernel` field defaults to `hnsw-rs` when absent in metadata.

## DFRR ready-state sidecar

When SCA uses the experimental DFRR kernel and the linked
`semantic-code-dfrr-hnsw` build supports ready-state persistence, passing a
collection `.v2/` directory through `VectorKernel::set_snapshot_dir()` lets
the kernel populate an internal `dfrr/` cache beside the base v2 bundle:

- `dfrr/manifest.json`
- `dfrr/graph.bin`
- `dfrr/vectors.bin`
- `dfrr/binary_vectors.bin`
- `dfrr/rank_index.json`
- `dfrr/node_to_id.json`
- `dfrr/dfrr_state.json`

The base v2 bundle remains the authoritative collection snapshot. The `dfrr/`
directory is a kernel-private cache that is rebuilt automatically when missing,
invalid, or stale relative to the active collection geometry.

## Upgrade behavior (v1 -> v2)

`upgrade_v1_to_v2(snapshot_dir)` reads `snapshot.v1.json`, quantizes vectors,
writes `snapshot.meta` + `vectors.u8.bin`, and returns v2 metadata.

`read_snapshot_v2_with_options` supports optional auto-upgrade:

- `ReadSnapshotV2Options { auto_upgrade_v1: true }`
- If `snapshot.meta` is missing and `snapshot.v1.json` exists, v1 is upgraded
  in-place and then loaded as v2.

## Safety and determinism notes

- Mmap stays encapsulated in `semantic_code_vector::mmap`.
- Hash and length checks are deterministic and explicit.
- Unsupported versions, invalid magic, invalid hash formatting, and upgrade
  mismatches are surfaced as typed errors.

## Snapshot metrics

`VectorIndex::snapshot_stats(VectorSnapshotWriteVersion::V2)` reports:

- version
- dimension
- count
- total bytes (meta + vectors + ids)
- deterministic metadata keys (sorted map)
