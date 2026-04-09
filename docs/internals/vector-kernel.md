# Vector Kernel

The vector kernel layer defines how `VectorIndex` dispatches search across
kernel families. Phase 06 introduces explicit kernel selection and snapshot
kernel metadata plumbing.

## Core API

- `VectorIndex::insert` stores dense vectors by id.
- `VectorIndex::search` returns similarity-scored matches (higher is better).
- `VectorIndex::delete` removes ids (best-effort).
- `VectorIndex::search_with_kernel` dispatches by `VectorKernelKind`.
- `VectorIndex::snapshot` and `VectorIndex::from_snapshot` support persistence.
- `VectorIndex::snapshot_v2` and `VectorIndex::from_snapshot_v2` persist/load
  v2 bundles for quantized storage.
- `VectorIndex::snapshot_v2_for_kernel` writes v2 bundle metadata with explicit
  kernel family.
- `VectorIndex::snapshot_stats` computes deterministic snapshot metrics
  (version, dimension, count, bytes, metadata).
- `VectorIndex::write_snapshot` selects v1 or v2 write format via
  `VectorSnapshotWriteVersion`.

## Kernel types and dispatch

Kernel enum:

- `VectorKernelKind::HnswRs`
- Additional variants may be available via optional dependencies

Kernel trait:

- `VectorKernel::kind()`
- `VectorKernel::search(...)`
- `VectorKernel::search_with_config_override(...)`
- `VectorKernel::set_snapshot_dir(...)`
- `VectorKernel::materialize_runtime(...)`
- `VectorKernel::warm_from_source(...)`

Source/runtime contracts:

- `legacy_vector_index_v0`
  The compatibility path. The kernel is warmed and searched with a concrete
  `VectorIndex` and may materialize a borrowed runtime wrapper over that same
  index.
- `segmented_source_v1`
  The canonical DFRR path. The loader may warm ready state from a
  `PublishedGenerationKernelSource` derived from the immutable exact-row
  generation bundle instead of rebuilding the kernel state from
  `VectorIndex::active_records()`.

This means the collection loader and DFRR prewarm flow can use the published
generation as the source of truth for ready-state materialization while still
binding the ready state to the runtime `VectorIndex` that the local adapter
keeps for collection bookkeeping and compatibility.

Current runtime behavior is build-dependent:

- `HnswKernel` is the concrete in-tree implementation used by default.
- Requesting an unsupported kernel returns `vector:kernel_unsupported`.

## Parameters

The `HnswParams` struct controls index sizing and search width:

- `max_nb_connection`
- `max_layer`
- `ef_construction`
- `ef_search`
- `max_elements`

Defaults are tuned for small to mid-sized local repos and can be adjusted later
as we add CLI tuning options.

## Snapshot format

Snapshots are versioned and include:

- `version`
- `dimension`
- `params`
- `records` (id + vector)

Mismatched versions return a `vector:snapshot_version_mismatch` error.

For binary companion snapshots, v2 metadata + bundle read/write helpers live in
`semantic_code_vector::snapshot`, including quantization metadata, CRC32
integrity checks, and v1->v2 upgrade helpers. See
[Vector Snapshot Format](./vector-snapshot-format.md).

Kernel metadata in v2 snapshots supports future kernel variants. Local adapter
load validates snapshot kernel metadata against configured kernel and either
fails (`vector:snapshot_kernel_mismatch`) or rewrites metadata when
`forceReindexOnKernelChange` is enabled.

At the `VectorIndex` boundary, v2 persistence also writes an `ids.json` sidecar
to preserve stable record IDs for dequantized reload. When loading v2 with
auto-upgrade enabled, missing `ids.json` falls back to IDs from
`snapshot.v1.json`.

The trait-level `set_snapshot_dir()` hook lets a host adapter hand the
collection `.v2/` directory to kernels that support private caches. In
persistence-capable experimental DFRR builds, that hook is used to persist a
`dfrr/` ready-state sidecar (graph, vectors, rank index, node mapping, and
state metadata) and restore it on later processes. If the sidecar is missing,
corrupt, or stale relative to node count/dimension, DFRR rebuilds from the
loaded collection and rewrites the cache.

For generation-backed DFRR loads, the local adapter now prefers warming from
`PublishedGenerationKernelSource` whenever the kernel family reports
`segmented_source_v1` as its canonical source path. Legacy kernels continue to
use the `legacy_vector_index_v0` path.

## Snapshot stats and limits

`SnapshotStats` exposes deterministic snapshot metrics with a sorted metadata
map (`BTreeMap`) so logs and diagnostics remain stable.

Writes can be guarded with a size cap:

- `VectorIndex::write_snapshot_with_size_limit(...)`
- `VectorIndex::snapshot_v2_with_size_limit(...)`

Oversize writes return `vector:snapshot_oversize`.

## Experimental u8 search

`VectorIndex` also exposes an experimental quantized search path (u8 cosine)
that can be toggled at runtime by `vectorDb.experimentalU8Search`. This path is
feature-gated and falls back to f32 search when disabled.

See [Vector Quantization](./vector-quantization.md) for limits and
usage guidance.

## Experimental hooks

The crate exposes an `experimental` feature gate reserved for future extension
points. The feature is disabled by default to keep the kernel stable.

## Selection plumbing

Kernel selection flows through config and infra:

- Config enum: `semantic_code_config::VectorKernelKind`
- Effective selector: `VectorDbConfig::effective_vector_kernel()`
- Local adapter factory: `semantic_code_infra::build_vectordb_port(...)`

CLI overrides provide `vectorDb.vectorKernel` on selected commands, then config
normalization determines the effective kernel passed to local adapter
construction.
