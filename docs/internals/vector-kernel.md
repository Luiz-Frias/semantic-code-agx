# Vector Kernel

The vector kernel provides a lightweight HNSW-backed index for local vector
search. It is designed to be deterministic and easy to persist for CLI-first
workflows.

## Core API

- `VectorIndex::insert` stores dense vectors by id.
- `VectorIndex::search` returns similarity-scored matches (higher is better).
- `VectorIndex::delete` removes ids (best-effort).
- `VectorIndex::snapshot` and `VectorIndex::from_snapshot` support persistence.

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

## Experimental hooks

The crate exposes an `experimental` feature gate reserved for future extension
points. The feature is disabled by default to keep the kernel stable.
