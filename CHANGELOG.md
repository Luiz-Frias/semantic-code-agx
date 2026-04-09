# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]



### Added

### Changed

### Fixed

## [0.4.0] - 2026-04-09

### Added
- feat(config): make HNSW M and ef_construction configurable (#45)
- feat(config): wire shadow PQ config + dimension-aware factory — FOR-109 (#44)
- feat(docs): +/public-sync cmd for published releases
- feat(docs): +important architectural designs

### Changed
- refactor(config): decompose apply_vector_db_env_overrides into sub-concern helpers (#46)
- refactor(vector): add versioned kernel source path contract (#43)

### Fixed
- fix(ci): prevent root mise.toml leak + clippy auto-fixes

### Infrastructure
- build(deps): major version bumps — sha2, md5, rusqlite, toml
- chore(deps): cargo update — 64 patch/minor bumps
- chore(ci): sync mise.ci.toml with root mise.toml (nightly-2026-04-01)
- chore(deps): bump workspace crates to v0.3.0 (#42)

## [0.3.0] - 2026-04-01

### Breaking Changes
- The `snapshot-subset` CLI command has been removed. Subset/tile tooling now lives in the sibling `bench-lab-rs` repo.
- Collection storage layout has changed: local collections now use a `generations/` directory structure with SQLite catalog tracking, an `ACTIVE` pointer file, and kernel-namespaced ready-state directories. Existing `.v2/` snapshot bundles are auto-migrated on first load, but the new layout is not backward-compatible with v0.2.0 tooling.

### Added

#### Generation-Based Collection Lifecycle
- Full generation control system for local vector collections: exact f32 row storage, SQLite catalog tracking, staged insert-then-publish lifecycle, and durable generation pointers (`ACTIVE` file). Each generation bundles base vectors, origin mappings, and kernel-ready state under `generations/{gen_id}/`.
- Kernel capability model (`VectorKernelKind`) enables the vector backend to advertise what it supports (graph, quantization, exact scan) and the adapter to select the appropriate load/build strategy.
- Staged publish flow: inserts accumulate in a clean-slate journal, then atomically publish via `flush` — prevents partial collections from being visible to readers.
- Tombstone reclamation: dead entries are garbage-collected on snapshot persist via `rebuild_active_index`, compacting the collection.
- Origin-ID mapping provides stable cross-checkpoint identity for incremental operations (quantization cache, DFRR graph reuse).

#### Collection Loader Actor
- Actor-based collection initialization replaces per-collection `OnceCell` gates. The actor serializes `Load`, `Evict`, and `Reload` commands through an `mpsc` channel, eliminating TOCTOU races in concurrent access paths. Hot reads/writes still use `Arc<RwLock<HashMap>>` directly — the actor only controls *when* the map is populated or cleared.
- Per-load `CancellationToken` support: individual collection loads can be cancelled without affecting other in-flight operations.

#### DFRR Ready-State Prewarm
- DFRR kernel state (graph, rank structures) is now persisted to a `dfrr/` sidecar inside `.v2/` bundles. On subsequent loads, the kernel restores from cache instead of rebuilding — eliminating the ~34s cold-start graph reconstruction for large collections.
- Prewarm lifecycle with heartbeat logging and post-index finalization hooks. The adapter gates DFRR search on prewarmed ready state to prevent queries against partially-built structures.

#### CLI
- `sca agent-doc [COMMAND]`: machine-readable AGX protocol spec in YAML. Full invocation emits the complete protocol (commands, NDJSON shapes, exit codes, recovery table, filter syntax, workflows). Scoped invocation emits a single command contract.

#### Configuration
- `DfrrBq1ThresholdMode` enum with `ClusterPercentile` variant for BQ1 percentile-assist threshold selection.
- `max_chunks` field in sync configuration for corpus-size limiting.
- DFRR BQ1 environment variable overrides (`SCA_DFRR_BQ1_*`).

#### Vector Kernel
- Cooperative cancellation for HNSW insert and snapshot load operations via `CancellationToken`.
- Distance evaluation tracking in HNSW search for observability.
- Exact row and generation foundation types (`ExactVectorRow`, `ExactVectorRows`, `GenerationId`, `OriginId`).

#### Tooling
- Claude Code slash command workflows: `gather-context`, `implement`, `review-fix`, `review-patterns`.
- Linear status update post-commit hook: auto-moves Linear issues to "In Review" when a PR exists on `FOR-*` branches.
- Pre-commit configuration via `prek` with 10 active git hooks.

#### Documentation
- Architecture: collection loader actor design, incremental DFRR design, curvature-guided DFRR maintenance.
- Research: DFRR mutation concurrency report.

### Changed
- Benchmark infrastructure extracted to sibling [`bench-lab-rs`](https://github.com/Luiz-Frias/bench-lab-rs) repo. Removed ~15 scripts (`bench-run-all.sh`, `bench-scaling-sweep.sh`, `bench-analyze-*.py`, `bench-dfrr-vs-hnsw.py`, etc.), the in-tree `bench-runner` crate, DVC-tracked scaling data, and golden query fixtures.
- Local adapter now defaults to V2 snapshots with DFRR persistence and skips HNSW graph rebuild for graph-agnostic kernels.
- Collection loader refactored from `OnceCell` per-collection gates to actor-based architecture with serialized lifecycle commands.
- CLI runtime unified to `multi_thread(1)` for all async paths — fixes macOS kqueue edge-triggered notification loss while maintaining single-worker scheduling semantics.
- Shared cancellation token backed by `tokio-util` (`CancellationToken` replaces manual implementation).
- Checkpoint build path uses typestate v2 for compile-time correctness guarantees.
- Splitter deduplicates fallback spans before chunk materialization via fragment-aware chunk identity.
- Vector index tracks origin IDs for stable cross-checkpoint identity mapping.

### Fixed
- **Three-layer race condition** in collection initialization: (1) macOS kqueue edge-triggered notification loss between `spawn_blocking` and runtime park, (2) TOCTOU in check-then-load allowing duplicate 800MB+ loads, (3) no reload/failure recovery after `OnceCell` drop. All resolved by the actor architecture and `multi_thread(1)` runtime.
- Pre-embed hang and collection load races resolved via actor command serialization.
- HNSW graph identity decoupled from payload slots — prevents stale graph references on unsafe checkpoint.
- HNSW insertion path serialized to prevent concurrent data corruption.
- Staging upsert parity restored for vector snapshot insert operations.
- Duplicate-ID sidecar load errors now raised explicitly instead of silently skipping.
- Fragment-aware chunk identity prevents duplicate chunks from splitter dedup failures.
- HNSW `ef` parameter now honored when zero-threshold DFRR search doesn't widen (search quality fix).
- Panic vs cancellation properly discriminated in `spawn_blocking` `JoinError` handling.

### Performance
- Parallel HNSW insertion via `rayon` for batch inserts.
- Incremental SQ8 quantization across checkpoints — avoids re-quantizing the entire collection on every persist.
- Shared quantized vectors via `Arc` between snapshot and cache, eliminating redundant copies.
- Checkpoint frequency throttled by vector count to reduce I/O overhead on small inserts.
- Triple quantization eliminated in the V2 checkpoint path.
- Parallel index + sidecar loading via `try_join!`.
- HNSW graph build skipped entirely for graph-agnostic kernels (DFRR, flat-scan).
- Origin-ID mapping enables incremental quantization cache across checkpoints.

## [0.2.0] - 2026-03-06

### Breaking Changes
- MSRV is now Rust `1.96.0`.
- The public library surface tightened across `facade`, `ports`, and `config`: `VectorDbPort::search` now returns `VectorSearchResponse`, `VectorDbPort::flush` was added, lending shims are sealed, and several request/config parsing helpers were narrowed or removed.
- Experimental DFRR and ANE integrations now come from private git dependencies instead of an in-tree kernel crate, which changes contributor build assumptions for feature-enabled local builds.

### Added
- Snapshot v2 companion bundles with mmap-safe loads, CRC/integrity checks, quantization metadata, and supporting docs in [Vector Snapshot Format](docs/vector-snapshot-format.md), [Vector Quantization](docs/vector-quantization.md), and [mmap Safety Invariants](docs/mmap-safety.md).
- New local kernel and query controls across config and CLI, including `hnsw-rs`, experimental `dfrr`, exact `flat-scan`, richer search stats, and warm-session search paths.
- New CLI workflows: `estimate-storage`, `calibrate`, `snapshot-subset`, `search --stdin-batch`, vector-kernel overrides, and richer `self-check` build metadata.
- BQ1 calibration across domain, ports, app, and infra, including optional auto-calibration and EMA drift tracking for DFRR tuning.
- Apple Neural Engine embedding support and execution controls for compatible builds.
- Expanded proof lanes and bridge/e2e coverage, with guidance in [Proof Lanes](docs/proof-lanes.md) and [Nightly Toolchain Policy](docs/nightly-toolchain-policy.md).

### Changed
- The CLI moved from a monolithic `main.rs` to reusable command/library modules with structured tracing and redacted logging. See [CLI Reference](docs/reference/cli.md) and [Observability](docs/observability.md).
- Local vector persistence now uses WAL/checkpoint/flush durability flows, persisted HNSW graph data, and faster v2 collection loading.
- The public boundary now favors owned facade DTOs, curated re-exports, and a runtime-focused environment view instead of broad parser internals.
- Benchmarking and release engineering now include DVC-tracked scaling data, parity gates, and PGO tooling documented in [PGO Guide](docs/guides/pgo.md).

### Fixed
- Indexing now fails fast on embed/insert stage errors instead of carrying partial success deeper into the pipeline.
- ANE config/env handling and model-path resolution were corrected for non-ANE and override-driven builds.
- DFRR and snapshot flows were stabilized with deterministic sampling, frontier/guard fixes, score stability work, and better result deduplication.
- CLI and test harness reliability improved through stricter calibration/input validation, insert-concurrency fixes, proof-lane coverage, and fixture race fixes.

### Migration Notes
- Update local toolchains, CI images, and contributor docs to Rust `1.96.0` before adopting `v0.2.0`.
- If you implement `ports` directly, adapt search call sites to `VectorSearchResponse`, wire the new `flush` hook where durability matters, and stop relying on external `*PortLend` implementations.
- If you build experimental DFRR or ANE paths locally, ensure the sibling private dependencies referenced from the workspace are available in your environment.

## [0.1.1] - 2026-02-07

### Added

#### Highlights
- CLI-first semantic code search with agent-friendly output (`--agent`, NDJSON).
- Local indexing backed by an HNSW vector index; optional Milvus support (REST or gRPC).
- Embedding adapters for ONNX (local) and remote providers (OpenAI, Gemini, Voyage, Ollama).
- Incremental indexing via Merkle snapshots plus `.contextignore`-based scan excludes.

#### CLI
- Commands: `init`, `index`, `search`, `status`, `clear`, `reindex`.
- Background jobs for `index`/`reindex` via `--background`, with `jobs status`/`jobs cancel`.
- Output control: `--output text|json|ndjson`, plus `--agent` for machine-friendly defaults.
- Developer-only `self-check` command (debug builds or `dev-tools` feature).

#### Configuration
- TOML/JSON configuration with default discovery of `.context/config.toml`.
- Config inspection and validation via `config check`, `config show`, and `config validate`.

#### Embeddings
- Providers: ONNX (local), OpenAI, Gemini, Ollama, Voyage.
- Request-scoped retries and timeouts around embedding calls.
- Embedding cache (in-memory) with optional disk-backed cache (SQLite; Postgres/MySQL/MSSQL behind feature flags).

#### Vector Databases
- Local vector database backed by HNSW with persistence.
- Milvus vector database adapters (REST and gRPC) with structured error mapping.

#### Indexing & File Sync
- Incremental indexing using Merkle snapshots to detect changes.
- `.contextignore` support for scan-time ignore rules.

#### Splitting
- Tree-sitter-based semantic splitting for Rust, Go, Java, JavaScript, TypeScript/TSX, Python, C, and C++, with line-based fallback for other file types.

#### API & Types
- API v1 DTOs and fixtures with validation and parity tests.
- Domain primitives and metadata/search/state types with parse constructors and validation.
- Shared error envelopes across crates.

#### Tooling & Quality
- `mise.toml` for local toolchain/dev tooling setup.
- `cargo nextest` via `just` recipes for faster test runs.
- Pre-commit hooks configuration.

#### Documentation
- Getting started, configuration guide, CLI reference, troubleshooting, and architecture docs under `docs/`.

#### Compatibility
- MSRV: Rust 1.95.0 (Rust 2024 edition).

### Security

- Best-effort secret redaction helpers for config and error output.

### Fixed

- Ignore Cargo advisory DB and lock artifacts via `.gitignore`.
- Developer setup now prefers `.env.local` for local overrides.

[0.3.0]: https://github.com/Luiz-Frias/semantic-code-agx/releases/tag/v0.3.0
[0.2.0]: https://github.com/Luiz-Frias/semantic-code-agx/releases/tag/v0.2.0
[0.4.0]: https://github.com/Luiz-Frias/semantic-code-agx/compare/v0.3.0...v0.4.0
[Unreleased]: https://github.com/Luiz-Frias/semantic-code-agx/compare/v0.4.0...HEAD
[0.1.1]: https://github.com/Luiz-Frias/semantic-code-agx/releases/tag/v0.1.1
