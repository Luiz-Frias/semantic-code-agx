# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added

### Changed

### Fixed

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

[0.2.0]: https://github.com/Luiz-Frias/semantic-code-agents-rs/releases/tag/v0.2.0
[Unreleased]: https://github.com/Luiz-Frias/semantic-code-agents-rs/compare/v0.2.0...HEAD
[0.1.1]: https://github.com/Luiz-Frias/semantic-code-agents-rs/releases/tag/v0.1.1
