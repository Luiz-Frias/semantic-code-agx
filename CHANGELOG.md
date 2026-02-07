# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

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

[Unreleased]: https://github.com/Luiz-Frias/semantic-code-agx/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Luiz-Frias/semantic-code-agx/releases/tag/v0.1.1
