# Documentation

Welcome to the **semantic-code-agents-rs** documentation -- a CLI-first semantic code search engine in Rust.

## Quick Links

| I want to... | Go to |
|---|---|
| Get started in 5 minutes | [Getting Started](./getting-started.md) |
| Index my codebase | [Indexing Guide](./guides/indexing.md) |
| Search semantically | [Searching Guide](./guides/searching.md) |
| Understand the architecture | [Architecture Overview](./architecture/README.md) |
| Look up a CLI command | [CLI Reference](./reference/cli.md) |
| Configure the system | [Configuration Guide](./guides/configuration.md) |
| Solve a problem | [Troubleshooting](./troubleshooting.md) |
| Understand a design decision | [ADRs](./adrs/README.md) |
| Contribute code | [Contributing](../CONTRIBUTING.md) |

---

## Getting Started

- [Getting Started](./getting-started.md) -- Install and run your first search in 5 minutes
- [Release & Install](./release.md) -- Release artifacts and install methods
- [FAQ](./faq.md) -- Frequently asked questions
- [Troubleshooting](./troubleshooting.md) -- Common issues and solutions

## Guides

How-to instructions for specific tasks.

- [Configuration](./guides/configuration.md) -- Config sources, format, profiles, and validation
- [Embedding Providers](./guides/embedding-providers.md) -- Setup for ONNX, OpenAI, Gemini, Voyage, Ollama
- [Indexing](./guides/indexing.md) -- Full indexing, incremental reindex, and index management
- [Searching](./guides/searching.md) -- Semantic search, query writing, and result interpretation
- [Profile-Guided Optimization](./guides/pgo.md) -- PGO build pipeline for 10-20% throughput gains

## Reference

Exact specifications and API contracts.

- [CLI Reference](./reference/cli.md) -- Commands, flags, output formats, agent integration
- [Config Schema](./reference/config-schema.md) -- Full validated configuration schema
- [Environment Variables](./reference/env-vars.md) -- All `SCA_*` env vars and provider overrides
- [Error Codes](./reference/error-codes.md) -- Error envelope structure, kinds, and retry policy
- [API v1 Contract](./reference/api-v1.md) -- DTO overview, error mapping, and redaction policy
- [Port Traits](./reference/ports.md) -- Boundary contracts for adapters

## Architecture

System design and design rationale.

- [Architecture Overview](./architecture/README.md) -- Hexagonal design, crate map summary, data flow
- [Hexagonal Architecture](./architecture/hexagonal.md) -- Ports and adapters pattern deep-dive
- [Crate Map](./architecture/crate-map.md) -- Workspace structure and full dependency graph
- [Data Flow](./architecture/data-flow.md) -- Sequence diagrams for index, search, reindex, error, and config flows
- [Collection Loader Actor](./architecture/collection-loader-actor.md) -- Actor-based collection lifecycle (replaces OnceCell)
- [DFRR Design](./architecture/dfrr-design.md) -- Incremental maintenance and curvature-guided algorithms
- [Advanced Patterns](./architecture/patterns.md) -- FSM, GAT, and typestate refactoring opportunities
- [ADRs](./adrs/README.md) -- Architecture Decision Records

## Internals

Deep implementation details for contributors and curious readers.

- [Vector Kernel](./internals/vector-kernel.md) -- Kernel dispatch, HNSW params, snapshot format plumbing
- [Local Index](./internals/local-index.md) -- Persistence modes, snapshot loading, kernel metadata, filters
- [Splitter](./internals/splitter.md) -- Tree-sitter chunking, supported languages, fallback behavior
- [Concurrency](./internals/concurrency.md) -- CancellationToken, BoundedQueue, WorkerPool, performance knobs
- [File Sync](./internals/file-sync.md) -- Merkle DAG change detection for incremental reindex
- [Resilience](./internals/resilience.md) -- Embedding cache, retry policy, timeouts, telemetry counters
- [Validation](./internals/validation.md) -- Request DTOs, filter expression grammar, ownership matrix
- [Ignore Policy](./internals/ignore-policy.md) -- Pattern normalization, matching semantics, .contextignore
- [mmap Safety](./internals/mmap-safety.md) -- Safety invariants for read-only memory mapping
- [Vector Snapshot Format](./internals/vector-snapshot-format.md) -- v2 bundle layout, integrity checks, DFRR sidecar
- [Vector Quantization](./internals/vector-quantization.md) -- SQ8 quantization and experimental u8 search path
- [Observability](./internals/observability.md) -- Structured JSON logs, telemetry, tracing, and sampling

## Research

Design explorations and analysis reports.

- [DFRR Mutation & Concurrency](./research/dfrr-mutation-concurrency.md) -- Static analysis of DFRR kernel mutation and concurrency model

## Project

- [Security](./security.md) -- Path policy, state directories, log redaction
- [Release & Install](./release.md) -- Artifacts, verification, install methods, maintainer notes
- [Changelog](../CHANGELOG.md) -- Version history
- [Contributing](../CONTRIBUTING.md) -- How to contribute
- [Security Policy](../SECURITY.md) -- Reporting vulnerabilities
- [Code of Conduct](../CODE_OF_CONDUCT.md) -- Community guidelines
- [License](../LICENSE) -- Dual MIT/Apache-2.0
