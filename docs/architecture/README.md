# Architecture Overview

semantic-code-agx is designed as a **hexagonal (ports and adapters) modular monolith** optimized for extensibility, determinism, and clean separation of concerns.

## High-Level Architecture

```
+-----------------------------------------------------------------+
|                         CLI / API Layer                          |
|               (infra crate, API types only)                     |
+-----------------------------------------------------------------+
                                 |
                                 v
+-----------------------------------------------------------------+
|                      Application Layer                          |
|                        (app crate)                              |
|  +-----------+  +-----------+  +---------------------------+   |
|  |  Index    |  |  Search   |  |  Reindex / Clear          |   |
|  |  Codebase |  |  Codebase |  |  Index                    |   |
|  +-----------+  +-----------+  +---------------------------+   |
+-----------------------------------------------------------------+
                                 |
                                 v
+-----------------------------------------------------------------+
|                         Port Traits                             |
|                       (ports crate)                             |
|  +----------+  +----------+  +----------+  +--------------+    |
|  | Embedder |  | VectorDb |  | FileSys  |  | CodeSplitter |    |
|  +----------+  +----------+  +----------+  +--------------+    |
+-----------------------------------------------------------------+
                                 |
                                 v
+-----------------------------------------------------------------+
|                        Adapter Layer                            |
|                      (adapters crate)                           |
|  +-----------------------------------------------------------+ |
|  | Embeddings: ONNX, OpenAI, Gemini, Voyage, Ollama          | |
|  | VectorDB: Local (HNSW), Milvus (gRPC/REST)                | |
|  | FileSystem: Standard FS with ignore patterns               | |
|  | Splitter: tree-sitter (Rust/Go/Java/JS/TS/TSX/Python/C/C++)| |
|  |          + line-based fallback for other files              | |
|  +-----------------------------------------------------------+ |
+-----------------------------------------------------------------+
                                 |
                                 v
+-----------------------------------------------------------------+
|                         Domain Layer                            |
|                       (domain crate)                            |
|  +----------------------------------------------------------+  |
|  | ChunkId, CodebaseId, Embedding, SearchResult, Metadata    |  |
|  | Parse constructors with validation, Newtype wrappers      |  |
|  +----------------------------------------------------------+  |
+-----------------------------------------------------------------+
```

## Design Principles

1. **Hexagonal Architecture** -- Domain logic stays pure and independent of storage or embedding providers. See [hexagonal.md](./hexagonal.md).

2. **Type-Driven Design** -- Newtype wrappers, parse constructors, and derive macros enforce invariants at compile time. Invalid states are unrepresentable.

3. **Typestate Pattern** -- The indexing pipeline enforces correct operation order via the type system. See [ADR-0002](../adrs/0002-typestate-indexing.md).

4. **Port Traits at the Boundary** -- Boxed futures for async boundaries with clear ownership and consistent error propagation.

5. **Typed Error Handling** -- All errors are strongly typed enums with context, secret redaction, and clean mapping to CLI exit codes and API surfaces.

## Crate Organization

| Crate | Purpose | Dependencies |
|-------|---------|--------------|
| `domain` | Core types, validation | None (pure) |
| `ports` | Abstract trait definitions | `domain` |
| `adapters` | Concrete implementations | `ports`, `domain` |
| `app` | Use cases (index, search) | `ports`, `domain` |
| `config` | Configuration parsing | `domain` |
| `infra` | CLI, factories | All |
| `api` | REST API types | `domain` |
| `shared` | Cross-cutting utilities | None |
| `facade` | Simplified API | `app`, `adapters` |
| `vector` | HNSW kernel, mmap, snapshots | None |
| `dfrr_hnsw` | DFRR search kernel (BSL) | `vector` |
| `core` | Build-time metadata | None |
| `validate-derive` | Proc macro for validation | None |
| `testkit` | In-memory test doubles | Relaxed lints |

See [Crate Map](./crate-map.md) for the full dependency graph.

## Data Flow

```
Index:  Scan files -> Split chunks (tree-sitter) -> Embed (batch) -> Insert vector DB
Search: Embed query -> Vector similarity search -> Rank/filter -> Return results
Reindex: Merkle-based change detection -> Selective re-embed -> Upsert
```

See [Data Flow](./data-flow.md) for detailed sequence diagrams.

## Key Architectural Decisions

| Decision | Rationale | ADR |
|----------|-----------|-----|
| Hexagonal architecture | Extensibility, isolation | [ADR-0001](../adrs/0001-hexagonal-architecture.md) |
| Typestate for indexing | Compile-time correctness | [ADR-0002](../adrs/0002-typestate-indexing.md) |
| GAT-based port traits | Type-safe async boundaries | [ADR-0003](../adrs/0003-gat-ports.md) |
| Owned requests | Simplicity, ownership clarity | [ADR-0004](../adrs/0004-owned-requests.md) |

## Architecture Documents

| Document | Description |
|----------|-------------|
| [Hexagonal Architecture](./hexagonal.md) | Ports and adapters pattern deep-dive |
| [Crate Map](./crate-map.md) | Workspace structure and dependency graph |
| [Data Flow](./data-flow.md) | Request lifecycle and sequence diagrams |
| [Collection Loader Actor](./collection-loader-actor.md) | Actor-based collection lifecycle management |
| [DFRR Design](./dfrr-design.md) | Incremental maintenance and curvature-guided algorithms |
| [Advanced Patterns](./patterns.md) | FSM, GAT, and typestate refactoring opportunities |

## Extension Points

### Adding a New Embedding Provider

1. Implement `Embedder` trait in `adapters/src/embedding/`
2. Add configuration variant in `config/src/schema.rs`
3. Register in factory (`infra/src/embedding_factory.rs`)

### Adding a New Vector Database

1. Implement `VectorDb` trait in `adapters/src/vectordb/`
2. Add configuration variant
3. Register in factory

### Adding a New File Type

1. Add tree-sitter grammar to `splitter` adapter
2. Update language detection
