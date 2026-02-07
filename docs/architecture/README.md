# Architecture Overview

semantic-code-agx is designed as a **hexagonal (ports and adapters) modular monolith** optimized for extensibility, determinism, and clean separation of concerns.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI / API Layer                          │
│               (infra crate, API types only)                      │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                           │
│                        (app crate)                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Index     │  │   Search    │  │   Reindex / Clear       │  │
│  │  Codebase   │  │  Codebase   │  │   Index                 │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Port Traits                              │
│                       (ports crate)                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────┐  │
│  │ Embedder │  │ VectorDb │  │ FileSys  │  │ CodeSplitter    │  │
│  └──────────┘  └──────────┘  └──────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Adapter Layer                             │
│                      (adapters crate)                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Embeddings: ONNX, OpenAI, Gemini, Voyage, Ollama          │  │
│  │ VectorDB: Local (HNSW), Milvus (gRPC/REST)                │  │
│  │ FileSystem: Standard FS with ignore patterns              │  │
│  │ Splitter: Tree-sitter (Rust/Go/Java/JS/TS/TSX/Python/C/C++)│  │
│  │          + line-based fallback for other files             │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Domain Layer                             │
│                       (domain crate)                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ChunkId, CodebaseId, Embedding, SearchResult, Metadata   │   │
│  │ Parse constructors with validation, Newtype wrappers     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Hexagonal Architecture (Ports & Adapters)

The system separates:
- **Domain logic** (pure business rules, no I/O)
- **Ports** (abstract interfaces/traits)
- **Adapters** (concrete implementations)

This enables:
- Swapping implementations without changing business logic
- Clear dependency direction (inward)

See [Hexagonal Architecture](./hexagonal.md) for details.

### 2. Type-Driven Design

All domain types use:
- **Newtype wrappers** (e.g., `ChunkId(String)` not raw `String`)
- **Parse constructors** that validate on construction
- **Derive macros** for consistent validation

Invalid states are unrepresentable at compile time.

### 3. Typestate Pattern

The indexing pipeline uses typestate to enforce correct operation order:

```rust
// Can only transition: Prepared → Scanned → Embedded → Inserted → Completed
IndexPipeline<Prepared> → IndexPipeline<Scanned> → IndexPipeline<Embedded> → ...
```

This prevents runtime errors from operations in wrong order.

### 4. Port Traits at the Boundary

Port traits use boxed futures for I/O boundaries:
- Predictable async signatures
- Clear ownership at crate boundaries
- Consistent error propagation

### 5. Error Handling

All errors:
- Are strongly typed (`enum` with variants)
- Carry context (what failed, why, where)
- Support secret redaction in logs
- Map cleanly to CLI exit codes (and future API surfaces)

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
See [Crate Map](./crate-map.md) for detailed dependency graph.

## Data Flow

### Indexing Flow

```
User Request → CLI/API → IndexCodebase Use Case
                              │
                              ▼
                    ┌─────────────────┐
                    │  Scan Files     │ ← FileSystem Adapter
                    └────────┬────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Split Chunks   │ ← Splitter Adapter
                    └────────┬────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Generate Embeds │ ← Embedder Adapter
                    └────────┬────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Store Vectors  │ ← VectorDb Adapter
                    └─────────────────┘
```

### Search Flow

```
User Query → CLI/API → SemanticSearch Use Case
                              │
                              ▼
                    ┌─────────────────┐
                    │ Embed Query     │ ← Embedder Adapter
                    └────────┬────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Vector Search   │ ← VectorDb Adapter
                    └────────┬────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Rank & Return   │
                    └─────────────────┘
```

See [Data Flow](./data-flow.md) for detailed sequence diagrams.

## Key Architectural Decisions

The following decisions shaped the architecture:

| Decision | Rationale | ADR |
|----------|-----------|-----|
| Hexagonal architecture | Extensibility, isolation | [ADR-001](../adrs/001-hexagonal-architecture.md) |
| Typestate for indexing | Compile-time correctness | [ADR-002](../adrs/002-typestate-indexing.md) |
| Owned requests | Simplicity, ownership clarity | [ADR-004](../adrs/004-owned-requests.md) |

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

## Performance Characteristics

- **Indexing**: O(n) files, parallelized
- **Search**: O(log n) vector similarity (HNSW)
- **Memory**: Configurable (local vs. Milvus)

## Further Reading

- [Hexagonal Architecture](./hexagonal.md) - Pattern deep-dive
- [Crate Map](./crate-map.md) - Dependency visualization
- [Data Flow](./data-flow.md) - Sequence diagrams
- [ADRs](../adrs/README.md) - Decision rationale
