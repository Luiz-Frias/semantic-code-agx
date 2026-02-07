# Crate Map

This document provides a visual overview of all crates in the workspace and their dependencies.

## Workspace Structure

```
semantic-code-agx/
├── crates/
│   ├── adapters/      # Concrete implementations (embeddings, vectordb, fs)
│   ├── api/           # REST API types and validation
│   ├── app/           # Application use cases
│   ├── config/        # Configuration parsing and validation
│   ├── core/          # Re-exports for convenience
│   ├── domain/        # Core domain types and primitives
│   ├── facade/        # Simplified high-level API
│   ├── infra/         # CLI, factories, wiring
│   ├── ports/         # Abstract trait definitions
│   ├── shared/        # Cross-cutting utilities
│   ├── validate-derive/# Proc macro for validation
│   └── vector/        # Vector operations (HNSW, similarity)
```

## Dependency Graph

```
                           ┌──────────────┐
                           │    infra     │  CLI, factories
                           │   (binary)   │
                           └──────┬───────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
    ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
    │    facade     │     │     app       │     │    config     │
    │  (high-level) │     │  (use cases)  │     │   (parsing)   │
    └───────┬───────┘     └───────┬───────┘     └───────┬───────┘
            │                     │                     │
            │              ┌──────┴──────┐              │
            │              │             │              │
            ▼              ▼             ▼              ▼
    ┌───────────────┐  ┌───────────┐  ┌───────────┐
    │   adapters    │  │   ports   │  │    api    │
    │ (implementations)│  │ (traits)  │  │  (DTOs)   │
    └───────┬───────┘  └─────┬─────┘  └─────┬─────┘
            │                │              │
            │          ┌─────┴──────────────┘
            │          │
            ▼          ▼
        ┌──────────────────┐
        │      domain      │
        │  (core types)    │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐     ┌──────────────────┐
        │      shared      │     │      vector      │
        │   (utilities)    │     │   (HNSW, math)   │
        └──────────────────┘     └──────────────────┘

        ┌──────────────────┐
        │  validate-derive │
        │   (proc macro)   │
        └──────────────────┘
```

## Crate Details

### `domain` - Core Domain Types

**Purpose**: Define the ubiquitous language and core types.

**Key types**:
- `ChunkId`, `CodebaseId`, `FileId` - Identity types
- `Embedding` - Vector representation
- `SearchResult`, `Metadata` - Query results
- `IndexState` - Pipeline state machine

**Dependencies**: None (pure domain)

**Dependents**: All other crates

---

### `ports` - Abstract Interfaces

**Purpose**: Define what the system can do without specifying how.

**Key traits**:
- `Embedder` - Generate embeddings from text
- `VectorDb` - Store and search vectors
- `FileSystem` - Read files from disk
- `CodeSplitter` - Split code into chunks

**Dependencies**: `domain`

**Dependents**: `app`, `adapters`, `facade`

---

### `adapters` - Concrete Implementations

**Purpose**: Implement ports for specific technologies.

**Modules**:
- `embedding/` - OpenAI, Gemini, Voyage, Ollama, ONNX
- `vectordb/` - Local HNSW, Milvus (gRPC + REST)
- `fs.rs` - Standard filesystem
- `splitter.rs` - Tree-sitter code splitter
- `ignore.rs` - .contextignore support
- `cache.rs` - Embedding cache
- `file_sync.rs` - Merkle-based sync

**Dependencies**: `ports`, `domain`, `shared`, `vector`

**Dependents**: `facade`, `infra`

---

### `app` - Application Use Cases

**Purpose**: Orchestrate domain logic through ports.

**Use cases**:
- `IndexCodebase` - Index source files
- `SemanticSearch` - Search by meaning
- `ReindexByChange` - Incremental updates
- `ClearIndex` - Remove index data

**Dependencies**: `ports`, `domain`

**Dependents**: `facade`, `infra`

---

### `config` - Configuration

**Purpose**: Parse and validate configuration from files and environment.

**Key types**:
- `BackendConfig` - Main configuration
- `EmbeddingsConfig` - Embedding provider settings
- `MilvusConfig` - Milvus connection settings

**Features**:
- TOML file parsing
- Environment variable expansion
- Validation with helpful errors

**Dependencies**: `domain`, `shared`

**Dependents**: `infra`, `facade`

---

### `api` - REST API Types

**Purpose**: Define API contracts (DTOs, validation, mapping).

**Modules**:
- `v1/schema.rs` - Request/response types
- `v1/validation.rs` - Input validation
- `v1/mappers.rs` - Domain ↔ API conversion

**Dependencies**: `domain`

**Dependents**: `infra`

---

### `infra` - Infrastructure

**Purpose**: CLI, factories, and application wiring.

**Modules**:
- `cli_local.rs` - Local index commands
- `cli_manifest.rs` - Manifest commands
- `embedding_factory.rs` - Create embedders
- `vectordb_factory.rs` - Create vector DBs
- `config_check.rs`, `env_check.rs`, `request_check.rs` - Validation commands

**Dependencies**: `app`, `adapters`, `config`, `api`, `domain`

**Dependents**: Binary entry point

---

### `facade` - Simplified API

**Purpose**: Provide a high-level API hiding internal complexity.

**Key types**:
- `SemanticCodeSearch` - Main entry point

**Dependencies**: `app`, `adapters`, `config`

**Dependents**: External consumers

---

### `shared` - Utilities

**Purpose**: Cross-cutting concerns used by multiple crates.

**Modules**:
- `errors.rs` - Error types and handling
- `result.rs` - Result aliases
- `redaction.rs` - Secret redaction
- `retry.rs` - Retry logic
- `timeout.rs` - Timeout handling
- `concurrency.rs` - Async utilities
- `merkle.rs` - Merkle tree for change detection
- `validation.rs` - Validation helpers
- `invariants.rs` - Runtime invariant checks

**Dependencies**: None

**Dependents**: Most crates

---

### `vector` - Vector Operations

**Purpose**: Low-level vector math and indexing.

**Features**:
- HNSW index implementation
- Cosine similarity
- Persistence to disk

**Dependencies**: None

**Dependents**: `adapters`

---

### `validate-derive` - Proc Macro

**Purpose**: Derive macro for automatic validation.

**Usage**:
```rust
#[derive(Validate)]
pub struct ChunkId {
    #[validate(non_empty)]
    value: String,
}
```

**Dependencies**: `syn`, `quote`, `proc-macro2`

**Dependents**: `domain`

---

### `core` - Re-exports

**Purpose**: Convenience re-exports for common use.

**Dependencies**: `domain`, `ports`, `app`

**Dependents**: External consumers

## Feature Flags

### `adapters` crate features

| Feature | Description | Default |
|---------|-------------|---------|
| `onnx` | Local ONNX embeddings | ✅ |
| `openai` | OpenAI API embeddings | ✅ |
| `gemini` | Google Gemini embeddings | ✅ |
| `voyage` | Voyage AI embeddings | ✅ |
| `ollama` | Ollama local embeddings | ✅ |
| `milvus-grpc` | Milvus gRPC transport | ✅ |
| `milvus-rest` | Milvus REST transport | ✅ |

## Build Order

The workspace builds in dependency order:

1. `shared`, `vector`, `validate-derive` (no deps)
2. `domain` (depends on validate-derive)
3. `ports`, `api`, `config` (depend on domain)
4. `adapters` (depends on ports)
5. `app` (depends on ports)
6. `facade` (depends on app, adapters)
7. `infra` (depends on everything)

## Adding a New Crate

1. Create directory under `crates/`
2. Add `Cargo.toml` with appropriate dependencies
3. Add to workspace in root `Cargo.toml`
4. Follow the dependency rules (no cycles, inward dependencies)
5. Update this document

## Dependency Rules

### ✅ Allowed

- Higher layers can depend on lower layers
- All crates can depend on `shared`

### ❌ Forbidden

- `domain` cannot depend on any crate except `shared` and `validate-derive`
- `ports` cannot depend on `adapters` (abstraction over implementation)
- `app` cannot depend on `adapters` (use cases are adapter-agnostic)
- No circular dependencies
