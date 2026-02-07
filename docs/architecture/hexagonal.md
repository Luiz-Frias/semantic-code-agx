# Hexagonal Architecture

This document explains the hexagonal (ports and adapters) architecture pattern as implemented in semantic-code-agx.

## What is Hexagonal Architecture?

Hexagonal architecture (also known as Ports and Adapters) isolates the application's core logic from external concerns like databases, APIs, and user interfaces.

```
                    ┌─────────────────────────┐
                    │     Primary Adapters    │
                    │   (CLI, API types)      │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │      Primary Ports      │
                    │   (Use Case interfaces) │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────▼─────────────────┐
              │                                   │
              │         Application Core          │
              │                                   │
              │   ┌───────────────────────────┐   │
              │   │      Domain Model         │   │
              │   │   (Entities, Value Objs)  │   │
              │   └───────────────────────────┘   │
              │                                   │
              │   ┌───────────────────────────┐   │
              │   │       Use Cases           │   │
              │   │  (Application Services)   │   │
              │   └───────────────────────────┘   │
              │                                   │
              └─────────────────┬─────────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │    Secondary Ports      │
                    │  (Repository, Gateway)  │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Secondary Adapters    │
                    │  (DB, API clients, FS)  │
                    └─────────────────────────┘
```

## Why Hexagonal Here

- **Core isolation**: indexing and search logic stays pure and independent of storage or embedding providers.
- **Adapter substitution**: local HNSW vs Milvus, and ONNX vs cloud embeddings are selectable without modifying use cases.
- **Strict dependency direction**: outer layers depend inward, keeping the architecture stable as the system grows.

## Implementation in This Codebase

### Domain Layer (`crates/domain`)

- Core types and validation (e.g., identifiers, metadata, chunk spans).
- No external I/O dependencies.

### Port Layer (`crates/ports`)

- Boundary contracts for embedding, vector search, filesystem, splitter, logging, and telemetry.
- Async methods use boxed futures for predictable boundaries and ownership.

### Adapter Layer (`crates/adapters`)

- Embeddings: ONNX, OpenAI, Gemini, Voyage, Ollama.
- Vector DB: Local HNSW, Milvus (gRPC/REST).
- File system + ignore rules + tree-sitter splitter with line-based fallback.

### Application Layer (`crates/app`)

- Use cases: index, search, reindex, clear.
- Orchestrates ports, logging, and telemetry.

### Infrastructure Layer (`crates/infra`)

- Composition root and factories.
- CLI wiring and local command orchestration.

### API Types (`crates/api`)

- API request/response DTOs and validation.
- Types only (no server surface in this repo).

## Dependency Rules

### ✅ Allowed

```
infra → app → ports → domain
infra → adapters → ports → domain
infra → config → domain
```

### ❌ Not Allowed

```
domain → anything external
ports → adapters (ports are abstract)
app → adapters (use cases are adapter-agnostic)
```

## Benefits Realized

### Easy Adapter Swapping

Adding a new vector DB or embedding provider requires:

1. A new adapter implementation
2. A config entry for selection
3. A factory registration

Domain and use-case logic remain unchanged.

### Clear Boundaries

Each crate has a focused responsibility:

- `domain`: what things are
- `ports`: what the system can do (abstract)
- `adapters`: how it does it (concrete)
- `app`: workflows and orchestration
- `infra`: composition and CLI glue

## Trade-offs

Hexagonal design introduces more files and indirection than a monolith. That cost is justified for:

- Long-lived CLI tooling
- Multiple provider backends
- A growing set of integration surfaces
