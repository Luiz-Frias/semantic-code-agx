# ADR-0001: Hexagonal Architecture

**Status**: Accepted
**Date**: 2025-01-14
**Deciders**: Core team

## Context

We need an architecture that:
1. Supports multiple embedding providers (OpenAI, Gemini, Voyage, Ollama, ONNX)
2. Supports multiple vector databases (local, Milvus)
3. Is testable without external dependencies
4. Allows adding new adapters without modifying business logic
5. Maintains clear separation of concerns

## Decision

Adopt **hexagonal architecture** (ports and adapters) with the following structure:

```
domain → ports → adapters
              ↘ app (use cases)
```

### Crate Organization

| Crate | Layer | Responsibility |
|-------|-------|----------------|
| `domain` | Core | Entity types, value objects, validation |
| `ports` | Core | Abstract trait definitions |
| `app` | Application | Use case orchestration |
| `adapters` | Infrastructure | Concrete implementations |
| `infra` | Infrastructure | CLI, wiring, factories |

### Dependency Rules

- Domain has no external dependencies
- Ports depend only on domain
- Adapters implement ports, depend on domain
- App depends on ports (not adapters)
- Infra wires everything together

## Consequences

### Positive

- **Testability**: Use cases tested with mock adapters
- **Flexibility**: Add providers without changing core logic
- **Maintainability**: Clear boundaries between concerns
- **Team scaling**: Teams can work on different adapters independently

### Negative

- **Complexity**: More files and indirection than simple architecture
- **Learning curve**: Contributors must understand the pattern
- **Boilerplate**: Trait definitions for every adapter type

### Risks

- Over-abstraction if too many port traits created
- Mitigated by: Only abstract what varies (embedders, vector DBs, file systems)

## Alternatives Considered

### Layered Architecture

Traditional layers (presentation → business → data).

**Rejected because**: Tight coupling between layers, harder to swap implementations.

### Microservices

Separate services for indexing, search, embeddings.

**Rejected because**: Premature complexity for current scale. Can evolve later if needed.

### Simple Module Structure

Flat module organization without formal architecture.

**Rejected because**: Would become unmaintainable as adapter count grows.

## References

- [Alistair Cockburn's Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)
- [Architecture documentation](../architecture/hexagonal.md)
