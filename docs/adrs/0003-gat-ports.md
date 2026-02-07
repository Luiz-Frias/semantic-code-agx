# ADR-0003: GAT-Based Port Traits

**Status**: Accepted
**Date**: 2025-01-16
**Deciders**: Core team

## Context

Port traits need to define abstract interfaces for:
- Embedders (multiple providers with different error types)
- Vector databases (different storage backends)
- File systems (real, mock, in-memory)

Challenges with traditional trait design:
1. **Associated error types** need to be implementation-specific
2. **Async methods** require careful lifetime handling
3. **Batching** requires flexible request/response types
4. **Testing** requires easy mock implementations

## Decision

Use **Generic Associated Types (GATs)** for port traits where flexibility is needed.

### Implementation

```rust
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Implementation-specific error type
    type Error: std::error::Error + Send + Sync + 'static;

    /// Generate embeddings for a batch of texts
    async fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>, Self::Error>;

    /// Embedding dimension (fixed per implementation)
    fn dimension(&self) -> usize;
}

#[async_trait]
pub trait VectorDb: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    async fn upsert(&self, chunks: Vec<ChunkWithEmbedding>) -> Result<(), Self::Error>;
    async fn search(&self, query: Embedding, top_k: usize) -> Result<Vec<SearchResult>, Self::Error>;
    async fn delete(&self, ids: &[ChunkId]) -> Result<(), Self::Error>;
}
```

### Usage in Use Cases

```rust
pub struct SemanticSearch<E, V>
where
    E: Embedder,
    V: VectorDb,
{
    embedder: E,
    vectordb: V,
}

impl<E: Embedder, V: VectorDb> SemanticSearch<E, V> {
    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>, SearchError>
    where
        SearchError: From<E::Error> + From<V::Error>,
    {
        let embedding = self.embedder.embed(&[query.to_string()]).await?;
        let results = self.vectordb.search(embedding[0].clone(), top_k).await?;
        Ok(results)
    }
}
```

### Why Not `Box<dyn Error>`?

```rust
// We avoided this pattern:
async fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>, Box<dyn Error>>;
```

Because:
- Loses type information for error handling
- Can't pattern match on specific errors
- Less efficient (heap allocation)

## Consequences

### Positive

- **Type safety**: Each adapter has its own error type
- **Zero-cost**: No dynamic dispatch or boxing required
- **Flexibility**: Adapters can define custom associated types
- **Testability**: Mock implementations with simple error types

### Negative

- **Complexity**: Generic bounds can become verbose
- **Compilation**: More monomorphization, longer compile times
- **Learning curve**: GATs are advanced Rust feature

### Risks

- **Trait coherence issues** with complex bounds
- Mitigated by: Keep trait definitions simple, use helper traits

## Alternatives Considered

### Dynamic Dispatch with `Box<dyn Trait>`

```rust
pub type DynEmbedder = Box<dyn Embedder<Error = anyhow::Error>>;
```

**Rejected because**: Requires standardizing on one error type, loses type information.

### Enum Error Types

```rust
enum EmbedderError {
    OpenAi(OpenAiError),
    Gemini(GeminiError),
    // ...
}
```

**Rejected because**: Port layer shouldn't know about specific implementations.

### anyhow/eyre for Errors

Use `anyhow::Error` throughout.

**Rejected because**: Loses ability to match on specific error variants in application layer.

## References

- [GAT Stabilization RFC](https://github.com/rust-lang/rust/issues/44265)
- [Async Traits in Rust](https://blog.rust-lang.org/inside-rust/2022/11/17/async-fn-in-trait-nightly.html)
- Ports implementation: `crates/ports/src/lib.rs`
