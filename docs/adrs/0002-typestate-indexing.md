# ADR-0002: Typestate Pattern for Indexing Pipeline

**Status**: Accepted
**Date**: 2025-01-16
**Deciders**: Core team

## Context

The indexing pipeline has distinct phases that must execute in order:

1. **Unindexed** → Scan files from filesystem
2. **Scanning** → Split files into chunks
3. **Chunking** → Generate embeddings
4. **Embedding** → Store in vector database
5. **Indexed** → Complete

Runtime errors occurred when:
- Attempting to search before indexing
- Calling embedding before splitting
- Skipping pipeline stages

We need compile-time guarantees that operations happen in the correct order.

## Decision

Use the **typestate pattern** to encode pipeline state in the type system.

### Implementation

```rust
// State markers (zero-sized types)
pub struct Unindexed;
pub struct Scanning;
pub struct Chunking;
pub struct Embedding;
pub struct Indexed;

// Pipeline with state parameter
pub struct IndexPipeline<S> {
    state: PhantomData<S>,
    // ... data fields
}

// Transitions consume self and return next state
impl IndexPipeline<Unindexed> {
    pub async fn scan_files(self, fs: &impl FileSystem) -> Result<IndexPipeline<Scanning>, ScanError> {
        // ...
    }
}

impl IndexPipeline<Scanning> {
    pub async fn split_chunks(self, splitter: &impl CodeSplitter) -> Result<IndexPipeline<Chunking>, SplitError> {
        // ...
    }
}

// Methods only available in specific states
impl IndexPipeline<Indexed> {
    pub fn chunk_count(&self) -> usize { /* ... */ }
}
```

### Usage

```rust
// Compile-time enforced sequence
let pipeline = IndexPipeline::new()
    .scan_files(&fs).await?       // Unindexed → Scanning
    .split_chunks(&splitter).await?  // Scanning → Chunking
    .generate_embeddings(&embedder).await?  // Chunking → Embedding
    .store_index(&vectordb).await?;  // Embedding → Indexed

// This would NOT compile:
// let pipeline = IndexPipeline::new()
//     .generate_embeddings(&embedder).await?;  // Error: no method on Unindexed
```

## Consequences

### Positive

- **Compile-time safety**: Invalid sequences are compiler errors
- **Self-documenting**: Type signature shows valid operations
- **No runtime checks**: Zero overhead from state validation
- **Clear API**: Users can't misuse the pipeline

### Negative

- **Verbosity**: Each state needs trait implementations
- **Learning curve**: Pattern less familiar to some developers
- **Inflexibility**: Adding states requires code changes

### Risks

- **Complex state machines** become unwieldy
- Mitigated by: Keep states minimal (5 states is manageable)

## Alternatives Considered

### Runtime State Enum

```rust
enum PipelineState { Unindexed, Scanning, ... }
```

**Rejected because**: Requires runtime checks, errors at runtime not compile time.

### Builder Pattern

```rust
PipelineBuilder::new().with_files().with_chunks().build()
```

**Rejected because**: Doesn't prevent out-of-order calls, all methods available at all times.

### Session Types

Full session type implementation with continuations.

**Rejected because**: Too complex for this use case. Simple typestate sufficient.

## References

- [Typestate Pattern in Rust](https://cliffle.com/blog/rust-typestate/)
- [Architecture: Data Flow](../architecture/data-flow.md)
- Ports crate implementation: `crates/ports/src/lib.rs`
