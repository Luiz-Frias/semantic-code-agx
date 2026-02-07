# ADR-0004: Owned Request/Response Types

**Status**: Accepted
**Date**: 2025-01-16
**Deciders**: Core team

## Context

API boundaries (CLI input, REST requests, adapter calls) need to handle data ownership.

Options for data passing:
1. **References with lifetimes**: `fn search(&self, query: &str) -> &[Result]`
2. **Owned values**: `fn search(&self, query: String) -> Vec<Result>`
3. **Cow/borrowed-or-owned**: `fn search(&self, query: Cow<str>) -> Cow<[Result]>`

Challenges:
- Lifetime annotations become complex across async boundaries
- Serialization/deserialization naturally produces owned data
- Cloning at boundaries adds overhead

## Decision

Use **owned request/response types** at all API boundaries.

### Implementation

```rust
// Request types are owned
pub struct SearchRequest {
    pub query: String,           // Owned, not &str
    pub codebase_id: CodebaseId, // Owned
    pub top_k: usize,
}

// Response types are owned
pub struct SearchResponse {
    pub results: Vec<SearchResult>,  // Owned Vec
    pub total_count: usize,
}

// Port methods take ownership
#[async_trait]
pub trait VectorDb {
    async fn search(&self, request: SearchRequest) -> Result<SearchResponse, Self::Error>;
}
```

### Rationale for Owned Types

1. **Async compatibility**: No lifetime issues across await points
2. **Serialization**: JSON/TOML naturally deserialize to owned types
3. **Simplicity**: No complex lifetime annotations
4. **Flexibility**: Callers can move or clone as needed

### When References Are Acceptable

Internal methods within a crate can use references when:
- No async boundaries crossed
- Performance critical hot paths
- Lifetime is clearly bounded

```rust
// Internal helper - references OK
fn normalize_text(text: &str) -> &str {
    text.trim()
}

// Public API - owned types
pub fn process_text(text: String) -> ProcessedText {
    let normalized = normalize_text(&text);
    ProcessedText { value: normalized.to_string() }
}
```

## Consequences

### Positive

- **Simplicity**: No lifetime annotations at boundaries
- **Async-friendly**: Owned data crosses await points freely
- **Predictable**: Ownership is clear at API boundaries
- **Serialization**: Natural fit for JSON/TOML

### Negative

- **Memory**: More allocations than zero-copy approaches
- **Performance**: Clone cost at boundaries (usually negligible)

### Trade-offs

Memory vs. simplicity trade-off is acceptable because:
- Request/response data is typically small (< 1KB)
- Boundaries are crossed infrequently (once per API call)
- Developer productivity from simpler code outweighs micro-optimizations

## Alternatives Considered

### Reference-Heavy Design

```rust
pub struct SearchRequest<'a> {
    pub query: &'a str,
    pub codebase_id: &'a CodebaseId,
}
```

**Rejected because**:
- Lifetime annotations proliferate through call stack
- Complex interactions with async/await
- Harder for contributors to understand

### Cow (Copy-on-Write)

```rust
pub struct SearchRequest<'a> {
    pub query: Cow<'a, str>,
}
```

**Rejected because**:
- Adds complexity without clear benefit
- Still requires lifetime parameter
- API becomes harder to use

### Arc-Based Sharing

```rust
pub struct SearchRequest {
    pub query: Arc<str>,
}
```

**Rejected because**:
- Atomic reference counting overhead
- Unclear ownership semantics
- Overkill for request/response patterns

## Measurements

Micro-benchmarks showed:
- Clone cost for typical request: ~50ns
- String allocation: ~100ns
- Network latency: ~10-100ms

Conclusion: Allocation cost is < 0.001% of total request time.

## References

- [Rust API Guidelines: Ownership](https://rust-lang.github.io/api-guidelines/flexibility.html)
- [Async Rust Patterns](https://rust-lang.github.io/async-book/)
- Related: [ADR-0003: GAT Ports](./0003-gat-ports.md)
