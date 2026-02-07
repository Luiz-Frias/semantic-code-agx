# Data Flow

This document describes the data flow through the system for key operations.

## Indexing Flow

### Overview

Indexing transforms source code files into searchable vector embeddings.

```
Source Files → Chunks → Embeddings → Vector Index
```

### Sequence Diagram

```
┌───────┐     ┌───────────┐     ┌────────────┐     ┌──────────┐     ┌──────────┐
│  CLI  │     │ IndexUse  │     │ FileSystem │     │ Splitter │     │ Embedder │
│       │     │   Case    │     │  Adapter   │     │ Adapter  │     │ Adapter  │
└───┬───┘     └─────┬─────┘     └──────┬─────┘     └────┬─────┘     └────┬─────┘
    │               │                  │                │                │
    │ index(path)   │                  │                │                │
    │──────────────>│                  │                │                │
    │               │                  │                │                │
    │               │ scan_files()     │                │                │
    │               │─────────────────>│                │                │
    │               │                  │                │                │
    │               │   Vec<FilePath>  │                │                │
    │               │<─────────────────│                │                │
    │               │                  │                │                │
    │               │ for each file:   │                │                │
    │               │ read_file()      │                │                │
    │               │─────────────────>│                │                │
    │               │   FileContent    │                │                │
    │               │<─────────────────│                │                │
    │               │                  │                │                │
    │               │ split(content)   │                │                │
    │               │──────────────────────────────────>│                │
    │               │   Vec<Chunk>     │                │                │
    │               │<──────────────────────────────────│                │
    │               │                  │                │                │
    │               │ embed(chunks)    │                │                │
    │               │─────────────────────────────────────────────────────>│
    │               │   Vec<Embedding> │                │                │
    │               │<─────────────────────────────────────────────────────│
    │               │                  │                │                │
    │               │                  │                │                │
    │               │                  │                │                │
    │               │                  │    ┌──────────┐
    │               │ upsert(embeddings)   │ VectorDb │
    │               │─────────────────────>│ Adapter  │
    │               │   Ok(())             └────┬─────┘
    │               │<─────────────────────────│
    │               │                          │
    │  IndexResult  │                          │
    │<──────────────│                          │
    │               │                          │
```

### Typestate Progression

The indexing pipeline uses typestate to enforce correct operation order:

```rust
IndexPipeline<Unindexed>
    │
    │ .scan_files(fs_adapter)
    ▼
IndexPipeline<Scanning>
    │
    │ .split_chunks(splitter_adapter)
    ▼
IndexPipeline<Chunking>
    │
    │ .generate_embeddings(embedder_adapter)
    ▼
IndexPipeline<Embedding>
    │
    │ .store_index(vectordb_adapter)
    ▼
IndexPipeline<Indexed>
```

Each transition is a method that consumes `self` and returns the next state. Invalid transitions are compile-time errors.

### Data Transformations

| Stage | Input | Output | Adapter |
|-------|-------|--------|---------|
| Scan | Directory path | List of file paths | FileSystem |
| Read | File path | File content (string) | FileSystem |
| Split | File content | Code chunks with metadata | Splitter |
| Embed | Code chunks | Embedding vectors | Embedder |
| Store | Embeddings + metadata | Persisted index | VectorDb |

### Chunk Structure

```rust
struct Chunk {
    id: ChunkId,           // Unique identifier
    content: String,       // Code text
    file_path: FilePath,   // Source file
    start_line: u32,       // Start line number
    end_line: u32,         // End line number
    language: Language,    // Detected language
    metadata: Metadata,    // Additional info
}
```

---

## Search Flow

### Overview

Search converts a natural language query into similar code chunks.

```
Query String → Query Embedding → Vector Search → Ranked Results
```

### Sequence Diagram

```
┌───────┐     ┌───────────┐     ┌──────────┐     ┌──────────┐
│  CLI  │     │ SearchUse │     │ Embedder │     │ VectorDb │
│       │     │   Case    │     │ Adapter  │     │ Adapter  │
└───┬───┘     └─────┬─────┘     └────┬─────┘     └────┬─────┘
    │               │                │                │
    │ search(query) │                │                │
    │──────────────>│                │                │
    │               │                │                │
    │               │ embed([query]) │                │
    │               │───────────────>│                │
    │               │                │                │
    │               │ Vec<Embedding> │                │
    │               │<───────────────│                │
    │               │                │                │
    │               │ search(embedding, top_k)       │
    │               │───────────────────────────────>│
    │               │                │                │
    │               │ Vec<SearchResult>              │
    │               │<───────────────────────────────│
    │               │                │                │
    │               │ rank_results() │                │
    │               │──────┐         │                │
    │               │      │ (internal)               │
    │               │<─────┘         │                │
    │               │                │                │
    │ SearchResults │                │                │
    │<──────────────│                │                │
    │               │                │                │
```

### Search Result Structure

```rust
struct SearchResult {
    chunk_id: ChunkId,      // Matching chunk
    score: f32,             // Similarity score (0.0 - 1.0)
    file_path: FilePath,    // Source file
    start_line: u32,        // Location
    end_line: u32,
    content: String,        // Code snippet
    metadata: Metadata,     // Additional info
}
```

### Similarity Scoring

The vector database uses **cosine similarity**:

```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Scores range from -1.0 (opposite) to 1.0 (identical). In practice, code embeddings typically score:
- **0.85-1.0**: Very similar (likely same concept)
- **0.70-0.85**: Related (similar domain)
- **0.50-0.70**: Loosely related
- **< 0.50**: Unlikely related

---

## Incremental Reindex Flow

### Overview

Incremental reindexing only processes changed files using Merkle tree comparison.

```
Current State → Merkle Diff → Changed Files → Re-embed → Update Index
```

### Sequence Diagram

```
┌───────┐     ┌───────────┐     ┌────────────┐     ┌──────────┐
│  CLI  │     │ Reindex   │     │ FileSync   │     │ VectorDb │
│       │     │ UseCase   │     │ Adapter    │     │ Adapter  │
└───┬───┘     └─────┬─────┘     └──────┬─────┘     └────┬─────┘
    │               │                  │                │
    │ reindex()     │                  │                │
    │──────────────>│                  │                │
    │               │                  │                │
    │               │ compute_merkle() │                │
    │               │─────────────────>│                │
    │               │                  │                │
    │               │ MerkleSnapshot   │                │
    │               │<─────────────────│                │
    │               │                  │                │
    │               │ load_previous_snapshot()         │
    │               │─────────────────>│                │
    │               │                  │                │
    │               │ diff(old, new)   │                │
    │               │─────────────────>│                │
    │               │                  │                │
    │               │ ChangedFiles {   │                │
    │               │   added,         │                │
    │               │   modified,      │                │
    │               │   deleted        │                │
    │               │ }                │                │
    │               │<─────────────────│                │
    │               │                  │                │
    │               │ (re-index changed files only)    │
    │               │ ...              │                │
    │               │                  │                │
    │               │ delete_chunks(deleted_files)     │
    │               │───────────────────────────────────>│
    │               │                  │                │
    │               │ upsert(new_embeddings)           │
    │               │───────────────────────────────────>│
    │               │                  │                │
    │               │ save_snapshot()  │                │
    │               │─────────────────>│                │
    │               │                  │                │
    │ ReindexResult │                  │                │
    │<──────────────│                  │                │
```

### Merkle Tree Structure

```
              Root Hash
             /         \
        Hash(L)       Hash(R)
        /    \        /    \
    H(a.rs) H(b.rs) H(c.rs) H(d.rs)
```

Each file is hashed. Directory hashes combine child hashes. When comparing snapshots:
- Different root hash → something changed
- Traverse to find changed subtrees
- Only process changed files

---

## Error Flow

### Overview

Errors flow upward through the call stack with context enrichment.

```
Adapter Error → Port Error → Use Case Error → API Error → User Message
```

### Error Enrichment

```rust
// Adapter layer: raw error
OpenAiError::RateLimit { retry_after: 60 }

// Port layer: contextualized
EmbedderError::ProviderError {
    provider: "openai",
    cause: Box::new(openai_error),
}

// Use case layer: user-facing
IndexError::EmbeddingFailed {
    file: "src/main.rs",
    cause: Box::new(embedder_error),
}

// API layer: HTTP response
{
    "error": {
        "code": "EMBEDDING_FAILED",
        "message": "Failed to generate embedding for src/main.rs",
        "details": { ... }
    },
    "status": 500
}
```

### Secret Redaction

Sensitive data is automatically redacted in logs:

```rust
// Input
error!("Failed to authenticate: api_key={}", api_key);

// Output (redacted)
error!("Failed to authenticate: api_key=[REDACTED]");
```

---

## Configuration Flow

### Overview

Configuration is loaded from multiple sources with precedence.

```
Defaults → Config File → Environment Variables → CLI Arguments
```

### Loading Sequence

```
┌─────────────────┐
│  Default values │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ .context/config.toml    │
│ (or specified config)   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Environment variables   │
│ (SEMANTIC_*, API keys)  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ CLI arguments           │
│ (--port, --config, etc) │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Validated Configuration │
└─────────────────────────┘
```

Later sources override earlier ones. Environment variables support `${VAR}` expansion in config files:

```toml
[backend.embeddings]
openai_api_key = "${OPENAI_API_KEY}"
```

---

## Summary

| Flow | Key Operations | Adapters Used |
|------|----------------|---------------|
| **Index** | Scan → Split → Embed → Store | FileSystem, Splitter, Embedder, VectorDb |
| **Search** | Embed query → Vector search → Rank | Embedder, VectorDb |
| **Reindex** | Merkle diff → Index changes | FileSync, all indexing adapters |
| **Error** | Enrich context → Redact secrets → Format | N/A |
| **Config** | Load sources → Merge → Validate | N/A |
