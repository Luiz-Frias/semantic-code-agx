# Searching Guide

This guide covers how to perform semantic searches on your indexed codebase.

## Overview

Semantic search finds code by meaning, not just keywords:

```
"error handling" → finds try/catch blocks, Result types, error utilities
"database connection" → finds DB setup code, connection pooling, queries
```

## Quick Start

```bash
# Search indexed codebase
semantic-code-agents search "error handling"

# Limit results
semantic-code-agents search "authentication logic" --top-k 10
```

## How Semantic Search Works

1. **Embed query** - Convert your question to a vector
2. **Vector search** - Find similar vectors in the index
3. **Rank results** - Order by similarity score
4. **Return matches** - Show relevant code chunks

## CLI Usage

### Basic Search

```bash
semantic-code-agents search "your query here"
```

### Options

```bash
# Limit number of results (default: 5)
semantic-code-agents search "query" --top-k 10

# Search specific codebase (if multiple indexed)
semantic-code-agents search "query" --codebase my-project

# JSON output (for scripting)
semantic-code-agents search "query" --json

# Verbose output with scores
semantic-code-agents search "query" --verbose
```

### Output Format

```
Found 5 results:

1. src/error.rs:24-45 (score: 0.92)
   Custom error types and error handling utilities
   │ pub enum AppError {
   │     NotFound(String),
   │     Validation(ValidationError),
   │     ...
   │ }

2. src/handlers/auth.rs:89-112 (score: 0.87)
   Authentication error handling
   │ fn handle_auth_error(err: AuthError) -> Response {
   │     ...
   │ }
```

## Writing Effective Queries

### Good Queries

| Query | Why It Works |
|-------|--------------|
| "user authentication with JWT tokens" | Specific, domain-aware |
| "database connection pooling" | Technical, clear intent |
| "error handling and recovery" | Conceptual, finds patterns |
| "API rate limiting implementation" | Feature-focused |

### Poor Queries

| Query | Why It Fails |
|-------|--------------|
| "stuff" | Too vague |
| "the thing that does the thing" | Meaningless |
| "foo" | Identifier-specific (use grep) |
| "class User" | Exact match (use grep) |

### Tips

1. **Be specific** - "user session management" > "sessions"
2. **Use domain terms** - "authentication" > "checking users"
3. **Describe behavior** - "retry failed requests" > "retry"
4. **Think conceptually** - "caching strategy" finds cache implementations

## Understanding Results

### Similarity Scores

| Score | Meaning |
|-------|---------|
| 0.90+ | Very relevant (likely exact match) |
| 0.80-0.90 | Highly relevant |
| 0.70-0.80 | Related |
| 0.60-0.70 | Loosely related |
| < 0.60 | Probably not relevant |

### Result Ordering

Results are ordered by:
1. Similarity score (descending)
2. File path (ascending, for ties)
3. Line number (ascending, for ties)

This ensures deterministic, reproducible results.

## API Usage

### REST API

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "error handling",
    "codebase_id": "my-project",
    "top_k": 5
  }'
```

### Response

```json
{
  "results": [
    {
      "chunk_id": "abc123",
      "score": 0.92,
      "file_path": "src/error.rs",
      "start_line": 24,
      "end_line": 45,
      "content": "pub enum AppError { ... }",
      "metadata": {
        "language": "rust"
      }
    }
  ],
  "total_count": 5
}
```

See [API Reference](../reference/api-v1.md) for full details.

## Advanced Searching

### Filtering by Codebase

If you've indexed multiple projects:

```bash
# Search only in backend code
semantic-code-agents search "API endpoint" --codebase backend

# Search only in frontend code
semantic-code-agents search "React component" --codebase frontend
```

### Combining with grep

For best of both worlds:

```bash
# Semantic: find relevant files
semantic-code-agents search "user authentication" --json | jq -r '.results[].file_path'

# Then grep for specific identifiers
rg "AuthService" $(semantic-code-agents search "authentication" --json | jq -r '.results[].file_path')
```

## Performance

### Search Latency

Typical latency breakdown:
- Query embedding: 50-500ms (depends on provider)
- Vector search: 1-50ms (depends on index size)
- Total: ~100ms-1s per query

### Optimization

For faster searches:
1. Use ONNX embeddings (local, instant)
2. Keep index size manageable (exclude unnecessary files)
3. Use Milvus for large indexes (optimized search)

## Troubleshooting

### No Results

- Verify index exists: `semantic-code-agents status`
- Try broader query
- Check codebase_id is correct

### Irrelevant Results

- Try more specific query
- Use different embeddings provider (Voyage recommended for code)
- Re-index after improving code comments

### Slow Search

- Check embedding provider latency
- Consider local ONNX for speed
- Reduce top_k if you don't need many results

See [Troubleshooting](../TROUBLESHOOTING.md) for more.

## Related Guides

- [Indexing Guide](./indexing.md) - How to create the index
- [Configuration Guide](./configuration.md) - Search configuration
- [CLI Reference](../reference/cli.md) - Command details
