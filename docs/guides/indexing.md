# Indexing Guide

This guide covers how to index your codebase for semantic search, including initial indexing, incremental updates, and index management.

## Overview

Indexing transforms your source code into searchable vector embeddings:

```
Source Files → Code Chunks → Embeddings → Vector Index
```

## Quick Start

```bash
# Index current directory
semantic-code-agents index

# Index a specific directory
semantic-code-agents index /path/to/project
```

## Prerequisites

1. **Configuration file** at `.context/config.toml` (or specify with `--config`)
2. **Embeddings provider** configured (ONNX by default, no API key needed)
3. **Optional**: `.contextignore` file to exclude files

## Full Indexing

### Basic Usage

```bash
# Index with default settings
semantic-code-agents index

# Force re-index (drop existing and rebuild)
semantic-code-agents index --force
```

### What Gets Indexed

By default, the indexer processes:
- All text files in supported languages (25+ languages via tree-sitter)
- Files not matching `.contextignore` patterns
- Files not in `.git`, `node_modules`, etc. (default excludes)

### .contextignore File

Create a `.contextignore` file (like `.gitignore`) to exclude files:

```
# Dependencies
node_modules/
vendor/
venv/

# Build outputs
target/
dist/
build/

# Generated files
*.generated.ts
*.min.js

# Large files
*.lock
*.log

# Sensitive
.env*
credentials/
```

### Indexing Process

The indexer follows this pipeline:

1. **Scan files** - Find all files matching filters
2. **Split into chunks** - Use tree-sitter to create semantic code chunks
3. **Generate embeddings** - Convert chunks to vector representations
4. **Store in index** - Persist to vector database

### Progress Monitoring

```bash
# Verbose output
semantic-code-agents index --verbose

# JSON output (for scripting)
semantic-code-agents index --json
```

Output shows:
- Files scanned
- Chunks created
- Embeddings generated
- Time elapsed

## Incremental Reindexing

After initial indexing, use incremental reindex for efficiency:

```bash
# Only process changed files
semantic-code-agents reindex --incremental
```

### How It Works

1. Computes file hashes using Merkle tree
2. Compares against previous snapshot
3. Identifies added, modified, and deleted files
4. Only processes changes

### Change Detection

The system detects:
- **Added files**: New files not in previous index
- **Modified files**: Files with content changes
- **Deleted files**: Files removed since last index

### Best Practices

- Run incremental reindex after code changes
- Run full reindex weekly or after major refactoring
- Use CI/CD to automate reindexing

## Managing the Index

### Check Index Status

```bash
semantic-code-agents status
```

Shows:
- Index location
- Total chunks
- Last indexed timestamp
- Storage size

### Clear the Index

```bash
# Remove all indexed data
semantic-code-agents clear-index
```

This removes:
- All vectors from the database
- File sync snapshots
- Associated metadata

### Multiple Codebases

Index multiple projects separately:

```bash
# Index project A
semantic-code-agents index /path/to/project-a --codebase project-a

# Index project B
semantic-code-agents index /path/to/project-b --codebase project-b
```

Search specific codebase:

```bash
semantic-code-agents search "query" --codebase project-a
```

## Configuration Options

### Indexing Settings

```toml
[backend]
embeddings_provider = "onnx"  # or "openai", "voyage", etc.
vector_db = "local"           # or "milvus"

[backend.local_index]
path = ".context/index"

[backend.indexing]
batch_size = 100              # Chunks per embedding batch
max_concurrent_files = 10     # Parallel file processing
max_chunk_size = 1000         # Max tokens per chunk
```

### Performance Tuning

For large codebases:

```toml
[backend.indexing]
batch_size = 200              # Larger batches (if memory allows)
max_concurrent_files = 20     # More parallelism

[backend]
vector_db = "milvus"          # Scalable vector DB
```

## Troubleshooting

### "Out of memory"

- Reduce `batch_size`
- Reduce `max_concurrent_files`
- Use Milvus instead of local index

### Slow indexing

- Use ONNX embeddings (local, fastest)
- Increase `max_concurrent_files`
- Exclude large directories in `.contextignore`

### Missing files

- Check `.contextignore` patterns
- Verify file extensions are supported
- Check file permissions

See [Troubleshooting](../TROUBLESHOOTING.md) for more solutions.

## API Reference

For programmatic indexing, see:
- [REST API](../reference/api-v1.md) - HTTP endpoints
- [CLI Reference](../reference/cli.md) - Command details

## Related Guides

- [Searching Guide](./searching.md) - How to search the index
- [Configuration Guide](./configuration.md) - Full config options
- [Embedding Providers](./embedding-providers.md) - Provider setup
