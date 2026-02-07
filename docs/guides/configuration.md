# Configuration Guide

This guide covers all configuration options for semantic-code-agx.

## Configuration Sources

Configuration is loaded from multiple sources with this precedence (later wins):

```
Defaults → Config File → Environment Variables → CLI Arguments
```

## Quick Start

Create `.context/config.toml` in your project root:

```toml
[backend]
embeddings_provider = "onnx"
vector_db = "local"

[backend.local_index]
path = ".context/index"
```

## Configuration File

### Location

The CLI looks for configuration in this order:
1. Path specified with `--config`
2. `.context/config.toml` in current directory
3. Default values

### Formats

Both TOML and JSON are supported:

**TOML** (recommended):
```toml
[backend]
embeddings_provider = "openai"
```

**JSON**:
```json
{
  "backend": {
    "embeddings_provider": "openai"
  }
}
```

## Full Configuration Reference

### Backend Section

```toml
[backend]
# Embeddings provider: "onnx" | "openai" | "gemini" | "voyage" | "ollama"
embeddings_provider = "onnx"

# Vector database: "local" | "milvus"
vector_db = "local"
```

### Embeddings Configuration

```toml
[backend.embeddings]
# API keys (can use environment variable expansion)
openai_api_key = "${OPENAI_API_KEY}"
gemini_api_key = "${GEMINI_API_KEY}"
voyage_api_key = "${VOYAGE_API_KEY}"

# Model overrides (optional, uses defaults if not set)
openai_model = "text-embedding-ada-002"
voyage_model = "voyage-code-2"

# Request settings
request_timeout_secs = 30
batch_size = 100
max_retries = 3
```

### Local Index Configuration

```toml
[backend.local_index]
# Path to store the index
path = ".context/index"

# HNSW parameters (advanced)
ef_construction = 200
m = 16
```

### Milvus Configuration

```toml
[backend.milvus]
# Connection URI
uri = "http://localhost:19530"

# Authentication (optional)
username = ""
password = ""

# Collection settings
collection_name = "code_embeddings"
```

### Ollama Configuration

```toml
[backend.ollama]
# Ollama server URL
base_url = "http://localhost:11434"

# Model to use
model = "nomic-embed-text"
```

### Indexing Configuration

```toml
[backend.indexing]
# Chunks per embedding batch
batch_size = 100

# Parallel file processing
max_concurrent_files = 10

# Maximum tokens per chunk
max_chunk_size = 1000

# Minimum tokens per chunk (avoid tiny fragments)
min_chunk_size = 50
```

### Cache Configuration

```toml
[backend.cache]
# Enable embedding cache
enabled = true

# Cache TTL in seconds (1 hour)
ttl_secs = 3600

# Max cache entries
max_entries = 10000
```

### Resilience Configuration

```toml
[backend.resilience]
# Retry settings
max_retries = 3
retry_delay_ms = 1000
retry_backoff_multiplier = 2.0

# Timeout settings
request_timeout_secs = 30
connect_timeout_secs = 10
```

## Environment Variables

All configuration can be overridden via environment variables:

| Variable | Config Path | Example |
|----------|-------------|---------|
| `SCA_EMBEDDINGS_PROVIDER` | `backend.embeddings_provider` | `openai` |
| `SCA_VECTOR_DB` | `backend.vector_db` | `milvus` |
| `OPENAI_API_KEY` | `backend.embeddings.openai_api_key` | `sk-...` |
| `GEMINI_API_KEY` | `backend.embeddings.gemini_api_key` | `AIza...` |
| `VOYAGE_API_KEY` | `backend.embeddings.voyage_api_key` | `pa-...` |

### Environment Variable Expansion

Config files support `${VAR}` syntax:

```toml
[backend.embeddings]
openai_api_key = "${OPENAI_API_KEY}"
```

This expands at load time from the environment.

## CLI Arguments

CLI arguments override all other sources:

```bash
# Override config file
semantic-code-agents index --config /path/to/config.toml

# Override specific settings
semantic-code-agents search "query" --top-k 10
```

## Configuration Profiles

### Development (Local, Fast)

```toml
[backend]
embeddings_provider = "onnx"
vector_db = "local"

[backend.local_index]
path = ".context/index"
```

### Production (Cloud, Scalable)

```toml
[backend]
embeddings_provider = "voyage"
vector_db = "milvus"

[backend.embeddings]
voyage_api_key = "${VOYAGE_API_KEY}"

[backend.milvus]
uri = "${MILVUS_URI}"
```

### Testing (In-Memory)

```toml
[backend]
embeddings_provider = "onnx"
vector_db = "local"

[backend.local_index]
path = "/tmp/test-index"
```

## Validation

### Check Configuration

```bash
# Validate config file
semantic-code-agents config check --path .context/config.toml

# JSON output
semantic-code-agents config check --path .context/config.toml --json
```

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `config:invalid_toml` | Syntax error in TOML | Check TOML syntax |
| `config:config_file_not_found` | File doesn't exist | Verify path |
| `config:missing_api_key` | API key not set | Set env var or add to config |
| `config:invalid_provider` | Unknown provider | Use valid provider name |

## Security

### Sensitive Data

- **Never commit API keys** to version control
- Use environment variables for secrets
- Add `.context/config.toml` to `.gitignore` if it contains keys

### Example .gitignore

```
# Config with secrets
.context/config.toml

# But allow template
!.context/config.toml.example

# Index data
.context/index/
```

### Config Template

Create `.context/config.toml.example` for the team:

```toml
[backend]
embeddings_provider = "openai"
vector_db = "local"

[backend.embeddings]
# Set OPENAI_API_KEY environment variable
openai_api_key = "${OPENAI_API_KEY}"

[backend.local_index]
path = ".context/index"
```

## Troubleshooting

### "Invalid configuration"

```bash
# Check syntax
semantic-code-agents config check --path .context/config.toml --json
```

### "API key not found"

```bash
# Verify env var is set
echo $OPENAI_API_KEY

# Or add directly to config (not recommended for shared configs)
```

### "Unknown provider"

Valid providers:
- `onnx` (local, default)
- `openai`
- `gemini`
- `voyage`
- `ollama`

See [Troubleshooting](../TROUBLESHOOTING.md) for more.

## Related

- [Embedding Providers](./embedding-providers.md) - Provider-specific setup
- [Config Schema Reference](../reference/config-schema.md) - Full schema
- [Environment Variables](../reference/env-vars.md) - All env vars
