# Embedding Providers Guide

This guide covers setup and configuration for each supported embedding provider.

## Provider Comparison

| Provider | Cost | Speed | Quality | Best For |
|----------|------|-------|---------|----------|
| **ONNX** | Free | Fastest | Good | Development, cost-sensitive |
| **Voyage** | $$ | Fast | Excellent | Code search (recommended) |
| **OpenAI** | $$$ | Medium | Very Good | General purpose |
| **Gemini** | $$ | Medium | Good | Google ecosystem |
| **Ollama** | Free | Varies | Good | Privacy, local LLMs |

## ONNX (Local, Default)

ONNX runs embeddings locally using the ONNX Runtime. No API key required.

### Setup

```toml
[backend]
embeddings_provider = "onnx"
```

That's it! No additional configuration needed.

### Characteristics

- **Cost**: Free
- **Latency**: ~10-50ms per batch
- **Dimension**: 384 (all-MiniLM-L6-v2)
- **Privacy**: 100% local, no data leaves your machine

### When to Use

- Development and testing
- Cost-sensitive projects
- Privacy-critical applications
- Offline usage

### Limitations

- Fixed model (all-MiniLM-L6-v2)
- Lower quality than cloud providers for complex queries
- CPU-bound (no GPU acceleration in default build)

---

## Voyage AI

Voyage specializes in embeddings for code. Highest quality for code search.

### Setup

1. Get API key from [Voyage AI](https://www.voyageai.com/)

2. Set environment variable:
   ```bash
   export VOYAGE_API_KEY="pa-..."
   ```

3. Configure:
   ```toml
   [backend]
   embeddings_provider = "voyage"

   [backend.embeddings]
   voyage_api_key = "${VOYAGE_API_KEY}"
   voyage_model = "voyage-code-2"  # Optional, this is the default
   ```

### Characteristics

- **Cost**: ~$0.02 per 1M tokens
- **Latency**: ~50ms per batch
- **Dimension**: 1024 (voyage-code-2)
- **Best for**: Code understanding

### Available Models

| Model | Dimension | Use Case |
|-------|-----------|----------|
| `voyage-code-2` | 1024 | Code (recommended) |
| `voyage-large-2` | 1536 | General text |
| `voyage-2` | 1024 | Balanced |

### When to Use

- Production code search
- Best-in-class code understanding
- Moderate budget

---

## OpenAI

OpenAI's embeddings are general-purpose and widely used.

### Setup

1. Get API key from [OpenAI](https://platform.openai.com/)

2. Set environment variable:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. Configure:
   ```toml
   [backend]
   embeddings_provider = "openai"

   [backend.embeddings]
   openai_api_key = "${OPENAI_API_KEY}"
   openai_model = "text-embedding-ada-002"  # Optional
   ```

### Characteristics

- **Cost**: ~$0.10 per 1M tokens
- **Latency**: ~100-200ms per batch
- **Dimension**: 1536 (text-embedding-ada-002)
- **Best for**: General purpose

### Available Models

| Model | Dimension | Notes |
|-------|-----------|-------|
| `text-embedding-ada-002` | 1536 | Standard, recommended |
| `text-embedding-3-small` | 1536 | Newer, slightly better |
| `text-embedding-3-large` | 3072 | Highest quality |

### When to Use

- Already using OpenAI ecosystem
- General-purpose embeddings
- Well-documented, stable API

---

## Google Gemini

Gemini provides embeddings through Google's AI platform.

### Setup

1. Get API key from [Google AI Studio](https://aistudio.google.com/)

2. Set environment variable:
   ```bash
   export GEMINI_API_KEY="AIza..."
   ```

3. Configure:
   ```toml
   [backend]
   embeddings_provider = "gemini"

   [backend.embeddings]
   gemini_api_key = "${GEMINI_API_KEY}"
   ```

### Characteristics

- **Cost**: Free tier available, then ~$0.025 per 1M tokens
- **Latency**: ~200-300ms per batch
- **Dimension**: 768 (embedding-001)
- **Best for**: Google Cloud users

### When to Use

- Google Cloud ecosystem
- Free tier available
- Moderate quality requirements

---

## Ollama (Local LLM)

Ollama runs LLMs locally, including embedding models.

### Setup

1. Install Ollama from [ollama.ai](https://ollama.ai/)

2. Pull an embedding model:
   ```bash
   ollama pull nomic-embed-text
   ```

3. Start Ollama server:
   ```bash
   ollama serve
   ```

4. Configure:
   ```toml
   [backend]
   embeddings_provider = "ollama"

   [backend.ollama]
   base_url = "http://localhost:11434"
   model = "nomic-embed-text"
   ```

### Characteristics

- **Cost**: Free (local hardware)
- **Latency**: Varies by model and hardware
- **Dimension**: Varies by model
- **Privacy**: 100% local

### Available Models

| Model | Dimension | Notes |
|-------|-----------|-------|
| `nomic-embed-text` | 768 | Good quality, fast |
| `mxbai-embed-large` | 1024 | Higher quality |
| `all-minilm` | 384 | Lightweight |

### When to Use

- Privacy requirements
- No external API access
- GPU available for acceleration

### Requirements

- Ollama installed and running
- Sufficient RAM (4GB+ for most models)
- Optional: GPU for faster inference

---

## Choosing a Provider

### Decision Matrix

| Requirement | Best Choice |
|-------------|-------------|
| Cost-free | ONNX or Ollama |
| Best code quality | Voyage |
| General purpose | OpenAI |
| Google ecosystem | Gemini |
| Complete privacy | ONNX or Ollama |
| Fastest | ONNX |

### Migration Between Providers

Changing providers requires re-indexing because embedding dimensions differ:

```bash
# 1. Clear existing index
semantic-code-agents clear-index

# 2. Update config
# Edit .context/config.toml

# 3. Re-index
semantic-code-agents index
```

### Fallback Strategy

Configure ONNX as automatic fallback:

```toml
[backend]
embeddings_provider = "openai"

[backend.embeddings]
fallback_provider = "onnx"  # Use ONNX if OpenAI fails
```

---

## Performance Tuning

### Batch Size

Larger batches are more efficient for cloud providers:

```toml
[backend.embeddings]
batch_size = 100  # Texts per API call
```

Recommended:
- ONNX: 50-100
- Voyage: 100-200
- OpenAI: 100-200
- Gemini: 50-100
- Ollama: 10-50 (depends on model)

### Caching

Enable embedding cache to avoid re-computing:

```toml
[backend.cache]
enabled = true
ttl_secs = 3600  # 1 hour
```

### Timeouts

Adjust for slow networks:

```toml
[backend.resilience]
request_timeout_secs = 60
max_retries = 3
```

---

## Troubleshooting

### "API key not found"

```bash
# Check if set
echo $OPENAI_API_KEY

# Set it
export OPENAI_API_KEY="sk-..."
```

### "Rate limit exceeded"

- Reduce `batch_size`
- Add delay between requests
- Upgrade API tier

### "Connection timeout"

- Check network connectivity
- Increase `request_timeout_secs`
- Verify provider URL

### "Dimension mismatch"

- Clear and re-index after changing providers
- Each provider has different embedding dimensions

See [Troubleshooting](../TROUBLESHOOTING.md) for more.

---

## Related

- [Configuration Guide](./configuration.md) - Full config options
- [Indexing Guide](./indexing.md) - Using embeddings for indexing
