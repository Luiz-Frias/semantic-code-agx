# Frequently Asked Questions

## General

### What is semantic-code-agx?

It's a semantic code search engine that uses embeddings and vector databases to find code based on meaning, not keywords. Ask questions like "error handling" and find all error handling code, regardless of the exact variable names or syntax used.

### How is this different from grep/ripgrep?

- **grep/ripgrep**: Find exact text matches (fast but limited)
- **semantic-code-agx**: Find by meaning using AI embeddings (powerful but requires computation)

Use semantic search when:
- You don't know the exact code name
- You want conceptually similar code
- You're exploring unfamiliar codebases

Use grep when:
- You know the exact identifier
- You need speed on every single search
- You're working with tiny codebases

### Is it production-ready?

Yes! The code follows production standards:
- Comprehensive error handling
- Type-safe design with Rust
- Extensive testing
- Hexagonal architecture for extensibility
- Multiple production-grade integrations (Milvus, OpenAI, etc.)

### Is my code private?

**It depends on your configuration:**
- **Local (ONNX) + Local index**: All processing local, completely private
- **Cloud embeddings**: Your code is sent to the embedding provider (OpenAI, Gemini, etc.)

For sensitive code, use local ONNX embeddings.

### How much storage does the index need?

Typical ratios:
- **Source code**: 1 MB → **5-10 MB index** (with embeddings)
- **10,000 files** → **500 MB - 2 GB** (depending on file size)

Example:
- A 100 MB codebase → ~500 MB to 1 GB index

Use `semantic-code-agents status` to check current index size.

## Configuration & Setup

### Do I need an API key?

**No, not required.** The default is ONNX (local embeddings, no key needed).

To use cloud embeddings, you'll need:
- **OpenAI**: `OPENAI_API_KEY`
- **Gemini**: `GEMINI_API_KEY`
- **Voyage**: `VOYAGE_API_KEY`
- **Ollama**: Local (no key)

### How do I choose an embeddings provider?

**For best results** (in order):
1. **Voyage** - Fast, accurate, designed for code
2. **OpenAI** - General purpose, good for code
3. **Gemini** - Good semantic understanding
4. **ONNX** - Free, local, good enough

**For speed**:
1. **ONNX** - Instant (local)
2. **Voyage** - ~50ms
3. **OpenAI** - ~100-200ms
4. **Gemini** - ~200-300ms

**For cost**:
1. **ONNX** - $0
2. **Ollama** - $0
3. **Voyage** - $$$$ (cheapest cloud)
4. **OpenAI** - $$$

Choose ONNX unless you need higher quality.

### How do I use it with my project?

1. Add `.context/config.toml` to your project root
2. Add `.contextignore` (like `.gitignore`)
3. Run `semantic-code-agents index`
4. Run `semantic-code-agents search "your query"`

See [Getting Started](./GETTING_STARTED.md).

### Can I use Milvus instead of local index?

Yes! For production deployments, use Milvus:

```toml
[backend]
vector_db = "milvus"

[backend.milvus]
uri = "http://milvus:19530"
```

See [Configuration Guide](./guides/configuration.md).

## Usage

### Why are my search results bad?

**Common causes:**
1. **Query is too vague** - Be specific
2. **Code is poorly named** - Better names → better results
3. **Wrong embeddings provider** - Try Voyage or OpenAI
4. **Small index** - Need more code examples

**Solutions:**
1. Try a more specific query
2. Improve code comments and variable names
3. Try a different embeddings provider
4. Increase `top_k` parameter

### How accurate is the search?

Accuracy depends on:
- **Embeddings quality** (Voyage best, ONNX okay)
- **Code quality** (well-named variables, good comments)
- **Query specificity** (specific beats vague)
- **Index size** (more examples → better)

Typical: 70-90% relevant results in top-5.

### Can I search multiple codebases?

Yes! Use `codebase_id`:

```bash
semantic-code-agents search "query" --codebase my-project
```

See [Searching Guide](./guides/searching.md).

### How often should I re-index?

**Recommendations:**
- **Development**: After major changes (new features, refactoring)
- **CI/CD**: Nightly or weekly full re-index
- **Incremental**: After each commit (if supported)

For most teams, nightly re-indexing is sufficient.

## Performance & Scaling

### How fast is it?

**Indexing:**
- ~1,000 files/minute (single-threaded)
- Faster with parallel processing enabled

**Searching:**
- Embedding: 50ms - 500ms (depends on provider)
- Vector search: 1-50ms (depends on index size)
- Total: ~100ms - 1 second per query

### Can it handle large codebases (100k+ files)?

Yes, with proper configuration:
1. Use Milvus vector database (not local)
2. Use Voyage or OpenAI embeddings
3. Exclude non-essential files in `.contextignore`
4. Use parallel indexing

See [Configuration Guide](./guides/configuration.md) for production setup.

### What's the maximum codebase size?

No hard limit, but practical considerations:
- **Local index**: 1-10 GB (depends on RAM)
- **Milvus**: 100+ GB (scales horizontally)

For 1M+ files, use Milvus with cloud.

## Development

### How do I contribute?

1. Read [Contributing Guide](../CONTRIBUTING.md)
2. Fork the repository
3. Create a feature branch
4. Make changes and test
5. Submit a pull request

### How can I extend it (add adapters)?

The system uses hexagonal architecture:

**Add a new embedding provider:**
1. Implement `Embedder` trait
2. Add config option
3. Add tests

**Add a new vector database:**
1. Implement `VectorDb` trait
2. Add config option
3. Add tests

See [Contributing Guide](../CONTRIBUTING.md) for details.

### Where should I report security vulnerabilities?

Email **security@example.com** instead of opening a public issue.

See [Security Policy](../SECURITY.md).

### Can I use this commercially?

Yes! It's dual-licensed MIT/Apache-2.0, so you can use it in commercial products.

## Troubleshooting

### My search is slow

See [Troubleshooting: Performance Issues](./TROUBLESHOOTING.md#performance-issues).

### I get "API key not found"

See [Troubleshooting: Configuration Issues](./TROUBLESHOOTING.md#api-key-not-found).

### Index is using too much memory

See [Troubleshooting: Out of Memory](./TROUBLESHOOTING.md#out-of-memory-error).

### Nothing seems to work

1. Check [Getting Started](./GETTING_STARTED.md) - start fresh
2. Check [Troubleshooting](./TROUBLESHOOTING.md) - common issues
3. Check [Configuration Guide](./guides/configuration.md) - verify config
4. Open an issue on GitHub with:
   - What you tried
   - Error message
   - Config (without secrets)

## Licensing & Legal

### What license is this?

Dual-licensed MIT/Apache-2.0. You can choose either.

### Can I use this in a commercial product?

Yes, both licenses allow commercial use.

### Can I modify the code?

Yes, both licenses allow modifications. MIT requires attribution; Apache-2.0 is more permissive.

### Do I need to open-source my project?

No, neither MIT nor Apache-2.0 require you to open-source anything.

---

**Still have questions?** Check the [documentation](./README.md) or open an issue on GitHub.
