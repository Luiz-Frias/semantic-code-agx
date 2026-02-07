# Troubleshooting

Common issues and how to solve them.

## Indexing Issues

### "Index not found" Error

**Problem**: You try to search but get "Index not found" error.

**Solution**:
1. Run `semantic-code-agents index` to create the initial index
2. Verify the index path in `.context/config.toml`:
   ```toml
   [backend.local_index]
   path = ".context/index"
   ```
3. Check that the path exists: `ls -la .context/index`

### Indexing Takes Too Long

**Problem**: Indexing is slow or seems stuck.

**Causes**:
- Large codebase with many files
- Slow disk I/O
- Network latency (if using cloud embeddings)
- CPU-bound (ONNX embeddings)

**Solutions**:
1. **Use faster embeddings provider**:
   - ONNX (local, fastest) - default
   - Voyage (cloud, very fast)
   - OpenAI (cloud, standard)
   - Ollama (local, depends on model)

2. **Optimize .contextignore**:
   ```
   # Exclude large directories
   node_modules/
   .git/
   dist/
   build/
   venv/
   __pycache__/
   ```

3. **Use incremental indexing**:
   ```bash
   semantic-code-agents reindex --incremental
   ```

4. **Check system resources**:
   ```bash
   # Monitor memory and CPU
   top -p $(pgrep -f semantic-code-agents)
   ```

### "Out of Memory" Error

**Problem**: Process killed or memory usage exceeds system limits.

**Causes**:
- Very large codebase
- Local vector database keeping entire index in memory
- Multiple large files

**Solutions**:
1. **Use Milvus vector database** (persistent, scalable):
   ```toml
   [backend]
   vector_db = "milvus"

   [backend.milvus]
   uri = "http://milvus:19530"
   ```

2. **Exclude large directories** in `.contextignore`

3. **Split indexing** by subdirectory:
   ```bash
   semantic-code-agents index src/
   semantic-code-agents index tests/
   # Merge results
   ```

4. **Increase system memory** or use a more powerful machine

## Search Issues

### Search Returns No Results

**Problem**: Search queries return empty results.

**Causes**:
- Index not created yet
- Query is too specific
- Wrong codebase selected
- Index is corrupted

**Solutions**:
1. **Verify index exists**:
   ```bash
   semantic-code-agents status
   ```

2. **Try a broader query**:
   ```bash
   # Instead of: "PostgreSQL connection pooling"
   # Try: "database connection"
   ```

3. **Check codebase_id** (if using multiple codebases):
   ```bash
   semantic-code-agents search "query" --codebase my-project
   ```

4. **Re-index** if index might be corrupted:
   ```bash
   semantic-code-agents clear-index
   semantic-code-agents index
   ```

### Search Results Are Irrelevant

**Problem**: Search returns results but they're not related to your query.

**Causes**:
- Query is ambiguous
- Not enough code examples in index
- Embeddings provider has poor semantic understanding
- Low quality code comments/naming

**Solutions**:
1. **Use better embeddings provider**:
   - Try: Voyage > OpenAI > Gemini (in that order for code)
   - Avoid: ONNX for very complex semantic queries

2. **Improve query clarity**:
   ```bash
   # Instead of: "handle stuff"
   # Try: "error handling with retry logic"
   ```

3. **Add context to code**:
   - Better variable names
   - Clear function documentation
   - Meaningful comments

4. **Increase top_k**:
   ```bash
   semantic-code-agents search "query" --top-k 10
   ```

## API/Server Issues

### "Connection refused" on API

**Problem**: Can't connect to API endpoint.

**Solution**:
1. Start the server if not running:
   ```bash
   semantic-code-agents serve --port 8000
   ```

2. Verify server is running:
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

3. Check if port is already in use:
   ```bash
   lsof -i :8000
   ```

4. Use different port:
   ```bash
   semantic-code-agents serve --port 9000
   ```

### API Returns 400 Bad Request

**Problem**: Your API request returns 400 error.

**Causes**:
- Invalid JSON in request body
- Missing required fields
- Invalid field types

**Solution**:
1. Check request format against [API Reference](./reference/api-v1.md)
2. Validate JSON:
   ```bash
   echo '{"query":"..."}' | jq .
   ```
3. Check server logs for detailed error
4. Example valid request:
   ```bash
   curl -X POST http://localhost:8000/api/v1/search \
     -H "Content-Type: application/json" \
     -d '{
       "query": "error handling",
       "codebase_id": "my-project",
       "top_k": 5
     }'
   ```

### API Returns 500 Server Error

**Problem**: API returns 500 internal server error.

**Solution**:
1. Check server logs for stack trace
2. Verify index exists: `semantic-code-agents status`
3. Check system resources (disk space, memory)
4. Restart server:
   ```bash
   # Stop server (Ctrl+C)
   # Restart with debug logging
   RUST_LOG=debug semantic-code-agents serve
   ```

## Configuration Issues

### "Invalid configuration" Error

**Problem**: Startup fails with configuration error.

**Solution**:
1. Validate TOML syntax:
   ```bash
   # Check for TOML errors
   cat .context/config.toml | head -20
   ```

2. Verify required fields:
   ```toml
   [backend]
   embeddings_provider = "onnx"  # Required
   vector_db = "local"           # Required
   ```

3. Check environment variables:
   ```bash
   # If using env vars in config
   echo $OPENAI_API_KEY  # Should not be empty
   ```

### API Key Not Found

**Problem**: "API key not found" error when using cloud embeddings.

**Solution**:
1. Set the environment variable:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

2. Or add to `.context/config.toml`:
   ```toml
   [backend.embeddings]
   openai_api_key = "sk-..."
   ```

3. Verify it's set:
   ```bash
   echo $OPENAI_API_KEY
   ```

4. Restart the application

### "Network timeout" on Embeddings

**Problem**: Embeddings request times out (cloud provider slow).

**Solution**:
1. **Increase timeout** in config:
   ```toml
   [backend.embeddings]
   request_timeout_secs = 60  # Default is 30
   ```

2. **Check network connectivity**:
   ```bash
   curl -I https://api.openai.com  # For OpenAI
   ```

3. **Use faster provider**:
   - Try Voyage (very fast)
   - Or use local ONNX embeddings

4. **Reduce batch size**:
   ```toml
   [backend.embeddings]
   batch_size = 10  # Default might be 100
   ```

## Performance Issues

### Search Latency is High

**Problem**: Search requests take too long to respond.

**Causes**:
- Large index
- Slow embeddings provider
- Slow vector database lookup
- Network latency

**Solutions**:
1. **Profile the operation**:
   ```bash
   time semantic-code-agents search "query"
   ```

2. **Use faster embeddings** (in order):
   - ONNX (local)
   - Voyage (cloud, optimized)
   - OpenAI
   - Gemini

3. **Optimize index size**:
   - Better `.contextignore`
   - Smaller `max_chunk_size` (faster embedding, less accurate)

4. **Use caching** (if available):
   ```toml
   [backend.cache]
   enabled = true
   ttl_secs = 3600  # Cache embeddings for 1 hour
   ```

## Getting Help

If you can't find a solution:

1. **Check [FAQ](./FAQ.md)** - Frequently asked questions
2. **Read [Configuration Guide](./guides/configuration.md)** - Understand configuration options
3. **Check [API Reference](./reference/api-v1.md)** - Verify API usage
4. **Open a GitHub Issue** - Include:
   - Command/request you ran
   - Full error message
   - Output of `semantic-code-agents --version`
   - Configuration (without sensitive data)
   - System info (OS, Rust version)

---

**Still stuck?** Open an issue on GitHub with the information above.
