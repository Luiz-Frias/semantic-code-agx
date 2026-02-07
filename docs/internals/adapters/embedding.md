# Embedding Adapters

Phase 06 adds **external embeddings** and a **local ONNX** adapter, all behind
feature gates. Adapters live in `semantic-code-adapters` and are selected by
`semantic-code-infra`.

## Provider selection

The provider is chosen from `embedding.provider` (or `SCA_EMBEDDING_PROVIDER`):

- `onnx` / `local`: local ONNX embeddings
- `openai`, `gemini`, `voyage`, `ollama`: remote providers
- `test`: deterministic dummy embeddings (fixtures only)
- `auto`: pick a configured remote provider, otherwise fall back to ONNX

Local-only mode can be enforced with `embedding.localOnly` (or
`SCA_EMBEDDING_LOCAL_ONLY=true`).

## Detecting dimensions

All adapters implement `detect_dimension`:

- If `embedding.dimension` is set, that value is returned without an API call.
- Otherwise, the adapter performs a minimal probe using a small test string.

## Remote providers

Remote adapters are feature-gated and use `reqwest` with request timeouts and
`RequestContext` cancellation checks. Provider-specific overrides are taken
from env vars (e.g. `OPENAI_API_KEY`, `VOYAGEAI_API_KEY`) and do **not** persist
into the config file.

## Local ONNX (ORT)

The ONNX adapter uses the Rust `ort` crate only (no Python bindings). Model
assets are resolved in this order:

1. Explicit `embedding.onnx.modelDir`
2. Default cache: `.context/models/onnx/<repo>`
3. Legacy cache fallback: `.context/onnx-cache/<repo>`

If assets are missing and `embedding.onnx.downloadOnMissing=true`, the adapter
downloads using the `hf` CLI.

### ONNX assets

Required files:

- `onnx/model.onnx` (or `model.onnx` at the repo root)
- `tokenizer.json`

Optional files:

- `tokenizer_config.json`
- `config.json`
- `special_tokens_map.json`
- `vocab.txt`

## Test fallback (optional)

When `SCA_EMBEDDING_TEST_FALLBACK=true`, the embedding factory will fall back to
the deterministic test adapter **only** if ONNX assets are missing. This keeps
production-like ONNX behavior while allowing local testing to proceed without
manual downloads.

## Adapter versioning notes

Adapters target the current provider API shapes and **are not a stable API**.
When upgrading providers or models:

- pin the model name explicitly (`embedding.model`)
- verify request/response compatibility in tests
- update fixtures as needed
