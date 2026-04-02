# Vector Quantization And Experimental U8 Search

This project stores v2 local snapshots with SQ8 quantized vectors (`u8`) and
supports an optional experimental search path that runs cosine distance directly
on quantized bytes.

## Feature Gate

- Cargo feature: `experimental-u8-search`
- Current default: enabled
- Runtime flag: `vectorDb.experimentalU8Search` (default: `false`)

When the runtime flag is disabled, search always falls back to the existing f32
HNSW path.

## Config

TOML:

```toml
[vectorDb]
experimentalU8Search = true
```

Env override:

```bash
SCA_VECTOR_DB_EXPERIMENTAL_U8_SEARCH=true
```

## Limitations

- The u8 path is experimental and currently uses exact scan over quantized
  vectors, not HNSW over u8 payloads.
- Quantization is fit from the current active record set at query time.
- Search quality can degrade on datasets with strong outliers or very tight
  near-duplicate clusters.
- Tie behavior can differ from the f32 path when quantization collapses nearby
  values into identical bins.

## Recommendation

Keep `experimentalU8Search = false` unless you are explicitly evaluating the
quality/performance tradeoff for your dataset.
