# API v1 Contract

## Scope

API v1 defines stable DTOs and error payloads for external boundaries (CLI, HTTP, MCP).
It is a versioned contract layer that maps internal `ErrorEnvelope` values into
wire-safe responses.

## DTO overview

- `ApiV1ErrorDto`: `code`, `message`, `kind`, optional `meta`.
- `ApiV1Result<T>`: `{ ok: true, data: T } | { ok: false, error: ApiV1ErrorDto }`.
- Requests: index, search, reindex-by-change, clear-index.
- Responses: index, search, reindex-by-change, clear-index.

## Error mapping rules

- `ErrorCode` maps to `ERR_<NAMESPACE>_<CODE>` with uppercase segments.
- `ErrorKind::Invariant` maps to `INVARIANT`; `Expected` and `Unexpected` map to `EXPECTED`.
- `ErrorClass` is not exposed in API v1.
- `cause` is never exposed in API v1 payloads.

## Redaction policy

`ApiV1ErrorDto.meta` is redacted on output:

- Keys containing `api_key`, `apikey`, `token`, `password`, `secret`, `authorization`, or `bearer`
  are replaced with `[REDACTED]`.
- Keys equal to `query`, keys ending in `query`, and `content` values are replaced with
  `[REDACTED,len=<n>]`, where `<n>` is the original string length.

## Validation policy

API v1 DTO validation helpers only check shape and limits:

- `codebaseRoot` and `query` must be non-empty.
- `topK` must be in `1..=50` when provided.
- `threshold` must be in `0..=1` when provided.
- `filterExpr` is deny-by-default (non-empty values are rejected).

Domain invariants (IDs, spans, collection naming) are validated by domain constructors.

## Evolution policy

- Additive changes are allowed (new optional fields, new response types).
- Renames or removals require a new version (v2+).
- `ApiV1ErrorDto.code` values are stable and must remain backward compatible.
