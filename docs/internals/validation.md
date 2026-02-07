# Request Validation

Phase 03 Milestone 3 adds **boundary request DTOs** and **validation helpers**
in the config crate (`semantic_code_config`).

Validation goals:

- Catch shape errors early (missing/empty fields).
- Enforce stable bounds for query options (topK/threshold).
- Enforce an allowlist grammar for `filterExpr` (deny unknown expressions).

This is **not a security boundary**: callers must still treat `filterExpr` as
untrusted input.

## Ownership matrix

| Concern | Owner | Notes |
|---|---|---|
| `codebaseRoot` non-empty, not a URL, no NUL | `config` | Shape + basic safety checks only (no filesystem I/O). |
| `collectionName` allowlist pattern | `domain` | Delegated to `CollectionName::parse()` (no duplicated regex/pattern). |
| `query` non-empty after trim | `config` | Query text is a boundary input. |
| `topK` bounds | `config` | Currently `1..=50`. |
| `threshold` bounds | `config` | Must be finite and `0..=1`. |
| `filterExpr` allowlist | `config` | Deny unknown; see grammar below. |

## Request DTOs

The config crate defines DTOs (serde JSON) and validated forms:

- `IndexRequestDto` → `IndexRequest`
- `SearchRequestDto` → `SearchRequest`
- `ReindexByChangeRequestDto` → `ReindexByChangeRequest`
- `ClearIndexRequestDto` → `ClearIndexRequest`

These DTOs are the shared boundary contract for the CLI and local adapters.

## `filterExpr` allowlist grammar

Accepted forms (single comparison only):

- `relativePath == '<value>'`
- `relativePath != '<value>'`
- `language == '<value>'`
- `fileExtension == '<value>'`

Rules:

- `<value>` must be a single-quoted or double-quoted string.
- Newlines are rejected.
- Any other operators/fields/boolean expressions are rejected.

## Tools

- `sca validate-request --kind <...> --input-json <json>` (hidden command)
  validates a request payload and returns `status: ok` on success.
