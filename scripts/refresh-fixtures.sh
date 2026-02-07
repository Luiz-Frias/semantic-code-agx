#!/usr/bin/env bash
# =============================================================================
# REFRESH API V1 FIXTURES
# =============================================================================
# Regenerate API v1 JSON fixtures used by parity tests. Requires the reference
# snapshot to exist so we can keep fixture inputs aligned with the TS source.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REFERENCE_ROOT="$PROJECT_ROOT/references/semantic-code-for-agents"
REFERENCE_TYPES="$REFERENCE_ROOT/packages/backend/src/api/v1/types.ts"
FIXTURE_DIR="$PROJECT_ROOT/crates/testkit/fixtures/api-v1"
REFERENCE_FIXTURE_DIR="$REFERENCE_ROOT/packages/backend/src/api/v1/fixtures"

if [[ ! -f "$REFERENCE_TYPES" ]]; then
  echo "Reference snapshot missing at $REFERENCE_ROOT" >&2
  echo "Run scripts/refresh-references.sh first." >&2
  exit 1
fi

mkdir -p "$FIXTURE_DIR"
mkdir -p "$REFERENCE_FIXTURE_DIR"

cat >"$FIXTURE_DIR/json-fixtures.json" <<'JSON'
{
  "errorDto": {
    "code": "ERR_DOMAIN_INVALID_COLLECTION_NAME",
    "message": "CollectionName must match /^[a-zA-Z][a-zA-Z0-9_]*$/",
    "kind": "EXPECTED",
    "meta": {
      "input": "bad-name",
      "token": "[REDACTED]",
      "query": "[REDACTED,len=11]"
    }
  },
  "okResult": {
    "ok": true,
    "data": {
      "indexedFiles": 3,
      "totalChunks": 7,
      "status": "completed"
    }
  },
  "errorResult": {
    "ok": false,
    "error": {
      "code": "ERR_DOMAIN_INVALID_COLLECTION_NAME",
      "message": "CollectionName must match /^[a-zA-Z][a-zA-Z0-9_]*$/",
      "kind": "EXPECTED",
      "meta": {
        "input": "bad-name",
        "token": "[REDACTED]",
        "query": "[REDACTED,len=11]"
      }
    }
  },
  "indexRequest": {
    "codebaseRoot": "/tmp/repo",
    "collectionName": "code_chunks_123",
    "forceReindex": true
  },
  "indexResponse": {
    "indexedFiles": 3,
    "totalChunks": 7,
    "status": "completed"
  },
  "searchRequest": {
    "codebaseRoot": "/tmp/repo",
    "query": "hello",
    "topK": 5,
    "threshold": 0.42,
    "filterExpr": ""
  },
  "searchResult": {
    "content": "fn main() {}",
    "relativePath": "src/main.rs",
    "startLine": 1,
    "endLine": 1,
    "language": "rust",
    "score": 0.88
  },
  "searchResponse": {
    "results": [
      {
        "content": "fn main() {}",
        "relativePath": "src/main.rs",
        "startLine": 1,
        "endLine": 1,
        "language": "rust",
        "score": 0.88
      }
    ]
  },
  "reindexByChangeRequest": {
    "codebaseRoot": "/tmp/repo"
  },
  "reindexByChangeResponse": {
    "added": 1,
    "removed": 2,
    "modified": 3
  },
  "clearIndexRequest": {
    "codebaseRoot": "/tmp/repo"
  },
  "clearIndexResponse": {
    "cleared": true
  }
}
JSON

cat >"$FIXTURE_DIR/id-parity.json" <<'JSON'
{
  "collection": {
    "codebaseRoot": "/tmp/example-codebase",
    "indexMode": "dense",
    "expected": "code_chunks_ea6f3b5e"
  },
  "codebase": {
    "codebaseRoot": "/tmp/example-codebase-2",
    "expected": "codebase_dbdae6de5a20"
  },
  "chunk": {
    "relativePath": "src/main.rs",
    "startLine": 1,
    "endLine": 3,
    "content": "fn main() {\n    println!(\"hi\");\n}\n",
    "expected": "chunk_60f65cfc556c5638"
  }
}
JSON

cp "$FIXTURE_DIR/json-fixtures.json" "$REFERENCE_FIXTURE_DIR/json-fixtures.json"
cp "$FIXTURE_DIR/id-parity.json" "$REFERENCE_FIXTURE_DIR/id-parity.json"

echo "✓ API v1 fixtures refreshed"
