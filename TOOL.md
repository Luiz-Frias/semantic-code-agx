# TOOL.md — semantic-code-agx

## State Machine

> Agent quick-path: use `TOOL.agents.md` for compact invocation flow, retry logic, and debug playbooks.
> `TOOL.md` is the authoritative full wire contract (all schemas, flags, and error-code mappings).

| State ID | State name | Observable condition (source-of-truth) |
|---|---|---|
| `S0` | `uninitialized` | `.context/manifest.json` missing. `.context/config.toml` may also be missing. |
| `S1` | `configured` | Config file exists (`--config` path or `.context/config.toml`), but manifest missing. |
| `S2` | `initialized` | Manifest exists and validates for current root (`.context/manifest.json`), but searchable collection may be absent. |
| `S3` | `indexed_searchable` | Manifest exists and vector collection exists for manifest `collectionName` (search returns non-error; may return zero results). |
| `JQ` | `job_queued` | `.context/jobs/<job_id>/status.json` has `state: "queued"`. |
| `JR` | `job_running` | `.context/jobs/<job_id>/status.json` has `state: "running"`. |
| `JC` | `job_completed` | `.context/jobs/<job_id>/status.json` has `state: "completed"`. |
| `JF` | `job_failed` | `.context/jobs/<job_id>/status.json` has `state: "failed"` + `error`. |
| `JX` | `job_cancelled` | `.context/jobs/<job_id>/status.json` has `state: "cancelled"`. |

| Transition | Command | Preconditions | Resulting state |
|---|---|---|---|
| `S0 -> S2` | `sca init` | none | writes config+manifest (unless already present and `--force` not set). |
| `S0 -> S3` | `sca index --init` | valid config/env + storage preflight pass | auto-creates manifest/config, then indexes. |
| `S1 -> S3` | `sca index --init` | same as above | same as above. |
| `S2 -> S3` | `sca index` | valid config/env + storage preflight pass | indexes collection and updates manifest timestamp. |
| `S2 -> S3` | `sca reindex` | valid config/env | reindex-by-change; may create/update collection path. |
| `S2 -> S2` | `sca clear` | valid config/env | drops collection if present, deletes sync snapshot. |
| `S2 -> S2` | `sca status` | valid config/env | read-only. |
| `S2 -> S2` | `sca search` | valid config/env | read-only. returns empty results if collection absent. |
| `S* -> JQ` | `sca index --background` / `sca reindex --background` | job request creation succeeds | writes job request+status and spawns `jobs run`. |
| `JQ -> JR -> JC/JF/JX` | `sca jobs run --job-id ...` | job files exist | executes queued job; terminal state persisted. |

| Wrong-state call | Internal code | API-v1 code | kind/class | Notes |
|---|---|---|---|---|
| `sca index` (without `--init`) when manifest missing | `core:invalid_input` | `ERR_CORE_INVALID_INPUT` | `EXPECTED` / non-retriable | message: `manifest missing; run \`sca index --init\`` |
| `sca search` when manifest missing | `core:invalid_input` | `ERR_CORE_INVALID_INPUT` | `EXPECTED` / non-retriable | same manifest message path. |
| `sca reindex` when manifest missing | `core:invalid_input` | `ERR_CORE_INVALID_INPUT` | `EXPECTED` / non-retriable | same manifest message path. |
| `sca clear` when manifest missing | `core:invalid_input` | `ERR_CORE_INVALID_INPUT` | `EXPECTED` / non-retriable | same manifest message path. |
| `sca status` when manifest missing | `core:invalid_input` | `ERR_CORE_INVALID_INPUT` | `EXPECTED` / non-retriable | same manifest message path. |
| `sca jobs status/cancel/run` with unknown id | `core:not_found` | `ERR_CORE_NOT_FOUND` | `EXPECTED` / non-retriable | metadata includes `jobId`. |
| manifest path exists but belongs to another root | `core:invalid_input` | `ERR_CORE_INVALID_INPUT` | `EXPECTED` / non-retriable | message `manifest root mismatch`. |

<!-- TODO: publish an authoritative state-transition conformance test matrix that validates every command/state pair end-to-end. -->

## Commands

### Global Invocation Contract

| Flag | Rust type | Behavior |
|---|---|---|
| `--output <text|json|ndjson>` | `Option<OutputFormat>` | Explicit formatter selection. |
| `--json` (hidden) | `bool` | Legacy alias for `--output json` when `--output` is not set. |
| `--agent` | `bool` | If `--output` unset and `--json` unset, forces `ndjson`; always forces `no_progress=true`. |
| `--no-progress` | `bool` | Suppresses informational stderr logs unless `--interactive` overrides. |
| `--interactive` | `bool` | Forces `no_progress=false` when `--agent` is false. |

Output selection precedence (`OutputMode::from_args`):
1. `--output` (highest)
2. `--json`
3. `--agent`
4. default `text`

`no_progress` precedence:
1. `--agent` => `true`
2. else if `--interactive` => `false`
3. else `--no-progress`

Common error output envelope (`InfraError` path):

- JSON (`--output json`):
```json
{
  "status": "error",
  "error": {
    "code": "ERR_<NAMESPACE>_<CODE>",
    "message": "...",
    "kind": "EXPECTED|INVARIANT",
    "meta": {
      "key": "value"
    }
  }
}
```

- NDJSON (`--output ndjson` or `--agent`):
```json
{"type":"error","status":"error","error":{...ApiV1ErrorDto...}}
```

- Text:
```text
status: error
code: ERR_...
message: ...
kind: EXPECTED|INVARIANT
meta:
  key: value
```

Important: CLI-local argument/resolve failures (`CliError::InvalidInput`, `CliError::Io`, `CliError::Serialization`) are emitted as plain stderr (`error: ...`) and do not use the JSON/NDJSON error envelope.

### Common Override Inputs

Embedding override flags (used by `estimate-storage`, `index`, `reindex`):
- `--embedding-provider <String>`
- `--embedding-model <String>`
- `--embedding-base-url <String>`
- `--embedding-dimension <u32>`
- `--embedding-local-first <bool>`
- `--embedding-local-only <bool>`
- `--embedding-routing-mode <String>`
- `--embedding-split-remote-batches <u32>`

Vector DB override flags (used by `estimate-storage`, `index`, `search`, `clear`, `status`, `reindex`):
- `--vector-db-provider <String>`
- `--vector-db-address <String>`
- `--vector-db-base-url <String>`
- `--vector-db-database <String>`
- `--vector-db-ssl <bool>`
- `--vector-db-token <String>`
- `--vector-db-username <String>`
- `--vector-db-password <String>`

These are merged into runtime override JSON (`vectorDb.*`, `embedding.*`) before config loading.

### `sca info`

- Preconditions: none.
- Input:
  - no command-specific flags.
- Output schema:
  - JSON:
```json
{
  "status": "ok",
  "build": {
    "name": "string",            // &'static str
    "version": "string",         // &'static str
    "facadeVersion": "string",   // &str
    "rustcVersion": "string",    // &'static str
    "target": "string",          // &'static str
    "profile": "string",         // &'static str
    "gitHash": "string|null",    // Option<&'static str>
    "gitDirty": true               // bool
  }
}
```
  - NDJSON:
```json
{"type":"summary","status":"ok","kind":"info","build":{...same fields...}}
```
- Error codes:

| Code | When | Retry |
|---|---|---|
| none (infra) | normal path | n/a |
| no API code (`CliError::Serialization`) | JSON serialization failure | non-retriable |
| no API code (`CliError::Io`) | stdout/stderr write failure | retriable if transient IO |

- State change: none.
- Performance:
  - Complexity: `O(1)`.
  - 1K/10K/100K files: not applicable.

### `sca config check`

- Preconditions: none.
- Input:
  - `--path <PathBuf?>`
  - `--overrides-json <String?>`
  - consumes `SCA_*` env subset.
- Output schema:
  - JSON:
```json
{
  "status": "ok",
  "configPath": "string|null",
  "effectiveConfig": {"...": "..."} // serde_json::Value of validated config
}
```
  - NDJSON:
```json
{"type":"summary","status":"ok","kind":"config"}
```
  - Text: `status: ok`, `config: ok`, optional `path: ...`.
- Error codes:

| API code | When | Retry |
|---|---|---|
| `ERR_CONFIG_INVALID_JSON` | invalid config/overrides JSON | non-retriable (fix input) |
| `ERR_CONFIG_INVALID_TOML` | invalid config TOML | non-retriable |
| `ERR_CONFIG_CONFIG_FILE_NOT_FOUND` | `--path` missing | non-retriable |
| `ERR_CONFIG_UNSUPPORTED_FORMAT` | config extension not `.json|.toml` | non-retriable |
| `ERR_CONFIG_INVALID_ENV_*` | env parse failures (`BOOL/INT/URL/ENUM/CSV`) | non-retriable |
| `ERR_CONFIG_INVALID_*` | schema validation failures (timeout/limit/url/etc.) | non-retriable |

- State change: none.
- Performance:
  - Complexity: `O(config_size + env_var_count)`.
  - 1K/10K/100K files: not applicable.

### `sca config show`

- Preconditions: none.
- Input: same as `config check`.
- Output schema:
  - JSON/NDJSON same shape as `config check`.
  - Text includes full pretty config text under `config:`.
- Error codes: same as `config check`.
- State change: none.
- Performance: same as `config check`.

### `sca config validate`

- Preconditions: none.
- Input: same as `config check`.
- Output schema:
  - JSON:
```json
{"status":"ok","configPath":"string|null"}
```
  - NDJSON:
```json
{"type":"summary","status":"ok","kind":"config"}
```
  - Text: `status: ok`, `config: ok`, optional `path`.
- Error codes: same config/env/schema family as `config check`.
- State change: none.
- Performance:
  - Complexity: `O(config_size + env_var_count)`.
  - 1K/10K/100K files: not applicable.

### `sca init`

- Preconditions: none.
- Input:
  - `--config <PathBuf?>`
  - `--codebase-root <PathBuf?>` (defaults to current working directory)
  - `--storage-mode <String?>` where accepted values are:
    - `disabled`
    - `project`
    - `custom:/abs/path`
    - `custom=/abs/path`
  - `--force <bool>`
- Output schema:
  - JSON:
```json
{
  "status":"ok",
  "configPath":"string",     // PathBuf
  "manifestPath":"string",   // PathBuf
  "created":{"config":true,"manifest":true}
}
```
  - NDJSON:
```json
{"type":"summary","status":"ok","kind":"init","configPath":"...","manifestPath":"...","createdConfig":true,"createdManifest":true}
```
- Error codes:

| Code | When | Retry |
|---|---|---|
| no API code (`CliError::InvalidInput`) | invalid `--storage-mode` syntax/value | non-retriable |
| `ERR_CONFIG_*` | config/env parsing and validation failures | non-retriable |
| `ERR_CORE_INVALID_INPUT` | manifest root mismatch | non-retriable |
| `ERR_DOMAIN_INVALID_COLLECTION_NAME` | derived/parsed collection invalid | non-retriable |
| `ERR_CORE_IO` / `ERR_CORE_PERMISSION_DENIED` / `ERR_CORE_NOT_FOUND` | filesystem failures | retriable only for transient IO |

- State change:
  - `S0/S1 -> S2` (creates manifest; may create config).
  - with existing manifest and `--force=false`, remains initialized and only appends `.context/` to `.gitignore` if needed.
- Performance:
  - Complexity: `O(1)` metadata/filesystem ops.
  - 1K/10K/100K files: not file-count dependent.

### `sca estimate-storage`

- Preconditions: readable codebase root.
- Input:
  - `--config <PathBuf?>`
  - `--codebase-root <PathBuf?>`
  - common embedding and vector overrides
  - hidden: `--danger-close-storage <bool>`
- Output schema:
  - JSON:
```json
{
  "status":"ok",
  "kind":"estimateStorage",
  "codebaseRoot":"string",          // PathBuf
  "vectorProvider":"string",        // Box<str>
  "localStorageEnforced":true,        // bool
  "localStorageRoot":"string|null", // Option<PathBuf>
  "indexMode":"dense|hybrid",       // IndexMode
  "filesScanned":0,                   // u64
  "filesIndexable":0,                 // u64
  "bytesIndexable":0,                 // u64
  "charsIndexable":0,                 // u64
  "estimatedChunks":0,                // u64
  "dimensionLow":384,                 // u32
  "dimensionHigh":1536,               // u32
  "estimatedBytesLow":0,              // u64
  "estimatedBytesHigh":0,             // u64
  "requiredFreeBytes":0,              // u64
  "safetyFactorNum":2,                // u64
  "safetyFactorDen":1,                // u64
  "safetyFactor":"2.00",            // string derived
  "availableBytes":null,              // Option<u64>
  "thresholdStatus":"pass|fail|unknown"
}
```
  - NDJSON: same fields in one `summary` line with `kind:"estimateStorage"`.
- Error codes:

| API code | When | Retry |
|---|---|---|
| `ERR_CONFIG_*` | config/env/override validation | non-retriable |
| `ERR_CORE_IO` / `ERR_CORE_PERMISSION_DENIED` | filesystem scan/stat read failures | retriable if transient |
| `ERR_CORE_INTERNAL` | internal conversion/runtime failures | non-retriable |

- State change: none.
- Performance:
  - Complexity: `O(number_of_scanned_files + total_indexable_bytes)`.
  - 1K files: `UNVERIFIED`.
  - 10K files: `UNVERIFIED`.
  - 100K files: `UNVERIFIED`.
  - <!-- TODO: add benchmark-backed wall-time envelopes for typical codebase profiles. -->

### `sca index`

- Preconditions:
  - valid index request (`codebaseRoot` path validation).
  - storage headroom check passes unless provider is non-local.
  - manifest exists, or `--init` set.
- Input:
  - `--config <PathBuf?>`
  - `--codebase-root <PathBuf?>`
  - `--init <bool>`
  - `--background <bool>`
  - common embedding/vector overrides
  - hidden: `--danger-close-storage <bool>`
- Output schema:
  - Foreground JSON:
```json
{
  "status":"ok",
  "indexedFiles":0,          // usize
  "totalChunks":0,           // usize
  "indexStatus":"completed|limitReached",
  "stageStats":{
    "scan":{"files":0,"durationMs":0},
    "split":{"files":0,"chunks":0,"durationMs":0},
    "embed":{"batches":0,"chunks":0,"durationMs":0},
    "insert":{"batches":0,"chunks":0,"durationMs":0}
  }
}
```
  - Foreground NDJSON:
```json
{"type":"summary","status":"ok","indexedFiles":0,"totalChunks":0,"indexStatus":"completed|limitReached","stageStats":{...}}
```
  - Background JSON/NDJSON: job envelope (`job_status`) identical to `sca jobs status` output.
- Error codes:

| API code | When | Retry |
|---|---|---|
| `ERR_CONFIG_INVALID_CODEBASE_ROOT` / `ERR_CONFIG_EMPTY_FIELD` | invalid request fields | non-retriable |
| `ERR_CORE_INVALID_INPUT` | manifest missing without `--init`; provider unsupported; invalid split/local settings | non-retriable |
| `ERR_STORAGE_INSUFFICIENT_FREE_SPACE` | local preflight fail | non-retriable until disk/input changes |
| `ERR_VECTOR_*` | vector adapter/snapshot/provider failures | depends on code/class |
| `ERR_CORE_CANCELLED` | cancellation during async pipeline | non-retriable |

- State change:
  - foreground success: `S2 -> S3` (or remains `S3`).
  - background enqueue: `S* -> JQ`.
- Performance:
  - Complexity: `O(files + chunks + embedding_batches + insert_batches)`.
  - 1K files: `UNVERIFIED`.
  - 10K files: `UNVERIFIED`.
  - 100K files: `UNVERIFIED`.
  - <!-- TODO: publish profile-separated index throughput baselines by provider and index mode. -->

### `sca search`

- Preconditions:
  - manifest exists.
  - query provided via `--query` or `--stdin`.
- Input:
  - `--query <String?>`
  - `--stdin <bool>` (conflicts with `--query`)
  - `--top-k <u32?>` (alias: `--max-results`)
  - `--threshold <f32?>`
  - `--filter-expr <String?>`
  - `--include-content <bool>`
  - `--config <PathBuf?>`
  - `--codebase-root <PathBuf?>`
  - common vector overrides
- Output schema:
  - JSON:
```json
{
  "status":"ok",
  "results":[
    {
      "key":{"relativePath":"string","span":{"startLine":1,"endLine":2}},
      "content":"string|null",      // Option<Box<str>>
      "language":"rust|null",        // Option<Language>
      "score":0.42                    // f32
    }
  ]
}
```
  - NDJSON (streamed lines):
```json
{"type":"result","relativePath":"string","startLine":1,"endLine":2,"score":0.42,"content":"string|null"}
{"type":"summary","status":"ok","count":1}
```
  - Text:
```text
status: ok
results: <usize>
<relativePath>:<start>-<end> score=<f32 with 4 decimals>
```
- Error codes:

| Code | When | Retry |
|---|---|---|
| no API code (`CliError::InvalidInput`) | missing `--query/--stdin`, empty stdin query | non-retriable |
| `ERR_CORE_INVALID_INPUT` | manifest missing | non-retriable |
| `ERR_CONFIG_INVALID_FILTER_EXPR` | filter expression fails request allowlist | non-retriable |
| `ERR_CONFIG_OUT_OF_RANGE` | `topK`/`threshold` invalid | non-retriable |
| `ERR_VECTOR_*` | vector provider/search errors | depends on code/class |

- State change: none.
- Performance:
  - Complexity: `O(embed(query) + vector_search(top_k))`.
  - 1K files: `UNVERIFIED`.
  - 10K files: `UNVERIFIED`.
  - 100K files: `UNVERIFIED`.
  - <!-- TODO: add p50/p95 latency by index size and provider. -->

### `sca reindex`

- Preconditions: manifest exists.
- Input:
  - `--config <PathBuf?>`
  - `--codebase-root <PathBuf?>`
  - `--background <bool>`
  - common embedding/vector overrides
- Output schema:
  - Foreground JSON:
```json
{"status":"ok","added":0,"removed":0,"modified":0} // usize counters
```
  - Foreground NDJSON:
```json
{"type":"summary","status":"ok","kind":"reindex","added":0,"removed":0,"modified":0}
```
  - Background JSON/NDJSON: job envelope (`job_status`) identical to `sca jobs status` output.
- Error codes:

| API code | When | Retry |
|---|---|---|
| `ERR_CORE_INVALID_INPUT` | manifest missing | non-retriable |
| `ERR_CONFIG_INVALID_CODEBASE_ROOT` | invalid request root | non-retriable |
| `ERR_VECTOR_*` | vector/snapshot/provider failures | depends on code/class |
| `ERR_CORE_CANCELLED` | cancellation | non-retriable |

- State change:
  - foreground success: usually remains/enters `S3`.
  - background enqueue: `S* -> JQ`.
- Performance:
  - Complexity: `O(changed_files + changed_chunks)` plus snapshot diff.
  - 1K/10K/100K files: `UNVERIFIED`.
  - <!-- TODO: publish reindex cost model for low/high churn datasets. -->

### `sca clear`

- Preconditions: manifest exists.
- Input:
  - `--config <PathBuf?>`
  - `--codebase-root <PathBuf?>`
  - common vector overrides
- Output schema:
  - JSON: `{"status":"ok"}`
  - NDJSON: `{"type":"summary","status":"ok","kind":"clear"}`
  - Text: `status: ok`
- Error codes:

| API code | When | Retry |
|---|---|---|
| `ERR_CORE_INVALID_INPUT` | manifest missing | non-retriable |
| `ERR_VECTOR_VDB_UNKNOWN` | provider-specific drop failure | retriable for milvus drop path in clear retry loop |
| `ERR_VECTOR_*` | other vector adapter failures | depends on code/class |

- State change: `S3 -> S2` (collection dropped/snapshot deleted), or `S2 -> S2` if already empty.
- Performance:
  - Complexity: `O(collection_drop + snapshot_delete)`.
  - 1K/10K/100K files: mostly independent of source file count.

### `sca status`

- Preconditions: manifest exists.
- Input:
  - `--config <PathBuf?>`
  - `--codebase-root <PathBuf?>`
  - common vector overrides
- Output schema:
  - JSON:
```json
{
  "status":"ok",
  "manifest":{
    "schemaVersion":1,            // u32
    "codebaseRoot":"string",    // PathBuf
    "collectionName":"string",  // CollectionName
    "indexMode":"dense|hybrid", // IndexMode
    "snapshotStorage":"disabled|project|{\"custom\":\"/abs/path\"}", // SnapshotStorageMode
    "createdAtMs":0,              // u64
    "updatedAtMs":0               // u64
  },
  "vectorSnapshot":{"path":"string|null","exists":true,"updatedAtMs":0,"recordCount":0},
  "syncSnapshot":{"path":"string|null","exists":true,"updatedAtMs":0,"recordCount":null},
  "config":{
    "indexMode":"dense|hybrid",
    "snapshotStorage":"disabled|project|{\"custom\":\"/abs/path\"}",
    "embeddingDimension":384,
    "embeddingCache":{"enabled":false,"diskEnabled":false,"maxEntries":2048,"maxBytes":134217728,"diskPath":"string|null","diskProvider":"string|null","diskConnection":"string|null","diskTable":"string|null","diskMaxBytes":1073741824},
    "retry":{"maxAttempts":3,"baseDelayMs":250,"maxDelayMs":5000,"jitterRatioPct":20},
    "limits":{"maxInFlightFiles":null,"maxInFlightEmbeddingBatches":null,"maxInFlightInserts":null,"maxBufferedChunks":null,"maxBufferedEmbeddings":null}
  }
}
```
  - NDJSON: same payload in one `summary` line.
- Error codes:

| API code | When | Retry |
|---|---|---|
| `ERR_CORE_INVALID_INPUT` | manifest missing/root mismatch, invalid snapshot parse path | non-retriable |
| `ERR_CONFIG_*` | config/env load failures | non-retriable |
| `ERR_CORE_IO` | filesystem metadata/read failures | retriable if transient |

- State change: none.
- Performance:
  - Complexity: `O(1)` plus snapshot file parse for vector record count.

### `sca jobs status`

- Preconditions: `job_id` exists under `.context/jobs/<job_id>`.
- Input:
  - positional `JOB_ID: String`
  - `--codebase-root <PathBuf?>`
- Output schema:
  - JSON:
```json
{
  "status":"ok",
  "job":{
    "id":"string",                      // Box<str>
    "kind":"index|reindex",            // JobKind
    "state":"queued|running|completed|failed|cancelled", // JobState
    "createdAtMs":0,                     // u64
    "updatedAtMs":0,                     // u64
    "progress":{"stage":"string","phase":"string","current":0,"total":0,"percentage":0}, // optional
    "result":{
      "type":"index|reindex",          // serde tag
      "...":"..."
    },                                     // optional
    "error":{"code":"string","message":"string","class":"retriable|nonretriable"}, // optional
    "cancelRequested":false,
    "warnings":[{"code":"string","message":"string","class":"..."}]
  }
}
```
  - NDJSON:
```json
{"type":"job_status","status":"ok","job":{...JobStatus...}}
```
- Error codes:

| API code | When | Retry |
|---|---|---|
| `ERR_CORE_NOT_FOUND` | unknown job id | non-retriable |
| `ERR_CORE_INVALID_INPUT` | corrupt `request.json`/`status.json` parse | non-retriable |
| `ERR_CORE_IO` | disk IO while reading status | retriable if transient |

- State change: none.
- Performance: `O(1)` file read + JSON parse.

### `sca jobs cancel`

- Preconditions: same as `jobs status`.
- Input: same as `jobs status`.
- Output schema: same `job` envelope as `jobs status`, with `cancelRequested=true` when write succeeds.
- Error codes: same as `jobs status` + IO write failures.
- State change: sets `cancelRequested=true`, writes `.context/jobs/<id>/cancel` marker.
- Performance: `O(1)` file write.

### `sca jobs run` (hidden/internal)

- Preconditions:
  - job request/status files exist.
  - should be invoked by spawned worker process.
- Input:
  - `--job-id <String>`
  - `--codebase-root <PathBuf?>`
- Output schema: same `job` envelope as `jobs status`.
- Error codes:

| API code | When | Retry |
|---|---|---|
| `ERR_CORE_NOT_FOUND` | missing request/status files | non-retriable |
| `ERR_CORE_INVALID_INPUT` | request/status JSON parse failure | non-retriable |
| `ERR_STORAGE_INSUFFICIENT_FREE_SPACE` | index job preflight fail | non-retriable until storage changes |
| `ERR_VECTOR_*`, `ERR_CONFIG_*`, `ERR_CORE_*` | propagated job execution failures | code-dependent |

- State change: `JQ -> JR -> JC|JF|JX`.
- Performance: same as the underlying job (`index` or `reindex`).

### `sca validate-request` (hidden)

- Preconditions: none.
- Input:
  - `--kind <ValidateRequestKind>` where variants are `Index|Search|ReindexByChange|ClearIndex`.
  - `--input-json <String>`.
- Output schema:
  - JSON: `{"status":"ok","kind":"index|search|reindexByChange|clearIndex"}`
  - NDJSON: `{"type":"summary","status":"ok","kind":"..."}`
- Error codes:

| API code | When | Retry |
|---|---|---|
| `ERR_CONFIG_INVALID_JSON` | malformed input JSON | non-retriable |
| `ERR_CONFIG_EMPTY_FIELD` / `ERR_CONFIG_INVALID_FIELD` / `ERR_CONFIG_OUT_OF_RANGE` | schema validation failure | non-retriable |
| `ERR_CONFIG_INVALID_FILTER_EXPR` | unsupported filter expression | non-retriable |
| `ERR_DOMAIN_INVALID_COLLECTION_NAME` | invalid collection override | non-retriable |

- State change: none.
- Performance: `O(request_json_size)`.

### `sca self-check` (hidden; debug/dev-tools gated)

- Preconditions:
  - command exists only when `cfg(any(debug_assertions, feature = "dev-tools"))`.
- Input: no command-specific flags.
- Output schema:
  - JSON:
```json
{
  "status":"ok",
  "env":{"status":"ok"},
  "index":{"status":"ok|error"},
  "search":{"status":"ok|error"},
  "clear":{"status":"ok|error"},
  "build":{...same shape as `sca info` build block...}
}
```
  - Text fields: status/env/index/search/clear + build info.
  - NDJSON: `UNVERIFIED` (no explicit NDJSON branch in implementation).
  - <!-- TODO: add explicit NDJSON support or formally declare unsupported for self-check. -->
- Error codes:

| API code | When | Retry |
|---|---|---|
| `ERR_CONFIG_INVALID_ENV_*` | env parse validation fails | non-retriable |
| `ERR_CORE_INTERNAL` | smoke test failure in index/search/clear | non-retriable |

- State change: none (uses in-memory self-check adapters).
- Performance: small fixed-cost smoke run (`UNVERIFIED` exact duration).

## Filter Expression Syntax

Grammar implemented in both request validator and local vector adapter:

```ebnf
expr        := comparison
comparison  := field ws op ws quoted
field       := "relativePath" | "language" | "fileExtension"
op          := "==" | "!="
quoted      := single_quoted | double_quoted
single_quoted := "'" <chars-no-newline> "'"
double_quoted := '"' <chars-no-newline> '"'
ws          := one-or-more whitespace between field and operator
```

Rules:
- single comparison only.
- no newline (`\n`, `\r`).
- quoted value required.
- empty quoted value rejected.
- unknown fields/operators rejected.

Valid examples:
- `relativePath == 'src/main.rs'`
- `language != "rust"`
- `fileExtension == 'ts'`

Invalid examples:
- `score > 0.5` (unknown field/operator)
- `relativePath == src/main.rs` (missing quotes)
- `relativePath == ''` (empty value)

Error codes:
- request-validation path: `config:invalid_filter_expr` => `ERR_CONFIG_INVALID_FILTER_EXPR`.
- vector-adapter parser path: `vector:invalid_filter_expr` => `ERR_VECTOR_INVALID_FILTER_EXPR`.

Note for CLI `sca search`: filter expressions are validated at request layer, but the current semantic-search pipeline does not pass `filterExpr` to vector queries.

## Error Contract

Error envelope (internal):
- `kind: expected|invariant|unexpected`
- `class: retriable|non-retriable`
- `code: <namespace>:<code>`
- `message: String`
- `metadata: BTreeMap<String,String>`

API-v1 mapping (`infra_error_to_api_v1`):
- `code` => `ERR_<NAMESPACE>_<CODE>` (uppercase, non-alnum => `_`).
- `kind` mapping:
  - `expected` -> `EXPECTED`
  - `unexpected` -> `EXPECTED`
  - `invariant` -> `INVARIANT`
- `class` is not exposed in API-v1 payload.

CLI exit codes:
- `0`: success
- `1`: internal (`CliError::Serialization` or invariant-mapped infra error)
- `2`: invalid input (includes API kind `EXPECTED`, which includes expected+unexpected infra kinds)
- `3`: IO (`CliError::Io`)

### Code Catalog (tool-relevant namespaces)

| Internal code | API code | kind/class (typical) | Recovery guidance |
|---|---|---|---|
| `core:cancelled` | `ERR_CORE_CANCELLED` | expected / non-retriable | stop; do not retry same cancelled context |
| `core:invalid_input` | `ERR_CORE_INVALID_INPUT` | expected / non-retriable | fix arguments/state |
| `core:not_found` | `ERR_CORE_NOT_FOUND` | expected / non-retriable | create/init resource then retry |
| `core:permission_denied` | `ERR_CORE_PERMISSION_DENIED` | unexpected / non-retriable | fix permissions |
| `core:timeout` | `ERR_CORE_TIMEOUT` | unexpected / retriable | retry with backoff |
| `core:io` | `ERR_CORE_IO` | unexpected / mixed | retry if transient IO |
| `core:internal` | `ERR_CORE_INTERNAL` | invariant/unexpected / non-retriable | escalate |
| `config:invalid_json` | `ERR_CONFIG_INVALID_JSON` | expected / non-retriable | fix JSON |
| `config:invalid_toml` | `ERR_CONFIG_INVALID_TOML` | expected / non-retriable | fix TOML |
| `config:config_file_not_found` | `ERR_CONFIG_CONFIG_FILE_NOT_FOUND` | expected / non-retriable | fix path/create file |
| `config:config_file_permission_denied` | `ERR_CONFIG_CONFIG_FILE_PERMISSION_DENIED` | expected / non-retriable | fix permissions |
| `config:config_file_io` | `ERR_CONFIG_CONFIG_FILE_IO` | expected / non-retriable | inspect filesystem |
| `config:unsupported_format` | `ERR_CONFIG_UNSUPPORTED_FORMAT` | expected / non-retriable | use `.json` or `.toml` |
| `config:serialize_toml` | `ERR_CONFIG_SERIALIZE_TOML` | unexpected / non-retriable | escalate |
| `config:unsupported_version` | `ERR_CONFIG_UNSUPPORTED_VERSION` | expected / non-retriable | migrate config version |
| `config:invalid_timeout` | `ERR_CONFIG_INVALID_TIMEOUT` | expected / non-retriable | adjust timeout bounds |
| `config:invalid_limit` | `ERR_CONFIG_INVALID_LIMIT` | expected / non-retriable | adjust numeric limits |
| `config:list_too_large` | `ERR_CONFIG_LIST_TOO_LARGE` | expected / non-retriable | reduce list size |
| `config:invalid_extension` | `ERR_CONFIG_INVALID_EXTENSION` | expected / non-retriable | fix extension tokens |
| `config:invalid_ignore_pattern` | `ERR_CONFIG_INVALID_IGNORE_PATTERN` | expected / non-retriable | fix ignore patterns |
| `config:invalid_url` | `ERR_CONFIG_INVALID_URL` | expected / non-retriable | fix URL syntax/scheme |
| `config:invalid_snapshot_storage` | `ERR_CONFIG_INVALID_SNAPSHOT_STORAGE` | expected / non-retriable | use absolute custom path |
| `config:invalid_cache_config` | `ERR_CONFIG_INVALID_CACHE_CONFIG` | expected / non-retriable | fix provider/path/connection combination |
| `config:invalid_index_config` | `ERR_CONFIG_INVALID_INDEX_CONFIG` | expected / non-retriable | fix index type/metric/params |
| `config:empty_env_var` | `ERR_CONFIG_EMPTY_ENV_VAR` | expected / non-retriable | set non-empty value |
| `config:invalid_env_bool` | `ERR_CONFIG_INVALID_ENV_BOOL` | expected / non-retriable | set valid boolean |
| `config:invalid_env_int` | `ERR_CONFIG_INVALID_ENV_INT` | expected / non-retriable | set valid integer |
| `config:invalid_env_url` | `ERR_CONFIG_INVALID_ENV_URL` | expected / non-retriable | set valid `http/https` URL |
| `config:invalid_env_enum` | `ERR_CONFIG_INVALID_ENV_ENUM` | expected / non-retriable | set supported enum value |
| `config:invalid_env_csv` | `ERR_CONFIG_INVALID_ENV_CSV` | expected / non-retriable | fix CSV entries |
| `config:empty_field` | `ERR_CONFIG_EMPTY_FIELD` | expected / non-retriable | provide field value |
| `config:invalid_field` | `ERR_CONFIG_INVALID_FIELD` | expected / non-retriable | fix field value |
| `config:invalid_codebase_root` | `ERR_CONFIG_INVALID_CODEBASE_ROOT` | expected / non-retriable | use filesystem path |
| `config:out_of_range` | `ERR_CONFIG_OUT_OF_RANGE` | expected / non-retriable | adjust `topK/threshold` etc |
| `config:invalid_filter_expr` | `ERR_CONFIG_INVALID_FILTER_EXPR` | expected / non-retriable | use allowlisted grammar |
| `domain:invalid_collection_name` | `ERR_DOMAIN_INVALID_COLLECTION_NAME` | expected / non-retriable | fix collection name pattern |
| `storage:insufficient_free_space` | `ERR_STORAGE_INSUFFICIENT_FREE_SPACE` | expected / non-retriable | free disk / reduce index scope |
| `vector:invalid_filter_expr` | `ERR_VECTOR_INVALID_FILTER_EXPR` | expected / non-retriable | fix provider filter expression |
| `vector:snapshot_parse_failed` | `ERR_VECTOR_SNAPSHOT_PARSE_FAILED` | unexpected / non-retriable | repair/remove snapshot |
| `vector:snapshot_serialize_failed` | `ERR_VECTOR_SNAPSHOT_SERIALIZE_FAILED` | unexpected / non-retriable | escalate |
| `vector:snapshot_load_task_failed` | `ERR_VECTOR_SNAPSHOT_LOAD_TASK_FAILED` | unexpected / non-retriable | retry once; then repair snapshot |
| `vector:snapshot_count_overflow` | `ERR_VECTOR_SNAPSHOT_COUNT_OVERFLOW` | unexpected / non-retriable | escalate |
| `vector:snapshot_dimension_mismatch` | `ERR_VECTOR_SNAPSHOT_DIMENSION_MISMATCH` | expected / non-retriable | rebuild snapshot/index |
| `vector:snapshot_record_missing` | `ERR_VECTOR_SNAPSHOT_RECORD_MISSING` | expected / non-retriable | rebuild snapshot/index |
| `vector:snapshot_version_mismatch` | `ERR_VECTOR_SNAPSHOT_VERSION_MISMATCH` | expected / non-retriable | migrate/reindex |
| `vector:snapshot_oversize` | `ERR_VECTOR_SNAPSHOT_OVERSIZE` | expected / non-retriable | raise limit or reduce data |
| `vector:snapshot_missing_companion` | `ERR_VECTOR_SNAPSHOT_MISSING_COMPANION` | expected / non-retriable | regenerate v2 companion files |
| `vector:vdb_auth` | `ERR_VECTOR_VDB_AUTH` | unexpected / non-retriable | fix auth/token |
| `vector:vdb_timeout` | `ERR_VECTOR_VDB_TIMEOUT` | unexpected / retriable | retry with backoff |
| `vector:vdb_connection` | `ERR_VECTOR_VDB_CONNECTION` | unexpected / retriable | retry; validate endpoint |
| `vector:vdb_schema_mismatch` | `ERR_VECTOR_VDB_SCHEMA_MISMATCH` | unexpected / non-retriable | fix schema/index setup |
| `vector:vdb_query_invalid` | `ERR_VECTOR_VDB_QUERY_INVALID` | unexpected / non-retriable | fix query/filter |
| `vector:vdb_collection_limit` | `ERR_VECTOR_VDB_COLLECTION_LIMIT` | unexpected / non-retriable | cleanup or increase provider limits |
| `vector:vdb_unknown` | `ERR_VECTOR_VDB_UNKNOWN` | unexpected / non-retriable (except special clear retry case) | inspect provider metadata |

<!-- TODO: generate and publish an authoritative code catalog from all `ErrorCode` constructors across the workspace; `docs/reference/error-codes.md` is policy-only today. -->

## Configuration

### Required vs Optional

Top-level config object (`BackendConfig`):
- required keys with defaults: `version`, `core`, `embedding`, `vectorDb`, `sync`.
- unknown fields rejected (`deny_unknown_fields`).

Defaults (selected, source from `Default` impls):
- `version: 1`
- `core.timeoutMs: 30000`
- `core.maxConcurrency: 8`
- `core.maxChunkChars: 2500`
- `core.retry: {maxAttempts:3, baseDelayMs:250, maxDelayMs:5000, jitterRatioPct:20}`
- `embedding.timeoutMs: 60000`, `batchSize: 32`, `localFirst:false`, `localOnly:false`
- `embedding.onnx.downloadOnMissing:true`, `sessionPoolSize:1`
- `embedding.cache.enabled:false`, `maxEntries:2048`, `maxBytes:134217728`, `diskEnabled:false`, `diskMaxBytes:1073741824`
- `vectorDb.indexMode:dense`, `timeoutMs:60000`, `indexTimeoutMs:60000`, `batchSize:128`, `snapshotStorage:project`, `snapshotFormat:v1`
- `vectorDb.index.dense: {indexType:"AUTOINDEX", metricType:"COSINE"}`
- `vectorDb.index.sparse: {indexType:"SPARSE_INVERTED_INDEX", metricType:"BM25"}`
- `sync.allowedExtensions: [c, cpp, go, java, js, jsx, md, py, rs, ts, tsx]`
- `sync.ignorePatterns: [.context/, .git/, node_modules/, target/]`
- `sync.maxFiles: 250000`, `sync.maxFileSizeBytes: 2000000`

### Merge Precedence

Applied order (last wins):
1. `BackendConfig::default()`
2. config file (`--config` or default `.context/config.toml` when present)
3. CLI override JSON (from command flags)
4. environment (`BackendEnv` parsed from `SCA_*` + aliases)

### Environment Variable Mappings

Core:
- `SCA_CORE_TIMEOUT_MS`
- `SCA_CORE_MAX_CONCURRENCY`
- `SCA_CORE_MAX_IN_FLIGHT_FILES`
- `SCA_CORE_MAX_IN_FLIGHT_EMBEDDING_BATCHES`
- `SCA_CORE_MAX_IN_FLIGHT_INSERTS`
- `SCA_CORE_MAX_BUFFERED_CHUNKS`
- `SCA_CORE_MAX_BUFFERED_EMBEDDINGS`
- `SCA_CORE_MAX_CHUNK_CHARS`
- `SCA_CORE_RETRY_MAX_ATTEMPTS`
- `SCA_CORE_RETRY_BASE_DELAY_MS`
- `SCA_CORE_RETRY_MAX_DELAY_MS`
- `SCA_CORE_RETRY_JITTER_RATIO_PCT`

Embedding core (with aliases):
- `SCA_EMBEDDING_PROVIDER` / `EMBEDDING_PROVIDER`
- `SCA_EMBEDDING_MODEL` / `EMBEDDING_MODEL`
- `SCA_EMBEDDING_TIMEOUT_MS` / `EMBEDDING_TIMEOUT_MS`
- `SCA_EMBEDDING_BATCH_SIZE` / `EMBEDDING_BATCH_SIZE`
- `SCA_EMBEDDING_DIMENSION` / `EMBEDDING_DIMENSION`
- `SCA_EMBEDDING_BASE_URL` / `EMBEDDING_BASE_URL`
- `SCA_EMBEDDING_API_KEY` / `EMBEDDING_API_KEY` (secret)
- `SCA_EMBEDDING_LOCAL_FIRST` / `EMBEDDING_LOCAL_FIRST`
- `SCA_EMBEDDING_LOCAL_ONLY` / `EMBEDDING_LOCAL_ONLY`
- `SCA_EMBEDDING_ROUTING_MODE` / `EMBEDDING_ROUTING_MODE`
- `SCA_EMBEDDING_SPLIT_MAX_REMOTE_BATCHES` / `EMBEDDING_SPLIT_MAX_REMOTE_BATCHES`
- `SCA_EMBEDDING_JOBS_PROGRESS_INTERVAL_MS` / `EMBEDDING_JOBS_PROGRESS_INTERVAL_MS`
- `SCA_EMBEDDING_JOBS_CANCEL_POLL_INTERVAL_MS` / `EMBEDDING_JOBS_CANCEL_POLL_INTERVAL_MS`
- `SCA_EMBEDDING_TEST_FALLBACK` / `EMBEDDING_TEST_FALLBACK`

Embedding ONNX (with aliases):
- `SCA_EMBEDDING_ONNX_MODEL_DIR` / `EMBEDDING_ONNX_MODEL_DIR`
- `SCA_EMBEDDING_ONNX_MODEL_FILENAME` / `EMBEDDING_ONNX_MODEL_FILENAME`
- `SCA_EMBEDDING_ONNX_TOKENIZER_FILENAME` / `EMBEDDING_ONNX_TOKENIZER_FILENAME`
- `SCA_EMBEDDING_ONNX_REPO` / `EMBEDDING_ONNX_REPO`
- `SCA_EMBEDDING_ONNX_DOWNLOAD` / `EMBEDDING_ONNX_DOWNLOAD`
- `SCA_EMBEDDING_ONNX_SESSION_POOL_SIZE` / `EMBEDDING_ONNX_SESSION_POOL_SIZE`

Embedding cache:
- `SCA_EMBEDDING_CACHE_ENABLED`
- `SCA_EMBEDDING_CACHE_MAX_ENTRIES`
- `SCA_EMBEDDING_CACHE_MAX_BYTES`
- `SCA_EMBEDDING_CACHE_DISK_ENABLED`
- `SCA_EMBEDDING_CACHE_DISK_PATH`
- `SCA_EMBEDDING_CACHE_DISK_MAX_BYTES`
- `SCA_EMBEDDING_CACHE_DISK_PROVIDER`
- `SCA_EMBEDDING_CACHE_DISK_CONNECTION` (secret-like)
- `SCA_EMBEDDING_CACHE_DISK_TABLE`

Provider-specific:
- OpenAI: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`
- Gemini: `GEMINI_API_KEY`, `GEMINI_BASE_URL`, `GEMINI_MODEL`
- Voyage: `VOYAGEAI_API_KEY`, `VOYAGEAI_BASE_URL`, `VOYAGEAI_MODEL`
- Ollama: `OLLAMA_MODEL`, `OLLAMA_HOST`

Vector DB:
- `SCA_VECTOR_DB_PROVIDER`
- `SCA_VECTOR_DB_INDEX_MODE`
- `SCA_VECTOR_DB_TIMEOUT_MS`
- `SCA_VECTOR_DB_INDEX_TIMEOUT_MS`
- `SCA_VECTOR_DB_BATCH_SIZE`
- `SCA_VECTOR_DB_SNAPSHOT_FORMAT`
- `SCA_VECTOR_DB_SNAPSHOT_MAX_BYTES`
- `SCA_VECTOR_DB_SEARCH_STRATEGY`
- `SCA_VECTOR_DB_EXPERIMENTAL_U8_SEARCH`
- `SCA_VECTOR_DB_BASE_URL`
- `SCA_VECTOR_DB_ADDRESS`
- `SCA_VECTOR_DB_DATABASE`
- `SCA_VECTOR_DB_SSL`
- `SCA_VECTOR_DB_TOKEN` (secret)
- `SCA_VECTOR_DB_USERNAME`
- `SCA_VECTOR_DB_PASSWORD` (secret)

Sync:
- `SCA_SYNC_ALLOWED_EXTENSIONS`
- `SCA_SYNC_IGNORE_PATTERNS`
- `SCA_SYNC_MAX_FILES`
- `SCA_SYNC_MAX_FILE_SIZE_BYTES`

### Validation Rules (selected bounds)

- `core.timeoutMs`: `1000..=600000`
- `core.maxConcurrency`: `1..=256`
- `core.maxInFlight*`: `1..=256` (when set)
- `core.maxBuffered*`: `1..=1000000` (when set)
- `core.maxChunkChars`: `1..=20000`
- `core.retry.maxAttempts`: `1..=10`
- `core.retry.baseDelayMs`: `1..=60000`
- `core.retry.maxDelayMs`: `1..=600000`, and `>= baseDelayMs`
- `core.retry.jitterRatioPct`: `0..=100`
- `embedding.timeoutMs`: `1000..=1200000`
- `embedding.batchSize`: `1..=8192`
- `embedding.dimension`: `1..=65536` (if set)
- `embedding.onnx.sessionPoolSize`: `1..=64`
- `embedding.routing.split.maxRemoteBatches`: `1..=1000000` (if set)
- `embedding.jobs.progressIntervalMs`: `50..=60000`
- `embedding.jobs.cancelPollIntervalMs`: `50..=60000`
- `embedding.cache.maxEntries`: `1..=100000` (if cache enabled)
- `embedding.cache.maxBytes`: `1..=10000000000` (if cache enabled)
- `embedding.cache.diskMaxBytes`: `1..=100000000000` (if disk cache enabled)
- `vectorDb.timeoutMs`: `1000..=1200000`
- `vectorDb.indexTimeoutMs`: `1000..=3600000`
- `vectorDb.batchSize`: `1..=16384`
- `vectorDb.snapshotMaxBytes`: `1..=100000000000` (if set)
- `sync.maxFiles`: `1..=10000000`
- `sync.maxFileSizeBytes`: `1..=100000000`
- `sync.allowedExtensions` max entries: `128`
- `sync.ignorePatterns` max entries: `512`

### Startup vs Runtime Failures

Startup/load-time failures (before command execution):
- config parse (`invalid_json`, `invalid_toml`, `unsupported_format`)
- env parse (`invalid_env_*`, `empty_env_var`)
- schema validation (`invalid_timeout`, `invalid_limit`, etc.)
- request DTO validation (`empty_field`, `out_of_range`, etc.)

Runtime failures:
- manifest/state (`core:invalid_input`, `core:not_found`)
- storage preflight (`storage:insufficient_free_space`)
- vector provider/network/snapshot (`vector:*`)
- filesystem/process (`core:io`, `core:permission_denied`, etc.)

## Composition Patterns

### Agent-Mode Invocation

Use `--agent` for machine callers:
- default output => NDJSON
- progress/info stderr suppressed
- one-line JSON objects suitable for streaming parsers

Example:
```bash
sca --agent search --query "retry policy" --top-k 5
```

### Typical Chain: estimate -> index -> search

```bash
sca --agent estimate-storage --codebase-root /repo
sca --agent index --codebase-root /repo --init
sca --agent search --codebase-root /repo --query "embedding cache"
```

### Background Job Pattern

```bash
sca --agent index --background --init --codebase-root /repo
sca --agent jobs status <job_id> --codebase-root /repo
sca --agent jobs cancel <job_id> --codebase-root /repo
```

### Piping into Other Tools

NDJSON examples:
```bash
sca --agent search --query "vector" | jq -c 'select(.type=="result")'
sca --agent jobs status <job_id> | jq '.job.state'
```

### Idempotency and Concurrency

| Command | Idempotent (same inputs) | Concurrency safety |
|---|---|---|
| `info`, `config *`, `estimate-storage`, `status`, `jobs status` | yes | read-only |
| `init` | mostly (without `--force`) | safe for serial execution; concurrent writes `UNVERIFIED` |
| `index` foreground | `UNVERIFIED` final-state idempotency | concurrent index/search across processes not strongly coordinated |
| `index/reindex --background` | no (new job id each call) | jobs are independent; shared storage contention possible |
| `reindex` foreground | `UNVERIFIED` | same caveats as index |
| `clear` | effectively yes if manifest exists | concurrent with index/search can race |
| `jobs cancel` | yes (repeated cancel marker writes) | best-effort cancellation |

<!-- TODO: define explicit cross-process locking/consistency guarantees for local snapshot + collection writes. -->

### MCP Tool-Use Integration

When wrapping in MCP tools:
- prefer `--agent` + NDJSON parsing.
- key line discriminators:
  - `type=summary`
  - `type=result`
  - `type=error`
  - `type=job_status`

## Limitations

- `sca search` currently validates `filterExpr`/`includeContent` but semantic-search execution does not forward these flags to vector queries.
- CLI-local argument errors are not emitted as JSON/NDJSON error envelopes (stderr plain text only).
- `self-check` has no explicit NDJSON serializer branch.
- Most operational commands require a local manifest; missing manifest is a hard invalid-input error.
- With `snapshotStorage=disabled` and local provider, indexed state persistence across separate CLI processes is not guaranteed.
- Cross-process strong consistency for concurrent index/search/clear is not guaranteed by explicit locks.
- Performance SLOs for 1K/10K/100K file workloads are not encoded in code.
- Hidden/internal commands (`jobs run`, `validate-request`, `self-check`) are implementation surfaces and may change more frequently.

<!-- TODO: add benchmark artifacts and stable SLO targets for command latency/throughput. -->
<!-- TODO: expose a generated machine-readable schema package for all command outputs (JSON + NDJSON line variants). -->
