# TOOL.agents.md — semantic-code-agx

## Agent Entry
- Use `sca --agent ...` for all machine calls unless you explicitly need human text output.
- `--agent` defaults output to NDJSON and forces `no_progress=true`.
- Output mode precedence is: `--output` > `--json` > `--agent` > default `text`.

## AGX Flow (State-Aware)
| Goal | Command | Success signal | If it fails |
|---|---|---|---|
| Validate runtime config/env | `sca --agent config check [--path ...]` | `{"type":"summary","status":"ok","kind":"config"}` | Fix `ERR_CONFIG_*` input/env issues. |
| Bootstrap repo for indexing/search | `sca --agent index --init --codebase-root <repo>` | `summary` line with `status:"ok"` and `indexStatus` | If `ERR_STORAGE_INSUFFICIENT_FREE_SPACE`, reduce scope/free space. |
| Rebuild changed files only | `sca --agent reindex --codebase-root <repo>` | `summary` line with `kind:"reindex"` | If `ERR_CORE_INVALID_INPUT`, repo is not initialized (manifest missing). |
| Query semantic index | `sca --agent search --query "<q>" --top-k <u32>` | streamed `result` lines + final `summary` | If no results, rerun after index/reindex and adjust query/threshold. |
| Clear indexed data | `sca --agent clear --codebase-root <repo>` | `summary` line with `kind:"clear"` | If vector provider error, inspect `ERR_VECTOR_*`. |
| Track background work | `sca --agent jobs status <job_id>` | `job_status` line with terminal `job.state` | If `failed`, inspect `job.error.code` and apply recovery table below. |

## NDJSON Parsing Contract (Compact)
Parse one JSON object per line, branch on `type`.

```json
{"type":"summary","status":"ok", "...":"..."}
{"type":"result","relativePath":"...","startLine":1,"endLine":2,"score":0.42,"content":"...|null"}
{"type":"job_status","status":"ok","job":{"id":"...","kind":"index|reindex","state":"queued|running|completed|failed|cancelled","...":"..."}}
{"type":"error","status":"error","error":{"code":"ERR_*","message":"...","kind":"EXPECTED|INVARIANT","meta":{"...":"..."}}}
```

Notes:
- `sca search` streams multiple `result` lines, then a `summary`.
- `sca index`/`reindex` with `--background` returns job envelope (`job_status` shape).
- Some CLI-local failures (argument parsing/serialization/io wrappers) print plain `stderr` `error: ...` and skip API error envelope.

## Exit Codes (Automation)
- `0`: success.
- `1`: internal/serialization/invariant path.
- `2`: invalid input (includes API `EXPECTED` kind).
- `3`: IO error (`CliError::Io` path).

## Retry / Repair Loop
1. If error code prefix is `ERR_CONFIG_` or `ERR_DOMAIN_`: do not blind-retry; fix request/env/config.
2. If error is `ERR_CORE_INVALID_INPUT` with manifest context: run `sca --agent index --init ...` or `sca init`.
3. If error is `ERR_STORAGE_INSUFFICIENT_FREE_SPACE`: stop and reduce/free resources.
4. If error is `ERR_VECTOR_VDB_TIMEOUT` or `ERR_VECTOR_VDB_CONNECTION`: retry with bounded backoff.
5. If error is `ERR_VECTOR_SNAPSHOT_*`: rebuild index/snapshot (`clear` then `index --init`).
6. If error is `ERR_CORE_TIMEOUT` or transient `ERR_CORE_IO`: retry with bounded backoff and jitter.
7. If error kind is `INVARIANT` or code is `ERR_CORE_INTERNAL`: escalate.
8. If search returns empty after successful execution: verify index via `status`, rerun `reindex`, then lower threshold.
9. If a background job appears stuck in `running`: use `jobs status`, then `jobs cancel`, then rerun in foreground for direct errors.
10. If vector auth/connection failures persist: verify `SCA_VECTOR_DB_*` env and endpoint.
11. If config/env parse fails: run `config check` and fix the exact `ERR_CONFIG_INVALID_ENV_*` code.

## Search + Filter Behavior
- Allowed filter syntax is one comparison only:
  - fields: `relativePath | language | fileExtension`
  - operators: `== | !=`
  - value must be quoted (`'...'` or `"..."`)
- Invalid filter returns `ERR_CONFIG_INVALID_FILTER_EXPR` (request validation) or `ERR_VECTOR_INVALID_FILTER_EXPR` (adapter path).

## Practical Command Patterns
1. Happy path (fresh repo to first query):
```bash
sca --agent config check
sca --agent index --init --codebase-root /repo
sca --agent search --codebase-root /repo --query "retry policy" --top-k 5
```

## Hidden/Internal Commands (Use With Care)
- `sca jobs run --job-id <id>`: internal worker entrypoint.
- `sca validate-request --kind <...> --input-json <...>`: request DTO pre-validation.
- `sca self-check`: available only under `cfg(any(debug_assertions, feature = "dev-tools"))`.

## Concurrency Note
- For multi-process agent orchestration, avoid overlapping mutating calls (`index`, `reindex`, `clear`) on the same repo. See `TOOL.md` for full idempotency/operational semantics.

## Known Gaps
- `filterExpr` and `includeContent` are validated at request layer, but semantic-search execution currently does not forward filter to vector queries.

## Escalate to Full Contract
- Use `TOOL.md` when you need full JSON/NDJSON schemas, exact field types, full error catalog, and full configuration bounds.
