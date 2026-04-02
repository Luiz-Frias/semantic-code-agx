# CLI Reference

Complete reference for the `sca` command-line interface.

Primary command: `sca` (alias: `semantic-code`).

## Commands

### init

Create `.context/manifest.json` and `.context/config.toml` for a new codebase.

```bash
sca init [--config <path>] [--codebase-root <path>] \
  [--storage-mode disabled|project|custom:/abs/path] [--force]
```

### estimate-storage

Preview index storage requirements and free-space headroom.

```bash
sca estimate-storage [--config <path>] [--codebase-root <path>]
```

Embedding and vector DB overrides match `index`.

### index

Build the vector index from source files.

```bash
sca index [--config <path>] [--codebase-root <path>] [--init] [--background]
```

Embedding overrides (optional):

- `--embedding-provider <id>`
- `--embedding-model <id>`
- `--embedding-base-url <url>`
- `--embedding-dimension <n>`
- `--embedding-local-first <true|false>`
- `--embedding-local-only <true|false>`
- `--embedding-routing-mode <localFirst|remoteFirst|split>`
- `--embedding-split-remote-batches <n>`

Vector DB overrides (optional):

- `--vector-db-provider <id>`
- `--vector-kernel <hnsw-rs|dfrr|flat-scan>`
- `--vector-db-address <host:port>`
- `--vector-db-base-url <url>`
- `--vector-db-database <name>`
- `--vector-db-ssl <true|false>`
- `--vector-db-token <token>`
- `--vector-db-username <name>`
- `--vector-db-password <password>`

### search

Perform semantic search against the index.

```bash
sca search --query <text> [--top-k <n>] [--threshold <f>] \
  [--filter-expr <expr>] [--include-content] [--config <path>] [--codebase-root <path>]
sca search --stdin [--top-k <n>] [--threshold <f>] \
  [--filter-expr <expr>] [--include-content] [--config <path>] [--codebase-root <path>]
sca search --stdin-batch [--config <path>] [--codebase-root <path>]
```

Input modes:

- `--query <text>`: inline query string
- `--stdin`: read single query from stdin
- `--stdin-batch`: read NDJSON queries from stdin, emit one NDJSON result per line (loads index once)

```bash
sca search --query "error handling"
echo "error handling" | sca search --stdin --output ndjson
printf '%s\n' '{"query":"error handling","topK":10}' | sca search --stdin-batch --output ndjson
```

Vector DB overrides: same as `index`.

### calibrate

Find a BQ1 threshold for the local DFRR kernel.

```bash
sca calibrate [--config <path>] [--codebase-root <path>] \
  [--target-recall <f>] [--precision <f>] [--num-queries <n>] [--top-k <n>]
```

Vector DB overrides: same as `index`.

### reindex

Incrementally update the index based on file changes (Merkle diff).

```bash
sca reindex [--config <path>] [--codebase-root <path>] [--background]
```

Embedding and vector DB overrides match `index`.

### clear

Remove all indexed data.

```bash
sca clear [--config <path>] [--codebase-root <path>]
```

### status

Show index metadata and health.

```bash
sca status [--config <path>] [--codebase-root <path>]
```

### config

Inspect and validate configuration.

```bash
sca config check [--path <path>] [--overrides-json <json>]
sca config show [--path <path>] [--overrides-json <json>]
sca config validate [--path <path>] [--overrides-json <json>]
```

### jobs

Manage background jobs started with `--background`.

```bash
sca jobs status <job-id> [--codebase-root <path>]
sca jobs cancel <job-id> [--codebase-root <path>]
```

### info

Display build metadata (version, git hash, target triple, profile).

```bash
sca info
```

### self-check (developer-only)

Available in debug builds or with the `dev-tools` feature.

```bash
sca self-check
sca self-check --output json
```

Reports: status blocks, build metadata, kernel metadata.

## Global Flags

| Flag | Description |
|---|---|
| `--output <text\|json\|ndjson>` | Select output format |
| `--agent` | Machine-friendly defaults (NDJSON output, no prompts, no progress) |
| `--no-progress` | Suppress progress/logs on stderr |
| `--interactive` | Enable prompts (no prompts are used yet) |
| `--json` | Legacy alias for `--output json` |

## Kernel Selection

`--vector-kernel` is accepted on `estimate-storage`, `index`, `search`, `calibrate`, and `reindex`.

Accepted values:

- `hnsw-rs` -- default local HNSW kernel
- `dfrr` -- experimental DFRR kernel (requires feature flag)
- `flat-scan` -- exact search for benchmarking and ground truth

Requesting `dfrr` without build support returns `vector:kernel_unsupported`. `clear` and `status` do not currently accept `--vector-kernel`.

## Output Routing

- Machine-readable output (`--output json|ndjson`) writes to stdout.
- Logs and diagnostics write to stderr.
- `--agent` forces NDJSON and suppresses progress.

## Agent and Automation Usage

For CI, scripts, and AI agent workflows:

```bash
sca --agent search --query "auth flow"
sca --agent --output ndjson search --stdin
sca --output ndjson search --query "error handling" \
  | jq -r 'select(.type=="result") | .relativePath'
```

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Internal error |
| `2` | Invalid input or validation failure |
| `3` | I/O failure |

## Common Workflows

### First-time setup

```bash
sca init
sca index --init
sca search --query "error handling and recovery"
```

### Daily development

```bash
sca reindex
sca search --query "auth middleware"
```

### Background indexing

```bash
sca index --background
sca jobs status <job-id>
sca jobs cancel <job-id>
```

### Machine-readable output

```bash
sca --output json search --query "embedding"
sca --output ndjson search --query "embedding"
sca --agent search --query "embedding"
```
