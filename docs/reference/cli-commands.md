# CLI Commands

This reference summarizes the command set and key flags.
Primary command: `sca` (alias: `semantic-code`).

## init

```bash
sca init [--config <path>] [--codebase-root <path>] \
  [--storage-mode disabled|project|custom:/abs/path] [--force]
```

## index

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
- `--vector-db-address <host:port>`
- `--vector-db-base-url <url>`
- `--vector-db-database <name>`
- `--vector-db-ssl <true|false>`
- `--vector-db-token <token>`
- `--vector-db-username <name>`
- `--vector-db-password <password>`

## search

```bash
sca search --query <text> [--top-k <n>] [--threshold <f>] \
  [--filter-expr <expr>] [--include-content] [--config <path>] [--codebase-root <path>]
sca search --stdin [--top-k <n>] [--threshold <f>] \
  [--filter-expr <expr>] [--include-content] [--config <path>] [--codebase-root <path>]
```

Vector DB overrides (optional): same as `index`.

## reindex

```bash
sca reindex [--config <path>] [--codebase-root <path>] [--background]
```

Embedding and vector DB overrides match `index`.

## clear

```bash
sca clear [--config <path>] [--codebase-root <path>]
```

## status

```bash
sca status [--config <path>] [--codebase-root <path>]
```

## config

```bash
sca config check [--path <path>] [--overrides-json <json>]
sca config show [--path <path>] [--overrides-json <json>]
sca config validate [--path <path>] [--overrides-json <json>]
```

## jobs

```bash
sca jobs status <job-id> [--codebase-root <path>]
sca jobs cancel <job-id> [--codebase-root <path>]
```

## info

```bash
sca info
```

## Global flags

- `--output <text|json|ndjson>`: select output format.
- `--agent`: machine-friendly defaults (NDJSON output, no prompts, no progress).
- `--no-progress`: suppress progress/logs on stderr.
- `--interactive`: enable prompts (no prompts are used yet).
- `--json`: legacy alias for `--output json`.
