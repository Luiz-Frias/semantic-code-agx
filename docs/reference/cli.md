# CLI Reference

The CLI provides local entrypoints for init, indexing, search, reindex, clear,
status, config inspection, jobs, and build info. The developer-only
`self-check` command is available in debug builds (or when built with the
`dev-tools` feature).

Primary command: `sca` (alias: `semantic-code`).

For end-to-end local flows, see `docs/reference/cli-usage.md`.

## Usage

```bash
sca --version
sca init
sca index --init
sca index --init --background
sca search --query "local-index"
sca reindex
sca reindex --background
sca clear
sca status
sca jobs status <job-id>
sca jobs cancel <job-id>
sca config show
sca config validate
sca info
```

`self-check` is developer-only:

```bash
sca self-check
sca self-check --output json
```

## Output routing

- Machine-readable output uses `--output json|ndjson` and is written to stdout.
- Logs and diagnostics are written to stderr.
- `--agent` forces NDJSON output and suppresses progress output.
- `--json` remains supported as a legacy alias for `--output json`.

## Search input

`search` accepts `--query` or `--stdin`:

```bash
sca search --query "error handling"
echo "error handling" | sca search --stdin --output ndjson
```

## Background jobs

Use `--background` on `index` or `reindex` to run asynchronously. The command
returns a job id; use `sca jobs status` to poll or `sca jobs cancel` to request
cancellation.

## Related references

- `docs/reference/cli-commands.md`
- `docs/reference/cli-usage.md`
- `docs/reference/agent-usage.md`
