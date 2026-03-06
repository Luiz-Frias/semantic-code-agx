# CLI Reference

The CLI provides local entrypoints for init, storage estimation, indexing,
search, calibration, snapshot utilities, reindex, clear, status, config
inspection, jobs, and build info. The developer-only `self-check` command is
available in debug builds (or when built with the `dev-tools` feature).

Phase 06 adds local vector kernel override flags on selected commands and
kernel metadata in v2 snapshot flows (see internals docs below).

Primary command: `sca` (alias: `semantic-code`).

For end-to-end local flows, see `docs/reference/cli-usage.md`.

## Usage

```bash
sca --version
sca init
sca estimate-storage
sca index --init
sca index --init --background
sca index --vector-kernel hnsw-rs
sca search --query "local-index"
sca search --stdin-batch --output ndjson
sca search --query "local-index" --vector-kernel hnsw-rs
sca calibrate --target-recall 0.99 --vector-kernel dfrr
sca snapshot-subset --source .context/snapshots/current --dest /tmp/subset --target-count 1000
sca reindex
sca reindex --vector-kernel hnsw-rs
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

## Kernel selection flags

`--vector-kernel` is currently accepted on:

- `estimate-storage`
- `index`
- `search`
- `calibrate`
- `reindex`

Accepted values:

- `hnsw-rs`
- `dfrr`
- `flat-scan`

Behavior notes:

- Effective kernel still passes through config validation and adapter/build
  support checks.
- `flat-scan` is intended for exact local runs and benchmark ground-truth
  comparisons.
- Requesting `dfrr` on a build without DFRR support returns
  `vector:kernel_unsupported`.
- `clear` and `status` do not currently expose `--vector-kernel`.

## Output routing

- Machine-readable output uses `--output json|ndjson` and is written to stdout.
- Logs and diagnostics are written to stderr.
- `--agent` forces NDJSON output and suppresses progress output.
- `--json` remains supported as a legacy alias for `--output json`.

## Search input

`search` accepts `--query`, `--stdin`, or `--stdin-batch`:

```bash
sca search --query "error handling"
echo "error handling" | sca search --stdin --output ndjson
```

`--stdin-batch` reads NDJSON queries from stdin and writes one NDJSON result per
line after loading the local index once:

```bash
printf '%s\n' '{"query":"error handling","topK":10}' | sca search --stdin-batch --output ndjson
```

## Background jobs

Use `--background` on `index` or `reindex` to run asynchronously. The command
returns a job id; use `sca jobs status` to poll or `sca jobs cancel` to request
cancellation.

## Self-check output metadata

`self-check --output json` reports:

- top-level status blocks: `status`, `env`, `index`, `search`, `clear`
- build metadata block: `build.name`, `build.version`, `build.facadeVersion`,
  `build.rustcVersion`, `build.target`, `build.profile`, `build.gitHash`,
  `build.gitDirty`
- kernel metadata block: `vectorKernel.effective`

## Storage preflight

Use `estimate-storage` to preview index storage requirements and free-space
headroom based on your current config and codebase:

```bash
sca estimate-storage
sca --output json estimate-storage
```

## Calibration

Use `calibrate` to find a BQ1 threshold for the local DFRR kernel:

```bash
sca calibrate --vector-kernel dfrr --target-recall 0.99 --num-queries 50
```

## Snapshot subset

Use `snapshot-subset` to sample or perturb a v2 snapshot for benchmarking and
reproducible smaller fixtures:

```bash
sca snapshot-subset --source .context/snapshots/current --dest /tmp/subset --target-count 5000
```

## Related references

- `docs/reference/cli-commands.md`
- `docs/reference/cli-usage.md`
- `docs/reference/agent-usage.md`
