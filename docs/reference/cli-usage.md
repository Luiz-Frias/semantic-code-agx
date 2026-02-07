# CLI Usage

This guide shows the minimal CLI flows for local indexing and semantic search.
Primary command: `sca` (alias: `semantic-code`).

## Initialize a repo

```bash
sca init
```

This creates `.context/manifest.json` and `.context/config.toml` when missing.

## Index

```bash
sca index --init
```

## Background jobs

Run long operations asynchronously and poll status:

```bash
sca index --background
sca reindex --background
sca jobs status <job-id>
sca jobs cancel <job-id>
```

## Search

```bash
sca search --query "error handling"
```

Use stdin for scripted flows:

```bash
echo "error handling" | sca search --stdin --output ndjson
```

## Reindex by changes

```bash
sca reindex
```

## Clear local data

```bash
sca clear
```

## Status

```bash
sca status
```

## Output formats

```bash
sca --output json search --query "embedding"
sca --output ndjson search --query "embedding"
sca --agent search --query "embedding"
```
