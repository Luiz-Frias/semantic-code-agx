# Security

Phase 07 Milestone 3 hardens local CLI workflows by tightening path handling
and enforcing safe logging behavior.

## Path Policy

All file system access from core workflows uses the `SafeRelativePath` policy:

- Absolute paths are rejected.
- Traversal segments (`..`) are rejected.
- The `.context/` state directory is rejected.

This prevents index and reindex flows from reading internal state files.

## State Directories

The CLI uses `.context/` under the codebase root for state:

- `config.toml`
- `manifest.json`
- vector and sync snapshots
- background job metadata

Security rules:

- `.context/` is **always** included in ignore patterns.
- `.context/` is never indexed as part of the codebase.

## Snapshot Storage Paths

`vectorDb.snapshotStorage` is validated at config load:

- `custom` paths **must** be absolute.
- Invalid paths fail config validation with a structured error.

## Log Redaction

Structured logs and telemetry redact sensitive keys automatically:

- Keys that look like secrets (`apiKey`, `token`, `password`, etc.) are replaced
  with `[REDACTED]`.
- Redaction applies to nested JSON objects.

This prevents accidental secret leakage at observability boundaries.
