# Filesystem Adapter

The filesystem adapter provides async access to local files for indexing and
sync workflows.

## Responsibilities

- Read directories and return deterministic, sorted entries.
- Read UTF-8 file contents.
- Provide file metadata (kind, size, mtime).
- Enforce optional max file size limits on reads.

## Path handling

- Paths are normalized to use `/` separators.
- Absolute paths and `..` traversal segments are rejected by the path policy.
- Empty inputs normalize to `"."`.

## Determinism

- Directory listings are sorted by entry name.
- Normalization removes duplicate slashes and redundant `./`.
