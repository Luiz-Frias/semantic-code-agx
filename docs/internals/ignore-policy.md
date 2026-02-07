# Ignore Policy

Ignore patterns are used to filter files and directories during scans.

## Normalization

- Trim whitespace.
- Replace `\\` with `/`.
- Collapse duplicate `/`.
- Trim leading `./` and surrounding `/`.

## Matching semantics

Patterns are treated as path segment sequences. A path is ignored when it
contains a contiguous segment sequence equal to the normalized pattern.

Examples:

- `node_modules/` matches `node_modules/pkg/index.js`
- `target/` matches `src/target/output.log`
- `src/generated` matches `src/generated/code.rs`

## .contextignore

If a `.contextignore` file is present at the codebase root, its non-empty,
non-comment lines (starting with `#`) are merged with configured ignore
patterns. The `.contextignore` file itself is always ignored.

## Notes

- Patterns are order-independent; duplicates are ignored.
- Glob and negation syntax are not supported in Phase 05.
- The `.context/` state directory is always ignored.
