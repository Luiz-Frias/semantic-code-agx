# Error Policy

This project uses a shared `ErrorEnvelope` to keep failures typed, consistent,
and safe to expose at boundaries.

## Error envelope

Every error carries:

- `kind`: origin category (`expected`, `invariant`, `unexpected`).
- `class`: retry classification (`retriable`, `non-retriable`).
- `code`: stable namespace + identifier (ex: `core:timeout`).
- `message`: human-readable summary.
- `metadata`: optional diagnostics, redacted before public exposure.

## Kinds

- **Expected**: validation failures, user input errors, cancellation.
- **Invariant**: domain assumptions violated (logic bugs).
- **Unexpected**: I/O, external systems, or failures outside domain control.

## Retry classification

- **Retriable**: transient I/O or dependency failures.
- **Non-retriable**: invalid input, invariants, or explicit cancellation.

## Cancellation

Cancellation uses `core:cancelled` with `expected` + `non-retriable`. Use the
shared helpers to check cancellation instead of matching strings.

## Redaction

Metadata may include sensitive fields (tokens, paths). Use redaction helpers
before mapping errors to public DTOs or CLI output.

## Unknown errors

Do not introduce `Unknown` variants in domain errors. If something truly
unexpected happens, normalize it to `unexpected` with `core:internal` and
an explicit retry class.
