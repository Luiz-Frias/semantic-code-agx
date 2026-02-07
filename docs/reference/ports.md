## Ports

The `ports` layer defines boundary contracts between the app/use-case layer and
adapters (filesystem, splitter, embedding, vector DB, sync, and observability).

### Rules

- Ports define **traits + DTOs only** (no provider logic, no business logic).
- Ports depend only on `domain` and `shared`.
- All I/O-touching methods accept a `shared::RequestContext` for **cancellation**
  and correlation.
- All fallible methods return `shared::Result<_, ErrorEnvelope>` (typed error
  envelope, no panics).

### Async strategy

Rust stable does not provide ergonomic `async fn` in traits for all use-cases.
For boundary traits we use **boxed futures**:

- `ports::BoxFuture<'a, T>`

This keeps the trait surface stable and object-safe. Ports should prefer
**batch APIs** for hot paths (e.g. `embed_batch`) to reduce allocation overhead.

### Error and cancellation policy

- Cancellation is represented as `core:cancelled` (`ErrorEnvelope::cancelled(...)`).
- Adapters should check `ctx.is_cancelled()` (or call `ctx.ensure_not_cancelled(...)`)
  at boundaries and before expensive work.
- Unexpected failures should be mapped to `ErrorEnvelope` with the correct
  retriable vs non-retriable class.
