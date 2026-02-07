# Observability

Phase 07 Milestone 3 adds structured JSON logging and telemetry adapters for
CLI workflows. Logs and metrics are emitted as **one JSON object per line** to
stderr.

## Enabling Structured Output

Set the following environment variables when running the CLI:

- `SCA_LOG_FORMAT=json` enables structured logs.
- `SCA_TELEMETRY_FORMAT=json` enables telemetry (defaults to log format when unset).
- `SCA_LOG_LEVEL=debug|info|warn|error` controls the minimum log level (default: `info`).
- `SCA_TRACE_SAMPLE_RATE=0.0-1.0` controls span sampling (default: `1.0`).

Example:

```bash
SCA_LOG_FORMAT=json SCA_TELEMETRY_FORMAT=json sca search --query "auth"
```

## Log Format

Each log line is a single JSON object with the following keys:

- `timestampMs` (u64): epoch milliseconds.
- `level`: `debug`, `info`, `warn`, or `error`.
- `event`: stable event name (e.g. `backend.search.completed`).
- `message`: human-readable message (already redacted).
- `fields`: optional structured fields (redacted when keys look secret).
- `error`: optional error payload (recursively redacted by secret key names).

Correlation IDs are injected into `fields.correlationId` for every CLI request.

## Telemetry Format

Telemetry emits counters and timers as JSON metrics:

- `type`: `metric`
- `metricType`: `counter` or `timer`
- `name`: metric name (e.g. `backend.search.total`)
- `value`: integer value (timers are in ms)
- `unit`: optional (for timers)
- `tags`: optional tags (redacted when keys look secret)

Timers also emit lightweight span events:

- `type`: `span`
- `event`: `start` or `end`
- `name`: span name
- `spanId`: numeric id
- `durationMs`: only on `end`

## Log Volume + Sampling Policy

Default behavior logs all events and emits all spans. For higher-volume
workloads, reduce span volume by setting `SCA_TRACE_SAMPLE_RATE`:

- `1.0` (default): sample every span
- `0.1`: sample ~10% of spans
- `0.0`: disable spans entirely (metrics still emit)

Recommended practice:

- Keep **error** logs unsampled.
- Sample **info/debug** spans for high-throughput indexing/search.
- Use metrics for aggregate visibility and spans for targeted tracing.

## Troubleshooting Runbook

1. **No logs appear**: ensure `SCA_LOG_FORMAT=json` is set for the CLI process.
2. **No telemetry**: set `SCA_TELEMETRY_FORMAT=json` (or rely on log format).
3. **Missing correlation IDs**: verify logs contain `fields.correlationId`; if
   missing, check the CLI is running with the Phase 07 observability adapters.
4. **Too much output**: reduce spans via `SCA_TRACE_SAMPLE_RATE` or increase
   log level with `SCA_LOG_LEVEL=warn`.
