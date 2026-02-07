//! Telemetry boundary contract (metrics + timings).

use std::collections::BTreeMap;

/// Telemetry tags. Keep tags low-cardinality.
pub type TelemetryTags = BTreeMap<Box<str>, Box<str>>;

/// Timer handle.
pub trait TelemetryTimer: Send + Sync {
    /// Stop the timer and record its duration.
    fn stop(&self);
}

/// Boundary contract for telemetry (metrics + timings).
pub trait TelemetryPort: Send + Sync {
    /// Increment a counter by `value` (default should be 1 at call sites).
    fn increment_counter(&self, name: &str, value: u64, tags: Option<&TelemetryTags>);

    /// Record a duration (in milliseconds) for an operation.
    fn record_timer_ms(&self, name: &str, duration_ms: u64, tags: Option<&TelemetryTags>);

    /// Start a timer and return a handle that records on `stop()`.
    fn start_timer(&self, name: &str, tags: Option<&TelemetryTags>) -> Box<dyn TelemetryTimer>;
}
