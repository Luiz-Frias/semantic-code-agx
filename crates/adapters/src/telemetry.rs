//! JSON telemetry adapter (counters, timers, and spans).

use crate::log_sink::LogSink;
use semantic_code_ports::{TelemetryPort, TelemetryTags, TelemetryTimer};
use semantic_code_shared::redaction::{REDACTED, is_secret_key};
use serde_json::Value;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Telemetry adapter that emits JSON lines.
#[derive(Clone)]
pub struct JsonTelemetry {
    sink: Arc<dyn LogSink>,
    base_tags: TelemetryTags,
    sampler: Arc<SpanSampler>,
}

impl JsonTelemetry {
    /// Create a telemetry adapter backed by the provided sink.
    #[must_use]
    pub fn new(sink: Arc<dyn LogSink>) -> Self {
        Self {
            sink,
            base_tags: TelemetryTags::new(),
            sampler: Arc::new(SpanSampler::new(1.0)),
        }
    }

    /// Set base tags applied to every metric.
    #[must_use]
    pub fn with_base_tags(mut self, tags: TelemetryTags) -> Self {
        self.base_tags = tags;
        self
    }

    /// Set span sample rate (0.0 - 1.0). Default is 1.0.
    #[must_use]
    pub fn with_span_sample_rate(mut self, rate: f64) -> Self {
        self.sampler = Arc::new(SpanSampler::new(rate));
        self
    }
}

impl TelemetryPort for JsonTelemetry {
    fn increment_counter(&self, name: &str, value: u64, tags: Option<&TelemetryTags>) {
        let tags = merge_tags(&self.base_tags, tags);
        let payload = metric_payload("counter", name, value, None, &tags);
        self.sink.write_line(&payload);
    }

    fn record_timer_ms(&self, name: &str, duration_ms: u64, tags: Option<&TelemetryTags>) {
        let tags = merge_tags(&self.base_tags, tags);
        let payload = metric_payload("timer", name, duration_ms, Some("ms"), &tags);
        self.sink.write_line(&payload);
    }

    fn start_timer(&self, name: &str, tags: Option<&TelemetryTags>) -> Box<dyn TelemetryTimer> {
        let tags = merge_tags(&self.base_tags, tags);
        let span_id = self.sampler.sample_span_id();
        if let Some(span_id) = span_id {
            let payload = span_payload("start", name, None, span_id, &tags);
            self.sink.write_line(&payload);
        }
        Box::new(JsonTelemetryTimer::new(
            Arc::clone(&self.sink),
            name.to_owned().into_boxed_str(),
            tags,
            span_id,
        ))
    }
}

/// Telemetry adapter that applies base tags to an inner telemetry sink.
#[derive(Clone)]
pub struct TaggedTelemetry {
    inner: Arc<dyn TelemetryPort>,
    tags: TelemetryTags,
}

impl TaggedTelemetry {
    /// Wrap a telemetry sink with base tags.
    #[must_use]
    pub fn new(inner: Arc<dyn TelemetryPort>, tags: TelemetryTags) -> Self {
        Self { inner, tags }
    }
}

impl TelemetryPort for TaggedTelemetry {
    fn increment_counter(&self, name: &str, value: u64, tags: Option<&TelemetryTags>) {
        let merged = merge_tags(&self.tags, tags);
        self.inner.increment_counter(name, value, Some(&merged));
    }

    fn record_timer_ms(&self, name: &str, duration_ms: u64, tags: Option<&TelemetryTags>) {
        let merged = merge_tags(&self.tags, tags);
        self.inner.record_timer_ms(name, duration_ms, Some(&merged));
    }

    fn start_timer(&self, name: &str, tags: Option<&TelemetryTags>) -> Box<dyn TelemetryTimer> {
        let merged = merge_tags(&self.tags, tags);
        self.inner.start_timer(name, Some(&merged))
    }
}

struct JsonTelemetryTimer {
    sink: Arc<dyn LogSink>,
    name: Box<str>,
    tags: TelemetryTags,
    span_id: Option<u64>,
    started_at: Instant,
    stopped: AtomicBool,
}

impl JsonTelemetryTimer {
    fn new(
        sink: Arc<dyn LogSink>,
        name: Box<str>,
        tags: TelemetryTags,
        span_id: Option<u64>,
    ) -> Self {
        Self {
            sink,
            name,
            tags,
            span_id,
            started_at: Instant::now(),
            stopped: AtomicBool::new(false),
        }
    }
}

impl TelemetryTimer for JsonTelemetryTimer {
    fn stop(&self) {
        if self.stopped.swap(true, Ordering::SeqCst) {
            return;
        }
        let duration_ms = self.started_at.elapsed().as_millis();
        let duration_ms = u64::try_from(duration_ms).unwrap_or_default();

        let metric = metric_payload("timer", &self.name, duration_ms, Some("ms"), &self.tags);
        self.sink.write_line(&metric);

        if let Some(span_id) = self.span_id {
            let payload = span_payload("end", &self.name, Some(duration_ms), span_id, &self.tags);
            self.sink.write_line(&payload);
        }
    }
}

struct SpanSampler {
    numerator: u64,
    denominator: u64,
    counter: AtomicU64,
}

impl SpanSampler {
    fn new(rate: f64) -> Self {
        let (numerator, denominator) = rate_fraction(rate);
        Self {
            numerator,
            denominator,
            counter: AtomicU64::new(1),
        }
    }

    fn sample_span_id(&self) -> Option<u64> {
        if self.numerator == 0 {
            return None;
        }
        let id = self.counter.fetch_add(1, Ordering::Relaxed);
        if self.numerator >= self.denominator {
            return Some(id);
        }
        if id % self.denominator < self.numerator {
            Some(id)
        } else {
            None
        }
    }
}

fn rate_fraction(rate: f64) -> (u64, u64) {
    let rate = if rate.is_finite() { rate } else { 1.0 };
    if rate <= 0.0 {
        return (0, 1);
    }
    if rate >= 1.0 {
        return (1, 1);
    }
    let rendered = format!("{rate:.6}");
    let mut parts = rendered.split('.');
    let int_part = parts.next().unwrap_or("0");
    let frac_part = parts.next().unwrap_or("");
    let frac_trimmed = frac_part.trim_end_matches('0');
    let scale = match frac_trimmed.len() {
        0 => 1,
        1 => 10,
        2 => 100,
        3 => 1_000,
        4 => 10_000,
        5 => 100_000,
        _ => 1_000_000,
    };
    let int_value = int_part.parse::<u64>().unwrap_or(0);
    let frac_value = if frac_trimmed.is_empty() {
        0
    } else {
        frac_trimmed.parse::<u64>().unwrap_or(0)
    };
    let numerator = int_value.saturating_mul(scale).saturating_add(frac_value);
    if numerator == 0 {
        (0, 1)
    } else {
        (numerator, scale)
    }
}

fn metric_payload(
    metric_type: &str,
    name: &str,
    value: u64,
    unit: Option<&str>,
    tags: &TelemetryTags,
) -> String {
    let mut payload = serde_json::Map::new();
    payload.insert("type".to_string(), Value::String("metric".to_string()));
    payload.insert("timestampMs".to_string(), Value::from(now_epoch_ms()));
    payload.insert(
        "metricType".to_string(),
        Value::String(metric_type.to_string()),
    );
    payload.insert("name".to_string(), Value::String(name.to_string()));
    payload.insert("value".to_string(), Value::from(value));
    if let Some(unit) = unit {
        payload.insert("unit".to_string(), Value::String(unit.to_string()));
    }
    if !tags.is_empty() {
        payload.insert("tags".to_string(), tags_to_json(tags));
    }
    to_line(payload)
}

fn span_payload(
    event: &str,
    name: &str,
    duration_ms: Option<u64>,
    span_id: u64,
    tags: &TelemetryTags,
) -> String {
    let mut payload = serde_json::Map::new();
    payload.insert("type".to_string(), Value::String("span".to_string()));
    payload.insert("timestampMs".to_string(), Value::from(now_epoch_ms()));
    payload.insert("event".to_string(), Value::String(event.to_string()));
    payload.insert("name".to_string(), Value::String(name.to_string()));
    payload.insert("spanId".to_string(), Value::from(span_id));
    if let Some(duration_ms) = duration_ms {
        payload.insert("durationMs".to_string(), Value::from(duration_ms));
    }
    if !tags.is_empty() {
        payload.insert("tags".to_string(), tags_to_json(tags));
    }
    to_line(payload)
}

fn to_line(payload: serde_json::Map<String, Value>) -> String {
    serde_json::to_string(&Value::Object(payload)).map_or_else(
        |_| {
            "{\"type\":\"metric\",\"metricType\":\"error\",\"name\":\"telemetry.serialize_failed\",\"value\":1}\n"
                .to_string()
        },
        |mut encoded| {
            encoded.push('\n');
            encoded
        },
    )
}

fn merge_tags(base: &TelemetryTags, extra: Option<&TelemetryTags>) -> TelemetryTags {
    if base.is_empty() && extra.is_none() {
        return TelemetryTags::new();
    }
    let mut merged = base.clone();
    if let Some(extra) = extra {
        for (key, value) in extra {
            merged.insert(key.clone(), value.clone());
        }
    }
    redact_tags(&mut merged);
    merged
}

fn tags_to_json(tags: &TelemetryTags) -> Value {
    let mut map = serde_json::Map::new();
    for (key, value) in tags {
        map.insert(key.to_string(), Value::String(value.to_string()));
    }
    Value::Object(map)
}

fn redact_tags(tags: &mut TelemetryTags) {
    for (key, value) in tags.iter_mut() {
        if is_secret_key(key) {
            *value = REDACTED.to_string().into_boxed_str();
        }
    }
}

fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log_sink::LogSink;
    use std::sync::Mutex;

    #[derive(Debug, Default)]
    struct MemorySink {
        lines: Mutex<Vec<String>>,
    }

    impl MemorySink {
        fn take(&self) -> Vec<String> {
            let mut guard = self.lines.lock().expect("memory sink lock");
            std::mem::take(&mut *guard)
        }
    }

    impl LogSink for MemorySink {
        fn write_line(&self, line: &str) {
            let mut guard = self.lines.lock().expect("memory sink lock");
            guard.push(line.to_string());
        }
    }

    #[test]
    fn telemetry_emits_counter_and_timer() -> Result<(), Box<dyn std::error::Error>> {
        let sink = Arc::new(MemorySink::default());
        let telemetry = JsonTelemetry::new(sink.clone()).with_span_sample_rate(1.0);

        telemetry.increment_counter("test.counter", 2, None);
        let timer = telemetry.start_timer("test.timer", None);
        timer.stop();

        let lines = sink.take();
        assert!(lines.len() >= 3);
        let parsed: Vec<Value> = lines
            .iter()
            .map(|line| serde_json::from_str(line.trim()))
            .collect::<Result<_, _>>()?;

        let counter = parsed
            .iter()
            .find(|value| value.get("metricType") == Some(&Value::String("counter".to_string())))
            .ok_or_else(|| "missing counter")?;
        assert_eq!(
            counter.get("name"),
            Some(&Value::String("test.counter".to_string()))
        );

        let timer = parsed
            .iter()
            .find(|value| value.get("metricType") == Some(&Value::String("timer".to_string())))
            .ok_or_else(|| "missing timer")?;
        assert_eq!(
            timer.get("name"),
            Some(&Value::String("test.timer".to_string()))
        );
        Ok(())
    }

    #[test]
    fn tagged_telemetry_merges_correlation_tag() {
        struct CaptureTelemetry {
            tags: Mutex<Option<TelemetryTags>>,
        }

        impl CaptureTelemetry {
            fn new() -> Self {
                Self {
                    tags: Mutex::new(None),
                }
            }
        }

        impl TelemetryPort for CaptureTelemetry {
            fn increment_counter(&self, _name: &str, _value: u64, tags: Option<&TelemetryTags>) {
                let mut guard = self.tags.lock().expect("tags lock");
                *guard = tags.cloned();
            }

            fn record_timer_ms(
                &self,
                _name: &str,
                _duration_ms: u64,
                tags: Option<&TelemetryTags>,
            ) {
                let mut guard = self.tags.lock().expect("tags lock");
                *guard = tags.cloned();
            }

            fn start_timer(
                &self,
                _name: &str,
                _tags: Option<&TelemetryTags>,
            ) -> Box<dyn TelemetryTimer> {
                Box::new(NoopTimer)
            }
        }

        struct NoopTimer;
        impl TelemetryTimer for NoopTimer {
            fn stop(&self) {}
        }

        let base = Arc::new(CaptureTelemetry::new());
        let mut tags = TelemetryTags::new();
        tags.insert(
            "correlationId".to_owned().into_boxed_str(),
            "req_456".to_owned().into_boxed_str(),
        );
        let telemetry = TaggedTelemetry::new(base.clone(), tags);
        telemetry.increment_counter("counter", 1, None);

        let captured = base.tags.lock().expect("tags lock").clone();
        let captured = captured.expect("tags missing");
        assert_eq!(
            captured.get("correlationId").map(|value| value.as_ref()),
            Some("req_456")
        );
    }
}
