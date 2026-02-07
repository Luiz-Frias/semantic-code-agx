//! Structured JSON logger adapter.

use crate::log_sink::LogSink;
use semantic_code_ports::{LogEvent, LogFields, LogLevel, LoggerPort};
use semantic_code_shared::redaction::{REDACTED, is_secret_key};
use serde_json::Value;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// JSON logger emitting one line per event.
#[derive(Clone)]
pub struct JsonLogger {
    sink: Arc<dyn LogSink>,
    base_fields: LogFields,
    min_level: LogLevel,
}

impl JsonLogger {
    /// Create a JSON logger backed by the provided sink.
    #[must_use]
    pub fn new(sink: Arc<dyn LogSink>) -> Self {
        Self {
            sink,
            base_fields: LogFields::new(),
            min_level: LogLevel::Info,
        }
    }

    /// Set base fields applied to every event.
    #[must_use]
    pub fn with_base_fields(mut self, fields: LogFields) -> Self {
        self.base_fields = fields;
        self
    }

    /// Set the minimum log level.
    #[must_use]
    pub const fn with_min_level(mut self, level: LogLevel) -> Self {
        self.min_level = level;
        self
    }
}

impl LoggerPort for JsonLogger {
    fn log(&self, event: LogEvent) {
        if !should_log(self.min_level, event.level) {
            return;
        }

        let mut fields = self.base_fields.clone();
        if let Some(extra) = event.fields {
            for (key, value) in extra {
                fields.insert(key, value);
            }
        }
        redact_fields(&mut fields);

        let mut error = event.error;
        if let Some(ref mut value) = error {
            redact_value(value);
        }

        let mut payload = serde_json::Map::new();
        payload.insert("timestampMs".to_string(), Value::from(now_epoch_ms()));
        payload.insert("level".to_string(), Value::String(level_str(event.level)));
        payload.insert("event".to_string(), Value::String(event.event.to_string()));
        payload.insert(
            "message".to_string(),
            Value::String(event.message.to_string()),
        );
        if !fields.is_empty() {
            payload.insert("fields".to_string(), fields_to_json(&fields));
        }
        if let Some(error) = error {
            payload.insert("error".to_string(), error);
        }

        let line = serde_json::to_string(&Value::Object(payload)).map_or_else(
            |_| {
                "{\"timestampMs\":0,\"level\":\"error\",\"event\":\"logger.serialize_failed\",\"message\":\"log serialization failed\"}\n"
                    .to_string()
            },
            |mut encoded| {
                encoded.push('\n');
                encoded
            },
        );
        self.sink.write_line(&line);
    }

    fn child(&self, fields: LogFields) -> Box<dyn LoggerPort> {
        let mut merged = self.base_fields.clone();
        for (key, value) in fields {
            merged.insert(key, value);
        }
        Box::new(Self {
            sink: Arc::clone(&self.sink),
            base_fields: merged,
            min_level: self.min_level,
        })
    }
}

const fn should_log(min_level: LogLevel, level: LogLevel) -> bool {
    level_rank(level) >= level_rank(min_level)
}

const fn level_rank(level: LogLevel) -> u8 {
    match level {
        LogLevel::Debug => 10,
        LogLevel::Info => 20,
        LogLevel::Warn => 30,
        LogLevel::Error => 40,
    }
}

fn level_str(level: LogLevel) -> String {
    match level {
        LogLevel::Debug => "debug".to_string(),
        LogLevel::Info => "info".to_string(),
        LogLevel::Warn => "warn".to_string(),
        LogLevel::Error => "error".to_string(),
    }
}

fn fields_to_json(fields: &LogFields) -> Value {
    let mut map = serde_json::Map::new();
    for (key, value) in fields {
        map.insert(key.to_string(), value.clone());
    }
    Value::Object(map)
}

fn redact_fields(fields: &mut LogFields) {
    for (key, value) in fields.iter_mut() {
        if is_secret_key(key) {
            *value = Value::String(REDACTED.to_string());
        } else {
            redact_value(value);
        }
    }
}

fn redact_value(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for (key, nested) in map.iter_mut() {
                if is_secret_key(key) {
                    *nested = Value::String(REDACTED.to_string());
                } else {
                    redact_value(nested);
                }
            }
        },
        Value::Array(items) => {
            for item in items {
                redact_value(item);
            }
        },
        _ => {},
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
    use serde_json::json;
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
    fn json_logger_redacts_sensitive_fields() -> Result<(), Box<dyn std::error::Error>> {
        let sink = Arc::new(MemorySink::default());
        let logger = JsonLogger::new(sink.clone()).with_min_level(LogLevel::Debug);

        let mut fields = LogFields::new();
        fields.insert(
            "apiKey".to_owned().into_boxed_str(),
            Value::String("secret".to_string()),
        );
        fields.insert(
            "safe".to_owned().into_boxed_str(),
            Value::String("ok".to_string()),
        );

        logger.log(LogEvent {
            event: "test.event".into(),
            level: LogLevel::Info,
            message: "testing".into(),
            fields: Some(fields),
            error: Some(json!({ // pragma: allowlist secret
                "token": "should-hide",
                "nested": { "password": "nope", "value": 7 } // pragma: allowlist secret
            })),
        });

        let lines = sink.take();
        assert_eq!(lines.len(), 1);
        let payload: Value = serde_json::from_str(lines[0].trim())?;
        let fields = payload
            .get("fields")
            .and_then(Value::as_object)
            .ok_or_else(|| "missing fields")?;
        assert_eq!(
            fields.get("apiKey"),
            Some(&Value::String(REDACTED.to_string()))
        );
        assert_eq!(fields.get("safe"), Some(&Value::String("ok".to_string())));

        let error = payload
            .get("error")
            .and_then(Value::as_object)
            .ok_or_else(|| "missing error")?;
        assert_eq!(
            error.get("token"),
            Some(&Value::String(REDACTED.to_string()))
        );
        let nested = error
            .get("nested")
            .and_then(Value::as_object)
            .ok_or_else(|| "missing nested")?;
        assert_eq!(
            nested.get("password"),
            Some(&Value::String(REDACTED.to_string()))
        );
        assert_eq!(nested.get("value"), Some(&Value::from(7)));
        Ok(())
    }

    #[test]
    fn child_logger_merges_fields() -> Result<(), Box<dyn std::error::Error>> {
        let sink = Arc::new(MemorySink::default());
        let logger = JsonLogger::new(sink.clone()).with_min_level(LogLevel::Debug);

        let mut base = LogFields::new();
        base.insert(
            "correlationId".to_owned().into_boxed_str(),
            Value::String("req_123".to_string()),
        );
        let child = logger.child(base);
        child.info("test.child", "child log", None);

        let lines = sink.take();
        assert_eq!(lines.len(), 1);
        let payload: Value = serde_json::from_str(lines[0].trim())?;
        let fields = payload
            .get("fields")
            .and_then(Value::as_object)
            .ok_or_else(|| "missing fields")?;
        assert_eq!(
            fields.get("correlationId"),
            Some(&Value::String("req_123".to_string()))
        );
        Ok(())
    }
}
