//! Structured logging boundary contract.

use std::collections::BTreeMap;

/// Log level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Debug.
    Debug,
    /// Info.
    Info,
    /// Warn.
    Warn,
    /// Error.
    Error,
}

/// Additional event fields.
pub type LogFields = BTreeMap<Box<str>, serde_json::Value>;

/// Structured log event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogEvent {
    /// Stable event name.
    pub event: Box<str>,
    /// Severity.
    pub level: LogLevel,
    /// Human-readable message (safe, redacted).
    pub message: Box<str>,
    /// Optional structured fields.
    pub fields: Option<LogFields>,
    /// Optional error payload.
    pub error: Option<serde_json::Value>,
}

/// Boundary contract for structured logging.
pub trait LoggerPort: Send + Sync {
    /// Emit a structured event.
    fn log(&self, event: LogEvent);

    /// Create a child logger with base fields applied to every event.
    fn child(&self, fields: LogFields) -> Box<dyn LoggerPort>;

    /// Convenience: debug event.
    fn debug(&self, event: &str, message: &str, fields: Option<LogFields>) {
        self.log(LogEvent {
            event: event.to_owned().into_boxed_str(),
            level: LogLevel::Debug,
            message: message.to_owned().into_boxed_str(),
            fields,
            error: None,
        });
    }

    /// Convenience: info event.
    fn info(&self, event: &str, message: &str, fields: Option<LogFields>) {
        self.log(LogEvent {
            event: event.to_owned().into_boxed_str(),
            level: LogLevel::Info,
            message: message.to_owned().into_boxed_str(),
            fields,
            error: None,
        });
    }

    /// Convenience: warn event.
    fn warn(&self, event: &str, message: &str, fields: Option<LogFields>) {
        self.log(LogEvent {
            event: event.to_owned().into_boxed_str(),
            level: LogLevel::Warn,
            message: message.to_owned().into_boxed_str(),
            fields,
            error: None,
        });
    }

    /// Convenience: error event.
    fn error(&self, event: &str, message: &str, fields: Option<LogFields>) {
        self.log(LogEvent {
            event: event.to_owned().into_boxed_str(),
            level: LogLevel::Error,
            message: message.to_owned().into_boxed_str(),
            fields,
            error: None,
        });
    }
}
