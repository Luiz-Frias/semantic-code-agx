//! Redacting writer for JSON tracing output.
//!
//! This module provides a `MakeWriter` implementation that intercepts each
//! JSON log line, redacts sensitive keys recursively, and writes sanitized
//! output to stderr.

use semantic_code_shared::{REDACTED, is_secret_key};
use serde_json::Value;
use std::io::{self, Write};
use tracing_subscriber::fmt::MakeWriter;

/// A tracing writer factory that redacts sensitive JSON fields before writing.
#[derive(Debug, Clone, Copy, Default)]
pub struct RedactingStderrMakeWriter;

impl<'writer> MakeWriter<'writer> for RedactingStderrMakeWriter {
    type Writer = RedactingWriter<io::StderrLock<'writer>>;

    fn make_writer(&'writer self) -> Self::Writer {
        RedactingWriter::new(io::stderr().lock())
    }
}

/// A buffered line writer that redacts secret-like JSON keys.
#[derive(Debug)]
pub struct RedactingWriter<W> {
    inner: W,
    buffer: Vec<u8>,
}

impl<W> RedactingWriter<W>
where
    W: Write,
{
    const fn new(inner: W) -> Self {
        Self {
            inner,
            buffer: Vec::new(),
        }
    }

    fn drain_complete_lines(&mut self) -> io::Result<()> {
        while let Some(newline_pos) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line = self.buffer.drain(..=newline_pos).collect::<Vec<_>>();
            let had_newline = matches!(line.last(), Some(b'\n'));
            if had_newline {
                let _ = line.pop();
            }
            self.write_line(&line, had_newline)?;
        }
        Ok(())
    }

    fn write_line(&mut self, line: &[u8], append_newline: bool) -> io::Result<()> {
        if line.is_empty() {
            if append_newline {
                self.inner.write_all(b"\n")?;
            }
            return Ok(());
        }

        match serde_json::from_slice::<Value>(line) {
            Ok(mut payload) => {
                redact_json_value(&mut payload);
                match serde_json::to_vec(&payload) {
                    Ok(mut encoded) => {
                        if append_newline {
                            encoded.push(b'\n');
                        }
                        self.inner.write_all(&encoded)
                    },
                    Err(_) => self.write_raw(line, append_newline),
                }
            },
            Err(_) => self.write_raw(line, append_newline),
        }
    }

    fn write_raw(&mut self, line: &[u8], append_newline: bool) -> io::Result<()> {
        self.inner.write_all(line)?;
        if append_newline {
            self.inner.write_all(b"\n")?;
        }
        Ok(())
    }
}

impl<W> Write for RedactingWriter<W>
where
    W: Write,
{
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        self.drain_complete_lines()?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if !self.buffer.is_empty() {
            let remaining = std::mem::take(&mut self.buffer);
            self.write_line(&remaining, false)?;
        }
        self.inner.flush()
    }
}

fn redact_json_value(value: &mut Value) {
    match value {
        Value::Object(object) => {
            for (key, nested) in object.iter_mut() {
                if is_secret_key(key) {
                    *nested = Value::String(REDACTED.to_string());
                } else {
                    redact_json_value(nested);
                }
            }
        },
        Value::Array(items) => {
            for item in items {
                redact_json_value(item);
            }
        },
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {},
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_nested_secret_keys() {
        let mut payload = serde_json::json!({
            "fields": {
                "api_key": "fixture-secret", // pragma: allowlist secret
                "level": "info"
            },
            "spans": [
                {
                    "password": "fixture-secret" // pragma: allowlist secret
                }
            ]
        });

        redact_json_value(&mut payload);

        assert_eq!(
            payload["fields"]["api_key"],
            Value::String(REDACTED.to_owned())
        );
        assert_eq!(payload["fields"]["level"], Value::String("info".to_owned()));
        assert_eq!(
            payload["spans"][0]["password"],
            Value::String(REDACTED.to_owned())
        );
    }
}
