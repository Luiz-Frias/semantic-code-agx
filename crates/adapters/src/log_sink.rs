//! Log sink helpers for observability adapters.

use std::io::Write;

/// A sink that receives pre-formatted log lines.
pub trait LogSink: Send + Sync {
    /// Write a line to the sink.
    fn write_line(&self, line: &str);
}

/// Log sink that writes to stderr.
#[derive(Debug, Default)]
pub struct StderrLogSink;

impl LogSink for StderrLogSink {
    fn write_line(&self, line: &str) {
        let mut stderr = std::io::stderr();
        if let Err(error) = stderr.write_all(line.as_bytes()) {
            eprintln!("log sink write failed: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LogSink;
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
    fn memory_sink_captures_lines() {
        let sink = MemorySink::default();
        sink.write_line("hello\n");
        sink.write_line("world\n");

        let lines = sink.take();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "hello\n");
        assert_eq!(lines[1], "world\n");
    }
}
