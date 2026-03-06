//! Error formatting, NDJSON output, sanitization, and output writing.
//!
//! This module owns the [`CliOutput`] envelope that every command handler
//! returns, plus all helpers that serialize errors and summaries into the
//! three output formats (text, JSON, NDJSON).

use crate::error::{CliError, ExitCode};
use crate::format::OutputMode;
use semantic_code_facade::{ApiV1ErrorDto, ApiV1ErrorKind, InfraError, infra_error_to_api_v1};
use semantic_code_shared::is_secret_key;
use std::io::{self, Write};

pub struct CliOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: ExitCode,
}

pub fn format_error_output(mode: OutputMode, error: &InfraError, exit_code: ExitCode) -> CliOutput {
    let api_error = sanitize_api_error(infra_error_to_api_v1(error));

    let mut stderr = String::new();
    log_info(&mut stderr, "command failed", mode.no_progress);

    let stdout = if mode.is_ndjson() {
        format_ndjson_error(&api_error)
    } else if mode.is_json() {
        let payload = serde_json::json!({
            "status": "error",
            "error": api_error,
        });

        // This is a CLI boundary, so JSON serialization errors are internal.
        let mut output = serde_json::to_string_pretty(&payload).unwrap_or_else(|_| {
            "{\"status\":\"error\",\"error\":{\"code\":\"ERR_CORE_INTERNAL\",\"message\":\"internal error\",\"kind\":\"INVARIANT\"}}".to_string()
        });
        output.push('\n');
        output
    } else {
        format_api_error_text(&api_error)
    };

    CliOutput {
        stdout,
        stderr,
        exit_code,
    }
}

pub fn infra_exit_code(error: &InfraError) -> ExitCode {
    match infra_error_to_api_v1(error).kind {
        ApiV1ErrorKind::Expected => ExitCode::InvalidInput,
        ApiV1ErrorKind::Invariant => ExitCode::Internal,
    }
}

pub fn sanitize_api_error(mut error: ApiV1ErrorDto) -> ApiV1ErrorDto {
    if let Some(meta) = error.meta.as_mut() {
        for (key, value) in meta.iter_mut() {
            if is_secret_key(key) {
                *value = "<redacted>".to_string();
            }
        }
    }
    error
}

fn format_api_error_text(error: &ApiV1ErrorDto) -> String {
    let mut out = String::new();
    out.push_str("status: error\n");
    out.push_str("code: ");
    out.push_str(&error.code);
    out.push('\n');
    out.push_str("message: ");
    out.push_str(&error.message);
    out.push('\n');
    out.push_str("kind: ");
    out.push_str(match error.kind {
        ApiV1ErrorKind::Expected => "EXPECTED",
        ApiV1ErrorKind::Invariant => "INVARIANT",
    });
    out.push('\n');

    if let Some(meta) = error.meta.as_ref()
        && !meta.is_empty()
    {
        out.push_str("meta:\n");
        for (key, value) in meta {
            out.push_str("  ");
            out.push_str(key);
            out.push_str(": ");
            out.push_str(value);
            out.push('\n');
        }
    }

    out
}

pub fn log_info(stderr: &mut String, message: &str, no_progress: bool) {
    if no_progress {
        return;
    }
    stderr.push_str("info: ");
    stderr.push_str(message);
    stderr.push('\n');
}

pub fn format_ndjson_summary(status: &str, kind: &str, extra: Option<serde_json::Value>) -> String {
    let mut payload = serde_json::Map::new();
    payload.insert(
        "type".to_string(),
        serde_json::Value::String("summary".to_string()),
    );
    payload.insert(
        "status".to_string(),
        serde_json::Value::String(status.to_string()),
    );
    payload.insert(
        "kind".to_string(),
        serde_json::Value::String(kind.to_string()),
    );
    if let Some(serde_json::Value::Object(map)) = extra {
        for (key, value) in map {
            payload.insert(key, value);
        }
    }
    let mut out = serde_json::to_string(&serde_json::Value::Object(payload)).unwrap_or_else(|_| {
        "{\"type\":\"summary\",\"status\":\"error\",\"kind\":\"internal\"}".to_string()
    });
    out.push('\n');
    out
}

pub fn format_ndjson_error(error: &ApiV1ErrorDto) -> String {
    let payload = serde_json::json!({
        "type": "error",
        "status": "error",
        "error": error,
    });
    let mut out = serde_json::to_string(&payload).unwrap_or_else(|_| {
        "{\"type\":\"error\",\"status\":\"error\",\"error\":{\"code\":\"ERR_CORE_INTERNAL\",\"message\":\"internal error\",\"kind\":\"INVARIANT\"}}".to_string()
    });
    out.push('\n');
    out
}

pub fn write_output(output: &CliOutput) -> Result<(), CliError> {
    let mut stdout = io::stdout();
    stdout.write_all(output.stdout.as_bytes())?;

    if !output.stderr.is_empty() {
        let mut stderr = io::stderr();
        stderr.write_all(output.stderr.as_bytes())?;
        stderr.flush()?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_facade::ApiV1ErrorKind;
    use std::collections::BTreeMap;

    #[test]
    fn error_formatting_redacts_sensitive_meta_keys() {
        let error = ApiV1ErrorDto {
            code: "ERR_CONFIG_INVALID_ENV_URL".to_string(),
            message: "bad env".to_string(),
            kind: ApiV1ErrorKind::Expected,
            meta: Some(BTreeMap::from([
                ("apiKey".to_string(), "secret-value".to_string()),
                ("field".to_string(), "timeoutMs".to_string()),
            ])),
        };

        let sanitized = sanitize_api_error(error);
        let meta = sanitized.meta.expect("meta should be present");
        assert_eq!(meta.get("apiKey").map(String::as_str), Some("<redacted>"));
        assert_eq!(meta.get("field").map(String::as_str), Some("timeoutMs"));
    }

    #[test]
    fn log_info_respects_no_progress() {
        let mut stderr = String::new();
        log_info(&mut stderr, "message", true);
        assert!(stderr.is_empty());
    }
}
