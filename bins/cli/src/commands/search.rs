//! Search command handler.

use crate::error::{CliError, ExitCode};
use crate::format::OutputMode;
use crate::{CliOutput, format_error_output, infra_exit_code};
use semantic_code_config::{SearchRequestDto, validate_search_request};
use semantic_code_facade::{SearchResult, run_search_local};
use std::fmt::Write;
use std::path::Path;

/// Inputs for search command execution.
pub struct SearchCommandInput<'a> {
    pub config_path: Option<&'a Path>,
    pub overrides_json: Option<&'a str>,
    pub codebase_root: &'a Path,
    pub query: &'a str,
    pub top_k: Option<u32>,
    pub threshold: Option<f32>,
    pub filter_expr: Option<&'a str>,
    pub include_content: bool,
}

/// Run the search command.
pub fn run_search(mode: OutputMode, input: &SearchCommandInput<'_>) -> Result<CliOutput, CliError> {
    let request = SearchRequestDto {
        codebase_root: input.codebase_root.to_string_lossy().to_string(),
        query: input.query.to_string(),
        top_k: input.top_k,
        threshold: input.threshold.map(f64::from),
        filter_expr: input.filter_expr.map(str::to_owned),
        include_content: input.include_content.then_some(true),
    };
    let request = match validate_search_request(&request) {
        Ok(request) => request,
        Err(error) => return Ok(format_error_output(mode, &error, infra_exit_code(&error))),
    };

    match run_search_local(input.config_path, input.overrides_json, &request) {
        Ok(results) => format_search_output(mode, &results),
        Err(error) => Ok(format_error_output(mode, &error, infra_exit_code(&error))),
    }
}

fn format_search_output(mode: OutputMode, results: &[SearchResult]) -> Result<CliOutput, CliError> {
    let stdout = if mode.is_ndjson() {
        format_search_ndjson(results)?
    } else if mode.is_json() {
        format_search_json(results)?
    } else {
        format_search_text(results)
    };

    Ok(CliOutput {
        stdout,
        stderr: String::new(),
        exit_code: ExitCode::Ok,
    })
}

fn format_search_json(results: &[SearchResult]) -> Result<String, CliError> {
    let payload = serde_json::json!({
        "status": "ok",
        "results": results,
    });
    let mut out = serde_json::to_string_pretty(&payload)?;
    out.push('\n');
    Ok(out)
}

fn format_search_ndjson(results: &[SearchResult]) -> Result<String, CliError> {
    let mut out = String::new();
    for result in results {
        let payload = serde_json::json!({
            "type": "result",
            "relativePath": result.key.relative_path.as_ref(),
            "startLine": result.key.span.start_line(),
            "endLine": result.key.span.end_line(),
            "score": result.score,
            "content": result.content,
        });
        let line = serde_json::to_string(&payload)?;
        out.push_str(&line);
        out.push('\n');
    }
    let summary = serde_json::json!({
        "type": "summary",
        "status": "ok",
        "count": results.len(),
    });
    let line = serde_json::to_string(&summary)?;
    out.push_str(&line);
    out.push('\n');
    Ok(out)
}

fn format_search_text(results: &[SearchResult]) -> String {
    let mut out = String::new();
    out.push_str("status: ok\n");
    out.push_str("results: ");
    out.push_str(&results.len().to_string());
    out.push('\n');

    for result in results {
        let start = result.key.span.start_line();
        let end = result.key.span.end_line();
        out.push_str(result.key.relative_path.as_ref());
        out.push(':');
        out.push_str(&start.to_string());
        out.push('-');
        out.push_str(&end.to_string());
        out.push_str(" score=");
        let _ = write!(&mut out, "{:.4}", result.score);
        out.push('\n');
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndjson_output_includes_summary_line() -> Result<(), CliError> {
        let result: SearchResult = serde_json::from_value(serde_json::json!({
            "key": {
                "relativePath": "src/lib.rs",
                "span": { "startLine": 1, "endLine": 2 }
            },
            "content": "fn main() {}",
            "score": 0.42
        }))?;
        let output = format_search_ndjson(&[result])?;
        let lines: Vec<&str> = output.lines().collect();
        assert!(lines.len() >= 2);
        let summary: serde_json::Value =
            serde_json::from_str(lines.last().unwrap()).map_err(std::io::Error::other)?;
        assert_eq!(
            summary.get("type").and_then(|v| v.as_str()),
            Some("summary")
        );
        assert_eq!(summary.get("status").and_then(|v| v.as_str()), Some("ok"));
        Ok(())
    }
}
