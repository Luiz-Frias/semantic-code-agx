//! Search command handler.

use crate::error::{CliError, ExitCode};
use crate::format::OutputMode;
use crate::vector_kernel::{
    VectorKernelMetadata, resolve_vector_kernel_metadata_std_env, warn_if_experimental,
};
use crate::{CliOutput, format_error_output, infra_exit_code};
use semantic_code_facade::{SearchOutput, run_search_local, validate_search_request_for_query};
use std::fmt::Write;
use std::io::{self, BufRead};
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
    let request = match validate_search_request_for_query(
        input.codebase_root,
        input.query,
        input.top_k,
        input.threshold,
        input.filter_expr,
        input.include_content,
    ) {
        Ok(request) => request,
        Err(error) => return Ok(format_error_output(mode, &error, infra_exit_code(&error))),
    };
    let vector_kernel =
        match resolve_vector_kernel_metadata_std_env(input.config_path, input.overrides_json) {
            Ok(metadata) => metadata,
            Err(error) => return Ok(format_error_output(mode, &error, infra_exit_code(&error))),
        };
    warn_if_experimental(vector_kernel);

    match run_search_local(input.config_path, input.overrides_json, &request) {
        Ok(output) => format_search_output(mode, &output, vector_kernel),
        Err(error) => Ok(format_error_output(mode, &error, infra_exit_code(&error))),
    }
}

fn format_search_output(
    mode: OutputMode,
    output: &SearchOutput,
    vector_kernel: VectorKernelMetadata,
) -> Result<CliOutput, CliError> {
    let stdout = if mode.is_ndjson() {
        format_search_ndjson(output, vector_kernel)?
    } else if mode.is_json() {
        format_search_json(output, vector_kernel)?
    } else {
        format_search_text(output)?
    };

    Ok(CliOutput {
        stdout,
        stderr: String::new(),
        exit_code: ExitCode::Ok,
    })
}

fn format_search_json(
    output: &SearchOutput,
    vector_kernel: VectorKernelMetadata,
) -> Result<String, CliError> {
    let mut payload = serde_json::Map::new();
    payload.insert("status".into(), serde_json::Value::String("ok".into()));
    payload.insert("results".into(), serde_json::to_value(&output.results)?);
    payload.insert("vectorKernel".into(), vector_kernel.as_json());
    if let Some(stats) = output.stats.as_ref() {
        payload.insert("searchStats".into(), serde_json::to_value(stats)?);
    }
    let mut out = serde_json::to_string_pretty(&serde_json::Value::Object(payload))?;
    out.push('\n');
    Ok(out)
}

fn format_search_ndjson(
    output: &SearchOutput,
    vector_kernel: VectorKernelMetadata,
) -> Result<String, CliError> {
    let mut out = String::new();
    for result in &output.results {
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
    let mut summary = serde_json::Map::new();
    summary.insert("type".into(), serde_json::Value::String("summary".into()));
    summary.insert("status".into(), serde_json::Value::String("ok".into()));
    summary.insert(
        "count".into(),
        serde_json::Value::from(output.results.len()),
    );
    summary.insert("vectorKernel".into(), vector_kernel.as_json());
    if let Some(stats) = output.stats.as_ref() {
        summary.insert("searchStats".into(), serde_json::to_value(stats)?);
    }
    let line = serde_json::to_string(&serde_json::Value::Object(summary))?;
    out.push_str(&line);
    out.push('\n');
    Ok(out)
}

fn format_search_text(output: &SearchOutput) -> Result<String, CliError> {
    let mut out = String::new();
    out.push_str("status: ok\n");
    out.push_str("results: ");
    out.push_str(&output.results.len().to_string());
    out.push('\n');

    for result in &output.results {
        let start = result.key.span.start_line();
        let end = result.key.span.end_line();
        out.push_str(result.key.relative_path.as_ref());
        out.push(':');
        out.push_str(&start.to_string());
        out.push('-');
        out.push_str(&end.to_string());
        out.push_str(" score=");
        if let Err(error) = write!(&mut out, "{:.4}", result.score) {
            return Err(CliError::Io(io::Error::other(error.to_string())));
        }
        out.push('\n');
    }

    Ok(out)
}

// ── stdin-batch mode ─────────────────────────────────────────────────────────

/// Parse a batch query from a JSON value.
///
/// Expected shape:
/// - Basic:      `{"query":"...","topK":10,"threshold":0.0}`
/// - Pre-embed:  `{"query":"label","queryVector":[0.1,0.2,...],"topK":10}`
/// - Warmup:     `{"query":"...","returnQueryVector":true}`
struct BatchQuery {
    query: String,
    top_k: Option<u32>,
    threshold: Option<f32>,
    /// Pre-computed embedding; skips inference when present.
    query_vector: Option<Vec<f32>>,
    /// When true, include the query embedding in the response.
    return_query_vector: bool,
}

impl BatchQuery {
    fn from_json(value: &serde_json::Value) -> Option<Self> {
        let query = value.get("query")?.as_str()?.to_owned();
        let top_k = value
            .get("topK")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| u32::try_from(v).ok());
        let threshold = value
            .get("threshold")
            .and_then(serde_json::Value::as_f64)
            .map(|v| {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "JSON f64 threshold intentionally narrowed to f32"
                )]
                let t = v as f32;
                t
            });
        let query_vector = value.get("queryVector").and_then(|arr| {
            arr.as_array().map(|items| {
                items
                    .iter()
                    .filter_map(|v| {
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "JSON f64 embedding components intentionally narrowed to f32"
                        )]
                        v.as_f64().map(|f| f as f32)
                    })
                    .collect()
            })
        });
        let return_query_vector = value
            .get("returnQueryVector")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        Some(Self {
            query,
            top_k,
            threshold,
            query_vector,
            return_query_vector,
        })
    }
}

/// Run search in stdin-batch mode: load the index once, process NDJSON queries
/// from stdin, write NDJSON results to stdout.
///
/// Protocol:
/// - Input (stdin):  one JSON object per line `{"query":"...","topK":10,"threshold":0.0}`
/// - Output (stdout): one JSON object per line `{"status":"ok","results":[...],...}`
/// - On error: `{"status":"error","message":"..."}`
/// - On EOF: process exits with code 0
pub fn run_search_stdin_batch(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    codebase_root: &Path,
    default_top_k: Option<u32>,
    default_threshold: Option<f32>,
    include_content: bool,
) -> Result<CliOutput, CliError> {
    let session =
        semantic_code_facade::open_search_session(config_path, overrides_json, codebase_root)
            .map_err(|error| CliError::Io(io::Error::other(error.to_string())))?;

    let vector_kernel = resolve_vector_kernel_metadata_std_env(config_path, overrides_json)
        .unwrap_or_else(|_| {
            VectorKernelMetadata::new(semantic_code_facade::CliVectorKernelKind::HnswRs)
        });
    warn_if_experimental(vector_kernel);

    let stdin = io::stdin().lock();
    let mut stdout = io::stdout().lock();
    let mut count: u64 = 0;

    for line_result in stdin.lines() {
        let line = line_result?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let json_value: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(parsed) => parsed,
            Err(error) => {
                write_batch_error(&mut stdout, &format!("invalid JSON: {error}"))?;
                continue;
            },
        };
        let Some(batch_query) = BatchQuery::from_json(&json_value) else {
            write_batch_error(&mut stdout, "missing required field: \"query\"")?;
            continue;
        };

        let top_k = batch_query.top_k.or(default_top_k);
        let threshold = batch_query.threshold.or(default_threshold);

        // If the caller wants the query vector back, embed first so we can
        // include it in the response (warmup pass pattern).
        let query_vector = if batch_query.return_query_vector && batch_query.query_vector.is_none()
        {
            match session.embed(&batch_query.query) {
                Ok(vec) => Some(vec),
                Err(error) => {
                    write_batch_error(&mut stdout, &error.to_string())?;
                    continue;
                },
            }
        } else {
            batch_query.query_vector
        };

        let search_result = if let Some(ref vec) = query_vector {
            session.search_with_vector(&batch_query.query, vec.clone(), top_k, threshold)
        } else {
            session.search(&batch_query.query, top_k, threshold)
        };

        match search_result {
            Ok(output) => {
                let return_vec = if batch_query.return_query_vector {
                    query_vector.as_deref()
                } else {
                    None
                };
                write_batch_result(
                    &mut stdout,
                    &output,
                    vector_kernel,
                    include_content,
                    return_vec,
                )?;
            },
            Err(error) => {
                write_batch_error(&mut stdout, &error.to_string())?;
            },
        }
        count += 1;
    }

    Ok(CliOutput {
        stdout: String::new(),
        stderr: format!("stdin-batch: processed {count} queries\n"),
        exit_code: ExitCode::Ok,
    })
}

fn write_batch_result(
    writer: &mut impl io::Write,
    output: &SearchOutput,
    vector_kernel: VectorKernelMetadata,
    _include_content: bool,
    query_vector: Option<&[f32]>,
) -> Result<(), CliError> {
    // Reuse the same JSON shape as `format_search_json` (nested key/span)
    // so consumers (e.g. bench-runner) can parse both formats identically.
    let mut payload = serde_json::Map::new();
    payload.insert("status".into(), serde_json::Value::String("ok".into()));
    payload.insert("results".into(), serde_json::to_value(&output.results)?);
    payload.insert("vectorKernel".into(), vector_kernel.as_json());
    if let Some(stats) = output.stats.as_ref() {
        payload.insert("searchStats".into(), serde_json::to_value(stats)?);
    }
    if let Some(vec) = query_vector {
        payload.insert("queryVector".into(), serde_json::to_value(vec)?);
    }

    serde_json::to_writer(&mut *writer, &serde_json::Value::Object(payload))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn write_batch_error(writer: &mut impl io::Write, message: &str) -> Result<(), CliError> {
    let payload = serde_json::json!({
        "status": "error",
        "message": message,
    });
    serde_json::to_writer(&mut *writer, &payload)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndjson_output_includes_summary_line() -> Result<(), CliError> {
        let result: semantic_code_facade::SearchResult =
            serde_json::from_value(serde_json::json!({
                "key": {
                    "relativePath": "src/lib.rs",
                    "span": { "startLine": 1, "endLine": 2 }
                },
                "content": "fn main() {}",
                "score": 0.42
            }))?;
        let output = SearchOutput {
            results: vec![result],
            stats: None,
        };
        let output = format_search_ndjson(
            &output,
            VectorKernelMetadata::new(semantic_code_facade::CliVectorKernelKind::HnswRs),
        )?;
        let lines: Vec<&str> = output.lines().collect();
        assert!(lines.len() >= 2);
        let summary: serde_json::Value =
            serde_json::from_str(lines.last().unwrap()).map_err(io::Error::other)?;
        assert_eq!(
            summary.get("type").and_then(|v| v.as_str()),
            Some("summary")
        );
        assert_eq!(summary.get("status").and_then(|v| v.as_str()), Some("ok"));
        assert_eq!(
            summary
                .get("vectorKernel")
                .and_then(|v| v.get("effective"))
                .and_then(|v| v.as_str()),
            Some("hnsw-rs")
        );
        assert!(summary.get("searchStats").is_none());
        Ok(())
    }

    #[test]
    fn ndjson_summary_includes_search_stats_when_present() -> Result<(), CliError> {
        let output = SearchOutput {
            results: Vec::new(),
            stats: Some(semantic_code_facade::SearchStats {
                expansions: Some(7),
                kernel: "hnsw-rs".into(),
                extra: std::collections::BTreeMap::new(),
                kernel_search_duration_ns: None,
                index_size: None,
            }),
        };
        let output = format_search_ndjson(
            &output,
            VectorKernelMetadata::new(semantic_code_facade::CliVectorKernelKind::HnswRs),
        )?;
        let lines: Vec<&str> = output.lines().collect();
        let summary: serde_json::Value =
            serde_json::from_str(lines.last().unwrap()).map_err(io::Error::other)?;
        let stats = summary
            .get("searchStats")
            .ok_or_else(|| io::Error::other("missing searchStats"))?;
        assert_eq!(stats.get("expansions").and_then(|v| v.as_u64()), Some(7));
        assert_eq!(
            stats.get("kernel").and_then(|v| v.as_str()),
            Some("hnsw-rs")
        );
        Ok(())
    }

    #[test]
    fn ndjson_summary_includes_extra_metrics_when_present() -> Result<(), CliError> {
        let mut extra = std::collections::BTreeMap::new();
        extra.insert("pulls".into(), 12.0);
        extra.insert("peakBucketUtilization".into(), 0.75);
        let output = SearchOutput {
            results: Vec::new(),
            stats: Some(semantic_code_facade::SearchStats {
                expansions: Some(42),
                kernel: "dfrr".into(),
                extra,
                kernel_search_duration_ns: None,
                index_size: None,
            }),
        };
        let output = format_search_ndjson(
            &output,
            VectorKernelMetadata::new(semantic_code_facade::CliVectorKernelKind::Dfrr),
        )?;
        let lines: Vec<&str> = output.lines().collect();
        let summary: serde_json::Value =
            serde_json::from_str(lines.last().unwrap()).map_err(io::Error::other)?;
        let stats = summary
            .get("searchStats")
            .ok_or_else(|| io::Error::other("missing searchStats"))?;
        let extra = stats
            .get("extra")
            .ok_or_else(|| io::Error::other("missing extra in searchStats"))?;
        assert_eq!(
            extra.get("pulls").and_then(|v| v.as_f64()),
            Some(12.0),
            "extra metric 'pulls' should be in searchStats.extra"
        );
        assert_eq!(
            extra.get("peakBucketUtilization").and_then(|v| v.as_f64()),
            Some(0.75),
            "extra metric 'peakBucketUtilization' should be in searchStats.extra"
        );
        Ok(())
    }
}
