//! Index command handler.

use crate::commands::jobs::{format_job_status, spawn_job_runner};
use crate::error::{CliError, ExitCode};
use crate::format::OutputMode;
use crate::{CliOutput, format_error_output, infra_exit_code};
use semantic_code_config::{IndexRequestDto, validate_index_request};
use semantic_code_facade::{
    IndexCodebaseOutput, IndexCodebaseStatus, JobKind, JobRequest, create_job, run_index_local,
};
use std::path::Path;

/// Run the index command.
pub fn run_index(
    mode: OutputMode,
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    codebase_root: &Path,
    init_if_missing: bool,
    background: bool,
) -> Result<CliOutput, CliError> {
    let request = IndexRequestDto {
        codebase_root: codebase_root.to_string_lossy().to_string(),
        collection_name: None,
        force_reindex: None,
    };
    let request = match validate_index_request(&request) {
        Ok(request) => request,
        Err(error) => return Ok(format_error_output(mode, &error, infra_exit_code(&error))),
    };

    if background {
        let job_request = match JobRequest::new(
            JobKind::Index,
            codebase_root,
            config_path,
            overrides_json.map(str::to_string),
            init_if_missing,
        ) {
            Ok(request) => request,
            Err(error) => return Ok(format_error_output(mode, &error, infra_exit_code(&error))),
        };
        let status = match create_job(&job_request) {
            Ok(status) => status,
            Err(error) => return Ok(format_error_output(mode, &error, infra_exit_code(&error))),
        };
        spawn_job_runner(&status.id, codebase_root)?;
        return format_job_status(mode, &status);
    }

    match run_index_local(config_path, overrides_json, &request, init_if_missing) {
        Ok(output) => format_index_output(mode, &output),
        Err(error) => Ok(format_error_output(mode, &error, infra_exit_code(&error))),
    }
}

fn format_index_output(
    mode: OutputMode,
    output: &IndexCodebaseOutput,
) -> Result<CliOutput, CliError> {
    let stdout = if mode.is_ndjson() {
        let payload = serde_json::json!({
            "type": "summary",
            "status": "ok",
            "indexedFiles": output.indexed_files,
            "totalChunks": output.total_chunks,
            "indexStatus": index_status_label(output.status),
            "stageStats": stage_stats_json(output),
        });
        let mut out = serde_json::to_string(&payload)?;
        out.push('\n');
        out
    } else if mode.is_json() {
        format_index_json(output)?
    } else {
        format_index_text(output)
    };

    Ok(CliOutput {
        stdout,
        stderr: String::new(),
        exit_code: ExitCode::Ok,
    })
}

fn format_index_json(output: &IndexCodebaseOutput) -> Result<String, CliError> {
    let payload = serde_json::json!({
        "status": "ok",
        "indexedFiles": output.indexed_files,
        "totalChunks": output.total_chunks,
        "indexStatus": index_status_label(output.status),
        "stageStats": stage_stats_json(output),
    });
    let mut out = serde_json::to_string_pretty(&payload)?;
    out.push('\n');
    Ok(out)
}

fn format_index_text(output: &IndexCodebaseOutput) -> String {
    let mut out = String::new();
    out.push_str("status: ok\n");
    out.push_str("indexedFiles: ");
    out.push_str(&output.indexed_files.to_string());
    out.push('\n');
    out.push_str("totalChunks: ");
    out.push_str(&output.total_chunks.to_string());
    out.push('\n');
    out.push_str("indexStatus: ");
    out.push_str(index_status_label(output.status));
    out.push('\n');
    out.push_str("stageStats:\n");
    out.push_str("  scan: files=");
    out.push_str(&output.stage_stats.scan.files.to_string());
    out.push_str(" durationMs=");
    out.push_str(&output.stage_stats.scan.duration_ms.to_string());
    out.push('\n');
    out.push_str("  split: files=");
    out.push_str(&output.stage_stats.split.files.to_string());
    out.push_str(" chunks=");
    out.push_str(&output.stage_stats.split.chunks.to_string());
    out.push_str(" durationMs=");
    out.push_str(&output.stage_stats.split.duration_ms.to_string());
    out.push('\n');
    out.push_str("  embed: batches=");
    out.push_str(&output.stage_stats.embed.batches.to_string());
    out.push_str(" chunks=");
    out.push_str(&output.stage_stats.embed.chunks.to_string());
    out.push_str(" durationMs=");
    out.push_str(&output.stage_stats.embed.duration_ms.to_string());
    out.push('\n');
    out.push_str("  insert: batches=");
    out.push_str(&output.stage_stats.insert.batches.to_string());
    out.push_str(" chunks=");
    out.push_str(&output.stage_stats.insert.chunks.to_string());
    out.push_str(" durationMs=");
    out.push_str(&output.stage_stats.insert.duration_ms.to_string());
    out.push('\n');
    out
}

const fn index_status_label(status: IndexCodebaseStatus) -> &'static str {
    match status {
        IndexCodebaseStatus::Completed => "completed",
        IndexCodebaseStatus::LimitReached => "limitReached",
    }
}

fn stage_stats_json(output: &IndexCodebaseOutput) -> serde_json::Value {
    serde_json::json!({
        "scan": {
            "files": output.stage_stats.scan.files,
            "durationMs": output.stage_stats.scan.duration_ms,
        },
        "split": {
            "files": output.stage_stats.split.files,
            "chunks": output.stage_stats.split.chunks,
            "durationMs": output.stage_stats.split.duration_ms,
        },
        "embed": {
            "batches": output.stage_stats.embed.batches,
            "chunks": output.stage_stats.embed.chunks,
            "durationMs": output.stage_stats.embed.duration_ms,
        },
        "insert": {
            "batches": output.stage_stats.insert.batches,
            "chunks": output.stage_stats.insert.chunks,
            "durationMs": output.stage_stats.insert.duration_ms,
        },
    })
}
