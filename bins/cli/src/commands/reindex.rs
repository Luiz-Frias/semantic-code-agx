//! Reindex command handler.

use crate::commands::jobs::{format_job_status, spawn_job_runner};
use crate::error::{CliError, ExitCode};
use crate::format::OutputMode;
use crate::{CliOutput, format_error_output, infra_exit_code};
use semantic_code_config::{ReindexByChangeRequestDto, validate_reindex_by_change_request};
use semantic_code_facade::{
    JobKind, JobRequest, ReindexByChangeOutput, create_job, run_reindex_local,
};
use std::path::Path;

/// Run the reindex command.
pub fn run_reindex(
    mode: OutputMode,
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    codebase_root: &Path,
    background: bool,
) -> Result<CliOutput, CliError> {
    let request = ReindexByChangeRequestDto {
        codebase_root: codebase_root.to_string_lossy().to_string(),
    };
    let request = match validate_reindex_by_change_request(&request) {
        Ok(request) => request,
        Err(error) => return Ok(format_error_output(mode, &error, infra_exit_code(&error))),
    };

    if background {
        let job_request = match JobRequest::new(
            JobKind::Reindex,
            codebase_root,
            config_path,
            overrides_json.map(str::to_string),
            false,
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

    match run_reindex_local(config_path, overrides_json, &request) {
        Ok(output) => format_reindex_output(mode, output),
        Err(error) => Ok(format_error_output(mode, &error, infra_exit_code(&error))),
    }
}

fn format_reindex_output(
    mode: OutputMode,
    output: ReindexByChangeOutput,
) -> Result<CliOutput, CliError> {
    let stdout = if mode.is_ndjson() {
        let payload = serde_json::json!({
            "type": "summary",
            "status": "ok",
            "kind": "reindex",
            "added": output.added,
            "removed": output.removed,
            "modified": output.modified,
        });
        let mut out = serde_json::to_string(&payload)?;
        out.push('\n');
        out
    } else if mode.is_json() {
        let payload = serde_json::json!({
            "status": "ok",
            "added": output.added,
            "removed": output.removed,
            "modified": output.modified,
        });
        let mut out = serde_json::to_string_pretty(&payload)?;
        out.push('\n');
        out
    } else {
        format!(
            "status: ok\nadded: {}\nremoved: {}\nmodified: {}\n",
            output.added, output.removed, output.modified
        )
    };

    Ok(CliOutput {
        stdout,
        stderr: String::new(),
        exit_code: ExitCode::Ok,
    })
}
