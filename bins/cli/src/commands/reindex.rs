//! Reindex command handler.

use crate::commands::jobs::{format_job_status, spawn_job_runner};
use crate::error::{CliError, ExitCode};
use crate::format::OutputMode;
use crate::vector_kernel::{
    VectorKernelMetadata, resolve_vector_kernel_metadata_std_env, warn_if_experimental,
};
use crate::{CliOutput, format_error_output, infra_exit_code};
use semantic_code_facade::{
    JobKind, JobRequest, ReindexByChangeOutput, create_job, run_reindex_local,
    validate_reindex_request_for_root,
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
    let request = match validate_reindex_request_for_root(codebase_root) {
        Ok(request) => request,
        Err(error) => return Ok(format_error_output(mode, &error, infra_exit_code(&error))),
    };
    let vector_kernel = match resolve_vector_kernel_metadata_std_env(config_path, overrides_json) {
        Ok(metadata) => metadata,
        Err(error) => return Ok(format_error_output(mode, &error, infra_exit_code(&error))),
    };
    warn_if_experimental(vector_kernel);

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
        return format_job_status(mode, &status, Some(vector_kernel));
    }

    match run_reindex_local(config_path, overrides_json, &request) {
        Ok(output) => format_reindex_output(mode, output, vector_kernel),
        Err(error) => Ok(format_error_output(mode, &error, infra_exit_code(&error))),
    }
}

fn format_reindex_output(
    mode: OutputMode,
    output: ReindexByChangeOutput,
    vector_kernel: VectorKernelMetadata,
) -> Result<CliOutput, CliError> {
    let stdout = if mode.is_ndjson() {
        let payload = serde_json::json!({
            "type": "summary",
            "status": "ok",
            "kind": "reindex",
            "added": output.added,
            "removed": output.removed,
            "modified": output.modified,
            "vectorKernel": vector_kernel.as_json(),
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
            "vectorKernel": vector_kernel.as_json(),
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
