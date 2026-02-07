//! Clear command handler.

use crate::error::{CliError, ExitCode};
use crate::format::OutputMode;
use crate::{CliOutput, format_error_output, infra_exit_code};
use semantic_code_config::{ClearIndexRequestDto, validate_clear_index_request};
use semantic_code_facade::run_clear_local;
use std::path::Path;

/// Run the clear command.
pub fn run_clear(
    mode: OutputMode,
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    codebase_root: &Path,
) -> Result<CliOutput, CliError> {
    let request = ClearIndexRequestDto {
        codebase_root: codebase_root.to_string_lossy().to_string(),
    };
    let request = match validate_clear_index_request(&request) {
        Ok(request) => request,
        Err(error) => return Ok(format_error_output(mode, &error, infra_exit_code(&error))),
    };

    match run_clear_local(config_path, overrides_json, &request) {
        Ok(()) => format_clear_output(mode),
        Err(error) => Ok(format_error_output(mode, &error, infra_exit_code(&error))),
    }
}

fn format_clear_output(mode: OutputMode) -> Result<CliOutput, CliError> {
    let stdout = if mode.is_ndjson() {
        let payload = serde_json::json!({
            "type": "summary",
            "status": "ok",
            "kind": "clear",
        });
        let mut out = serde_json::to_string(&payload)?;
        out.push('\n');
        out
    } else if mode.is_json() {
        let payload = serde_json::json!({ "status": "ok" });
        let mut out = serde_json::to_string_pretty(&payload)?;
        out.push('\n');
        out
    } else {
        "status: ok\n".to_string()
    };

    Ok(CliOutput {
        stdout,
        stderr: String::new(),
        exit_code: ExitCode::Ok,
    })
}
