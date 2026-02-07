//! Init command handler.

use crate::error::{CliError, ExitCode};
use crate::format::OutputMode;
use crate::{CliOutput, format_error_output, infra_exit_code};
use semantic_code_facade::{CliInitStatus, run_init_local};
use std::path::Path;

/// Run the init command.
pub fn run_init(
    mode: OutputMode,
    config_path: Option<&Path>,
    codebase_root: &Path,
    storage_mode: Option<semantic_code_config::SnapshotStorageMode>,
    force: bool,
) -> Result<CliOutput, CliError> {
    let status = match run_init_local(config_path, codebase_root, storage_mode, force) {
        Ok(status) => status,
        Err(error) => return Ok(format_error_output(mode, &error, infra_exit_code(&error))),
    };

    format_init_output(mode, &status)
}

fn format_init_output(mode: OutputMode, status: &CliInitStatus) -> Result<CliOutput, CliError> {
    let stdout = if mode.is_ndjson() {
        let payload = serde_json::json!({
            "type": "summary",
            "status": "ok",
            "kind": "init",
            "configPath": status.config_path,
            "manifestPath": status.manifest_path,
            "createdConfig": status.created_config,
            "createdManifest": status.created_manifest,
        });
        let mut out = serde_json::to_string(&payload)?;
        out.push('\n');
        out
    } else if mode.is_json() {
        let payload = serde_json::json!({
            "status": "ok",
            "configPath": status.config_path,
            "manifestPath": status.manifest_path,
            "created": {
                "config": status.created_config,
                "manifest": status.created_manifest,
            }
        });
        let mut out = serde_json::to_string_pretty(&payload)?;
        out.push('\n');
        out
    } else {
        format!(
            "status: ok\nconfigPath: {}\nmanifestPath: {}\ncreatedConfig: {}\ncreatedManifest: {}\n",
            status.config_path.to_string_lossy(),
            status.manifest_path.to_string_lossy(),
            status.created_config,
            status.created_manifest
        )
    };

    Ok(CliOutput {
        stdout,
        stderr: String::new(),
        exit_code: ExitCode::Ok,
    })
}
