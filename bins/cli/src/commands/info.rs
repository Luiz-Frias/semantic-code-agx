//! Info command handler.

use crate::CliOutput;
use crate::error::{CliError, ExitCode};
use crate::format::OutputMode;
use semantic_code_core::{BuildInfo, build_info};
use semantic_code_facade::facade_crate_version;

/// Run the info command.
pub fn run_info(mode: OutputMode) -> Result<CliOutput, CliError> {
    let build = build_info();
    let facade_version = facade_crate_version();

    let stdout = if mode.is_ndjson() {
        format_info_ndjson(&build, facade_version)?
    } else if mode.is_json() {
        format_info_json(&build, facade_version)?
    } else {
        format_info_text(&build, facade_version)
    };

    Ok(CliOutput {
        stdout,
        stderr: String::new(),
        exit_code: ExitCode::Ok,
    })
}

fn format_info_text(build: &BuildInfo, facade_version: &str) -> String {
    format!(
        "status: ok\nname: {}\nversion: {}\nfacade: {}\nrustc: {}\ntarget: {}\nprofile: {}\ngit: {}{}\n",
        build.name,
        build.version,
        facade_version,
        build.rustc_version,
        build.target,
        build.profile,
        build.git_hash.unwrap_or("none"),
        if build.git_dirty { " (dirty)" } else { "" }
    )
}

fn format_info_json(build: &BuildInfo, facade_version: &str) -> Result<String, CliError> {
    let payload = serde_json::json!({
        "status": "ok",
        "build": {
            "name": build.name,
            "version": build.version,
            "facadeVersion": facade_version,
            "rustcVersion": build.rustc_version,
            "target": build.target,
            "profile": build.profile,
            "gitHash": build.git_hash,
            "gitDirty": build.git_dirty,
        }
    });
    let mut output = serde_json::to_string_pretty(&payload)?;
    output.push('\n');
    Ok(output)
}

fn format_info_ndjson(build: &BuildInfo, facade_version: &str) -> Result<String, CliError> {
    let payload = serde_json::json!({
        "type": "summary",
        "status": "ok",
        "kind": "info",
        "build": {
            "name": build.name,
            "version": build.version,
            "facadeVersion": facade_version,
            "rustcVersion": build.rustc_version,
            "target": build.target,
            "profile": build.profile,
            "gitHash": build.git_hash,
            "gitDirty": build.git_dirty,
        }
    });
    let mut output = serde_json::to_string(&payload)?;
    output.push('\n');
    Ok(output)
}
