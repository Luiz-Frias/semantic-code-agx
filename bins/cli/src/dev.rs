//! Dev/diagnostic commands: `self-check` and `validate-request`.
//!
//! `self-check` is behind `#[cfg(any(debug_assertions, feature = "dev-tools"))]`
//! and exercises the facade smoke tests. `validate-request` is always compiled
//! (hidden from `--help`) for CI pre-flight validation.

use crate::args::ValidateRequestKind;
use crate::error::{CliError, ExitCode};
use crate::format::OutputMode;
use crate::output::{CliOutput, format_error_output, format_ndjson_summary, log_info};
#[cfg(any(debug_assertions, feature = "dev-tools"))]
use crate::resolve::collect_scoped_env;
#[cfg(any(debug_assertions, feature = "dev-tools"))]
use crate::vector_kernel::resolve_vector_kernel_metadata_from_env;
#[cfg(any(debug_assertions, feature = "dev-tools"))]
use semantic_code_facade::build_info;
use semantic_code_facade::validate_request_json;
#[cfg(any(debug_assertions, feature = "dev-tools"))]
use semantic_code_facade::{
    BuildInfo, facade_crate_version, run_clear_smoke, run_index_smoke, run_search_smoke,
};
#[cfg(any(debug_assertions, feature = "dev-tools"))]
use std::collections::BTreeMap;

#[cfg(any(debug_assertions, feature = "dev-tools"))]
pub fn self_check(mode: OutputMode) -> Result<CliOutput, CliError> {
    let env = collect_scoped_env("SCA_");
    self_check_with_env(mode, &env)
}

#[cfg(any(debug_assertions, feature = "dev-tools"))]
pub fn self_check_with_env(
    mode: OutputMode,
    env: &BTreeMap<String, String>,
) -> Result<CliOutput, CliError> {
    let vector_kernel = match resolve_vector_kernel_metadata_from_env(env) {
        Ok(kernel) => kernel,
        Err(error) => return Ok(format_error_output(mode, &error, ExitCode::InvalidInput)),
    };

    if let Err(error) = run_index_smoke() {
        return Ok(format_error_output(mode, &error, ExitCode::Internal));
    }
    if let Err(error) = run_search_smoke() {
        return Ok(format_error_output(mode, &error, ExitCode::Internal));
    }
    if let Err(error) = run_clear_smoke() {
        return Ok(format_error_output(mode, &error, ExitCode::Internal));
    }

    let build = build_info();
    let facade_version = facade_crate_version();
    let mut stderr = String::new();

    log_info(&mut stderr, "self-check completed", mode.no_progress);

    let stdout = if mode.is_json() {
        format_self_check_json(&build, facade_version, true, true, true, vector_kernel)?
    } else {
        format_self_check_text(&build, facade_version, true, true, true, vector_kernel)
    };

    Ok(CliOutput {
        stdout,
        stderr,
        exit_code: ExitCode::Ok,
    })
}

pub fn validate_request(
    kind: ValidateRequestKind,
    input_json: &str,
    mode: OutputMode,
) -> Result<CliOutput, CliError> {
    if let Err(error) = validate_request_json(kind.into(), input_json) {
        return Ok(format_error_output(mode, &error, ExitCode::InvalidInput));
    }

    let mut stderr = String::new();
    log_info(
        &mut stderr,
        "request validation completed",
        mode.no_progress,
    );

    let stdout = if mode.is_ndjson() {
        format_ndjson_summary("ok", kind.as_str(), None)
    } else if mode.is_json() {
        let payload = serde_json::json!({
            "status": "ok",
            "kind": kind.as_str(),
        });
        let mut output = serde_json::to_string_pretty(&payload)?;
        output.push('\n');
        output
    } else {
        format!("status: ok\nkind: {}\n", kind.as_str())
    };

    Ok(CliOutput {
        stdout,
        stderr,
        exit_code: ExitCode::Ok,
    })
}

#[cfg(any(debug_assertions, feature = "dev-tools"))]
fn format_self_check_text(
    build: &BuildInfo,
    facade_version: &str,
    index_ok: bool,
    search_ok: bool,
    clear_ok: bool,
    vector_kernel: crate::vector_kernel::VectorKernelMetadata,
) -> String {
    format!(
        "status: ok\nenv: ok\nindex: {}\nsearch: {}\nclear: {}\nvectorKernel: {}\nname: {}\nversion: {}\nfacade: {}\nrustc: {}\ntarget: {}\nprofile: {}\ngit: {}{}\n",
        if index_ok { "ok" } else { "error" },
        if search_ok { "ok" } else { "error" },
        if clear_ok { "ok" } else { "error" },
        vector_kernel.effective_label(),
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

#[cfg(any(debug_assertions, feature = "dev-tools"))]
fn format_self_check_json(
    build: &BuildInfo,
    facade_version: &str,
    index_ok: bool,
    search_ok: bool,
    clear_ok: bool,
    vector_kernel: crate::vector_kernel::VectorKernelMetadata,
) -> Result<String, CliError> {
    let payload = serde_json::json!({
        "status": "ok",
        "env": { "status": "ok" },
        "index": { "status": if index_ok { "ok" } else { "error" } },
        "search": { "status": if search_ok { "ok" } else { "error" } },
        "clear": { "status": if clear_ok { "ok" } else { "error" } },
        "vectorKernel": vector_kernel.as_json(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{LogLevel, OutputArgs, OutputFormat, OutputMode};

    #[test]
    #[cfg(any(debug_assertions, feature = "dev-tools"))]
    fn self_check_json_output_shape() -> Result<(), Box<dyn std::error::Error>> {
        let mode = OutputMode::from_args(&OutputArgs {
            output: Some(OutputFormat::Json),
            json: false,
            agent: false,
            no_progress: true,
            interactive: false,
            log_level: LogLevel::Info,
        });
        let output = self_check_with_env(mode, &BTreeMap::new())?;
        let value: serde_json::Value = serde_json::from_str(output.stdout.trim())?;

        let status = value
            .get("status")
            .and_then(|value| value.as_str())
            .ok_or_else(|| std::io::Error::other("missing status field"))?;
        assert_eq!(status, "ok");

        let env_status = value
            .get("env")
            .and_then(|value| value.get("status"))
            .and_then(|value| value.as_str())
            .ok_or_else(|| std::io::Error::other("missing env status"))?;
        assert_eq!(env_status, "ok");
        let vector_kernel = value
            .get("vectorKernel")
            .and_then(|value| value.get("effective"))
            .and_then(|value| value.as_str())
            .ok_or_else(|| std::io::Error::other("missing vectorKernel.effective"))?;
        assert_eq!(vector_kernel, "hnsw-rs");

        let build = value
            .get("build")
            .ok_or_else(|| std::io::Error::other("missing build field"))?;
        let build_obj = build
            .as_object()
            .ok_or_else(|| std::io::Error::other("build object missing"))?;
        let keys = [
            "name",
            "version",
            "facadeVersion",
            "rustcVersion",
            "target",
            "profile",
            "gitHash",
            "gitDirty",
        ];

        for key in keys {
            assert!(build_obj.contains_key(key), "missing {key}");
        }

        Ok(())
    }

    #[test]
    fn validate_request_errors_are_invalid_input() -> Result<(), Box<dyn std::error::Error>> {
        let mode = OutputMode::from_args(&OutputArgs {
            output: None,
            json: false,
            agent: false,
            no_progress: true,
            interactive: false,
            log_level: LogLevel::Info,
        });
        let output = validate_request(ValidateRequestKind::Search, "{bad", mode)?;
        assert_eq!(output.exit_code, ExitCode::InvalidInput);
        assert!(output.stdout.contains("status: error"));
        Ok(())
    }
}
