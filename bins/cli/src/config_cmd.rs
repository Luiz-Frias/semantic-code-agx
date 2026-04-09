//! Config subcommand handlers (`config check`, `config show`, `config validate`).

use crate::error::{CliError, ExitCode};
use crate::format::OutputMode;
use crate::output::{CliOutput, format_error_output, format_ndjson_summary, log_info};
use crate::resolve::collect_scoped_env;
use semantic_code_facade::load_effective_config_json;
use std::collections::BTreeMap;
use std::path::Path;

pub fn config_check(
    mode: OutputMode,
    path: Option<&Path>,
    overrides_json: Option<&str>,
) -> Result<CliOutput, CliError> {
    let env = collect_scoped_env("SCA_");
    config_check_with_env(mode, &env, path, overrides_json)
}

pub fn config_check_with_env(
    mode: OutputMode,
    env: &BTreeMap<String, String>,
    path: Option<&Path>,
    overrides_json: Option<&str>,
) -> Result<CliOutput, CliError> {
    let config_json = match load_effective_config_json(env, path, overrides_json) {
        Ok(config) => config,
        Err(error) => return Ok(format_error_output(mode, &error, ExitCode::InvalidInput)),
    };

    let mut stderr = String::new();
    log_info(&mut stderr, "config check completed", mode.no_progress);

    let stdout = if mode.is_ndjson() {
        format_ndjson_summary("ok", "config", None)
    } else if mode.is_json() {
        let config_value: serde_json::Value = serde_json::from_str(config_json.trim())?;
        let payload = serde_json::json!({
            "status": "ok",
            "configPath": path.map(|value| value.to_string_lossy().to_string()),
            "effectiveConfig": config_value,
        });
        let mut output = serde_json::to_string_pretty(&payload)?;
        output.push('\n');
        output
    } else {
        path.map_or_else(
            || "status: ok\nconfig: ok\n".to_string(),
            |path| format!("status: ok\nconfig: ok\npath: {}\n", path.to_string_lossy()),
        )
    };

    Ok(CliOutput {
        stdout,
        stderr,
        exit_code: ExitCode::Ok,
    })
}

pub fn config_show(
    mode: OutputMode,
    path: Option<&Path>,
    overrides_json: Option<&str>,
) -> Result<CliOutput, CliError> {
    let env = collect_scoped_env("SCA_");
    let config_json = match load_effective_config_json(&env, path, overrides_json) {
        Ok(config) => config,
        Err(error) => return Ok(format_error_output(mode, &error, ExitCode::InvalidInput)),
    };

    let mut stderr = String::new();
    log_info(&mut stderr, "config show completed", mode.no_progress);

    let stdout = if mode.is_ndjson() {
        format_ndjson_summary("ok", "config", None)
    } else if mode.is_json() {
        let config_value: serde_json::Value = serde_json::from_str(config_json.trim())?;
        let payload = serde_json::json!({
            "status": "ok",
            "configPath": path.map(|value| value.to_string_lossy().to_string()),
            "effectiveConfig": config_value,
        });
        let mut output = serde_json::to_string_pretty(&payload)?;
        output.push('\n');
        output
    } else {
        let mut out = String::new();
        out.push_str("status: ok\nconfig:\n");
        out.push_str(&config_json);
        out
    };

    Ok(CliOutput {
        stdout,
        stderr,
        exit_code: ExitCode::Ok,
    })
}

pub fn config_validate(
    mode: OutputMode,
    path: Option<&Path>,
    overrides_json: Option<&str>,
) -> Result<CliOutput, CliError> {
    let env = collect_scoped_env("SCA_");
    if let Err(error) = load_effective_config_json(&env, path, overrides_json) {
        return Ok(format_error_output(mode, &error, ExitCode::InvalidInput));
    }

    let mut stderr = String::new();
    log_info(&mut stderr, "config validate completed", mode.no_progress);

    let stdout = if mode.is_ndjson() {
        format_ndjson_summary("ok", "config", None)
    } else if mode.is_json() {
        let payload = serde_json::json!({
            "status": "ok",
            "configPath": path.map(|value| value.to_string_lossy().to_string()),
        });
        let mut output = serde_json::to_string_pretty(&payload)?;
        output.push('\n');
        output
    } else {
        path.map_or_else(
            || "status: ok\nconfig: ok\n".to_string(),
            |path| format!("status: ok\nconfig: ok\npath: {}\n", path.to_string_lossy()),
        )
    };

    Ok(CliOutput {
        stdout,
        stderr,
        exit_code: ExitCode::Ok,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ExitCode;
    use crate::format::{LogLevel, OutputArgs, OutputFormat, OutputMode};
    use std::path::{Path, PathBuf};

    fn workspace_root() -> PathBuf {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        manifest_dir
            .parent()
            .and_then(Path::parent)
            .map_or_else(|| manifest_dir.to_path_buf(), Path::to_path_buf)
    }

    fn fixture_path(relative: &str) -> PathBuf {
        workspace_root()
            .join("crates")
            .join("testkit")
            .join("fixtures")
            .join(relative)
    }

    #[test]
    fn config_check_failure_exit_code_is_invalid_input() -> Result<(), Box<dyn std::error::Error>> {
        let env = BTreeMap::new();
        let missing = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("missing-config.json");

        let mode = OutputMode::from_args(&OutputArgs {
            output: None,
            json: false,
            agent: false,
            no_progress: true,
            interactive: false,
            log_level: LogLevel::Info,
        });
        let output = config_check_with_env(mode, &env, Some(missing.as_path()), None)?;
        assert_eq!(output.exit_code, ExitCode::InvalidInput);
        assert!(output.stdout.contains("status: error"));
        Ok(())
    }

    #[test]
    fn config_overrides_are_applied() -> Result<(), Box<dyn std::error::Error>> {
        let env = BTreeMap::new();
        let path = fixture_path("config/backend-config.valid.json");
        let overrides = r#"{"core":{"timeoutMs":12345}}"#;
        let mode = OutputMode::from_args(&OutputArgs {
            output: Some(OutputFormat::Json),
            json: false,
            agent: false,
            no_progress: true,
            interactive: false,
            log_level: LogLevel::Info,
        });
        let output = config_check_with_env(mode, &env, Some(path.as_path()), Some(overrides))?;
        let value: serde_json::Value = serde_json::from_str(output.stdout.trim())?;
        let timeout_ms = value
            .get("effectiveConfig")
            .and_then(|value| value.get("core"))
            .and_then(|value| value.get("timeoutMs"))
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| std::io::Error::other("missing core.timeoutMs"))?;
        assert_eq!(timeout_ms, 12345);
        Ok(())
    }
}
