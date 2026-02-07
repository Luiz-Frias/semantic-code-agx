//! CLI integration tests.

use std::path::Path;
use std::process::Command;

fn run_cli(args: &[&str]) -> std::io::Result<std::process::Output> {
    Command::new(env!("CARGO_BIN_EXE_sca")).args(args).output()
}

fn run_cli_clean_env(args: &[&str]) -> std::io::Result<std::process::Output> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_sca"));
    command.args(args);
    scrub_scoped_env(&mut command);
    command.output()
}

fn scrub_scoped_env(command: &mut Command) {
    for (key, _) in std::env::vars() {
        if key.starts_with("SCA_") {
            command.env_remove(key);
        }
    }
    for key in embedding_env_aliases() {
        command.env_remove(key);
    }
    for key in provider_env_keys() {
        command.env_remove(key);
    }
}

fn embedding_env_aliases() -> &'static [&'static str] {
    &[
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "EMBEDDING_TIMEOUT_MS",
        "EMBEDDING_BATCH_SIZE",
        "EMBEDDING_DIMENSION",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_LOCAL_FIRST",
        "EMBEDDING_LOCAL_ONLY",
        "EMBEDDING_ROUTING_MODE",
        "EMBEDDING_SPLIT_MAX_REMOTE_BATCHES",
        "EMBEDDING_JOBS_PROGRESS_INTERVAL_MS",
        "EMBEDDING_JOBS_CANCEL_POLL_INTERVAL_MS",
        "EMBEDDING_TEST_FALLBACK",
        "EMBEDDING_ONNX_MODEL_DIR",
        "EMBEDDING_ONNX_MODEL_FILENAME",
        "EMBEDDING_ONNX_TOKENIZER_FILENAME",
        "EMBEDDING_ONNX_REPO",
        "EMBEDDING_ONNX_DOWNLOAD",
        "EMBEDDING_ONNX_SESSION_POOL_SIZE",
        "EMBEDDING_CACHE_ENABLED",
        "EMBEDDING_CACHE_MAX_ENTRIES",
        "EMBEDDING_CACHE_MAX_BYTES",
        "EMBEDDING_CACHE_DISK_ENABLED",
        "EMBEDDING_CACHE_DISK_PATH",
        "EMBEDDING_CACHE_DISK_PROVIDER",
        "EMBEDDING_CACHE_DISK_CONNECTION",
        "EMBEDDING_CACHE_DISK_TABLE",
        "EMBEDDING_CACHE_DISK_MAX_BYTES",
        "EMBEDDING_API_KEY",
    ]
}

fn provider_env_keys() -> &'static [&'static str] {
    &[
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "GEMINI_API_KEY",
        "GEMINI_BASE_URL",
        "GEMINI_MODEL",
        "VOYAGE_API_KEY",
        "VOYAGE_BASE_URL",
        "VOYAGE_MODEL",
        "OLLAMA_HOST",
        "OLLAMA_MODEL",
    ]
}

fn workspace_root() -> std::path::PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_dir.to_path_buf())
}

fn fixture_path(relative: &str) -> std::path::PathBuf {
    workspace_root()
        .join("crates")
        .join("testkit")
        .join("fixtures")
        .join(relative)
}

#[test]
fn cli_self_check_runs() -> std::io::Result<()> {
    let output = Command::new(env!("CARGO_BIN_EXE_sca"))
        .args(["self-check"])
        .env("SCA_EMBEDDING_PROVIDER", "test")
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "self-check failed: {stderr}");
    assert!(stdout.contains("status: ok"));

    Ok(())
}

#[test]
fn cli_version_runs() -> std::io::Result<()> {
    let output = run_cli(&["--version"])?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "--version failed: {stderr}");
    assert!(stdout.starts_with("sca "));
    assert!(stdout.contains(env!("CARGO_PKG_VERSION")));

    Ok(())
}

#[test]
fn cli_validate_request_runs() -> std::io::Result<()> {
    let payload = r#"{"codebaseRoot":"/tmp/repo","query":"hello","topK":5}"#;
    let output = run_cli(&[
        "validate-request",
        "--kind",
        "search",
        "--input-json",
        payload,
    ])?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "validate-request failed: {stderr}");
    assert!(stdout.contains("status: ok"));
    assert!(stdout.contains("kind: search"));

    Ok(())
}

#[test]
fn cli_config_check_runs_on_valid_fixture() -> std::io::Result<()> {
    let path = fixture_path("config/backend-config.valid.json");
    let output = run_cli(&["config", "check", "--path", path.to_string_lossy().as_ref()])?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "config check failed: {stderr}");
    assert!(stdout.contains("status: ok"));

    Ok(())
}

#[test]
fn cli_config_check_runs_on_toml_fixture() -> std::io::Result<()> {
    let path = fixture_path("config/backend-config.default.toml");
    let output = run_cli(&["config", "check", "--path", path.to_string_lossy().as_ref()])?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "config check failed: {stderr}");
    assert!(stdout.contains("status: ok"));

    Ok(())
}

#[test]
fn cli_config_check_fails_on_invalid_fixture() -> std::io::Result<()> {
    let path = fixture_path("config/backend-config.invalid.json");
    let output = run_cli(&["config", "check", "--path", path.to_string_lossy().as_ref()])?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert_eq!(output.status.code(), Some(2));
    assert!(stdout.contains("status: error"));

    Ok(())
}

#[test]
fn cli_config_check_env_overrides_win() -> std::io::Result<()> {
    let path = fixture_path("config/backend-config.valid.json");
    let output = Command::new(env!("CARGO_BIN_EXE_sca"))
        .args(["--json", "config", "check", "--path"])
        .arg(path)
        .env("SCA_CORE_TIMEOUT_MS", "60000")
        .output()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());

    let value: serde_json::Value =
        serde_json::from_str(stdout.trim()).map_err(std::io::Error::other)?;
    let timeout_ms = value
        .get("effectiveConfig")
        .and_then(|value| value.get("core"))
        .and_then(|value| value.get("timeoutMs"))
        .and_then(|value| value.as_u64())
        .ok_or_else(|| std::io::Error::other("missing core.timeoutMs"))?;
    assert_eq!(timeout_ms, 60000);

    Ok(())
}

#[test]
fn cli_config_show_runs_on_valid_fixture() -> std::io::Result<()> {
    let path = fixture_path("config/backend-config.valid.json");
    let output = run_cli(&[
        "--json",
        "config",
        "show",
        "--path",
        path.to_string_lossy().as_ref(),
    ])?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "config show failed: {stderr}");
    let value: serde_json::Value =
        serde_json::from_str(stdout.trim()).map_err(std::io::Error::other)?;
    assert_eq!(value.get("status").and_then(|v| v.as_str()), Some("ok"));
    assert!(value.get("effectiveConfig").is_some());

    Ok(())
}

#[test]
fn cli_config_show_reflects_external_profile() -> std::io::Result<()> {
    let path = fixture_path("config/backend-config.milvus-external.toml");
    let output = run_cli_clean_env(&[
        "--json",
        "config",
        "show",
        "--path",
        path.to_string_lossy().as_ref(),
    ])?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "config show failed: {stderr}");
    let value: serde_json::Value =
        serde_json::from_str(stdout.trim()).map_err(std::io::Error::other)?;
    let effective = value
        .get("effectiveConfig")
        .ok_or_else(|| std::io::Error::other("missing effectiveConfig"))?;
    assert_eq!(
        effective
            .get("embedding")
            .and_then(|value| value.get("provider"))
            .and_then(|value| value.as_str()),
        Some("auto")
    );
    assert_eq!(
        effective
            .get("vectorDb")
            .and_then(|value| value.get("provider"))
            .and_then(|value| value.as_str()),
        Some("milvus")
    );
    assert_eq!(
        effective
            .get("vectorDb")
            .and_then(|value| value.get("address"))
            .and_then(|value| value.as_str()),
        Some("127.0.0.1:19530")
    );

    Ok(())
}

#[test]
fn cli_config_validate_runs_on_valid_fixture() -> std::io::Result<()> {
    let path = fixture_path("config/backend-config.valid.json");
    let output = run_cli(&[
        "config",
        "validate",
        "--path",
        path.to_string_lossy().as_ref(),
    ])?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "config validate failed: {stderr}");
    assert!(stdout.contains("status: ok"));

    Ok(())
}
