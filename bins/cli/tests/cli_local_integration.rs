//! Integration tests for local CLI commands.

use std::io;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn run_cli_in_dir(dir: &Path, args: &[&str]) -> io::Result<std::process::Output> {
    Command::new(env!("CARGO_BIN_EXE_sca"))
        .current_dir(dir)
        .env("SCA_EMBEDDING_PROVIDER", "test")
        .args(args)
        .output()
}

fn run_cli_in_dir_with_env(
    dir: &Path,
    args: &[&str],
    extra_env: &[(&str, &str)],
) -> io::Result<std::process::Output> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_sca"));
    command
        .current_dir(dir)
        .env("SCA_EMBEDDING_PROVIDER", "test");
    for (key, value) in extra_env {
        command.env(key, value);
    }
    command.args(args).output()
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

fn golden_fixture_path(relative: &str) -> std::path::PathBuf {
    workspace_root()
        .join("crates")
        .join("testkit")
        .join("fixtures")
        .join("cli-output")
        .join(relative)
}

fn copy_fixture_repo(relative: &str) -> io::Result<std::path::PathBuf> {
    let source = fixture_path(relative);
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dest = std::env::temp_dir().join(format!("sca-cli-fixture-{unique}"));
    copy_dir_recursive(&source, &dest)?;
    Ok(dest)
}

fn copy_dir_recursive(source: &Path, dest: &Path) -> io::Result<()> {
    std::fs::create_dir_all(dest)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let source_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_recursive(&source_path, &dest_path)?;
        } else if file_type.is_file() {
            std::fs::copy(&source_path, &dest_path)?;
        }
    }
    Ok(())
}

#[test]
fn cli_index_init_creates_manifest_and_gitignore() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    std::fs::write(temp.join(".gitignore"), "target/\n")?;

    let output = run_cli_in_dir(&temp, &["index", "--init"])?;
    assert!(output.status.success(), "index --init failed");
    assert!(temp.join(".context/manifest.json").is_file());
    assert!(temp.join(".context/config.toml").is_file());
    let gitignore = std::fs::read_to_string(temp.join(".gitignore")).unwrap_or_default();
    assert!(gitignore.lines().any(|line| line.trim() == ".context/"));
    Ok(())
}

#[test]
fn cli_init_creates_manifest_and_config() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let output = run_cli_in_dir(&temp, &["init"])?;
    assert!(output.status.success(), "init failed");
    assert!(temp.join(".context/manifest.json").is_file());
    assert!(temp.join(".context/config.toml").is_file());
    Ok(())
}

#[test]
fn cli_index_auto_loads_context_config() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let context_dir = temp.join(".context");
    std::fs::create_dir_all(&context_dir)?;
    std::fs::write(
        context_dir.join("config.toml"),
        "version = 1\n\n[vectorDb]\nindexMode = \"hybrid\"\n",
    )?;

    let output = run_cli_in_dir(&temp, &["index", "--init"])?;
    assert!(output.status.success(), "index --init failed");

    let manifest_path = temp.join(".context/manifest.json");
    let manifest = std::fs::read_to_string(manifest_path)?;
    let manifest_json: serde_json::Value =
        serde_json::from_str(&manifest).map_err(io::Error::other)?;
    assert_eq!(
        manifest_json
            .get("indexMode")
            .and_then(|value| value.as_str()),
        Some("hybrid")
    );
    Ok(())
}

#[test]
fn cli_search_errors_without_manifest() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let output = run_cli_in_dir(&temp, &["search", "--query", "needle"])?;
    assert_eq!(output.status.code(), Some(2));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("manifest"),
        "expected manifest error in stdout"
    );
    Ok(())
}

#[test]
fn cli_status_errors_without_manifest() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let output = run_cli_in_dir(&temp, &["status"])?;
    assert_eq!(output.status.code(), Some(2));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("manifest"),
        "expected manifest error in stdout"
    );
    Ok(())
}

#[test]
fn cli_search_json_output_schema() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    run_cli_in_dir(&temp, &["index", "--init"])?;
    let output = run_cli_in_dir(&temp, &["--json", "search", "--query", "local-index"])?;
    let value: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(value.get("status").and_then(|v| v.as_str()), Some("ok"));
    assert!(value.get("results").and_then(|v| v.as_array()).is_some());
    Ok(())
}

#[test]
fn cli_search_ndjson_output_schema() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    run_cli_in_dir(&temp, &["index", "--init"])?;
    let output = run_cli_in_dir(
        &temp,
        &["--output", "ndjson", "search", "--query", "local-index"],
    )?;
    assert!(output.status.success(), "search failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = stdout.lines();
    let last = lines
        .next_back()
        .ok_or_else(|| io::Error::other("missing summary"))?;
    let summary: serde_json::Value = serde_json::from_str(last).map_err(io::Error::other)?;
    assert_eq!(
        summary.get("type").and_then(|v| v.as_str()),
        Some("summary")
    );
    assert_eq!(summary.get("status").and_then(|v| v.as_str()), Some("ok"));
    Ok(())
}

#[test]
fn cli_search_reads_query_from_stdin() -> io::Result<()> {
    use std::io::Write;
    use std::process::Stdio;

    let temp = copy_fixture_repo("local-index/basic")?;
    run_cli_in_dir(&temp, &["index", "--init"])?;

    let mut child = Command::new(env!("CARGO_BIN_EXE_sca"))
        .current_dir(&temp)
        .env("SCA_EMBEDDING_PROVIDER", "test")
        .args(["search", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;

    if let Some(stdin) = child.stdin.as_mut() {
        stdin.write_all(b"local-index\n")?;
    }

    let output = child.wait_with_output()?;
    assert!(output.status.success(), "search --stdin failed");
    Ok(())
}

#[test]
fn cli_search_output_matches_golden_fixtures() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let output = run_cli_in_dir(&temp, &["index", "--init"])?;
    assert!(output.status.success(), "index --init failed");

    let expected_json = std::fs::read_to_string(golden_fixture_path("search.basic.json"))?;
    let expected_json: serde_json::Value =
        serde_json::from_str(&expected_json).map_err(io::Error::other)?;
    let output = run_cli_in_dir(
        &temp,
        &["--output", "json", "search", "--query", "local-index"],
    )?;
    assert!(output.status.success(), "search --output json failed");
    let actual_json: serde_json::Value =
        serde_json::from_slice(&output.stdout).map_err(io::Error::other)?;
    assert_eq!(actual_json, expected_json);

    let expected_ndjson = std::fs::read_to_string(golden_fixture_path("search.basic.ndjson"))?;
    let expected_lines: Vec<serde_json::Value> = expected_ndjson
        .lines()
        .map(|line| serde_json::from_str(line).map_err(io::Error::other))
        .collect::<Result<_, _>>()?;
    let output = run_cli_in_dir(
        &temp,
        &["--output", "ndjson", "search", "--query", "local-index"],
    )?;
    assert!(output.status.success(), "search --output ndjson failed");
    let actual_lines: Vec<serde_json::Value> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| serde_json::from_str(line).map_err(io::Error::other))
        .collect::<Result<_, _>>()?;
    assert_eq!(actual_lines, expected_lines);
    Ok(())
}

#[test]
fn cli_agent_scripted_flow_emits_ndjson() -> io::Result<()> {
    use std::io::Write;
    use std::process::Stdio;

    let temp = copy_fixture_repo("local-index/basic")?;
    let output = run_cli_in_dir(&temp, &["--agent", "index", "--init"])?;
    assert!(output.status.success(), "agent index failed");

    let summary: serde_json::Value =
        serde_json::from_slice(&output.stdout).map_err(io::Error::other)?;
    assert_eq!(
        summary.get("type").and_then(|value| value.as_str()),
        Some("summary")
    );
    assert_eq!(
        summary.get("status").and_then(|value| value.as_str()),
        Some("ok")
    );

    let mut child = Command::new(env!("CARGO_BIN_EXE_sca"))
        .current_dir(&temp)
        .env("SCA_EMBEDDING_PROVIDER", "test")
        .args(["--agent", "search", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;

    if let Some(stdin) = child.stdin.as_mut() {
        stdin.write_all(b"local-index\n")?;
    }

    let output = child.wait_with_output()?;
    assert!(output.status.success(), "agent search failed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut saw_result = false;
    let mut summary_line: Option<serde_json::Value> = None;
    for line in stdout.lines() {
        let value: serde_json::Value = serde_json::from_str(line).map_err(io::Error::other)?;
        match value.get("type").and_then(|value| value.as_str()) {
            Some("result") => saw_result = true,
            Some("summary") => summary_line = Some(value),
            _ => {},
        }
    }

    let summary = summary_line.ok_or_else(|| io::Error::other("missing summary"))?;
    assert_eq!(
        summary.get("status").and_then(|value| value.as_str()),
        Some("ok")
    );
    assert!(saw_result, "expected at least one result line");
    Ok(())
}

#[test]
fn cli_clear_removes_vector_snapshot() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let output = run_cli_in_dir(&temp, &["index", "--init"])?;
    assert!(output.status.success(), "index --init failed");

    let manifest_path = temp.join(".context/manifest.json");
    let manifest = std::fs::read_to_string(manifest_path)?;
    let manifest_json: serde_json::Value =
        serde_json::from_str(&manifest).map_err(io::Error::other)?;
    let collection = manifest_json
        .get("collectionName")
        .and_then(|value| value.as_str())
        .ok_or_else(|| io::Error::other("manifest missing collectionName"))?;
    let snapshot_path = temp
        .join(".context")
        .join("vector")
        .join("collections")
        .join(format!("{collection}.json"));
    assert!(
        snapshot_path.is_file(),
        "expected vector snapshot after index"
    );

    let output = run_cli_in_dir(&temp, &["clear"])?;
    assert!(output.status.success(), "clear failed");
    assert!(
        !snapshot_path.exists(),
        "expected vector snapshot removed after clear"
    );
    Ok(())
}

#[test]
fn cli_reindex_runs_after_index() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let output = run_cli_in_dir(&temp, &["index", "--init"])?;
    assert!(output.status.success(), "index --init failed");

    let output = run_cli_in_dir(&temp, &["reindex"])?;
    assert!(output.status.success(), "reindex failed");
    Ok(())
}

#[test]
fn cli_status_reports_after_index() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let output = run_cli_in_dir(&temp, &["index", "--init"])?;
    assert!(output.status.success(), "index --init failed");

    let output = run_cli_in_dir(&temp, &["status"])?;
    assert!(output.status.success(), "status failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("status: ok"));
    Ok(())
}

#[test]
fn cli_emits_structured_logs_when_enabled() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let output = run_cli_in_dir(&temp, &["index", "--init"])?;
    assert!(output.status.success(), "index --init failed");

    let output = run_cli_in_dir_with_env(
        &temp,
        &["search", "--query", "local-index"],
        &[("SCA_LOG_FORMAT", "json"), ("SCA_TELEMETRY_FORMAT", "json")],
    )?;
    assert!(output.status.success(), "search failed");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut saw_event = false;
    let mut saw_correlation = false;
    for line in stderr.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if value
            .get("event")
            .and_then(|event| event.as_str())
            .is_some()
        {
            saw_event = true;
        }
        if let Some(fields) = value.get("fields").and_then(|fields| fields.as_object()) {
            if fields.contains_key("correlationId") {
                saw_correlation = true;
            }
        }
    }

    assert!(saw_event, "expected structured log event in stderr");
    assert!(
        saw_correlation,
        "expected correlationId to be present in structured logs"
    );
    Ok(())
}
