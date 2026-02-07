//! Phase 07 CLI flow E2E tests.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_FIXTURE_REPO: &str = "tmp/dspy/dspy";
const ENV_FIXTURE_REPO: &str = "SCA_E2E_FIXTURE_REPO";

fn run_cli_in_dir(dir: &Path, args: &[&str]) -> io::Result<std::process::Output> {
    Command::new(env!("CARGO_BIN_EXE_sca"))
        .current_dir(dir)
        .env("SCA_EMBEDDING_PROVIDER", "test")
        .args(args)
        .output()
}

fn copy_fixture_repo() -> io::Result<PathBuf> {
    let source = fixture_repo_root()?;
    if !source.is_dir() {
        return Err(io::Error::other(format!(
            "fixture repo missing at {}; set {ENV_FIXTURE_REPO} or update .env.local",
            source.display()
        )));
    }
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dest = std::env::temp_dir().join(format!("sca-cli-e2e-p7-{unique}"));
    copy_dir_recursive(&source, &dest)?;
    Ok(dest)
}

fn fixture_repo_root() -> io::Result<PathBuf> {
    if let Ok(value) = std::env::var(ENV_FIXTURE_REPO) {
        return Ok(resolve_repo_path(&value));
    }
    if let Some(value) = read_env_local_value(ENV_FIXTURE_REPO)? {
        return Ok(resolve_repo_path(&value));
    }
    Ok(PathBuf::from(DEFAULT_FIXTURE_REPO))
}

fn resolve_repo_path(value: &str) -> PathBuf {
    let trimmed = value.trim();
    let candidate = PathBuf::from(trimmed);
    if candidate.is_absolute() {
        candidate
    } else {
        workspace_root().join(candidate)
    }
}

fn read_env_local_value(key: &str) -> io::Result<Option<String>> {
    let path = env_local_path();
    let contents = match std::fs::read_to_string(&path) {
        Ok(value) => value,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((left, right)) = line.split_once('=') else {
            continue;
        };
        if left.trim() == key {
            return Ok(Some(unquote_env_value(right)));
        }
    }
    Ok(None)
}

fn unquote_env_value(value: &str) -> String {
    let trimmed = value.trim();
    if let Some(stripped) = trimmed
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
    {
        return stripped.to_string();
    }
    if let Some(stripped) = trimmed
        .strip_prefix('\'')
        .and_then(|value| value.strip_suffix('\''))
    {
        return stripped.to_string();
    }
    trimmed.to_string()
}

fn env_local_path() -> PathBuf {
    workspace_root().join(".env.local")
}

fn workspace_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_dir.to_path_buf())
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
fn phase7_cli_flow_init_index_status() -> io::Result<()> {
    let temp = copy_fixture_repo()?;

    let output = run_cli_in_dir(&temp, &["init"])?;
    assert!(output.status.success(), "init failed");

    let output = run_cli_in_dir(&temp, &["index"])?;
    assert!(output.status.success(), "index failed");

    let output = run_cli_in_dir(&temp, &["status"])?;
    assert!(output.status.success(), "status failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("status: ok"));
    Ok(())
}

#[test]
fn phase7_cli_flow_scripted_agent() -> io::Result<()> {
    let temp = copy_fixture_repo()?;

    let output = run_cli_in_dir(&temp, &["--agent", "index", "--init"])?;
    assert!(output.status.success(), "agent index failed");

    let summary: serde_json::Value =
        serde_json::from_slice(&output.stdout).map_err(io::Error::other)?;
    assert_eq!(summary.get("type").and_then(|value| value.as_str()), Some("summary"));
    assert_eq!(summary.get("status").and_then(|value| value.as_str()), Some("ok"));

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
    assert_eq!(summary.get("status").and_then(|value| value.as_str()), Some("ok"));
    assert!(saw_result, "expected at least one result line");
    Ok(())
}
