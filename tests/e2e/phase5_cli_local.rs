//! Phase 05 local CLI flow E2E tests.

use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
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
    let dest = std::env::temp_dir().join(format!("sca-cli-e2e-{unique}"));
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
fn phase5_cli_local_flow() -> io::Result<()> {
    let temp = copy_fixture_repo()?;
    assert!(
        run_cli_in_dir(&temp, &["index", "--init"])?
            .status
            .success()
    );
    assert!(
        run_cli_in_dir(&temp, &["search", "--query", "local-index"])?
            .status
            .success()
    );
    assert!(run_cli_in_dir(&temp, &["clear"])?.status.success());
    Ok(())
}
