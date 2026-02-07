//! Phase 03 config/request validation E2E tests.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{fs, io};

fn workspace_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_dir.to_path_buf())
}

fn fixture_path(relative: &str) -> PathBuf {
    workspace_root()
        .join("crates")
        .join("testkit")
        .join("fixtures")
        .join(relative)
}

fn read_fixture(relative: &str) -> io::Result<String> {
    fs::read_to_string(fixture_path(relative))
}

#[test]
fn phase3_config_and_request_validation_e2e() -> io::Result<()> {
    let valid_config = fixture_path("config/backend-config.valid.json");
    let invalid_config = fixture_path("config/backend-config.invalid.json");
    let search_request = read_fixture("requests/search.valid.json")?;

    // Config check (valid).
    let output = Command::new(env!("CARGO_BIN_EXE_sca"))
        .args(["--json", "config", "check", "--path"])
        .arg(&valid_config)
        .output()?;
    assert!(output.status.success());

    let value: serde_json::Value =
        serde_json::from_slice(&output.stdout).map_err(io::Error::other)?;
    let status = value
        .get("status")
        .and_then(|value| value.as_str())
        .ok_or_else(|| io::Error::other("missing status"))?;
    assert_eq!(status, "ok");

    // Request validation (valid).
    let output = Command::new(env!("CARGO_BIN_EXE_sca"))
        .args([
            "--json",
            "validate-request",
            "--kind",
            "search",
            "--input-json",
            &search_request,
        ])
        .output()?;
    assert!(output.status.success());

    let value: serde_json::Value =
        serde_json::from_slice(&output.stdout).map_err(io::Error::other)?;
    let status = value
        .get("status")
        .and_then(|value| value.as_str())
        .ok_or_else(|| io::Error::other("missing status"))?;
    assert_eq!(status, "ok");

    // Config check (invalid) emits structured errors.
    let output = Command::new(env!("CARGO_BIN_EXE_sca"))
        .args(["--json", "config", "check", "--path"])
        .arg(&invalid_config)
        .output()?;
    assert_eq!(output.status.code(), Some(2));

    let value: serde_json::Value =
        serde_json::from_slice(&output.stdout).map_err(io::Error::other)?;
    let status = value
        .get("status")
        .and_then(|value| value.as_str())
        .ok_or_else(|| io::Error::other("missing status"))?;
    assert_eq!(status, "error");

    let code = value
        .get("error")
        .and_then(|value| value.get("code"))
        .and_then(|value| value.as_str())
        .ok_or_else(|| io::Error::other("missing error.code"))?;
    assert_eq!(code, "ERR_CONFIG_INVALID_TIMEOUT");

    Ok(())
}
