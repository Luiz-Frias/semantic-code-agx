//! Phase 01 CLI E2E smoke tests.

use std::io;
use std::process::Command;

fn run_self_check_json() -> io::Result<String> {
    let output = Command::new(env!("CARGO_BIN_EXE_sca"))
        .args(["self-check", "--json"])
        .env("SCA_EMBEDDING_PROVIDER", "test")
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(io::Error::other(format!("self-check failed: {stderr}")));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

#[test]
fn phase1_self_check_is_deterministic() -> io::Result<()> {
    let first = run_self_check_json()?;
    let second = run_self_check_json()?;

    assert_eq!(first, second, "self-check output should be deterministic");

    Ok(())
}
