//! Phase 04 CLI E2E tests.

use std::io;
use std::process::Command;

#[test]
fn phase4_self_check_includes_index_search_clear() -> io::Result<()> {
    let output = Command::new(env!("CARGO_BIN_EXE_sca"))
        .args(["self-check", "--json"])
        .env("SCA_EMBEDDING_PROVIDER", "test")
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(io::Error::other(format!("self-check failed: {stderr}")));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).map_err(io::Error::other)?;

    assert_eq!(value.get("status").and_then(|v| v.as_str()), Some("ok"));
    assert_eq!(
        value
            .get("index")
            .and_then(|v| v.get("status"))
            .and_then(|v| v.as_str()),
        Some("ok")
    );
    assert_eq!(
        value
            .get("search")
            .and_then(|v| v.get("status"))
            .and_then(|v| v.as_str()),
        Some("ok")
    );
    assert_eq!(
        value
            .get("clear")
            .and_then(|v| v.get("status"))
            .and_then(|v| v.as_str()),
        Some("ok")
    );

    Ok(())
}
