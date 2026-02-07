//! Integration tests for parsing config fixtures from the workspace testkit.

use semantic_code_config::{
    CURRENT_CONFIG_VERSION, SnapshotStorageMode, parse_backend_config_json,
    parse_backend_config_toml,
};
use semantic_code_shared::ErrorCode;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_dir.to_path_buf())
}

fn read_fixture(relative: &str) -> Result<String, Box<dyn Error>> {
    let path = workspace_root()
        .join("crates")
        .join("testkit")
        .join("fixtures")
        .join(relative);
    Ok(fs::read_to_string(path)?)
}

#[test]
fn parses_valid_fixture_and_normalizes() -> Result<(), Box<dyn Error>> {
    let contents = read_fixture("config/backend-config.valid.json")?;
    let config = parse_backend_config_json(&contents)?;

    assert_eq!(config.version, CURRENT_CONFIG_VERSION);
    assert_eq!(config.core.max_concurrency, 16);
    assert_eq!(config.core.timeout_ms, 45_000);

    assert_eq!(
        config.embedding.provider.as_deref(),
        Some("openai"),
        "provider should be trimmed"
    );
    assert_eq!(config.embedding.dimension, Some(1536));
    assert_eq!(config.vector_db.provider.as_deref(), Some("local"));

    let extensions: Vec<&str> = config
        .sync
        .allowed_extensions
        .iter()
        .map(AsRef::as_ref)
        .collect();
    assert_eq!(extensions, vec!["rs", "ts", "tsx"]);

    let patterns: Vec<&str> = config
        .sync
        .ignore_patterns
        .iter()
        .map(AsRef::as_ref)
        .collect();
    assert_eq!(
        patterns,
        vec![".context/", "dist/", "node_modules/", "target/"]
    );

    Ok(())
}

#[test]
fn parses_default_toml_fixture() -> Result<(), Box<dyn Error>> {
    let contents = read_fixture("config/backend-config.default.toml")?;
    let config = parse_backend_config_toml(&contents)?;

    assert_eq!(config.core.timeout_ms, 30_000);
    assert_eq!(config.core.max_chunk_chars, 2_500);
    assert_eq!(
        config.vector_db.snapshot_storage,
        SnapshotStorageMode::Project
    );

    Ok(())
}

#[test]
fn invalid_fixture_reports_error_code() -> Result<(), Box<dyn Error>> {
    let contents = read_fixture("config/backend-config.invalid.json")?;
    let result = parse_backend_config_json(&contents);
    assert!(result.is_err());

    let error = result
        .err()
        .ok_or_else(|| std::io::Error::other("expected invalid fixture error"))?;

    assert_eq!(error.code, ErrorCode::new("config", "invalid_timeout"));
    assert_eq!(
        error.metadata.get("section").map(String::as_str),
        Some("core")
    );
    assert_eq!(
        error.metadata.get("field").map(String::as_str),
        Some("timeoutMs")
    );

    Ok(())
}
