//! Integration tests for env parsing and env-to-config merging.

use semantic_code_config::{BackendConfig, BackendEnv, EnvParseError, apply_env_overrides};
use semantic_code_shared::ErrorCode;
use std::collections::BTreeMap;
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

fn read_env_map(relative: &str) -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    let contents = read_fixture(relative)?;
    Ok(serde_json::from_str(&contents)?)
}

#[test]
fn env_fixtures_merge_into_effective_config() -> Result<(), Box<dyn Error>> {
    let env_map = read_env_map("env/backend-env.valid.json")?;
    let env = BackendEnv::from_map(&env_map)?;

    let config = apply_env_overrides(BackendConfig::default(), &env)?;

    assert_eq!(config.core.timeout_ms, 45_000);
    assert_eq!(config.core.max_concurrency, 16);

    assert_eq!(config.embedding.provider.as_deref(), Some("openai"));
    assert_eq!(
        config.embedding.base_url.as_deref(),
        Some("https://example.com/v1")
    );
    assert_eq!(config.embedding.dimension, Some(1536));

    assert_eq!(config.vector_db.provider.as_deref(), Some("local"));
    assert_eq!(
        config.vector_db.index_mode,
        semantic_code_domain::IndexMode::Hybrid
    );
    assert_eq!(
        config.vector_db.base_url.as_deref(),
        Some("http://localhost:19530/")
    );

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
fn invalid_env_fixture_is_rejected() -> Result<(), Box<dyn Error>> {
    let env_map = read_env_map("env/backend-env.invalid.json")?;
    let error = BackendEnv::from_map(&env_map).err();
    assert!(matches!(error, Some(EnvParseError::InvalidUrl { .. })));

    let envelope: semantic_code_shared::ErrorEnvelope = error
        .ok_or_else(|| std::io::Error::other("expected invalid env error"))?
        .into();
    assert_eq!(envelope.code, ErrorCode::new("config", "invalid_env_url"));

    Ok(())
}
