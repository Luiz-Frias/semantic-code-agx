//! Integration tests for parsing config fixtures from the workspace testkit.

use semantic_code_config::{
    BackendConfig, SnapshotStorageMode, VectorKernelKind, VectorSearchStrategy,
    VectorSnapshotFormat,
};
use semantic_code_shared::{ErrorCode, ErrorEnvelope};
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

fn parse_json_config(
    input: &str,
) -> Result<semantic_code_config::ValidatedBackendConfig, ErrorEnvelope> {
    semantic_code_config::load_backend_config_from_sources(Some(input), None, &BTreeMap::new())
}

fn parse_toml_config(
    input: &str,
) -> Result<semantic_code_config::ValidatedBackendConfig, ErrorEnvelope> {
    let config: BackendConfig = toml::from_str(input).map_err(|error| {
        ErrorEnvelope::expected(
            ErrorCode::new("config", "invalid_toml"),
            format!("invalid config TOML: {error}"),
        )
    })?;
    Ok(config
        .validate_and_normalize()
        .map_err(ErrorEnvelope::from)?)
}

#[test]
fn parses_valid_fixture_and_normalizes() -> Result<(), Box<dyn Error>> {
    let contents = read_fixture("config/backend-config.valid.json")?;
    let config = parse_json_config(&contents)?;

    assert_eq!(config.version, BackendConfig::default().version);
    assert_eq!(config.core.max_concurrency, 16);
    assert_eq!(config.core.timeout_ms, 45_000);

    assert_eq!(
        config.embedding.provider.as_deref(),
        Some("openai"),
        "provider should be trimmed"
    );
    assert_eq!(config.embedding.dimension, Some(1536));
    assert_eq!(config.vector_db.provider.as_deref(), Some("local"));
    assert_eq!(config.vector_db.snapshot_format, VectorSnapshotFormat::V1);
    assert_eq!(config.vector_db.snapshot_max_bytes, None);
    assert_eq!(
        config.vector_db.effective_vector_kernel(),
        VectorKernelKind::HnswRs
    );
    assert!(!config.vector_db.experimental_u8_search);
    assert!(!config.vector_db.enable_search_metrics);
    assert_eq!(
        config.vector_db.effective_search_strategy(),
        VectorSearchStrategy::F32Hnsw
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
fn parses_default_toml_fixture() -> Result<(), Box<dyn Error>> {
    let contents = read_fixture("config/backend-config.default.toml")?;
    let config = parse_toml_config(&contents)?;

    assert_eq!(config.core.timeout_ms, 30_000);
    assert_eq!(config.core.max_chunk_chars, 2_500);
    assert_eq!(
        config.vector_db.snapshot_storage,
        SnapshotStorageMode::Project
    );
    assert_eq!(config.vector_db.snapshot_format, VectorSnapshotFormat::V1);
    assert_eq!(config.vector_db.snapshot_max_bytes, None);
    assert_eq!(
        config.vector_db.effective_vector_kernel(),
        VectorKernelKind::HnswRs
    );
    assert!(!config.vector_db.experimental_u8_search);
    assert_eq!(
        config.vector_db.effective_search_strategy(),
        VectorSearchStrategy::F32Hnsw
    );

    Ok(())
}

#[test]
fn parses_v2_snapshot_format_fixture() -> Result<(), Box<dyn Error>> {
    let contents = read_fixture("config/backend-config.snapshot-v2.toml")?;
    let config = parse_toml_config(&contents)?;

    assert_eq!(config.vector_db.provider.as_deref(), Some("local"));
    assert_eq!(config.vector_db.snapshot_format, VectorSnapshotFormat::V2);
    assert_eq!(config.vector_db.snapshot_max_bytes, Some(4_194_304));
    assert_eq!(
        config.vector_db.effective_vector_kernel(),
        VectorKernelKind::HnswRs
    );
    assert!(!config.vector_db.experimental_u8_search);
    assert_eq!(
        config.vector_db.effective_search_strategy(),
        VectorSearchStrategy::F32Hnsw
    );

    Ok(())
}

#[test]
fn inline_fixture_honors_explicit_kernel_and_force_reindex() -> Result<(), Box<dyn Error>> {
    let contents = r#"{
      "version": 1,
      "vectorDb": {
        "vectorKernel": "dfrr",
        "forceReindexOnKernelChange": true
      }
    }"#;
    let config = parse_json_config(contents)?;

    assert_eq!(config.vector_db.vector_kernel, Some(VectorKernelKind::Dfrr));
    assert_eq!(
        config.vector_db.effective_vector_kernel(),
        VectorKernelKind::Dfrr
    );
    assert!(config.vector_db.force_reindex_on_kernel_change);
    Ok(())
}

#[test]
fn inline_fixture_enable_search_metrics_flag() -> Result<(), Box<dyn Error>> {
    let with_flag = r#"{
      "version": 1,
      "vectorDb": {
        "enableSearchMetrics": true
      }
    }"#;
    let config = parse_json_config(with_flag)?;
    assert!(config.vector_db.enable_search_metrics);

    let without_flag = r#"{
      "version": 1,
      "vectorDb": {}
    }"#;
    let config = parse_json_config(without_flag)?;
    assert!(!config.vector_db.enable_search_metrics);

    Ok(())
}

#[test]
fn invalid_fixture_reports_error_code() -> Result<(), Box<dyn Error>> {
    let contents = read_fixture("config/backend-config.invalid.json")?;
    let result = parse_json_config(&contents);
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
