//! Adapter selection tests for infra factories.

use semantic_code_config::{BackendConfig, BackendEnv, SnapshotStorageMode};
use semantic_code_infra::{build_embedding_port, build_vectordb_port};
use semantic_code_shared::ErrorCode;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(label: &str) -> std::io::Result<PathBuf> {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("sca-infra-{label}-{unique}"));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn empty_env() -> BackendEnv {
    BackendEnv::from_map(&BTreeMap::new()).expect("env parse")
}

fn config_root(path: &Path) -> Box<str> {
    path.to_string_lossy().to_string().into_boxed_str()
}

#[tokio::test]
async fn embedding_selection_uses_config_provider_test() {
    let root = temp_dir("embedding-test").expect("temp dir");
    let mut config = BackendConfig::default();
    config.embedding.provider = Some("test".to_string().into_boxed_str());
    let config = config.validate_and_normalize().expect("config");

    let port = build_embedding_port(&config, &empty_env(), &root).expect("port");
    assert_eq!(port.provider().id.as_str(), "test");
}

#[tokio::test]
async fn vectordb_selection_uses_config_provider_local() {
    let root = temp_dir("vectordb-local").expect("temp dir");
    let mut config = BackendConfig::default();
    config.vector_db.provider = Some("local".to_string().into_boxed_str());
    let config = config.validate_and_normalize().expect("config");

    let port = build_vectordb_port(&config, &root, SnapshotStorageMode::Disabled)
        .await
        .expect("port");
    assert_eq!(port.provider().id.as_str(), "local");
}

#[tokio::test]
async fn embedding_openai_requires_api_key_message() {
    let root = temp_dir("embedding-openai").expect("temp dir");
    let mut config = BackendConfig::default();
    config.embedding.provider = Some("openai".to_string().into_boxed_str());
    let config = config.validate_and_normalize().expect("config");

    let error = match build_embedding_port(&config, &empty_env(), &root) {
        Ok(_) => panic!("expected OpenAI config to fail without API key"),
        Err(error) => error,
    };
    assert!(
        error.message.contains("OpenAI API key is required"),
        "unexpected error message: {}",
        error.message
    );
}

#[tokio::test]
async fn onnx_missing_assets_error_is_user_friendly() {
    let root = temp_dir("embedding-onnx").expect("temp dir");
    let model_dir = root.join("missing-onnx");
    std::fs::create_dir_all(&model_dir).expect("create model dir");

    let mut config = BackendConfig::default();
    config.embedding.provider = Some("onnx".to_string().into_boxed_str());
    config.embedding.onnx.model_dir = Some(config_root(&model_dir));
    config.embedding.onnx.download_on_missing = false;
    let config = config.validate_and_normalize().expect("config");

    let error = match build_embedding_port(&config, &empty_env(), &root) {
        Ok(_) => panic!("expected ONNX config to fail when assets are missing"),
        Err(error) => error,
    };
    assert_eq!(
        error.code,
        ErrorCode::new("embedding", "onnx_assets_missing")
    );
    assert!(error.message.contains("SCA_EMBEDDING_ONNX_MODEL_DIR"));
}
