//! Adapter selection tests for infra factories.

use crate::embedding_factory::build_embedding_port_with_telemetry;
use crate::vectordb_factory::build_vectordb_port;
use semantic_code_config::{
    BackendConfig, RuntimeEnv, SnapshotStorageMode, VectorKernelKind, VectorSearchStrategy,
    VectorSnapshotFormat,
};
use semantic_code_domain::{CollectionName, LineSpan, VectorDocumentMetadata};
use semantic_code_ports::{
    VectorDbPort, VectorDocumentForInsert, VectorSearchOptions, VectorSearchRequest,
};
use semantic_code_shared::ErrorCode;
use std::path::{Path, PathBuf};
use std::sync::Arc;
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

fn empty_env() -> RuntimeEnv {
    RuntimeEnv::default()
}

fn config_root(path: &Path) -> Box<str> {
    path.to_string_lossy().to_string().into_boxed_str()
}

fn sample_metadata(path: &str) -> Result<VectorDocumentMetadata, Box<dyn std::error::Error>> {
    Ok(VectorDocumentMetadata {
        relative_path: path.into(),
        language: None,
        file_extension: Some("rs".into()),
        span: LineSpan::new(1, 1)?,
        fragment_start_byte: None,
        fragment_end_byte: None,
        node_kind: None,
    })
}

#[tokio::test]
async fn embedding_selection_uses_config_provider_test() {
    let root = temp_dir("embedding-test").expect("temp dir");
    let mut config = BackendConfig::default();
    config.embedding.provider = Some("test".to_string().into_boxed_str());
    let config = config.validate_and_normalize().expect("config");

    let port =
        build_embedding_port_with_telemetry(&config, &empty_env(), &root, None).expect("port");
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
async fn vectordb_search_strategy_config_toggles_local_search_path()
-> Result<(), Box<dyn std::error::Error>> {
    let root = temp_dir("vectordb-u8-toggle")?;
    let collection = CollectionName::parse("u8_toggle")?;
    let ctx = semantic_code_shared::RequestContext::new_request();

    let mut base = BackendConfig::default();
    base.vector_db.provider = Some("local".into());
    base.vector_db.snapshot_storage = SnapshotStorageMode::Disabled;

    let mut f32_cfg = base.clone();
    f32_cfg.vector_db.search_strategy = Some(VectorSearchStrategy::F32Hnsw);
    let f32_cfg = f32_cfg.validate_and_normalize()?;
    let f32_port: Arc<dyn VectorDbPort> =
        build_vectordb_port(&f32_cfg, &root, SnapshotStorageMode::Disabled).await?;

    let mut u8_exact_cfg = base;
    u8_exact_cfg.vector_db.search_strategy = Some(VectorSearchStrategy::U8Exact);
    let u8_exact_cfg = u8_exact_cfg.validate_and_normalize()?;
    let u8_exact_port: Arc<dyn VectorDbPort> =
        build_vectordb_port(&u8_exact_cfg, &root, SnapshotStorageMode::Disabled).await?;

    f32_port
        .create_collection(&ctx, collection.clone(), 2, None)
        .await?;
    u8_exact_port
        .create_collection(&ctx, collection.clone(), 2, None)
        .await?;

    let docs = vec![
        VectorDocumentForInsert {
            id: "a".into(),
            vector: Arc::from(vec![1.0, 0.0]),
            content: "a".into(),
            metadata: sample_metadata("src/a.rs")?,
        },
        VectorDocumentForInsert {
            id: "b".into(),
            vector: Arc::from(vec![1.0, 0.001]),
            content: "b".into(),
            metadata: sample_metadata("src/b.rs")?,
        },
        VectorDocumentForInsert {
            id: "outlier".into(),
            vector: Arc::from(vec![0.0, 1000.0]),
            content: "outlier".into(),
            metadata: sample_metadata("src/c.rs")?,
        },
    ];

    f32_port
        .insert(&ctx, collection.clone(), docs.clone())
        .await?;
    u8_exact_port.insert(&ctx, collection.clone(), docs).await?;

    let request = VectorSearchRequest {
        collection_name: collection,
        query_vector: Arc::from(vec![1.0, 0.001]),
        options: VectorSearchOptions {
            top_k: Some(1),
            filter_expr: None,
            threshold: Some(0.1),
        },
    };
    let f32_results = f32_port.search(&ctx, request.clone()).await?;
    let u8_results = u8_exact_port.search(&ctx, request).await?;

    assert_eq!(
        f32_results
            .results
            .first()
            .map(|item| item.document.id.as_ref()),
        Some("b")
    );
    assert_eq!(
        u8_results
            .results
            .first()
            .map(|item| item.document.id.as_ref()),
        Some("a")
    );
    Ok(())
}

#[tokio::test]
async fn vectordb_snapshot_max_bytes_config_enforces_limit()
-> Result<(), Box<dyn std::error::Error>> {
    let root = temp_dir("vectordb-size-limit")?;
    let mut config = BackendConfig::default();
    config.vector_db.provider = Some("local".into());
    config.vector_db.snapshot_format = VectorSnapshotFormat::V2;
    config.vector_db.snapshot_max_bytes = Some(1);
    let config = config.validate_and_normalize()?;

    let port =
        build_vectordb_port(&config, &root, SnapshotStorageMode::Custom(root.clone())).await?;
    let ctx = semantic_code_shared::RequestContext::new_request();
    let collection = CollectionName::parse("size_limit_cfg")?;
    let error = port
        .create_collection(&ctx, collection, 3, None)
        .await
        .err()
        .ok_or_else(|| std::io::Error::other("expected snapshot size limit error"))?;
    assert_eq!(error.code, ErrorCode::new("vector", "snapshot_oversize"));
    Ok(())
}

#[tokio::test]
async fn vectordb_dfrr_kernel_selection_matches_build_support()
-> Result<(), Box<dyn std::error::Error>> {
    let root = temp_dir("vectordb-dfrr-kernel")?;
    let mut config = BackendConfig::default();
    config.vector_db.provider = Some("local".into());
    config.vector_db.vector_kernel = Some(VectorKernelKind::Dfrr);
    let config = config.validate_and_normalize()?;

    match build_vectordb_port(&config, &root, SnapshotStorageMode::Custom(root.clone())).await {
        Ok(port) => {
            assert_eq!(port.provider().id.as_str(), "local");
        },
        Err(error) => {
            assert_eq!(error.code, ErrorCode::new("vector", "kernel_unsupported"));
            assert_eq!(
                error.metadata.get("requestedKernel").map(String::as_str),
                Some("dfrr")
            );
        },
    }
    Ok(())
}

#[tokio::test]
async fn embedding_openai_requires_api_key_message() {
    let root = temp_dir("embedding-openai").expect("temp dir");
    let mut config = BackendConfig::default();
    config.embedding.provider = Some("openai".to_string().into_boxed_str());
    let config = config.validate_and_normalize().expect("config");

    let error = match build_embedding_port_with_telemetry(&config, &empty_env(), &root, None) {
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

    let error = match build_embedding_port_with_telemetry(&config, &empty_env(), &root, None) {
        Ok(_) => panic!("expected ONNX config to fail when assets are missing"),
        Err(error) => error,
    };
    assert_eq!(
        error.code,
        ErrorCode::new("embedding", "onnx_assets_missing")
    );
    assert!(error.message.contains("SCA_EMBEDDING_ONNX_MODEL_DIR"));
}
