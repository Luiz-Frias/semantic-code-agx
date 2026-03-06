//! Integration coverage for local vector DB snapshot v2 flows.

use semantic_code_adapters::LocalVectorDb;
use semantic_code_config::{SnapshotStorageMode, VectorSearchStrategy, VectorSnapshotFormat};
use semantic_code_domain::{CollectionName, LineSpan};
use semantic_code_ports::{
    VectorDbPort, VectorDocumentForInsert, VectorDocumentMetadata, VectorSearchOptions,
    VectorSearchRequest,
};
use semantic_code_shared::{ErrorEnvelope, RequestContext, Result};
use semantic_code_vector::HnswKernel;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(prefix: &str) -> Result<PathBuf> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!("{prefix}-{nanos}"));
    std::fs::create_dir_all(&path).map_err(ErrorEnvelope::from)?;
    Ok(path)
}

fn sample_metadata(path: &str) -> Result<VectorDocumentMetadata> {
    Ok(VectorDocumentMetadata {
        relative_path: path.into(),
        language: None,
        file_extension: Some("rs".into()),
        span: LineSpan::new(1, 1)?,
        node_kind: None,
    })
}

fn v1_snapshot_path(root: &Path, collection: &CollectionName) -> PathBuf {
    root.join("vector")
        .join("collections")
        .join(format!("{}.json", collection.as_str()))
}

fn v2_bundle_dir(root: &Path, collection: &CollectionName) -> PathBuf {
    root.join("vector")
        .join("collections")
        .join(format!("{}.v2", collection.as_str()))
}

#[tokio::test]
async fn index_snapshot_reload_with_v2() -> Result<()> {
    let root = temp_dir("sca-adapters-localdb-v2")?;
    let collection = CollectionName::parse("local_snapshot").map_err(ErrorEnvelope::from)?;
    let ctx = RequestContext::new_request();
    let db = LocalVectorDb::new(
        root.clone(),
        SnapshotStorageMode::Custom(root.clone()),
        VectorSnapshotFormat::V2,
        None,
        Arc::new(HnswKernel::new()),
        false,
        VectorSearchStrategy::F32Hnsw,
    )?;

    db.create_collection(&ctx, collection.clone(), 3, None)
        .await?;
    db.insert(
        &ctx,
        collection.clone(),
        vec![VectorDocumentForInsert {
            id: "doc1".into(),
            vector: Arc::from(vec![0.1, 0.2, 0.3]),
            content: "hello".into(),
            metadata: sample_metadata("src/lib.rs")?,
        }],
    )
    .await?;
    db.flush(&ctx, collection.clone()).await?;

    let bundle = v2_bundle_dir(&root, &collection);
    assert!(bundle.join("snapshot.meta").is_file());
    assert!(bundle.join("vectors.u8.bin").is_file());

    let restored = LocalVectorDb::new(
        root.clone(),
        SnapshotStorageMode::Custom(root.clone()),
        VectorSnapshotFormat::V2,
        None,
        Arc::new(HnswKernel::new()),
        false,
        VectorSearchStrategy::F32Hnsw,
    )?;
    let results = restored
        .search(
            &ctx,
            VectorSearchRequest {
                collection_name: collection,
                query_vector: Arc::from(vec![0.1, 0.2, 0.3]),
                options: VectorSearchOptions {
                    top_k: Some(1),
                    filter_expr: None,
                    threshold: None,
                },
            },
        )
        .await?;

    assert_eq!(results.results.len(), 1);
    assert_eq!(results.results[0].document.id, "doc1".into());
    Ok(())
}

#[tokio::test]
async fn upgrade_v1_snapshot_on_v2_load() -> Result<()> {
    let root = temp_dir("sca-adapters-localdb-upgrade")?;
    let collection = CollectionName::parse("local_upgrade").map_err(ErrorEnvelope::from)?;
    let ctx = RequestContext::new_request();

    let db_v1 = LocalVectorDb::new(
        root.clone(),
        SnapshotStorageMode::Custom(root.clone()),
        VectorSnapshotFormat::V1,
        None,
        Arc::new(HnswKernel::new()),
        false,
        VectorSearchStrategy::F32Hnsw,
    )?;
    db_v1
        .create_collection(&ctx, collection.clone(), 3, None)
        .await?;
    db_v1
        .insert(
            &ctx,
            collection.clone(),
            vec![VectorDocumentForInsert {
                id: "doc1".into(),
                vector: Arc::from(vec![0.1, 0.2, 0.3]),
                content: "hello".into(),
                metadata: sample_metadata("src/lib.rs")?,
            }],
        )
        .await?;
    db_v1.flush(&ctx, collection.clone()).await?;

    let v1_path = v1_snapshot_path(&root, &collection);
    let v2_bundle = v2_bundle_dir(&root, &collection);
    assert!(v1_path.is_file());
    assert!(!v2_bundle.join("snapshot.meta").is_file());
    assert!(!v2_bundle.join("vectors.u8.bin").is_file());

    let db_v2 = LocalVectorDb::new(
        root.clone(),
        SnapshotStorageMode::Custom(root.clone()),
        VectorSnapshotFormat::V2,
        None,
        Arc::new(HnswKernel::new()),
        false,
        VectorSearchStrategy::F32Hnsw,
    )?;
    let results = db_v2
        .search(
            &ctx,
            VectorSearchRequest {
                collection_name: collection,
                query_vector: Arc::from(vec![0.1, 0.2, 0.3]),
                options: VectorSearchOptions {
                    top_k: Some(1),
                    filter_expr: None,
                    threshold: None,
                },
            },
        )
        .await?;

    assert_eq!(results.results.len(), 1);
    assert!(v2_bundle.join("snapshot.meta").is_file());
    assert!(v2_bundle.join("vectors.u8.bin").is_file());
    Ok(())
}

#[tokio::test]
async fn v2_snapshot_size_limit_is_enforced() -> Result<()> {
    let root = temp_dir("sca-adapters-localdb-size-limit")?;
    let collection = CollectionName::parse("local_limit").map_err(ErrorEnvelope::from)?;
    let ctx = RequestContext::new_request();
    let db = LocalVectorDb::new(
        root.clone(),
        SnapshotStorageMode::Custom(root),
        VectorSnapshotFormat::V2,
        Some(1),
        Arc::new(HnswKernel::new()),
        false,
        VectorSearchStrategy::F32Hnsw,
    )?;

    let error = db
        .create_collection(&ctx, collection, 3, None)
        .await
        .err()
        .ok_or_else(|| std::io::Error::other("expected oversize snapshot error"))?;
    assert_eq!(
        error.code,
        semantic_code_shared::ErrorCode::new("vector", "snapshot_oversize")
    );
    assert_eq!(
        error.metadata.get("version").map(String::as_str),
        Some("v2")
    );
    Ok(())
}
