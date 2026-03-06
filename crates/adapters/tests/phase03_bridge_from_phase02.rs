//! Bridge tests validating Phase 02 outputs for Phase 03.

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
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn create(prefix: &str) -> std::io::Result<Self> {
        let path = unique_temp_path(prefix);
        std::fs::create_dir_all(&path)?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        self.path.as_path()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn unique_temp_path(prefix: &str) -> PathBuf {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let seq = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}-{pid}-{nanos}-{seq}"))
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

fn v2_bundle_dir(root: &Path, collection: &CollectionName) -> PathBuf {
    root.join("vector")
        .join("collections")
        .join(format!("{}.v2", collection.as_str()))
}

#[tokio::test]
async fn phase03_bt02_localdb_snapshot_v2_load_path() -> Result<()> {
    let temp = TempDir::create("phase03-bridge-localdb-v2").map_err(ErrorEnvelope::from)?;
    let root = temp.path().to_path_buf();
    let collection = CollectionName::parse("phase03_snapshot").map_err(ErrorEnvelope::from)?;
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

    let restarted = LocalVectorDb::new(
        root.clone(),
        SnapshotStorageMode::Custom(root),
        VectorSnapshotFormat::V2,
        None,
        Arc::new(HnswKernel::new()),
        false,
        VectorSearchStrategy::F32Hnsw,
    )?;
    let results = restarted
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
    assert_eq!(results.results[0].document.id.as_ref(), "doc1");
    Ok(())
}
