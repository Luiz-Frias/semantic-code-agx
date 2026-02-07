//! Integration tests for the local file sync adapter.

use semantic_code_adapters::file_sync::LocalFileSync;
use semantic_code_config::SnapshotStorageMode;
use semantic_code_ports::{FileSyncInitOptions, FileSyncOptions, FileSyncPort};
use semantic_code_shared::{ErrorEnvelope, RequestContext, Result};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{nanos}"))
}

async fn write_file(path: &Path, contents: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(ErrorEnvelope::from)?;
    }
    tokio::fs::write(path, contents)
        .await
        .map_err(ErrorEnvelope::from)?;
    Ok(())
}

#[tokio::test]
async fn detects_added_removed_modified_files() -> Result<()> {
    let root = temp_dir("sync-change");
    tokio::fs::create_dir_all(&root)
        .await
        .map_err(ErrorEnvelope::from)?;
    write_file(&root.join("a.txt"), "alpha").await?;
    write_file(&root.join("b.txt"), "bravo").await?;

    let sync = LocalFileSync::new(root.clone(), SnapshotStorageMode::Disabled);
    let ctx = RequestContext::new_request();
    sync.initialize(
        &ctx,
        FileSyncInitOptions {
            codebase_root: root.clone(),
            ignore_patterns: None,
        },
    )
    .await?;

    let first = sync
        .check_for_changes(&ctx, FileSyncOptions::default())
        .await?;
    assert_eq!(first.added, vec!["a.txt".into(), "b.txt".into()]);
    assert!(first.removed.is_empty());
    assert!(first.modified.is_empty());

    write_file(&root.join("a.txt"), "alpha-updated").await?;
    tokio::fs::remove_file(root.join("b.txt"))
        .await
        .map_err(ErrorEnvelope::from)?;
    write_file(&root.join("c.txt"), "charlie").await?;

    let second = sync
        .check_for_changes(&ctx, FileSyncOptions::default())
        .await?;
    assert_eq!(second.added, vec!["c.txt".into()]);
    assert_eq!(second.removed, vec!["b.txt".into()]);
    assert_eq!(second.modified, vec!["a.txt".into()]);

    Ok(())
}

#[tokio::test]
async fn ignores_configured_patterns() -> Result<()> {
    let root = temp_dir("sync-ignore");
    tokio::fs::create_dir_all(&root)
        .await
        .map_err(ErrorEnvelope::from)?;
    write_file(&root.join("visible.txt"), "ok").await?;
    write_file(&root.join("node_modules/pkg/index.js"), "skip").await?;

    let sync = LocalFileSync::new(root.clone(), SnapshotStorageMode::Disabled);
    let ctx = RequestContext::new_request();
    sync.initialize(
        &ctx,
        FileSyncInitOptions {
            codebase_root: root.clone(),
            ignore_patterns: Some(vec!["node_modules/".into()]),
        },
    )
    .await?;

    let changes = sync
        .check_for_changes(&ctx, FileSyncOptions::default())
        .await?;
    assert_eq!(changes.added, vec!["visible.txt".into()]);
    Ok(())
}

#[tokio::test]
async fn snapshot_persists_across_instances() -> Result<()> {
    let root = temp_dir("sync-persist");
    tokio::fs::create_dir_all(&root)
        .await
        .map_err(ErrorEnvelope::from)?;
    write_file(&root.join("a.txt"), "alpha").await?;

    let snapshot_dir = temp_dir("sync-snapshots");
    tokio::fs::create_dir_all(&snapshot_dir)
        .await
        .map_err(ErrorEnvelope::from)?;

    let storage = SnapshotStorageMode::Custom(snapshot_dir.clone());
    let ctx = RequestContext::new_request();

    let first = LocalFileSync::new(root.clone(), storage.clone());
    first
        .initialize(
            &ctx,
            FileSyncInitOptions {
                codebase_root: root.clone(),
                ignore_patterns: None,
            },
        )
        .await?;
    let changes = first
        .check_for_changes(&ctx, FileSyncOptions::default())
        .await?;
    assert_eq!(changes.added, vec!["a.txt".into()]);

    let second = LocalFileSync::new(root.clone(), storage);
    second
        .initialize(
            &ctx,
            FileSyncInitOptions {
                codebase_root: root.clone(),
                ignore_patterns: None,
            },
        )
        .await?;
    let changes = second
        .check_for_changes(&ctx, FileSyncOptions::default())
        .await?;
    assert!(changes.added.is_empty());
    assert!(changes.removed.is_empty());
    assert!(changes.modified.is_empty());

    Ok(())
}
