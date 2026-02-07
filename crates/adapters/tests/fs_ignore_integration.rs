//! Filesystem + ignore integration tests.

use semantic_code_adapters::fs::{LocalFileSystem, LocalPathPolicy};
use semantic_code_adapters::ignore::IgnoreMatcher;
use semantic_code_ports::{
    FileSystemEntryKind, FileSystemPort, IgnoreMatchInput, IgnorePort, PathPolicyPort,
    SafeRelativePath,
};
use semantic_code_shared::{RequestContext, Result};
use std::path::{Path, PathBuf};

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../testkit/fixtures/fs/basic")
}

#[tokio::test]
async fn scan_fixture_with_ignore_patterns() -> Result<()> {
    let ctx = RequestContext::new_request();
    let fs = LocalFileSystem::default();
    let policy = LocalPathPolicy::new();
    let ignore = IgnoreMatcher::new();
    let root = fixture_root();

    let mut pending = vec![SafeRelativePath::new(".")?];
    let mut files = Vec::new();
    while let Some(dir) = pending.pop() {
        let entries = fs.read_dir(&ctx, root.clone(), dir.clone()).await?;
        for entry in entries {
            let rel = if dir.as_str() == "." {
                entry.name.to_string()
            } else {
                format!("{}/{}", dir.as_str(), entry.name)
            };
            let ignored = ignore.is_ignored(&IgnoreMatchInput {
                ignore_patterns: vec!["node_modules/".into(), "target/".into()],
                relative_path: rel.clone().into_boxed_str(),
            });
            if ignored {
                continue;
            }
            match entry.kind {
                FileSystemEntryKind::Directory => {
                    let safe = policy.to_safe_relative_path(&rel)?;
                    pending.push(safe);
                },
                FileSystemEntryKind::File => files.push(rel),
                FileSystemEntryKind::Other => {},
            }
        }
    }

    files.sort();
    assert_eq!(files, vec!["README.md", "src/lib.rs", "src/main.rs"]);
    Ok(())
}

#[tokio::test]
async fn read_dir_returns_sorted_entries() -> Result<()> {
    let ctx = RequestContext::new_request();
    let fs = LocalFileSystem::default();
    let root = fixture_root();
    let entries = fs
        .read_dir(&ctx, root.clone(), SafeRelativePath::new("src")?)
        .await?;

    let names: Vec<&str> = entries.iter().map(|entry| entry.name.as_ref()).collect();
    assert_eq!(names, vec!["lib.rs", "main.rs"]);
    Ok(())
}

#[tokio::test]
async fn stat_reports_file_metadata() -> Result<()> {
    let ctx = RequestContext::new_request();
    let fs = LocalFileSystem::default();
    let root = fixture_root();
    let stat = fs
        .stat(&ctx, root.clone(), SafeRelativePath::new("src/lib.rs")?)
        .await?;
    assert_eq!(stat.kind, FileSystemEntryKind::File);
    assert!(stat.size_bytes > 0);
    Ok(())
}
