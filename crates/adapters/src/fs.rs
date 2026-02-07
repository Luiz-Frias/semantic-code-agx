//! Filesystem and path policy adapters.

use semantic_code_ports::{
    FileSystemDirEntry, FileSystemEntryKind, FileSystemPort, FileSystemStat, PathPolicyPort,
    SafeRelativePath,
};
use semantic_code_shared::{ErrorCode, ErrorEnvelope, RequestContext, Result};
use std::path::PathBuf;
use std::time::{Duration, UNIX_EPOCH};

/// Local filesystem adapter using async IO.
#[derive(Debug, Clone, Default)]
pub struct LocalFileSystem {
    max_file_size_bytes: Option<u64>,
}

impl LocalFileSystem {
    /// Build a filesystem adapter with an optional max file size.
    pub const fn new(max_file_size_bytes: Option<u64>) -> Self {
        Self {
            max_file_size_bytes,
        }
    }
}

impl FileSystemPort for LocalFileSystem {
    fn read_dir(
        &self,
        _ctx: &RequestContext,
        codebase_root: PathBuf,
        dir: SafeRelativePath,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<FileSystemDirEntry>>> {
        Box::pin(async move {
            let full_path = codebase_root.join(dir.as_str());
            let mut entries = Vec::new();
            let mut read_dir = tokio::fs::read_dir(&full_path)
                .await
                .map_err(ErrorEnvelope::from)?;

            while let Some(entry) = read_dir.next_entry().await.map_err(ErrorEnvelope::from)? {
                let file_type = entry.file_type().await.map_err(ErrorEnvelope::from)?;
                let kind = if file_type.is_file() {
                    FileSystemEntryKind::File
                } else if file_type.is_dir() {
                    FileSystemEntryKind::Directory
                } else {
                    FileSystemEntryKind::Other
                };
                let name = entry
                    .file_name()
                    .to_string_lossy()
                    .to_string()
                    .into_boxed_str();
                entries.push(FileSystemDirEntry { name, kind });
            }

            entries.sort_by(|a, b| a.name.cmp(&b.name));
            Ok(entries)
        })
    }

    fn read_file_text(
        &self,
        _ctx: &RequestContext,
        codebase_root: PathBuf,
        file: SafeRelativePath,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Box<str>>> {
        let max_file_size_bytes = self.max_file_size_bytes;
        Box::pin(async move {
            let full_path = codebase_root.join(file.as_str());
            let metadata = tokio::fs::metadata(&full_path)
                .await
                .map_err(ErrorEnvelope::from)?;
            if !metadata.is_file() {
                return Err(ErrorEnvelope::expected(
                    ErrorCode::invalid_input(),
                    "path is not a file",
                ));
            }
            if let Some(limit) = max_file_size_bytes
                && metadata.len() > limit
            {
                return Err(ErrorEnvelope::expected(
                    ErrorCode::invalid_input(),
                    "file exceeds max size",
                ));
            }

            let contents = tokio::fs::read_to_string(&full_path)
                .await
                .map_err(ErrorEnvelope::from)?;
            Ok(contents.into_boxed_str())
        })
    }

    fn stat(
        &self,
        _ctx: &RequestContext,
        codebase_root: PathBuf,
        path: SafeRelativePath,
    ) -> semantic_code_ports::BoxFuture<'_, Result<FileSystemStat>> {
        Box::pin(async move {
            let full_path = codebase_root.join(path.as_str());
            let metadata = tokio::fs::metadata(&full_path)
                .await
                .map_err(ErrorEnvelope::from)?;
            let file_type = metadata.file_type();
            let kind = if file_type.is_file() {
                FileSystemEntryKind::File
            } else if file_type.is_dir() {
                FileSystemEntryKind::Directory
            } else {
                FileSystemEntryKind::Other
            };

            let mtime_ms = metadata
                .modified()
                .ok()
                .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
                .unwrap_or(Duration::from_secs(0))
                .as_millis();
            let mtime_ms = u64::try_from(mtime_ms).unwrap_or(0);

            Ok(FileSystemStat {
                kind,
                size_bytes: metadata.len(),
                mtime_ms,
            })
        })
    }
}

/// Local path policy for safe relative paths.
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalPathPolicy;

impl LocalPathPolicy {
    /// Build a default path policy.
    pub const fn new() -> Self {
        Self
    }
}

impl PathPolicyPort for LocalPathPolicy {
    fn to_safe_relative_path(&self, input: &str) -> Result<SafeRelativePath> {
        let safe = SafeRelativePath::new(input)?;
        if is_state_dir(safe.as_str()) {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "state directory paths are not allowed",
            )
            .with_metadata("path", safe.as_str().to_string()));
        }
        Ok(safe)
    }
}

fn is_state_dir(path: &str) -> bool {
    path == ".context" || path.starts_with(".context/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn path_policy_rejects_absolute_paths() {
        let policy = LocalPathPolicy::new();
        let err = policy
            .to_safe_relative_path("/tmp/repo")
            .expect_err("expected absolute path to fail");
        assert_eq!(err.code, ErrorCode::invalid_input());
    }

    #[test]
    fn path_policy_rejects_traversal() {
        let policy = LocalPathPolicy::new();
        let err = policy
            .to_safe_relative_path("../repo")
            .expect_err("expected traversal path to fail");
        assert_eq!(err.code, ErrorCode::invalid_input());
    }

    #[test]
    fn path_policy_rejects_state_dir() {
        let policy = LocalPathPolicy::new();
        let err = policy
            .to_safe_relative_path(".context/manifest.json")
            .expect_err("expected state dir to fail");
        assert_eq!(err.code, ErrorCode::invalid_input());
    }

    #[tokio::test]
    async fn read_file_text_enforces_max_size() -> Result<()> {
        let root = std::env::temp_dir().join("sca_fs_limit_test");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).map_err(ErrorEnvelope::from)?;
        let file_path = root.join("big.txt");
        fs::write(&file_path, "0123456789").map_err(ErrorEnvelope::from)?;

        let fs = LocalFileSystem::new(Some(5));
        let result = fs
            .read_file_text(
                &RequestContext::new_request(),
                root.clone(),
                SafeRelativePath::new("big.txt")?,
            )
            .await;

        let err = result.expect_err("expected max size error");
        assert_eq!(err.code, ErrorCode::invalid_input());
        Ok(())
    }
}
