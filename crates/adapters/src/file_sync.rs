//! Local file sync adapter backed by Merkle snapshots.

use crate::ignore::IgnoreMatcher;
use semantic_code_config::SnapshotStorageMode;
use semantic_code_ports::{
    FileChangeSet, FileSyncInitOptions, FileSyncOptions, FileSyncPort, IgnoreMatchInput, IgnorePort,
};
use semantic_code_shared::merkle::{MerkleDag, MerkleDagSerialized};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

const SNAPSHOT_VERSION: u32 = 1;
const SNAPSHOT_DIR: &str = "sync";
const SNAPSHOT_FILE_EXT: &str = "json";
const CONTEXT_DIR_PATTERN: &str = ".context/";

/// Local filesystem-based file sync adapter.
#[derive(Clone)]
pub struct LocalFileSync {
    codebase_root: PathBuf,
    storage_mode: SnapshotStorageMode,
    state: Arc<RwLock<SyncState>>,
}

#[derive(Debug, Clone)]
struct SyncState {
    ignore_patterns: Vec<Box<str>>,
    file_hashes: BTreeMap<Box<str>, Box<str>>,
    merkle_dag: MerkleDag,
}

impl Default for SyncState {
    fn default() -> Self {
        Self {
            ignore_patterns: Vec::new(),
            file_hashes: BTreeMap::new(),
            merkle_dag: MerkleDag::new(),
        }
    }
}

impl LocalFileSync {
    /// Create a local file sync adapter scoped to a codebase root.
    #[must_use]
    pub fn new(codebase_root: PathBuf, storage_mode: SnapshotStorageMode) -> Self {
        Self {
            codebase_root,
            storage_mode,
            state: Arc::new(RwLock::new(SyncState::default())),
        }
    }

    fn snapshot_root(&self) -> Option<PathBuf> {
        self.storage_mode
            .resolve_root(&self.codebase_root)
            .map(|root| root.join(SNAPSHOT_DIR))
    }

    fn snapshot_path(&self) -> Option<PathBuf> {
        let root = self.snapshot_root()?;
        let normalized = normalize_root_path(&self.codebase_root);
        let digest = md5::compute(normalized.to_string_lossy().as_bytes());
        let hash = format!("{digest:x}");
        Some(root.join(format!("{hash}.{SNAPSHOT_FILE_EXT}")))
    }

    fn resolve_snapshot_mode(&self) -> bool {
        self.snapshot_root().is_some()
    }

    async fn load_snapshot(&self) -> Result<Option<SyncSnapshot>> {
        let Some(path) = self.snapshot_path() else {
            return Ok(None);
        };

        match tokio::fs::read(&path).await {
            Ok(payload) => {
                let snapshot: SyncSnapshot = serde_json::from_slice(&payload).map_err(|error| {
                    snapshot_error("snapshot_parse_failed", "failed to parse snapshot", error)
                })?;
                Ok(Some(snapshot))
            },
            Err(error) => {
                if error.kind() == std::io::ErrorKind::NotFound {
                    Ok(None)
                } else {
                    Err(ErrorEnvelope::from(error))
                }
            },
        }
    }

    async fn write_snapshot(&self, snapshot: &SyncSnapshot) -> Result<()> {
        let Some(path) = self.snapshot_path() else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(ErrorEnvelope::from)?;
        }
        let payload = serde_json::to_vec_pretty(snapshot).map_err(|error| {
            snapshot_error(
                "snapshot_serialize_failed",
                "failed to serialize snapshot",
                error,
            )
        })?;
        tokio::fs::write(&path, payload)
            .await
            .map_err(ErrorEnvelope::from)?;
        Ok(())
    }

    fn ensure_root_matches(&self, provided: &Path) -> Result<()> {
        let expected = normalize_root_path(&self.codebase_root);
        let provided = normalize_root_path(provided);
        if expected != provided {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "codebase root mismatch",
            )
            .with_metadata("expected", expected.to_string_lossy().to_string())
            .with_metadata("provided", provided.to_string_lossy().to_string()));
        }
        Ok(())
    }

    fn merged_ignore_patterns(input: Option<Vec<Box<str>>>) -> Vec<Box<str>> {
        let mut patterns = input.unwrap_or_default();
        if !patterns
            .iter()
            .any(|pattern| pattern.as_ref() == CONTEXT_DIR_PATTERN)
        {
            patterns.push(CONTEXT_DIR_PATTERN.into());
        }
        patterns
    }

    async fn generate_file_hashes(&self, ignore_patterns: &[Box<str>]) -> Result<FileHashMap> {
        let mut file_hashes = BTreeMap::new();
        self.scan_dir(&self.codebase_root, ignore_patterns, &mut file_hashes)
            .await?;
        Ok(file_hashes)
    }

    async fn scan_dir(
        &self,
        dir: &Path,
        ignore_patterns: &[Box<str>],
        file_hashes: &mut FileHashMap,
    ) -> Result<()> {
        let mut pending = VecDeque::new();
        pending.push_back(dir.to_path_buf());

        while let Some(current) = pending.pop_front() {
            let mut entries = tokio::fs::read_dir(&current)
                .await
                .map_err(ErrorEnvelope::from)?;
            let mut collected = Vec::new();
            while let Some(entry) = entries.next_entry().await.map_err(ErrorEnvelope::from)? {
                collected.push(entry);
            }
            collected.sort_by_key(tokio::fs::DirEntry::file_name);

            for entry in collected {
                let path = entry.path();
                let relative = self.relative_path_for(&path);
                if let Some(ref relative) = relative
                    && is_ignored(ignore_patterns, relative)
                {
                    continue;
                }

                let metadata = entry.metadata().await.map_err(ErrorEnvelope::from)?;
                if metadata.is_dir() {
                    pending.push_back(path);
                } else if metadata.is_file() {
                    let Some(relative) = relative else {
                        continue;
                    };
                    let hash = hash_file(&path).await?;
                    file_hashes.insert(relative.into_boxed_str(), hash.into_boxed_str());
                }
            }
        }
        Ok(())
    }

    fn relative_path_for(&self, path: &Path) -> Option<String> {
        let stripped = path.strip_prefix(&self.codebase_root).ok()?;
        let raw = stripped.to_string_lossy();
        let normalized = raw.replace('\\', "/");
        let normalized = normalized.trim_start_matches("./");
        let normalized = normalized.trim_start_matches('/');
        if normalized.is_empty() {
            None
        } else {
            Some(normalized.to_owned())
        }
    }

    fn build_merkle_dag(file_hashes: &FileHashMap) -> MerkleDag {
        let mut dag = MerkleDag::new();
        let mut values = String::new();
        for hash in file_hashes.values() {
            values.push_str(hash.as_ref());
        }
        let root_data = format!("root:{values}");
        let root_id = dag.add_node(&root_data, None);

        for (path, hash) in file_hashes {
            let file_data = format!("{path}:{hash}");
            dag.add_node(&file_data, Some(&root_id));
        }
        dag
    }

    fn diff_file_hashes(old: &FileHashMap, new: &FileHashMap) -> FileChangeSet {
        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut modified = Vec::new();

        for (path, hash) in new {
            match old.get(path) {
                None => added.push(path.clone()),
                Some(previous) => {
                    if previous != hash {
                        modified.push(path.clone());
                    }
                },
            }
        }

        for path in old.keys() {
            if !new.contains_key(path) {
                removed.push(path.clone());
            }
        }

        FileChangeSet {
            added,
            removed,
            modified,
        }
    }
}

impl FileSyncPort for LocalFileSync {
    fn initialize(
        &self,
        ctx: &RequestContext,
        options: FileSyncInitOptions,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let sync = self.clone();
        Box::pin(async move {
            ctx.ensure_not_cancelled("file_sync.initialize")?;
            sync.ensure_root_matches(&options.codebase_root)?;

            let ignore_patterns = Self::merged_ignore_patterns(options.ignore_patterns);
            let snapshot = if sync.resolve_snapshot_mode() {
                sync.load_snapshot().await?
            } else {
                None
            };

            let (file_hashes, merkle_dag) = if let Some(snapshot) = snapshot {
                snapshot.into_state()?
            } else {
                (FileHashMap::new(), MerkleDag::new())
            };

            let mut state = sync.state.write().await;
            state.ignore_patterns = ignore_patterns;
            state.file_hashes = file_hashes;
            state.merkle_dag = merkle_dag;
            drop(state);
            Ok(())
        })
    }

    fn check_for_changes(
        &self,
        ctx: &RequestContext,
        _options: FileSyncOptions,
    ) -> semantic_code_ports::BoxFuture<'_, Result<FileChangeSet>> {
        let ctx = ctx.clone();
        let sync = self.clone();
        Box::pin(async move {
            ctx.ensure_not_cancelled("file_sync.check_for_changes")?;
            let (previous_hashes, previous_dag, ignore_patterns) = {
                let state = sync.state.read().await;
                (
                    state.file_hashes.clone(),
                    state.merkle_dag.clone(),
                    state.ignore_patterns.clone(),
                )
            };

            let new_hashes = sync.generate_file_hashes(&ignore_patterns).await?;
            let new_dag = Self::build_merkle_dag(&new_hashes);

            let dag_changes = MerkleDag::compare(&previous_dag, &new_dag);
            if dag_changes.is_empty() {
                return Ok(FileChangeSet::default());
            }

            let changes = Self::diff_file_hashes(&previous_hashes, &new_hashes);

            let snapshot = if sync.resolve_snapshot_mode() {
                Some(SyncSnapshot::from_state(
                    SNAPSHOT_VERSION,
                    &new_hashes,
                    &new_dag,
                ))
            } else {
                None
            };

            let mut state = sync.state.write().await;
            state.file_hashes = new_hashes;
            state.merkle_dag = new_dag;
            drop(state);

            if let Some(snapshot) = snapshot {
                sync.write_snapshot(&snapshot).await?;
            }

            Ok(changes)
        })
    }

    fn delete_snapshot(
        &self,
        ctx: &RequestContext,
        codebase_root: PathBuf,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let sync = self.clone();
        Box::pin(async move {
            ctx.ensure_not_cancelled("file_sync.delete_snapshot")?;
            sync.ensure_root_matches(&codebase_root)?;

            let Some(path) = sync.snapshot_path() else {
                return Ok(());
            };
            match tokio::fs::remove_file(&path).await {
                Ok(()) => Ok(()),
                Err(error) => {
                    if error.kind() == std::io::ErrorKind::NotFound {
                        Ok(())
                    } else {
                        Err(ErrorEnvelope::from(error))
                    }
                },
            }
        })
    }
}

type FileHashMap = BTreeMap<Box<str>, Box<str>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SyncSnapshot {
    #[serde(default)]
    version: u32,
    #[serde(rename = "fileHashes")]
    file_hashes: Vec<(Box<str>, Box<str>)>,
    #[serde(rename = "merkleDAG")]
    merkle_dag: MerkleDagSerialized,
}

impl SyncSnapshot {
    fn from_state(version: u32, file_hashes: &FileHashMap, merkle_dag: &MerkleDag) -> Self {
        Self {
            version,
            file_hashes: file_hashes
                .iter()
                .map(|(path, hash)| (path.clone(), hash.clone()))
                .collect(),
            merkle_dag: merkle_dag.serialize(),
        }
    }

    fn into_state(self) -> Result<(FileHashMap, MerkleDag)> {
        if self.version != 0 && self.version != SNAPSHOT_VERSION {
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("sync", "snapshot_version_mismatch"),
                "snapshot version mismatch",
            )
            .with_metadata("found", self.version.to_string())
            .with_metadata("expected", SNAPSHOT_VERSION.to_string()));
        }

        let mut file_hashes = BTreeMap::new();
        for (path, hash) in self.file_hashes {
            file_hashes.insert(path, hash);
        }
        let merkle_dag = MerkleDag::deserialize(self.merkle_dag);
        Ok((file_hashes, merkle_dag))
    }
}

async fn hash_file(path: &Path) -> Result<String> {
    let bytes = tokio::fs::read(path).await.map_err(ErrorEnvelope::from)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn snapshot_error(
    code: &'static str,
    message: &str,
    error: impl std::fmt::Display,
) -> ErrorEnvelope {
    ErrorEnvelope::unexpected(
        ErrorCode::new("sync", code),
        format!("{message}: {error}"),
        ErrorClass::NonRetriable,
    )
}

fn normalize_root_path(path: &Path) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}

fn is_ignored(ignore_patterns: &[Box<str>], relative_path: &str) -> bool {
    if ignore_patterns.is_empty() {
        return false;
    }
    let matcher = IgnoreMatcher::new();
    matcher.is_ignored(&IgnoreMatchInput {
        ignore_patterns: ignore_patterns.to_vec(),
        relative_path: relative_path.to_owned().into_boxed_str(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_config::SnapshotStorageMode;
    use semantic_code_shared::RequestContext;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nanos}"))
    }

    #[tokio::test]
    async fn snapshot_roundtrip_is_stable() -> Result<()> {
        let root = temp_dir("sync-snap");
        tokio::fs::create_dir_all(&root)
            .await
            .map_err(ErrorEnvelope::from)?;
        tokio::fs::write(root.join("a.txt"), "hello")
            .await
            .map_err(ErrorEnvelope::from)?;

        let sync = LocalFileSync::new(root.clone(), SnapshotStorageMode::Disabled);
        sync.initialize(
            &RequestContext::new_request(),
            FileSyncInitOptions {
                codebase_root: root.clone(),
                ignore_patterns: None,
            },
        )
        .await?;

        let hashes = sync.generate_file_hashes(&[]).await?;
        let dag = LocalFileSync::build_merkle_dag(&hashes);
        let snapshot = SyncSnapshot::from_state(SNAPSHOT_VERSION, &hashes, &dag);
        let decoded = snapshot.clone().into_state()?;
        assert_eq!(decoded.0, hashes);
        assert_eq!(decoded.1.serialize(), dag.serialize());
        Ok(())
    }
}
