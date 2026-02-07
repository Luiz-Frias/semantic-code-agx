//! Manifest helpers for local CLI commands.

use crate::{InfraError, InfraResult};
use semantic_code_config::{BackendConfig, SnapshotStorageMode, to_pretty_toml};
use semantic_code_domain::{CollectionName, IndexMode};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope};
use serde::{Deserialize, Serialize};
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const CLI_MANIFEST_VERSION: u32 = 1;
const CONTEXT_DIR_NAME: &str = ".context";
const MANIFEST_FILE_NAME: &str = "manifest.json";
const CONFIG_FILE_NAME: &str = "config.toml";

/// Manifest persisted for local CLI operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CliManifest {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Codebase root path for this manifest.
    pub codebase_root: PathBuf,
    /// Collection name used for indexing.
    pub collection_name: CollectionName,
    /// Index mode used for collection naming.
    pub index_mode: IndexMode,
    /// Snapshot storage mode for local persistence.
    pub snapshot_storage: SnapshotStorageMode,
    /// Manifest creation timestamp (milliseconds since epoch).
    pub created_at_ms: u64,
    /// Manifest last updated timestamp (milliseconds since epoch).
    pub updated_at_ms: u64,
}

impl CliManifest {
    /// Build a new manifest for a codebase.
    pub fn new(
        codebase_root: &Path,
        collection_name: impl AsRef<str>,
        index_mode: IndexMode,
        snapshot_storage: SnapshotStorageMode,
    ) -> InfraResult<Self> {
        let now_ms = now_epoch_ms()?;
        let normalized_root =
            std::path::absolute(codebase_root).unwrap_or_else(|_| codebase_root.to_path_buf());
        let collection_name =
            CollectionName::parse(collection_name.as_ref()).map_err(ErrorEnvelope::from)?;
        Ok(Self {
            schema_version: CLI_MANIFEST_VERSION,
            codebase_root: normalized_root,
            collection_name,
            index_mode,
            snapshot_storage,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
        })
    }
}

/// Read the manifest from `.context/manifest.json`, if present.
pub fn read_manifest(root: &Path) -> InfraResult<Option<CliManifest>> {
    let path = manifest_path(root);
    match std::fs::read_to_string(&path) {
        Ok(contents) => {
            let manifest = serde_json::from_str(&contents).map_err(|error| {
                ErrorEnvelope::unexpected(
                    ErrorCode::invalid_input(),
                    format!("manifest parse failed: {error}"),
                    ErrorClass::NonRetriable,
                )
            })?;
            Ok(Some(manifest))
        },
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(InfraError::from(error)),
    }
}

/// Write the manifest to `.context/manifest.json`.
pub fn write_manifest(root: &Path, manifest: &CliManifest) -> InfraResult<()> {
    let context_dir = context_dir(root);
    std::fs::create_dir_all(&context_dir)?;
    let mut payload = serde_json::to_string_pretty(manifest).map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            format!("manifest serialize failed: {error}"),
            ErrorClass::NonRetriable,
        )
    })?;
    payload.push('\n');
    std::fs::write(context_dir.join(MANIFEST_FILE_NAME), payload)?;
    Ok(())
}

/// Write the default config to `.context/config.toml` if it does not exist.
pub fn ensure_default_config(root: &Path) -> InfraResult<()> {
    let path = config_path(root);
    if path.is_file() {
        return Ok(());
    }
    let context_dir = context_dir(root);
    std::fs::create_dir_all(&context_dir)?;
    let payload = to_pretty_toml(&BackendConfig::default())?;
    std::fs::write(path, payload)?;
    Ok(())
}

/// Update the manifest updated timestamp and persist it.
pub fn touch_manifest(root: &Path, manifest: &CliManifest) -> InfraResult<CliManifest> {
    let mut updated = manifest.clone();
    updated.updated_at_ms = now_epoch_ms()?;
    write_manifest(root, &updated)?;
    Ok(updated)
}

/// Append `.context/` to `.gitignore` if the file already exists.
pub fn append_context_gitignore(root: &Path) -> InfraResult<()> {
    let path = root.join(".gitignore");
    let contents = match std::fs::read_to_string(&path) {
        Ok(value) => value,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(InfraError::from(error)),
    };
    let has_context = contents.lines().any(|line| line.trim() == ".context/");
    if has_context {
        return Ok(());
    }
    let mut updated = contents;
    if !updated.ends_with('\n') {
        updated.push('\n');
    }
    updated.push_str(".context/\n");
    std::fs::write(path, updated)?;
    Ok(())
}

fn context_dir(root: &Path) -> PathBuf {
    root.join(CONTEXT_DIR_NAME)
}

/// Manifest path under `.context/`.
pub fn manifest_path(root: &Path) -> PathBuf {
    context_dir(root).join(MANIFEST_FILE_NAME)
}

/// Default config path under `.context/`.
pub fn config_path(root: &Path) -> PathBuf {
    context_dir(root).join(CONFIG_FILE_NAME)
}

fn now_epoch_ms() -> InfraResult<u64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                format!("clock error: {error}"),
                ErrorClass::NonRetriable,
            )
        })?;
    u64::try_from(duration.as_millis()).map_err(|_| {
        ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            "timestamp overflow",
            ErrorClass::NonRetriable,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::{
        CliManifest, append_context_gitignore, config_path, ensure_default_config, read_manifest,
        write_manifest,
    };
    use crate::InfraResult;
    use semantic_code_config::SnapshotStorageMode;
    use semantic_code_domain::IndexMode;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nanos}"))
    }

    #[test]
    fn manifest_roundtrip_and_gitignore_append() -> InfraResult<()> {
        let temp = temp_dir("cli-manifest");
        std::fs::create_dir_all(&temp)?;
        std::fs::write(temp.join(".gitignore"), "target/\n")?;

        let manifest = CliManifest::new(
            &temp,
            "code_chunks_local",
            IndexMode::Dense,
            SnapshotStorageMode::Project,
        )?;
        write_manifest(&temp, &manifest)?;
        append_context_gitignore(&temp)?;
        ensure_default_config(&temp)?;

        let loaded = read_manifest(&temp)?.expect("manifest");
        assert_eq!(loaded.collection_name.as_str(), "code_chunks_local");
        let gitignore = std::fs::read_to_string(temp.join(".gitignore"))?;
        assert!(gitignore.lines().any(|line| line.trim() == ".context/"));
        assert!(config_path(&temp).is_file());
        Ok(())
    }
}
