//! Snapshot storage configuration.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Persistence mode for local vector snapshots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum SnapshotStorageMode {
    /// Disable persistence (in-memory only).
    Disabled,
    /// Store under `.context/` inside the codebase root.
    #[default]
    Project,
    /// Store under a custom absolute path.
    Custom(PathBuf),
}

impl SnapshotStorageMode {
    /// Resolve the snapshot root directory for a codebase.
    #[must_use]
    pub fn resolve_root(&self, codebase_root: &Path) -> Option<PathBuf> {
        match self {
            Self::Disabled => None,
            Self::Project => Some(codebase_root.join(".context")),
            Self::Custom(path) => Some(path.clone()),
        }
    }
}

/// Snapshot format used by the local vector DB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum VectorSnapshotFormat {
    /// Legacy JSON snapshot (`<collection>.json`).
    V1,
    /// Binary companion snapshot (`snapshot.meta` + `vectors.u8.bin`).
    #[default]
    V2,
}

impl VectorSnapshotFormat {
    /// Canonical string representation used in config/env surfaces.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::V1 => "v1",
            Self::V2 => "v2",
        }
    }
}
