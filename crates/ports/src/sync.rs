//! Sync / change detection boundary contract.

use crate::BoxFuture;
use semantic_code_shared::{RequestContext, Result};
use std::path::PathBuf;

/// Set of detected file changes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FileChangeSet {
    /// Added files (relative paths).
    pub added: Vec<Box<str>>,
    /// Removed files (relative paths).
    pub removed: Vec<Box<str>>,
    /// Modified files (relative paths).
    pub modified: Vec<Box<str>>,
}

/// Options for initial snapshot creation.
#[derive(Debug, Clone)]
pub struct FileSyncInitOptions {
    /// Absolute codebase root.
    pub codebase_root: PathBuf,
    /// Ignore patterns applied before file access.
    pub ignore_patterns: Option<Vec<Box<str>>>,
}

/// Options for sync operations.
#[derive(Debug, Clone, Default)]
pub struct FileSyncOptions {}

/// Boundary contract for change detection (snapshot + diff).
pub trait FileSyncPort: Send + Sync {
    /// Initialize state for a codebase root (e.g. create an initial snapshot).
    fn initialize(
        &self,
        ctx: &RequestContext,
        options: FileSyncInitOptions,
    ) -> BoxFuture<'_, Result<()>>;

    /// Check for changes since the last snapshot.
    fn check_for_changes(
        &self,
        ctx: &RequestContext,
        options: FileSyncOptions,
    ) -> BoxFuture<'_, Result<FileChangeSet>>;

    /// Delete any persisted snapshot state for the codebase root.
    fn delete_snapshot(
        &self,
        ctx: &RequestContext,
        codebase_root: PathBuf,
    ) -> BoxFuture<'_, Result<()>>;
}
