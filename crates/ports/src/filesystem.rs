//! Filesystem boundary contract.

use crate::BoxFuture;
use semantic_code_shared::{ErrorCode, ErrorEnvelope, RequestContext, Result};
use std::path::PathBuf;
use std::sync::Arc;

/// A validated, normalized path relative to a codebase root.
///
/// Implementations MUST reject absolute paths and traversal (e.g. `..` segments).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SafeRelativePath(Box<str>);

impl SafeRelativePath {
    /// Validate and normalize an untrusted relative path.
    pub fn new(input: &str) -> Result<Self> {
        let normalized = normalize_relative_path(input)?;
        Ok(Self(normalized.into_boxed_str()))
    }

    /// Borrow the path as a string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Boundary contract for safe path normalization.
pub trait PathPolicyPort: Send + Sync {
    /// Convert untrusted input into a normalized `SafeRelativePath` or reject it.
    fn to_safe_relative_path(&self, input: &str) -> Result<SafeRelativePath>;
}

/// Borrowing filesystem session bound to a codebase root.
pub struct FileSystemSession<'a> {
    fs: &'a dyn FileSystemPort,
    codebase_root: PathBuf,
}

impl<'a> FileSystemSession<'a> {
    /// Create a borrowing session for a codebase root.
    #[must_use]
    pub fn new(fs: &'a dyn FileSystemPort, codebase_root: PathBuf) -> Self {
        Self { fs, codebase_root }
    }

    /// Read and list directory entries relative to the codebase root.
    pub fn read_dir(
        &self,
        ctx: &RequestContext,
        dir: SafeRelativePath,
    ) -> BoxFuture<'_, Result<Vec<FileSystemDirEntry>>> {
        self.fs.read_dir(ctx, self.codebase_root.clone(), dir)
    }

    /// Read a UTF-8 text file relative to the codebase root.
    pub fn read_file_text(
        &self,
        ctx: &RequestContext,
        file: SafeRelativePath,
    ) -> BoxFuture<'_, Result<Box<str>>> {
        self.fs
            .read_file_text(ctx, self.codebase_root.clone(), file)
    }

    /// Read file metadata relative to the codebase root.
    pub fn stat(
        &self,
        ctx: &RequestContext,
        path: SafeRelativePath,
    ) -> BoxFuture<'_, Result<FileSystemStat>> {
        self.fs.stat(ctx, self.codebase_root.clone(), path)
    }
}

/// Extension helpers for filesystem ports.
pub trait FileSystemPortExt {
    /// Create a borrowing session for a codebase root.
    fn session(&self, codebase_root: PathBuf) -> FileSystemSession<'_>;
}

impl<T> FileSystemPortExt for T
where
    T: FileSystemPort,
{
    fn session(&self, codebase_root: PathBuf) -> FileSystemSession<'_> {
        FileSystemSession::new(self, codebase_root)
    }
}

impl FileSystemPortExt for Arc<dyn FileSystemPort> {
    fn session(&self, codebase_root: PathBuf) -> FileSystemSession<'_> {
        FileSystemSession::new(self.as_ref(), codebase_root)
    }
}

fn normalize_relative_path(input: &str) -> Result<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(".".to_owned());
    }
    let replaced = trimmed.replace('\\', "/");
    let collapsed = collapse_forward_slashes(&replaced);
    if is_absolute_like(&collapsed) {
        return Err(ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            "absolute paths are not allowed",
        ));
    }
    let collapsed = collapsed.trim_start_matches("./");
    let collapsed = collapsed.trim_matches('/');

    if collapsed.is_empty() {
        return Ok(".".to_owned());
    }

    if collapsed.split('/').any(|segment| segment == "..") {
        return Err(ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            "path traversal is not allowed",
        ));
    }

    Ok(collapsed.to_owned())
}

fn is_absolute_like(path: &str) -> bool {
    if path.starts_with('/') || path.starts_with("//") {
        return true;
    }
    let bytes = path.as_bytes();
    matches!(bytes, [drive, b':', b'/', ..] if drive.is_ascii_alphabetic())
}

fn collapse_forward_slashes(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut previous_was_slash = false;

    for ch in input.chars() {
        if ch == '/' {
            if previous_was_slash {
                continue;
            }
            previous_was_slash = true;
        } else {
            previous_was_slash = false;
        }
        output.push(ch);
    }

    output
}

/// File system entry kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileSystemEntryKind {
    /// Regular file.
    File,
    /// Directory.
    Directory,
    /// Other / unknown.
    Other,
}

/// A directory listing entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSystemDirEntry {
    /// Entry name (single path segment).
    pub name: Box<str>,
    /// Entry kind.
    pub kind: FileSystemEntryKind,
}

/// File system stat info.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSystemStat {
    /// Kind of the entry.
    pub kind: FileSystemEntryKind,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Modification time as milliseconds since epoch.
    pub mtime_ms: u64,
}

/// Boundary contract for filesystem access.
///
/// Note: `codebase_root` is an absolute path owned by the caller/infra composition.
pub trait FileSystemPort: Send + Sync {
    /// Read and list directory entries.
    fn read_dir(
        &self,
        ctx: &RequestContext,
        codebase_root: PathBuf,
        dir: SafeRelativePath,
    ) -> BoxFuture<'_, Result<Vec<FileSystemDirEntry>>>;

    /// Read a UTF-8 text file.
    fn read_file_text(
        &self,
        ctx: &RequestContext,
        codebase_root: PathBuf,
        file: SafeRelativePath,
    ) -> BoxFuture<'_, Result<Box<str>>>;

    /// Read file metadata (kind/size/mtime).
    fn stat(
        &self,
        ctx: &RequestContext,
        codebase_root: PathBuf,
        path: SafeRelativePath,
    ) -> BoxFuture<'_, Result<FileSystemStat>>;
}
