//! Safe read-only memory mapping for vector snapshot bytes.
//!
//! This module keeps mmap-specific `unsafe` isolated behind a small API.

use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Result type for mmap operations.
pub type MmapResult<T> = Result<T, MmapError>;

/// Typed mmap errors.
#[derive(Debug, Error)]
pub enum MmapError {
    /// File open failed.
    #[error("failed to open file for read-only mapping: {path}")]
    Open {
        /// Path that failed to open.
        path: PathBuf,
        /// Underlying IO error.
        #[source]
        source: io::Error,
    },
    /// Metadata read failed.
    #[error("failed to read file metadata: {path}")]
    Metadata {
        /// Path whose metadata read failed.
        path: PathBuf,
        /// Underlying IO error.
        #[source]
        source: io::Error,
    },
    /// File length did not match the expected value.
    #[error("file size mismatch for {path}: expected {expected} bytes, found {found} bytes")]
    SizeMismatch {
        /// Path that mismatched.
        path: PathBuf,
        /// Expected byte length.
        expected: u64,
        /// Actual byte length.
        found: u64,
    },
    /// File length could not be represented as `usize`.
    #[error("file size overflow for {path}: {length} bytes does not fit in usize")]
    LengthOverflow {
        /// Path that overflowed.
        path: PathBuf,
        /// File length in bytes.
        length: u64,
    },
    /// Mapping call failed.
    #[error("failed to create read-only memory map: {path}")]
    Map {
        /// Path whose mmap creation failed.
        path: PathBuf,
        /// Underlying IO error.
        #[source]
        source: io::Error,
    },
    /// Mapped length changed after pre-map validation.
    #[error(
        "mapped length mismatch for {path}: validated {validated} bytes, mapped {mapped} bytes"
    )]
    MappedLengthMismatch {
        /// Path with mismatch.
        path: PathBuf,
        /// Length read from metadata before mapping.
        validated: usize,
        /// Length visible in mapped view.
        mapped: usize,
    },
    /// `offset + len` overflowed `usize`.
    #[error("slice range overflow: offset={offset}, len={len}")]
    SliceRangeOverflow {
        /// Start offset.
        offset: usize,
        /// Requested length.
        len: usize,
    },
    /// Requested slice was outside mapped range.
    #[error("slice out of bounds: offset={offset}, len={len}, total={total}")]
    SliceOutOfBounds {
        /// Start offset.
        offset: usize,
        /// Requested length.
        len: usize,
        /// Total mapped bytes.
        total: usize,
    },
}

/// Open a file in read-only mode with typed errors.
pub fn open_readonly(path: impl AsRef<Path>) -> MmapResult<File> {
    let path = path.as_ref();
    File::open(path).map_err(|source| MmapError::Open {
        path: path.to_path_buf(),
        source,
    })
}

/// Read-only mmap byte view with RAII lifetime management.
///
/// Dropping this value unmaps bytes automatically via `memmap2::Mmap` drop.
#[derive(Debug)]
pub struct MmapBytes {
    mmap: Mmap,
}

impl MmapBytes {
    /// Open and validate a read-only mmap file view.
    ///
    /// The file length is validated against `expected_len` before mapping.
    pub fn open_readonly(path: impl AsRef<Path>, expected_len: u64) -> MmapResult<Self> {
        let path = path.as_ref().to_path_buf();
        let file = open_readonly(&path)?;
        Self::from_file(path, &file, expected_len)
    }

    fn from_file(path: PathBuf, file: &File, expected_len: u64) -> MmapResult<Self> {
        let metadata = file.metadata().map_err(|source| MmapError::Metadata {
            path: path.clone(),
            source,
        })?;
        let found_len = metadata.len();
        if found_len != expected_len {
            return Err(MmapError::SizeMismatch {
                path,
                expected: expected_len,
                found: found_len,
            });
        }

        let validated_len = usize::try_from(found_len).map_err(|_| MmapError::LengthOverflow {
            path: path.clone(),
            length: found_len,
        })?;
        let mmap = map_file_readonly(file, &path)?;
        if mmap.len() != validated_len {
            return Err(MmapError::MappedLengthMismatch {
                path,
                validated: validated_len,
                mapped: mmap.len(),
            });
        }
        Ok(Self { mmap })
    }

    /// Borrow the full mapped byte range.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        self.mmap.as_ref()
    }

    /// Return mapped length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Return true when the mapped view has no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Borrow a checked slice from the mapped bytes.
    pub fn slice_at(&self, offset: usize, len: usize) -> MmapResult<&[u8]> {
        let total = self.len();
        let end = offset
            .checked_add(len)
            .ok_or(MmapError::SliceRangeOverflow { offset, len })?;
        self.as_slice()
            .get(offset..end)
            .ok_or(MmapError::SliceOutOfBounds { offset, len, total })
    }
}

#[expect(
    unsafe_code,
    reason = "mmap creation requires an unsafe OS call; preconditions are validated first"
)]
fn map_file_readonly(file: &File, path: &Path) -> MmapResult<Mmap> {
    // SAFETY:
    // - The file descriptor comes from `File::open` in read-only mode.
    // - We map read-only bytes, so there is no mutable aliasing through this API.
    // - The file length is validated before this call and rechecked after mapping.
    // - The resulting `Mmap` is owned by `MmapBytes` and unmapped on drop (RAII).
    unsafe { MmapOptions::new().map(file) }.map_err(|source| MmapError::Map {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::{MmapBytes, MmapError, open_readonly};
    use std::io;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    struct TempFile {
        path: PathBuf,
    }

    impl TempFile {
        fn write(prefix: &str, bytes: &[u8]) -> io::Result<Self> {
            let path = unique_temp_path(prefix);
            std::fs::write(&path, bytes)?;
            Ok(Self { path })
        }

        fn path(&self) -> &Path {
            self.path.as_path()
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn unique_temp_path(prefix: &str) -> PathBuf {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);

        let pid = std::process::id();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let seq = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("{prefix}-{pid}-{nanos}-{seq}.bin"))
    }

    #[test]
    fn map_temp_file_and_read_bytes() -> TestResult {
        let bytes = b"hello-mmap-wrapper";
        let temp = TempFile::write("vector-mmap-read", bytes)?;
        let expected_len = u64::try_from(bytes.len())?;

        let mapped = MmapBytes::open_readonly(temp.path(), expected_len)?;
        assert_eq!(mapped.len(), bytes.len());
        assert!(!mapped.is_empty());
        assert_eq!(mapped.as_slice(), bytes);
        assert_eq!(mapped.slice_at(0, 5)?, b"hello");
        assert_eq!(mapped.slice_at(6, 4)?, b"mmap");
        Ok(())
    }

    #[test]
    fn size_mismatch_returns_error() -> TestResult {
        let bytes = b"abcd";
        let temp = TempFile::write("vector-mmap-size", bytes)?;
        let expected = u64::try_from(bytes.len())? + 1;

        let error = MmapBytes::open_readonly(temp.path(), expected)
            .err()
            .ok_or_else(|| {
                io::Error::other("expected MmapBytes::open_readonly to return size mismatch")
            })?;
        assert!(matches!(
            error,
            MmapError::SizeMismatch {
                expected: 5,
                found: 4,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn open_readonly_missing_file_returns_typed_error() -> TestResult {
        let path = unique_temp_path("vector-mmap-missing");
        let error = open_readonly(&path)
            .err()
            .ok_or_else(|| io::Error::other("expected open_readonly to return an error"))?;
        assert!(matches!(error, MmapError::Open { .. }));
        Ok(())
    }
}
