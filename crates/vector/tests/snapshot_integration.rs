//! Integration tests for snapshot metadata and v2 fixture compatibility.

use semantic_code_vector::{
    HnswParams, MmapBytes, QuantizationParams, SNAPSHOT_V2_META_FILE_NAME,
    SNAPSHOT_V2_VECTORS_FILE_NAME, SnapshotError, SnapshotResult, VectorSnapshotMeta,
    VectorSnapshotVersion, read_metadata, write_metadata,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

type TestResult = Result<(), Box<dyn std::error::Error>>;

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

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
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

fn default_quantization(dimension: usize) -> SnapshotResult<QuantizationParams> {
    QuantizationParams::new(vec![1.0; dimension], vec![0.0; dimension])
        .map_err(|source| SnapshotError::Quantization { source })
}

#[test]
fn metadata_roundtrip_in_temp_dir() -> TestResult {
    let temp = TempDir::create("vector-snapshot-meta")?;
    let path = temp.path().join(SNAPSHOT_V2_META_FILE_NAME);

    let meta = VectorSnapshotMeta::new(8, 5, HnswParams::default(), default_quantization(8)?, 0)?;
    write_metadata(&path, &meta)?;
    let restored = read_metadata(&path)?;

    assert_eq!(restored, meta);
    assert_eq!(restored.version, VectorSnapshotVersion::V2);
    Ok(())
}

#[test]
fn metadata_fixture_decodes() -> SnapshotResult<()> {
    let path = fixture_path("snapshot_v2.meta");
    let meta = read_metadata(path)?;

    assert_eq!(meta.version, VectorSnapshotVersion::V2);
    assert_eq!(meta.dimension, 3);
    assert_eq!(meta.count, 2);
    assert_eq!(meta.quantization.dimension(), 3);
    Ok(())
}

#[test]
fn fixture_v2_bundle_loads_with_mmap() -> TestResult {
    let bundle = fixture_path("snapshot_v2_bundle");
    let meta = read_metadata(bundle.join(SNAPSHOT_V2_META_FILE_NAME))?;
    let expected_len = u64::from(meta.dimension)
        .checked_mul(meta.count)
        .ok_or_else(|| std::io::Error::other("snapshot vectors byte length overflow"))?;
    let vectors_path = bundle.join(SNAPSHOT_V2_VECTORS_FILE_NAME);
    let mapped = MmapBytes::open_readonly(&vectors_path, expected_len)?;
    let expected = std::fs::read(&vectors_path)?;

    assert_eq!(meta.version, VectorSnapshotVersion::V2);
    assert_eq!(mapped.as_slice(), expected.as_slice());
    Ok(())
}
