//! Bridge tests validating Phase 01 outputs for Phase 02.

use semantic_code_vector::{
    MmapBytes, Quantizer, SNAPSHOT_V2_META_FILE_NAME, SNAPSHOT_V2_VECTORS_FILE_NAME, read_metadata,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

type TestResult = Result<(), Box<dyn std::error::Error>>;

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn write(prefix: &str, bytes: &[u8]) -> std::io::Result<Self> {
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

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

#[test]
fn phase02_bt01_quantization_exports_are_usable() -> TestResult {
    let dataset = vec![
        vec![0.0, 1.0, 2.0, 3.0],
        vec![3.0, 2.0, 1.0, 0.0],
        vec![1.5, 2.5, 3.5, 4.5],
    ];
    let quantizer = Quantizer::from_dataset(&dataset)?;
    let quantized = quantizer.quantize_batch(&dataset)?;
    let params = quantizer.params();

    let dimension = params.dimension();
    assert_eq!(quantized.len(), dataset.len() * dimension);
    let view = quantized.chunks_exact(dimension);
    assert!(view.remainder().is_empty());
    assert_eq!(view.len(), dataset.len());
    assert_eq!(params.dimension(), dataset[0].len());

    let restored = quantizer.dequantize(
        quantized
            .chunks_exact(dimension)
            .next()
            .ok_or_else(|| std::io::Error::other("expected at least one quantized vector"))?,
    )?;
    assert_eq!(restored.len(), dataset[0].len());
    Ok(())
}

#[test]
fn phase02_bt02_mmap_wrapper_with_temp_file_fixture() -> TestResult {
    let bytes = b"phase02-bridge-mmap";
    let temp = TempFile::write("phase02-bridge-mmap", bytes)?;
    let expected_len = u64::try_from(bytes.len())?;
    let mapped = MmapBytes::open_readonly(temp.path(), expected_len)?;

    assert_eq!(mapped.len(), bytes.len());
    assert_eq!(mapped.as_slice(), bytes);
    assert_eq!(mapped.slice_at(0, 7)?, b"phase02");
    Ok(())
}

#[test]
fn phase02_bt03_snapshot_v2_format_reads_correctly() -> TestResult {
    let bundle = fixture_path("snapshot_v2_bundle");
    let meta = read_metadata(bundle.join(SNAPSHOT_V2_META_FILE_NAME))?;
    let expected_len = u64::from(meta.dimension)
        .checked_mul(meta.count)
        .ok_or_else(|| std::io::Error::other("snapshot vectors byte length overflow"))?;
    let mapped =
        MmapBytes::open_readonly(bundle.join(SNAPSHOT_V2_VECTORS_FILE_NAME), expected_len)?;
    let expected_len = usize::try_from(expected_len)?;

    assert_eq!(meta.version.as_u8(), 2);
    assert_eq!(mapped.len(), expected_len);
    assert_eq!(
        mapped.len() / usize::try_from(meta.dimension)?,
        usize::try_from(meta.count)?
    );
    Ok(())
}
