//! Integration tests for read-only mmap wrappers.

use semantic_code_vector::MmapBytes;
use std::path::PathBuf;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

#[test]
fn map_snapshot_bytes_fixture() -> TestResult {
    let path = fixture_path("snapshot.bytes");
    let expected = std::fs::read(&path)?;
    let expected_len = u64::try_from(expected.len())?;

    let mapped = MmapBytes::open_readonly(&path, expected_len)?;
    assert_eq!(mapped.len(), expected.len());
    assert_eq!(mapped.as_slice(), expected.as_slice());
    assert_eq!(mapped.slice_at(0, 3)?, b"SCA");

    Ok(())
}
