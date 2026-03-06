//! Bridge tests validating Phase 02 outputs for Phase 03.

use semantic_code_vector::{
    HnswParams, VectorIndex, VectorRecord, VectorSearchBackend, VectorSnapshotWriteVersion,
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

fn unique_temp_path(prefix: &str) -> PathBuf {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let seq = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}-{pid}-{nanos}-{seq}"))
}

fn make_vector(seed: u32, dimension: usize) -> Vec<f32> {
    let values = (0..dimension)
        .map(|offset| {
            let base = seed as f32;
            let axis = offset as f32;
            let sine = (base.mul_add(0.17, axis * 0.11)).sin() * 0.75;
            let cosine = (base.mul_add(0.07, axis * 0.19)).cos() * 0.25;
            sine + cosine + 0.05
        })
        .collect::<Vec<_>>();
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= 0.0 {
        return values;
    }
    values.into_iter().map(|value| value / norm).collect()
}

#[test]
fn phase03_bt01_vectorindex_v2_snapshot_roundtrip() -> TestResult {
    let temp = TempDir::create("phase03-bridge-v2-roundtrip")?;
    let mut index = VectorIndex::new(3, HnswParams::default())?;
    index.insert(vec![
        VectorRecord {
            id: "doc_a".into(),
            vector: vec![1.0, 0.0, 0.0],
        },
        VectorRecord {
            id: "doc_b".into(),
            vector: vec![0.8, 0.2, 0.0],
        },
        VectorRecord {
            id: "doc_c".into(),
            vector: vec![0.1, 0.2, 0.97],
        },
    ])?;

    let before = index.search(&[1.0, 0.0, 0.0], 3)?;
    index.snapshot_v2(temp.path())?;
    let restored = VectorIndex::from_snapshot_v2(temp.path())?;
    let after = restored.search(&[1.0, 0.0, 0.0], 3)?;

    let before_ids = before
        .matches
        .iter()
        .map(|item| item.id.to_string())
        .collect::<Vec<_>>();
    let after_ids = after
        .matches
        .iter()
        .map(|item| item.id.to_string())
        .collect::<Vec<_>>();
    assert_eq!(after_ids, before_ids);

    let stats = restored.snapshot_stats(VectorSnapshotWriteVersion::V2)?;
    assert_eq!(stats.version.as_str(), "v2");
    assert_eq!(stats.count, 3);
    Ok(())
}

#[test]
fn phase03_bt03_u8_search_investigation_results_available() -> TestResult {
    let dimension = 24usize;
    let mut index = VectorIndex::new(u32::try_from(dimension)?, HnswParams::default())?;
    let records = (0u32..128u32)
        .map(|seed| VectorRecord {
            id: format!("doc_{seed:04}").into_boxed_str(),
            vector: make_vector(seed, dimension),
        })
        .collect::<Vec<_>>();
    index.insert(records)?;
    let query = make_vector(7_777, dimension);

    let f32_results =
        index.search_with_backend(query.as_slice(), 5, VectorSearchBackend::F32Hnsw)?;
    let u8_results = index.search_with_backend(
        query.as_slice(),
        5,
        VectorSearchBackend::ExperimentalU8ThenF32Rerank,
    )?;

    // Verify enum-typed routing returns expected result counts.
    assert_eq!(f32_results.matches.len(), 5);
    assert_eq!(u8_results.matches.len(), 5);
    Ok(())
}
