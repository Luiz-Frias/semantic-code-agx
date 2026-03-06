//! Integration tests for `VectorIndex` v2 snapshot persistence.

use semantic_code_vector::{
    HnswParams, VectorIndex, VectorRecord, VectorSnapshot, VectorSnapshotV2LoadOptions,
    VectorSnapshotWriteVersion,
};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

type TestResult = Result<(), Box<dyn std::error::Error>>;
const SNAPSHOT_V1_FILE_NAME: &str = "snapshot.v1.json";

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SnapshotStatsFixture {
    version: String,
    dimension: u32,
    count: u64,
    min_bytes: u64,
    expected_metadata_keys: Vec<String>,
}

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

fn load_snapshot_stats_fixture(
    name: &str,
) -> Result<SnapshotStatsFixture, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(fixture_path(name))?;
    Ok(serde_json::from_slice(&bytes)?)
}

#[test]
fn vector_index_v2_fixture_reload_from_disk() -> TestResult {
    let bundle = fixture_path("vector_index_v2_bundle");
    let restored = VectorIndex::from_snapshot_v2(&bundle)?;
    let matches = restored.search(&[10.0, 20.0, 30.0], 1)?;

    assert_eq!(restored.dimension(), 3);
    assert_eq!(matches.matches.len(), 1);
    assert_eq!(matches.matches[0].id.as_ref(), "doc_a");
    Ok(())
}

#[test]
fn vector_index_loads_v1_via_auto_upgrade_path() -> TestResult {
    let temp = TempDir::create("vector-index-upgrade-v1-v2")?;

    let legacy = VectorSnapshot {
        version: 1,
        dimension: 3,
        params: HnswParams::default(),
        records: vec![
            VectorRecord {
                id: "doc_1".into(),
                vector: vec![0.0, 1.0, 2.0],
            },
            VectorRecord {
                id: "doc_2".into(),
                vector: vec![3.0, 4.0, 5.0],
            },
        ],
    };

    let payload = serde_json::to_vec(&legacy)?;
    std::fs::write(temp.path().join(SNAPSHOT_V1_FILE_NAME), payload)?;

    let restored = VectorIndex::from_snapshot_v2_with_options(
        temp.path(),
        VectorSnapshotV2LoadOptions {
            auto_upgrade_v1: true,
        },
    )?;

    assert!(temp.path().join("snapshot.meta").is_file());
    assert!(temp.path().join("vectors.u8.bin").is_file());

    let matches = restored.search(&[0.0, 1.0, 2.0], 1)?;
    assert_eq!(matches.matches.len(), 1);
    assert_eq!(matches.matches[0].id.as_ref(), "doc_1");
    Ok(())
}

#[test]
fn snapshot_stats_from_v2_snapshot_matches_fixture_shape() -> TestResult {
    let fixture = load_snapshot_stats_fixture("snapshot_stats_large.expected.json")?;
    let mut index = VectorIndex::new(fixture.dimension, HnswParams::default())?;
    let mut records = Vec::new();
    for i in 0..fixture.count {
        records.push(VectorRecord {
            id: format!("doc_{i:03}").into(),
            vector: vec![
                i as f32 + 1.0,
                (i as f32 + 2.0) * 0.5,
                (i as f32 + 3.0) * 0.25,
            ],
        });
    }
    index.insert(records)?;

    let stats = index.snapshot_stats(VectorSnapshotWriteVersion::V2)?;
    assert_eq!(fixture.version, stats.version.as_str());
    assert_eq!(fixture.dimension, stats.dimension);
    assert_eq!(fixture.count, stats.count);
    assert!(stats.bytes >= fixture.min_bytes);

    let keys = stats
        .metadata
        .keys()
        .map(|key| key.to_string())
        .collect::<Vec<_>>();
    assert_eq!(keys, fixture.expected_metadata_keys);
    Ok(())
}

#[test]
fn size_limit_enforcement_blocks_small_limit() -> TestResult {
    let temp = TempDir::create("vector-index-size-limit-enforcement")?;
    let mut index = VectorIndex::new(3, HnswParams::default())?;
    index.insert(vec![
        VectorRecord {
            id: "doc_a".into(),
            vector: vec![1.0, 2.0, 3.0],
        },
        VectorRecord {
            id: "doc_b".into(),
            vector: vec![4.0, 5.0, 6.0],
        },
    ])?;

    let error = index
        .write_snapshot_with_size_limit(temp.path(), VectorSnapshotWriteVersion::V2, Some(8))
        .err()
        .ok_or_else(|| std::io::Error::other("expected oversize snapshot error"))?;
    assert_eq!(
        error.code,
        semantic_code_shared::ErrorCode::new("vector", "snapshot_oversize")
    );
    Ok(())
}
