//! # semantic-code-vector
//!
//! Vector indexing kernel and related APIs.
//! This crate depends only on `shared`.

use hnsw_rs::prelude::{DistCosine, Hnsw, Neighbour};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

const VECTOR_SNAPSHOT_VERSION: u32 = 1;

/// Configuration for the HNSW index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HnswParams {
    /// Maximum number of connections per node.
    pub max_nb_connection: usize,
    /// Maximum graph layer count.
    pub max_layer: usize,
    /// Construction search width.
    pub ef_construction: usize,
    /// Search width.
    pub ef_search: usize,
    /// Expected number of elements (allocation hint).
    pub max_elements: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            max_nb_connection: 16,
            max_layer: 16,
            ef_construction: 200,
            ef_search: 50,
            max_elements: 100_000,
        }
    }
}

/// Record stored inside the vector kernel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorRecord {
    /// Stable external identifier for this vector.
    pub id: Box<str>,
    /// Dense vector payload.
    pub vector: Vec<f32>,
}

/// Serialized snapshot for local persistence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorSnapshot {
    /// Snapshot schema version.
    pub version: u32,
    /// Vector dimensionality.
    pub dimension: u32,
    /// HNSW parameters.
    pub params: HnswParams,
    /// Stored vector records.
    pub records: Vec<VectorRecord>,
}

/// Search match with similarity score.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorMatch {
    /// External identifier for this vector.
    pub id: Box<str>,
    /// Similarity score in [0, 1].
    pub score: f32,
}

/// Fixed-dimension vector wrapper.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedVector<const D: usize>(Vec<f32>);

impl<const D: usize> FixedVector<D> {
    /// Validate and build a fixed-size vector.
    pub fn new(values: Vec<f32>) -> Result<Self> {
        if values.len() != D {
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("vector", "invalid_dimension"),
                "vector dimension mismatch",
            )
            .with_metadata("expected", D.to_string())
            .with_metadata("found", values.len().to_string()));
        }
        Ok(Self(values))
    }

    /// Borrow the vector as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Consume and return the raw vector.
    #[must_use]
    pub fn into_inner(self) -> Vec<f32> {
        self.0
    }
}

/// Record stored inside a fixed-dimension index.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorRecordFixed<const D: usize> {
    /// Stable external identifier for this vector.
    pub id: Box<str>,
    /// Dense vector payload.
    pub vector: FixedVector<D>,
}

/// Fixed-dimension wrapper around `VectorIndex`.
pub struct VectorIndexFixed<const D: usize> {
    inner: VectorIndex,
}

impl<const D: usize> VectorIndexFixed<D> {
    /// Create a new fixed-dimension vector index.
    pub fn new(params: HnswParams) -> Result<Self> {
        let dimension = u32::try_from(D).map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "invalid_dimension"),
                "dimension conversion overflow",
                ErrorClass::NonRetriable,
            )
        })?;
        Ok(Self {
            inner: VectorIndex::new(dimension, params)?,
        })
    }

    /// Insert or update records in the index.
    pub fn insert(&mut self, records: Vec<VectorRecordFixed<D>>) -> Result<()> {
        let records = records
            .into_iter()
            .map(|record| VectorRecord {
                id: record.id,
                vector: record.vector.into_inner(),
            })
            .collect();
        self.inner.insert(records)
    }

    /// Search for nearest neighbours and return sorted matches.
    pub fn search(&self, query: &FixedVector<D>, limit: usize) -> Result<Vec<VectorMatch>> {
        self.inner.search(query.as_slice(), limit)
    }

    /// Return the record for a given id.
    #[must_use]
    pub fn record_for_id(&self, id: &str) -> Option<&VectorRecord> {
        self.inner.record_for_id(id)
    }

    /// Export the index into a snapshot.
    #[must_use]
    pub fn snapshot(&self) -> VectorSnapshot {
        self.inner.snapshot()
    }
}

/// In-memory vector index backed by HNSW.
pub struct VectorIndex {
    dimension: u32,
    params: HnswParams,
    hnsw: Hnsw<'static, f32, DistCosine>,
    records: Vec<VectorRecord>,
    id_to_index: HashMap<Box<str>, usize>,
    deleted: HashSet<usize>,
}

impl VectorIndex {
    /// Create a new vector index for the given dimension.
    pub fn new(dimension: u32, params: HnswParams) -> Result<Self> {
        if dimension == 0 {
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("vector", "invalid_dimension"),
                "dimension must be greater than zero",
            ));
        }
        let max_elements = params.max_elements.max(1);
        let hnsw = Hnsw::new(
            params.max_nb_connection,
            max_elements,
            params.max_layer,
            params.ef_construction,
            DistCosine,
        );
        Ok(Self {
            dimension,
            params,
            hnsw,
            records: Vec::new(),
            id_to_index: HashMap::new(),
            deleted: HashSet::new(),
        })
    }

    /// Return the vector dimension.
    #[must_use]
    pub const fn dimension(&self) -> u32 {
        self.dimension
    }

    /// Insert or update records in the index.
    pub fn insert(&mut self, records: Vec<VectorRecord>) -> Result<()> {
        for record in records {
            ensure_dimension(self.dimension, &record.vector)?;

            let index = self.records.len();
            if let Some(previous) = self.id_to_index.insert(record.id.clone(), index) {
                self.deleted.insert(previous);
            }

            self.hnsw.insert((record.vector.as_slice(), index));
            self.records.push(record);
        }
        Ok(())
    }

    /// Delete records by external id (best-effort).
    pub fn delete(&mut self, ids: &[Box<str>]) -> Result<()> {
        for id in ids {
            if let Some(index) = self.id_to_index.remove(id.as_ref()) {
                self.deleted.insert(index);
            }
        }
        Ok(())
    }

    /// Search for nearest neighbours and return sorted matches.
    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<VectorMatch>> {
        if self.records.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        ensure_dimension(self.dimension, query)?;

        let total = self.records.len();
        let requested = limit.min(total);
        let knbn = (requested.saturating_mul(5)).max(requested).min(total);
        let ef_search = self.params.ef_search.max(knbn);

        let neighbours = self.hnsw.search(query, knbn, ef_search);
        let mut matches = to_matches(&self.records, &self.deleted, neighbours);

        matches.sort_by(|a, b| {
            let score = b.score.total_cmp(&a.score);
            if score != std::cmp::Ordering::Equal {
                return score;
            }
            a.id.cmp(&b.id)
        });
        matches.truncate(requested);
        Ok(matches)
    }

    /// Return the record for a given id.
    #[must_use]
    pub fn record_for_id(&self, id: &str) -> Option<&VectorRecord> {
        self.id_to_index
            .get(id)
            .and_then(|index| self.records.get(*index))
    }

    /// Export the index into a snapshot.
    #[must_use]
    pub fn snapshot(&self) -> VectorSnapshot {
        let mut ordered: BTreeMap<&str, &VectorRecord> = BTreeMap::new();
        for (id, index) in &self.id_to_index {
            if let Some(record) = self.records.get(*index) {
                ordered.insert(id.as_ref(), record);
            }
        }

        let records = ordered
            .into_values()
            .cloned()
            .collect::<Vec<VectorRecord>>();

        VectorSnapshot {
            version: VECTOR_SNAPSHOT_VERSION,
            dimension: self.dimension,
            params: self.params,
            records,
        }
    }

    /// Restore a vector index from a snapshot.
    pub fn from_snapshot(snapshot: VectorSnapshot) -> Result<Self> {
        if snapshot.version != VECTOR_SNAPSHOT_VERSION {
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("vector", "snapshot_version_mismatch"),
                "snapshot version mismatch",
            )
            .with_metadata("found", snapshot.version.to_string())
            .with_metadata("expected", VECTOR_SNAPSHOT_VERSION.to_string()));
        }

        let mut params = snapshot.params;
        params.max_elements = params.max_elements.max(snapshot.records.len().max(1));

        let mut index = Self::new(snapshot.dimension, params)?;
        index.insert(snapshot.records)?;
        Ok(index)
    }
}

fn ensure_dimension(dimension: u32, vector: &[f32]) -> Result<()> {
    let dimension = usize::try_from(dimension).map_err(|_| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "invalid_dimension"),
            "dimension conversion overflow",
            ErrorClass::NonRetriable,
        )
    })?;
    if vector.len() != dimension {
        return Err(ErrorEnvelope::expected(
            ErrorCode::new("vector", "invalid_dimension"),
            "vector dimension mismatch",
        )
        .with_metadata("expected", dimension.to_string())
        .with_metadata("found", vector.len().to_string()));
    }
    Ok(())
}

fn to_matches(
    records: &[VectorRecord],
    deleted: &HashSet<usize>,
    neighbours: Vec<Neighbour>,
) -> Vec<VectorMatch> {
    neighbours
        .into_iter()
        .filter_map(|neighbour| {
            let index = neighbour.d_id;
            if deleted.contains(&index) {
                return None;
            }
            let record = records.get(index)?;
            let score = (1.0 - neighbour.distance).max(0.0);
            Some(VectorMatch {
                id: record.id.clone(),
                score,
            })
        })
        .collect()
}

/// Returns the vector crate version.
#[must_use]
pub const fn vector_crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(feature = "experimental")]
/// Experimental extension hooks for local vector kernels.
pub mod experimental {
    /// Placeholder trait for experimental extensions.
    pub trait VectorKernelExtension {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_shared::shared_crate_version;

    #[test]
    fn vector_crate_compiles() {
        let version = vector_crate_version();
        assert!(!version.is_empty());
    }

    #[test]
    fn vector_can_use_shared() {
        let shared_version = shared_crate_version();
        assert!(!shared_version.is_empty());
    }

    #[test]
    fn snapshot_roundtrip_restores_index() -> Result<()> {
        let mut index = VectorIndex::new(2, HnswParams::default())?;
        index.insert(vec![VectorRecord {
            id: "a".into(),
            vector: vec![0.5, 0.5],
        }])?;

        let snapshot = index.snapshot();
        let restored = VectorIndex::from_snapshot(snapshot)?;
        let matches = restored.search(&[0.5, 0.5], 1)?;

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, "a".into());
        Ok(())
    }

    #[test]
    fn search_prefers_closer_vectors() -> Result<()> {
        let mut index = VectorIndex::new(2, HnswParams::default())?;
        index.insert(vec![
            VectorRecord {
                id: "near".into(),
                vector: vec![0.1, 0.1],
            },
            VectorRecord {
                id: "far".into(),
                vector: vec![0.9, 0.9],
            },
        ])?;

        let matches = index.search(&[0.1, 0.1], 2)?;
        assert_eq!(matches.first().map(|m| m.id.as_ref()), Some("near"));
        Ok(())
    }

    #[test]
    fn invalid_dimension_rejected() {
        let result = VectorIndex::new(0, HnswParams::default());
        assert!(result.is_err());
    }

    #[test]
    #[cfg(not(feature = "experimental"))]
    fn experimental_feature_disabled_by_default() {
        assert!(
            !cfg!(feature = "experimental"),
            "experimental feature should be disabled by default"
        );
    }

    #[test]
    fn fixed_dimension_index_accepts_only_matching_vectors() -> Result<()> {
        let mut index = VectorIndexFixed::<2>::new(HnswParams::default())?;
        let record = VectorRecordFixed {
            id: "a".into(),
            vector: FixedVector::new(vec![0.5, 0.5])?,
        };
        index.insert(vec![record])?;
        let query = FixedVector::new(vec![0.5, 0.5])?;
        let matches = index.search(&query, 1)?;
        assert_eq!(matches.len(), 1);
        Ok(())
    }
}
