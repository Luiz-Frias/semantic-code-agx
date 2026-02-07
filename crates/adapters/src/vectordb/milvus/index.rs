//! Milvus index configuration shared by gRPC and REST adapters.

use std::collections::BTreeMap;

/// Index configuration for dense + sparse vector fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MilvusIndexConfig {
    /// Dense vector index spec.
    pub dense: MilvusIndexSpec,
    /// Sparse vector index spec.
    pub sparse: MilvusIndexSpec,
}

impl Default for MilvusIndexConfig {
    fn default() -> Self {
        Self {
            dense: MilvusIndexSpec::dense_default(),
            sparse: MilvusIndexSpec::sparse_default(),
        }
    }
}

/// Single-field Milvus index spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MilvusIndexSpec {
    /// Index type (e.g. `AUTOINDEX`, `HNSW`).
    pub index_type: Box<str>,
    /// Metric type (e.g. `COSINE`, `IP`, `BM25`).
    pub metric_type: Box<str>,
    /// Index build parameters.
    pub params: BTreeMap<Box<str>, Box<str>>,
}

impl MilvusIndexSpec {
    fn dense_default() -> Self {
        Self {
            index_type: "AUTOINDEX".into(),
            metric_type: "COSINE".into(),
            params: BTreeMap::new(),
        }
    }

    fn sparse_default() -> Self {
        Self {
            index_type: "SPARSE_INVERTED_INDEX".into(),
            metric_type: "BM25".into(),
            params: BTreeMap::new(),
        }
    }
}
