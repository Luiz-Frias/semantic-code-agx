//! # semantic-code-vector
//!
//! Vector indexing kernel and related APIs.
//! This crate depends only on `shared`.

use hnsw_rs::prelude::{AnnT, Distance, Hnsw, HnswIo, Neighbour};

/// Accelerate-backed BLAS distance primitives with scalar fallbacks.
mod accelerate;
use accelerate::DistAccelerateCosine;
use rayon::prelude::*;
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, Result};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io;
use std::path::Path;
use tracing::instrument;

/// Safe read-only mmap wrappers for binary snapshot workflows.
mod mmap;
/// Quantization primitives for SQ8 workflows.
pub(crate) mod quantization;
/// Snapshot v2 metadata format helpers.
pub(crate) mod snapshot;

use crate::quantization::{QuantizedSlice, decode_u8_to_f32, fit_min_max};
use crate::snapshot::{
    ReadSnapshotV2Options, SNAPSHOT_V1_FILE_NAME, compute_vectors_crc32, encode_metadata,
    read_snapshot_v2_with_options, write_snapshot_v2_with_kernel,
};

pub use mmap::MmapBytes;
pub use quantization::{QuantizationError, QuantizationParams, Quantizer, quantize_f32_to_u8};
pub use snapshot::{
    SNAPSHOT_V2_HNSW_GRAPH_BASENAME, SNAPSHOT_V2_META_FILE_NAME, SNAPSHOT_V2_VECTORS_FILE_NAME,
    SnapshotError, SnapshotResult, VectorSnapshotMeta, VectorSnapshotVersion, read_metadata,
    read_snapshot_ids, subset_snapshot_v2, tile_snapshot_v2, write_metadata, write_snapshot_ids,
};

const VECTOR_SNAPSHOT_VERSION: u32 = 1;
const VECTOR_SNAPSHOT_V2_IDS_FILE_NAME: &str = "ids.json";
const SNAPSHOT_STATS_CONTEXT: &str = "<snapshot-stats>";
const U8_RERANK_CANDIDATE_MULTIPLIER: usize = 4;
const U8_RERANK_MIN_CANDIDATES: usize = 32;
const U8_RERANK_MAX_CANDIDATES: usize = 512;
// hnsw_rs graph dumping requires the runtime max layer to match its fixed
// internal layer constant (NB_LAYER_MAX == 16 in hnsw_rs 0.3.x).
const HNSW_RS_GRAPH_DUMP_REQUIRED_MAX_LAYER: usize = 16;

/// Default minimum squared L2 norm for cosine-distance validation.
///
/// Vectors with `norm² < min_norm_squared` are rejected at insertion because
/// cosine distance is degenerate for near-zero vectors. The default (`1e-30`)
/// is intentionally permissive — it catches only truly-zero and numerically-zero
/// vectors while allowing very small but legitimate embedding outputs.
const DEFAULT_MIN_NORM_SQUARED: f32 = 1e-30;

/// Return the default minimum norm² threshold (for `serde(default)`).
const fn default_min_norm_squared() -> f32 {
    DEFAULT_MIN_NORM_SQUARED
}

/// Configuration for the HNSW index.
///
/// `PartialEq` and `Eq` are implemented manually because `min_norm_squared`
/// is `f32` (which lacks `Eq`). We use `f32::to_bits()` for bitwise equality,
/// which is sound because the field is validated to be finite and positive.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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
    /// Minimum squared L2 norm for inserted vectors.
    ///
    /// Cosine distance is degenerate for zero/near-zero vectors, producing
    /// non-deterministic HNSW search results. Vectors with `norm² < threshold`
    /// are rejected at insertion with a descriptive error.
    #[serde(default = "default_min_norm_squared")]
    pub min_norm_squared: f32,
}

impl PartialEq for HnswParams {
    fn eq(&self, other: &Self) -> bool {
        self.max_nb_connection == other.max_nb_connection
            && self.max_layer == other.max_layer
            && self.ef_construction == other.ef_construction
            && self.ef_search == other.ef_search
            && self.max_elements == other.max_elements
            && self.min_norm_squared.to_bits() == other.min_norm_squared.to_bits()
    }
}

impl Eq for HnswParams {}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            max_nb_connection: 16,
            max_layer: 16,
            ef_construction: 200,
            ef_search: 200,
            max_elements: 100_000,
            min_norm_squared: DEFAULT_MIN_NORM_SQUARED,
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

/// Snapshot format selector for on-disk persistence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum VectorSnapshotWriteVersion {
    /// Legacy JSON snapshot (`snapshot.v1.json`).
    #[default]
    V1,
    /// v2 metadata + binary payload (`snapshot.meta` + `vectors.u8.bin`).
    V2,
}

impl VectorSnapshotWriteVersion {
    /// Canonical lowercase version label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::V1 => "v1",
            Self::V2 => "v2",
        }
    }
}

/// Deterministic snapshot size and shape stats.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotStats {
    /// Snapshot output version these stats are computed for.
    pub version: VectorSnapshotWriteVersion,
    /// Vector dimensionality.
    pub dimension: u32,
    /// Number of vector records.
    pub count: u64,
    /// Total estimated bytes written for this snapshot output.
    pub bytes: u64,
    /// Extra deterministic metadata sorted by key.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<Box<str>, Box<str>>,
}

/// Load options for `VectorIndex::from_snapshot_v2_with_options`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VectorSnapshotV2LoadOptions {
    /// Allow automatic v1 -> v2 upgrade when `snapshot.meta` is missing.
    pub auto_upgrade_v1: bool,
}

/// Search match with similarity score.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorMatch {
    /// External identifier for this vector.
    pub id: Box<str>,
    /// Similarity score in [0, 1].
    pub score: f32,
}

/// Search backend used by `VectorIndex::search_with_backend`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorSearchBackend {
    /// Default path backed by f32 HNSW with exact-scan shortfall recovery.
    F32Hnsw,
    /// Experimental exact search on quantized u8 vectors.
    ExperimentalU8Quantized,
    /// Experimental two-stage search (`u8` candidate generation + `f32` rerank).
    ExperimentalU8ThenF32Rerank,
}

/// Kernel family used by the local vector index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum VectorKernelKind {
    /// Built-in HNSW kernel backed by `hnsw_rs`.
    #[default]
    HnswRs,
    /// Experimental DFRR kernel.
    Dfrr,
    /// Brute-force exact nearest-neighbor scan for benchmark ground truth.
    FlatScan,
}

/// Kernel-level search stats for benchmarking and diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelSearchStats {
    /// Kernel-specific expansion count, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expansions: Option<u64>,
    /// Kernel that produced this search output.
    pub kernel: VectorKernelKind,
    /// Kernel-specific extended metrics (e.g. DFRR pulls, splits, bucket utilization).
    /// Keys use camelCase naming. Empty map is omitted from JSON.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<Box<str>, f64>,
    /// Wall-clock nanoseconds spent in the kernel search algorithm.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_search_duration_ns: Option<u64>,
}

/// Search output container with matches and kernel-level stats.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorSearchOutput {
    /// Sorted search matches.
    pub matches: Vec<VectorMatch>,
    /// Kernel-level stats captured during search.
    pub stats: KernelSearchStats,
}

/// Search-kernel contract for local vector indexes.
pub trait VectorKernel {
    /// Kernel family identifier.
    fn kind(&self) -> VectorKernelKind;

    /// Search through this kernel using the selected backend strategy.
    fn search(
        &self,
        index: &VectorIndex,
        query: &[f32],
        limit: usize,
        backend: VectorSearchBackend,
    ) -> Result<VectorSearchOutput>;

    /// Search with an optional per-call kernel-specific config override.
    ///
    /// `config_json` is an opaque JSON string whose schema depends on the
    /// kernel implementation. For DFRR kernels this is a serialized
    /// `DfrrLoopConfig`; other kernels may ignore it entirely.
    ///
    /// The default implementation ignores `config_json` and delegates to
    /// [`search()`](VectorKernel::search).
    fn search_with_config_override(
        &self,
        index: &VectorIndex,
        query: &[f32],
        limit: usize,
        backend: VectorSearchBackend,
        _config_json: Option<&str>,
    ) -> Result<VectorSearchOutput> {
        self.search(index, query, limit, backend)
    }
}

/// Built-in HNSW kernel implementation.
///
/// Carries an optional `ef_search` override that takes precedence over the
/// index-level `HnswParams::ef_search`.  When `None`, the index default is
/// used.  The **actual** `ef_search` applied is always emitted in
/// `KernelSearchStats::extra["efSearch"]` so benchmarks and diagnostics
/// record the true value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HnswKernel {
    ef_search_override: Option<usize>,
}

impl Default for HnswKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl HnswKernel {
    /// Create a kernel that falls back to the index-level `ef_search`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ef_search_override: None,
        }
    }

    /// Create a kernel with an explicit `ef_search` override.
    #[must_use]
    pub const fn with_ef_search(ef_search: usize) -> Self {
        Self {
            ef_search_override: Some(ef_search),
        }
    }
}

impl VectorKernel for HnswKernel {
    fn kind(&self) -> VectorKernelKind {
        VectorKernelKind::HnswRs
    }

    fn search(
        &self,
        index: &VectorIndex,
        query: &[f32],
        limit: usize,
        backend: VectorSearchBackend,
    ) -> Result<VectorSearchOutput> {
        let start = std::time::Instant::now();
        let matches = match backend {
            VectorSearchBackend::F32Hnsw => {
                index.search_f32_hnsw(query, limit, self.ef_search_override)
            },
            VectorSearchBackend::ExperimentalU8Quantized => index.search_u8_quantized(query, limit),
            VectorSearchBackend::ExperimentalU8ThenF32Rerank => {
                Self::search_u8_then_f32_rerank(index, query, limit)
            },
        }?;
        let kernel_search_duration_ns = u64::try_from(start.elapsed().as_nanos()).ok();

        // Record the actual ef_search used so benchmarks capture the true value.
        let total = index.records.len();
        let requested = limit.min(total);
        let base_ef = self.ef_search_override.unwrap_or(index.params.ef_search);
        let actual_ef = base_ef.max(requested);
        let mut extra = BTreeMap::new();
        #[expect(
            clippy::cast_precision_loss,
            reason = "ef_search is a small tuning parameter (typically <10_000); f64 is exact up to 2^53"
        )]
        let ef_search_f64 = actual_ef as f64;
        extra.insert("efSearch".into(), ef_search_f64);

        Ok(VectorSearchOutput {
            matches,
            stats: KernelSearchStats {
                expansions: None,
                kernel: self.kind(),
                extra,
                kernel_search_duration_ns,
            },
        })
    }
}

impl HnswKernel {
    #[cfg(feature = "experimental-u8-search")]
    fn search_u8_then_f32_rerank(
        index: &VectorIndex,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<VectorMatch>> {
        if index.records.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let query = prepare_search_query(index.dimension, index.params.min_norm_squared, query)?;

        let records = index.ordered_record_refs();
        if records.is_empty() {
            return Ok(Vec::new());
        }

        let requested = limit.min(records.len());
        let prepared = prepare_quantized_search(records.as_slice(), query.as_ref())?;
        let coarse = score_quantized_records(
            records.as_slice(),
            prepared.quantized_query.as_slice(),
            prepared.quantized_vectors.as_slice(),
            prepared.dimension,
        )?;

        let candidate_count = rerank_candidate_count(requested, records.len());
        let mut reranked = rerank_candidates(query.as_ref(), coarse.as_slice(), candidate_count);
        sort_matches_by_score_then_id(reranked.as_mut_slice());
        reranked.truncate(requested);
        Ok(reranked)
    }

    #[cfg(not(feature = "experimental-u8-search"))]
    fn search_u8_then_f32_rerank(
        index: &VectorIndex,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<VectorMatch>> {
        index.search_f32_hnsw(query, limit, None)
    }
}

/// Brute-force exact nearest-neighbor kernel.
///
/// O(N×D) per query — intended only as benchmark ground truth, not production use.
/// Iterates all active records, computes cosine similarity, sorts, and truncates to limit.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlatScanKernel;

impl VectorKernel for FlatScanKernel {
    fn kind(&self) -> VectorKernelKind {
        VectorKernelKind::FlatScan
    }

    fn search(
        &self,
        index: &VectorIndex,
        query: &[f32],
        limit: usize,
        _backend: VectorSearchBackend,
    ) -> Result<VectorSearchOutput> {
        let start = std::time::Instant::now();
        let matches = Self::flat_scan(index, query, limit)?;
        let kernel_search_duration_ns = u64::try_from(start.elapsed().as_nanos()).ok();

        Ok(VectorSearchOutput {
            matches,
            stats: KernelSearchStats {
                expansions: None,
                kernel: self.kind(),
                extra: BTreeMap::new(),
                kernel_search_duration_ns,
            },
        })
    }
}

impl FlatScanKernel {
    fn flat_scan(index: &VectorIndex, query: &[f32], limit: usize) -> Result<Vec<VectorMatch>> {
        if index.records.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let query = prepare_search_query(index.dimension, index.params.min_norm_squared, query)?;
        let metric = DistAccelerateCosine;

        let mut matches: Vec<VectorMatch> = index
            .records
            .par_iter()
            .enumerate()
            .filter(|(idx, _)| !index.deleted.contains(idx))
            .map(|(_, record)| {
                let score = (1.0 - metric.eval(query.as_ref(), record.vector.as_slice())).max(0.0);
                VectorMatch {
                    id: record.id.clone(),
                    score,
                }
            })
            .collect();

        sort_matches_by_score_then_id(matches.as_mut_slice());
        let requested = limit.min(matches.len());
        matches.truncate(requested);
        Ok(matches)
    }
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
        self.inner
            .search(query.as_slice(), limit)
            .map(|output| output.matches)
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

/// Erase the lifetime parameter on a non-mmap `Hnsw` loaded from disk.
///
/// # Safety
///
/// The caller MUST guarantee that the `Hnsw` was loaded **without mmap** — i.e.
/// `ReloadOptions::use_mmap()` returned `(false, _)`. When mmap is off, every
/// `Point` inside the HNSW stores its data as `PointData::V(Vec<f32>)` (fully
/// owned). The lifetime `'b` on `Hnsw<'b, T, D>` only borrows from `HnswIo`
/// when `PointData::S(&'b [T])` (mmap slice) is used. With mmap disabled the
/// borrow is vacuous, so the transmute to `'static` is sound.
///
/// After this call the source `HnswIo` may be dropped without affecting the
/// returned `Hnsw<'static, …>`.
#[expect(unsafe_code, reason = "sound lifetime erasure for non-mmap HNSW load")]
unsafe fn erase_hnsw_lifetime(
    hnsw: Hnsw<'_, f32, DistAccelerateCosine>,
) -> Hnsw<'static, f32, DistAccelerateCosine> {
    // SAFETY: see doc-comment above — all PointData variants are V(Vec<f32>)
    // when mmap is disabled, so the lifetime is vacuously 'static.
    unsafe { std::mem::transmute(hnsw) }
}

/// In-memory vector index backed by HNSW.
pub struct VectorIndex {
    dimension: u32,
    params: HnswParams,
    hnsw: Hnsw<'static, f32, DistAccelerateCosine>,
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
        validate_min_norm_squared(params.min_norm_squared)?;
        let max_elements = params.max_elements.max(1);
        let hnsw = Hnsw::new(
            params.max_nb_connection,
            max_elements,
            params.max_layer,
            params.ef_construction,
            DistAccelerateCosine,
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

    /// Return HNSW parameters used for this index.
    #[must_use]
    pub const fn params(&self) -> &HnswParams {
        &self.params
    }

    /// Return read-only references to active (non-deleted) records in id-sorted order.
    ///
    /// This provides deterministic ordering regardless of insertion sequence,
    /// which is essential for building DFRR structures with stable `NodeId` assignment.
    #[must_use]
    pub fn active_records(&self) -> Vec<&VectorRecord> {
        self.ordered_record_refs()
    }

    /// Return the number of active (non-deleted) records.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.id_to_index.len()
    }

    /// Insert or update records in the index.
    #[instrument(
        name = "vector.index.insert_batch",
        skip_all,
        fields(dimension = self.dimension, record_count = records.len())
    )]
    pub fn insert(&mut self, records: Vec<VectorRecord>) -> Result<()> {
        for record in records {
            ensure_dimension(self.dimension, &record.vector)?;
            let indexed_vector = prepare_vector_for_cosine(
                record.vector.as_slice(),
                self.params.min_norm_squared,
                "insert",
            )
            .map_err(|error| error.with_metadata("id", record.id.to_string()))?;

            let index = self.records.len();
            if let Some(previous) = self.id_to_index.insert(record.id.clone(), index) {
                self.deleted.insert(previous);
            }

            self.hnsw.insert((indexed_vector.as_ref(), index));
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
    #[instrument(name = "vector.index.search", skip_all, fields(dimension = self.dimension, limit))]
    pub fn search(&self, query: &[f32], limit: usize) -> Result<VectorSearchOutput> {
        self.search_with_kernel(
            query,
            limit,
            &HnswKernel::new(),
            VectorSearchBackend::F32Hnsw,
        )
    }

    /// Search with an explicit backend strategy.
    pub fn search_with_backend(
        &self,
        query: &[f32],
        limit: usize,
        backend: VectorSearchBackend,
    ) -> Result<VectorSearchOutput> {
        self.search_with_kernel(query, limit, &HnswKernel::new(), backend)
    }

    /// Search with an explicit kernel + backend strategy.
    #[instrument(
        name = "vector.index.search_with_kernel",
        skip_all,
        fields(
            dimension = self.dimension,
            limit,
            kernel = ?kernel.kind(),
            backend = ?backend
        )
    )]
    pub fn search_with_kernel(
        &self,
        query: &[f32],
        limit: usize,
        kernel: &dyn VectorKernel,
        backend: VectorSearchBackend,
    ) -> Result<VectorSearchOutput> {
        kernel.search(self, query, limit, backend)
    }

    fn search_f32_hnsw(
        &self,
        query: &[f32],
        limit: usize,
        ef_search_override: Option<usize>,
    ) -> Result<Vec<VectorMatch>> {
        if self.records.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let query = prepare_search_query(self.dimension, self.params.min_norm_squared, query)?;

        let total = self.records.len();
        let requested = limit.min(total);
        let knbn = requested;
        let base_ef = ef_search_override.unwrap_or(self.params.ef_search);
        let ef_search = base_ef.max(knbn);

        let neighbours = self.hnsw.search(query.as_ref(), knbn, ef_search);
        let mut matches = to_matches(&self.records, &self.deleted, neighbours);
        self.fill_shortfall_with_exact_scan(query.as_ref(), requested, &mut matches);

        sort_matches_by_score_then_id(matches.as_mut_slice());
        matches.truncate(requested);
        Ok(matches)
    }

    #[cfg(feature = "experimental-u8-search")]
    fn search_u8_quantized(&self, query: &[f32], limit: usize) -> Result<Vec<VectorMatch>> {
        if self.records.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let query = prepare_search_query(self.dimension, self.params.min_norm_squared, query)?;

        let records = self.ordered_record_refs();
        if records.is_empty() {
            return Ok(Vec::new());
        }

        let requested = limit.min(records.len());
        let prepared = prepare_quantized_search(records.as_slice(), query.as_ref())?;
        let scored = score_quantized_records(
            records.as_slice(),
            prepared.quantized_query.as_slice(),
            prepared.quantized_vectors.as_slice(),
            prepared.dimension,
        )?;

        let mut matches = Vec::with_capacity(scored.len());
        for (record, score) in scored {
            matches.push(VectorMatch {
                id: record.id.clone(),
                score,
            });
        }

        sort_matches_by_score_then_id(matches.as_mut_slice());
        matches.truncate(requested);
        Ok(matches)
    }

    #[cfg(not(feature = "experimental-u8-search"))]
    fn search_u8_quantized(&self, query: &[f32], limit: usize) -> Result<Vec<VectorMatch>> {
        self.search_f32_hnsw(query, limit, None)
    }

    fn fill_shortfall_with_exact_scan(
        &self,
        query: &[f32],
        requested: usize,
        matches: &mut Vec<VectorMatch>,
    ) {
        if matches.len() >= requested {
            return;
        }

        let mut seen_ids = matches
            .iter()
            .map(|item| item.id.clone())
            .collect::<HashSet<_>>();
        let metric = DistAccelerateCosine;
        for (index, record) in self.records.iter().enumerate() {
            if matches.len() >= requested {
                break;
            }
            if self.deleted.contains(&index) {
                continue;
            }
            if !seen_ids.insert(record.id.clone()) {
                continue;
            }

            let score = (1.0 - metric.eval(query, record.vector.as_slice())).max(0.0);
            matches.push(VectorMatch {
                id: record.id.clone(),
                score,
            });
        }
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
        VectorSnapshot {
            version: VECTOR_SNAPSHOT_VERSION,
            dimension: self.dimension,
            params: self.params,
            records: self.ordered_records(),
        }
    }

    /// Compute deterministic snapshot stats for the selected format.
    pub fn snapshot_stats(&self, version: VectorSnapshotWriteVersion) -> Result<SnapshotStats> {
        let records = self.ordered_records();
        self.snapshot_stats_for_records(version, records.as_slice())
    }

    fn snapshot_stats_for_records(
        &self,
        version: VectorSnapshotWriteVersion,
        records: &[VectorRecord],
    ) -> Result<SnapshotStats> {
        match version {
            VectorSnapshotWriteVersion::V1 => self.snapshot_stats_v1(records),
            VectorSnapshotWriteVersion::V2 => self.snapshot_stats_v2(records),
        }
    }

    fn snapshot_stats_v1(&self, records: &[VectorRecord]) -> Result<SnapshotStats> {
        let snapshot = VectorSnapshot {
            version: VECTOR_SNAPSHOT_VERSION,
            dimension: self.dimension,
            params: self.params,
            records: records.to_vec(),
        };
        let payload = serde_json::to_vec(&snapshot).map_err(|source| {
            ErrorEnvelope::expected(
                ErrorCode::new("vector", "snapshot_serialize_failed"),
                "failed to serialize v1 snapshot stats",
            )
            .with_metadata("source", source.to_string())
        })?;
        let payload_bytes = usize_to_u64(payload.len(), "snapshot v1 payload size overflow")?;
        let count = usize_to_u64(records.len(), "snapshot record count conversion overflow")?;
        let mut metadata = BTreeMap::new();
        metadata.insert("fileCount".into(), "1".into());
        metadata.insert(
            "files.snapshot.v1.json.bytes".into(),
            payload_bytes.to_string().into(),
        );

        Ok(SnapshotStats {
            version: VectorSnapshotWriteVersion::V1,
            dimension: self.dimension,
            count,
            bytes: payload_bytes,
            metadata,
        })
    }

    fn snapshot_stats_v2(&self, records: &[VectorRecord]) -> Result<SnapshotStats> {
        let stats_path = Path::new(SNAPSHOT_STATS_CONTEXT);
        let quantization = fit_quantization(records, self.dimension, stats_path)?;
        let quantizer = Quantizer::new(quantization.clone())
            .map_err(|source| map_quantization_error(stats_path, &source, "build quantizer"))?;
        let dataset = records
            .iter()
            .map(|record| record.vector.as_slice())
            .collect::<Vec<_>>();
        let vectors = quantizer
            .quantize_batch(dataset.as_slice())
            .map_err(|source| {
                map_quantization_error(stats_path, &source, "quantize vectors for v2 stats")
            })?;
        let count = usize_to_u64(records.len(), "snapshot record count conversion overflow")?;
        let vectors_crc32 = compute_vectors_crc32(vectors.as_slice());
        let meta = VectorSnapshotMeta::new(
            self.dimension,
            count,
            self.params,
            quantization,
            vectors_crc32,
        )
        .map_err(|source| map_snapshot_error(stats_path, &source, "build v2 stats metadata"))?;
        let meta_bytes = encode_metadata(&meta)
            .map_err(|source| map_snapshot_error(stats_path, &source, "encode v2 stats metadata"))
            .and_then(|bytes| usize_to_u64(bytes.len(), "snapshot meta size overflow"))?;
        let ids = records
            .iter()
            .map(|record| record.id.clone())
            .collect::<Vec<_>>();
        let ids_payload = serde_json::to_vec(ids.as_slice()).map_err(|source| {
            ErrorEnvelope::expected(
                ErrorCode::new("vector", "snapshot_ids_serialize_failed"),
                "failed to serialize v2 snapshot ids for stats",
            )
            .with_metadata("source", source.to_string())
        })?;
        let ids_bytes = usize_to_u64(ids_payload.len(), "snapshot ids size overflow")?;
        let vectors_bytes = usize_to_u64(vectors.len(), "snapshot vectors size overflow")?;
        let total_bytes = add_u64_checked(
            add_u64_checked(vectors_bytes, meta_bytes, "snapshot stats size overflow")?,
            ids_bytes,
            "snapshot stats size overflow",
        )?;

        let mut metadata = BTreeMap::new();
        metadata.insert("fileCount".into(), "3".into());
        metadata.insert("files.ids.json.bytes".into(), ids_bytes.to_string().into());
        metadata.insert(
            "files.snapshot.meta.bytes".into(),
            meta_bytes.to_string().into(),
        );
        metadata.insert(
            "files.vectors.u8.bin.bytes".into(),
            vectors_bytes.to_string().into(),
        );

        Ok(SnapshotStats {
            version: VectorSnapshotWriteVersion::V2,
            dimension: self.dimension,
            count,
            bytes: total_bytes,
            metadata,
        })
    }

    /// Write an on-disk snapshot in the selected format.
    pub fn write_snapshot(
        &self,
        snapshot_dir: impl AsRef<Path>,
        version: VectorSnapshotWriteVersion,
    ) -> Result<()> {
        self.write_snapshot_with_size_limit(snapshot_dir, version, None)
    }

    /// Write an on-disk snapshot with an optional max-bytes cap.
    pub fn write_snapshot_with_size_limit(
        &self,
        snapshot_dir: impl AsRef<Path>,
        version: VectorSnapshotWriteVersion,
        max_snapshot_bytes: Option<u64>,
    ) -> Result<()> {
        let snapshot_dir = snapshot_dir.as_ref();
        std::fs::create_dir_all(snapshot_dir).map_err(|source| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "snapshot_create_dir_failed"),
                "failed to create snapshot directory",
                ErrorClass::NonRetriable,
            )
            .with_metadata("snapshotDir", snapshot_dir.display().to_string())
            .with_metadata("source", source.to_string())
        })?;

        match version {
            VectorSnapshotWriteVersion::V1 => {
                let stats = self.snapshot_stats(VectorSnapshotWriteVersion::V1)?;
                enforce_snapshot_size_limit(snapshot_dir, &stats, max_snapshot_bytes)?;
                let payload = serde_json::to_vec(&self.snapshot()).map_err(|source| {
                    ErrorEnvelope::expected(
                        ErrorCode::new("vector", "snapshot_serialize_failed"),
                        "failed to serialize v1 snapshot",
                    )
                    .with_metadata("snapshotDir", snapshot_dir.display().to_string())
                    .with_metadata("source", source.to_string())
                })?;
                let path = snapshot_dir.join(SNAPSHOT_V1_FILE_NAME);
                std::fs::write(&path, payload).map_err(|source| {
                    ErrorEnvelope::unexpected(
                        ErrorCode::new("vector", "snapshot_write_failed"),
                        "failed to write v1 snapshot",
                        ErrorClass::NonRetriable,
                    )
                    .with_metadata("path", path.display().to_string())
                    .with_metadata("source", source.to_string())
                })?;
                Ok(())
            },
            VectorSnapshotWriteVersion::V2 => self
                .snapshot_v2_with_kernel_and_size_limit(
                    snapshot_dir,
                    VectorKernelKind::HnswRs,
                    max_snapshot_bytes,
                )
                .map(|_| ()),
        }
    }

    /// Write a v2 snapshot bundle to disk.
    pub fn snapshot_v2(&self, snapshot_dir: impl AsRef<Path>) -> Result<VectorSnapshotMeta> {
        self.snapshot_v2_with_kernel_and_size_limit(snapshot_dir, VectorKernelKind::HnswRs, None)
    }

    /// Write a v2 snapshot bundle to disk with an explicit kernel family.
    pub fn snapshot_v2_for_kernel(
        &self,
        snapshot_dir: impl AsRef<Path>,
        kernel: VectorKernelKind,
    ) -> Result<VectorSnapshotMeta> {
        self.snapshot_v2_with_kernel_and_size_limit(snapshot_dir, kernel, None)
    }

    /// Write a v2 snapshot bundle with an optional max-bytes cap.
    pub fn snapshot_v2_with_size_limit(
        &self,
        snapshot_dir: impl AsRef<Path>,
        max_snapshot_bytes: Option<u64>,
    ) -> Result<VectorSnapshotMeta> {
        self.snapshot_v2_with_kernel_and_size_limit(
            snapshot_dir,
            VectorKernelKind::HnswRs,
            max_snapshot_bytes,
        )
    }

    /// Write a v2 snapshot bundle with kernel metadata and an optional size cap.
    pub fn snapshot_v2_with_kernel_and_size_limit(
        &self,
        snapshot_dir: impl AsRef<Path>,
        kernel: VectorKernelKind,
        max_snapshot_bytes: Option<u64>,
    ) -> Result<VectorSnapshotMeta> {
        let snapshot_dir = snapshot_dir.as_ref();
        let stats = self.snapshot_stats(VectorSnapshotWriteVersion::V2)?;
        enforce_snapshot_size_limit(snapshot_dir, &stats, max_snapshot_bytes)?;
        let records = self.ordered_records();
        let quantization = fit_quantization(records.as_slice(), self.dimension, snapshot_dir)?;
        let quantizer = Quantizer::new(quantization.clone())
            .map_err(|source| map_quantization_error(snapshot_dir, &source, "build quantizer"))?;
        let dataset = records
            .iter()
            .map(|record| record.vector.as_slice())
            .collect::<Vec<_>>();
        let vectors = quantizer
            .quantize_batch(dataset.as_slice())
            .map_err(|source| {
                map_quantization_error(snapshot_dir, &source, "quantize vectors for v2 snapshot")
            })?;
        let count = u64::try_from(records.len()).map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "snapshot_count_overflow"),
                "snapshot record count conversion overflow",
                ErrorClass::NonRetriable,
            )
            .with_metadata("snapshotDir", snapshot_dir.display().to_string())
            .with_metadata("count", records.len().to_string())
        })?;

        let meta = write_snapshot_v2_with_kernel(
            snapshot_dir,
            self.params,
            kernel,
            quantization,
            count,
            &vectors,
        )
        .map_err(|source| {
            map_snapshot_error(snapshot_dir, &source, "write v2 snapshot metadata+vectors")
        })?;

        let ids = records
            .into_iter()
            .map(|record| record.id)
            .collect::<Vec<_>>();
        write_snapshot_v2_ids(snapshot_dir, ids.as_slice())?;

        // Persist the HNSW graph topology alongside the quantized vectors.
        // hnsw_rs `file_dump` writes `$basename.hnsw.graph` + `$basename.hnsw.data`
        // into the snapshot directory. The graph file captures the full HNSW layer
        // structure and neighbourhood lists; the data file stores the f32 vectors
        // that the graph nodes refer to during search.
        //
        // Empty indexes have no entry point, so `file_dump` would fail — skip.
        if self.active_count() > 0 {
            let max_layer = self.hnsw.get_max_level();
            if max_layer != HNSW_RS_GRAPH_DUMP_REQUIRED_MAX_LAYER {
                tracing::warn!(
                    snapshot_dir = %snapshot_dir.display(),
                    max_layer,
                    required_max_layer = HNSW_RS_GRAPH_DUMP_REQUIRED_MAX_LAYER,
                    "skipping persisted HNSW graph dump; runtime max_layer is incompatible with hnsw_rs dump format"
                );
            } else if let Err(source) = self
                .hnsw
                .file_dump(snapshot_dir, SNAPSHOT_V2_HNSW_GRAPH_BASENAME)
            {
                tracing::warn!(
                    snapshot_dir = %snapshot_dir.display(),
                    error = %source,
                    "persisted HNSW graph dump failed; continuing with metadata+vectors only"
                );
            }
        }

        Ok(meta)
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

    /// Restore a vector index from a v2 snapshot bundle.
    pub fn from_snapshot_v2(snapshot_dir: impl AsRef<Path>) -> Result<Self> {
        Self::from_snapshot_v2_with_options(snapshot_dir, VectorSnapshotV2LoadOptions::default())
    }

    /// Restore a vector index from a v2 bundle with explicit load options.
    ///
    /// When a persisted HNSW graph (`hnsw_graph.hnsw.graph` + `hnsw_graph.hnsw.data`)
    /// exists in the snapshot directory, it is loaded directly — skipping the O(n log n)
    /// graph rebuild entirely. Falls back to full reconstruction when graph files are
    /// missing (e.g. snapshots produced before graph persistence was added).
    #[instrument(name = "vector.index.from_snapshot_v2", skip_all)]
    pub fn from_snapshot_v2_with_options(
        snapshot_dir: impl AsRef<Path>,
        options: VectorSnapshotV2LoadOptions,
    ) -> Result<Self> {
        let snapshot_dir = snapshot_dir.as_ref().to_path_buf();
        let loaded = read_snapshot_v2_with_options(
            &snapshot_dir,
            ReadSnapshotV2Options {
                auto_upgrade_v1: options.auto_upgrade_v1,
            },
        )
        .map_err(|source| map_snapshot_error(&snapshot_dir, &source, "read v2 snapshot bundle"))?;

        let quantized_vectors = loaded.quantized_vectors().map_err(|source| {
            map_snapshot_error(&snapshot_dir, &source, "validate quantized vectors")
        })?;
        let ids = read_snapshot_v2_ids(&snapshot_dir, quantized_vectors.len())?;
        if ids.len() != quantized_vectors.len() {
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("vector", "snapshot_record_count_mismatch"),
                "snapshot ids and vector counts do not match",
            )
            .with_metadata("snapshotDir", snapshot_dir.display().to_string())
            .with_metadata("ids", ids.len().to_string())
            .with_metadata("vectors", quantized_vectors.len().to_string()));
        }

        let quantizer = Quantizer::new(loaded.meta.quantization.clone())
            .map_err(|source| map_quantization_error(&snapshot_dir, &source, "build quantizer"))?;
        let mut records = Vec::with_capacity(ids.len());
        for (record_index, (id, quantized_vector)) in
            ids.into_iter().zip(quantized_vectors.iter()).enumerate()
        {
            let vector = quantizer.dequantize(quantized_vector).map_err(|source| {
                map_quantization_error(&snapshot_dir, &source, "dequantize snapshot vector")
                    .with_metadata("recordIndex", record_index.to_string())
            })?;
            ensure_dimension(loaded.meta.dimension, vector.as_slice()).map_err(|error| {
                error
                    .with_metadata("snapshotDir", snapshot_dir.display().to_string())
                    .with_metadata("recordIndex", record_index.to_string())
            })?;
            records.push(VectorRecord { id, vector });
        }

        let mut params = loaded.meta.params;
        params.max_elements = params.max_elements.max(records.len().max(1));

        // Try the fast path: load a persisted HNSW graph from disk.
        // When the graph files exist, we skip the O(n log n) HNSW rebuild entirely.
        // On failure we fall back to full reconstruction by re-dequantizing from
        // the same snapshot (records are consumed by load_persisted_graph).
        let graph_file = snapshot_dir.join(format!("{SNAPSHOT_V2_HNSW_GRAPH_BASENAME}.hnsw.graph"));
        if graph_file.exists() {
            match Self::load_persisted_graph(&snapshot_dir, loaded.meta.dimension, params, records)
            {
                Ok(index) => return Ok(index),
                Err(error) => {
                    tracing::warn!(
                        path = %graph_file.display(),
                        %error,
                        "persisted HNSW graph load failed, falling back to full rebuild"
                    );
                    // Re-dequantize records for the fallback path since the fast
                    // path consumed the original Vec.
                    let ids_again = read_snapshot_v2_ids(&snapshot_dir, quantized_vectors.len())?;
                    let mut rebuilt_records = Vec::with_capacity(ids_again.len());
                    for (id, qv) in ids_again.into_iter().zip(quantized_vectors.iter()) {
                        let vector = quantizer.dequantize(qv).map_err(|source| {
                            map_quantization_error(
                                &snapshot_dir,
                                &source,
                                "dequantize snapshot vector (fallback)",
                            )
                        })?;
                        rebuilt_records.push(VectorRecord { id, vector });
                    }
                    let mut index = Self::new(loaded.meta.dimension, params)?;
                    index.insert(rebuilt_records)?;
                    return Ok(index);
                },
            }
        }

        // No persisted graph: rebuild HNSW graph from scratch (O(n log n)).
        let mut index = Self::new(loaded.meta.dimension, params)?;
        index.insert(records)?;
        Ok(index)
    }

    /// Load a persisted HNSW graph from the snapshot directory.
    ///
    /// Uses `HnswIo::load_hnsw_with_dist` to reload the graph topology and
    /// embedded f32 vectors from `hnsw_graph.hnsw.{graph,data}`. The loaded
    /// `Hnsw` already contains the full graph structure with all vectors — no
    /// per-record insertion is needed.
    ///
    /// The loaded `Hnsw` is transmuted from `Hnsw<'_, …>` to `Hnsw<'static, …>`
    /// because mmap is disabled and all point data is fully owned.
    #[instrument(name = "vector.index.load_persisted_graph", skip_all)]
    fn load_persisted_graph(
        snapshot_dir: &Path,
        dimension: u32,
        params: HnswParams,
        records: Vec<VectorRecord>,
    ) -> Result<Self> {
        let hnswio = HnswIo::new(snapshot_dir, SNAPSHOT_V2_HNSW_GRAPH_BASENAME);
        // `load_hnsw_with_dist` uses no mmap (default ReloadOptions).
        // All Point data is PointData::V(Vec<f32>) — fully owned, no borrows
        // from HnswIo. See `erase_hnsw_lifetime` safety doc for the proof.
        let loaded_hnsw = hnswio
            .load_hnsw_with_dist(DistAccelerateCosine)
            .map_err(|source| {
                ErrorEnvelope::unexpected(
                    ErrorCode::new("vector", "snapshot_graph_load_failed"),
                    format!("failed to load persisted HNSW graph: {source}"),
                    ErrorClass::NonRetriable,
                )
                .with_metadata("snapshotDir", snapshot_dir.display().to_string())
            })?;

        // SAFETY: mmap is disabled (default ReloadOptions), so the lifetime is
        // vacuously 'static. See `erase_hnsw_lifetime` doc-comment.
        #[expect(unsafe_code, reason = "sound lifetime erasure for non-mmap HNSW load")]
        let hnsw = unsafe { erase_hnsw_lifetime(loaded_hnsw) };

        let mut id_to_index = HashMap::with_capacity(records.len());
        for (index, record) in records.iter().enumerate() {
            id_to_index.insert(record.id.clone(), index);
        }

        tracing::info!(
            record_count = records.len(),
            "loaded persisted HNSW graph from snapshot"
        );

        Ok(Self {
            dimension,
            params,
            hnsw,
            records,
            id_to_index,
            deleted: HashSet::new(),
        })
    }

    fn ordered_record_refs(&self) -> Vec<&VectorRecord> {
        let mut ordered: BTreeMap<&str, &VectorRecord> = BTreeMap::new();
        for (id, index) in &self.id_to_index {
            if let Some(record) = self.records.get(*index) {
                ordered.insert(id.as_ref(), record);
            }
        }

        ordered.into_values().collect::<Vec<&VectorRecord>>()
    }

    fn ordered_records(&self) -> Vec<VectorRecord> {
        self.ordered_record_refs()
            .into_iter()
            .cloned()
            .collect::<Vec<VectorRecord>>()
    }
}

#[cfg(feature = "experimental-u8-search")]
#[derive(Default, Copy, Clone)]
struct DistU8Cosine;

#[cfg(feature = "experimental-u8-search")]
impl Distance<u8> for DistU8Cosine {
    fn eval(&self, va: &[u8], vb: &[u8]) -> f32 {
        if va.len() != vb.len() {
            return 1.0;
        }

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for (lhs, rhs) in va.iter().copied().zip(vb.iter().copied()) {
            let lhs = decode_u8_to_f32(lhs);
            let rhs = decode_u8_to_f32(rhs);
            dot += lhs * rhs;
            norm_a += lhs * lhs;
            norm_b += rhs * rhs;
        }

        if norm_a <= 0.0 || norm_b <= 0.0 {
            return 0.0;
        }

        let distance = 1.0 - (dot / (norm_a * norm_b).sqrt());
        if distance.is_finite() {
            return distance.max(0.0);
        }
        1.0
    }
}

fn fit_quantization(
    records: &[VectorRecord],
    dimension: u32,
    snapshot_dir: &Path,
) -> Result<QuantizationParams> {
    let dimension = usize::try_from(dimension).map_err(|_| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "invalid_dimension"),
            "dimension conversion overflow",
            ErrorClass::NonRetriable,
        )
        .with_metadata("snapshotDir", snapshot_dir.display().to_string())
    })?;
    if records.is_empty() {
        return QuantizationParams::new(vec![1.0; dimension], vec![0.0; dimension]).map_err(
            |source| map_quantization_error(snapshot_dir, &source, "build default quantization"),
        );
    }

    let dataset = records
        .iter()
        .map(|record| record.vector.as_slice())
        .collect::<Vec<_>>();
    fit_min_max(dataset.as_slice())
        .map_err(|source| map_quantization_error(snapshot_dir, &source, "fit quantization"))
}

fn write_snapshot_v2_ids(snapshot_dir: &Path, ids: &[Box<str>]) -> Result<()> {
    let path = snapshot_dir.join(VECTOR_SNAPSHOT_V2_IDS_FILE_NAME);
    let payload = serde_json::to_vec(ids).map_err(|source| {
        ErrorEnvelope::expected(
            ErrorCode::new("vector", "snapshot_ids_serialize_failed"),
            "failed to serialize v2 snapshot ids",
        )
        .with_metadata("path", path.display().to_string())
        .with_metadata("source", source.to_string())
    })?;
    std::fs::write(&path, payload).map_err(|source| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "snapshot_ids_write_failed"),
            "failed to write v2 snapshot ids",
            ErrorClass::NonRetriable,
        )
        .with_metadata("path", path.display().to_string())
        .with_metadata("source", source.to_string())
    })
}

fn read_snapshot_v2_ids(snapshot_dir: &Path, expected_count: usize) -> Result<Vec<Box<str>>> {
    let path = snapshot_dir.join(VECTOR_SNAPSHOT_V2_IDS_FILE_NAME);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(source) if source.kind() == io::ErrorKind::NotFound => {
            return read_snapshot_v1_ids(snapshot_dir, expected_count);
        },
        Err(source) => {
            return Err(ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "snapshot_ids_read_failed"),
                "failed to read v2 snapshot ids",
                ErrorClass::NonRetriable,
            )
            .with_metadata("path", path.display().to_string())
            .with_metadata("source", source.to_string()));
        },
    };

    let ids = serde_json::from_slice::<Vec<Box<str>>>(&bytes).map_err(|source| {
        ErrorEnvelope::expected(
            ErrorCode::new("vector", "snapshot_ids_parse_failed"),
            "failed to parse v2 snapshot ids",
        )
        .with_metadata("path", path.display().to_string())
        .with_metadata("source", source.to_string())
    })?;

    if ids.len() != expected_count {
        return Err(ErrorEnvelope::expected(
            ErrorCode::new("vector", "snapshot_record_count_mismatch"),
            "snapshot ids and vectors count mismatch",
        )
        .with_metadata("snapshotDir", snapshot_dir.display().to_string())
        .with_metadata("ids", ids.len().to_string())
        .with_metadata("vectors", expected_count.to_string()));
    }
    Ok(ids)
}

fn read_snapshot_v1_ids(snapshot_dir: &Path, expected_count: usize) -> Result<Vec<Box<str>>> {
    let path = snapshot_dir.join(SNAPSHOT_V1_FILE_NAME);
    let bytes = std::fs::read(&path).map_err(|source| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "snapshot_ids_read_failed"),
            "failed to read v1 snapshot fallback ids",
            ErrorClass::NonRetriable,
        )
        .with_metadata("path", path.display().to_string())
        .with_metadata("source", source.to_string())
    })?;
    let snapshot = serde_json::from_slice::<VectorSnapshot>(&bytes).map_err(|source| {
        ErrorEnvelope::expected(
            ErrorCode::new("vector", "snapshot_parse_failed"),
            "failed to parse v1 snapshot for ids fallback",
        )
        .with_metadata("path", path.display().to_string())
        .with_metadata("source", source.to_string())
    })?;

    let ids = snapshot
        .records
        .into_iter()
        .map(|record| record.id)
        .collect::<Vec<_>>();
    if ids.len() != expected_count {
        return Err(ErrorEnvelope::expected(
            ErrorCode::new("vector", "snapshot_record_count_mismatch"),
            "snapshot ids and vectors count mismatch",
        )
        .with_metadata("snapshotDir", snapshot_dir.display().to_string())
        .with_metadata("ids", ids.len().to_string())
        .with_metadata("vectors", expected_count.to_string()));
    }
    Ok(ids)
}

fn map_snapshot_error(snapshot_dir: &Path, source: &SnapshotError, action: &str) -> ErrorEnvelope {
    let code = match source {
        SnapshotError::ReadMetadata { .. }
        | SnapshotError::WriteMetadata { .. }
        | SnapshotError::CreateMetadataDir { .. }
        | SnapshotError::CreateSnapshotDir { .. }
        | SnapshotError::WriteVectors { .. }
        | SnapshotError::MapVectors { .. }
        | SnapshotError::ReadLegacySnapshot { .. } => {
            ErrorCode::new("vector", "snapshot_io_failed")
        },
        _ => ErrorCode::new("vector", "snapshot_invalid"),
    };

    let base = if code == ErrorCode::new("vector", "snapshot_invalid") {
        ErrorEnvelope::expected(code, "snapshot v2 validation failed")
    } else {
        ErrorEnvelope::unexpected(
            code,
            "snapshot v2 IO operation failed",
            ErrorClass::NonRetriable,
        )
    };

    base.with_metadata("snapshotDir", snapshot_dir.display().to_string())
        .with_metadata("action", action.to_string())
        .with_metadata("source", source.to_string())
}

fn map_quantization_error(
    snapshot_dir: &Path,
    source: &QuantizationError,
    action: &str,
) -> ErrorEnvelope {
    ErrorEnvelope::expected(
        ErrorCode::new("vector", "snapshot_quantization_failed"),
        "snapshot quantization operation failed",
    )
    .with_metadata("snapshotDir", snapshot_dir.display().to_string())
    .with_metadata("action", action.to_string())
    .with_metadata("source", source.to_string())
}

fn map_search_quantization_error(source: &QuantizationError, action: &str) -> ErrorEnvelope {
    ErrorEnvelope::expected(
        ErrorCode::new("vector", "search_quantization_failed"),
        "u8 search quantization operation failed",
    )
    .with_metadata("action", action.to_string())
    .with_metadata("source", source.to_string())
}

fn enforce_snapshot_size_limit(
    snapshot_dir: &Path,
    stats: &SnapshotStats,
    max_snapshot_bytes: Option<u64>,
) -> Result<()> {
    let Some(max_snapshot_bytes) = max_snapshot_bytes else {
        return Ok(());
    };
    if stats.bytes <= max_snapshot_bytes {
        return Ok(());
    }

    Err(ErrorEnvelope::expected(
        ErrorCode::new("vector", "snapshot_oversize"),
        "snapshot exceeds configured size limit",
    )
    .with_metadata("snapshotDir", snapshot_dir.display().to_string())
    .with_metadata("version", stats.version.as_str().to_string())
    .with_metadata("bytes", stats.bytes.to_string())
    .with_metadata("maxBytes", max_snapshot_bytes.to_string()))
}

fn usize_to_u64(value: usize, context: &'static str) -> Result<u64> {
    u64::try_from(value).map_err(|_| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "snapshot_stats_overflow"),
            context,
            ErrorClass::NonRetriable,
        )
        .with_metadata("value", value.to_string())
    })
}

fn add_u64_checked(lhs: u64, rhs: u64, context: &'static str) -> Result<u64> {
    lhs.checked_add(rhs).ok_or_else(|| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "snapshot_stats_overflow"),
            context,
            ErrorClass::NonRetriable,
        )
        .with_metadata("left", lhs.to_string())
        .with_metadata("right", rhs.to_string())
    })
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

fn validate_min_norm_squared(min_norm_squared: f32) -> Result<()> {
    if !min_norm_squared.is_finite() || min_norm_squared <= 0.0 {
        return Err(ErrorEnvelope::expected(
            ErrorCode::new("vector", "invalid_hnsw_params"),
            "min_norm_squared must be finite and greater than zero",
        )
        .with_metadata("minNormSquared", min_norm_squared.to_string()));
    }
    Ok(())
}

fn prepare_vector_for_cosine<'a>(
    vector: &'a [f32],
    min_norm_squared: f32,
    context: &'static str,
) -> Result<Cow<'a, [f32]>> {
    let norm_squared = vector_norm_squared(vector, context)?;
    if norm_squared < f64::from(min_norm_squared) {
        return Err(ErrorEnvelope::expected(
            ErrorCode::new("vector", "vector_norm_too_small"),
            "vector norm is below configured minimum for cosine distance",
        )
        .with_metadata("context", context.to_string())
        .with_metadata("normSquared", norm_squared.to_string())
        .with_metadata("minNormSquared", min_norm_squared.to_string()));
    }
    Ok(Cow::Borrowed(vector))
}

fn prepare_search_query(
    dimension: u32,
    min_norm_squared: f32,
    query: &[f32],
) -> Result<Cow<'_, [f32]>> {
    ensure_dimension(dimension, query)?;
    prepare_vector_for_cosine(query, min_norm_squared, "search")
}

fn vector_norm_squared(vector: &[f32], context: &'static str) -> Result<f64> {
    let mut sum = 0.0f64;
    for (dimension, value) in vector.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("vector", "invalid_vector_value"),
                "vector contains a non-finite component",
            )
            .with_metadata("context", context.to_string())
            .with_metadata("dimension", dimension.to_string())
            .with_metadata("value", value.to_string()));
        }
        let value = f64::from(value);
        sum += value * value;
    }
    if !sum.is_finite() {
        return Err(ErrorEnvelope::expected(
            ErrorCode::new("vector", "invalid_vector_value"),
            "vector norm is not finite",
        )
        .with_metadata("context", context.to_string()));
    }
    Ok(sum)
}

fn normalize_for_cosine(vector: &[f32]) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= 0.0 {
        return vector.to_vec();
    }
    vector.iter().map(|value| value / norm).collect::<Vec<_>>()
}

#[cfg(feature = "experimental-u8-search")]
struct PreparedQuantizedSearch {
    quantized_query: Vec<u8>,
    quantized_vectors: Vec<u8>,
    dimension: usize,
}

#[cfg(feature = "experimental-u8-search")]
fn prepare_quantized_search(
    records: &[&VectorRecord],
    query: &[f32],
) -> Result<PreparedQuantizedSearch> {
    let normalized_dataset = records
        .iter()
        .map(|record| normalize_for_cosine(record.vector.as_slice()))
        .collect::<Vec<_>>();
    let dataset = normalized_dataset
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let normalized_query = normalize_for_cosine(query);
    let quantization = fit_min_max(dataset.as_slice())
        .map_err(|source| map_search_quantization_error(&source, "fit quantization"))?;
    let quantizer = Quantizer::new(quantization)
        .map_err(|source| map_search_quantization_error(&source, "build quantizer"))?;
    let quantized_query = quantizer
        .quantize(normalized_query.as_slice())
        .map_err(|source| map_search_quantization_error(&source, "quantize query"))?;
    let quantized_vectors = quantizer
        .quantize_batch(dataset.as_slice())
        .map_err(|source| map_search_quantization_error(&source, "quantize vectors"))?;

    Ok(PreparedQuantizedSearch {
        quantized_query,
        quantized_vectors,
        dimension: quantizer.params().dimension(),
    })
}

#[cfg(feature = "experimental-u8-search")]
fn score_quantized_records<'a>(
    records: &[&'a VectorRecord],
    quantized_query: &[u8],
    quantized_vectors: &[u8],
    dimension: usize,
) -> Result<Vec<(&'a VectorRecord, f32)>> {
    let view = QuantizedSlice::new(quantized_vectors, dimension)
        .map_err(|source| map_search_quantization_error(&source, "validate quantized view"))?;
    let metric = DistU8Cosine;
    let mut coarse = Vec::with_capacity(records.len());
    for (record, encoded) in records.iter().zip(view.iter()) {
        let score = (1.0 - metric.eval(quantized_query, encoded)).max(0.0);
        coarse.push((*record, score));
    }
    sort_scored_records_by_score_then_id(coarse.as_mut_slice());
    Ok(coarse)
}

fn sort_matches_by_score_then_id(matches: &mut [VectorMatch]) {
    matches.sort_by(|a, b| {
        let score = b.score.total_cmp(&a.score);
        if score != std::cmp::Ordering::Equal {
            return score;
        }
        a.id.cmp(&b.id)
    });
}

#[cfg(feature = "experimental-u8-search")]
fn sort_scored_records_by_score_then_id(scored: &mut [(&VectorRecord, f32)]) {
    scored.sort_by(|(left_record, left_score), (right_record, right_score)| {
        let score = right_score.total_cmp(left_score);
        if score != std::cmp::Ordering::Equal {
            return score;
        }
        left_record.id.cmp(&right_record.id)
    });
}

#[cfg(feature = "experimental-u8-search")]
fn rerank_candidate_count(requested: usize, total: usize) -> usize {
    requested
        .saturating_mul(U8_RERANK_CANDIDATE_MULTIPLIER)
        .clamp(U8_RERANK_MIN_CANDIDATES, U8_RERANK_MAX_CANDIDATES)
        .min(total)
}

#[cfg(feature = "experimental-u8-search")]
fn rerank_candidates(
    query: &[f32],
    coarse: &[(&VectorRecord, f32)],
    candidate_count: usize,
) -> Vec<VectorMatch> {
    let metric = DistAccelerateCosine;
    let mut reranked = Vec::with_capacity(candidate_count);
    for (record, _) in coarse.iter().copied().take(candidate_count) {
        let score = (1.0 - metric.eval(query, record.vector.as_slice())).max(0.0);
        reranked.push(VectorMatch {
            id: record.id.clone(),
            score,
        });
    }
    reranked
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
pub(crate) mod experimental {
    /// Placeholder trait for experimental extensions.
    #[expect(
        dead_code,
        reason = "reserved hook for experimental kernel extension work"
    )]
    pub trait VectorKernelExtension {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_shared::shared_crate_version;
    use std::io;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn create(prefix: &str) -> io::Result<Self> {
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
        let matches = restored.search(&[0.5, 0.5], 1)?.matches;

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

        let matches = index.search(&[0.1, 0.1], 2)?.matches;
        assert_eq!(matches.first().map(|m| m.id.as_ref()), Some("near"));
        Ok(())
    }

    #[test]
    fn invalid_dimension_rejected() {
        let result = VectorIndex::new(0, HnswParams::default());
        assert!(result.is_err());
    }

    #[test]
    fn invalid_min_norm_squared_rejected() {
        let params = HnswParams {
            min_norm_squared: 0.0,
            ..HnswParams::default()
        };
        let result = VectorIndex::new(2, params);
        assert!(result.is_err());
    }

    #[test]
    fn insert_rejects_vector_below_min_norm() -> Result<()> {
        let params = HnswParams {
            min_norm_squared: 1e-8,
            ..HnswParams::default()
        };
        let mut index = VectorIndex::new(2, params)?;
        let result = index.insert(vec![VectorRecord {
            id: "tiny".into(),
            vector: vec![1e-5, 0.0],
        }]);

        assert!(result.is_err());
        let error = result.err().expect("insert should reject tiny vector");
        assert_eq!(
            error.code,
            ErrorCode::new("vector", "vector_norm_too_small")
        );
        Ok(())
    }

    #[test]
    fn search_rejects_query_below_min_norm() -> Result<()> {
        let params = HnswParams {
            min_norm_squared: 1e-8,
            ..HnswParams::default()
        };
        let mut index = VectorIndex::new(2, params)?;
        index.insert(vec![VectorRecord {
            id: "a".into(),
            vector: vec![1.0, 0.0],
        }])?;

        let result = index.search(&[1e-5, 0.0], 1);
        assert!(result.is_err());
        let error = result.err().expect("search should reject tiny query");
        assert_eq!(
            error.code,
            ErrorCode::new("vector", "vector_norm_too_small")
        );
        Ok(())
    }

    #[test]
    #[cfg(feature = "experimental-u8-search")]
    fn u8_distance_correctness_on_small_vectors() {
        let metric = DistU8Cosine;

        let identical = metric.eval(&[255, 0], &[255, 0]);
        assert!(identical <= 1e-6);

        let orthogonal = metric.eval(&[255, 0], &[0, 255]);
        assert!((orthogonal - 1.0).abs() <= 1e-6);

        let diagonal = metric.eval(&[255, 255], &[255, 0]);
        let expected = 1.0 - std::f32::consts::FRAC_1_SQRT_2;
        assert!((diagonal - expected).abs() <= 1e-4);
    }

    #[test]
    fn fallback_path_uses_f32_search_when_u8_disabled() -> Result<()> {
        let mut index = VectorIndex::new(2, HnswParams::default())?;
        index.insert(vec![
            VectorRecord {
                id: "a".into(),
                vector: vec![1.0, 0.0],
            },
            VectorRecord {
                id: "b".into(),
                vector: vec![0.0, 1.0],
            },
        ])?;

        let expected = index.search(&[1.0, 0.0], 2)?;
        let with_fallback =
            index.search_with_backend(&[1.0, 0.0], 2, VectorSearchBackend::F32Hnsw)?;
        assert_eq!(with_fallback.matches, expected.matches);
        Ok(())
    }

    #[test]
    fn default_kernel_search_matches_explicit_hnsw_kernel() -> Result<()> {
        let mut index = VectorIndex::new(2, HnswParams::default())?;
        index.insert(vec![
            VectorRecord {
                id: "a".into(),
                vector: vec![1.0, 0.0],
            },
            VectorRecord {
                id: "b".into(),
                vector: vec![0.0, 1.0],
            },
            VectorRecord {
                id: "c".into(),
                vector: vec![0.5, 0.5],
            },
        ])?;

        assert_eq!(VectorKernelKind::default(), VectorKernelKind::HnswRs);
        let query = [0.7, 0.3];
        let default_results = index.search(query.as_slice(), 3)?;
        let explicit_results = index.search_with_kernel(
            query.as_slice(),
            3,
            &HnswKernel::new(),
            VectorSearchBackend::F32Hnsw,
        )?;
        assert_eq!(default_results.matches, explicit_results.matches);
        Ok(())
    }

    #[test]
    fn hnsw_kernel_rerank_backend_matches_index_backend() -> Result<()> {
        let mut index = VectorIndex::new(3, HnswParams::default())?;
        index.insert(vec![
            VectorRecord {
                id: "doc_a".into(),
                vector: vec![0.9, 0.1, 0.0],
            },
            VectorRecord {
                id: "doc_b".into(),
                vector: vec![0.8, 0.2, 0.0],
            },
            VectorRecord {
                id: "doc_c".into(),
                vector: vec![0.1, 0.9, 0.0],
            },
        ])?;

        let query = [0.85, 0.15, 0.0];
        let via_index = index.search_with_backend(
            query.as_slice(),
            3,
            VectorSearchBackend::ExperimentalU8ThenF32Rerank,
        )?;
        let via_kernel = HnswKernel::new().search(
            &index,
            query.as_slice(),
            3,
            VectorSearchBackend::ExperimentalU8ThenF32Rerank,
        )?;
        assert_eq!(via_kernel.matches, via_index.matches);
        Ok(())
    }

    #[test]
    fn search_with_kernel_trait_object_dispatches_to_hnsw() -> Result<()> {
        let mut index = VectorIndex::new(2, HnswParams::default())?;
        index.insert(vec![
            VectorRecord {
                id: "a".into(),
                vector: vec![1.0, 0.0],
            },
            VectorRecord {
                id: "b".into(),
                vector: vec![0.0, 1.0],
            },
        ])?;

        let kernel: &dyn VectorKernel = &HnswKernel::new();
        assert_eq!(kernel.kind(), VectorKernelKind::HnswRs);

        let via_trait_object =
            index.search_with_kernel(&[1.0, 0.0], 2, kernel, VectorSearchBackend::F32Hnsw)?;
        let via_direct = index.search(&[1.0, 0.0], 2)?;
        assert_eq!(via_trait_object.matches, via_direct.matches);
        Ok(())
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

    #[test]
    fn snapshot_v2_roundtrip_restores_search() -> Result<()> {
        let temp = TempDir::create("vector-index-v2-roundtrip").map_err(ErrorEnvelope::from)?;
        let mut index = VectorIndex::new(3, HnswParams::default())?;
        index.insert(vec![
            VectorRecord {
                id: "doc_a".into(),
                vector: vec![10.0, 20.0, 30.0],
            },
            VectorRecord {
                id: "doc_b".into(),
                vector: vec![40.0, 50.0, 60.0],
            },
        ])?;

        let _meta = index.snapshot_v2(temp.path())?;
        let restored = VectorIndex::from_snapshot_v2(temp.path())?;
        let matches = restored.search(&[10.0, 20.0, 30.0], 1)?.matches;

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id.as_ref(), "doc_a");
        Ok(())
    }

    #[test]
    fn snapshot_v1_v2_parity_on_small_set() -> Result<()> {
        let v1_temp = TempDir::create("vector-index-v1-parity").map_err(ErrorEnvelope::from)?;
        let v2_temp = TempDir::create("vector-index-v2-parity").map_err(ErrorEnvelope::from)?;

        let mut index = VectorIndex::new(2, HnswParams::default())?;
        index.insert(vec![
            VectorRecord {
                id: "low".into(),
                vector: vec![1.0, 0.0],
            },
            VectorRecord {
                id: "mid".into(),
                vector: vec![100.0, 0.0],
            },
            VectorRecord {
                id: "high".into(),
                vector: vec![200.0, 0.0],
            },
        ])?;

        index.write_snapshot(v1_temp.path(), VectorSnapshotWriteVersion::V1)?;
        index.write_snapshot(v2_temp.path(), VectorSnapshotWriteVersion::V2)?;

        let v1_bytes = std::fs::read(v1_temp.path().join(SNAPSHOT_V1_FILE_NAME))
            .map_err(ErrorEnvelope::from)?;
        let v1_snapshot =
            serde_json::from_slice::<VectorSnapshot>(&v1_bytes).map_err(|source| {
                ErrorEnvelope::expected(
                    ErrorCode::new("vector", "snapshot_parse_failed"),
                    "failed to parse v1 snapshot in parity test",
                )
                .with_metadata("source", source.to_string())
            })?;
        let v1_restored = VectorIndex::from_snapshot(v1_snapshot)?;
        let v2_restored = VectorIndex::from_snapshot_v2(v2_temp.path())?;

        let v1 = v1_restored.search(&[1.0, 0.0], 3)?.matches;
        let v2 = v2_restored.search(&[1.0, 0.0], 3)?.matches;
        let v1_ids = v1.iter().map(|item| item.id.as_ref()).collect::<Vec<_>>();
        let v2_ids = v2.iter().map(|item| item.id.as_ref()).collect::<Vec<_>>();
        assert_eq!(v1_ids.first(), v2_ids.first());
        assert_eq!(v1_ids.len(), v2_ids.len());
        for id in v1_ids {
            assert!(v2_ids.contains(&id));
        }
        Ok(())
    }

    #[test]
    fn snapshot_v2_empty_index_loads() -> Result<()> {
        let temp = TempDir::create("vector-index-v2-empty").map_err(ErrorEnvelope::from)?;
        let index = VectorIndex::new(4, HnswParams::default())?;

        let _meta = index.snapshot_v2(temp.path())?;
        let restored = VectorIndex::from_snapshot_v2(temp.path())?;
        let matches = restored.search(&[0.0, 0.0, 0.0, 0.0], 10)?.matches;

        assert!(matches.is_empty());
        Ok(())
    }

    #[test]
    fn snapshot_stats_calculation_reports_expected_fields() -> Result<()> {
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

        let stats = index.snapshot_stats(VectorSnapshotWriteVersion::V2)?;
        assert_eq!(stats.version, VectorSnapshotWriteVersion::V2);
        assert_eq!(stats.dimension, 3);
        assert_eq!(stats.count, 2);
        assert_eq!(
            stats
                .metadata
                .get("files.vectors.u8.bin.bytes")
                .map(|value| value.as_ref()),
            Some("6")
        );
        let keys = stats
            .metadata
            .keys()
            .map(|key| key.as_ref())
            .collect::<Vec<_>>();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(keys, sorted);
        assert!(stats.bytes >= 6);
        Ok(())
    }

    #[test]
    fn snapshot_v2_graph_persistence_roundtrip() -> Result<()> {
        let temp = TempDir::create("vector-index-v2-graph-persist").map_err(ErrorEnvelope::from)?;
        let mut index = VectorIndex::new(3, HnswParams::default())?;
        index.insert(vec![
            VectorRecord {
                id: "alpha".into(),
                vector: vec![1.0, 0.0, 0.0],
            },
            VectorRecord {
                id: "beta".into(),
                vector: vec![0.0, 1.0, 0.0],
            },
            VectorRecord {
                id: "gamma".into(),
                vector: vec![0.0, 0.0, 1.0],
            },
        ])?;

        // Write v2 snapshot (includes graph dump).
        let _meta = index.snapshot_v2(temp.path())?;

        // Verify graph files were written.
        let graph_file = temp
            .path()
            .join(format!("{}.hnsw.graph", SNAPSHOT_V2_HNSW_GRAPH_BASENAME));
        let data_file = temp
            .path()
            .join(format!("{}.hnsw.data", SNAPSHOT_V2_HNSW_GRAPH_BASENAME));
        assert!(
            graph_file.exists(),
            "hnsw graph file should exist: {}",
            graph_file.display()
        );
        assert!(
            data_file.exists(),
            "hnsw data file should exist: {}",
            data_file.display()
        );

        // Reload from persisted graph.
        let restored = VectorIndex::from_snapshot_v2(temp.path())?;

        // Search results should match the original.
        let original_matches = index.search(&[1.0, 0.0, 0.0], 3)?.matches;
        let restored_matches = restored.search(&[1.0, 0.0, 0.0], 3)?.matches;

        assert_eq!(original_matches.len(), restored_matches.len());
        assert_eq!(
            original_matches.first().map(|m| m.id.as_ref()),
            restored_matches.first().map(|m| m.id.as_ref()),
            "top result should match"
        );

        // Also verify metadata is intact.
        assert_eq!(restored.dimension(), 3);
        assert_eq!(restored.active_count(), 3);

        Ok(())
    }

    #[test]
    fn snapshot_v2_loads_without_graph_files() -> Result<()> {
        let temp = TempDir::create("vector-index-v2-no-graph").map_err(ErrorEnvelope::from)?;
        let mut index = VectorIndex::new(3, HnswParams::default())?;
        index.insert(vec![VectorRecord {
            id: "only".into(),
            vector: vec![1.0, 2.0, 3.0],
        }])?;

        // Write v2 snapshot, then delete graph files to simulate legacy snapshot.
        let _meta = index.snapshot_v2(temp.path())?;
        let graph_file = temp
            .path()
            .join(format!("{}.hnsw.graph", SNAPSHOT_V2_HNSW_GRAPH_BASENAME));
        let data_file = temp
            .path()
            .join(format!("{}.hnsw.data", SNAPSHOT_V2_HNSW_GRAPH_BASENAME));
        std::fs::remove_file(&graph_file).ok();
        std::fs::remove_file(&data_file).ok();

        // Should fall back to full rebuild.
        let restored = VectorIndex::from_snapshot_v2(temp.path())?;
        assert_eq!(restored.active_count(), 1);
        let matches = restored.search(&[1.0, 2.0, 3.0], 1)?.matches;
        assert_eq!(matches.first().map(|m| m.id.as_ref()), Some("only"));
        Ok(())
    }

    #[test]
    fn snapshot_v2_skips_graph_dump_for_non_default_max_layer() -> Result<()> {
        let temp = TempDir::create("vector-index-v2-graph-skip").map_err(ErrorEnvelope::from)?;
        let params = HnswParams {
            max_layer: 4,
            ..HnswParams::default()
        };
        let mut index = VectorIndex::new(3, params)?;
        index.insert(vec![
            VectorRecord {
                id: "alpha".into(),
                vector: vec![1.0, 0.0, 0.0],
            },
            VectorRecord {
                id: "beta".into(),
                vector: vec![0.0, 1.0, 0.0],
            },
        ])?;

        // Snapshot should succeed even when graph dump is incompatible.
        let _meta = index.snapshot_v2(temp.path())?;
        let graph_file = temp
            .path()
            .join(format!("{}.hnsw.graph", SNAPSHOT_V2_HNSW_GRAPH_BASENAME));
        let data_file = temp
            .path()
            .join(format!("{}.hnsw.data", SNAPSHOT_V2_HNSW_GRAPH_BASENAME));
        assert!(
            !graph_file.exists(),
            "graph file should be absent when dump is skipped: {}",
            graph_file.display()
        );
        assert!(
            !data_file.exists(),
            "graph data file should be absent when dump is skipped: {}",
            data_file.display()
        );

        // Reload should still work via full rebuild from metadata+vectors.
        let restored = VectorIndex::from_snapshot_v2(temp.path())?;
        assert_eq!(restored.active_count(), 2);
        let matches = restored.search(&[1.0, 0.0, 0.0], 1)?.matches;
        assert_eq!(matches.first().map(|m| m.id.as_ref()), Some("alpha"));
        Ok(())
    }

    #[test]
    fn snapshot_size_limit_error_is_reported() -> Result<()> {
        let temp = TempDir::create("vector-index-v2-size-limit").map_err(ErrorEnvelope::from)?;
        let mut index = VectorIndex::new(3, HnswParams::default())?;
        index.insert(vec![VectorRecord {
            id: "doc_a".into(),
            vector: vec![10.0, 20.0, 30.0],
        }])?;

        let error = index
            .snapshot_v2_with_size_limit(temp.path(), Some(1))
            .err()
            .ok_or_else(|| std::io::Error::other("expected oversize snapshot error"))?;
        assert_eq!(error.code, ErrorCode::new("vector", "snapshot_oversize"));
        assert_eq!(
            error.metadata.get("maxBytes").map(String::as_str),
            Some("1")
        );
        Ok(())
    }
}
