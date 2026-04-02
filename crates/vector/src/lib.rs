//! # semantic-code-vector
//!
//! Vector indexing kernel and related APIs.
//! This crate depends only on `shared`.

use hnsw_rs::prelude::{AnnT, Distance, Hnsw, HnswIo, Neighbour};

/// Accelerate-backed BLAS distance primitives with scalar fallbacks.
pub(crate) mod accelerate;
use accelerate::{DistAccelerateCosine, with_distance_eval_tracking};
use rayon::prelude::*;
use semantic_code_shared::{CancellationToken, ErrorClass, ErrorCode, ErrorEnvelope, Result};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::instrument;

mod exact_rows;
mod generation;
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

pub use exact_rows::{
    ExactVectorRow, ExactVectorRowRef, ExactVectorRowSource, ExactVectorRowView, ExactVectorRows,
    ExactVectorRowsIter, fingerprint_exact_rows,
};
pub use generation::{
    CollectionGenerationPaths, EXACT_GENERATION_IDS_FILE_NAME, EXACT_GENERATION_META_FILE_NAME,
    EXACT_GENERATION_ORIGINS_FILE_NAME, EXACT_GENERATION_VECTORS_FILE_NAME,
    EXACT_GENERATION_VERSION_V3, ExactGenerationMeta, ExactVectorNormalization,
    GENERATION_ACTIVE_FILE_NAME, GENERATION_BASE_DIR_NAME, GENERATION_CATALOG_DB_FILE_NAME,
    GENERATION_DERIVED_DIR_NAME, GENERATION_KERNELS_DIR_NAME, GENERATIONS_DIR_NAME, GenerationId,
    PublishedGenerationPaths, read_exact_generation, write_exact_generation,
};
pub use mmap::MmapBytes;
pub use quantization::{QuantizationError, QuantizationParams, Quantizer, quantize_f32_to_u8};
pub use snapshot::{
    SNAPSHOT_V2_HNSW_GRAPH_BASENAME, SNAPSHOT_V2_META_FILE_NAME, SNAPSHOT_V2_VECTORS_FILE_NAME,
    SnapshotError, SnapshotResult, VectorSnapshotMeta, VectorSnapshotVersion, read_metadata,
    read_snapshot_ids, write_metadata, write_snapshot_ids,
};

const VECTOR_SNAPSHOT_VERSION: u32 = 1;
const VECTOR_SNAPSHOT_V2_IDS_FILE_NAME: &str = "ids.json";
const VECTOR_SNAPSHOT_V2_ORIGINS_FILE_NAME: &str = "origins.json";
const SNAPSHOT_STATS_CONTEXT: &str = "<snapshot-stats>";
const U8_RERANK_CANDIDATE_MULTIPLIER: usize = 4;
const U8_RERANK_MIN_CANDIDATES: usize = 32;
const U8_RERANK_MAX_CANDIDATES: usize = 512;
// hnsw_rs graph dumping requires the runtime max layer to match its fixed
// internal layer constant (NB_LAYER_MAX == 16 in hnsw_rs 0.3.x).
const HNSW_RS_GRAPH_DUMP_REQUIRED_MAX_LAYER: usize = 16;
static NEXT_VECTOR_INDEX_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

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

/// Stable graph/search identity used as the HNSW `origin_id`.
///
/// Origin IDs are monotonically assigned at insertion time and never reused.
/// Ascending origin order therefore preserves insertion ordering, making it
/// the canonical persistence ordering for tombstone-free snapshot writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OriginId(usize);

impl OriginId {
    /// Create an `OriginId` from a raw `usize` value.
    pub const fn from_usize(value: usize) -> Self {
        Self(value)
    }

    /// Return the raw `usize` value of this origin ID.
    pub const fn as_usize(self) -> usize {
        self.0
    }

    fn as_u64(self) -> Result<u64> {
        usize_to_u64(self.0, "origin id conversion overflow")
    }
}

/// Append-only payload slot inside `VectorIndex.records`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PayloadSlot(usize);

impl PayloadSlot {
    const fn from_usize(value: usize) -> Self {
        Self(value)
    }

    const fn as_usize(self) -> usize {
        self.0
    }
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

/// Pre-computed V2 snapshot data ready for writing.
///
/// Captures the expensive computation (record ordering, quantization fitting,
/// and batch quantization) in a single pass so callers can check size limits
/// before committing to disk I/O — without redundant re-quantization.
///
/// The quantized vector bytes are shared with [`QuantizationCache`] via
/// `Arc` to avoid cloning the full buffer (which can be tens of megabytes).
pub struct PreparedV2Snapshot {
    /// Deterministic snapshot stats (bytes, count, metadata).
    pub stats: SnapshotStats,
    /// Ordered records in canonical persistence order (ascending `origin_id`).
    records: Vec<VectorRecord>,
    /// Stable origin ids aligned with `records`.
    origins: Vec<OriginId>,
    /// Fitted quantization parameters.
    quantization: QuantizationParams,
    /// Quantized vector bytes (flattened `n × d` u8 array), shared with cache.
    vectors: Arc<Vec<u8>>,
    /// Whether the record order matches HNSW internal node order.
    graph_safe: bool,
}

impl PreparedV2Snapshot {
    /// Record IDs in the same order used for `ids.json` and `vectors.u8.bin`.
    ///
    /// Callers that write companion files (e.g. the JSONL metadata sidecar)
    /// should iterate this list to guarantee row-aligned ordering with the
    /// binary snapshot files.
    pub fn ordered_ids(&self) -> Vec<Box<str>> {
        self.records.iter().map(|r| r.id.clone()).collect()
    }
}

/// Cached quantization state for incremental snapshot preparation.
///
/// Stores the fitted quantization parameters and the number of records
/// that were quantized. On subsequent snapshots, only new records
/// (appended since the cache was built) need quantization — provided
/// the new vectors don't expand the per-dimension min/max range.
///
/// The quantized bytes are shared with [`PreparedV2Snapshot`] via `Arc`
/// to avoid a ~38 MB clone at 100K vectors × 384 dimensions.
pub struct QuantizationCache {
    /// Per-dimension SQ8 parameters from the last snapshot.
    params: QuantizationParams,
    /// Number of records that were quantized into the cache.
    record_count: usize,
    /// The quantized bytes for the first `record_count` records, shared with snapshot.
    vectors: Arc<Vec<u8>>,
    /// Whether the cached records were in graph-safe (insertion) order.
    graph_safe: bool,
}

impl QuantizationCache {
    /// Build a cache from a freshly prepared snapshot.
    ///
    /// Use this when the incremental path is not applicable (e.g. after a
    /// tombstone-reclaiming rebuild where the previous cache is invalid).
    pub fn from_prepared(prepared: &PreparedV2Snapshot) -> Self {
        Self {
            params: prepared.quantization.clone(),
            record_count: prepared.records.len(),
            vectors: Arc::clone(&prepared.vectors),
            graph_safe: prepared.graph_safe,
        }
    }
}

enum PersistableIndex<'a> {
    Current(&'a VectorIndex),
    Rebuilt(Box<VectorIndex>),
}

struct PersistableV2Snapshot<'a> {
    index: PersistableIndex<'a>,
    prepared: PreparedV2Snapshot,
}

impl PersistableV2Snapshot<'_> {
    const fn stats(&self) -> &SnapshotStats {
        &self.prepared.stats
    }

    fn write(self, snapshot_dir: &Path, kernel: VectorKernelKind) -> Result<VectorSnapshotMeta> {
        let Self { index, prepared } = self;
        match index {
            PersistableIndex::Current(index) => {
                index.write_prepared_v2_snapshot(snapshot_dir, kernel, prepared)
            },
            PersistableIndex::Rebuilt(index) => {
                index.write_prepared_v2_snapshot(snapshot_dir, kernel, prepared)
            },
        }
    }
}

/// Load options for `VectorIndex::from_snapshot_v2_with_options`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VectorSnapshotV2LoadOptions {
    /// Allow automatic v1 -> v2 upgrade when `snapshot.meta` is missing.
    pub auto_upgrade_v1: bool,
    /// Skip loading the persisted HNSW graph even when it exists on disk,
    /// forcing a full O(n log n) rebuild from the dequantized vectors.
    ///
    /// Use this when the caller's kernel differs from the snapshot kernel
    /// (e.g. loading an hnsw-rs snapshot for DFRR search). The persisted
    /// graph's internal node IDs only align with the snapshot's ID array
    /// when the write path had no deleted records, so callers that cannot
    /// guarantee this should also set this flag.
    pub skip_persisted_graph: bool,
    /// Skip building the HNSW graph entirely — populate records, ID mappings,
    /// and origin tracking but leave the graph empty.
    ///
    /// Use this for graph-agnostic kernels (e.g. `FlatScan`) that only need
    /// the raw vectors for linear scan.  Avoids an O(n log n) graph build
    /// that can take hours for large collections (383k+ vectors).
    pub skip_graph_build: bool,
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

/// Loader/runtime capabilities for a concrete kernel family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorKernelLoadCapabilities {
    /// Whether the kernel requires a host-side HNSW graph to be present in the
    /// loaded [`VectorIndex`].
    pub requires_host_hnsw_graph: bool,
    /// Whether the kernel can safely load a snapshot produced by a different
    /// kernel family without forcing a full host-graph rebuild.
    pub tolerates_snapshot_kernel_mismatch: bool,
    /// Whether the kernel supports an explicit warm/materialization lifecycle
    /// with persisted ready-state beyond the base collection snapshot.
    pub supports_kernel_ready_state: bool,
}

impl VectorKernelKind {
    /// Runtime capabilities for this kernel family.
    #[must_use]
    pub const fn load_capabilities(self) -> VectorKernelLoadCapabilities {
        match self {
            Self::HnswRs => VectorKernelLoadCapabilities {
                requires_host_hnsw_graph: true,
                tolerates_snapshot_kernel_mismatch: false,
                supports_kernel_ready_state: false,
            },
            Self::Dfrr => VectorKernelLoadCapabilities {
                requires_host_hnsw_graph: false,
                tolerates_snapshot_kernel_mismatch: true,
                supports_kernel_ready_state: true,
            },
            Self::FlatScan => VectorKernelLoadCapabilities {
                requires_host_hnsw_graph: false,
                tolerates_snapshot_kernel_mismatch: true,
                supports_kernel_ready_state: false,
            },
        }
    }
}

/// Process-local key that identifies one live [`VectorIndex`] state revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorIndexStateKey {
    /// Stable per-index instance identifier for the current process.
    pub instance_id: u64,
    /// Monotonic revision for mutations applied to that live instance.
    pub revision: u64,
}

/// Context passed to [`VectorKernel::warm`] for collection-scoped
/// materialization and persisted ready-state restore.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorKernelWarmContext {
    /// Stable caller-provided collection identity for diagnostics and
    /// collection-scoped cache cleanup.
    pub collection_identity: Box<str>,
    /// Snapshot directory that owns the persisted collection state, when one
    /// exists for this collection.
    pub snapshot_dir: Option<std::path::PathBuf>,
    /// Whether the warmed state is allowed to be written back beside the base
    /// collection snapshot. Callers should disable this when the live index has
    /// diverged from the on-disk collection state (for example after WAL replay
    /// but before the next checkpoint).
    pub allow_persist: bool,
}

impl VectorKernelWarmContext {
    /// Build a warm context for one collection state.
    #[must_use]
    pub fn new(
        collection_identity: impl Into<Box<str>>,
        snapshot_dir: Option<std::path::PathBuf>,
        allow_persist: bool,
    ) -> Self {
        Self {
            collection_identity: collection_identity.into(),
            snapshot_dir,
            allow_persist,
        }
    }
}

/// One row to materialize into a derived V2 HNSW snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotSubsetRow {
    /// Record ID to copy from the source snapshot.
    pub id: Box<str>,
    /// Durable origin ID to preserve in the derived snapshot.
    pub origin: OriginId,
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

    /// Provide a snapshot directory for persisting/restoring kernel state.
    ///
    /// Kernels that support state persistence (e.g., DFRR) will attempt to
    /// restore cached state from this directory on first access, avoiding
    /// expensive cold-start rebuilds. If no cached state is found, the
    /// kernel builds from scratch and persists the result.
    ///
    /// Default implementation is a no-op — kernels that do not support
    /// persistence simply ignore this call.
    fn set_snapshot_dir(&self, _dir: &Path) {}

    /// Eagerly materialize any kernel-specific ready-state for this collection.
    ///
    /// The default implementation is a no-op for kernels whose search path uses
    /// only the base [`VectorIndex`] data (e.g. host HNSW or flat scan).
    fn warm(
        &self,
        _index: &VectorIndex,
        _context: &VectorKernelWarmContext,
        _cancellation: Option<&CancellationToken>,
    ) -> Result<()> {
        Ok(())
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
        let (matches, expansions, extra) = match backend {
            VectorSearchBackend::F32Hnsw => {
                let (matches, expansions, ef_search) =
                    index.search_f32_hnsw(query, limit, self.ef_search_override)?;
                let mut extra = BTreeMap::new();
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "ef_search is a small tuning parameter (typically <10_000); f64 is exact up to 2^53"
                )]
                let ef_search_f64 = ef_search as f64;
                extra.insert("efSearch".into(), ef_search_f64);
                (matches, Some(expansions), extra)
            },
            VectorSearchBackend::ExperimentalU8Quantized => (
                index.search_u8_quantized(query, limit)?,
                None,
                BTreeMap::new(),
            ),
            VectorSearchBackend::ExperimentalU8ThenF32Rerank => (
                Self::search_u8_then_f32_rerank(index, query, limit)?,
                None,
                BTreeMap::new(),
            ),
        };
        let kernel_search_duration_ns = u64::try_from(start.elapsed().as_nanos()).ok();

        Ok(VectorSearchOutput {
            matches,
            stats: KernelSearchStats {
                expansions,
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
        let (matches, _, _) = index.search_f32_hnsw(query, limit, None)?;
        Ok(matches)
    }
}

impl ExactVectorRowSource for VectorIndex {
    type Row<'a>
        = ExactVectorRowRef<'a>
    where
        Self: 'a;

    type Iter<'a>
        = std::vec::IntoIter<ExactVectorRowRef<'a>>
    where
        Self: 'a;

    fn dimension(&self) -> u32 {
        self.dimension
    }

    fn row_count(&self) -> usize {
        self.active_count()
    }

    fn rows(&self) -> Self::Iter<'_> {
        self.active_entries_by_origin()
            .into_iter()
            .map(|(origin, record)| {
                ExactVectorRowRef::new(origin, record.id.as_ref(), record.vector.as_slice())
            })
            .collect::<Vec<ExactVectorRowRef<'_>>>()
            .into_iter()
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
            .filter(|(idx, _)| !index.deleted_slots.contains(&PayloadSlot::from_usize(*idx)))
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
    slot_origins: Vec<OriginId>,
    id_to_origin: HashMap<Box<str>, OriginId>,
    origin_to_slot: Vec<Option<PayloadSlot>>,
    deleted_slots: HashSet<PayloadSlot>,
    next_origin_id: usize,
    instance_id: u64,
    state_revision: u64,
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
            slot_origins: Vec::new(),
            id_to_origin: HashMap::new(),
            origin_to_slot: Vec::new(),
            deleted_slots: HashSet::new(),
            next_origin_id: 0,
            instance_id: NEXT_VECTOR_INDEX_INSTANCE_ID.fetch_add(1, Ordering::Relaxed),
            state_revision: 0,
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

    /// Return the current live-state key for this index instance.
    #[must_use]
    pub const fn state_key(&self) -> VectorIndexStateKey {
        VectorIndexStateKey {
            instance_id: self.instance_id,
            revision: self.state_revision,
        }
    }

    /// Stable fingerprint of the active collection payload in canonical
    /// origin-order. Used to validate persisted kernel-ready sidecars.
    #[must_use]
    pub fn state_fingerprint(&self) -> u64 {
        fingerprint_exact_rows(
            self.dimension,
            self.active_count(),
            ExactVectorRowSource::rows(self),
        )
    }

    /// Return the number of active (non-deleted) records.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.id_to_origin.len()
    }

    /// Host-side HNSW node count.
    ///
    /// This is primarily a diagnostics/debugging hook used by higher-level
    /// loaders and tests to distinguish records-only collection loads from
    /// loads that materialize the host HNSW graph.
    #[must_use]
    pub fn host_hnsw_count(&self) -> usize {
        self.hnsw.get_nb_point()
    }

    /// Return active entries in ascending `OriginId` order, skipping tombstoned slots.
    ///
    /// Origin IDs are monotonically assigned at insertion time, so ascending
    /// origin order preserves insertion ordering naturally. This is the
    /// canonical persistence ordering for tombstone-free snapshot writes.
    #[must_use]
    pub fn active_entries_by_origin(&self) -> Vec<(OriginId, &VectorRecord)> {
        let mut entries: Vec<(OriginId, &VectorRecord)> = self
            .slot_origins
            .iter()
            .enumerate()
            .filter(|&(slot_idx, _)| {
                !self
                    .deleted_slots
                    .contains(&PayloadSlot::from_usize(slot_idx))
            })
            .filter_map(|(slot_idx, &origin)| {
                self.records.get(slot_idx).map(|record| (origin, record))
            })
            .collect();
        entries.sort_by_key(|(origin, _)| origin.as_usize());
        entries
    }

    /// Materialize the active collection payload as exact canonical rows.
    ///
    /// The returned rows are ordered by ascending durable origin and are ready
    /// for staging or publication into kernel-neutral base generations.
    pub fn exact_rows(&self) -> Result<ExactVectorRows> {
        let rows = self
            .active_entries_by_origin()
            .into_iter()
            .map(|(origin, record)| {
                ExactVectorRow::new(record.id.clone(), origin, record.vector.clone())
            })
            .collect::<Vec<ExactVectorRow>>();
        ExactVectorRows::new(self.dimension, rows)
    }

    /// Return `true` if any slots have been tombstoned (deleted or superseded by upsert).
    #[must_use]
    pub fn has_tombstones(&self) -> bool {
        !self.deleted_slots.is_empty()
    }

    /// Build a fresh `VectorIndex` from active entries with a clean HNSW graph.
    ///
    /// `hnsw_rs` has no public delete/prune API, so tombstone reclamation
    /// requires building a fresh index. The rebuilt index has no
    /// `deleted_slots`, so `graph_safe` is naturally true — enabling HNSW
    /// graph persistence.
    pub fn rebuild_active_index(&self, cancellation: Option<&CancellationToken>) -> Result<Self> {
        let entries = self.active_entries_by_origin();
        let records: Vec<(OriginId, VectorRecord)> = entries
            .into_iter()
            .map(|(origin, record)| (origin, record.clone()))
            .collect();
        let mut params = self.params;
        params.max_elements = params.max_elements.max(records.len().max(1));
        let mut new_index = Self::new(self.dimension, params)?;
        new_index.insert_with_origins(records, cancellation)?;
        Ok(new_index)
    }

    /// Insert records while skipping host-HNSW graph construction.
    ///
    /// This is intended for kernels that only need the collection payload and
    /// manage their own search-ready state (for example DFRR or exact flat
    /// scan). Origin IDs are still assigned monotonically so persistence and
    /// higher-level metadata remain stable.
    pub fn insert_records_without_graph(&mut self, records: Vec<VectorRecord>) -> Result<()> {
        let mut records_with_origins = Vec::with_capacity(records.len());
        for record in records {
            let origin = self.allocate_origin_id()?;
            records_with_origins.push((origin, record));
        }
        self.insert_records_only(records_with_origins)
    }

    /// Insert records with caller-assigned origin IDs while building the host
    /// HNSW graph.
    ///
    /// This is used by snapshot/materialization helpers that must preserve an
    /// existing durable origin mapping instead of allocating fresh origin IDs.
    pub fn insert_with_assigned_origins(
        &mut self,
        records: Vec<(OriginId, VectorRecord)>,
        cancellation: Option<&CancellationToken>,
    ) -> Result<()> {
        self.insert_with_origins(records, cancellation)
    }

    /// Insert records with caller-assigned origin IDs without building the
    /// host HNSW graph.
    ///
    /// This is the rebuild-grade records-only twin of
    /// [`insert_with_assigned_origins`](Self::insert_with_assigned_origins),
    /// used when the caller must preserve durable origin identity while
    /// deferring graph construction to a later kernel-specific phase.
    pub fn insert_records_without_graph_with_assigned_origins(
        &mut self,
        records: Vec<(OriginId, VectorRecord)>,
    ) -> Result<()> {
        self.insert_records_only(records)
    }

    /// Insert or update records in the index.
    ///
    /// Record bookkeeping remains batched, but HNSW graph insertion is kept
    /// sequential.
    ///
    /// `hnsw_rs::parallel_insert_slice` regressed exact self-hit identity in
    /// fresh multi-batch builds, which then poisoned persisted-graph
    /// roundtrips. The graph must preserve the caller-assigned `origin_id` →
    /// record-index mapping, so correctness takes precedence over the broken
    /// parallel insertion optimization.
    #[instrument(
        name = "vector.index.insert_batch",
        skip_all,
        fields(dimension = self.dimension, record_count = records.len())
    )]
    pub fn insert(&mut self, records: Vec<VectorRecord>) -> Result<()> {
        // Phase 1: sequential bookkeeping — validate dimensions, prepare
        // vectors for cosine distance, assign HNSW node indices, and
        // track upserts. This must be sequential because it mutates
        // self.records, self.id_to_origin, and self.deleted_slots.
        let mut hnsw_batch: Vec<(OriginId, PayloadSlot)> = Vec::with_capacity(records.len());

        for record in records {
            ensure_dimension(self.dimension, &record.vector)?;
            prepare_vector_for_cosine(
                record.vector.as_slice(),
                self.params.min_norm_squared,
                "insert",
            )
            .map_err(|error| error.with_metadata("id", record.id.to_string()))?;

            let origin = self.allocate_origin_id()?;
            let slot = self.push_record_with_origin(origin, record);
            hnsw_batch.push((origin, slot));
        }

        // Phase 2: HNSW graph insertion. Build the batch of (slice, id)
        // pairs referencing the vectors now owned by self.records.
        let insert_pairs: Vec<(&[f32], usize)> = hnsw_batch
            .iter()
            .filter_map(|&(origin, slot)| {
                self.records
                    .get(slot.as_usize())
                    .map(|record| (record.vector.as_slice(), origin.as_usize()))
            })
            .collect();

        for &(vector, index) in &insert_pairs {
            self.hnsw.insert_slice((vector, index));
        }

        self.bump_state_revision();

        Ok(())
    }

    fn insert_with_origins(
        &mut self,
        records: Vec<(OriginId, VectorRecord)>,
        cancellation: Option<&CancellationToken>,
    ) -> Result<()> {
        let mut hnsw_batch: Vec<(OriginId, PayloadSlot)> = Vec::with_capacity(records.len());

        for (origin, record) in records {
            ensure_dimension(self.dimension, &record.vector)?;
            prepare_vector_for_cosine(
                record.vector.as_slice(),
                self.params.min_norm_squared,
                "insert",
            )
            .map_err(|error| error.with_metadata("id", record.id.to_string()))?;

            let slot = self.push_record_with_origin(origin, record);
            hnsw_batch.push((origin, slot));
        }

        for (batch_idx, &(origin, slot)) in hnsw_batch.iter().enumerate() {
            if batch_idx % 1000 == 0
                && let Some(token) = cancellation
                && token.is_cancelled()
            {
                return Err(ErrorEnvelope::cancelled(
                    "HNSW graph construction cancelled",
                ));
            }
            let vector = self
                .records
                .get(slot.as_usize())
                .ok_or_else(|| {
                    ErrorEnvelope::unexpected(
                        ErrorCode::new("vector", "record_slot_missing"),
                        "record slot missing during HNSW rebuild",
                        ErrorClass::NonRetriable,
                    )
                })?
                .vector
                .as_slice();
            self.hnsw.insert_slice((vector, origin.as_usize()));
        }

        self.bump_state_revision();

        Ok(())
    }

    /// Populate records, ID mappings, and origin tracking **without** building
    /// the HNSW graph.  The graph is left empty — only graph-agnostic kernels
    /// (e.g. `FlatScan`) can search the resulting index.
    fn insert_records_only(&mut self, records: Vec<(OriginId, VectorRecord)>) -> Result<()> {
        for (origin, record) in records {
            ensure_dimension(self.dimension, &record.vector)?;
            prepare_vector_for_cosine(
                record.vector.as_slice(),
                self.params.min_norm_squared,
                "insert",
            )
            .map_err(|error| error.with_metadata("id", record.id.to_string()))?;

            self.push_record_with_origin(origin, record);
        }
        self.bump_state_revision();
        Ok(())
    }

    fn allocate_origin_id(&mut self) -> Result<OriginId> {
        let value = self.next_origin_id;
        self.next_origin_id = self.next_origin_id.checked_add(1).ok_or_else(|| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "origin_id_overflow"),
                "origin id overflow",
                ErrorClass::NonRetriable,
            )
        })?;
        Ok(OriginId::from_usize(value))
    }

    fn bind_origin_to_slot(&mut self, origin: OriginId, slot: PayloadSlot) {
        if self.origin_to_slot.len() <= origin.as_usize() {
            self.origin_to_slot.resize(origin.as_usize() + 1, None);
        }
        if let Some(binding) = self.origin_to_slot.get_mut(origin.as_usize()) {
            *binding = Some(slot);
        }
    }

    fn push_record_with_origin(&mut self, origin: OriginId, record: VectorRecord) -> PayloadSlot {
        let slot = PayloadSlot::from_usize(self.records.len());
        self.next_origin_id = self.next_origin_id.max(origin.as_usize().saturating_add(1));
        if let Some(previous_origin) = self.id_to_origin.insert(record.id.clone(), origin) {
            self.retire_origin(previous_origin);
        }
        self.slot_origins.push(origin);
        self.bind_origin_to_slot(origin, slot);
        self.records.push(record);
        slot
    }

    const fn bump_state_revision(&mut self) {
        self.state_revision = self.state_revision.saturating_add(1);
    }

    fn retire_origin(&mut self, origin: OriginId) {
        if let Some(slot) = self.slot_for_origin(origin) {
            self.deleted_slots.insert(slot);
        }
        if let Some(binding) = self.origin_to_slot.get_mut(origin.as_usize()) {
            *binding = None;
        }
    }

    fn slot_for_origin(&self, origin: OriginId) -> Option<PayloadSlot> {
        self.origin_to_slot
            .get(origin.as_usize())
            .and_then(|slot| *slot)
    }

    fn record_for_origin(&self, origin: OriginId) -> Option<&VectorRecord> {
        self.slot_for_origin(origin)
            .and_then(|slot| self.records.get(slot.as_usize()))
    }

    /// Delete records by external id (best-effort).
    pub fn delete(&mut self, ids: &[Box<str>]) -> Result<()> {
        for id in ids {
            if let Some(origin) = self.id_to_origin.remove(id.as_ref()) {
                self.retire_origin(origin);
            }
        }
        self.bump_state_revision();
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
    ) -> Result<(Vec<VectorMatch>, u64, usize)> {
        if self.records.is_empty() || limit == 0 {
            return Ok((Vec::new(), 0, 0));
        }
        let query = prepare_search_query(self.dimension, self.params.min_norm_squared, query)?;

        let total = self.records.len();
        let requested = limit.min(total);
        let knbn = requested;
        let base_ef = ef_search_override.unwrap_or(self.params.ef_search);
        let ef_search = base_ef.max(knbn);

        let (matches, expansions) = with_distance_eval_tracking(|| {
            let neighbours = self.hnsw.search(query.as_ref(), knbn, ef_search);
            let mut matches = to_matches(&self.records, &self.origin_to_slot, neighbours);
            self.fill_shortfall_with_exact_scan(query.as_ref(), requested, &mut matches);

            sort_matches_by_score_then_id(matches.as_mut_slice());
            matches.truncate(requested);
            matches
        });
        Ok((matches, expansions, ef_search))
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
        let (matches, _, _) = self.search_f32_hnsw(query, limit, None)?;
        Ok(matches)
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
            if self.deleted_slots.contains(&PayloadSlot::from_usize(index)) {
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
        self.id_to_origin
            .get(id)
            .copied()
            .and_then(|origin| self.record_for_origin(origin))
    }

    /// Export the index into a snapshot.
    #[must_use]
    pub fn snapshot(&self) -> VectorSnapshot {
        let (records, _origins) = self.snapshot_records_and_origins();
        VectorSnapshot {
            version: VECTOR_SNAPSHOT_VERSION,
            dimension: self.dimension,
            params: self.params,
            records,
        }
    }

    /// Compute deterministic snapshot stats for the selected format.
    pub fn snapshot_stats(&self, version: VectorSnapshotWriteVersion) -> Result<SnapshotStats> {
        let (records, origins) = self.snapshot_records_and_origins();
        self.snapshot_stats_for_records(version, records.as_slice(), origins.as_slice())
    }

    fn snapshot_stats_for_records(
        &self,
        version: VectorSnapshotWriteVersion,
        records: &[VectorRecord],
        origins: &[OriginId],
    ) -> Result<SnapshotStats> {
        match version {
            VectorSnapshotWriteVersion::V1 => self.snapshot_stats_v1(records),
            VectorSnapshotWriteVersion::V2 => self.snapshot_stats_v2(records, origins),
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

    fn snapshot_stats_v2(
        &self,
        records: &[VectorRecord],
        origins: &[OriginId],
    ) -> Result<SnapshotStats> {
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
        let origins_payload = serde_json::to_vec(origins).map_err(|source| {
            ErrorEnvelope::expected(
                ErrorCode::new("vector", "snapshot_origins_serialize_failed"),
                "failed to serialize v2 snapshot origins for stats",
            )
            .with_metadata("source", source.to_string())
        })?;
        let origins_bytes = usize_to_u64(origins_payload.len(), "snapshot origins size overflow")?;
        let vectors_bytes = usize_to_u64(vectors.len(), "snapshot vectors size overflow")?;
        let total_bytes = add_u64_checked(
            add_u64_checked(
                add_u64_checked(vectors_bytes, meta_bytes, "snapshot stats size overflow")?,
                ids_bytes,
                "snapshot stats size overflow",
            )?,
            origins_bytes,
            "snapshot stats size overflow",
        )?;

        let mut metadata = BTreeMap::new();
        metadata.insert("fileCount".into(), "4".into());
        metadata.insert("files.ids.json.bytes".into(), ids_bytes.to_string().into());
        metadata.insert(
            "files.origins.json.bytes".into(),
            origins_bytes.to_string().into(),
        );
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

    /// Pre-compute all V2 snapshot data in a single pass.
    ///
    /// Performs record ordering, quantization fitting, and batch quantization
    /// exactly once. The returned [`PreparedV2Snapshot`] can then be checked
    /// against a size limit before writing via [`Self::write_prepared_v2_snapshot`].
    pub fn prepare_v2_snapshot(&self) -> Result<PreparedV2Snapshot> {
        let snapshot_dir = Path::new(SNAPSHOT_STATS_CONTEXT);
        let (records, origins) = self.snapshot_records_and_origins();
        let graph_safe = self.deleted_slots.is_empty();
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

        let count = usize_to_u64(records.len(), "snapshot record count conversion overflow")?;
        let vectors_crc32 = compute_vectors_crc32(vectors.as_slice());
        let meta = VectorSnapshotMeta::new(
            self.dimension,
            count,
            self.params,
            quantization.clone(),
            vectors_crc32,
        )
        .map_err(|source| map_snapshot_error(snapshot_dir, &source, "build v2 stats metadata"))?;
        let meta_bytes = encode_metadata(&meta)
            .map_err(|source| map_snapshot_error(snapshot_dir, &source, "encode v2 stats metadata"))
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
        let origins_payload = serde_json::to_vec(origins.as_slice()).map_err(|source| {
            ErrorEnvelope::expected(
                ErrorCode::new("vector", "snapshot_origins_serialize_failed"),
                "failed to serialize v2 snapshot origins for stats",
            )
            .with_metadata("source", source.to_string())
        })?;
        let origins_bytes = usize_to_u64(origins_payload.len(), "snapshot origins size overflow")?;
        let vectors_bytes = usize_to_u64(vectors.len(), "snapshot vectors size overflow")?;
        let total_bytes = add_u64_checked(
            add_u64_checked(
                add_u64_checked(vectors_bytes, meta_bytes, "snapshot stats size overflow")?,
                ids_bytes,
                "snapshot stats size overflow",
            )?,
            origins_bytes,
            "snapshot stats size overflow",
        )?;

        let mut metadata = BTreeMap::new();
        metadata.insert("fileCount".into(), "4".into());
        metadata.insert("files.ids.json.bytes".into(), ids_bytes.to_string().into());
        metadata.insert(
            "files.origins.json.bytes".into(),
            origins_bytes.to_string().into(),
        );
        metadata.insert(
            "files.snapshot.meta.bytes".into(),
            meta_bytes.to_string().into(),
        );
        metadata.insert(
            "files.vectors.u8.bin.bytes".into(),
            vectors_bytes.to_string().into(),
        );

        let stats = SnapshotStats {
            version: VectorSnapshotWriteVersion::V2,
            dimension: self.dimension,
            count,
            bytes: total_bytes,
            metadata,
        };

        Ok(PreparedV2Snapshot {
            stats,
            records,
            origins,
            quantization,
            vectors: Arc::new(vectors),
            graph_safe,
        })
    }

    fn prepare_persistable_v2_snapshot(&self) -> Result<PersistableV2Snapshot<'_>> {
        if self.has_tombstones() {
            let rebuilt = Box::new(self.rebuild_active_index(None)?);
            let prepared = rebuilt.prepare_v2_snapshot()?;
            Ok(PersistableV2Snapshot {
                index: PersistableIndex::Rebuilt(rebuilt),
                prepared,
            })
        } else {
            let prepared = self.prepare_v2_snapshot()?;
            Ok(PersistableV2Snapshot {
                index: PersistableIndex::Current(self),
                prepared,
            })
        }
    }

    /// Pre-compute V2 snapshot data, reusing a prior cache when possible.
    ///
    /// When the cache is valid (same record ordering, no deletions, and new
    /// vectors don't expand the per-dimension min/max range), only the new
    /// records since the cache was built are quantized. Returns the prepared
    /// snapshot and an updated cache for the next call.
    pub fn prepare_v2_snapshot_incremental(
        &self,
        cache: Option<&QuantizationCache>,
    ) -> Result<(PreparedV2Snapshot, QuantizationCache)> {
        let graph_safe = self.deleted_slots.is_empty();

        // Cache hit conditions: no deletions, same ordering mode, and
        // strictly more records than last time (append-only growth).
        let reusable_cache =
            cache.filter(|c| graph_safe && c.graph_safe && c.record_count <= self.records.len());

        let (prepared, new_cache) = if let Some(cached) = reusable_cache {
            self.prepare_v2_incremental_inner(cached)?
        } else {
            let prepared = self.prepare_v2_snapshot()?;
            let new_cache = QuantizationCache {
                params: prepared.quantization.clone(),
                record_count: prepared.records.len(),
                vectors: prepared.vectors.clone(),
                graph_safe: prepared.graph_safe,
            };
            (prepared, new_cache)
        };

        Ok((prepared, new_cache))
    }

    /// Inner incremental quantization path.
    ///
    /// Assumes the caller verified the cache is valid for reuse.
    fn prepare_v2_incremental_inner(
        &self,
        cached: &QuantizationCache,
    ) -> Result<(PreparedV2Snapshot, QuantizationCache)> {
        let snapshot_dir = Path::new(SNAPSHOT_STATS_CONTEXT);
        // In graph-safe mode, records are in insertion order — new records
        // are a suffix of self.records starting at cached.record_count.
        let records = self.records.clone();
        let origins = self.slot_origins.clone();
        let new_records = records.get(cached.record_count..).ok_or_else(|| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "cache_invariant_violation"),
                "cached record_count exceeds current record count — cache was not validated before calling prepare_v2_incremental_inner",
                ErrorClass::NonRetriable,
            )
        })?;

        if new_records.is_empty() {
            // No new records — reuse cache entirely. The Vec::clone here
            // is required because the cache is borrowed, but this path is
            // only hit when no vectors were added since the last checkpoint.
            return self.build_prepared_from_parts(
                records,
                origins,
                cached.params.clone(),
                (*cached.vectors).clone(),
                true,
            );
        }

        // Check if new records expand the per-dimension range.
        let new_dataset = new_records
            .iter()
            .map(|r| r.vector.as_slice())
            .collect::<Vec<_>>();
        let new_params = fit_min_max(new_dataset.as_slice()).map_err(|source| {
            map_quantization_error(snapshot_dir, &source, "fit quantization for new records")
        })?;

        let range_expanded = cached
            .params
            .zeros()
            .iter()
            .zip(new_params.zeros().iter())
            .any(|(cached_zero, new_zero)| new_zero < cached_zero)
            || cached
                .params
                .scales()
                .iter()
                .zip(cached.params.zeros().iter())
                .zip(new_params.scales().iter().zip(new_params.zeros().iter()))
                .any(|((c_scale, c_zero), (n_scale, n_zero))| {
                    // cached max = c_zero + c_scale * 255
                    // new max    = n_zero + n_scale * 255
                    let cached_max = c_zero + c_scale * 255.0;
                    let new_max = n_zero + n_scale * 255.0;
                    new_max > cached_max
                });

        if range_expanded {
            // Range expanded — must re-quantize everything with new params.
            let prepared = self.prepare_v2_snapshot()?;
            let new_cache = QuantizationCache {
                params: prepared.quantization.clone(),
                record_count: prepared.records.len(),
                vectors: prepared.vectors.clone(),
                graph_safe: true,
            };
            return Ok((prepared, new_cache));
        }

        // Range fits — quantize only new records with cached params.
        let quantizer = Quantizer::new(cached.params.clone()).map_err(|source| {
            map_quantization_error(snapshot_dir, &source, "build quantizer from cache")
        })?;
        let new_vectors = quantizer
            .quantize_batch(new_dataset.as_slice())
            .map_err(|source| {
                map_quantization_error(snapshot_dir, &source, "quantize new vectors incrementally")
            })?;

        let mut all_vectors = Vec::with_capacity(cached.vectors.len() + new_vectors.len());
        all_vectors.extend_from_slice(&cached.vectors);
        all_vectors.extend_from_slice(&new_vectors);

        self.build_prepared_from_parts(records, origins, cached.params.clone(), all_vectors, true)
    }

    /// Assemble a [`PreparedV2Snapshot`] and [`QuantizationCache`] from
    /// pre-computed parts (shared by full and incremental paths).
    ///
    /// The `vectors` buffer is wrapped in `Arc` and shared between the
    /// prepared snapshot and the cache, avoiding a ~38 MB clone at scale.
    fn build_prepared_from_parts(
        &self,
        records: Vec<VectorRecord>,
        origins: Vec<OriginId>,
        quantization: QuantizationParams,
        vectors: Vec<u8>,
        graph_safe: bool,
    ) -> Result<(PreparedV2Snapshot, QuantizationCache)> {
        let snapshot_dir = Path::new(SNAPSHOT_STATS_CONTEXT);
        let count = usize_to_u64(records.len(), "snapshot record count conversion overflow")?;
        let vectors_crc32 = compute_vectors_crc32(vectors.as_slice());
        let meta = VectorSnapshotMeta::new(
            self.dimension,
            count,
            self.params,
            quantization.clone(),
            vectors_crc32,
        )
        .map_err(|source| map_snapshot_error(snapshot_dir, &source, "build v2 stats metadata"))?;
        let meta_bytes = encode_metadata(&meta)
            .map_err(|source| map_snapshot_error(snapshot_dir, &source, "encode v2 stats metadata"))
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
        let origins_payload = serde_json::to_vec(origins.as_slice()).map_err(|source| {
            ErrorEnvelope::expected(
                ErrorCode::new("vector", "snapshot_origins_serialize_failed"),
                "failed to serialize v2 snapshot origins for stats",
            )
            .with_metadata("source", source.to_string())
        })?;
        let origins_bytes = usize_to_u64(origins_payload.len(), "snapshot origins size overflow")?;
        let vectors_bytes = usize_to_u64(vectors.len(), "snapshot vectors size overflow")?;
        let total_bytes = add_u64_checked(
            add_u64_checked(
                add_u64_checked(vectors_bytes, meta_bytes, "snapshot stats size overflow")?,
                ids_bytes,
                "snapshot stats size overflow",
            )?,
            origins_bytes,
            "snapshot stats size overflow",
        )?;

        let mut metadata = BTreeMap::new();
        metadata.insert("fileCount".into(), "4".into());
        metadata.insert("files.ids.json.bytes".into(), ids_bytes.to_string().into());
        metadata.insert(
            "files.origins.json.bytes".into(),
            origins_bytes.to_string().into(),
        );
        metadata.insert(
            "files.snapshot.meta.bytes".into(),
            meta_bytes.to_string().into(),
        );
        metadata.insert(
            "files.vectors.u8.bin.bytes".into(),
            vectors_bytes.to_string().into(),
        );

        let stats = SnapshotStats {
            version: VectorSnapshotWriteVersion::V2,
            dimension: self.dimension,
            count,
            bytes: total_bytes,
            metadata,
        };

        let shared_vectors = Arc::new(vectors);
        let cache = QuantizationCache {
            params: quantization.clone(),
            record_count: records.len(),
            vectors: Arc::clone(&shared_vectors),
            graph_safe,
        };

        Ok((
            PreparedV2Snapshot {
                stats,
                records,
                origins,
                quantization,
                vectors: shared_vectors,
                graph_safe,
            },
            cache,
        ))
    }

    /// Write a pre-computed V2 snapshot to disk.
    ///
    /// Accepts a [`PreparedV2Snapshot`] (from [`Self::prepare_v2_snapshot`])
    /// and writes it without any re-quantization.
    pub fn write_prepared_v2_snapshot(
        &self,
        snapshot_dir: impl AsRef<Path>,
        kernel: VectorKernelKind,
        prepared: PreparedV2Snapshot,
    ) -> Result<VectorSnapshotMeta> {
        let snapshot_dir = snapshot_dir.as_ref();
        let count = usize_to_u64(
            prepared.records.len(),
            "snapshot record count conversion overflow",
        )?;

        let meta = write_snapshot_v2_with_kernel(
            snapshot_dir,
            self.params,
            kernel,
            prepared.quantization,
            count,
            &prepared.vectors,
        )
        .map_err(|source| {
            map_snapshot_error(snapshot_dir, &source, "write v2 snapshot metadata+vectors")
        })?;

        let ids = prepared
            .records
            .into_iter()
            .map(|record| record.id)
            .collect::<Vec<_>>();
        write_snapshot_v2_ids(snapshot_dir, ids.as_slice())?;
        write_snapshot_v2_origins(snapshot_dir, prepared.origins.as_slice())?;

        if kernel.load_capabilities().requires_host_hnsw_graph
            && prepared.graph_safe
            && self.active_count() > 0
        {
            if let Err(error) = self.write_hnsw_ready_state(snapshot_dir) {
                tracing::warn!(
                    snapshot_dir = %snapshot_dir.display(),
                    %error,
                    "persisted HNSW graph dump failed; continuing with metadata+vectors only"
                );
            }
        } else {
            // Stale graph files from a previous checkpoint would be loaded on
            // the next `from_snapshot_v2_with_options` call even though the
            // record set has changed (upserts created deleted slots, making
            // the origin→slot mapping inconsistent with the persisted graph
            // topology).  Remove them so the load path falls through to a
            // full HNSW rebuild from the current vectors.
            let graph_file =
                snapshot_dir.join(format!("{SNAPSHOT_V2_HNSW_GRAPH_BASENAME}.hnsw.graph"));
            let data_file =
                snapshot_dir.join(format!("{SNAPSHOT_V2_HNSW_GRAPH_BASENAME}.hnsw.data"));
            if graph_file.exists() || data_file.exists() {
                tracing::info!(
                    snapshot_dir = %snapshot_dir.display(),
                    "removing stale persisted HNSW graph files (graph_safe=false)"
                );
                let _ = std::fs::remove_file(&graph_file);
                let _ = std::fs::remove_file(&data_file);
            }
        }

        Ok(meta)
    }

    /// Persist the host HNSW ready-state into an explicit ready-state
    /// directory.
    ///
    /// This is the kernel-private twin of the legacy root-level graph dump
    /// path used by v2 snapshots. Callers should point this at a dedicated
    /// `kernels/hnsw-rs/` directory under a published generation.
    pub fn write_hnsw_ready_state(&self, ready_state_dir: impl AsRef<Path>) -> Result<()> {
        let ready_state_dir = ready_state_dir.as_ref();
        std::fs::create_dir_all(ready_state_dir).map_err(|source| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "hnsw_ready_state_dir_create_failed"),
                "failed to create hnsw ready-state directory",
                ErrorClass::NonRetriable,
            )
            .with_metadata("readyStateDir", ready_state_dir.display().to_string())
            .with_metadata("source", source.to_string())
        })?;

        let max_layer = self.hnsw.get_max_level();
        if max_layer != HNSW_RS_GRAPH_DUMP_REQUIRED_MAX_LAYER {
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("vector", "hnsw_ready_state_max_layer_incompatible"),
                "runtime max_layer is incompatible with hnsw_rs dump format",
            )
            .with_metadata("readyStateDir", ready_state_dir.display().to_string())
            .with_metadata("maxLayer", max_layer.to_string())
            .with_metadata(
                "requiredMaxLayer",
                HNSW_RS_GRAPH_DUMP_REQUIRED_MAX_LAYER.to_string(),
            ));
        }

        self.hnsw
            .file_dump(ready_state_dir, SNAPSHOT_V2_HNSW_GRAPH_BASENAME)
            .map(|_| ())
            .map_err(|source| {
                ErrorEnvelope::unexpected(
                    ErrorCode::new("vector", "hnsw_ready_state_write_failed"),
                    format!("failed to persist HNSW ready state: {source}"),
                    ErrorClass::NonRetriable,
                )
                .with_metadata("readyStateDir", ready_state_dir.display().to_string())
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
    ///
    /// Delegates to [`Self::prepare_v2_snapshot`] +
    /// [`Self::write_prepared_v2_snapshot`] so quantization runs exactly once.
    pub fn snapshot_v2_with_kernel_and_size_limit(
        &self,
        snapshot_dir: impl AsRef<Path>,
        kernel: VectorKernelKind,
        max_snapshot_bytes: Option<u64>,
    ) -> Result<VectorSnapshotMeta> {
        let snapshot_dir = snapshot_dir.as_ref();
        let persistable = self.prepare_persistable_v2_snapshot()?;
        enforce_snapshot_size_limit(snapshot_dir, persistable.stats(), max_snapshot_bytes)?;
        persistable.write(snapshot_dir, kernel)
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
        Self::from_snapshot_v2_with_options(
            snapshot_dir,
            VectorSnapshotV2LoadOptions::default(),
            None,
        )
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
        cancellation: Option<&CancellationToken>,
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
        let origins = read_snapshot_v2_origins(&snapshot_dir, quantized_vectors.len())?;
        if ids.len() != quantized_vectors.len() {
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("vector", "snapshot_record_count_mismatch"),
                "snapshot ids and vector counts do not match",
            )
            .with_metadata("snapshotDir", snapshot_dir.display().to_string())
            .with_metadata("ids", ids.len().to_string())
            .with_metadata("vectors", quantized_vectors.len().to_string()));
        }
        if origins.len() != quantized_vectors.len() {
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("vector", "snapshot_record_count_mismatch"),
                "snapshot origins and vector counts do not match",
            )
            .with_metadata("snapshotDir", snapshot_dir.display().to_string())
            .with_metadata("origins", origins.len().to_string())
            .with_metadata("vectors", quantized_vectors.len().to_string()));
        }

        let quantizer = Quantizer::new(loaded.meta.quantization.clone())
            .map_err(|source| map_quantization_error(&snapshot_dir, &source, "build quantizer"))?;
        let records_with_origins = dequantize_snapshot_records(
            &quantizer,
            ids,
            origins,
            quantized_vectors,
            loaded.meta.dimension,
            &snapshot_dir,
            cancellation,
        )?;

        let mut params = loaded.meta.params;
        params.max_elements = params.max_elements.max(records_with_origins.len().max(1));

        // Try the fast path: load a persisted HNSW graph from disk.
        // When the graph files exist, we skip the O(n log n) HNSW rebuild entirely.
        // On failure we fall back to full reconstruction by re-dequantizing from
        // the same snapshot (records are consumed by load_persisted_graph).
        //
        // The fast path is only safe when the persisted graph's `origin_id`s
        // still align with an active-record mapping on disk. `origins.json`
        // provides that mapping for new snapshots; legacy snapshots fall back
        // to payload-slot order.
        //
        // Callers that load with a different kernel or cannot guarantee the
        // mapping must set `skip_persisted_graph = true` to force the rebuild path.
        let graph_file = snapshot_dir.join(format!("{SNAPSHOT_V2_HNSW_GRAPH_BASENAME}.hnsw.graph"));
        if graph_file.exists() && !options.skip_persisted_graph {
            match Self::load_persisted_graph(
                &snapshot_dir,
                loaded.meta.dimension,
                params,
                records_with_origins,
            ) {
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
                    let origins_again =
                        read_snapshot_v2_origins(&snapshot_dir, quantized_vectors.len())?;
                    let rebuilt_records = dequantize_snapshot_records(
                        &quantizer,
                        ids_again,
                        origins_again,
                        quantized_vectors,
                        loaded.meta.dimension,
                        &snapshot_dir,
                        cancellation,
                    )?;
                    let mut index = Self::new(loaded.meta.dimension, params)?;
                    index.insert_with_origins(rebuilt_records, cancellation)?;
                    return Ok(index);
                },
            }
        }

        // No persisted graph available.
        let mut index = Self::new(loaded.meta.dimension, params)?;
        if options.skip_graph_build {
            // Graph-agnostic kernel: populate records without the O(n log n)
            // HNSW build.  Only kernels that never call `self.hnsw.search()`
            // (e.g. FlatScan) may use the resulting index.
            index.insert_records_only(records_with_origins)?;
        } else {
            // Rebuild HNSW graph from scratch (O(n log n)).
            index.insert_with_origins(records_with_origins, cancellation)?;
        }
        Ok(index)
    }

    /// Materialize a derived HNSW-backed V2 snapshot from a source V2 snapshot.
    ///
    /// The source snapshot is loaded through the normal production read path so
    /// persisted HNSW graph/vector payloads are interpreted exactly the same way
    /// they are in the live product. The derived snapshot preserves the caller's
    /// selected origin IDs and writes a fresh host HNSW graph for the sampled
    /// root.
    pub fn materialize_subset_hnsw_snapshot_v2(
        source_snapshot_dir: impl AsRef<Path>,
        dest_snapshot_dir: impl AsRef<Path>,
        rows: &[SnapshotSubsetRow],
    ) -> Result<VectorSnapshotMeta> {
        let source = Self::from_snapshot_v2(source_snapshot_dir.as_ref())?;
        let mut params = *source.params();
        params.max_elements = params.max_elements.max(rows.len().max(1));
        let mut derived = Self::new(source.dimension(), params)?;

        let mut records_with_origins = Vec::with_capacity(rows.len());
        for row in rows {
            let record = source
                .record_for_id(row.id.as_ref())
                .cloned()
                .ok_or_else(|| {
                    ErrorEnvelope::expected(
                        ErrorCode::new("vector", "snapshot_subset_record_missing"),
                        "subset row id not found in source snapshot",
                    )
                    .with_metadata("id", row.id.to_string())
                    .with_metadata(
                        "sourceSnapshotDir",
                        source_snapshot_dir.as_ref().display().to_string(),
                    )
                })?;
            records_with_origins.push((row.origin, record));
        }

        derived.insert_with_assigned_origins(records_with_origins, None)?;
        derived.snapshot_v2_with_kernel_and_size_limit(
            dest_snapshot_dir,
            VectorKernelKind::HnswRs,
            None,
        )
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
        records_with_origins: Vec<(OriginId, VectorRecord)>,
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

        let mut records = Vec::with_capacity(records_with_origins.len());
        let mut slot_origins = Vec::with_capacity(records_with_origins.len());
        let mut id_to_origin = HashMap::with_capacity(records_with_origins.len());
        let max_origin = records_with_origins
            .iter()
            .map(|(origin, _)| origin.as_usize())
            .max()
            .unwrap_or(0);
        let mut origin_to_slot = vec![None; max_origin.saturating_add(1)];
        for (slot_index, (origin, record)) in records_with_origins.into_iter().enumerate() {
            let slot = PayloadSlot::from_usize(slot_index);
            if origin_to_slot.len() <= origin.as_usize() {
                origin_to_slot.resize(origin.as_usize() + 1, None);
            }
            let Some(binding) = origin_to_slot.get_mut(origin.as_usize()) else {
                return Err(ErrorEnvelope::unexpected(
                    ErrorCode::new("vector", "snapshot_origin_slot_missing"),
                    "origin slot missing while reconstructing persisted graph mappings",
                    ErrorClass::NonRetriable,
                )
                .with_metadata("snapshotDir", snapshot_dir.display().to_string())
                .with_metadata("origin", origin.as_usize().to_string()));
            };
            *binding = Some(slot);
            id_to_origin.insert(record.id.clone(), origin);
            slot_origins.push(origin);
            records.push(record);
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
            slot_origins,
            id_to_origin,
            origin_to_slot,
            deleted_slots: HashSet::new(),
            next_origin_id: max_origin.saturating_add(1),
            instance_id: NEXT_VECTOR_INDEX_INSTANCE_ID.fetch_add(1, Ordering::Relaxed),
            state_revision: 0,
        })
    }

    /// Load host HNSW ready-state from an explicit ready-state directory.
    pub fn from_hnsw_ready_state(
        ready_state_dir: impl AsRef<Path>,
        dimension: u32,
        params: HnswParams,
        records_with_origins: Vec<(OriginId, VectorRecord)>,
    ) -> Result<Self> {
        Self::load_persisted_graph(
            ready_state_dir.as_ref(),
            dimension,
            params,
            records_with_origins,
        )
    }

    /// Load host HNSW ready-state from an explicit ready-state directory using
    /// exact canonical rows as the durable origin-preserving payload source.
    pub fn from_hnsw_ready_state_with_exact_rows(
        ready_state_dir: impl AsRef<Path>,
        rows: &ExactVectorRows,
        params: HnswParams,
    ) -> Result<Self> {
        let records_with_origins = rows
            .rows()
            .map(|row| {
                (
                    row.origin(),
                    VectorRecord {
                        id: row.id().into(),
                        vector: row.vector().to_vec(),
                    },
                )
            })
            .collect::<Vec<(OriginId, VectorRecord)>>();
        Self::from_hnsw_ready_state(
            ready_state_dir,
            rows.dimension(),
            params,
            records_with_origins,
        )
    }

    fn ordered_active_entries(&self) -> Vec<(OriginId, &VectorRecord)> {
        let mut ordered: BTreeMap<&str, (OriginId, &VectorRecord)> = BTreeMap::new();
        for (id, origin) in &self.id_to_origin {
            if let Some(record) = self.record_for_origin(*origin) {
                ordered.insert(id.as_ref(), (*origin, record));
            }
        }

        ordered
            .into_values()
            .collect::<Vec<(OriginId, &VectorRecord)>>()
    }

    fn ordered_record_refs(&self) -> Vec<&VectorRecord> {
        self.ordered_active_entries()
            .into_iter()
            .map(|(_, record)| record)
            .collect::<Vec<&VectorRecord>>()
    }

    fn snapshot_records_and_origins(&self) -> (Vec<VectorRecord>, Vec<OriginId>) {
        if self.deleted_slots.is_empty() {
            return (self.records.clone(), self.slot_origins.clone());
        }

        let entries = self.active_entries_by_origin();
        let mut records = Vec::with_capacity(entries.len());
        let mut origins = Vec::with_capacity(entries.len());
        for (origin, record) in entries {
            origins.push(origin);
            records.push(record.clone());
        }
        (records, origins)
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

fn write_snapshot_v2_origins(snapshot_dir: &Path, origins: &[OriginId]) -> Result<()> {
    let path = snapshot_dir.join(VECTOR_SNAPSHOT_V2_ORIGINS_FILE_NAME);
    let payload = origins
        .iter()
        .copied()
        .map(OriginId::as_u64)
        .collect::<Result<Vec<u64>>>()?;
    let payload = serde_json::to_vec(payload.as_slice()).map_err(|source| {
        ErrorEnvelope::expected(
            ErrorCode::new("vector", "snapshot_origins_serialize_failed"),
            "failed to serialize v2 snapshot origins",
        )
        .with_metadata("path", path.display().to_string())
        .with_metadata("source", source.to_string())
    })?;
    std::fs::write(&path, payload).map_err(|source| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "snapshot_origins_write_failed"),
            "failed to write v2 snapshot origins",
            ErrorClass::NonRetriable,
        )
        .with_metadata("path", path.display().to_string())
        .with_metadata("source", source.to_string())
    })
}

/// Dequantize snapshot records with periodic cancellation checks.
fn dequantize_snapshot_records(
    quantizer: &Quantizer,
    ids: Vec<Box<str>>,
    origins: Vec<OriginId>,
    quantized_vectors: QuantizedSlice<'_>,
    dimension: u32,
    snapshot_dir: &Path,
    cancellation: Option<&CancellationToken>,
) -> Result<Vec<(OriginId, VectorRecord)>> {
    let mut records = Vec::with_capacity(ids.len());
    for (record_index, ((id, origin), quantized_vector)) in ids
        .into_iter()
        .zip(origins)
        .zip(quantized_vectors.iter())
        .enumerate()
    {
        if record_index % 1000 == 0
            && let Some(token) = cancellation
            && token.is_cancelled()
        {
            return Err(ErrorEnvelope::cancelled(
                "snapshot dequantization cancelled",
            ));
        }
        let vector = quantizer.dequantize(quantized_vector).map_err(|source| {
            map_quantization_error(snapshot_dir, &source, "dequantize snapshot vector")
                .with_metadata("recordIndex", record_index.to_string())
        })?;
        ensure_dimension(dimension, vector.as_slice()).map_err(|error| {
            error
                .with_metadata("snapshotDir", snapshot_dir.display().to_string())
                .with_metadata("recordIndex", record_index.to_string())
        })?;
        records.push((origin, VectorRecord { id, vector }));
    }
    Ok(records)
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

fn read_snapshot_v2_origins(snapshot_dir: &Path, expected_count: usize) -> Result<Vec<OriginId>> {
    let path = snapshot_dir.join(VECTOR_SNAPSHOT_V2_ORIGINS_FILE_NAME);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(source) if source.kind() == io::ErrorKind::NotFound => {
            return Ok((0..expected_count)
                .map(OriginId::from_usize)
                .collect::<Vec<OriginId>>());
        },
        Err(source) => {
            return Err(ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "snapshot_origins_read_failed"),
                "failed to read v2 snapshot origins",
                ErrorClass::NonRetriable,
            )
            .with_metadata("path", path.display().to_string())
            .with_metadata("source", source.to_string()));
        },
    };
    let origins: Vec<u64> = serde_json::from_slice(&bytes).map_err(|source| {
        ErrorEnvelope::expected(
            ErrorCode::new("vector", "snapshot_origins_parse_failed"),
            "failed to parse v2 snapshot origins",
        )
        .with_metadata("path", path.display().to_string())
        .with_metadata("source", source.to_string())
    })?;
    if origins.len() != expected_count {
        return Err(ErrorEnvelope::expected(
            ErrorCode::new("vector", "snapshot_record_count_mismatch"),
            "snapshot origins and vectors count mismatch",
        )
        .with_metadata("path", path.display().to_string())
        .with_metadata("origins", origins.len().to_string())
        .with_metadata("vectors", expected_count.to_string()));
    }
    origins
        .into_iter()
        .map(|origin| {
            usize::try_from(origin)
                .map(OriginId::from_usize)
                .map_err(|_| {
                    ErrorEnvelope::unexpected(
                        ErrorCode::new("vector", "snapshot_origins_overflow"),
                        "snapshot origin id conversion overflow",
                        ErrorClass::NonRetriable,
                    )
                    .with_metadata("path", path.display().to_string())
                    .with_metadata("origin", origin.to_string())
                })
        })
        .collect::<Result<Vec<OriginId>>>()
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
    origin_to_slot: &[Option<PayloadSlot>],
    neighbours: Vec<Neighbour>,
) -> Vec<VectorMatch> {
    neighbours
        .into_iter()
        .filter_map(|neighbour| {
            let slot = origin_to_slot.get(neighbour.d_id).and_then(|slot| *slot)?;
            let record = records.get(slot.as_usize())?;
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
    fn hnsw_search_emits_distance_eval_expansions() -> Result<()> {
        let mut index = VectorIndex::new(2, HnswParams::default())?;
        index.insert(vec![
            VectorRecord {
                id: "near".into(),
                vector: vec![1.0, 0.0],
            },
            VectorRecord {
                id: "far".into(),
                vector: vec![0.0, 1.0],
            },
            VectorRecord {
                id: "diag".into(),
                vector: vec![0.7, 0.7],
            },
        ])?;

        let query = [1.0, 0.0];
        let output = index.search_with_kernel(
            query.as_slice(),
            2,
            &HnswKernel::with_ef_search(5),
            VectorSearchBackend::F32Hnsw,
        )?;

        assert!(output.stats.expansions.is_some_and(|count| count > 0));
        assert_eq!(output.stats.extra.get("efSearch"), Some(&5.0));
        assert!(!output.matches.is_empty());
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
    fn kernel_load_capabilities_match_runtime_contract() {
        assert_eq!(
            VectorKernelKind::HnswRs.load_capabilities(),
            VectorKernelLoadCapabilities {
                requires_host_hnsw_graph: true,
                tolerates_snapshot_kernel_mismatch: false,
                supports_kernel_ready_state: false,
            }
        );
        assert_eq!(
            VectorKernelKind::Dfrr.load_capabilities(),
            VectorKernelLoadCapabilities {
                requires_host_hnsw_graph: false,
                tolerates_snapshot_kernel_mismatch: true,
                supports_kernel_ready_state: true,
            }
        );
        assert_eq!(
            VectorKernelKind::FlatScan.load_capabilities(),
            VectorKernelLoadCapabilities {
                requires_host_hnsw_graph: false,
                tolerates_snapshot_kernel_mismatch: true,
                supports_kernel_ready_state: false,
            }
        );
    }

    #[test]
    fn vector_index_state_key_tracks_mutations() -> Result<()> {
        let mut index = VectorIndex::new(2, HnswParams::default())?;
        let initial = index.state_key();

        index.insert(vec![VectorRecord {
            id: "a".into(),
            vector: vec![1.0, 0.0],
        }])?;
        let after_insert = index.state_key();
        assert_eq!(after_insert.instance_id, initial.instance_id);
        assert!(after_insert.revision > initial.revision);

        index.delete(&["a".into()])?;
        let after_delete = index.state_key();
        assert_eq!(after_delete.instance_id, initial.instance_id);
        assert!(after_delete.revision > after_insert.revision);
        Ok(())
    }

    #[test]
    fn no_op_kernels_accept_warm_requests() -> Result<()> {
        let mut index = VectorIndex::new(2, HnswParams::default())?;
        index.insert(vec![VectorRecord {
            id: "a".into(),
            vector: vec![1.0, 0.0],
        }])?;
        let context = VectorKernelWarmContext::new("warm-test", None, false);

        HnswKernel::new().warm(&index, &context, None)?;
        FlatScanKernel.warm(&index, &context, None)?;
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
    fn snapshot_v2_for_dfrr_omits_host_hnsw_graph_dump() -> Result<()> {
        let temp =
            TempDir::create("vector-index-v2-dfrr-no-host-graph").map_err(ErrorEnvelope::from)?;
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
        ])?;

        let _meta = index.snapshot_v2_with_kernel_and_size_limit(
            temp.path(),
            VectorKernelKind::Dfrr,
            None,
        )?;
        let graph_file = temp
            .path()
            .join(format!("{}.hnsw.graph", SNAPSHOT_V2_HNSW_GRAPH_BASENAME));
        let data_file = temp
            .path()
            .join(format!("{}.hnsw.data", SNAPSHOT_V2_HNSW_GRAPH_BASENAME));
        assert!(
            !graph_file.exists(),
            "DFRR collection snapshots should not persist host HNSW graph dumps"
        );
        assert!(
            !data_file.exists(),
            "DFRR collection snapshots should not persist host HNSW graph payloads"
        );
        Ok(())
    }

    #[test]
    fn materialize_subset_hnsw_snapshot_v2_preserves_origins_and_graph() -> Result<()> {
        let source =
            TempDir::create("vector-index-v2-subset-source").map_err(ErrorEnvelope::from)?;
        let dest = TempDir::create("vector-index-v2-subset-dest").map_err(ErrorEnvelope::from)?;
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
        let _ = index.snapshot_v2(source.path())?;
        let source_origins = read_snapshot_v2_origins(source.path(), 3)?;

        let rows = vec![
            SnapshotSubsetRow {
                id: "alpha".into(),
                origin: source_origins[0],
            },
            SnapshotSubsetRow {
                id: "gamma".into(),
                origin: source_origins[2],
            },
        ];

        let _ = VectorIndex::materialize_subset_hnsw_snapshot_v2(
            source.path(),
            dest.path(),
            rows.as_slice(),
        )?;

        let graph_file = dest
            .path()
            .join(format!("{}.hnsw.graph", SNAPSHOT_V2_HNSW_GRAPH_BASENAME));
        assert!(
            graph_file.exists(),
            "derived subset snapshot should persist an HNSW graph"
        );
        let restored = VectorIndex::from_snapshot_v2(dest.path())?;
        let derived_origins = read_snapshot_v2_origins(dest.path(), 2)?;
        assert_eq!(
            derived_origins,
            vec![source_origins[0], source_origins[2]],
            "derived subset snapshot must preserve sampled origin ids"
        );
        let matches = restored.search(&[0.0, 0.0, 1.0], 1)?.matches;
        assert_eq!(matches[0].id.as_ref(), "gamma");
        Ok(())
    }

    /// Regression test: insertion order != sorted order must produce correct
    /// search results after a persisted-graph roundtrip.
    ///
    /// Before the fix, `ids.json` was written in sorted (BTreeMap) order
    /// while the HNSW graph preserved insertion-order internal IDs.  Loading
    /// the persisted graph with sorted-order records caused `to_matches()`
    /// to map HNSW internal IDs to the wrong external IDs — producing 0.00
    /// recall despite finding the correct internal vectors.
    #[test]
    fn snapshot_v2_graph_persistence_unsorted_insertion_order() -> Result<()> {
        let temp =
            TempDir::create("vector-index-v2-graph-unsorted").map_err(ErrorEnvelope::from)?;
        let mut index = VectorIndex::new(3, HnswParams::default())?;

        // Insert in reverse-alphabetical order so insertion order ≠ sorted order.
        index.insert(vec![
            VectorRecord {
                id: "gamma".into(),
                vector: vec![0.0, 0.0, 1.0],
            },
            VectorRecord {
                id: "alpha".into(),
                vector: vec![1.0, 0.0, 0.0],
            },
            VectorRecord {
                id: "beta".into(),
                vector: vec![0.0, 1.0, 0.0],
            },
        ])?;

        let _meta = index.snapshot_v2(temp.path())?;
        let restored = VectorIndex::from_snapshot_v2(temp.path())?;

        // Query should return the correct external IDs, not shuffled ones.
        let original_matches = index.search(&[1.0, 0.0, 0.0], 3)?.matches;
        let restored_matches = restored.search(&[1.0, 0.0, 0.0], 3)?.matches;

        assert_eq!(
            original_matches.len(),
            restored_matches.len(),
            "result count should match"
        );
        for (original, restored) in original_matches.iter().zip(restored_matches.iter()) {
            assert_eq!(
                original.id, restored.id,
                "IDs must match after graph-persistence roundtrip (insertion order was non-sorted)"
            );
        }

        // Verify the top result is "alpha" (the vector most similar to the query).
        assert_eq!(
            restored_matches.first().map(|m| m.id.as_ref()),
            Some("alpha"),
            "top result must be 'alpha' — the vector [1,0,0] closest to query [1,0,0]"
        );

        Ok(())
    }

    #[test]
    fn hnsw_ready_state_roundtrip_uses_explicit_kernel_dir() -> Result<()> {
        let temp =
            TempDir::create("vector-index-hnsw-ready-state-dir").map_err(ErrorEnvelope::from)?;
        let generation_layout = CollectionGenerationPaths::new(temp.path());
        let generation_id = GenerationId::new("gen-hnsw-ready")?;
        let generation = generation_layout.generation(&generation_id);
        let ready_dir = generation.kernel_dir(VectorKernelKind::HnswRs);

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

        index.write_hnsw_ready_state(&ready_dir)?;

        let graph_file = ready_dir.join(format!("{}.hnsw.graph", SNAPSHOT_V2_HNSW_GRAPH_BASENAME));
        let data_file = ready_dir.join(format!("{}.hnsw.data", SNAPSHOT_V2_HNSW_GRAPH_BASENAME));
        assert!(
            graph_file.is_file(),
            "expected hnsw graph dump in explicit ready-state dir"
        );
        assert!(
            data_file.is_file(),
            "expected hnsw data dump in explicit ready-state dir"
        );

        let records_with_origins = index
            .active_entries_by_origin()
            .into_iter()
            .map(|(origin, record)| (origin, record.clone()))
            .collect::<Vec<(OriginId, VectorRecord)>>();
        let restored = VectorIndex::from_hnsw_ready_state(
            &ready_dir,
            index.dimension(),
            *index.params(),
            records_with_origins,
        )?;

        let original_matches = index.search(&[1.0, 0.0, 0.0], 3)?.matches;
        let restored_matches = restored.search(&[1.0, 0.0, 0.0], 3)?.matches;
        assert_eq!(original_matches.len(), restored_matches.len());
        assert_eq!(
            original_matches.first().map(|m| m.id.as_ref()),
            restored_matches.first().map(|m| m.id.as_ref()),
            "explicit ready-state roundtrip should preserve top result"
        );

        Ok(())
    }

    #[test]
    fn prepare_v2_snapshot_with_tombstones_uses_origin_order() -> Result<()> {
        let mut index = VectorIndex::new(3, HnswParams::default())?;
        index.insert(vec![
            VectorRecord {
                id: "gamma".into(),
                vector: vec![0.0, 0.0, 1.0],
            },
            VectorRecord {
                id: "alpha".into(),
                vector: vec![1.0, 0.0, 0.0],
            },
            VectorRecord {
                id: "beta".into(),
                vector: vec![0.0, 1.0, 0.0],
            },
        ])?;
        index.insert(vec![VectorRecord {
            id: "alpha".into(),
            vector: vec![0.9, 0.1, 0.0],
        }])?;

        let prepared = index.prepare_v2_snapshot()?;
        let expected_entries = index.active_entries_by_origin();
        let expected_ids = expected_entries
            .iter()
            .map(|(_, record)| record.id.clone())
            .collect::<Vec<_>>();
        let expected_origins = expected_entries
            .iter()
            .map(|(origin, _)| *origin)
            .collect::<Vec<_>>();

        assert_eq!(prepared.ordered_ids(), expected_ids);
        assert_eq!(prepared.origins, expected_origins);
        assert!(
            !prepared.graph_safe,
            "live tombstoned index should still mark the prepared snapshot as graph-unsafe"
        );
        Ok(())
    }

    #[test]
    fn snapshot_v2_with_tombstones_rebuilds_clean_graph_for_persistence() -> Result<()> {
        let temp =
            TempDir::create("vector-index-v2-graph-tombstones").map_err(ErrorEnvelope::from)?;
        let mut index = VectorIndex::new(3, HnswParams::default())?;
        index.insert(vec![
            VectorRecord {
                id: "gamma".into(),
                vector: vec![0.0, 0.0, 1.0],
            },
            VectorRecord {
                id: "alpha".into(),
                vector: vec![1.0, 0.0, 0.0],
            },
            VectorRecord {
                id: "beta".into(),
                vector: vec![0.0, 1.0, 0.0],
            },
        ])?;
        index.insert(vec![VectorRecord {
            id: "alpha".into(),
            vector: vec![0.95, 0.05, 0.0],
        }])?;

        let expected_entries = index.active_entries_by_origin();
        let expected_ids = expected_entries
            .iter()
            .map(|(_, record)| record.id.clone())
            .collect::<Vec<_>>();
        let expected_origins = expected_entries
            .iter()
            .map(|(origin, _)| *origin)
            .collect::<Vec<_>>();

        let _meta = index.snapshot_v2(temp.path())?;

        let graph_file = temp
            .path()
            .join(format!("{}.hnsw.graph", SNAPSHOT_V2_HNSW_GRAPH_BASENAME));
        let data_file = temp
            .path()
            .join(format!("{}.hnsw.data", SNAPSHOT_V2_HNSW_GRAPH_BASENAME));
        assert!(
            graph_file.exists(),
            "snapshot_v2 should rebuild a clean graph before persisting when tombstones exist"
        );
        assert!(
            data_file.exists(),
            "persisted graph payload should be present"
        );
        assert_eq!(
            read_snapshot_v2_ids(temp.path(), expected_ids.len())?,
            expected_ids
        );
        assert_eq!(
            read_snapshot_v2_origins(temp.path(), expected_origins.len())?,
            expected_origins
        );

        let restored = VectorIndex::from_snapshot_v2(temp.path())?;
        assert!(
            !restored.has_tombstones(),
            "reloaded snapshot should contain a clean active-only graph"
        );
        let original_matches = index.search(&[1.0, 0.0, 0.0], 3)?.matches;
        let restored_matches = restored.search(&[1.0, 0.0, 0.0], 3)?.matches;
        assert_eq!(original_matches.len(), restored_matches.len());
        for (original, restored) in original_matches.iter().zip(restored_matches.iter()) {
            assert_eq!(original.id, restored.id);
        }

        Ok(())
    }

    #[test]
    fn snapshot_v2_graph_persistence_survives_payload_reordering() -> Result<()> {
        let temp =
            TempDir::create("vector-index-v2-graph-reordered").map_err(ErrorEnvelope::from)?;
        let mut index = VectorIndex::new(3, HnswParams::default())?;
        index.insert(vec![
            VectorRecord {
                id: "gamma".into(),
                vector: vec![0.0, 0.0, 1.0],
            },
            VectorRecord {
                id: "alpha".into(),
                vector: vec![1.0, 0.0, 0.0],
            },
            VectorRecord {
                id: "beta".into(),
                vector: vec![0.0, 1.0, 0.0],
            },
        ])?;

        let _meta = index.snapshot_v2(temp.path())?;
        let ids = read_snapshot_v2_ids(temp.path(), 3)?;
        let origins = read_snapshot_v2_origins(temp.path(), 3)?;
        let meta_path = temp.path().join(SNAPSHOT_V2_META_FILE_NAME);
        let mut meta = read_metadata(&meta_path)
            .map_err(|source| map_snapshot_error(temp.path(), &source, "read snapshot metadata"))?;
        let dimension = usize::try_from(meta.dimension).map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "invalid_dimension"),
                "snapshot dimension conversion overflow",
                ErrorClass::NonRetriable,
            )
        })?;
        let vectors_path = temp.path().join(SNAPSHOT_V2_VECTORS_FILE_NAME);
        let vectors = std::fs::read(&vectors_path).map_err(ErrorEnvelope::from)?;

        let mut entries = ids
            .into_iter()
            .zip(origins.into_iter())
            .zip(vectors.chunks_exact(dimension))
            .map(|((id, origin), vector)| (id, origin, vector.to_vec()))
            .collect::<Vec<_>>();
        entries.reverse();

        let reordered_ids = entries
            .iter()
            .map(|(id, _, _)| id.clone())
            .collect::<Vec<Box<str>>>();
        let reordered_origins = entries
            .iter()
            .map(|(_, origin, _)| *origin)
            .collect::<Vec<OriginId>>();
        let reordered_vectors = entries
            .into_iter()
            .flat_map(|(_, _, vector)| vector)
            .collect::<Vec<u8>>();

        write_snapshot_v2_ids(temp.path(), reordered_ids.as_slice())?;
        write_snapshot_v2_origins(temp.path(), reordered_origins.as_slice())?;
        std::fs::write(&vectors_path, reordered_vectors.as_slice()).map_err(ErrorEnvelope::from)?;
        meta.vectors_crc32 = compute_vectors_crc32(reordered_vectors.as_slice());
        write_metadata(meta_path, &meta).map_err(|source| {
            map_snapshot_error(temp.path(), &source, "write snapshot metadata")
        })?;

        let restored = VectorIndex::from_snapshot_v2(temp.path())?;
        let restored_matches = restored.search(&[1.0, 0.0, 0.0], 3)?.matches;
        assert_eq!(
            restored_matches.first().map(|m| m.id.as_ref()),
            Some("alpha"),
            "persisted graph must survive payload reordering when origins.json is present"
        );

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

    #[test]
    fn insert_with_origins_respects_cancellation() -> Result<()> {
        let token = CancellationToken::new();
        token.cancel();
        let mut index = VectorIndex::new(2, HnswParams::default())?;
        let records = vec![
            (
                OriginId::from_usize(0),
                VectorRecord {
                    id: "a".into(),
                    vector: vec![1.0, 0.0],
                },
            ),
            (
                OriginId::from_usize(1),
                VectorRecord {
                    id: "b".into(),
                    vector: vec![0.0, 1.0],
                },
            ),
        ];
        let result = index.insert_with_origins(records, Some(&token));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("cancelled"),
            "expected cancellation error, got: {err}"
        );
        Ok(())
    }

    #[test]
    fn active_entries_by_origin_skips_tombstones_and_returns_sorted() -> Result<()> {
        let mut index = VectorIndex::new(2, HnswParams::default())?;

        // Insert 3 records.
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
                vector: vec![0.7, 0.7],
            },
        ])?;

        assert!(!index.has_tombstones(), "no tombstones before upsert");

        // Upsert "b" with a different vector — creates a tombstone for the old slot.
        index.insert(vec![VectorRecord {
            id: "b".into(),
            vector: vec![0.5, 0.5],
        }])?;

        assert!(index.has_tombstones(), "upsert should create a tombstone");

        let entries = index.active_entries_by_origin();

        // Should have 3 active entries (not 4), since the old "b" is tombstoned.
        assert_eq!(
            entries.len(),
            3,
            "expected 3 active entries after upsert, got {}",
            entries.len()
        );

        // Entries should be in ascending origin ID order.
        for window in entries.windows(2) {
            assert!(
                window[0].0.as_usize() < window[1].0.as_usize(),
                "entries should be in ascending origin ID order: {} < {}",
                window[0].0.as_usize(),
                window[1].0.as_usize()
            );
        }

        // The upserted record "b" should have the NEW vector.
        let b_entry = entries.iter().find(|(_, record)| record.id.as_ref() == "b");
        assert!(b_entry.is_some(), "record 'b' should be in active entries");
        let (_, b_record) = b_entry.expect("checked above");
        assert_eq!(
            b_record.vector,
            vec![0.5, 0.5],
            "upserted record should have the new vector"
        );

        Ok(())
    }

    #[test]
    fn exact_vector_rows_validate_dimension_and_origin_order() {
        let dimension_error = ExactVectorRows::new(
            2,
            vec![ExactVectorRow::new(
                "a",
                OriginId::from_usize(0),
                vec![1.0, 0.0, 0.5],
            )],
        )
        .err()
        .expect("dimension mismatch should fail");
        assert!(
            dimension_error.to_string().contains("dimension mismatch"),
            "unexpected dimension error: {dimension_error}"
        );

        let order_error = ExactVectorRows::new(
            2,
            vec![
                ExactVectorRow::new("a", OriginId::from_usize(2), vec![1.0, 0.0]),
                ExactVectorRow::new("b", OriginId::from_usize(1), vec![0.0, 1.0]),
            ],
        )
        .err()
        .expect("origin order validation should fail");
        assert!(
            order_error
                .to_string()
                .contains("strictly increasing origin order"),
            "unexpected origin-order error: {order_error}"
        );
    }

    #[test]
    fn exact_rows_from_vector_index_preserve_origin_order_and_fingerprint() -> Result<()> {
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
                vector: vec![0.7, 0.7],
            },
        ])?;
        index.insert(vec![VectorRecord {
            id: "b".into(),
            vector: vec![0.5, 0.5],
        }])?;

        let exact_rows = index.exact_rows()?;
        let observed = exact_rows
            .rows()
            .map(|row| {
                (
                    row.origin().as_usize(),
                    row.id().to_string(),
                    row.vector().to_vec(),
                )
            })
            .collect::<Vec<(usize, String, Vec<f32>)>>();

        assert_eq!(
            observed,
            vec![
                (0, "a".to_string(), vec![1.0, 0.0]),
                (2, "c".to_string(), vec![0.7, 0.7]),
                (3, "b".to_string(), vec![0.5, 0.5]),
            ],
            "exact rows should preserve canonical active origin order"
        );
        assert_eq!(
            exact_rows.fingerprint(),
            index.state_fingerprint(),
            "owned exact rows should hash identically to the lending VectorIndex view"
        );

        let source_rows = ExactVectorRowSource::rows(&index)
            .map(|row| (row.origin().as_usize(), row.id().to_string()))
            .collect::<Vec<(usize, String)>>();
        assert_eq!(
            source_rows,
            vec![
                (0, "a".to_string()),
                (2, "c".to_string()),
                (3, "b".to_string()),
            ],
            "lending row source should match owned exact row order"
        );

        Ok(())
    }

    #[test]
    fn records_only_insert_retires_prior_logical_ids() -> Result<()> {
        let mut index = VectorIndex::new(2, HnswParams::default())?;
        index.insert_records_without_graph(vec![
            VectorRecord {
                id: "a".into(),
                vector: vec![1.0, 0.0],
            },
            VectorRecord {
                id: "b".into(),
                vector: vec![0.0, 1.0],
            },
        ])?;
        index.insert_records_without_graph(vec![VectorRecord {
            id: "b".into(),
            vector: vec![0.5, 0.5],
        }])?;

        assert!(
            index.has_tombstones(),
            "records-only upsert should tombstone the prior slot"
        );
        let observed = index
            .exact_rows()?
            .rows()
            .map(|row| (row.id().to_string(), row.vector().to_vec()))
            .collect::<Vec<_>>();
        assert_eq!(
            observed,
            vec![
                ("a".to_string(), vec![1.0, 0.0]),
                ("b".to_string(), vec![0.5, 0.5]),
            ]
        );

        Ok(())
    }

    #[test]
    fn assigned_origin_records_only_insert_retires_prior_logical_ids() -> Result<()> {
        let mut index = VectorIndex::new(2, HnswParams::default())?;
        index.insert_records_without_graph_with_assigned_origins(vec![
            (
                OriginId::from_usize(0),
                VectorRecord {
                    id: "a".into(),
                    vector: vec![1.0, 0.0],
                },
            ),
            (
                OriginId::from_usize(1),
                VectorRecord {
                    id: "b".into(),
                    vector: vec![0.0, 1.0],
                },
            ),
        ])?;
        index.insert_records_without_graph_with_assigned_origins(vec![(
            OriginId::from_usize(2),
            VectorRecord {
                id: "b".into(),
                vector: vec![0.5, 0.5],
            },
        )])?;

        assert!(
            index.has_tombstones(),
            "assigned-origin records-only upsert should tombstone the prior slot"
        );
        let observed = index
            .exact_rows()?
            .rows()
            .map(|row| {
                (
                    row.origin().as_usize(),
                    row.id().to_string(),
                    row.vector().to_vec(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            observed,
            vec![
                (0, "a".to_string(), vec![1.0, 0.0]),
                (2, "b".to_string(), vec![0.5, 0.5]),
            ]
        );

        Ok(())
    }

    #[test]
    fn rebuild_active_index_reclaims_tombstones() -> Result<()> {
        let mut index = VectorIndex::new(2, HnswParams::default())?;

        // Insert 4 records.
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
                vector: vec![0.7, 0.7],
            },
            VectorRecord {
                id: "d".into(),
                vector: vec![0.3, 0.9],
            },
        ])?;

        // Upsert "c" with a new vector — creates a tombstone.
        index.insert(vec![VectorRecord {
            id: "c".into(),
            vector: vec![0.9, 0.1],
        }])?;

        assert!(index.has_tombstones());
        let original_active_count = index.active_count();
        assert_eq!(original_active_count, 4);

        // Capture origin IDs and search results before rebuild.
        let original_entries = index.active_entries_by_origin();
        let original_origin_ids: Vec<usize> =
            original_entries.iter().map(|(o, _)| o.as_usize()).collect();
        let original_search = index.search(&[1.0, 0.0], 4)?.matches;

        // Rebuild.
        let rebuilt = index.rebuild_active_index(None)?;

        // No tombstones in the rebuilt index.
        assert!(
            !rebuilt.has_tombstones(),
            "rebuilt index should have no tombstones"
        );

        // Same active count.
        assert_eq!(
            rebuilt.active_count(),
            original_active_count,
            "rebuilt index should have same active count"
        );

        // Search results should match.
        let rebuilt_search = rebuilt.search(&[1.0, 0.0], 4)?.matches;
        assert_eq!(
            rebuilt_search.len(),
            original_search.len(),
            "search result count should match"
        );
        for (orig, rebu) in original_search.iter().zip(rebuilt_search.iter()) {
            assert_eq!(orig.id, rebu.id, "search result IDs should match in order");
        }

        // Origin IDs should be preserved.
        let rebuilt_entries = rebuilt.active_entries_by_origin();
        let rebuilt_origin_ids: Vec<usize> =
            rebuilt_entries.iter().map(|(o, _)| o.as_usize()).collect();
        assert_eq!(
            rebuilt_origin_ids, original_origin_ids,
            "origin IDs should be preserved after rebuild"
        );

        Ok(())
    }
}
