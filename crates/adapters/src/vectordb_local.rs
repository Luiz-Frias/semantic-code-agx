//! Local vector database adapter backed by HNSW.

mod generation_control;

use self::generation_control::{
    CollectionBuildCoordinatorActor, CollectionBuildCoordinatorHandle, has_ready_dfrr_state,
    upsert_dfrr_ready_state,
};
use semantic_code_config::{
    SnapshotStorageMode, VectorKernelKind as ConfigVectorKernelKind, VectorSearchStrategy,
    VectorSnapshotFormat,
};
use semantic_code_domain::{IndexMode, Language, SearchStats};
use semantic_code_ports::{
    CollectionName, HybridSearchBatchRequest, HybridSearchData, HybridSearchResult, VectorDbPort,
    VectorDbProviderId, VectorDbProviderInfo, VectorDbRow, VectorDocument, VectorDocumentForInsert,
    VectorDocumentMetadata, VectorSearchRequest, VectorSearchResponse, VectorSearchResult,
};
use semantic_code_shared::{
    CancellationToken, ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result,
};
use semantic_code_vector::{
    CollectionGenerationPaths, ExactVectorRowSource, ExactVectorRowView, HnswParams,
    PreparedV2Snapshot, PublishedGenerationKernelSource, QuantizationCache,
    SNAPSHOT_V2_META_FILE_NAME, SNAPSHOT_V2_VECTORS_FILE_NAME, SnapshotStats, VectorIndex,
    VectorKernel, VectorKernelKind, VectorKernelSourcePathKind, VectorKernelWarmContext,
    VectorKernelWarmSource, VectorRecord, VectorSearchBackend, VectorSnapshotV2LoadOptions,
    VectorSnapshotWriteVersion, read_exact_generation, read_metadata,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::future::Future;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock as StdRwLock};
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;
use tokio::sync::{Mutex, Notify, RwLock, mpsc, oneshot};
use tokio::task::{JoinHandle, spawn_blocking};
use tracing::Instrument;

const LOCAL_SNAPSHOT_VERSION: u32 = 1;
const LOCAL_SNAPSHOT_DIR: &str = "vector";
const LOCAL_COLLECTIONS_DIR: &str = "collections";
const LOCAL_SNAPSHOT_V2_DIR_SUFFIX: &str = ".v2";
const LOCAL_SNAPSHOT_V2_IDS_FILE_NAME: &str = "ids.json";
const LOCAL_SNAPSHOT_V2_RECORDS_META_FILE_NAME: &str = "records.meta.jsonl";
const LOCAL_INSERT_WAL_FILE_SUFFIX: &str = ".wal.jsonl";
const LOCAL_BUILD_JOURNAL_DIR_NAME: &str = "build";
const LOCAL_BUILD_JOURNAL_META_FILE_NAME: &str = "journal.meta.json";
const LOCAL_BUILD_JOURNAL_ROWS_FILE_NAME: &str = "rows.jsonl";
const LOCAL_BUILD_JOURNAL_VECTORS_FILE_NAME: &str = "vectors.f32.bin";
const LOCAL_BUILD_JOURNAL_SEALED_FILE_NAME: &str = "SEALED";

#[derive(Debug, Clone)]
struct CollectionSnapshotPaths {
    v1_json: PathBuf,
    v2_dir: PathBuf,
    v2_meta: PathBuf,
    v2_vectors: PathBuf,
    v2_ids: PathBuf,
    v2_records_meta: PathBuf,
    insert_wal: PathBuf,
    generation_layout: CollectionGenerationPaths,
    build_meta: PathBuf,
    build_rows: PathBuf,
    build_vectors: PathBuf,
    build_sealed: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum V2CompanionState {
    Missing,
    Present,
}

struct CollectionCheckpointState {
    progress: Mutex<CollectionCheckpointProgress>,
    wal_io: Mutex<()>,
    notify: Notify,
    /// Cached quantization state for incremental snapshot preparation.
    quantization_cache: std::sync::Mutex<Option<QuantizationCache>>,
}

impl CollectionCheckpointState {
    fn new() -> Self {
        Self {
            progress: Mutex::new(CollectionCheckpointProgress::default()),
            wal_io: Mutex::new(()),
            notify: Notify::new(),
            quantization_cache: std::sync::Mutex::new(None),
        }
    }
}

#[derive(Default)]
struct CollectionCheckpointProgress {
    scheduled_sequence: u64,
    durable_sequence: u64,
    last_error: Option<ErrorEnvelope>,
    worker: Option<JoinHandle<()>>,
    /// Total vector count at the last durable checkpoint.
    vectors_at_last_checkpoint: u64,
}

struct StagedCollectionFinalizeData {
    generation_layout: Option<CollectionGenerationPaths>,
    build_host_graph: bool,
    exact_rows: semantic_code_vector::ExactVectorRows,
    snapshot: CollectionSnapshot,
}

struct CheckpointBuildStart;
struct CheckpointBuildCollected<'a> {
    index: std::sync::RwLockReadGuard<'a, VectorIndex>,
    /// When tombstones were present at snapshot time, this holds the freshly
    /// rebuilt (tombstone-free) index whose HNSW graph should be persisted.
    /// [`persist_bundle`] writes the graph dump from this index so the
    /// on-disk graph is always clean.
    rebuilt_index: Option<VectorIndex>,
    prepared: PreparedV2Snapshot,
    new_cache: QuantizationCache,
}
struct CheckpointBuildPersisted {
    new_cache: QuantizationCache,
    ordered_ids: Vec<Box<str>>,
}

struct V2CollectionCheckpointBuild<State> {
    state: State,
    _marker: PhantomData<fn() -> State>,
}

impl V2CollectionCheckpointBuild<CheckpointBuildStart> {
    const fn new() -> Self {
        Self {
            state: CheckpointBuildStart,
            _marker: PhantomData,
        }
    }

    fn collect_from_collection<'a>(
        self,
        collection_name: &CollectionName,
        paths: &CollectionSnapshotPaths,
        collection: &'a LocalCollection,
        snapshot_max_bytes: Option<u64>,
        checkpoint_state: Option<&CollectionCheckpointState>,
    ) -> Result<V2CollectionCheckpointBuild<CheckpointBuildCollected<'a>>> {
        let Self {
            state: _state,
            _marker,
        } = self;
        let index = collection.read_index()?;

        // When tombstones exist the live index has deleted slots that make
        // `graph_safe = false`, preventing the HNSW graph from being
        // persisted.  Rebuild a clean index from active entries so the
        // prepared snapshot is naturally graph-safe.  The incremental
        // quantization cache is invalid when the record set changes, so we
        // skip it and prepare a fresh (non-incremental) snapshot.
        let (prepared, new_cache, rebuilt_index) = if index.has_tombstones() {
            tracing::info!(
                collection = %collection_name,
                "tombstones detected — rebuilding clean index for checkpoint"
            );
            let clean = index
                .rebuild_active_index(None)
                .map_err(|error| map_snapshot_write_error(error, collection_name, &paths.v2_dir))?;
            let prepared = clean
                .prepare_v2_snapshot()
                .map_err(|error| map_snapshot_write_error(error, collection_name, &paths.v2_dir))?;
            let new_cache = QuantizationCache::from_prepared(&prepared);
            (prepared, new_cache, Some(clean))
        } else {
            let cached = checkpoint_state.and_then(|cs| {
                cs.quantization_cache
                    .lock()
                    .ok()
                    .and_then(|mut guard| guard.take())
            });
            let (prepared, new_cache) = index
                .prepare_v2_snapshot_incremental(cached.as_ref())
                .map_err(|error| map_snapshot_write_error(error, collection_name, &paths.v2_dir))?;
            (prepared, new_cache, None)
        };

        enforce_snapshot_limit(
            collection_name,
            &paths.v2_dir,
            VectorSnapshotWriteVersion::V2,
            prepared.stats.bytes,
            snapshot_max_bytes,
        )?;
        log_v2_snapshot_stats(collection_name, &paths.v2_dir, &prepared.stats);

        Ok(V2CollectionCheckpointBuild {
            state: CheckpointBuildCollected {
                index,
                rebuilt_index,
                prepared,
                new_cache,
            },
            _marker: PhantomData,
        })
    }
}

impl V2CollectionCheckpointBuild<CheckpointBuildCollected<'_>> {
    fn persist_bundle(
        self,
        collection_name: &CollectionName,
        paths: &CollectionSnapshotPaths,
        kernel: VectorKernelKind,
    ) -> Result<V2CollectionCheckpointBuild<CheckpointBuildPersisted>> {
        let Self {
            state:
                CheckpointBuildCollected {
                    index,
                    rebuilt_index,
                    prepared,
                    new_cache,
                },
            _marker,
        } = self;

        // Extract IDs in snapshot order before `prepared` is consumed by the
        // write call.  The sidecar must iterate in this same order so its rows
        // are aligned with `ids.json` and `vectors.u8.bin`.
        let ordered_ids = prepared.ordered_ids();

        // When a clean rebuild was performed the graph dump must come from the
        // rebuilt index (which has no tombstones and a contiguous HNSW graph).
        // The original read-guard is kept alive to prevent concurrent mutation.
        let write_index: &VectorIndex = rebuilt_index.as_ref().unwrap_or(&index);
        write_index
            .write_prepared_v2_snapshot(paths.v2_dir.as_path(), kernel, prepared)
            .map(|_| ())
            .map_err(|error| map_snapshot_write_error(error, collection_name, &paths.v2_dir))?;

        Ok(V2CollectionCheckpointBuild {
            state: CheckpointBuildPersisted {
                new_cache,
                ordered_ids,
            },
            _marker: PhantomData,
        })
    }
}

impl V2CollectionCheckpointBuild<CheckpointBuildPersisted> {
    fn finalize(
        self,
        paths: &CollectionSnapshotPaths,
        collection: &LocalCollection,
        checkpoint_state: Option<&CollectionCheckpointState>,
    ) -> Result<()> {
        let Self {
            state:
                CheckpointBuildPersisted {
                    new_cache,
                    ordered_ids,
                },
            _marker,
        } = self;
        if let Some(state) = checkpoint_state
            && let Ok(mut guard) = state.quantization_cache.lock()
        {
            *guard = Some(new_cache);
        }
        write_sidecar_from_collection(paths, collection, &ordered_ids)
    }
}

/// Default checkpoint divisor (k).
///
/// Checkpoint interval = `current_vector_count / k`. With k=5 and 100K
/// vectors the interval is 20K — at most 20% of data is un-checkpointed.
const DEFAULT_CHECKPOINT_DIVISOR: u32 = 5;

/// Configuration and dependencies needed to load collections from disk.
///
/// Decouples "how to load" from "when to load" — the actor decides
/// *when*; this struct does the actual I/O.
#[derive(Clone)]
struct CollectionLoaderContext {
    codebase_root: PathBuf,
    storage_mode: SnapshotStorageMode,
    snapshot_format: VectorSnapshotFormat,
    snapshot_max_bytes: Option<u64>,
    kernel: Arc<dyn VectorKernel + Send + Sync>,
    runtime_dfrr_ready_state: Option<DfrrReadyStateRequirement>,
    dfrr_prewarm_requests: Vec<DfrrReadyStatePrewarmRequest>,
    force_reindex_on_kernel_change: bool,
    search_backend: VectorSearchBackend,
    hnsw_params: HnswParams,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// DFRR ready-state identity required by the runtime loader/search path.
pub struct DfrrReadyStateRequirement {
    /// Stable DFRR ready-state fingerprint derived from build-shaping config.
    pub ready_state_fingerprint: Box<str>,
    /// JSON form of the representative DFRR search config that produced the fingerprint.
    pub config_json: Box<str>,
}

#[derive(Clone)]
/// Additional DFRR ready-state variant that should be prewarmed during publish.
pub struct DfrrReadyStatePrewarmRequest {
    /// Stable identity and representative config for the prewarm target.
    pub requirement: DfrrReadyStateRequirement,
    /// Concrete DFRR kernel instance used to materialize this variant.
    pub kernel: Arc<dyn VectorKernel + Send + Sync>,
}

impl std::fmt::Debug for DfrrReadyStatePrewarmRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DfrrReadyStatePrewarmRequest")
            .field("requirement", &self.requirement)
            .field("kernel_kind", &self.kernel.kind())
            .finish()
    }
}

/// Local vector DB backed by an HNSW index.
pub struct LocalVectorDb {
    provider: VectorDbProviderInfo,
    loader: CollectionLoaderContext,
    collections: Arc<RwLock<HashMap<CollectionName, LocalCollection>>>,
    checkpoint_states: Arc<RwLock<HashMap<CollectionName, Arc<CollectionCheckpointState>>>>,
    /// Checkpoint frequency divisor (k).
    ///
    /// Checkpoint interval = `current_vector_count / k`. Only triggers a
    /// checkpoint write when enough new vectors have accumulated since the
    /// last durable checkpoint. Set to `0` to disable throttling (checkpoint
    /// after every insert). The `flush` path always bypasses throttling.
    checkpoint_divisor: u32,
    build_coordinator: CollectionBuildCoordinatorHandle,
    /// Actor handle for collection lifecycle management.
    ///
    /// All `ensure_loaded` calls delegate to the actor, which serializes
    /// load/evict operations via a bounded channel.
    loader_handle: CollectionLoaderHandle,
    #[cfg(test)]
    v2_from_collection_write_calls: Arc<AtomicUsize>,
    #[cfg(test)]
    v2_bundle_write_calls: Arc<AtomicUsize>,
    #[cfg(test)]
    checkpoint_delay_ms: Arc<AtomicU64>,
}

/// Builder for constructing a [`LocalVectorDb`] from required and optional
/// config-derived parameters.
///
/// Replaces the positional-argument constructors with named, chainable setters
/// so new parameters can be added without touching existing call sites.
///
/// ```ignore
/// let db = LocalVectorDbBuilder::new(root, kernel, cancel)
///     .storage_mode(SnapshotStorageMode::Custom(path))
///     .snapshot_format(VectorSnapshotFormat::V2)
///     .build()?;
/// ```
pub struct LocalVectorDbBuilder {
    codebase_root: PathBuf,
    kernel: Arc<dyn VectorKernel + Send + Sync>,
    cancellation: CancellationToken,
    storage_mode: SnapshotStorageMode,
    snapshot_format: VectorSnapshotFormat,
    snapshot_max_bytes: Option<u64>,
    force_reindex_on_kernel_change: bool,
    search_strategy: VectorSearchStrategy,
    hnsw_build_config: Option<semantic_code_config::HnswBuildConfig>,
    runtime_dfrr_ready_state: Option<DfrrReadyStateRequirement>,
    dfrr_prewarm_requests: Vec<DfrrReadyStatePrewarmRequest>,
}

impl LocalVectorDbBuilder {
    /// Create a builder with the three required fields.
    #[must_use]
    pub fn new(
        codebase_root: PathBuf,
        kernel: Arc<dyn VectorKernel + Send + Sync>,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            codebase_root,
            kernel,
            cancellation,
            storage_mode: SnapshotStorageMode::default(),
            snapshot_format: VectorSnapshotFormat::default(),
            snapshot_max_bytes: None,
            force_reindex_on_kernel_change: false,
            search_strategy: VectorSearchStrategy::default(),
            hnsw_build_config: None,
            runtime_dfrr_ready_state: None,
            dfrr_prewarm_requests: Vec::new(),
        }
    }

    /// Set the snapshot storage mode.
    #[must_use]
    pub fn storage_mode(mut self, mode: SnapshotStorageMode) -> Self {
        self.storage_mode = mode;
        self
    }

    /// Set the snapshot format.
    #[must_use]
    pub const fn snapshot_format(mut self, format: VectorSnapshotFormat) -> Self {
        self.snapshot_format = format;
        self
    }

    /// Set the maximum allowed snapshot size in bytes.
    #[must_use]
    pub const fn snapshot_max_bytes(mut self, bytes: u64) -> Self {
        self.snapshot_max_bytes = Some(bytes);
        self
    }

    /// Force a full reindex when the kernel kind changes between restarts.
    #[must_use]
    pub const fn force_reindex_on_kernel_change(mut self, force: bool) -> Self {
        self.force_reindex_on_kernel_change = force;
        self
    }

    /// Set the vector search strategy.
    #[must_use]
    pub const fn search_strategy(mut self, strategy: VectorSearchStrategy) -> Self {
        self.search_strategy = strategy;
        self
    }

    /// Set the HNSW build configuration (graph construction params).
    #[must_use]
    pub const fn hnsw_build_config(
        mut self,
        config: &semantic_code_config::HnswBuildConfig,
    ) -> Self {
        self.hnsw_build_config = Some(*config);
        self
    }

    /// Set the runtime DFRR ready-state requirement.
    #[must_use]
    pub fn runtime_dfrr_ready_state(mut self, state: DfrrReadyStateRequirement) -> Self {
        self.runtime_dfrr_ready_state = Some(state);
        self
    }

    /// Set additional DFRR prewarm requests to materialize at publish time.
    #[must_use]
    pub fn dfrr_prewarm_requests(mut self, requests: Vec<DfrrReadyStatePrewarmRequest>) -> Self {
        self.dfrr_prewarm_requests = requests;
        self
    }

    /// Build the [`LocalVectorDb`]. Consumes the builder.
    pub fn build(self) -> Result<LocalVectorDb> {
        let provider = VectorDbProviderInfo {
            id: VectorDbProviderId::parse("local").map_err(ErrorEnvelope::from)?,
            name: "Local".into(),
        };
        let loader = CollectionLoaderContext {
            codebase_root: self.codebase_root,
            storage_mode: self.storage_mode,
            snapshot_format: self.snapshot_format,
            snapshot_max_bytes: self.snapshot_max_bytes,
            kernel: self.kernel,
            runtime_dfrr_ready_state: self.runtime_dfrr_ready_state,
            dfrr_prewarm_requests: self.dfrr_prewarm_requests,
            force_reindex_on_kernel_change: self.force_reindex_on_kernel_change,
            search_backend: resolve_search_backend(self.search_strategy),
            hnsw_params: HnswParams::from_build_config(self.hnsw_build_config.as_ref()),
        };
        tracing::debug!(
            runtime_dfrr_ready_state = loader
                .runtime_dfrr_ready_state
                .as_ref()
                .map(|requirement| requirement.ready_state_fingerprint.as_ref()),
            dfrr_prewarm_requests = loader.dfrr_prewarm_requests.len(),
            "configured local vectordb ready-state prewarm plan"
        );
        let collections = Arc::new(RwLock::new(HashMap::new()));
        let checkpoint_states = Arc::new(RwLock::new(HashMap::new()));
        let (build_coordinator, _build_join) =
            CollectionBuildCoordinatorActor::spawn(self.cancellation.clone());
        let (loader_handle, _loader_join) = CollectionLoaderActor::spawn(
            loader.clone(),
            Arc::clone(&collections),
            Arc::clone(&checkpoint_states),
            self.cancellation,
        );
        Ok(LocalVectorDb {
            provider,
            loader,
            collections,
            checkpoint_states,
            checkpoint_divisor: DEFAULT_CHECKPOINT_DIVISOR,
            build_coordinator,
            loader_handle,
            #[cfg(test)]
            v2_from_collection_write_calls: Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            v2_bundle_write_calls: Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            checkpoint_delay_ms: Arc::new(AtomicU64::new(0)),
        })
    }
}

impl LocalVectorDb {
    /// Return whether a local vector kernel is available in this build.
    pub const fn is_kernel_supported(kernel: ConfigVectorKernelKind) -> bool {
        match kernel {
            ConfigVectorKernelKind::Dfrr => cfg!(feature = "experimental-dfrr-kernel"),
            ConfigVectorKernelKind::HnswRs | ConfigVectorKernelKind::FlatScan => true,
        }
    }

    fn snapshot_root(&self) -> Option<PathBuf> {
        self.loader.snapshot_root()
    }

    fn snapshot_paths(&self, collection_name: &CollectionName) -> Option<CollectionSnapshotPaths> {
        self.loader.snapshot_paths(collection_name)
    }

    /// Ensure a collection is loaded from its on-disk snapshot exactly once.
    ///
    /// Delegates to the collection loader actor, which serializes load/evict
    /// operations via a bounded channel. The fast-path check avoids channel
    /// overhead for warm lookups.
    async fn ensure_loaded(&self, collection_name: &CollectionName) -> Result<()> {
        // Fast path: already loaded or created via create_collection.
        {
            let collections = self.collections.read().await;
            if collections.contains_key(collection_name) {
                return Ok(());
            }
        }
        self.loader_handle.load(collection_name.clone()).await
    }

    /// Read a v1 JSON snapshot from disk.
    ///
    /// Delegated to the loader context.  Only used in integration tests —
    /// production code calls `self.loader.read_snapshot_json` directly.
    #[cfg(test)]
    async fn read_snapshot_json(
        &self,
        collection_name: &CollectionName,
    ) -> Result<Option<CollectionSnapshot>> {
        self.loader.read_snapshot_json(collection_name).await
    }

    /// Write a V1 JSON snapshot to disk.  Only called when
    /// `snapshot_format == V1` (legacy path).  V2-mode callers use
    /// [`Self::write_v2_from_collection`] directly.
    async fn write_snapshot(
        &self,
        collection_name: &CollectionName,
        snapshot: &CollectionSnapshot,
    ) -> Result<()> {
        let Some(paths) = self.snapshot_paths(collection_name) else {
            return Ok(());
        };

        let payload = serialize_snapshot_json(snapshot)?;
        let payload_bytes = u64::try_from(payload.len()).map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "snapshot_count_overflow"),
                "snapshot JSON size conversion overflow",
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata("path", paths.v1_json.display().to_string())
            .with_metadata("bytes", payload.len().to_string())
        })?;
        enforce_snapshot_limit(
            collection_name,
            &paths.v1_json,
            VectorSnapshotWriteVersion::V1,
            payload_bytes,
            self.loader.snapshot_max_bytes,
        )?;
        log_json_snapshot_stats(collection_name, &paths.v1_json, snapshot, payload_bytes);
        Self::write_snapshot_json(paths.v1_json.as_path(), payload.as_slice()).await?;
        Ok(())
    }

    async fn append_insert_wal(
        &self,
        collection_name: &CollectionName,
        wal_record: &InsertWalRecord,
    ) -> Result<()> {
        let Some(paths) = self.snapshot_paths(collection_name) else {
            return Ok(());
        };
        let checkpoint_state = self.checkpoint_state_for_collection(collection_name).await;
        let _wal_io = checkpoint_state.wal_io.lock().await;
        append_insert_wal_record(paths.insert_wal.as_path(), wal_record).await
    }

    async fn finalize_staged_collection(&self, collection_name: &CollectionName) -> Result<()> {
        let should_close = {
            let collections = self.collections.read().await;
            let Some(collection) = collections.get(collection_name) else {
                return Err(ErrorEnvelope::expected(
                    ErrorCode::not_found(),
                    "collection not found",
                ));
            };
            let should_close = collection.is_staging();
            drop(collections);
            should_close
        } && self.snapshot_paths(collection_name).is_some();
        if !should_close {
            return Ok(());
        }

        self.build_coordinator
            .close_session(collection_name.clone())
            .await?
            .wait()
            .await;

        let Some(staged) = self.collect_staged_finalize_data(collection_name).await? else {
            return Ok(());
        };

        let runtime_index = build_runtime_index_from_exact_rows_async(
            collection_name,
            staged.exact_rows.clone(),
            staged.build_host_graph,
            self.loader.hnsw_params,
            None,
        )
        .await?;
        let mut runtime_index = Some(runtime_index);

        if let Some(generation_layout) = staged.generation_layout {
            if let Some(paths) = self.snapshot_paths(collection_name) {
                seal_build_journal(&paths, &staged.snapshot).await?;
            }
            let generation_id = self
                .build_coordinator
                .stage_base_generation(
                    collection_name.clone(),
                    generation_layout.clone(),
                    staged.exact_rows,
                    staged.snapshot,
                )
                .await?;
            runtime_index = Some(
                self.publish_kernel_ready_state_for_generation(
                    collection_name,
                    &generation_layout,
                    generation_layout.generation(&generation_id),
                    runtime_index.take().ok_or_else(|| {
                        ErrorEnvelope::unexpected(
                            ErrorCode::new("vector", "runtime_index_missing"),
                            "runtime index missing before kernel-ready publish",
                            ErrorClass::NonRetriable,
                        )
                    })?,
                    staged.build_host_graph,
                )
                .await?,
            );
            self.build_coordinator
                .activate_generation(collection_name.clone(), generation_layout, generation_id)
                .await?;
        }

        self.install_online_runtime_index(
            collection_name,
            runtime_index.take().ok_or_else(|| {
                ErrorEnvelope::unexpected(
                    ErrorCode::new("vector", "runtime_index_missing"),
                    "runtime index missing before collection publish",
                    ErrorClass::NonRetriable,
                )
            })?,
        )
        .await?;
        Ok(())
    }

    async fn collect_staged_finalize_data(
        &self,
        collection_name: &CollectionName,
    ) -> Result<Option<StagedCollectionFinalizeData>> {
        let generation_layout = self
            .snapshot_paths(collection_name)
            .map(|paths| paths.generation_layout);
        let build_host_graph = self.loader.runtime_kernel_requires_host_hnsw_graph();
        let collections = self.collections.read().await;
        let Some(collection) = collections.get(collection_name) else {
            return Err(ErrorEnvelope::expected(
                ErrorCode::not_found(),
                "collection not found",
            ));
        };
        if !collection.is_staging() {
            return Ok(None);
        }
        let rows = collection.exact_rows()?;
        let snapshot = collection.snapshot()?;
        drop(collections);
        Ok(Some(StagedCollectionFinalizeData {
            generation_layout,
            build_host_graph,
            exact_rows: rows,
            snapshot,
        }))
    }

    async fn publish_kernel_ready_state_for_generation(
        &self,
        collection_name: &CollectionName,
        generation_layout: &CollectionGenerationPaths,
        generation: semantic_code_vector::PublishedGenerationPaths,
        runtime_index: VectorIndex,
        build_host_graph: bool,
    ) -> Result<VectorIndex> {
        if build_host_graph {
            runtime_index
                .write_hnsw_ready_state(generation.kernel_dir(VectorKernelKind::HnswRs))?;
        }
        if !self.loader.runtime_kernel_supports_ready_state()
            && self.loader.dfrr_prewarm_requests.is_empty()
        {
            return Ok(runtime_index);
        }

        let shared_generation_source = if (self.loader.runtime_kernel_supports_ready_state()
            && self.loader.runtime_kernel_prefers_generation_source())
            || self.loader.dfrr_prewarm_requests.iter().any(|request| {
                request
                    .kernel
                    .kind()
                    .load_capabilities()
                    .canonical_source_path
                    == VectorKernelSourcePathKind::SegmentedSourceV1
            }) {
            Some(Arc::new(
                load_generation_kernel_source(collection_name, &generation, None).await?,
            ))
        } else {
            None
        };

        let runtime_index_handle = Arc::new(StdRwLock::new(runtime_index));
        if self.loader.runtime_kernel_supports_ready_state() {
            if self.loader.runtime_kernel_prefers_generation_source() {
                let Some(source) = shared_generation_source.as_ref() else {
                    return Err(ErrorEnvelope::unexpected(
                        ErrorCode::new("vector", "generation_source_missing"),
                        "shared generation source missing for source-backed kernel warm",
                        ErrorClass::NonRetriable,
                    ));
                };
                warm_loaded_generation_source_at_path(
                    Arc::clone(&self.loader.kernel),
                    collection_name,
                    Arc::clone(&runtime_index_handle),
                    Arc::clone(source),
                    Some(generation.kernels_dir().to_path_buf()),
                    true,
                    None,
                )
                .await?;
            } else {
                warm_collection_kernel_state_at_path(
                    &self.loader,
                    collection_name,
                    Arc::clone(&runtime_index_handle),
                    Some(generation.kernels_dir().to_path_buf()),
                    true,
                    None,
                )
                .await?;
            }
        }

        self.prewarm_dfrr_ready_states_for_generation(
            collection_name,
            generation_layout,
            &generation,
            shared_generation_source,
            Arc::clone(&runtime_index_handle),
        )
        .await?;

        let runtime_index_handle = Arc::try_unwrap(runtime_index_handle).map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "runtime_index_unwrap_failed"),
                "runtime index handle still shared after kernel warm",
                ErrorClass::NonRetriable,
            )
        })?;
        runtime_index_handle
            .into_inner()
            .map_err(|_| index_lock_error("into_inner"))
    }

    async fn prewarm_dfrr_ready_states_for_generation(
        &self,
        collection_name: &CollectionName,
        generation_layout: &CollectionGenerationPaths,
        generation: &semantic_code_vector::PublishedGenerationPaths,
        generation_source: Option<Arc<PublishedGenerationKernelSource>>,
        runtime_index_handle: Arc<StdRwLock<VectorIndex>>,
    ) -> Result<()> {
        let generation_id = generation.generation_id();
        let generation_root = Some(generation.kernels_dir().to_path_buf());
        let total_dfrr_prewarm_states = count_unique_dfrr_prewarm_states(
            self.loader.runtime_dfrr_ready_state.as_ref(),
            &self.loader.dfrr_prewarm_requests,
        );
        let mut current_dfrr_prewarm_state = 0_u64;

        if total_dfrr_prewarm_states == 0 {
            return Ok(());
        }

        tracing::info!(
            collection = %collection_name,
            generation = generation_id.as_str(),
            total = total_dfrr_prewarm_states,
            "starting dfrr ready-state prewarm stage"
        );

        if let Some(requirement) = self.loader.runtime_dfrr_ready_state.as_ref() {
            current_dfrr_prewarm_state = current_dfrr_prewarm_state.saturating_add(1);
            let prefer = self.loader.runtime_kernel_prefers_generation_source();
            prewarm_and_record(
                DfrrPrewarmContext {
                    kernel: Arc::clone(&self.loader.kernel),
                    collection_name,
                    generation,
                    generation_source: generation_source.clone(),
                    generation_id,
                    ready_state_fingerprint: requirement.ready_state_fingerprint.as_ref(),
                    current: current_dfrr_prewarm_state,
                    total: total_dfrr_prewarm_states,
                    index: Arc::clone(&runtime_index_handle),
                    generation_root: generation_root.clone(),
                    cancellation: None,
                },
                prefer,
                generation_layout,
                requirement,
            )
            .await?;
        }

        for request in &self.loader.dfrr_prewarm_requests {
            if self
                .loader
                .runtime_dfrr_ready_state
                .as_ref()
                .is_some_and(|rt| {
                    rt.ready_state_fingerprint == request.requirement.ready_state_fingerprint
                })
            {
                continue;
            }
            current_dfrr_prewarm_state = current_dfrr_prewarm_state.saturating_add(1);
            let prefer = request
                .kernel
                .kind()
                .load_capabilities()
                .canonical_source_path
                == VectorKernelSourcePathKind::SegmentedSourceV1;
            prewarm_and_record(
                DfrrPrewarmContext {
                    kernel: Arc::clone(&request.kernel),
                    collection_name,
                    generation,
                    generation_source: generation_source.clone(),
                    generation_id,
                    ready_state_fingerprint: request.requirement.ready_state_fingerprint.as_ref(),
                    current: current_dfrr_prewarm_state,
                    total: total_dfrr_prewarm_states,
                    index: Arc::clone(&runtime_index_handle),
                    generation_root: generation_root.clone(),
                    cancellation: None,
                },
                prefer,
                generation_layout,
                &request.requirement,
            )
            .await?;
        }

        tracing::info!(
            collection = %collection_name,
            generation = generation_id.as_str(),
            total = total_dfrr_prewarm_states,
            "completed dfrr ready-state prewarm stage"
        );
        Ok(())
    }

    async fn install_online_runtime_index(
        &self,
        collection_name: &CollectionName,
        runtime_index: VectorIndex,
    ) -> Result<()> {
        let mut collections = self.collections.write().await;
        let Some(collection) = collections.get_mut(collection_name) else {
            return Err(ErrorEnvelope::expected(
                ErrorCode::not_found(),
                "collection not found",
            ));
        };
        if collection.is_staging() {
            collection.replace_index(runtime_index)?;
            collection.mark_online();
        }
        drop(collections);
        Ok(())
    }

    async fn checkpoint_state_for_collection(
        &self,
        collection_name: &CollectionName,
    ) -> Arc<CollectionCheckpointState> {
        {
            let states = self.checkpoint_states.read().await;
            if let Some(state) = states.get(collection_name) {
                return Arc::clone(state);
            }
        }

        let mut states = self.checkpoint_states.write().await;
        Arc::clone(
            states
                .entry(collection_name.clone())
                .or_insert_with(|| Arc::new(CollectionCheckpointState::new())),
        )
    }

    async fn reap_finished_checkpoint_worker(
        &self,
        collection_name: &CollectionName,
        state: &Arc<CollectionCheckpointState>,
    ) {
        let handle = {
            let mut progress = state.progress.lock().await;
            if progress
                .worker
                .as_ref()
                .is_some_and(JoinHandle::is_finished)
            {
                progress.worker.take()
            } else {
                None
            }
        };

        let Some(handle) = handle else {
            return;
        };

        if let Err(join_error) = handle.await {
            let error = ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "checkpoint_task_failed"),
                "local checkpoint task failed",
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata("source", join_error.to_string());
            tracing::error!(
                collection = collection_name.as_str(),
                source = %join_error,
                "adapter.vectordb.local.checkpoint_task_failed"
            );
            let mut progress = state.progress.lock().await;
            progress.last_error = Some(error);
        }
        state.notify.notify_waiters();
    }

    /// Schedule a checkpoint, throttled by the checkpoint divisor.
    ///
    /// `current_vector_count` is the current total vector count in the
    /// collection. The worker is only spawned when enough new vectors have
    /// accumulated since the last durable checkpoint (determined by
    /// `current_vector_count / checkpoint_divisor`). Pass `force = true`
    /// to bypass throttling (used by the `flush` path).
    async fn schedule_checkpoint(
        &self,
        collection_name: &CollectionName,
        sequence: u64,
        current_vector_count: u64,
        force: bool,
    ) -> Arc<CollectionCheckpointState> {
        let state = self.checkpoint_state_for_collection(collection_name).await;
        self.reap_finished_checkpoint_worker(collection_name, &state)
            .await;

        let mut progress = state.progress.lock().await;
        if sequence > progress.scheduled_sequence {
            progress.scheduled_sequence = sequence;
            progress.last_error = None;
        } else if progress.last_error.is_some() && sequence >= progress.durable_sequence {
            // Allow a fresh scheduling attempt after a previous checkpoint failure.
            progress.last_error = None;
        }

        let should_checkpoint = if force || self.checkpoint_divisor == 0 {
            true
        } else {
            let vectors_since =
                current_vector_count.saturating_sub(progress.vectors_at_last_checkpoint);
            // interval = max(current_count / k, 1) — at least 1 to avoid
            // stalling when the index is very small.
            // Integer division: interval = max(current_count / k, 1) so that
            // a checkpoint is always triggered after at least one new vector.
            let divisor = u64::from(self.checkpoint_divisor.max(1));
            let interval = (current_vector_count / divisor).max(1);
            vectors_since >= interval
        };

        if progress.worker.is_none()
            && progress.scheduled_sequence > progress.durable_sequence
            && should_checkpoint
        {
            progress.vectors_at_last_checkpoint = current_vector_count;
            let db = self.clone();
            let collection_name = collection_name.clone();
            let state_for_task = Arc::clone(&state);
            let handle = tokio::spawn(async move {
                db.run_checkpoint_worker(collection_name, state_for_task)
                    .await;
            });
            progress.worker = Some(handle);
        }
        drop(progress);
        state.notify.notify_waiters();
        state
    }

    async fn run_checkpoint_worker(
        self,
        collection_name: CollectionName,
        state: Arc<CollectionCheckpointState>,
    ) {
        loop {
            let target_sequence = {
                let progress = state.progress.lock().await;
                if progress.scheduled_sequence <= progress.durable_sequence {
                    break;
                }
                progress.scheduled_sequence
            };

            match self
                .write_checkpoint_and_compact_wal(&collection_name, target_sequence, &state)
                .await
            {
                Ok(checkpoint_sequence) => {
                    let mut progress = state.progress.lock().await;
                    progress.durable_sequence = progress.durable_sequence.max(checkpoint_sequence);
                    progress.last_error = None;
                    drop(progress);
                    state.notify.notify_waiters();
                },
                Err(error) => {
                    tracing::error!(
                        collection = collection_name.as_str(),
                        target_sequence,
                        code = %error.code,
                        message = %error.message,
                        "adapter.vectordb.local.checkpoint_failed"
                    );
                    let mut progress = state.progress.lock().await;
                    progress.last_error = Some(error);
                    drop(progress);
                    state.notify.notify_waiters();
                    break;
                },
            }
        }

        state.notify.notify_waiters();
    }

    async fn write_checkpoint_and_compact_wal(
        &self,
        collection_name: &CollectionName,
        target_sequence: u64,
        checkpoint_state: &CollectionCheckpointState,
    ) -> Result<u64> {
        #[cfg(test)]
        self.maybe_delay_checkpoint_for_tests().await;

        let checkpoint_sequence = if self.loader.snapshot_format == VectorSnapshotFormat::V2 {
            // V2 fast path: borrow the in-memory index directly — no HNSW
            // rebuild, no V1 JSON serialization.  Read lock is held for the
            // duration of the V2 write (~200-400 ms at 100 K vectors).
            let collections = self.collections.read().await;
            let collection = collections.get(collection_name).ok_or_else(|| {
                ErrorEnvelope::expected(ErrorCode::not_found(), "collection not found")
            })?;
            let checkpoint_sequence = collection.last_insert_sequence;
            if checkpoint_sequence < target_sequence {
                return Err(ErrorEnvelope::unexpected(
                    ErrorCode::new("vector", "checkpoint_target_regressed"),
                    "local checkpoint sequence is behind scheduled target",
                    ErrorClass::NonRetriable,
                )
                .with_metadata("collection", collection_name.as_str().to_string())
                .with_metadata("targetSequence", target_sequence.to_string())
                .with_metadata("checkpointSequence", checkpoint_sequence.to_string()));
            }
            if let Some(paths) = self.snapshot_paths(collection_name) {
                self.write_v2_from_collection(
                    collection_name,
                    &paths,
                    collection,
                    self.loader.kernel.kind(),
                    self.loader.snapshot_max_bytes,
                    Some(checkpoint_state),
                )?;
                warm_collection_kernel_state(
                    &self.loader,
                    collection_name,
                    Arc::clone(&collection.index),
                    Some(&paths),
                    true,
                    None,
                )
                .await?;
            }
            drop(collections);
            checkpoint_sequence
        } else {
            // V1 legacy path: snapshot → serialize JSON → rebuild HNSW.
            let snapshot = {
                let collections = self.collections.read().await;
                let snapshot = collections
                    .get(collection_name)
                    .map(LocalCollection::snapshot)
                    .transpose()?;
                drop(collections);
                snapshot.ok_or_else(|| {
                    ErrorEnvelope::expected(ErrorCode::not_found(), "collection not found")
                })?
            };
            let checkpoint_sequence = snapshot.checkpoint_sequence.unwrap_or(0);
            if checkpoint_sequence < target_sequence {
                return Err(ErrorEnvelope::unexpected(
                    ErrorCode::new("vector", "checkpoint_target_regressed"),
                    "local checkpoint sequence is behind scheduled target",
                    ErrorClass::NonRetriable,
                )
                .with_metadata("collection", collection_name.as_str().to_string())
                .with_metadata("targetSequence", target_sequence.to_string())
                .with_metadata("checkpointSequence", checkpoint_sequence.to_string()));
            }
            self.write_snapshot(collection_name, &snapshot).await?;
            checkpoint_sequence
        };

        if let Some(paths) = self.snapshot_paths(collection_name) {
            let checkpoint_state = self.checkpoint_state_for_collection(collection_name).await;
            let _wal_io = checkpoint_state.wal_io.lock().await;
            compact_insert_wal_records(paths.insert_wal.as_path(), checkpoint_sequence).await?;
        }
        Ok(checkpoint_sequence)
    }

    async fn collection_vector_count(&self, collection_name: &CollectionName) -> u64 {
        let collections = self.collections.read().await;
        collections
            .get(collection_name)
            .map_or(0, LocalCollection::vector_count)
    }

    async fn collection_insert_sequence(&self, collection_name: &CollectionName) -> Result<u64> {
        let collections = self.collections.read().await;
        let sequence = collections
            .get(collection_name)
            .map(|collection| collection.last_insert_sequence);
        drop(collections);
        let Some(sequence) = sequence else {
            return Err(ErrorEnvelope::expected(
                ErrorCode::not_found(),
                "collection not found",
            ));
        };
        Ok(sequence)
    }

    async fn wait_for_checkpoint_durable(
        &self,
        ctx: &RequestContext,
        collection_name: &CollectionName,
        target_sequence: u64,
        state: Arc<CollectionCheckpointState>,
    ) -> Result<()> {
        loop {
            ctx.ensure_not_cancelled("vectordb_local.flush")?;
            self.reap_finished_checkpoint_worker(collection_name, &state)
                .await;
            let notified = state.notify.notified();
            {
                let progress = state.progress.lock().await;
                if let Some(error) = progress.last_error.as_ref() {
                    return Err(error
                        .clone()
                        .with_metadata("collection", collection_name.as_str().to_string())
                        .with_metadata("targetSequence", target_sequence.to_string()));
                }
                if progress.durable_sequence >= target_sequence {
                    return Ok(());
                }
            }
            notified.await;
        }
    }

    async fn remove_checkpoint_state(
        &self,
        collection_name: &CollectionName,
    ) -> Option<Arc<CollectionCheckpointState>> {
        let mut states = self.checkpoint_states.write().await;
        states.remove(collection_name)
    }

    async fn stop_checkpoint_worker_for_drop(
        collection_name: &CollectionName,
        state: &CollectionCheckpointState,
    ) -> Result<()> {
        let handle = {
            let mut progress = state.progress.lock().await;
            progress.worker.take()
        };
        let Some(handle) = handle else {
            return Ok(());
        };
        handle.abort();
        match handle.await {
            Ok(()) => Ok(()),
            Err(error) => {
                if error.is_cancelled() {
                    return Ok(());
                }
                Err(ErrorEnvelope::unexpected(
                    ErrorCode::new("vector", "checkpoint_task_failed"),
                    "local checkpoint task failed during collection drop",
                    ErrorClass::NonRetriable,
                )
                .with_metadata("collection", collection_name.as_str().to_string())
                .with_metadata("source", error.to_string()))
            },
        }
    }

    #[cfg(test)]
    fn set_checkpoint_delay_for_tests(&self, delay: Duration) {
        let delay_ms_u128 = delay.as_millis();
        let delay_ms = u64::try_from(delay_ms_u128).unwrap_or(u64::MAX);
        self.checkpoint_delay_ms.store(delay_ms, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn reset_v2_write_path_counters(&self) {
        self.v2_from_collection_write_calls
            .store(0, Ordering::Relaxed);
        self.v2_bundle_write_calls.store(0, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn v2_write_path_calls(&self) -> (usize, usize) {
        (
            self.v2_from_collection_write_calls.load(Ordering::Relaxed),
            self.v2_bundle_write_calls.load(Ordering::Relaxed),
        )
    }

    #[cfg(test)]
    async fn maybe_delay_checkpoint_for_tests(&self) {
        let delay_ms = self.checkpoint_delay_ms.load(Ordering::Relaxed);
        if delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        }
    }

    async fn write_snapshot_json(path: &Path, payload: &[u8]) -> Result<()> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(ErrorEnvelope::from)?;
        }
        tokio::fs::write(path, payload)
            .await
            .map_err(ErrorEnvelope::from)?;
        Ok(())
    }

    /// Write a V2 bundle directly from the in-memory collection, reusing the
    /// already-built HNSW graph instead of rebuilding it from a serialized
    /// snapshot. This eliminates the O(n log n) HNSW reconstruction that
    /// a full-rebuild would require via `from_snapshot`.
    ///
    /// Uses the incremental prepare/write split so only new vectors since the
    /// last checkpoint are quantized (when per-dimension ranges haven't expanded).
    fn write_v2_from_collection(
        &self,
        collection_name: &CollectionName,
        paths: &CollectionSnapshotPaths,
        collection: &LocalCollection,
        kernel: VectorKernelKind,
        snapshot_max_bytes: Option<u64>,
        checkpoint_state: Option<&CollectionCheckpointState>,
    ) -> Result<()> {
        let _ = self;
        #[cfg(test)]
        self.v2_from_collection_write_calls
            .fetch_add(1, Ordering::Relaxed);
        write_v2_from_collection(
            collection_name,
            paths,
            collection,
            kernel,
            snapshot_max_bytes,
            checkpoint_state,
        )
    }
}

/// Write a V2 snapshot bundle from an in-memory collection.
///
/// Shared implementation used by both [`LocalVectorDb::write_v2_from_collection`]
/// (checkpoint path) and [`CollectionLoaderActor::handle_load`] (migration path).
fn write_v2_from_collection(
    collection_name: &CollectionName,
    paths: &CollectionSnapshotPaths,
    collection: &LocalCollection,
    kernel: VectorKernelKind,
    snapshot_max_bytes: Option<u64>,
    checkpoint_state: Option<&CollectionCheckpointState>,
) -> Result<()> {
    V2CollectionCheckpointBuild::new()
        .collect_from_collection(
            collection_name,
            paths,
            collection,
            snapshot_max_bytes,
            checkpoint_state,
        )?
        .persist_bundle(collection_name, paths, kernel)?
        .finalize(paths, collection, checkpoint_state)
}

/// Result of a pure collection load from disk.
///
/// Contains everything the caller needs to complete the load: the assembled
/// collection, checkpoint sequence, and flags describing side-effects (v2
/// rewrites, sidecar migrations) that must happen *outside* the loader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeKernelWarmState {
    NotWarmed,
    WarmedFromGenerationSource,
}

struct CollectionLoadResult {
    collection: LocalCollection,
    checkpoint_sequence: u64,
    /// Snapshot paths (when the storage mode produced them).
    paths: Option<CollectionSnapshotPaths>,
    /// The runtime kernel was already warmed from the active generation source
    /// during load, so the later warm stage can skip a redundant rewarm
    /// unless WAL replay invalidates that state.
    runtime_kernel_warm_state: RuntimeKernelWarmState,
    /// The on-disk kernel differed from the configured kernel and the caller
    /// had to rebuild the host collection graph for correctness.
    kernel_mismatch_requires_rebuild: bool,
    /// The runtime kernel differs from the on-disk metadata and the caller
    /// should rewrite only the kernel metadata after a successful load.
    needs_metadata_rewrite: bool,
    /// The v2 companion is missing or stale and needs to be rebuilt from the
    /// v1 snapshot.
    needs_v2_rebuild: bool,
}

#[derive(Debug, Clone, Copy)]
struct ResolvedV2LoadPlan {
    options: VectorSnapshotV2LoadOptions,
    snapshot_kernel: VectorKernelKind,
    kernel_mismatch_requires_rebuild: bool,
    needs_metadata_rewrite: bool,
}

impl CollectionLoaderContext {
    fn snapshot_root(&self) -> Option<PathBuf> {
        self.storage_mode
            .resolve_root(&self.codebase_root)
            .map(|root| root.join(LOCAL_SNAPSHOT_DIR).join(LOCAL_COLLECTIONS_DIR))
    }

    fn snapshot_paths(&self, collection_name: &CollectionName) -> Option<CollectionSnapshotPaths> {
        let root = self.snapshot_root()?;
        let collection = collection_name.as_str();
        let v1_json = root.join(format!("{collection}.json"));
        let v2_dir = root.join(format!("{collection}{LOCAL_SNAPSHOT_V2_DIR_SUFFIX}"));
        let v2_meta = v2_dir.join(SNAPSHOT_V2_META_FILE_NAME);
        let v2_vectors = v2_dir.join(SNAPSHOT_V2_VECTORS_FILE_NAME);
        let v2_ids = v2_dir.join(LOCAL_SNAPSHOT_V2_IDS_FILE_NAME);
        let v2_records_meta = v2_dir.join(LOCAL_SNAPSHOT_V2_RECORDS_META_FILE_NAME);
        let insert_wal = root.join(format!("{collection}{LOCAL_INSERT_WAL_FILE_SUFFIX}"));
        let generation_layout = CollectionGenerationPaths::new(root.join(collection));
        let build_dir = generation_layout.root().join(LOCAL_BUILD_JOURNAL_DIR_NAME);
        Some(CollectionSnapshotPaths {
            v1_json,
            v2_dir,
            v2_meta,
            v2_vectors,
            v2_ids,
            v2_records_meta,
            insert_wal,
            generation_layout,
            build_meta: build_dir.join(LOCAL_BUILD_JOURNAL_META_FILE_NAME),
            build_rows: build_dir.join(LOCAL_BUILD_JOURNAL_ROWS_FILE_NAME),
            build_vectors: build_dir.join(LOCAL_BUILD_JOURNAL_VECTORS_FILE_NAME),
            build_sealed: build_dir.join(LOCAL_BUILD_JOURNAL_SEALED_FILE_NAME),
        })
    }

    fn runtime_kernel_requires_host_hnsw_graph(&self) -> bool {
        self.kernel
            .kind()
            .load_capabilities()
            .requires_host_hnsw_graph
    }

    fn runtime_kernel_supports_ready_state(&self) -> bool {
        self.kernel
            .kind()
            .load_capabilities()
            .supports_kernel_ready_state()
    }

    fn runtime_dfrr_requires_prewarmed_state(&self) -> bool {
        self.kernel.kind() == VectorKernelKind::Dfrr && self.runtime_dfrr_ready_state.is_some()
    }

    fn runtime_kernel_prefers_generation_source(&self) -> bool {
        self.kernel.kind().load_capabilities().canonical_source_path
            == VectorKernelSourcePathKind::SegmentedSourceV1
    }

    fn build_collection_from_snapshot(
        &self,
        snapshot: CollectionSnapshot,
    ) -> Result<LocalCollection> {
        if self.runtime_kernel_requires_host_hnsw_graph() {
            LocalCollection::from_snapshot(snapshot, self.hnsw_params)
        } else {
            LocalCollection::from_snapshot_records_only(snapshot, self.hnsw_params)
        }
    }

    /// Load a collection from its on-disk snapshot.
    ///
    /// Dispatches to v2 (sidecar + binary bundle) or v1 (legacy JSON) based
    /// on `snapshot_format`.  Returns the loaded collection and metadata about
    /// any side-effects the caller must handle (v2 rewrites, sidecar
    /// migrations) without modifying any shared state.
    async fn load_collection(
        &self,
        collection_name: &CollectionName,
        cancellation: Option<CancellationToken>,
    ) -> Result<CollectionLoadResult> {
        if let Some(result) = self
            .try_load_active_generation(collection_name, cancellation.clone())
            .await?
        {
            return Ok(result);
        }

        match self.snapshot_format {
            VectorSnapshotFormat::V2 => {
                match self
                    .try_load_v2_with_sidecar(collection_name, cancellation)
                    .await?
                {
                    Some(result) => Ok(result),
                    None => self.load_via_v1_json(collection_name).await,
                }
            },
            VectorSnapshotFormat::V1 => self.load_via_v1_json(collection_name).await,
        }
    }

    async fn try_load_active_generation(
        &self,
        collection_name: &CollectionName,
        cancellation: Option<CancellationToken>,
    ) -> Result<Option<CollectionLoadResult>> {
        let Some(paths) = self.snapshot_paths(collection_name) else {
            return Ok(None);
        };

        let Some(active_generation) = paths.generation_layout.read_active_generation_id()? else {
            return Ok(None);
        };
        let generation = paths.generation_layout.generation(&active_generation);
        let sidecar_path = generation
            .base_dir()
            .join(LOCAL_SNAPSHOT_V2_RECORDS_META_FILE_NAME);
        if !path_exists(sidecar_path.as_path()).await? {
            return Ok(None);
        }

        let collection_name_for_sidecar = collection_name.clone();
        let generation_base_for_rows = generation.base_dir().to_path_buf();
        let generation_base_for_sidecar = generation.base_dir().to_path_buf();
        let (exact_rows, sidecar) = tokio::try_join!(
            async {
                spawn_blocking(move || read_exact_generation(generation_base_for_rows))
                    .await
                    .map_err(|join_error| {
                        map_spawn_blocking_join_error(
                            &join_error,
                            ErrorCode::new("vector", "generation_load_task_failed"),
                            "exact generation load",
                            collection_name,
                        )
                    })?
            },
            async {
                spawn_blocking(move || {
                    read_records_meta_sidecar(
                        generation_base_for_sidecar
                            .join(LOCAL_SNAPSHOT_V2_RECORDS_META_FILE_NAME)
                            .as_path(),
                    )
                })
                .await
                .map_err(|join_error| {
                    map_spawn_blocking_join_error(
                        &join_error,
                        ErrorCode::new("vector", "generation_sidecar_load_task_failed"),
                        "exact generation sidecar load",
                        &collection_name_for_sidecar,
                    )
                })?
            },
        )?;
        let generation_source = Arc::new(PublishedGenerationKernelSource::from_exact_rows(
            generation.clone(),
            exact_rows.clone(),
        ));

        if self.runtime_dfrr_requires_prewarmed_state() {
            ensure_runtime_dfrr_ready_state_available(self, collection_name, Some(&paths)).await?;
        }

        let build_host_graph = self.runtime_kernel_requires_host_hnsw_graph();
        // Clone once for the warm step; move original into the load step.
        // Both callees need ownership for spawn_blocking move closures.
        // CancellationToken is Arc-based so clone is cheap.
        let cancellation_for_warm = cancellation.clone();
        let index = load_runtime_index_from_generation_async(
            collection_name,
            &generation,
            exact_rows,
            build_host_graph,
            self.hnsw_params,
            cancellation,
        )
        .await?;
        let (index, runtime_kernel_warm_state) = warm_generation_runtime_index_if_needed(
            self,
            collection_name,
            &generation,
            Some(Arc::clone(&generation_source)),
            index,
            cancellation_for_warm,
        )
        .await?;
        let collection = LocalCollection::from_index_and_parsed_sidecar(index, sidecar)?;

        tracing::info!(
            collection = %collection_name,
            generation = active_generation.as_str(),
            "loaded collection from published active generation"
        );
        Ok(Some(CollectionLoadResult {
            checkpoint_sequence: collection.last_insert_sequence,
            collection,
            paths: Some(paths),
            runtime_kernel_warm_state,
            kernel_mismatch_requires_rebuild: false,
            needs_metadata_rewrite: false,
            needs_v2_rebuild: false,
        }))
    }

    /// v2 fast path: load the index from the v2 binary bundle and metadata
    /// from the JSONL sidecar.  Skips the v1 JSON entirely.
    ///
    /// Returns `None` if the sidecar doesn't exist (caller should fall back).
    async fn try_load_v2_with_sidecar(
        &self,
        collection_name: &CollectionName,
        cancellation: Option<CancellationToken>,
    ) -> Result<Option<CollectionLoadResult>> {
        let Some(paths) = self.snapshot_paths(collection_name) else {
            return Ok(None);
        };
        if !path_exists(&paths.v2_records_meta).await? {
            return Ok(None);
        }
        if self
            .detect_v2_companion_state(collection_name, &paths)
            .await?
            == V2CompanionState::Missing
        {
            return Ok(None);
        }

        let load_plan = self
            .resolve_v2_load_options(collection_name, &paths)
            .await?;

        // Load the v2 index and parse the JSONL sidecar in parallel — they
        // are independent I/O operations and parallelising cuts cold-load
        // time from sum(index, sidecar) to max(index, sidecar).
        let sidecar_path = paths.v2_records_meta.clone();
        let sidecar_collection_name = collection_name.clone();
        let (index_opt, sidecar) = tokio::try_join!(
            async {
                match self
                    .load_index_from_v2_with_options(
                        collection_name,
                        paths.v2_dir.clone(),
                        load_plan.options,
                        cancellation,
                    )
                    .await
                {
                    Ok(index) => Ok(Some(index)),
                    Err(error) if is_v2_companion_repairable_error(&error) => {
                        tracing::warn!(
                            collection = %collection_name,
                            error_code = %error.code,
                            "v2 companion load failed; falling back to v1 JSON"
                        );
                        Ok(None)
                    },
                    Err(error) => Err(error),
                }
            },
            async {
                spawn_blocking(move || read_records_meta_sidecar(&sidecar_path))
                    .await
                    .map_err(|join_error| {
                        map_spawn_blocking_join_error(
                            &join_error,
                            ErrorCode::new("vector", "sidecar_load_task_failed"),
                            "metadata sidecar load",
                            &sidecar_collection_name,
                        )
                    })?
            },
        )?;

        let Some(index) = index_opt else {
            return Ok(None);
        };

        let collection = LocalCollection::from_index_and_parsed_sidecar(index, sidecar)?;

        let checkpoint_sequence = collection.last_insert_sequence;
        tracing::info!(
            collection = %collection_name,
            documents = collection.documents.len(),
            snapshot_kernel = ?load_plan.snapshot_kernel,
            kernel_mismatch_requires_rebuild = load_plan.kernel_mismatch_requires_rebuild,
            needs_metadata_rewrite = load_plan.needs_metadata_rewrite,
            "loaded collection from v2 bundle + metadata sidecar (skipped v1 JSON)"
        );
        Ok(Some(CollectionLoadResult {
            collection,
            checkpoint_sequence,
            paths: Some(paths),
            runtime_kernel_warm_state: RuntimeKernelWarmState::NotWarmed,
            kernel_mismatch_requires_rebuild: load_plan.kernel_mismatch_requires_rebuild,
            needs_metadata_rewrite: load_plan.needs_metadata_rewrite,
            needs_v2_rebuild: false,
        }))
    }

    /// Resolve the v2 load options by comparing the snapshot kernel with the
    /// requested kernel.
    async fn resolve_v2_load_options(
        &self,
        collection_name: &CollectionName,
        paths: &CollectionSnapshotPaths,
    ) -> Result<ResolvedV2LoadPlan> {
        let snapshot_kernel = self
            .read_v2_kernel_metadata(collection_name, paths.v2_meta.clone())
            .await?;
        let runtime_kernel = self.kernel.kind();
        let capabilities = runtime_kernel.load_capabilities();
        let kernel_mismatch = snapshot_kernel != runtime_kernel;
        let kernel_mismatch_requires_rebuild =
            kernel_mismatch && !capabilities.tolerates_snapshot_kernel_mismatch;
        if kernel_mismatch_requires_rebuild && !self.force_reindex_on_kernel_change {
            return Err(snapshot_kernel_mismatch_error(
                collection_name,
                &paths.v2_dir,
                snapshot_kernel,
                runtime_kernel,
            ));
        }
        let needs_metadata_rewrite =
            kernel_mismatch && capabilities.tolerates_snapshot_kernel_mismatch;
        let skip_host_hnsw_graph = !capabilities.requires_host_hnsw_graph;
        match (
            kernel_mismatch_requires_rebuild,
            needs_metadata_rewrite,
            skip_host_hnsw_graph,
        ) {
            (true, _, _) => tracing::info!(
                collection = %collection_name,
                snapshot_kernel = ?snapshot_kernel,
                requested_kernel = ?runtime_kernel,
                "kernel mismatch; loading v2 data with graph rebuild"
            ),
            (_, true, _) => tracing::info!(
                collection = %collection_name,
                snapshot_kernel = ?snapshot_kernel,
                requested_kernel = ?runtime_kernel,
                "kernel mismatch tolerated by runtime kernel; loading records-only and rewriting metadata"
            ),
            (_, _, true) => tracing::info!(
                collection = %collection_name,
                kernel = ?runtime_kernel,
                "runtime kernel does not require host HNSW graph; loading records-only"
            ),
            _ => {},
        }
        Ok(ResolvedV2LoadPlan {
            options: VectorSnapshotV2LoadOptions {
                skip_persisted_graph: kernel_mismatch || skip_host_hnsw_graph,
                skip_graph_build: skip_host_hnsw_graph,
                ..VectorSnapshotV2LoadOptions::default()
            },
            snapshot_kernel,
            kernel_mismatch_requires_rebuild,
            needs_metadata_rewrite,
        })
    }

    /// Legacy load path: read v1 JSON, then optionally use v2 index.
    ///
    /// When running in v2 mode and the metadata sidecar is missing, this also
    /// writes the sidecar as a one-time migration so subsequent loads skip the
    /// v1 JSON.
    async fn load_via_v1_json(
        &self,
        collection_name: &CollectionName,
    ) -> Result<CollectionLoadResult> {
        let snapshot = self.read_snapshot_json(collection_name).await?;
        let Some(snapshot) = snapshot else {
            // No v1 JSON on disk.  When running in V2 mode a V2 snapshot may
            // still contain the dimension — read it from the metadata so we
            // don't create a broken collection with dimension=0.
            let dimension = if self.snapshot_format == VectorSnapshotFormat::V2 {
                self.read_v2_dimension_hint(collection_name).await
            } else {
                0
            };
            let collection = LocalCollection::new(dimension, IndexMode::Dense, self.hnsw_params)?;
            return Ok(CollectionLoadResult {
                collection,
                checkpoint_sequence: 0,
                paths: self.snapshot_paths(collection_name),
                runtime_kernel_warm_state: RuntimeKernelWarmState::NotWarmed,
                kernel_mismatch_requires_rebuild: false,
                needs_metadata_rewrite: false,
                needs_v2_rebuild: false,
            });
        };
        let checkpoint_sequence = snapshot.checkpoint_sequence.unwrap_or(0);

        // Auto-migrate: write sidecar so subsequent loads skip the v1 JSON.
        if self.snapshot_format == VectorSnapshotFormat::V2
            && let Some(paths) = self.snapshot_paths(collection_name)
            && !paths.v2_records_meta.exists()
            && !snapshot.records.is_empty()
        {
            match write_records_meta_sidecar(&paths, &snapshot) {
                Ok(()) => {
                    tracing::info!(
                        collection = %collection_name,
                        records = snapshot.records.len(),
                        "auto-migrated: wrote records.meta.jsonl sidecar from v1 JSON"
                    );
                },
                Err(error) => {
                    tracing::warn!(
                        collection = %collection_name,
                        %error,
                        "auto-migration of records.meta.jsonl sidecar failed (non-fatal)"
                    );
                },
            }
        }

        let (collection, needs_v2_rebuild, needs_metadata_rewrite) = match self.snapshot_format {
            VectorSnapshotFormat::V1 => (
                self.build_collection_from_snapshot(snapshot.clone())?,
                false,
                false,
            ),
            VectorSnapshotFormat::V2 => {
                self.load_collection_v2(collection_name, snapshot.clone())
                    .await?
            },
        };
        Ok(CollectionLoadResult {
            collection,
            checkpoint_sequence,
            paths: self.snapshot_paths(collection_name),
            runtime_kernel_warm_state: RuntimeKernelWarmState::NotWarmed,
            kernel_mismatch_requires_rebuild: false,
            needs_metadata_rewrite,
            needs_v2_rebuild,
        })
    }

    async fn read_snapshot_json(
        &self,
        collection_name: &CollectionName,
    ) -> Result<Option<CollectionSnapshot>> {
        let Some(paths) = self.snapshot_paths(collection_name) else {
            return Ok(None);
        };
        let path = paths.v1_json;

        match tokio::fs::read(path).await {
            Ok(payload) => {
                let snapshot = serde_json::from_slice(&payload).map_err(|error| {
                    snapshot_error("snapshot_parse_failed", "failed to parse snapshot", error)
                })?;
                Ok(Some(snapshot))
            },
            Err(error) => {
                if error.kind() == std::io::ErrorKind::NotFound {
                    Ok(None)
                } else {
                    Err(ErrorEnvelope::from(error))
                }
            },
        }
    }

    /// Load a collection in v2 mode from a v1 JSON snapshot, using the v2
    /// index if available.
    ///
    /// Returns `(collection, needs_v2_rebuild, needs_metadata_rewrite)`. When
    /// `needs_v2_rebuild` is `true`, the caller must write a v2 bundle from the
    /// original snapshot. Metadata rewrite is only used when the runtime kernel
    /// can tolerate the on-disk kernel mismatch without rebuilding the host
    /// graph.
    async fn load_collection_v2(
        &self,
        collection_name: &CollectionName,
        snapshot: CollectionSnapshot,
    ) -> Result<(LocalCollection, bool, bool)> {
        let Some(paths) = self.snapshot_paths(collection_name) else {
            let collection = self.build_collection_from_snapshot(snapshot)?;
            return Ok((collection, false, false));
        };

        match self
            .detect_v2_companion_state(collection_name, &paths)
            .await?
        {
            V2CompanionState::Present => {
                let load_plan = self
                    .resolve_v2_load_options(collection_name, &paths)
                    .await?;
                if load_plan.kernel_mismatch_requires_rebuild {
                    let collection = self.build_collection_from_snapshot(snapshot)?;
                    return Ok((collection, true, false));
                }
                let index = match self
                    .load_index_from_v2_with_options(
                        collection_name,
                        paths.v2_dir.clone(),
                        load_plan.options,
                        None,
                    )
                    .await
                {
                    Ok(index) => index,
                    Err(error) if is_v2_companion_repairable_error(&error) => {
                        tracing::warn!(
                            collection = %collection_name,
                            error_code = %error.code,
                            "v2 companion load failed; rebuilding from v1 snapshot"
                        );
                        let collection = self.build_collection_from_snapshot(snapshot)?;
                        return Ok((collection, true, false));
                    },
                    Err(error) => return Err(error),
                };
                if let Some(missing_id) = snapshot
                    .records
                    .iter()
                    .find(|record| index.record_for_id(record.id.as_ref()).is_none())
                    .map(|record| record.id.clone())
                {
                    tracing::warn!(
                        collection = %collection_name,
                        missing_id = %missing_id,
                        "v2 companion is stale relative to v1 snapshot; rebuilding from v1 snapshot"
                    );
                    let collection = self.build_collection_from_snapshot(snapshot)?;
                    return Ok((collection, true, false));
                }
                let collection = LocalCollection::from_snapshot_with_index(snapshot, index)?;
                Ok((collection, false, load_plan.needs_metadata_rewrite))
            },
            V2CompanionState::Missing => {
                let collection = self.build_collection_from_snapshot(snapshot)?;
                Ok((collection, true, false))
            },
        }
    }

    async fn read_v2_kernel_metadata(
        &self,
        collection_name: &CollectionName,
        meta_path: PathBuf,
    ) -> Result<VectorKernelKind> {
        let meta_path_for_task = meta_path.clone();
        let meta = spawn_blocking(move || read_metadata(meta_path_for_task))
            .await
            .map_err(|join_error| {
                map_spawn_blocking_join_error(
                    &join_error,
                    ErrorCode::new("vector", "snapshot_load_task_failed"),
                    "snapshot v2 metadata read",
                    collection_name,
                )
                .with_metadata("path", meta_path.display().to_string())
            })?
            .map_err(|error| {
                ErrorEnvelope::expected(
                    ErrorCode::new("vector", "snapshot_invalid"),
                    "failed to read snapshot v2 metadata",
                )
                .with_metadata("collection", collection_name.as_str().to_string())
                .with_metadata("path", meta_path.display().to_string())
                .with_metadata("source", error.to_string())
            })?;
        Ok(meta.kernel)
    }

    /// Best-effort dimension read from V2 snapshot metadata.
    ///
    /// Returns 0 when the metadata cannot be read (e.g. no V2 snapshot
    /// exists yet).  Callers should treat 0 as "unknown dimension".
    async fn read_v2_dimension_hint(&self, collection_name: &CollectionName) -> u32 {
        let Some(paths) = self.snapshot_paths(collection_name) else {
            return 0;
        };
        let meta_path = paths.v2_meta;
        match spawn_blocking(move || read_metadata(meta_path)).await {
            Ok(Ok(meta)) => meta.dimension,
            _ => 0,
        }
    }

    async fn detect_v2_companion_state(
        &self,
        collection_name: &CollectionName,
        paths: &CollectionSnapshotPaths,
    ) -> Result<V2CompanionState> {
        let meta_exists = path_exists(&paths.v2_meta).await?;
        let vectors_exists = path_exists(&paths.v2_vectors).await?;
        let ids_exists = path_exists(&paths.v2_ids).await?;

        if !meta_exists && !vectors_exists && !ids_exists {
            return Ok(V2CompanionState::Missing);
        }
        if meta_exists && vectors_exists && ids_exists {
            return Ok(V2CompanionState::Present);
        }

        let mut missing = Vec::new();
        if !meta_exists {
            missing.push(SNAPSHOT_V2_META_FILE_NAME);
        }
        if !vectors_exists {
            missing.push(SNAPSHOT_V2_VECTORS_FILE_NAME);
        }
        if !ids_exists {
            missing.push(LOCAL_SNAPSHOT_V2_IDS_FILE_NAME);
        }
        Err(snapshot_missing_companion_error(
            collection_name,
            &paths.v2_dir,
            missing.as_slice(),
        ))
    }

    async fn load_index_from_v2_with_options(
        &self,
        collection_name: &CollectionName,
        snapshot_dir: PathBuf,
        options: VectorSnapshotV2LoadOptions,
        cancellation: Option<CancellationToken>,
    ) -> Result<VectorIndex> {
        let snapshot_dir_for_task = snapshot_dir.clone();
        spawn_blocking(move || {
            VectorIndex::from_snapshot_v2_with_options(
                snapshot_dir_for_task,
                options,
                cancellation.as_ref(),
            )
        })
        .await
        .map_err(|join_error| {
            map_spawn_blocking_join_error(
                &join_error,
                ErrorCode::new("vector", "snapshot_load_task_failed"),
                "snapshot v2 load",
                collection_name,
            )
            .with_metadata("snapshotDir", snapshot_dir.display().to_string())
        })?
    }
}

impl VectorDbPort for LocalVectorDb {
    fn provider(&self) -> &VectorDbProviderInfo {
        &self.provider
    }

    fn create_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        dimension: u32,
        _description: Option<Box<str>>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let db = self.clone();
        let collection = collection_name.as_str().to_owned();
        let span = tracing::info_span!(
            "adapter.vectordb.local.create_collection",
            collection = %collection,
            dimension
        );
        Box::pin(
            async move {
                ctx.ensure_not_cancelled("vectordb_local.create_collection")?;
                let collection = if db.loader.snapshot_format == VectorSnapshotFormat::V2 {
                    LocalCollection::new_staging(
                        dimension,
                        IndexMode::Dense,
                        db.loader.hnsw_params,
                    )?
                } else {
                    LocalCollection::new(dimension, IndexMode::Dense, db.loader.hnsw_params)?
                };
                let mut guard = db.collections.write().await;
                guard.insert(collection_name.clone(), collection);
                let generation_layout = db
                    .snapshot_paths(&collection_name)
                    .map(|paths| paths.generation_layout);
                drop(guard);
                if let Some(generation_layout) = generation_layout {
                    db.build_coordinator
                        .ensure_scaffold(collection_name.clone(), generation_layout)
                        .await?;
                }
                let guard = db.collections.write().await;
                if db.loader.snapshot_format == VectorSnapshotFormat::V2 {
                    if let (Some(paths), Some(coll)) = (
                        db.snapshot_paths(&collection_name),
                        guard.get(&collection_name),
                    ) {
                        db.write_v2_from_collection(
                            &collection_name,
                            &paths,
                            coll,
                            db.loader.kernel.kind(),
                            db.loader.snapshot_max_bytes,
                            None,
                        )?;
                    }
                    drop(guard);
                } else {
                    let snapshot = guard
                        .get(&collection_name)
                        .map(LocalCollection::snapshot)
                        .transpose()?;
                    drop(guard);
                    if let Some(snapshot) = snapshot {
                        db.write_snapshot(&collection_name, &snapshot).await?;
                    }
                }
                Ok(())
            }
            .instrument(span),
        )
    }

    fn create_hybrid_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        dimension: u32,
        _description: Option<Box<str>>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let db = self.clone();
        let collection = collection_name.as_str().to_owned();
        let span = tracing::info_span!(
            "adapter.vectordb.local.create_hybrid_collection",
            collection = %collection,
            dimension
        );
        Box::pin(
            async move {
                ctx.ensure_not_cancelled("vectordb_local.create_hybrid_collection")?;
                let collection = if db.loader.snapshot_format == VectorSnapshotFormat::V2 {
                    LocalCollection::new_staging(
                        dimension,
                        IndexMode::Hybrid,
                        db.loader.hnsw_params,
                    )?
                } else {
                    LocalCollection::new(dimension, IndexMode::Hybrid, db.loader.hnsw_params)?
                };
                let mut guard = db.collections.write().await;
                guard.insert(collection_name.clone(), collection);
                let generation_layout = db
                    .snapshot_paths(&collection_name)
                    .map(|paths| paths.generation_layout);
                drop(guard);
                if let Some(generation_layout) = generation_layout {
                    db.build_coordinator
                        .ensure_scaffold(collection_name.clone(), generation_layout)
                        .await?;
                }
                let guard = db.collections.write().await;
                if db.loader.snapshot_format == VectorSnapshotFormat::V2 {
                    if let (Some(paths), Some(coll)) = (
                        db.snapshot_paths(&collection_name),
                        guard.get(&collection_name),
                    ) {
                        db.write_v2_from_collection(
                            &collection_name,
                            &paths,
                            coll,
                            db.loader.kernel.kind(),
                            db.loader.snapshot_max_bytes,
                            None,
                        )?;
                    }
                    drop(guard);
                } else {
                    let snapshot = guard
                        .get(&collection_name)
                        .map(LocalCollection::snapshot)
                        .transpose()?;
                    drop(guard);
                    if let Some(snapshot) = snapshot {
                        db.write_snapshot(&collection_name, &snapshot).await?;
                    }
                }
                Ok(())
            }
            .instrument(span),
        )
    }

    fn drop_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let db = self.clone();
        let snapshot = self.snapshot_paths(&collection_name);
        let collection = collection_name.as_str().to_owned();
        let span = tracing::info_span!(
            "adapter.vectordb.local.drop_collection",
            collection = %collection
        );
        Box::pin(
            async move {
                ctx.ensure_not_cancelled("vectordb_local.drop_collection")?;
                let mut guard = db.collections.write().await;
                guard.remove(&collection_name);
                drop(guard);

                // Notify the actor so it can purge internal state.
                let _ = db.loader_handle.evict(collection_name.clone()).await;

                if let Some(state) = db.remove_checkpoint_state(&collection_name).await {
                    Self::stop_checkpoint_worker_for_drop(&collection_name, state.as_ref()).await?;
                }

                if let Some(paths) = snapshot {
                    db.build_coordinator
                        .drop_collection(collection_name.clone(), paths.generation_layout.clone())
                        .await?;
                    match tokio::fs::remove_file(paths.v1_json.as_path()).await {
                        Ok(()) => (),
                        Err(error) => {
                            if error.kind() != std::io::ErrorKind::NotFound {
                                return Err(ErrorEnvelope::from(error));
                            }
                        },
                    }
                    match tokio::fs::remove_file(paths.insert_wal.as_path()).await {
                        Ok(()) => (),
                        Err(error) => {
                            if error.kind() != std::io::ErrorKind::NotFound {
                                return Err(ErrorEnvelope::from(error));
                            }
                        },
                    }
                    match tokio::fs::remove_dir_all(paths.v2_dir.as_path()).await {
                        Ok(()) => (),
                        Err(error) => {
                            if error.kind() != std::io::ErrorKind::NotFound {
                                return Err(ErrorEnvelope::from(error));
                            }
                        },
                    }
                }
                Ok(())
            }
            .instrument(span),
        )
    }

    fn has_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
    ) -> semantic_code_ports::BoxFuture<'_, Result<bool>> {
        let ctx = ctx.clone();
        let collections = Arc::clone(&self.collections);
        let snapshot = self.snapshot_paths(&collection_name);
        let collection = collection_name.as_str().to_owned();
        let span = tracing::info_span!(
            "adapter.vectordb.local.has_collection",
            collection = %collection
        );
        Box::pin(
            async move {
                ctx.ensure_not_cancelled("vectordb_local.has_collection")?;
                let guard = collections.read().await;
                if guard.contains_key(&collection_name) {
                    return Ok(true);
                }
                drop(guard);

                let Some(paths) = snapshot else {
                    return Ok(false);
                };

                // Check v2 sidecar first (primary format), then v1 JSON
                // as legacy fallback.
                if path_exists(paths.v2_records_meta.as_path()).await? {
                    return Ok(true);
                }

                path_exists(paths.v1_json.as_path()).await
            }
            .instrument(span),
        )
    }

    fn list_collections(
        &self,
        ctx: &RequestContext,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<CollectionName>>> {
        let ctx = ctx.clone();
        let collections = Arc::clone(&self.collections);
        let snapshot_root = self.snapshot_root();
        let span = tracing::info_span!("adapter.vectordb.local.list_collections");
        Box::pin(
            async move {
                ctx.ensure_not_cancelled("vectordb_local.list_collections")?;
                let guard = collections.read().await;
                let mut names: BTreeMap<Box<str>, CollectionName> = guard
                    .keys()
                    .map(|name| (name.as_str().into(), name.clone()))
                    .collect();
                drop(guard);

                let Some(root) = snapshot_root else {
                    return Ok(names.into_values().collect());
                };

                let mut dir = match tokio::fs::read_dir(&root).await {
                    Ok(dir) => dir,
                    Err(error) => {
                        if error.kind() == std::io::ErrorKind::NotFound {
                            return Ok(names.into_values().collect());
                        }
                        return Err(ErrorEnvelope::from(error));
                    },
                };

                while let Some(entry) = dir.next_entry().await.map_err(ErrorEnvelope::from)? {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if let Some(collection) = collection_name_from_filename(&name) {
                        names
                            .entry(collection.as_str().into())
                            .or_insert(collection);
                    }
                }

                Ok(names.into_values().collect())
            }
            .instrument(span),
        )
    }

    fn insert(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let db = self.clone();
        let doc_count = documents.len();
        let collection = collection_name.as_str().to_owned();
        let span = tracing::info_span!(
            "adapter.vectordb.local.insert",
            collection = %collection,
            doc_count
        );
        Box::pin(
            async move {
                ctx.ensure_not_cancelled("vectordb_local.insert")?;
                db.ensure_loaded(&collection_name).await?;
                let build_admission = if db.loader.snapshot_format == VectorSnapshotFormat::V2
                    && db.snapshot_paths(&collection_name).is_some()
                {
                    let collections = db.collections.read().await;
                    let Some(collection) = collections.get(&collection_name) else {
                        return Err(ErrorEnvelope::expected(
                            ErrorCode::not_found(),
                            "collection not found",
                        ));
                    };
                    let should_track = collection.is_staging();
                    drop(collections);
                    if should_track {
                        Some(
                            db.build_coordinator
                                .begin_journal_append(collection_name.clone())
                                .await?,
                        )
                    } else {
                        None
                    }
                } else {
                    None
                };
                let mut guard = db.collections.write().await;
                let Some(collection) = guard.get_mut(&collection_name) else {
                    return Err(ErrorEnvelope::expected(
                        ErrorCode::not_found(),
                        "collection not found",
                    ));
                };

                let wal_record = collection.insert(documents)?;
                let vector_count = collection.vector_count();
                let staged_v2 = collection.is_staging()
                    && db.loader.snapshot_format == VectorSnapshotFormat::V2;
                drop(guard);
                if staged_v2 {
                    if let Some(paths) = db.snapshot_paths(&collection_name) {
                        let append_result = append_build_journal_record(&paths, &wal_record).await;
                        drop(build_admission);
                        append_result?;
                    }
                } else {
                    db.append_insert_wal(&collection_name, &wal_record).await?;
                    db.schedule_checkpoint(
                        &collection_name,
                        wal_record.sequence,
                        vector_count,
                        false,
                    )
                    .await;
                }
                Ok(())
            }
            .instrument(span),
        )
    }

    fn insert_hybrid(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        self.insert(ctx, collection_name, documents)
    }

    fn flush(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let db = self.clone();
        let collection = collection_name.as_str().to_owned();
        let span = tracing::info_span!("adapter.vectordb.local.flush", collection = %collection);
        Box::pin(
            async move {
                ctx.ensure_not_cancelled("vectordb_local.flush")?;
                db.ensure_loaded(&collection_name).await?;
                db.finalize_staged_collection(&collection_name).await?;
                let target_sequence = db.collection_insert_sequence(&collection_name).await?;
                if target_sequence == 0 {
                    return Ok(());
                }

                let vector_count = db.collection_vector_count(&collection_name).await;
                let checkpoint_state = db
                    .schedule_checkpoint(&collection_name, target_sequence, vector_count, true)
                    .await;
                db.wait_for_checkpoint_durable(
                    &ctx,
                    &collection_name,
                    target_sequence,
                    checkpoint_state,
                )
                .await
            }
            .instrument(span),
        )
    }

    fn search(
        &self,
        ctx: &RequestContext,
        request: VectorSearchRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<VectorSearchResponse>> {
        let ctx = ctx.clone();
        let db = self.clone();
        let VectorSearchRequest {
            collection_name,
            query_vector,
            options,
        } = request;
        let requested_top_k = options.top_k.unwrap_or(10).max(1);
        let has_threshold = options.threshold.is_some();
        let has_filter = options.filter_expr.is_some();
        let collection = collection_name.as_str().to_owned();
        let span = tracing::info_span!(
            "adapter.vectordb.local.search",
            collection = %collection,
            top_k = requested_top_k,
            has_threshold,
            has_filter
        );
        Box::pin(
            async move {
                ctx.ensure_not_cancelled("vectordb_local.search")?;
                db.ensure_loaded(&collection_name).await?;
                let top_k = options.top_k.unwrap_or(10).max(1) as usize;
                let threshold = options.threshold;
                let filter = parse_filter_expr(options.filter_expr.as_deref())?;
                let search_limit = local_search_limit(top_k, filter.is_some(), threshold);

                let response = {
                    let guard = db.collections.read().await;
                    let Some(collection) = guard.get(&collection_name) else {
                        return Err(ErrorEnvelope::expected(
                            ErrorCode::not_found(),
                            "collection not found",
                        ));
                    };

                    let (search_output, stats) = {
                        let index = collection.read_index()?;
                        let search_output = index.search_with_kernel(
                            query_vector.as_ref(),
                            search_limit,
                            &*db.loader.kernel,
                            db.loader.search_backend,
                        )?;
                        let search_stats = search_output.stats.clone();

                        tracing::debug!(
                            kernel = ?db.loader.kernel.kind(),
                            backend = ?db.loader.search_backend,
                            top_k,
                            match_count = search_output.matches.len(),
                            "adapter.vectordb.local.search_completed"
                        );

                        let index_size = u64::try_from(index.active_count()).ok();
                        drop(index);
                        (
                            search_output,
                            Some(SearchStats {
                                expansions: search_stats.expansions,
                                kernel: kernel_kind_name(search_stats.kernel).into(),
                                extra: search_stats.extra,
                                kernel_search_duration_ns: search_stats.kernel_search_duration_ns,
                                index_size,
                            }),
                        )
                    };

                    let mut results = Vec::new();
                    for candidate in search_output.matches {
                        let Some(doc) = collection.documents.get(candidate.id.as_ref()) else {
                            continue;
                        };
                        if !filter_matches(filter.as_ref(), doc) {
                            continue;
                        }
                        let score = candidate.score;
                        if threshold.is_some_and(|value| score < value) {
                            continue;
                        }
                        results.push(VectorSearchResult {
                            document: VectorDocument {
                                id: candidate.id,
                                vector: None,
                                content: doc.content.clone(),
                                metadata: doc.metadata.clone(),
                            },
                            score,
                        });
                        if results.len() >= top_k {
                            break;
                        }
                    }
                    drop(guard);

                    VectorSearchResponse { results, stats }
                };

                Ok(response)
            }
            .instrument(span),
        )
    }

    fn hybrid_search(
        &self,
        ctx: &RequestContext,
        request: HybridSearchBatchRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<HybridSearchResult>>> {
        let ctx = ctx.clone();
        let db = self.clone();
        let HybridSearchBatchRequest {
            collection_name,
            search_requests,
            options,
        } = request;
        let request_count = search_requests.len();
        let collection = collection_name.as_str().to_owned();
        let span = tracing::info_span!(
            "adapter.vectordb.local.hybrid_search",
            collection = %collection,
            request_count
        );
        Box::pin(
            async move {
                ctx.ensure_not_cancelled("vectordb_local.hybrid_search")?;
                db.ensure_loaded(&collection_name).await?;
                let mut merged: HashMap<Box<str>, HybridSearchResult> = HashMap::new();
                let global_limit = options.limit.map(|value| value.max(1) as usize);
                let filter = parse_filter_expr(options.filter_expr.as_deref())?;

                {
                    let guard = db.collections.read().await;
                    let Some(collection) = guard.get(&collection_name) else {
                        return Err(ErrorEnvelope::expected(
                            ErrorCode::not_found(),
                            "collection not found",
                        ));
                    };

                    for req in search_requests {
                        let limit = req.limit.max(1) as usize;
                        let query = match req.data {
                            HybridSearchData::DenseVector(vector) => vector,
                            HybridSearchData::SparseQuery(_) => {
                                continue;
                            },
                        };
                        let search_output = {
                            let index = collection.read_index()?;
                            index.search_with_kernel(
                                query.as_ref(),
                                limit.saturating_mul(5),
                                &*db.loader.kernel,
                                db.loader.search_backend,
                            )?
                        };

                        for candidate in search_output.matches {
                            let Some(doc) = collection.documents.get(candidate.id.as_ref()) else {
                                continue;
                            };
                            if !filter_matches(filter.as_ref(), doc) {
                                continue;
                            }
                            let entry = merged.entry(candidate.id.clone()).or_insert_with(|| {
                                HybridSearchResult {
                                    document: VectorDocument {
                                        id: candidate.id.clone(),
                                        vector: None,
                                        content: doc.content.clone(),
                                        metadata: doc.metadata.clone(),
                                    },
                                    score: candidate.score,
                                }
                            });
                            if candidate.score > entry.score {
                                entry.score = candidate.score;
                            }
                        }
                    }
                    drop(guard);
                }

                let mut out: Vec<HybridSearchResult> = merged.into_values().collect();
                out.sort_by(|a, b| {
                    let score = b.score.total_cmp(&a.score);
                    if score != std::cmp::Ordering::Equal {
                        return score;
                    }
                    a.document.id.cmp(&b.document.id)
                });

                if let Some(limit) = global_limit {
                    out.truncate(limit);
                }

                Ok(out)
            }
            .instrument(span),
        )
    }

    fn delete(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        ids: Vec<Box<str>>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let db = self.clone();
        let id_count = ids.len();
        let collection = collection_name.as_str().to_owned();
        let span = tracing::info_span!(
            "adapter.vectordb.local.delete",
            collection = %collection,
            id_count
        );
        Box::pin(
            async move {
                ctx.ensure_not_cancelled("vectordb_local.delete")?;
                db.ensure_loaded(&collection_name).await?;
                let mut guard = db.collections.write().await;
                let Some(collection) = guard.get_mut(&collection_name) else {
                    return Err(ErrorEnvelope::expected(
                        ErrorCode::not_found(),
                        "collection not found",
                    ));
                };
                collection.delete(&ids)?;
                if db.loader.snapshot_format == VectorSnapshotFormat::V2 {
                    let index_handle = Arc::clone(&collection.index);
                    let paths = db.snapshot_paths(&collection_name);
                    if let Some(ref paths) = paths {
                        db.write_v2_from_collection(
                            &collection_name,
                            paths,
                            collection,
                            db.loader.kernel.kind(),
                            db.loader.snapshot_max_bytes,
                            None,
                        )?;
                    }
                    drop(guard);
                    warm_collection_kernel_state(
                        &db.loader,
                        &collection_name,
                        index_handle,
                        paths.as_ref(),
                        true,
                        None,
                    )
                    .await?;
                } else {
                    let snapshot = collection.snapshot()?;
                    drop(guard);
                    db.write_snapshot(&collection_name, &snapshot).await?;
                }
                Ok(())
            }
            .instrument(span),
        )
    }

    fn query(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        filter: Box<str>,
        output_fields: Vec<Box<str>>,
        limit: Option<u32>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<VectorDbRow>>> {
        let ctx = ctx.clone();
        let db = self.clone();
        let field_count = output_fields.len();
        let collection = collection_name.as_str().to_owned();
        let span = tracing::info_span!(
            "adapter.vectordb.local.query",
            collection = %collection,
            field_count,
            limit = limit.unwrap_or(0)
        );
        Box::pin(
            async move {
                ctx.ensure_not_cancelled("vectordb_local.query")?;
                db.ensure_loaded(&collection_name).await?;
                let limit = limit.map(|value| value.max(1) as usize);
                let filter = parse_filter_expr(Some(filter.as_ref()))?;

                let rows = {
                    let guard = db.collections.read().await;
                    let Some(collection) = guard.get(&collection_name) else {
                        return Err(ErrorEnvelope::expected(
                            ErrorCode::not_found(),
                            "collection not found",
                        ));
                    };
                    let mut rows = Vec::new();
                    for (id, doc) in &collection.documents {
                        if !filter_matches(filter.as_ref(), doc) {
                            continue;
                        }
                        rows.push(build_row(id, doc, &output_fields));
                        if limit.is_some_and(|value| rows.len() >= value) {
                            break;
                        }
                    }
                    drop(guard);
                    rows
                };

                Ok(rows)
            }
            .instrument(span),
        )
    }
}

fn local_search_limit(top_k: usize, has_filter: bool, threshold: Option<f32>) -> usize {
    if has_filter || threshold.is_some_and(|value| value > 0.0) {
        top_k.saturating_mul(5)
    } else {
        top_k
    }
}

impl Clone for LocalVectorDb {
    fn clone(&self) -> Self {
        Self {
            provider: self.provider.clone(),
            loader: self.loader.clone(),
            collections: Arc::clone(&self.collections),
            checkpoint_states: Arc::clone(&self.checkpoint_states),
            checkpoint_divisor: self.checkpoint_divisor,
            build_coordinator: self.build_coordinator.clone(),
            loader_handle: self.loader_handle.clone(),
            #[cfg(test)]
            v2_from_collection_write_calls: Arc::clone(&self.v2_from_collection_write_calls),
            #[cfg(test)]
            v2_bundle_write_calls: Arc::clone(&self.v2_bundle_write_calls),
            #[cfg(test)]
            checkpoint_delay_ms: Arc::clone(&self.checkpoint_delay_ms),
        }
    }
}

const fn resolve_search_backend(strategy: VectorSearchStrategy) -> VectorSearchBackend {
    match strategy {
        VectorSearchStrategy::F32Hnsw => VectorSearchBackend::F32Hnsw,
        VectorSearchStrategy::U8Exact => VectorSearchBackend::ExperimentalU8Quantized,
        VectorSearchStrategy::U8ThenF32Rerank => VectorSearchBackend::ExperimentalU8ThenF32Rerank,
    }
}

async fn rewrite_v2_kernel_metadata(
    loader: &CollectionLoaderContext,
    collection_name: &CollectionName,
    paths: &CollectionSnapshotPaths,
    kernel: VectorKernelKind,
) -> Result<()> {
    let meta_path = paths.v2_meta.clone();
    let meta_path_for_task = meta_path.clone();
    spawn_blocking(move || {
        let mut meta = read_metadata(meta_path_for_task.as_path()).map_err(|error| {
            ErrorEnvelope::expected(
                ErrorCode::new("vector", "snapshot_invalid"),
                "failed to read snapshot v2 metadata for kernel rewrite",
            )
            .with_metadata("path", meta_path_for_task.display().to_string())
            .with_metadata("source", error.to_string())
        })?;
        meta.kernel = kernel;
        semantic_code_vector::write_metadata(meta_path_for_task.as_path(), &meta).map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "snapshot_kernel_metadata_write_failed"),
                "failed to rewrite snapshot kernel metadata",
                ErrorClass::NonRetriable,
            )
            .with_metadata("path", meta_path_for_task.display().to_string())
            .with_metadata("source", error.to_string())
        })
    })
    .await
    .map_err(|join_error| {
        map_spawn_blocking_join_error(
            &join_error,
            ErrorCode::new("vector", "snapshot_load_task_failed"),
            "snapshot kernel metadata rewrite",
            collection_name,
        )
        .with_metadata("snapshotDir", paths.v2_dir.display().to_string())
    })??;

    tracing::info!(
        collection = %collection_name,
        kernel = ?loader.kernel.kind(),
        snapshot_dir = %paths.v2_dir.display(),
        "rewrote v2 kernel metadata after tolerant load"
    );
    Ok(())
}

async fn warm_collection_kernel_state(
    loader: &CollectionLoaderContext,
    collection_name: &CollectionName,
    index: Arc<StdRwLock<VectorIndex>>,
    paths: Option<&CollectionSnapshotPaths>,
    allow_persist: bool,
    cancellation: Option<CancellationToken>,
) -> Result<()> {
    let snapshot_dir = resolve_kernel_snapshot_dir(loader, paths)?;
    warm_collection_kernel_state_at_path(
        loader,
        collection_name,
        index,
        snapshot_dir,
        allow_persist,
        cancellation,
    )
    .await
}

async fn warm_collection_kernel_state_at_path(
    loader: &CollectionLoaderContext,
    collection_name: &CollectionName,
    index: Arc<StdRwLock<VectorIndex>>,
    snapshot_dir: Option<PathBuf>,
    allow_persist: bool,
    cancellation: Option<CancellationToken>,
) -> Result<()> {
    warm_kernel_ready_state_at_path(
        Arc::clone(&loader.kernel),
        collection_name,
        index,
        snapshot_dir,
        allow_persist,
        cancellation,
    )
    .await
}

async fn warm_kernel_ready_state_at_path(
    kernel: Arc<dyn VectorKernel + Send + Sync>,
    collection_name: &CollectionName,
    index: Arc<StdRwLock<VectorIndex>>,
    snapshot_dir: Option<PathBuf>,
    allow_persist: bool,
    cancellation: Option<CancellationToken>,
) -> Result<()> {
    let context =
        VectorKernelWarmContext::new(collection_name.as_str(), snapshot_dir, allow_persist);

    if !kernel
        .kind()
        .load_capabilities()
        .supports_kernel_ready_state()
    {
        let guard = index.read().map_err(|_| index_lock_error("read"))?;
        return kernel.warm(&guard, &context, cancellation.as_ref());
    }

    let collection_for_task = collection_name.clone();
    spawn_blocking(move || {
        let guard = index.read().map_err(|_| index_lock_error("read"))?;
        kernel.warm(&guard, &context, cancellation.as_ref())
    })
    .await
    .map_err(|join_error| {
        map_spawn_blocking_join_error(
            &join_error,
            ErrorCode::new("vector", "kernel_ready_state_task_failed"),
            "kernel ready-state warm",
            &collection_for_task,
        )
    })?
}

async fn load_generation_kernel_source(
    collection_name: &CollectionName,
    generation: &semantic_code_vector::PublishedGenerationPaths,
    cancellation: Option<CancellationToken>,
) -> Result<PublishedGenerationKernelSource> {
    let collection_for_task = collection_name.clone();
    let generation_for_task = generation.clone();
    spawn_blocking(move || {
        PublishedGenerationKernelSource::load_cancellable(
            &generation_for_task,
            cancellation.as_ref(),
        )
    })
    .await
    .map_err(|join_error| {
        map_spawn_blocking_join_error(
            &join_error,
            ErrorCode::new("vector", "generation_source_load_task_failed"),
            "published generation source load",
            &collection_for_task,
        )
    })?
}

async fn warm_loaded_generation_source_at_path(
    kernel: Arc<dyn VectorKernel + Send + Sync>,
    collection_name: &CollectionName,
    index: Arc<StdRwLock<VectorIndex>>,
    source: Arc<PublishedGenerationKernelSource>,
    snapshot_dir: Option<PathBuf>,
    allow_persist: bool,
    cancellation: Option<CancellationToken>,
) -> Result<()> {
    let context =
        VectorKernelWarmContext::new(collection_name.as_str(), snapshot_dir, allow_persist);
    let collection_for_task = collection_name.clone();

    spawn_blocking(move || {
        let guard = index.read().map_err(|_| index_lock_error("read"))?;
        kernel.warm_from_source(
            &guard,
            VectorKernelWarmSource::PublishedGeneration(source.as_ref()),
            &context,
            cancellation.as_ref(),
        )
    })
    .await
    .map_err(|join_error| {
        map_spawn_blocking_join_error(
            &join_error,
            ErrorCode::new("vector", "kernel_ready_state_task_failed"),
            "kernel ready-state warm from generation source",
            &collection_for_task,
        )
    })?
}

async fn warm_kernel_ready_state_from_generation_source_at_path(
    kernel: Arc<dyn VectorKernel + Send + Sync>,
    collection_name: &CollectionName,
    index: Arc<StdRwLock<VectorIndex>>,
    generation: &semantic_code_vector::PublishedGenerationPaths,
    snapshot_dir: Option<PathBuf>,
    allow_persist: bool,
    cancellation: Option<CancellationToken>,
) -> Result<()> {
    let source = Arc::new(
        load_generation_kernel_source(collection_name, generation, cancellation.clone()).await?,
    );
    warm_loaded_generation_source_at_path(
        kernel,
        collection_name,
        index,
        source,
        snapshot_dir,
        allow_persist,
        cancellation,
    )
    .await
}

fn count_unique_dfrr_prewarm_states(
    runtime_requirement: Option<&DfrrReadyStateRequirement>,
    requests: &[DfrrReadyStatePrewarmRequest],
) -> u64 {
    let mut fingerprints = BTreeSet::<&str>::new();
    if let Some(requirement) = runtime_requirement {
        fingerprints.insert(requirement.ready_state_fingerprint.as_ref());
    }
    for request in requests {
        fingerprints.insert(request.requirement.ready_state_fingerprint.as_ref());
    }
    u64::try_from(fingerprints.len()).map_or(u64::MAX, |value| value)
}

async fn warm_dfrr_ready_state_with_telemetry(
    kernel: Arc<dyn VectorKernel + Send + Sync>,
    collection_name: &CollectionName,
    generation_id: &semantic_code_vector::GenerationId,
    ready_state_fingerprint: &str,
    current: u64,
    total: u64,
    index: Arc<StdRwLock<VectorIndex>>,
    generation_root: Option<PathBuf>,
) -> Result<()> {
    let start = Instant::now();
    tracing::info!(
        collection = %collection_name,
        generation = generation_id.as_str(),
        kernel = ?kernel.kind(),
        ready_state_fingerprint,
        current,
        total,
        "dfrr ready-state prewarm started"
    );

    let mut warm_fut = Box::pin(warm_kernel_ready_state_at_path(
        kernel,
        collection_name,
        index,
        generation_root,
        true,
        None,
    ));
    let mut ticker = tokio::time::interval(Duration::from_secs(15));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let _ = ticker.tick().await;

    loop {
        tokio::select! {
            result = &mut warm_fut => {
                let elapsed_ms =
                    u64::try_from(start.elapsed().as_millis()).map_or(u64::MAX, |value| value);
                match &result {
                    Ok(()) => {
                        tracing::info!(
                            collection = %collection_name,
                            generation = generation_id.as_str(),
                            ready_state_fingerprint,
                            current,
                            total,
                            elapsed_ms,
                            "dfrr ready-state prewarm completed"
                        );
                    },
                    Err(error) => {
                        tracing::warn!(
                            collection = %collection_name,
                            generation = generation_id.as_str(),
                            ready_state_fingerprint,
                            current,
                            total,
                            elapsed_ms,
                            %error,
                            "dfrr ready-state prewarm failed"
                        );
                    },
                }
                return result;
            }
            _ = ticker.tick() => {
                let elapsed_ms =
                    u64::try_from(start.elapsed().as_millis()).map_or(u64::MAX, |value| value);
                tracing::info!(
                    collection = %collection_name,
                    generation = generation_id.as_str(),
                    ready_state_fingerprint,
                    current,
                    total,
                    elapsed_ms,
                    "dfrr ready-state prewarm still running"
                );
            }
        }
    }
}

/// Shared context for DFRR ready-state prewarm operations.
struct DfrrPrewarmContext<'a> {
    kernel: Arc<dyn VectorKernel + Send + Sync>,
    collection_name: &'a CollectionName,
    generation: &'a semantic_code_vector::PublishedGenerationPaths,
    generation_source: Option<Arc<PublishedGenerationKernelSource>>,
    generation_id: &'a semantic_code_vector::GenerationId,
    ready_state_fingerprint: &'a str,
    current: u64,
    total: u64,
    index: Arc<StdRwLock<VectorIndex>>,
    generation_root: Option<PathBuf>,
    cancellation: Option<CancellationToken>,
}

async fn warm_dfrr_ready_state_from_generation_source_with_telemetry(
    ctx: DfrrPrewarmContext<'_>,
) -> Result<()> {
    let start = Instant::now();
    tracing::info!(
        collection = %ctx.collection_name,
        generation = ctx.generation_id.as_str(),
        kernel = ?ctx.kernel.kind(),
        ready_state_fingerprint = ctx.ready_state_fingerprint,
        current = ctx.current,
        total = ctx.total,
        "dfrr ready-state prewarm started"
    );

    let collection_name = ctx.collection_name;
    let generation_id = ctx.generation_id;
    let ready_state_fingerprint = ctx.ready_state_fingerprint;
    let current = ctx.current;
    let total = ctx.total;

    let mut warm_fut: std::pin::Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> =
        if let Some(source) = ctx.generation_source {
            Box::pin(warm_loaded_generation_source_at_path(
                ctx.kernel,
                collection_name,
                ctx.index,
                source,
                ctx.generation_root,
                true,
                ctx.cancellation,
            ))
        } else {
            Box::pin(warm_kernel_ready_state_from_generation_source_at_path(
                ctx.kernel,
                collection_name,
                ctx.index,
                ctx.generation,
                ctx.generation_root,
                true,
                ctx.cancellation,
            ))
        };
    let mut ticker = tokio::time::interval(Duration::from_secs(15));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let _ = ticker.tick().await;

    loop {
        tokio::select! {
            result = &mut warm_fut => {
                let elapsed_ms =
                    u64::try_from(start.elapsed().as_millis()).map_or(u64::MAX, |value| value);
                match &result {
                    Ok(()) => tracing::info!(
                        collection = %collection_name,
                        generation = generation_id.as_str(),
                        ready_state_fingerprint,
                        current,
                        total,
                        elapsed_ms,
                        "dfrr ready-state prewarm completed"
                    ),
                    Err(error) => tracing::warn!(
                        collection = %collection_name,
                        generation = generation_id.as_str(),
                        ready_state_fingerprint,
                        current,
                        total,
                        elapsed_ms,
                        %error,
                        "dfrr ready-state prewarm failed"
                    ),
                }
                return result;
            }
            _ = ticker.tick() => {
                let elapsed_ms =
                    u64::try_from(start.elapsed().as_millis()).map_or(u64::MAX, |value| value);
                tracing::info!(
                    collection = %collection_name,
                    generation = generation_id.as_str(),
                    ready_state_fingerprint,
                    current,
                    total,
                    elapsed_ms,
                    "dfrr ready-state prewarm still running"
                );
            }
        }
    }
}

/// Run a single prewarm + record cycle for one DFRR ready-state requirement.
async fn prewarm_and_record(
    ctx: DfrrPrewarmContext<'_>,
    prefer_source: bool,
    generation_layout: &CollectionGenerationPaths,
    requirement: &DfrrReadyStateRequirement,
) -> Result<()> {
    let collection_name = ctx.collection_name;
    let generation_id = ctx.generation_id;
    let generation = ctx.generation;
    prewarm_generation_ready_state_request(ctx, prefer_source).await?;
    record_dfrr_ready_state_for_generation(
        collection_name,
        generation_layout,
        generation_id,
        generation,
        requirement,
    )
    .await
}

async fn prewarm_generation_ready_state_request(
    ctx: DfrrPrewarmContext<'_>,
    prefer_generation_source: bool,
) -> Result<()> {
    if prefer_generation_source {
        warm_dfrr_ready_state_from_generation_source_with_telemetry(ctx).await
    } else {
        warm_dfrr_ready_state_with_telemetry(
            ctx.kernel,
            ctx.collection_name,
            ctx.generation_id,
            ctx.ready_state_fingerprint,
            ctx.current,
            ctx.total,
            ctx.index,
            ctx.generation_root,
        )
        .await
    }
}

async fn warm_generation_runtime_index_if_needed(
    loader: &CollectionLoaderContext,
    collection_name: &CollectionName,
    generation: &semantic_code_vector::PublishedGenerationPaths,
    generation_source: Option<Arc<PublishedGenerationKernelSource>>,
    index: VectorIndex,
    cancellation: Option<CancellationToken>,
) -> Result<(VectorIndex, RuntimeKernelWarmState)> {
    if !(loader.runtime_kernel_supports_ready_state()
        && loader.runtime_kernel_prefers_generation_source())
    {
        return Ok((index, RuntimeKernelWarmState::NotWarmed));
    }

    let runtime_index_handle = Arc::new(StdRwLock::new(index));
    if let Some(source) = generation_source {
        warm_loaded_generation_source_at_path(
            Arc::clone(&loader.kernel),
            collection_name,
            Arc::clone(&runtime_index_handle),
            source,
            Some(generation.kernels_dir().to_path_buf()),
            true,
            cancellation,
        )
        .await?;
    } else {
        warm_kernel_ready_state_from_generation_source_at_path(
            Arc::clone(&loader.kernel),
            collection_name,
            Arc::clone(&runtime_index_handle),
            generation,
            Some(generation.kernels_dir().to_path_buf()),
            true,
            cancellation,
        )
        .await?;
    }
    let runtime_index_handle = Arc::try_unwrap(runtime_index_handle).map_err(|_| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "runtime_index_unwrap_failed"),
            "runtime index handle still shared after source-backed kernel warm",
            ErrorClass::NonRetriable,
        )
    })?;
    runtime_index_handle
        .into_inner()
        .map(|index| (index, RuntimeKernelWarmState::WarmedFromGenerationSource))
        .map_err(|_| index_lock_error("into_inner"))
}

async fn record_dfrr_ready_state_for_generation(
    collection_name: &CollectionName,
    generation_layout: &CollectionGenerationPaths,
    generation_id: &semantic_code_vector::GenerationId,
    generation: &semantic_code_vector::PublishedGenerationPaths,
    requirement: &DfrrReadyStateRequirement,
) -> Result<()> {
    let collection_name_for_task = collection_name.clone();
    let generation_layout_for_task = generation_layout.clone();
    let generation_id_for_task = generation_id.clone();
    let artifact_root = generation.kernels_dir().to_path_buf();
    let ready_state_fingerprint = requirement.ready_state_fingerprint.clone();
    let config_json = requirement.config_json.clone();
    spawn_blocking(move || {
        upsert_dfrr_ready_state(
            &collection_name_for_task,
            &generation_layout_for_task,
            &generation_id_for_task,
            ready_state_fingerprint.as_ref(),
            "ready",
            artifact_root.as_path(),
            config_json.as_ref(),
        )
    })
    .await
    .map_err(|join_error| {
        map_spawn_blocking_join_error(
            &join_error,
            ErrorCode::new("vector", "generation_catalog_task_failed"),
            "record DFRR ready state",
            collection_name,
        )
    })?
}

async fn generation_has_dfrr_ready_state(
    collection_name: &CollectionName,
    generation_layout: &CollectionGenerationPaths,
    generation_id: &semantic_code_vector::GenerationId,
    ready_state_fingerprint: &str,
) -> Result<bool> {
    let collection_name_for_task = collection_name.clone();
    let generation_layout_for_task = generation_layout.clone();
    let generation_id_for_task = generation_id.clone();
    let ready_state_fingerprint = ready_state_fingerprint.to_owned();
    spawn_blocking(move || {
        has_ready_dfrr_state(
            &collection_name_for_task,
            &generation_layout_for_task,
            &generation_id_for_task,
            ready_state_fingerprint.as_str(),
        )
    })
    .await
    .map_err(|join_error| {
        map_spawn_blocking_join_error(
            &join_error,
            ErrorCode::new("vector", "generation_catalog_task_failed"),
            "query DFRR ready state",
            collection_name,
        )
    })?
}

fn resolve_kernel_snapshot_dir(
    loader: &CollectionLoaderContext,
    paths: Option<&CollectionSnapshotPaths>,
) -> Result<Option<PathBuf>> {
    if loader.snapshot_format != VectorSnapshotFormat::V2 {
        return Ok(None);
    }
    let Some(paths) = paths else {
        return Ok(None);
    };

    let Some(active_generation) = paths.generation_layout.read_active_generation_id()? else {
        return Ok(Some(paths.v2_dir.clone()));
    };
    let generation = paths.generation_layout.generation(&active_generation);
    let dir = match loader.kernel.kind() {
        VectorKernelKind::HnswRs => generation.kernel_dir(VectorKernelKind::HnswRs),
        VectorKernelKind::Dfrr | VectorKernelKind::FlatScan => {
            generation.kernels_dir().to_path_buf()
        },
    };
    Ok(Some(dir))
}

async fn ensure_runtime_dfrr_ready_state_available(
    loader: &CollectionLoaderContext,
    collection_name: &CollectionName,
    paths: Option<&CollectionSnapshotPaths>,
) -> Result<()> {
    let Some(requirement) = loader.runtime_dfrr_ready_state.as_ref() else {
        return Ok(());
    };
    let Some(paths) = paths else {
        return Err(ErrorEnvelope::expected(
            ErrorCode::new("vector", "dfrr_ready_state_missing"),
            "DFRR ready state is missing; the collection has no published generation to restore",
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata(
            "readyStateFingerprint",
            requirement.ready_state_fingerprint.to_string(),
        ));
    };
    let Some(active_generation) = paths.generation_layout.read_active_generation_id()? else {
        return Err(ErrorEnvelope::expected(
            ErrorCode::new("vector", "dfrr_ready_state_missing"),
            "DFRR ready state is missing; no active generation is published",
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata(
            "readyStateFingerprint",
            requirement.ready_state_fingerprint.to_string(),
        ));
    };
    let ready = generation_has_dfrr_ready_state(
        collection_name,
        &paths.generation_layout,
        &active_generation,
        requirement.ready_state_fingerprint.as_ref(),
    )
    .await?;
    if ready {
        return Ok(());
    }

    Err(ErrorEnvelope::expected(
        ErrorCode::new("vector", "dfrr_ready_state_missing"),
        "DFRR ready state is missing; rerun index/build with the required prewarm fingerprint",
    )
    .with_metadata("collection", collection_name.as_str().to_string())
    .with_metadata("generationId", active_generation.as_str().to_string())
    .with_metadata(
        "readyStateFingerprint",
        requirement.ready_state_fingerprint.to_string(),
    ))
}

fn snapshot_kernel_mismatch_error(
    collection_name: &CollectionName,
    snapshot_dir: &Path,
    snapshot_kernel: VectorKernelKind,
    configured_kernel: VectorKernelKind,
) -> ErrorEnvelope {
    ErrorEnvelope::expected(
        ErrorCode::new("vector", "snapshot_kernel_mismatch"),
        "snapshot kernel metadata does not match configured vector kernel",
    )
    .with_metadata("collection", collection_name.as_str().to_string())
    .with_metadata("snapshotDir", snapshot_dir.display().to_string())
    .with_metadata(
        "snapshotKernel",
        kernel_kind_name(snapshot_kernel).to_string(),
    )
    .with_metadata(
        "configuredKernel",
        kernel_kind_name(configured_kernel).to_string(),
    )
}

fn index_lock_error(operation: &'static str) -> ErrorEnvelope {
    ErrorEnvelope::unexpected(
        ErrorCode::new("vector", "index_lock_poisoned"),
        "vector index lock is poisoned",
        ErrorClass::NonRetriable,
    )
    .with_metadata("operation", operation.to_string())
}

const fn kernel_kind_name(kernel: VectorKernelKind) -> &'static str {
    match kernel {
        VectorKernelKind::HnswRs => "hnsw-rs",
        VectorKernelKind::Dfrr => "dfrr",
        VectorKernelKind::FlatScan => "flat-scan",
    }
}

// ── Config → Vector bridge ──────────────────────────────────────────────────

/// Extension trait bridging the config layer (`HnswBuildConfig`) to the vector
/// layer (`HnswParams`).
///
/// Implemented here in the adapter crate because this is the boundary between
/// configuration and construction — the vector crate has no knowledge of config
/// types, and the config crate has no knowledge of vector types.
trait HnswParamsExt {
    /// Construct `HnswParams` from an optional build config, falling back to
    /// `HnswParams::default()` for unset fields.
    fn from_build_config(config: Option<&semantic_code_config::HnswBuildConfig>) -> Self;
}

impl HnswParamsExt for HnswParams {
    fn from_build_config(config: Option<&semantic_code_config::HnswBuildConfig>) -> Self {
        let mut params = Self::default();
        if let Some(build) = config {
            params.max_nb_connection = build.max_nb_connection as usize;
            params.ef_construction = build.ef_construction as usize;
        }
        params
    }
}

fn build_runtime_index_from_exact_rows(
    rows: &semantic_code_vector::ExactVectorRows,
    build_host_graph: bool,
    hnsw_params: &HnswParams,
    cancellation: Option<&CancellationToken>,
) -> Result<VectorIndex> {
    let mut index = VectorIndex::new(rows.dimension(), *hnsw_params)?;
    let records = rows
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
        .collect::<Vec<(semantic_code_vector::OriginId, VectorRecord)>>();

    if build_host_graph {
        index.insert_with_assigned_origins(records, cancellation)?;
    } else {
        index.insert_records_without_graph_with_assigned_origins(records)?;
    }

    Ok(index)
}

async fn build_runtime_index_from_exact_rows_async(
    collection_name: &CollectionName,
    rows: semantic_code_vector::ExactVectorRows,
    build_host_graph: bool,
    hnsw_params: HnswParams,
    cancellation: Option<CancellationToken>,
) -> Result<VectorIndex> {
    let collection_name_for_task = collection_name.clone();
    spawn_blocking(move || {
        build_runtime_index_from_exact_rows(
            &rows,
            build_host_graph,
            &hnsw_params,
            cancellation.as_ref(),
        )
    })
    .await
    .map_err(|join_error| {
        map_spawn_blocking_join_error(
            &join_error,
            ErrorCode::new("vector", "build_runtime_index_task_failed"),
            "build runtime index from exact rows",
            &collection_name_for_task,
        )
    })?
}

async fn load_runtime_index_from_generation_async(
    collection_name: &CollectionName,
    generation: &semantic_code_vector::PublishedGenerationPaths,
    rows: semantic_code_vector::ExactVectorRows,
    build_host_graph: bool,
    hnsw_params: HnswParams,
    cancellation: Option<CancellationToken>,
) -> Result<VectorIndex> {
    if build_host_graph {
        let ready_dir = generation.kernel_dir(VectorKernelKind::HnswRs);
        let graph_file = ready_dir.join(format!(
            "{}.hnsw.graph",
            semantic_code_vector::SNAPSHOT_V2_HNSW_GRAPH_BASENAME
        ));
        if graph_file.exists() {
            let collection_name_for_task = collection_name.clone();
            let ready_dir_for_task = ready_dir.clone();
            let rows_for_task = rows.clone();
            let mut params = hnsw_params;
            params.max_elements = params.max_elements.max(rows_for_task.row_count().max(1));
            let records_with_origins = rows_for_task
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
                .collect::<Vec<(semantic_code_vector::OriginId, VectorRecord)>>();

            let ready_load = spawn_blocking(move || {
                VectorIndex::from_hnsw_ready_state(
                    &ready_dir_for_task,
                    rows_for_task.dimension(),
                    params,
                    records_with_origins,
                )
            })
            .await
            .map_err(|join_error| {
                map_spawn_blocking_join_error(
                    &join_error,
                    ErrorCode::new("vector", "hnsw_ready_state_load_task_failed"),
                    "load hnsw ready state",
                    &collection_name_for_task,
                )
            })?;

            match ready_load {
                Ok(index) => return Ok(index),
                Err(error) => {
                    tracing::warn!(
                        collection = %collection_name,
                        ready_state_dir = %ready_dir.display(),
                        %error,
                        "failed to load generation-scoped hnsw ready state; rebuilding from exact rows"
                    );
                    return build_runtime_index_from_exact_rows_async(
                        collection_name,
                        rows,
                        true,
                        hnsw_params,
                        cancellation,
                    )
                    .await;
                },
            }
        }
    }

    build_runtime_index_from_exact_rows_async(
        collection_name,
        rows,
        build_host_graph,
        hnsw_params,
        cancellation,
    )
    .await
}

#[cfg(test)]
fn _assert_vector_index_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<VectorIndex>();
}

struct LocalCollection {
    dimension: u32,
    index_mode: IndexMode,
    index: Arc<StdRwLock<VectorIndex>>,
    documents: BTreeMap<Box<str>, StoredDocument>,
    last_insert_sequence: u64,
    mode: LocalCollectionMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalCollectionMode {
    Online,
    Staging,
}

impl LocalCollection {
    fn new(dimension: u32, index_mode: IndexMode, hnsw_params: HnswParams) -> Result<Self> {
        Self::new_with_mode(
            dimension,
            index_mode,
            LocalCollectionMode::Online,
            hnsw_params,
        )
    }

    fn new_staging(dimension: u32, index_mode: IndexMode, hnsw_params: HnswParams) -> Result<Self> {
        Self::new_with_mode(
            dimension,
            index_mode,
            LocalCollectionMode::Staging,
            hnsw_params,
        )
    }

    fn new_with_mode(
        dimension: u32,
        index_mode: IndexMode,
        mode: LocalCollectionMode,
        hnsw_params: HnswParams,
    ) -> Result<Self> {
        let index = VectorIndex::new(dimension, hnsw_params)?;
        Ok(Self {
            dimension,
            index_mode,
            index: Arc::new(StdRwLock::new(index)),
            documents: BTreeMap::new(),
            last_insert_sequence: 0,
            mode,
        })
    }

    /// Current number of active vectors in this collection.
    fn vector_count(&self) -> u64 {
        self.index
            .read()
            .map_or(0, |guard| guard.active_count() as u64)
    }

    fn read_index(&self) -> Result<std::sync::RwLockReadGuard<'_, VectorIndex>, ErrorEnvelope> {
        self.index.read().map_err(|_| index_lock_error("read"))
    }

    fn write_index(&self) -> Result<std::sync::RwLockWriteGuard<'_, VectorIndex>, ErrorEnvelope> {
        self.index.write().map_err(|_| index_lock_error("write"))
    }

    fn insert(&mut self, documents: Vec<VectorDocumentForInsert>) -> Result<InsertWalRecord> {
        let sequence = self.next_insert_sequence()?;
        let mut records = Vec::new();
        let mut docs = BTreeMap::new();
        let mut wal_documents = Vec::new();
        for doc in documents {
            let id = doc.id.clone();
            let vector = doc.vector.as_ref().to_vec();
            records.push(VectorRecord {
                id: id.clone(),
                vector: vector.clone(),
            });
            wal_documents.push(InsertWalDocument {
                id: id.clone(),
                vector,
                content: doc.content.clone(),
                metadata: doc.metadata.clone(),
            });
            docs.insert(
                id,
                StoredDocument {
                    content: doc.content,
                    metadata: doc.metadata,
                },
            );
        }

        if self.mode == LocalCollectionMode::Staging {
            self.write_index()?.insert_records_without_graph(records)?;
        } else {
            self.write_index()?.insert(records)?;
        }
        for (id, doc) in docs {
            self.documents.insert(id, doc);
        }
        self.last_insert_sequence = sequence;

        Ok(InsertWalRecord {
            sequence,
            documents: wal_documents,
        })
    }

    const fn is_staging(&self) -> bool {
        matches!(self.mode, LocalCollectionMode::Staging)
    }

    const fn mark_online(&mut self) {
        self.mode = LocalCollectionMode::Online;
    }

    fn exact_rows(&self) -> Result<semantic_code_vector::ExactVectorRows> {
        self.read_index()?.exact_rows()
    }

    fn replace_index(&self, index: VectorIndex) -> Result<()> {
        *self.write_index()? = index;
        Ok(())
    }

    fn replay_insert_record(&mut self, record: &InsertWalRecord) -> Result<()> {
        let mut index_records = Vec::new();
        let mut documents = BTreeMap::new();
        for document in &record.documents {
            index_records.push(VectorRecord {
                id: document.id.clone(),
                vector: document.vector.clone(),
            });
            documents.insert(
                document.id.clone(),
                StoredDocument {
                    content: document.content.clone(),
                    metadata: document.metadata.clone(),
                },
            );
        }

        self.write_index()?.insert(index_records)?;
        for (id, document) in documents {
            self.documents.insert(id, document);
        }
        self.last_insert_sequence = record.sequence;
        Ok(())
    }

    fn delete(&mut self, ids: &[Box<str>]) -> Result<()> {
        self.write_index()?.delete(ids)?;
        for id in ids {
            self.documents.remove(id.as_ref());
        }
        Ok(())
    }

    fn snapshot(&self) -> Result<CollectionSnapshot> {
        let records = {
            let index = self.read_index()?;
            let mut records = Vec::with_capacity(index.active_count());
            for (_origin, record) in index.active_entries_by_origin() {
                let Some(doc) = self.documents.get(record.id.as_ref()) else {
                    return Err(ErrorEnvelope::unexpected(
                        ErrorCode::new("vector", "snapshot_document_missing"),
                        "active vector record missing from collection documents during snapshot",
                        ErrorClass::NonRetriable,
                    )
                    .with_metadata("id", record.id.to_string()));
                };
                records.push(CollectionRecord {
                    id: record.id.clone(),
                    vector: record.vector.clone(),
                    content: doc.content.clone(),
                    metadata: doc.metadata.clone(),
                });
            }
            drop(index);
            records
        };

        Ok(CollectionSnapshot {
            version: LOCAL_SNAPSHOT_VERSION,
            dimension: self.dimension,
            index_mode: self.index_mode,
            records,
            checkpoint_sequence: (self.last_insert_sequence > 0)
                .then_some(self.last_insert_sequence),
        })
    }

    fn from_snapshot(snapshot: CollectionSnapshot, hnsw_params: HnswParams) -> Result<Self> {
        let CollectionSnapshot {
            version,
            dimension,
            index_mode,
            records,
            checkpoint_sequence,
        } = snapshot;
        validate_local_snapshot_version(version)?;
        let mut index = VectorIndex::new(dimension, hnsw_params)?;
        let mut documents = BTreeMap::new();
        let mut index_records = Vec::new();
        for record in records {
            index_records.push(VectorRecord {
                id: record.id.clone(),
                vector: record.vector.clone(),
            });
            documents.insert(
                record.id.clone(),
                StoredDocument {
                    content: record.content,
                    metadata: record.metadata,
                },
            );
        }
        index.insert(index_records)?;
        Ok(Self {
            dimension,
            index_mode,
            index: Arc::new(StdRwLock::new(index)),
            documents,
            last_insert_sequence: checkpoint_sequence.unwrap_or(0),
            mode: LocalCollectionMode::Online,
        })
    }

    fn from_snapshot_records_only(
        snapshot: CollectionSnapshot,
        hnsw_params: HnswParams,
    ) -> Result<Self> {
        let CollectionSnapshot {
            version,
            dimension,
            index_mode,
            records,
            checkpoint_sequence,
        } = snapshot;
        validate_local_snapshot_version(version)?;
        let mut index = VectorIndex::new(dimension, hnsw_params)?;
        let mut documents = BTreeMap::new();
        let mut index_records = Vec::new();
        for record in records {
            index_records.push(VectorRecord {
                id: record.id.clone(),
                vector: record.vector.clone(),
            });
            documents.insert(
                record.id.clone(),
                StoredDocument {
                    content: record.content,
                    metadata: record.metadata,
                },
            );
        }
        index.insert_records_without_graph(index_records)?;
        Ok(Self {
            dimension,
            index_mode,
            index: Arc::new(StdRwLock::new(index)),
            documents,
            last_insert_sequence: checkpoint_sequence.unwrap_or(0),
            mode: LocalCollectionMode::Online,
        })
    }

    fn from_snapshot_with_index(snapshot: CollectionSnapshot, index: VectorIndex) -> Result<Self> {
        let CollectionSnapshot {
            version,
            dimension,
            index_mode,
            records,
            checkpoint_sequence,
        } = snapshot;
        validate_local_snapshot_version(version)?;
        if index.dimension() != dimension {
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("vector", "snapshot_dimension_mismatch"),
                "snapshot dimension mismatch",
            )
            .with_metadata("snapshotDimension", dimension.to_string())
            .with_metadata("indexDimension", index.dimension().to_string()));
        }

        let mut documents = BTreeMap::new();
        for record in records {
            if index.record_for_id(record.id.as_ref()).is_none() {
                return Err(ErrorEnvelope::expected(
                    ErrorCode::new("vector", "snapshot_record_missing"),
                    "snapshot record id not found in v2 companion index",
                )
                .with_metadata("id", record.id.to_string()));
            }
            documents.insert(
                record.id.clone(),
                StoredDocument {
                    content: record.content,
                    metadata: record.metadata,
                },
            );
        }

        Ok(Self {
            dimension,
            index_mode,
            index: Arc::new(StdRwLock::new(index)),
            documents,
            last_insert_sequence: checkpoint_sequence.unwrap_or(0),
            mode: LocalCollectionMode::Online,
        })
    }

    /// Construct from a pre-built v2 index and a pre-parsed sidecar.
    ///
    /// Use this when the index and sidecar were loaded in parallel (via
    /// `try_join!`) and only the final assembly needs both.
    fn from_index_and_parsed_sidecar(
        index: VectorIndex,
        sidecar: ParsedRecordsSidecar,
    ) -> Result<Self> {
        if index.dimension() != sidecar.dimension {
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("vector", "snapshot_dimension_mismatch"),
                "sidecar dimension mismatch",
            )
            .with_metadata("sidecarDimension", sidecar.dimension.to_string())
            .with_metadata("indexDimension", index.dimension().to_string()));
        }
        Ok(Self {
            dimension: sidecar.dimension,
            index_mode: sidecar.index_mode,
            index: Arc::new(StdRwLock::new(index)),
            documents: sidecar.documents,
            last_insert_sequence: sidecar.checkpoint_sequence.unwrap_or(0),
            mode: LocalCollectionMode::Online,
        })
    }

    fn next_insert_sequence(&self) -> Result<u64> {
        self.last_insert_sequence.checked_add(1).ok_or_else(|| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "insert_sequence_overflow"),
                "insert sequence overflow",
                ErrorClass::NonRetriable,
            )
        })
    }
}

/// A single record in the v2 `records.meta.jsonl` sidecar.
///
/// Position-aligned with `ids.json` and `vectors.u8.bin`: line N corresponds
/// to the Nth record in both files.  Contains everything the v1 JSON carries
/// per record **except** the float32 vector (which lives in the v2 binary).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RecordMetadataEntry {
    id: Box<str>,
    content: Box<str>,
    metadata: VectorDocumentMetadata,
}

/// Header written as the first line of `records.meta.jsonl` to allow
/// forward-compatible parsing and validation before streaming records.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RecordMetadataHeader {
    version: u32,
    dimension: u32,
    index_mode: IndexMode,
    count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    checkpoint_sequence: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BuildJournalHeader {
    version: u32,
    dimension: u32,
    index_mode: IndexMode,
    count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    checkpoint_sequence: Option<u64>,
}

fn validate_local_snapshot_version(version: u32) -> Result<()> {
    if version != LOCAL_SNAPSHOT_VERSION {
        return Err(ErrorEnvelope::expected(
            ErrorCode::new("vector", "snapshot_version_mismatch"),
            "snapshot version mismatch",
        )
        .with_metadata("found", version.to_string())
        .with_metadata("expected", LOCAL_SNAPSHOT_VERSION.to_string()));
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct StoredDocument {
    content: Box<str>,
    metadata: VectorDocumentMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CollectionSnapshot {
    version: u32,
    dimension: u32,
    index_mode: IndexMode,
    records: Vec<CollectionRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    checkpoint_sequence: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CollectionRecord {
    id: Box<str>,
    vector: Vec<f32>,
    content: Box<str>,
    metadata: VectorDocumentMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InsertWalRecord {
    sequence: u64,
    documents: Vec<InsertWalDocument>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InsertWalDocument {
    id: Box<str>,
    vector: Vec<f32>,
    content: Box<str>,
    metadata: VectorDocumentMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilterOp {
    Eq,
    NotEq,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilterField {
    RelativePath,
    Language,
    FileExtension,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FilterCondition {
    field: FilterField,
    op: FilterOp,
    value: Box<str>,
}

fn parse_filter_expr(expr: Option<&str>) -> Result<Option<FilterCondition>> {
    let Some(expr) = expr else {
        return Ok(None);
    };
    let expr = expr.trim();
    if expr.is_empty() {
        return Ok(None);
    }
    if expr.contains('\n') || expr.contains('\r') {
        return Err(invalid_filter_expr(expr));
    }

    let (field, op, value) =
        parse_simple_comparison(expr).ok_or_else(|| invalid_filter_expr(expr))?;
    let field = match field {
        "relativePath" => FilterField::RelativePath,
        "language" => FilterField::Language,
        "fileExtension" => FilterField::FileExtension,
        _ => return Err(invalid_filter_expr(expr)),
    };
    let op = match op {
        "==" => FilterOp::Eq,
        "!=" => FilterOp::NotEq,
        _ => return Err(invalid_filter_expr(expr)),
    };
    if value.is_empty() {
        return Err(invalid_filter_expr(expr));
    }

    Ok(Some(FilterCondition {
        field,
        op,
        value: value.to_owned().into_boxed_str(),
    }))
}

fn parse_simple_comparison(input: &str) -> Option<(&str, &str, &str)> {
    let input = input.trim();
    let (field, rest) = split_once_ws(input)?;
    let rest = rest.trim_start();

    let (op, rest) = if let Some(rest) = rest.strip_prefix("==") {
        ("==", rest)
    } else if let Some(rest) = rest.strip_prefix("!=") {
        ("!=", rest)
    } else {
        return None;
    };

    let value = rest.trim_start();
    let unquoted = strip_quotes(value)?;
    Some((field, op, unquoted))
}

fn split_once_ws(input: &str) -> Option<(&str, &str)> {
    for (idx, ch) in input.char_indices() {
        if ch.is_whitespace() {
            let (left, right) = input.split_at(idx);
            return Some((left, right));
        }
    }
    None
}

fn strip_quotes(input: &str) -> Option<&str> {
    let input = input.trim();
    if input.len() < 2 {
        return None;
    }
    let bytes = input.as_bytes();
    let first = *bytes.first()?;
    let last = *bytes.last()?;
    if (first == b'\'' && last == b'\'') || (first == b'"' && last == b'"') {
        Some(&input[1..input.len() - 1])
    } else {
        None
    }
}

fn filter_matches(filter: Option<&FilterCondition>, doc: &StoredDocument) -> bool {
    let Some(filter) = filter else {
        return true;
    };

    let value = match filter.field {
        FilterField::RelativePath => Some(doc.metadata.relative_path.as_ref()),
        FilterField::Language => doc.metadata.language.map(Language::as_str),
        FilterField::FileExtension => doc.metadata.file_extension.as_deref(),
    };

    match filter.op {
        FilterOp::Eq => value.is_some_and(|v| v == filter.value.as_ref()),
        FilterOp::NotEq => value.is_none_or(|v| v != filter.value.as_ref()),
    }
}

fn build_row(id: &str, doc: &StoredDocument, output_fields: &[Box<str>]) -> VectorDbRow {
    let mut row = BTreeMap::new();
    for field in output_fields {
        match field.as_ref() {
            "id" => {
                row.insert(field.clone(), Value::String(id.to_owned()));
            },
            "relativePath" => {
                row.insert(
                    field.clone(),
                    Value::String(doc.metadata.relative_path.as_ref().to_owned()),
                );
            },
            "language" => {
                if let Some(language) = doc.metadata.language {
                    row.insert(field.clone(), Value::String(language.as_str().to_owned()));
                }
            },
            "fileExtension" => {
                if let Some(ext) = doc.metadata.file_extension.as_ref() {
                    row.insert(field.clone(), Value::String(ext.as_ref().to_owned()));
                }
            },
            "startLine" => {
                row.insert(field.clone(), Value::from(doc.metadata.span.start_line()));
            },
            "endLine" => {
                row.insert(field.clone(), Value::from(doc.metadata.span.end_line()));
            },
            "content" => {
                row.insert(
                    field.clone(),
                    Value::String(doc.content.as_ref().to_owned()),
                );
            },
            _ => {},
        }
    }
    row
}

fn invalid_filter_expr(expr: &str) -> ErrorEnvelope {
    ErrorEnvelope::expected(
        ErrorCode::new("vector", "invalid_filter_expr"),
        format!("filterExpr is not supported: {expr}"),
    )
}

fn snapshot_error(
    code: &'static str,
    message: &str,
    error: impl std::error::Error,
) -> ErrorEnvelope {
    ErrorEnvelope::unexpected(
        ErrorCode::new("vector", code),
        format!("{message}: {error}"),
        ErrorClass::NonRetriable,
    )
}

fn duplicate_record_id_in_sidecar_error(
    path: &Path,
    id: &str,
    first_line: usize,
    duplicate_line: usize,
) -> ErrorEnvelope {
    ErrorEnvelope::expected(
        ErrorCode::new("vector", "duplicate_record_id_in_sidecar"),
        "metadata sidecar contains a duplicate record id",
    )
    .with_metadata("id", id.to_owned())
    .with_metadata("path", path.display().to_string())
    .with_metadata("first_line", first_line.to_string())
    .with_metadata("duplicate_line", duplicate_line.to_string())
}

fn serialize_snapshot_json(snapshot: &CollectionSnapshot) -> Result<Vec<u8>> {
    serde_json::to_vec_pretty(snapshot).map_err(|error| {
        snapshot_error(
            "snapshot_serialize_failed",
            "failed to serialize snapshot",
            error,
        )
    })
}

/// Write the `records.meta.jsonl` sidecar alongside the v2 binary bundle.
///
/// Format: header line + one JSONL line per record, position-aligned with
/// `ids.json` and `vectors.u8.bin`. Records are emitted in the canonical
/// snapshot order already stored in `CollectionSnapshot::records`
/// (active records in ascending `origin_id`).
fn write_records_meta_sidecar(
    paths: &CollectionSnapshotPaths,
    snapshot: &CollectionSnapshot,
) -> Result<()> {
    let count = u64::try_from(snapshot.records.len()).map_err(|_| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "snapshot_count_overflow"),
            "record count conversion overflow in metadata sidecar",
            ErrorClass::NonRetriable,
        )
    })?;

    let header = RecordMetadataHeader {
        version: snapshot.version,
        dimension: snapshot.dimension,
        index_mode: snapshot.index_mode,
        count,
        checkpoint_sequence: snapshot.checkpoint_sequence,
    };

    let entries = snapshot.records.iter().map(|record| RecordMetadataEntry {
        id: record.id.clone(),
        content: record.content.clone(),
        metadata: record.metadata.clone(),
    });
    let buf = serialize_records_meta_sidecar(&header, entries)?;

    let path = &paths.v2_records_meta;
    write_records_meta_payload(path, buf.as_slice())?;

    tracing::debug!(
        path = %path.display(),
        records = count,
        bytes = buf.len(),
        "wrote records.meta.jsonl sidecar"
    );
    Ok(())
}

/// Write the `records.meta.jsonl` sidecar directly from the in-memory
/// collection, iterating in the order dictated by `ordered_ids`.
///
/// The caller extracts `ordered_ids` from the same [`PreparedV2Snapshot`]
/// that produced `ids.json` and `vectors.u8.bin`, so the sidecar rows are
/// guaranteed to be row-aligned with the binary snapshot files by
/// construction — not by convention.
///
/// IDs that appear in `ordered_ids` but are missing from
/// `collection.documents` are skipped with a warning.  This should never
/// happen in practice but keeps the writer defensive against transient
/// race conditions during concurrent upserts.
fn write_sidecar_from_collection(
    paths: &CollectionSnapshotPaths,
    collection: &LocalCollection,
    ordered_ids: &[Box<str>],
) -> Result<()> {
    let count = u64::try_from(ordered_ids.len()).map_err(|_| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "snapshot_count_overflow"),
            "record count conversion overflow in metadata sidecar",
            ErrorClass::NonRetriable,
        )
    })?;

    let checkpoint_sequence =
        (collection.last_insert_sequence > 0).then_some(collection.last_insert_sequence);

    let header = RecordMetadataHeader {
        version: LOCAL_SNAPSHOT_VERSION,
        dimension: collection.dimension,
        index_mode: collection.index_mode,
        count,
        checkpoint_sequence,
    };

    let entries = ordered_ids.iter().filter_map(|id| {
        let Some(doc) = collection.documents.get(id) else {
            tracing::warn!(
                id = %id,
                "sidecar: ordered ID missing from collection documents, skipping"
            );
            return None;
        };
        Some(RecordMetadataEntry {
            id: id.clone(),
            content: doc.content.clone(),
            metadata: doc.metadata.clone(),
        })
    });
    let buf = serialize_records_meta_sidecar(&header, entries)?;

    let path = &paths.v2_records_meta;
    write_records_meta_payload(path, buf.as_slice())?;

    tracing::debug!(
        path = %path.display(),
        records = count,
        bytes = buf.len(),
        "wrote records.meta.jsonl sidecar (from collection)"
    );
    Ok(())
}

fn write_published_records_meta_sidecar(
    base_dir: &Path,
    snapshot: &CollectionSnapshot,
) -> Result<()> {
    let count = u64::try_from(snapshot.records.len()).map_err(|_| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "snapshot_count_overflow"),
            "record count conversion overflow in metadata sidecar",
            ErrorClass::NonRetriable,
        )
    })?;
    let header = RecordMetadataHeader {
        version: snapshot.version,
        dimension: snapshot.dimension,
        index_mode: snapshot.index_mode,
        count,
        checkpoint_sequence: snapshot.checkpoint_sequence,
    };
    let entries = snapshot.records.iter().map(|record| RecordMetadataEntry {
        id: record.id.clone(),
        content: record.content.clone(),
        metadata: record.metadata.clone(),
    });
    let payload = serialize_records_meta_sidecar(&header, entries)?;
    write_records_meta_payload(
        &base_dir.join(LOCAL_SNAPSHOT_V2_RECORDS_META_FILE_NAME),
        payload.as_slice(),
    )?;
    Ok(())
}

fn serialize_records_meta_sidecar(
    header: &RecordMetadataHeader,
    entries: impl IntoIterator<Item = RecordMetadataEntry>,
) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    serde_json::to_writer(&mut buf, header).map_err(|error| {
        snapshot_error(
            "sidecar_serialize_failed",
            "failed to serialize metadata sidecar header",
            error,
        )
    })?;
    buf.push(b'\n');

    for entry in entries {
        serde_json::to_writer(&mut buf, &entry).map_err(|error| {
            snapshot_error(
                "sidecar_serialize_failed",
                "failed to serialize metadata sidecar entry",
                error,
            )
        })?;
        buf.push(b'\n');
    }

    Ok(buf)
}

fn write_records_meta_payload(path: &Path, payload: &[u8]) -> Result<()> {
    use std::io::Write;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(ErrorEnvelope::from)?;
    }
    let mut file = std::fs::File::create(path).map_err(ErrorEnvelope::from)?;
    file.write_all(payload).map_err(ErrorEnvelope::from)?;
    file.flush().map_err(ErrorEnvelope::from)?;
    Ok(())
}

/// Parsed content of a `records.meta.jsonl` sidecar.
struct ParsedRecordsSidecar {
    dimension: u32,
    index_mode: IndexMode,
    checkpoint_sequence: Option<u64>,
    documents: BTreeMap<Box<str>, StoredDocument>,
}

/// Read the `records.meta.jsonl` sidecar and reconstruct the document map
/// for a `LocalCollection` without needing the v1 JSON.
fn read_records_meta_sidecar(path: &Path) -> Result<ParsedRecordsSidecar> {
    let payload = std::fs::read(path).map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "sidecar_read_failed"),
            format!("failed to read metadata sidecar: {}", path.display()),
            ErrorClass::NonRetriable,
        )
        .with_metadata("source", error.to_string())
    })?;

    let mut lines = payload.split(|&b| b == b'\n').filter(|l| !l.is_empty());

    let header_line = lines.next().ok_or_else(|| {
        ErrorEnvelope::expected(
            ErrorCode::new("vector", "sidecar_empty"),
            format!("metadata sidecar is empty: {}", path.display()),
        )
    })?;
    let header: RecordMetadataHeader = serde_json::from_slice(header_line).map_err(|error| {
        snapshot_error(
            "sidecar_parse_failed",
            "failed to parse metadata sidecar header",
            error,
        )
    })?;
    validate_local_snapshot_version(header.version)?;

    let mut documents = BTreeMap::new();
    let mut document_lines = BTreeMap::new();
    for (line_idx, line) in lines.enumerate() {
        let line_number = line_idx + 2;
        let entry: RecordMetadataEntry = serde_json::from_slice(line).map_err(|error| {
            snapshot_error(
                "sidecar_parse_failed",
                &format!("failed to parse metadata sidecar record at line {line_number}"),
                error,
            )
        })?;

        if let Some(first_line) = document_lines.get(entry.id.as_ref()) {
            return Err(duplicate_record_id_in_sidecar_error(
                path,
                entry.id.as_ref(),
                *first_line,
                line_number,
            ));
        }

        document_lines.insert(entry.id.clone(), line_number);
        documents.insert(
            entry.id,
            StoredDocument {
                content: entry.content,
                metadata: entry.metadata,
            },
        );
    }

    let expected = usize::try_from(header.count).map_err(|_| {
        ErrorEnvelope::expected(
            ErrorCode::new("vector", "sidecar_count_overflow"),
            "metadata sidecar count overflows usize",
        )
        .with_metadata("count", header.count.to_string())
        .with_metadata("path", path.display().to_string())
    })?;
    if documents.len() != expected {
        return Err(ErrorEnvelope::expected(
            ErrorCode::new("vector", "sidecar_record_count_mismatch"),
            "metadata sidecar record count does not match header",
        )
        .with_metadata("expected", expected.to_string())
        .with_metadata("found", documents.len().to_string())
        .with_metadata("path", path.display().to_string()));
    }

    Ok(ParsedRecordsSidecar {
        dimension: header.dimension,
        index_mode: header.index_mode,
        checkpoint_sequence: header.checkpoint_sequence,
        documents,
    })
}

fn serialize_insert_wal_record(record: &InsertWalRecord) -> Result<Vec<u8>> {
    serde_json::to_vec(record).map_err(|error| {
        snapshot_error(
            "wal_serialize_failed",
            "failed to serialize insert WAL record",
            error,
        )
    })
}

async fn append_build_journal_record(
    paths: &CollectionSnapshotPaths,
    record: &InsertWalRecord,
) -> Result<()> {
    if let Some(parent) = paths.build_rows.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(ErrorEnvelope::from)?;
    }

    let mut rows_file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.build_rows)
        .await
        .map_err(ErrorEnvelope::from)?;
    let mut vectors_file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.build_vectors)
        .await
        .map_err(ErrorEnvelope::from)?;

    for document in &record.documents {
        let entry = RecordMetadataEntry {
            id: document.id.clone(),
            content: document.content.clone(),
            metadata: document.metadata.clone(),
        };
        let mut row = serde_json::to_vec(&entry).map_err(|error| {
            snapshot_error(
                "build_journal_serialize_failed",
                "failed to serialize build journal row",
                error,
            )
        })?;
        row.push(b'\n');
        rows_file
            .write_all(row.as_slice())
            .await
            .map_err(ErrorEnvelope::from)?;

        for value in &document.vector {
            vectors_file
                .write_all(&value.to_le_bytes())
                .await
                .map_err(ErrorEnvelope::from)?;
        }
    }

    rows_file.flush().await.map_err(ErrorEnvelope::from)?;
    vectors_file.flush().await.map_err(ErrorEnvelope::from)?;
    Ok(())
}

async fn seal_build_journal(
    paths: &CollectionSnapshotPaths,
    snapshot: &CollectionSnapshot,
) -> Result<()> {
    let header = BuildJournalHeader {
        version: snapshot.version,
        dimension: snapshot.dimension,
        index_mode: snapshot.index_mode,
        count: u64::try_from(snapshot.records.len()).map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "snapshot_count_overflow"),
                "record count conversion overflow in build journal",
                ErrorClass::NonRetriable,
            )
        })?,
        checkpoint_sequence: snapshot.checkpoint_sequence,
    };
    let payload = serde_json::to_vec_pretty(&header).map_err(|error| {
        snapshot_error(
            "build_journal_serialize_failed",
            "failed to serialize build journal header",
            error,
        )
    })?;
    tokio::fs::write(&paths.build_meta, payload)
        .await
        .map_err(ErrorEnvelope::from)?;
    tokio::fs::write(&paths.build_sealed, b"sealed")
        .await
        .map_err(ErrorEnvelope::from)?;
    Ok(())
}

async fn append_insert_wal_record(path: &Path, record: &InsertWalRecord) -> Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(ErrorEnvelope::from)?;
    }

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await
        .map_err(ErrorEnvelope::from)?;
    let mut payload = serialize_insert_wal_record(record)?;
    payload.push(b'\n');
    file.write_all(payload.as_slice())
        .await
        .map_err(ErrorEnvelope::from)?;
    file.flush().await.map_err(ErrorEnvelope::from)?;
    Ok(())
}

async fn read_insert_wal_records(path: &Path) -> Result<Vec<InsertWalRecord>> {
    let payload = match tokio::fs::read(path).await {
        Ok(payload) => payload,
        Err(error) => {
            if error.kind() == std::io::ErrorKind::NotFound {
                return Ok(Vec::new());
            }
            return Err(ErrorEnvelope::from(error));
        },
    };

    let records = parse_insert_wal_records(path, payload.as_slice())?;
    validate_wal_sequence_order(path, records.as_slice())?;
    Ok(records)
}

async fn compact_insert_wal_records(path: &Path, checkpoint_sequence: u64) -> Result<()> {
    let records = read_insert_wal_records(path).await?;
    if records.is_empty() {
        return Ok(());
    }

    let keep_from = records.partition_point(|record| record.sequence <= checkpoint_sequence);
    if keep_from == 0 {
        return Ok(());
    }
    if keep_from >= records.len() {
        match tokio::fs::remove_file(path).await {
            Ok(()) => return Ok(()),
            Err(error) => {
                if error.kind() == std::io::ErrorKind::NotFound {
                    return Ok(());
                }
                return Err(ErrorEnvelope::from(error));
            },
        }
    }

    let mut compacted = Vec::new();
    for record in records.iter().skip(keep_from) {
        let mut line = serialize_insert_wal_record(record)?;
        line.push(b'\n');
        compacted.extend_from_slice(line.as_slice());
    }

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .await
        .map_err(ErrorEnvelope::from)?;
    file.write_all(compacted.as_slice())
        .await
        .map_err(ErrorEnvelope::from)?;
    file.flush().await.map_err(ErrorEnvelope::from)?;
    Ok(())
}

fn replay_insert_wal_records(
    path: &Path,
    collection: &mut LocalCollection,
    checkpoint_sequence: u64,
    records: &[InsertWalRecord],
) -> Result<()> {
    let replay_start = records.partition_point(|record| record.sequence <= checkpoint_sequence);
    let mut expected_sequence = checkpoint_sequence.checked_add(1);

    for (idx, record) in records.iter().enumerate().skip(replay_start) {
        let Some(expected_sequence_value) = expected_sequence else {
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("vector", "wal_replay_sequence_overflow"),
                "insert WAL replay sequence overflow",
            )
            .with_metadata("path", path.display().to_string())
            .with_metadata("line", (idx + 1).to_string())
            .with_metadata("checkpointSequence", checkpoint_sequence.to_string()));
        };
        if record.sequence != expected_sequence_value {
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("vector", "wal_replay_sequence_gap"),
                "insert WAL replay sequence does not match checkpoint progression",
            )
            .with_metadata("path", path.display().to_string())
            .with_metadata("line", (idx + 1).to_string())
            .with_metadata("checkpointSequence", checkpoint_sequence.to_string())
            .with_metadata("expectedSequence", expected_sequence_value.to_string())
            .with_metadata("sequence", record.sequence.to_string()));
        }

        collection.replay_insert_record(record).map_err(|error| {
            ErrorEnvelope::expected(
                ErrorCode::new("vector", "wal_replay_apply_failed"),
                "failed to replay insert WAL record",
            )
            .with_metadata("path", path.display().to_string())
            .with_metadata("line", (idx + 1).to_string())
            .with_metadata("sequence", record.sequence.to_string())
            .with_metadata("sourceCode", error.code.to_string())
            .with_metadata("source", error.message)
        })?;
        expected_sequence = record.sequence.checked_add(1);
    }
    Ok(())
}

fn parse_insert_wal_records(path: &Path, payload: &[u8]) -> Result<Vec<InsertWalRecord>> {
    let mut records = Vec::new();
    for (line_idx, raw_line) in payload.split(|byte| *byte == b'\n').enumerate() {
        if raw_line.is_empty() {
            continue;
        }
        let line = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
        if line.is_empty() {
            continue;
        }
        let record: InsertWalRecord = serde_json::from_slice(line).map_err(|error| {
            ErrorEnvelope::expected(
                ErrorCode::new("vector", "wal_parse_failed"),
                "failed to parse insert WAL record",
            )
            .with_metadata("path", path.display().to_string())
            .with_metadata("line", (line_idx + 1).to_string())
            .with_metadata("source", error.to_string())
        })?;
        records.push(record);
    }
    Ok(records)
}

fn validate_wal_sequence_order(path: &Path, records: &[InsertWalRecord]) -> Result<()> {
    let mut previous = None;
    for (idx, record) in records.iter().enumerate() {
        if let Some(prev) = previous
            && record.sequence <= prev
        {
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("vector", "wal_sequence_out_of_order"),
                "insert WAL contains non-monotonic sequence",
            )
            .with_metadata("path", path.display().to_string())
            .with_metadata("line", (idx + 1).to_string())
            .with_metadata("previousSequence", prev.to_string())
            .with_metadata("sequence", record.sequence.to_string()));
        }
        previous = Some(record.sequence);
    }
    Ok(())
}

fn enforce_snapshot_limit(
    collection_name: &CollectionName,
    snapshot_path: &Path,
    version: VectorSnapshotWriteVersion,
    bytes: u64,
    max_bytes: Option<u64>,
) -> Result<()> {
    let Some(max_bytes) = max_bytes else {
        return Ok(());
    };
    if bytes <= max_bytes {
        return Ok(());
    }

    Err(ErrorEnvelope::expected(
        ErrorCode::new("vector", "snapshot_oversize"),
        "local snapshot exceeds configured size limit",
    )
    .with_metadata("collection", collection_name.as_str().to_string())
    .with_metadata("snapshotPath", snapshot_path.display().to_string())
    .with_metadata("version", version.as_str().to_string())
    .with_metadata("bytes", bytes.to_string())
    .with_metadata("maxBytes", max_bytes.to_string()))
}

fn log_json_snapshot_stats(
    collection_name: &CollectionName,
    snapshot_path: &Path,
    snapshot: &CollectionSnapshot,
    bytes: u64,
) {
    let count = snapshot.records.len();
    tracing::info!(
        collection = collection_name.as_str(),
        version = "v1",
        dimension = snapshot.dimension,
        count,
        bytes,
        snapshot_path = %snapshot_path.display(),
        "vectordb local snapshot stats"
    );
}

fn log_v2_snapshot_stats(
    collection_name: &CollectionName,
    snapshot_dir: &Path,
    stats: &SnapshotStats,
) {
    tracing::info!(
        collection = collection_name.as_str(),
        version = stats.version.as_str(),
        dimension = stats.dimension,
        count = stats.count,
        bytes = stats.bytes,
        snapshot_path = %snapshot_dir.display(),
        metadata = ?stats.metadata,
        "vectordb local snapshot stats"
    );
}

fn map_snapshot_write_error(
    error: ErrorEnvelope,
    collection_name: &CollectionName,
    snapshot_dir: &Path,
) -> ErrorEnvelope {
    if error.code == ErrorCode::new("vector", "snapshot_oversize") {
        return ErrorEnvelope::expected(
            ErrorCode::new("vector", "snapshot_oversize"),
            "local snapshot exceeds configured size limit",
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("snapshotPath", snapshot_dir.display().to_string())
        .with_metadata(
            "version",
            error
                .metadata
                .get("version")
                .cloned()
                .unwrap_or_else(|| "v2".to_string()),
        )
        .with_metadata(
            "bytes",
            error
                .metadata
                .get("bytes")
                .cloned()
                .unwrap_or_else(|| "0".to_string()),
        )
        .with_metadata(
            "maxBytes",
            error
                .metadata
                .get("maxBytes")
                .cloned()
                .unwrap_or_else(|| "0".to_string()),
        );
    }

    error
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("snapshotDir", snapshot_dir.display().to_string())
}

async fn path_exists(path: &Path) -> Result<bool> {
    match tokio::fs::metadata(path).await {
        Ok(metadata) => Ok(metadata.is_file()),
        Err(error) => {
            if error.kind() == std::io::ErrorKind::NotFound {
                Ok(false)
            } else {
                Err(ErrorEnvelope::from(error))
            }
        },
    }
}

fn snapshot_missing_companion_error(
    collection_name: &CollectionName,
    snapshot_dir: &Path,
    missing_files: &[&str],
) -> ErrorEnvelope {
    ErrorEnvelope::expected(
        ErrorCode::new("vector", "snapshot_missing_companion"),
        "snapshot v2 companion files are missing",
    )
    .with_metadata("collection", collection_name.as_str().to_string())
    .with_metadata("snapshotDir", snapshot_dir.display().to_string())
    .with_metadata("missingFiles", missing_files.join(","))
}

fn is_v2_companion_repairable_error(error: &ErrorEnvelope) -> bool {
    error.code == ErrorCode::new("vector", "snapshot_ids_read_failed")
        || error.code == ErrorCode::new("vector", "snapshot_ids_parse_failed")
        || error.code == ErrorCode::new("vector", "snapshot_record_count_mismatch")
        || error.code == ErrorCode::new("vector", "snapshot_record_missing")
}

/// Map a `spawn_blocking` [`tokio::task::JoinError`] into a typed
/// [`ErrorEnvelope`], distinguishing panics (invariant violations) from
/// cancellations (graceful shutdown).
fn map_spawn_blocking_join_error(
    join_error: &tokio::task::JoinError,
    code: ErrorCode,
    operation: &str,
    collection_name: &CollectionName,
) -> ErrorEnvelope {
    if join_error.is_panic() {
        ErrorEnvelope::invariant(code, format!("{operation} task panicked"))
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata("source", join_error.to_string())
    } else {
        ErrorEnvelope::cancelled(format!("{operation} task cancelled"))
            .with_metadata("collection", collection_name.as_str().to_string())
    }
}

fn collection_name_from_filename(filename: &str) -> Option<CollectionName> {
    let trimmed = filename.strip_suffix(".json")?;
    CollectionName::parse(trimmed).ok()
}

// ─── Collection Loader Actor types ──────────────────────────────────────────

/// Bounded channel capacity for the collection loader actor.
///
/// Justification: the CLI typically loads one collection at a time.
/// 8 provides headroom for burst (concurrent Load + internal
/// completions) and future agent-mode expansion.  Per module 40
/// sizing heuristic: 2× expected concurrent requests × burst headroom.
const LOADER_CHANNEL_CAPACITY: usize = 8;

/// Default timeout for a load operation (5 minutes).
///
/// Guards against pathologically large or corrupted files causing
/// unbounded waits.  Per module 06 §3: "every step is bounded."
const LOADER_OPERATION_TIMEOUT: Duration = Duration::from_mins(5);

/// Commands accepted by the [`CollectionLoaderActor`].
#[derive(Debug)]
enum CollectionLoaderCommand {
    /// Load a collection from disk into the shared map.
    /// No-op if the collection is already loaded.
    Load {
        name: CollectionName,
        reply: oneshot::Sender<Result<()>>,
        cancellation: CancellationToken,
    },
    /// Evict a collection from the shared map (for drop/clear).
    Evict { name: CollectionName },
}

// Compile-time assertion: commands must be sendable across tasks.
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<CollectionLoaderCommand>();
};

/// Cloneable handle for sending commands to the collection loader actor.
///
/// All methods that expect a reply use a bounded timeout
/// ([`LOADER_OPERATION_TIMEOUT`]) on the reply channel so that callers
/// never block indefinitely if the actor is stuck.
#[derive(Clone)]
struct CollectionLoaderHandle {
    tx: mpsc::Sender<CollectionLoaderCommand>,
}

impl CollectionLoaderHandle {
    /// Request that `name` is loaded from disk if not already present.
    async fn load(&self, name: CollectionName) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let load_cancellation = CancellationToken::new();
        self.tx
            .send(CollectionLoaderCommand::Load {
                name: name.clone(),
                reply: reply_tx,
                cancellation: load_cancellation.clone(),
            })
            .await
            .map_err(|_| loader_channel_closed_error(&name))?;
        receive_loader_reply(reply_rx, &name, "load", load_cancellation).await
    }

    /// Evict `name` from the shared map.  Fire-and-forget — no reply.
    async fn evict(&self, name: CollectionName) -> Result<()> {
        self.tx
            .send(CollectionLoaderCommand::Evict { name: name.clone() })
            .await
            .map_err(|_| loader_channel_closed_error(&name))
    }
}

/// Await a loader reply with the standard operation timeout.
///
/// On timeout the per-load `cancellation` token is cancelled so that the
/// in-progress `spawn_blocking` work (e.g. HNSW rebuild) cooperatively
/// stops instead of running for minutes after the caller has given up.
async fn receive_loader_reply(
    reply_rx: oneshot::Receiver<Result<()>>,
    collection_name: &CollectionName,
    operation: &str,
    cancellation: CancellationToken,
) -> Result<()> {
    match tokio::time::timeout(LOADER_OPERATION_TIMEOUT, reply_rx).await {
        Ok(Ok(result)) => result,
        Ok(Err(_)) => Err(ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "loader_reply_dropped"),
            format!("collection loader {operation} reply channel dropped"),
            ErrorClass::NonRetriable,
        )
        .with_metadata("collection", collection_name.as_str().to_string())),
        Err(_) => {
            cancellation.cancel();
            Err(ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "loader_timeout"),
                format!("collection loader {operation} timed out"),
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata(
                "timeoutSecs",
                LOADER_OPERATION_TIMEOUT.as_secs().to_string(),
            ))
        },
    }
}

struct LoadStageStart<'a> {
    actor: &'a CollectionLoaderActor,
    name: &'a CollectionName,
    cancellation: Option<CancellationToken>,
}

struct LoadStageLoaded<'a> {
    actor: &'a CollectionLoaderActor,
    name: &'a CollectionName,
    cancellation: Option<CancellationToken>,
    load_result: CollectionLoadResult,
}

struct LoadStageSnapshotBound<'a> {
    actor: &'a CollectionLoaderActor,
    name: &'a CollectionName,
    cancellation: Option<CancellationToken>,
    load_result: CollectionLoadResult,
}

struct LoadStageWalReplayed<'a> {
    actor: &'a CollectionLoaderActor,
    name: &'a CollectionName,
    cancellation: Option<CancellationToken>,
    collection: LocalCollection,
    checkpoint_sequence: u64,
    wal_replayed: bool,
    paths: Option<CollectionSnapshotPaths>,
    runtime_kernel_warm_state: RuntimeKernelWarmState,
    needs_metadata_rewrite: bool,
}

struct LoadStageKernelMaterialized<'a> {
    actor: &'a CollectionLoaderActor,
    name: &'a CollectionName,
    collection: LocalCollection,
    checkpoint_sequence: u64,
}

fn loader_channel_closed_error(collection_name: &CollectionName) -> ErrorEnvelope {
    ErrorEnvelope::unexpected(
        ErrorCode::new("vector", "loader_channel_closed"),
        "collection loader actor channel closed",
        ErrorClass::NonRetriable,
    )
    .with_metadata("collection", collection_name.as_str().to_string())
}

/// Actor that serializes collection lifecycle operations.
///
/// Hot read/write paths (insert, search) access the shared `collections`
/// map directly via `Arc<RwLock<HashMap>>`.  The actor only controls
/// *when* the map is populated (load) or cleared (evict) — it never
/// mediates individual document operations.
struct CollectionLoaderActor {
    loader: CollectionLoaderContext,
    collections: Arc<RwLock<HashMap<CollectionName, LocalCollection>>>,
    checkpoint_states: Arc<RwLock<HashMap<CollectionName, Arc<CollectionCheckpointState>>>>,
    rx: mpsc::Receiver<CollectionLoaderCommand>,
    cancellation: CancellationToken,
}

impl<'a> LoadStageStart<'a> {
    const fn new(
        actor: &'a CollectionLoaderActor,
        name: &'a CollectionName,
        cancellation: Option<CancellationToken>,
    ) -> Self {
        Self {
            actor,
            name,
            cancellation,
        }
    }

    async fn load(self) -> Result<LoadStageLoaded<'a>> {
        let load_result = self
            .actor
            .loader
            .load_collection(self.name, self.cancellation.clone())
            .await?;
        Ok(LoadStageLoaded {
            actor: self.actor,
            name: self.name,
            cancellation: self.cancellation,
            load_result,
        })
    }
}

impl<'a> LoadStageLoaded<'a> {
    fn bind_snapshot_contract(self) -> LoadStageSnapshotBound<'a> {
        if let Some(ref paths) = self.load_result.paths {
            if self.load_result.kernel_mismatch_requires_rebuild
                && let Err(error) = write_v2_from_collection(
                    self.name,
                    paths,
                    &self.load_result.collection,
                    self.actor.loader.kernel.kind(),
                    self.actor.loader.snapshot_max_bytes,
                    None,
                )
            {
                tracing::warn!(
                    collection = %self.name,
                    %error,
                    "post-kernel-mismatch v2 rewrite failed (non-fatal)"
                );
            }

            if self.load_result.needs_v2_rebuild
                && let Err(error) = write_v2_from_collection(
                    self.name,
                    paths,
                    &self.load_result.collection,
                    self.actor.loader.kernel.kind(),
                    self.actor.loader.snapshot_max_bytes,
                    None,
                )
            {
                tracing::warn!(
                    collection = %self.name,
                    %error,
                    "v1->v2 migration rewrite failed (non-fatal)"
                );
            }

            if let Ok(Some(snapshot_dir)) =
                resolve_kernel_snapshot_dir(&self.actor.loader, Some(paths))
            {
                self.actor.loader.kernel.set_snapshot_dir(&snapshot_dir);
            }
        }

        LoadStageSnapshotBound {
            actor: self.actor,
            name: self.name,
            cancellation: self.cancellation,
            load_result: self.load_result,
        }
    }
}

impl<'a> LoadStageSnapshotBound<'a> {
    async fn replay_wal(self) -> Result<LoadStageWalReplayed<'a>> {
        let mut collection = self.load_result.collection;
        let checkpoint_sequence = self.load_result.checkpoint_sequence;
        let wal_replayed = self
            .actor
            .replay_wal(self.name, &mut collection, checkpoint_sequence)
            .await?;

        Ok(LoadStageWalReplayed {
            actor: self.actor,
            name: self.name,
            cancellation: self.cancellation,
            collection,
            checkpoint_sequence,
            wal_replayed,
            paths: self.load_result.paths,
            runtime_kernel_warm_state: self.load_result.runtime_kernel_warm_state,
            needs_metadata_rewrite: self.load_result.needs_metadata_rewrite,
        })
    }
}

impl<'a> LoadStageWalReplayed<'a> {
    async fn warm_kernel(self) -> Result<LoadStageKernelMaterialized<'a>> {
        if self.actor.loader.runtime_dfrr_requires_prewarmed_state() {
            ensure_runtime_dfrr_ready_state_available(
                &self.actor.loader,
                self.name,
                self.paths.as_ref(),
            )
            .await?;
        }

        if self.runtime_kernel_warm_state != RuntimeKernelWarmState::WarmedFromGenerationSource
            || self.wal_replayed
        {
            warm_collection_kernel_state(
                &self.actor.loader,
                self.name,
                Arc::clone(&self.collection.index),
                self.paths.as_ref(),
                !self.wal_replayed,
                self.cancellation.clone(),
            )
            .await?;
        }

        if self.needs_metadata_rewrite
            && let Some(ref paths) = self.paths
            && let Err(error) = rewrite_v2_kernel_metadata(
                &self.actor.loader,
                self.name,
                paths,
                self.actor.loader.kernel.kind(),
            )
            .await
        {
            tracing::warn!(
                collection = %self.name,
                %error,
                "kernel metadata rewrite failed after tolerant load (non-fatal)"
            );
        }

        Ok(LoadStageKernelMaterialized {
            actor: self.actor,
            name: self.name,
            collection: self.collection,
            checkpoint_sequence: self.checkpoint_sequence,
        })
    }
}

impl LoadStageKernelMaterialized<'_> {
    async fn publish(self) -> Result<()> {
        self.actor
            .collections
            .write()
            .await
            .entry(self.name.clone())
            .or_insert(self.collection);
        self.actor
            .set_durable_hint(self.name, self.checkpoint_sequence)
            .await;
        Ok(())
    }
}

impl CollectionLoaderActor {
    /// Spawn the actor on the current Tokio runtime.
    ///
    /// Returns a `(handle, join_handle)` pair.  The `handle` is used to
    /// send commands.  The `join_handle` is used to await actor shutdown
    /// and observe any panics (per module 40: never ignore `JoinHandle`).
    fn spawn(
        loader: CollectionLoaderContext,
        collections: Arc<RwLock<HashMap<CollectionName, LocalCollection>>>,
        checkpoint_states: Arc<RwLock<HashMap<CollectionName, Arc<CollectionCheckpointState>>>>,
        cancellation: CancellationToken,
    ) -> (CollectionLoaderHandle, JoinHandle<()>) {
        let (tx, rx) = mpsc::channel(LOADER_CHANNEL_CAPACITY);
        let actor = Self {
            loader,
            collections,
            checkpoint_states,
            rx,
            cancellation,
        };
        let join = tokio::spawn(actor.run());
        (CollectionLoaderHandle { tx }, join)
    }

    async fn run(mut self) {
        loop {
            tokio::select! {
                () = self.cancellation.cancelled() => {
                    tracing::info!("collection loader actor shutting down (cancellation)");
                    break;
                }
                cmd = self.rx.recv() => {
                    match cmd {
                        Some(CollectionLoaderCommand::Load { name, reply, cancellation }) => {
                            let result = self.handle_load(&name, Some(cancellation)).await;
                            let _ = reply.send(result);
                        }
                        Some(CollectionLoaderCommand::Evict { name }) => {
                            self.handle_evict(&name).await;
                        }
                        None => {
                            tracing::info!("collection loader actor shutting down (channel closed)");
                            break;
                        }
                    }
                }
            }
        }
    }

    async fn handle_load(
        &self,
        name: &CollectionName,
        cancellation: Option<CancellationToken>,
    ) -> Result<()> {
        // Fast path: already loaded.
        {
            let collections = self.collections.read().await;
            if collections.contains_key(name) {
                return Ok(());
            }
        }

        LoadStageStart::new(self, name, cancellation)
            .load()
            .await?
            .bind_snapshot_contract()
            .replay_wal()
            .await?
            .warm_kernel()
            .await?
            .publish()
            .await
    }

    async fn handle_evict(&self, name: &CollectionName) {
        self.collections.write().await.remove(name);
    }

    async fn replay_wal(
        &self,
        name: &CollectionName,
        collection: &mut LocalCollection,
        checkpoint_sequence: u64,
    ) -> Result<bool> {
        let Some(paths) = self.loader.snapshot_paths(name) else {
            return Ok(false);
        };
        let state = self.checkpoint_state_for(name).await;
        let _wal_io = state.wal_io.lock().await;
        let records = read_insert_wal_records(paths.insert_wal.as_path()).await?;
        let replayed = records
            .iter()
            .any(|record| record.sequence > checkpoint_sequence);
        replay_insert_wal_records(
            paths.insert_wal.as_path(),
            collection,
            checkpoint_sequence,
            records.as_slice(),
        )?;
        Ok(replayed)
    }

    async fn checkpoint_state_for(&self, name: &CollectionName) -> Arc<CollectionCheckpointState> {
        {
            let states = self.checkpoint_states.read().await;
            if let Some(state) = states.get(name) {
                return Arc::clone(state);
            }
        }
        let mut states = self.checkpoint_states.write().await;
        Arc::clone(
            states
                .entry(name.clone())
                .or_insert_with(|| Arc::new(CollectionCheckpointState::new())),
        )
    }

    async fn set_durable_hint(&self, name: &CollectionName, seq: u64) {
        let state = self.checkpoint_state_for(name).await;
        let mut progress = state.progress.lock().await;
        progress.durable_sequence = progress.durable_sequence.max(seq);
        progress.scheduled_sequence = progress.scheduled_sequence.max(progress.durable_sequence);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_domain::LineSpan;
    use semantic_code_ports::VectorSearchOptions;
    use semantic_code_vector::{
        FlatScanKernel, HnswKernel, KernelSearchStats, SNAPSHOT_V2_HNSW_GRAPH_BASENAME,
        VectorSearchOutput,
    };
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    use tokio::time::timeout;

    fn sample_metadata(path: &str) -> Result<VectorDocumentMetadata> {
        Ok(VectorDocumentMetadata {
            relative_path: path.into(),
            language: None,
            file_extension: Some("rs".into()),
            span: LineSpan::new(1, 1)?,
            fragment_start_byte: None,
            fragment_end_byte: None,
            node_kind: None,
        })
    }

    fn deterministic_dense_unit_vector(seed: usize, dimension: usize) -> Vec<f32> {
        let mut vector = Vec::with_capacity(dimension);
        for column in 0..dimension {
            let raw = ((seed + 1) * 97 + (column + 1) * 57 + (seed * column + 13) * 17) % 1000;
            let centered = (raw as f32 / 500.0) - 1.0;
            vector.push(((seed % 7) as f32).mul_add(0.01, centered));
        }
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        vector
            .into_iter()
            .map(|value| value / norm.max(f32::EPSILON))
            .collect()
    }

    #[derive(Default)]
    struct TestDfrrKernel {
        warm_calls: AtomicUsize,
        search_calls: AtomicUsize,
        snapshot_dir: StdRwLock<Option<PathBuf>>,
    }

    impl TestDfrrKernel {
        fn warm_calls(&self) -> usize {
            self.warm_calls.load(Ordering::Relaxed)
        }

        fn search_calls(&self) -> usize {
            self.search_calls.load(Ordering::Relaxed)
        }

        fn snapshot_dir(&self) -> Option<PathBuf> {
            self.snapshot_dir
                .read()
                .ok()
                .and_then(|guard| guard.clone())
        }
    }

    impl VectorKernel for TestDfrrKernel {
        fn kind(&self) -> VectorKernelKind {
            VectorKernelKind::Dfrr
        }

        fn set_snapshot_dir(&self, dir: &Path) {
            if let Ok(mut guard) = self.snapshot_dir.write() {
                *guard = Some(dir.to_path_buf());
            }
        }

        fn warm(
            &self,
            _index: &VectorIndex,
            _context: &VectorKernelWarmContext,
            cancellation: Option<&CancellationToken>,
        ) -> Result<()> {
            if cancellation.is_some_and(CancellationToken::is_cancelled) {
                return Err(ErrorEnvelope::cancelled(
                    "test DFRR warm cancelled before materialization",
                ));
            }
            self.warm_calls.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn search(
            &self,
            index: &VectorIndex,
            query: &[f32],
            limit: usize,
            backend: VectorSearchBackend,
        ) -> Result<VectorSearchOutput> {
            self.search_calls.fetch_add(1, Ordering::Relaxed);
            let mut output = FlatScanKernel.search(index, query, limit, backend)?;
            output.stats = KernelSearchStats {
                kernel: VectorKernelKind::Dfrr,
                ..output.stats
            };
            Ok(output)
        }
    }

    #[tokio::test]
    async fn loader_handle_round_trip_load_command() {
        let (tx, mut rx) = mpsc::channel::<CollectionLoaderCommand>(LOADER_CHANNEL_CAPACITY);
        let handle = CollectionLoaderHandle { tx };
        let name = CollectionName::parse("test_collection").expect("valid collection name in test");

        // Spawn a responder that replies with Ok(()) for Load commands.
        let responder = tokio::spawn(async move {
            if let Some(CollectionLoaderCommand::Load { reply, .. }) = rx.recv().await {
                let _ = reply.send(Ok(()));
            }
        });

        let result = handle.load(name).await;
        assert!(result.is_ok(), "load round-trip should succeed");
        responder.await.expect("responder should not panic");
    }

    #[tokio::test]
    async fn loader_handle_round_trip_evict_command() {
        let (tx, mut rx) = mpsc::channel::<CollectionLoaderCommand>(LOADER_CHANNEL_CAPACITY);
        let handle = CollectionLoaderHandle { tx };
        let name = CollectionName::parse("test_collection").expect("valid collection name in test");

        let result = handle.evict(name).await;
        assert!(result.is_ok(), "evict send should succeed");

        match rx.recv().await {
            Some(CollectionLoaderCommand::Evict { .. }) => {},
            other => panic!("expected Evict command, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn loader_handle_timeout_on_dropped_reply() {
        // Use a tiny timeout to avoid slow tests.
        let (tx, mut rx) = mpsc::channel::<CollectionLoaderCommand>(LOADER_CHANNEL_CAPACITY);
        let handle = CollectionLoaderHandle { tx };
        let name = CollectionName::parse("test_collection").expect("valid collection name in test");

        // Spawn a responder that drops the reply sender without responding.
        let responder = tokio::spawn(async move {
            if let Some(CollectionLoaderCommand::Load { reply, .. }) = rx.recv().await {
                drop(reply);
            }
        });

        let result = handle.load(name).await;
        assert!(result.is_err(), "dropped reply should produce an error");
        let error = result.unwrap_err();
        assert_eq!(error.code, ErrorCode::new("vector", "loader_reply_dropped"));
        responder.await.expect("responder should not panic");
    }

    async fn build_hnsw_test_db(ef_search: usize) -> Result<(LocalVectorDb, CollectionName)> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-hnsw-ef-{}-{}",
            ef_search,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let db = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::with_ef_search(ef_search)),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp))
        .snapshot_format(VectorSnapshotFormat::V1)
        .build()?;
        let collection = CollectionName::parse("ef_search_honored")?;
        let ctx = RequestContext::new_request();
        db.create_collection(&ctx, collection.clone(), 3, None)
            .await?;
        let documents = (0..80)
            .map(|index| {
                let vector = deterministic_dense_unit_vector(index, 3);
                Ok(VectorDocumentForInsert {
                    id: format!("doc-{index}").into_boxed_str(),
                    vector: Arc::from(vector),
                    content: format!("content-{index}").into_boxed_str(),
                    metadata: sample_metadata(&format!("src/doc_{index}.rs"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        db.insert(&ctx, collection.clone(), documents).await?;
        Ok((db, collection))
    }

    #[tokio::test]
    async fn search_honors_explicit_hnsw_ef_when_threshold_is_zero() -> Result<()> {
        let (db, collection) = build_hnsw_test_db(32).await?;
        let ctx = RequestContext::new_request();
        let response = db
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection,
                    query_vector: Arc::from(deterministic_dense_unit_vector(1, 3)),
                    options: VectorSearchOptions {
                        top_k: Some(25),
                        filter_expr: None,
                        threshold: Some(0.0),
                    },
                },
            )
            .await?;
        let stats = response.stats.ok_or_else(|| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "missing search stats",
                ErrorClass::NonRetriable,
            )
        })?;
        assert_eq!(stats.extra.get("efSearch"), Some(&32.0));
        Ok(())
    }

    #[tokio::test]
    async fn search_widens_hnsw_ef_when_positive_threshold_needs_post_filtering() -> Result<()> {
        let (db, collection) = build_hnsw_test_db(32).await?;
        let ctx = RequestContext::new_request();
        let response = db
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection,
                    query_vector: Arc::from(deterministic_dense_unit_vector(1, 3)),
                    options: VectorSearchOptions {
                        top_k: Some(25),
                        filter_expr: None,
                        threshold: Some(0.5),
                    },
                },
            )
            .await?;
        let stats = response.stats.ok_or_else(|| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "missing search stats",
                ErrorClass::NonRetriable,
            )
        })?;
        assert_eq!(stats.extra.get("efSearch"), Some(&80.0));
        Ok(())
    }

    #[tokio::test]
    async fn filter_expr_allowlist_accepts_valid_inputs() -> Result<()> {
        let parsed = parse_filter_expr(Some("relativePath == 'src/lib.rs'"))?;
        assert!(parsed.is_some());
        Ok(())
    }

    #[tokio::test]
    async fn filter_expr_rejects_unknown_field() {
        let error = parse_filter_expr(Some("score > 0.5")).err();
        assert!(matches!(
            error,
            Some(envelope) if envelope.code == ErrorCode::new("vector", "invalid_filter_expr")
        ));
    }

    #[tokio::test]
    async fn snapshot_paths_resolve_v2_bundle() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-paths-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let db = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp))
        .build()?;
        let collection = CollectionName::parse("local_snapshot")?;
        let paths = db.snapshot_paths(&collection).ok_or_else(|| {
            std::io::Error::other("expected snapshot paths for custom storage mode")
        })?;

        assert!(
            paths
                .v1_json
                .ends_with("vector/collections/local_snapshot.json")
        );
        assert!(
            paths
                .v2_dir
                .ends_with("vector/collections/local_snapshot.v2")
        );
        assert!(
            paths
                .v2_meta
                .ends_with(format!("local_snapshot.v2/{SNAPSHOT_V2_META_FILE_NAME}"))
        );
        assert!(
            paths
                .v2_vectors
                .ends_with(format!("local_snapshot.v2/{SNAPSHOT_V2_VECTORS_FILE_NAME}"))
        );
        assert!(paths.v2_ids.ends_with(format!(
            "local_snapshot.v2/{LOCAL_SNAPSHOT_V2_IDS_FILE_NAME}"
        )));
        assert!(paths.v2_records_meta.ends_with(format!(
            "local_snapshot.v2/{LOCAL_SNAPSHOT_V2_RECORDS_META_FILE_NAME}"
        )));
        assert!(paths.insert_wal.ends_with(format!(
            "vector/collections/local_snapshot{LOCAL_INSERT_WAL_FILE_SUFFIX}"
        )));
        assert!(
            paths
                .generation_layout
                .root()
                .ends_with("vector/collections/local_snapshot"),
            "unexpected generation layout root: {}",
            paths.generation_layout.root().display()
        );
        Ok(())
    }

    #[tokio::test]
    async fn insert_and_flush_persists_checkpoint_sequence_and_compacts_wal() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-wal-sequence-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let collection = CollectionName::parse("wal_sequence")?;
        let ctx = RequestContext::new_request();

        let db = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .snapshot_format(VectorSnapshotFormat::V1)
        .build()?;

        db.create_collection(&ctx, collection.clone(), 3, None)
            .await?;
        db.insert(
            &ctx,
            collection.clone(),
            vec![VectorDocumentForInsert {
                id: "doc1".into(),
                vector: Arc::from(vec![0.1, 0.2, 0.3]),
                content: "one".into(),
                metadata: sample_metadata("src/doc1.rs")?,
            }],
        )
        .await?;
        db.insert(
            &ctx,
            collection.clone(),
            vec![VectorDocumentForInsert {
                id: "doc2".into(),
                vector: Arc::from(vec![0.4, 0.5, 0.6]),
                content: "two".into(),
                metadata: sample_metadata("src/doc2.rs")?,
            }],
        )
        .await?;
        db.insert(
            &ctx,
            collection.clone(),
            vec![VectorDocumentForInsert {
                id: "doc3".into(),
                vector: Arc::from(vec![0.7, 0.8, 0.9]),
                content: "three".into(),
                metadata: sample_metadata("src/doc3.rs")?,
            }],
        )
        .await?;
        db.flush(&ctx, collection.clone()).await?;

        let paths = db.snapshot_paths(&collection).ok_or_else(|| {
            std::io::Error::other("expected snapshot paths for custom storage mode")
        })?;
        assert!(
            !path_exists(paths.insert_wal.as_path()).await?,
            "expected insert WAL to be compacted after flush"
        );

        let snapshot = db
            .read_snapshot_json(&collection)
            .await?
            .ok_or_else(|| std::io::Error::other("expected persisted snapshot"))?;
        assert_eq!(snapshot.checkpoint_sequence, Some(3));
        Ok(())
    }

    #[tokio::test]
    async fn clean_slate_collection_stages_inserts_until_flush_and_publishes_generation()
    -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-generation-flush-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let db = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .build()?;
        let collection = CollectionName::parse("generation_stage")?;
        let ctx = RequestContext::new_request();

        db.create_collection(&ctx, collection.clone(), 3, None)
            .await?;

        let paths = db.snapshot_paths(&collection).ok_or_else(|| {
            std::io::Error::other("expected snapshot paths for custom storage mode")
        })?;

        db.insert(
            &ctx,
            collection.clone(),
            vec![
                VectorDocumentForInsert {
                    id: "doc_a".into(),
                    vector: Arc::from(vec![1.0, 0.0, 0.0]),
                    content: "document a".into(),
                    metadata: sample_metadata("src/a.rs")?,
                },
                VectorDocumentForInsert {
                    id: "doc_b".into(),
                    vector: Arc::from(vec![0.0, 1.0, 0.0]),
                    content: "document b".into(),
                    metadata: sample_metadata("src/b.rs")?,
                },
            ],
        )
        .await?;

        assert!(
            !path_exists(paths.insert_wal.as_path()).await?,
            "staged clean-slate inserts should not use the collection WAL before flush"
        );
        assert!(
            path_exists(paths.build_rows.as_path()).await?,
            "staged clean-slate inserts should append to the build journal rows file"
        );
        assert!(
            path_exists(paths.build_vectors.as_path()).await?,
            "staged clean-slate inserts should append to the build journal vectors file"
        );

        {
            let collections = db.collections.read().await;
            let collection_state = collections.get(&collection).ok_or_else(|| {
                std::io::Error::other("expected in-memory collection after staged insert")
            })?;
            assert!(
                collection_state.is_staging(),
                "newly created collection should stay in staging mode before flush"
            );
            let index = collection_state.read_index()?;
            assert_eq!(
                index.host_hnsw_count(),
                0,
                "staged inserts should not materialize host HNSW before flush"
            );
            assert_eq!(index.active_count(), 2);
        }

        db.flush(&ctx, collection.clone()).await?;

        assert!(
            path_exists(paths.build_meta.as_path()).await?,
            "flush should seal the build journal with metadata"
        );
        assert!(
            path_exists(paths.build_sealed.as_path()).await?,
            "flush should mark the build journal as sealed"
        );

        let active_generation = std::fs::read_to_string(paths.generation_layout.active_file())
            .map_err(ErrorEnvelope::from)?;
        let active_generation = active_generation.trim().to_string();
        assert!(
            !active_generation.is_empty(),
            "ACTIVE generation pointer should be written after flush"
        );
        let generation_id = semantic_code_vector::GenerationId::new(active_generation.as_str())
            .map_err(|error| {
                std::io::Error::other(format!("invalid generation id in ACTIVE file: {error}"))
            })?;
        let generation = paths.generation_layout.generation(&generation_id);
        assert!(
            generation
                .base_dir()
                .join(semantic_code_vector::EXACT_GENERATION_META_FILE_NAME)
                .is_file(),
            "expected exact generation metadata after flush"
        );
        assert!(
            generation
                .base_dir()
                .join(semantic_code_vector::EXACT_GENERATION_VECTORS_FILE_NAME)
                .is_file(),
            "expected exact generation vectors after flush"
        );
        assert!(
            generation
                .kernel_dir(VectorKernelKind::HnswRs)
                .join(format!("{SNAPSHOT_V2_HNSW_GRAPH_BASENAME}.hnsw.graph"))
                .is_file(),
            "expected generation-scoped hnsw ready-state graph after flush"
        );

        {
            let collections = db.collections.read().await;
            let collection_state = collections.get(&collection).ok_or_else(|| {
                std::io::Error::other("expected in-memory collection after flush")
            })?;
            assert!(
                !collection_state.is_staging(),
                "flush should promote staged collections back to online mode"
            );
            let index = collection_state.read_index()?;
            assert_eq!(
                index.host_hnsw_count(),
                2,
                "flush should materialize host HNSW for hnsw-rs runtime collections"
            );
        }

        let response = db
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection.clone(),
                    query_vector: Arc::from(vec![1.0, 0.0, 0.0]),
                    options: VectorSearchOptions {
                        top_k: Some(1),
                        filter_expr: None,
                        threshold: None,
                    },
                },
            )
            .await?;
        assert_eq!(response.results[0].document.id, "doc_a".into());

        Ok(())
    }

    #[tokio::test]
    async fn restart_loads_from_published_active_generation() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-generation-restart-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let collection = CollectionName::parse("generation_restart")?;
        let ctx = RequestContext::new_request();

        let initial = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .build()?;
        initial
            .create_collection(&ctx, collection.clone(), 3, None)
            .await?;
        initial
            .insert(
                &ctx,
                collection.clone(),
                vec![VectorDocumentForInsert {
                    id: "doc_restart".into(),
                    vector: Arc::from(vec![0.1, 0.2, 0.3]),
                    content: "hello".into(),
                    metadata: sample_metadata("src/restart.rs")?,
                }],
            )
            .await?;
        initial.flush(&ctx, collection.clone()).await?;

        let paths = initial.snapshot_paths(&collection).ok_or_else(|| {
            std::io::Error::other("expected snapshot paths for custom storage mode")
        })?;
        let active_generation = paths
            .generation_layout
            .read_active_generation_id()?
            .ok_or_else(|| std::io::Error::other("expected active generation after flush"))?;
        assert!(
            paths
                .generation_layout
                .generation(&active_generation)
                .base_dir()
                .join(semantic_code_vector::EXACT_GENERATION_META_FILE_NAME)
                .is_file(),
            "expected exact generation metadata before restart"
        );
        let generation = paths.generation_layout.generation(&active_generation);
        let _ = std::fs::remove_file(
            paths
                .v2_dir
                .join(format!("{SNAPSHOT_V2_HNSW_GRAPH_BASENAME}.hnsw.graph")),
        );
        let _ = std::fs::remove_file(
            paths
                .v2_dir
                .join(format!("{SNAPSHOT_V2_HNSW_GRAPH_BASENAME}.hnsw.data")),
        );
        assert!(
            generation
                .kernel_dir(VectorKernelKind::HnswRs)
                .join(format!("{SNAPSHOT_V2_HNSW_GRAPH_BASENAME}.hnsw.graph"))
                .is_file(),
            "expected generation-scoped hnsw ready-state graph before restart"
        );
        drop(initial);

        let restarted = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .build()?;

        let response = restarted
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection.clone(),
                    query_vector: Arc::from(vec![0.1, 0.2, 0.3]),
                    options: VectorSearchOptions {
                        top_k: Some(1),
                        filter_expr: None,
                        threshold: None,
                    },
                },
            )
            .await?;
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].document.id, "doc_restart".into());

        {
            let collections = restarted.collections.read().await;
            let collection_state = collections.get(&collection).ok_or_else(|| {
                std::io::Error::other("expected collection to be loaded after restart search")
            })?;
            assert!(
                !collection_state.is_staging(),
                "restart-loaded collection should already be online"
            );
            let index = collection_state.read_index()?;
            assert_eq!(index.host_hnsw_count(), 1);
        }

        Ok(())
    }

    #[tokio::test]
    async fn restart_replays_wal_records_before_flush_and_flush_persists_state() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-wal-replay-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let collection = CollectionName::parse("wal_replay")?;
        let ctx = RequestContext::new_request();

        let initial = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .snapshot_format(VectorSnapshotFormat::V1)
        .build()?;
        initial.set_checkpoint_delay_for_tests(Duration::from_secs(30));
        initial
            .create_collection(&ctx, collection.clone(), 3, None)
            .await?;
        initial
            .insert(
                &ctx,
                collection.clone(),
                vec![VectorDocumentForInsert {
                    id: "doc1".into(),
                    vector: Arc::from(vec![0.1, 0.2, 0.3]),
                    content: "hello".into(),
                    metadata: sample_metadata("src/replayed.rs")?,
                }],
            )
            .await?;

        let paths = initial.snapshot_paths(&collection).ok_or_else(|| {
            std::io::Error::other("expected snapshot paths for custom storage mode")
        })?;
        assert!(
            path_exists(paths.insert_wal.as_path()).await?,
            "expected WAL record to exist before restart"
        );
        drop(initial);

        let restarted = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .snapshot_format(VectorSnapshotFormat::V1)
        .build()?;
        let replayed = restarted
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection.clone(),
                    query_vector: Arc::from(vec![0.1, 0.2, 0.3]),
                    options: VectorSearchOptions {
                        top_k: Some(1),
                        filter_expr: None,
                        threshold: None,
                    },
                },
            )
            .await?;
        assert_eq!(replayed.results.len(), 1);
        assert_eq!(replayed.results[0].document.id, "doc1".into());
        assert_eq!(replayed.results[0].document.content.as_ref(), "hello");
        assert_eq!(
            replayed.results[0].document.metadata.relative_path.as_ref(),
            "src/replayed.rs"
        );

        restarted.flush(&ctx, collection.clone()).await?;
        assert!(
            !path_exists(paths.insert_wal.as_path()).await?,
            "expected replayed WAL entries to compact after flush"
        );
        let snapshot = restarted
            .read_snapshot_json(&collection)
            .await?
            .ok_or_else(|| std::io::Error::other("expected persisted snapshot"))?;
        assert_eq!(snapshot.checkpoint_sequence, Some(1));
        drop(restarted);

        let reopened = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .snapshot_format(VectorSnapshotFormat::V1)
        .build()?;
        let restored = reopened
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection.clone(),
                    query_vector: Arc::from(vec![0.1, 0.2, 0.3]),
                    options: VectorSearchOptions {
                        top_k: Some(1),
                        filter_expr: None,
                        threshold: None,
                    },
                },
            )
            .await?;
        assert_eq!(restored.results.len(), 1);
        assert_eq!(restored.results[0].document.id, "doc1".into());
        Ok(())
    }

    #[tokio::test]
    async fn restart_rejects_wal_replay_sequence_gap() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-wal-gap-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let collection = CollectionName::parse("wal_gap")?;
        let ctx = RequestContext::new_request();

        let db = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .snapshot_format(VectorSnapshotFormat::V1)
        .build()?;
        db.create_collection(&ctx, collection.clone(), 3, None)
            .await?;
        db.insert(
            &ctx,
            collection.clone(),
            vec![VectorDocumentForInsert {
                id: "doc1".into(),
                vector: Arc::from(vec![0.1, 0.2, 0.3]),
                content: "hello".into(),
                metadata: sample_metadata("src/doc1.rs")?,
            }],
        )
        .await?;
        db.flush(&ctx, collection.clone()).await?;

        let paths = db.snapshot_paths(&collection).ok_or_else(|| {
            std::io::Error::other("expected snapshot paths for custom storage mode")
        })?;
        append_insert_wal_record(
            paths.insert_wal.as_path(),
            &InsertWalRecord {
                sequence: 3,
                documents: vec![InsertWalDocument {
                    id: "doc2".into(),
                    vector: vec![0.3, 0.2, 0.1],
                    content: "gap".into(),
                    metadata: sample_metadata("src/doc2.rs")?,
                }],
            },
        )
        .await?;
        drop(db);

        let restarted = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .snapshot_format(VectorSnapshotFormat::V1)
        .build()?;
        let error = restarted
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection.clone(),
                    query_vector: Arc::from(vec![0.1, 0.2, 0.3]),
                    options: VectorSearchOptions {
                        top_k: Some(1),
                        filter_expr: None,
                        threshold: None,
                    },
                },
            )
            .await
            .err()
            .ok_or_else(|| std::io::Error::other("expected WAL replay sequence error"))?;
        assert_eq!(
            error.code,
            ErrorCode::new("vector", "wal_replay_sequence_gap")
        );
        assert_eq!(
            error.metadata.get("expectedSequence").map(String::as_str),
            Some("2")
        );
        assert_eq!(
            error.metadata.get("sequence").map(String::as_str),
            Some("3")
        );
        Ok(())
    }

    #[tokio::test]
    async fn flush_waits_for_pending_checkpoint_and_returns_when_durable() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-flush-waits-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let collection = CollectionName::parse("flush_waits")?;
        let ctx = RequestContext::new_request();

        let db = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .snapshot_format(VectorSnapshotFormat::V1)
        .build()?;
        db.set_checkpoint_delay_for_tests(Duration::from_millis(150));
        db.create_collection(&ctx, collection.clone(), 3, None)
            .await?;
        db.insert(
            &ctx,
            collection.clone(),
            vec![VectorDocumentForInsert {
                id: "doc1".into(),
                vector: Arc::from(vec![0.1, 0.2, 0.3]),
                content: "hello".into(),
                metadata: sample_metadata("src/lib.rs")?,
            }],
        )
        .await?;

        let first_flush = timeout(
            Duration::from_millis(20),
            db.flush(&ctx, collection.clone()),
        )
        .await;
        assert!(
            first_flush.is_err(),
            "flush returned before pending checkpoint became durable"
        );

        db.set_checkpoint_delay_for_tests(Duration::ZERO);
        db.flush(&ctx, collection.clone()).await?;

        let state = db.checkpoint_state_for_collection(&collection).await;
        db.reap_finished_checkpoint_worker(&collection, &state)
            .await;
        let progress = state.progress.lock().await;
        assert!(progress.durable_sequence >= 1);
        assert_eq!(progress.scheduled_sequence, 1);
        Ok(())
    }

    #[test]
    fn hnsw_kernel_always_supported() {
        assert!(
            LocalVectorDb::is_kernel_supported(ConfigVectorKernelKind::HnswRs),
            "HnswRs kernel should always be supported"
        );
    }

    #[test]
    fn dfrr_kernel_support_matches_feature_flag() {
        let supported = LocalVectorDb::is_kernel_supported(ConfigVectorKernelKind::Dfrr);
        assert_eq!(
            supported,
            cfg!(feature = "experimental-dfrr-kernel"),
            "DFRR support should match the experimental-dfrr-kernel feature flag"
        );
    }

    #[tokio::test]
    async fn v2_load_falls_back_to_v1_snapshot_and_migrates() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-v2-fallback-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let collection = CollectionName::parse("local_snapshot")?;
        let ctx = RequestContext::new_request();
        let db_v1 = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .snapshot_format(VectorSnapshotFormat::V1)
        .build()?;

        db_v1
            .create_collection(&ctx, collection.clone(), 3, None)
            .await?;
        db_v1
            .insert(
                &ctx,
                collection.clone(),
                vec![VectorDocumentForInsert {
                    id: "doc1".into(),
                    vector: Arc::from(vec![0.1, 0.2, 0.3]),
                    content: "hello".into(),
                    metadata: sample_metadata("src/lib.rs")?,
                }],
            )
            .await?;
        db_v1.flush(&ctx, collection.clone()).await?;

        let paths = db_v1.snapshot_paths(&collection).ok_or_else(|| {
            std::io::Error::other("expected snapshot paths for custom storage mode")
        })?;
        assert!(paths.v1_json.is_file());
        assert!(!paths.v2_meta.is_file());
        assert!(!paths.v2_vectors.is_file());
        assert!(!paths.v2_ids.is_file());
        let _ = std::fs::remove_file(paths.generation_layout.active_file());

        let db_v2 = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .build()?;
        let response = db_v2
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection.clone(),
                    query_vector: Arc::from(vec![0.1, 0.2, 0.3]),
                    options: VectorSearchOptions {
                        top_k: Some(1),
                        filter_expr: None,
                        threshold: None,
                    },
                },
            )
            .await?;

        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].document.id, "doc1".into());
        assert!(response.stats.is_some());
        assert!(paths.v2_meta.is_file());
        assert!(paths.v2_vectors.is_file());
        assert!(paths.v2_ids.is_file());
        Ok(())
    }

    #[tokio::test]
    async fn v2_load_repairs_stale_companion_ids_from_v1_snapshot() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-v2-repair-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let collection = CollectionName::parse("local_snapshot_repair")?;
        let ctx = RequestContext::new_request();
        // Use V1 format to write a v1 JSON snapshot with the initial record.
        // The repair path requires float32 vectors stored in v1 JSON; in pure V2
        // mode those vectors are never persisted to disk, so this test exercises
        // the V1→V2 migration repair scenario.
        let db_v1 = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .snapshot_format(VectorSnapshotFormat::V1)
        .build()?;
        db_v1
            .create_collection(&ctx, collection.clone(), 3, None)
            .await?;
        db_v1
            .insert(
                &ctx,
                collection.clone(),
                vec![VectorDocumentForInsert {
                    id: "doc1".into(),
                    vector: Arc::from(vec![0.1, 0.2, 0.3]),
                    content: "hello".into(),
                    metadata: sample_metadata("src/lib.rs")?,
                }],
            )
            .await?;
        db_v1.flush(&ctx, collection.clone()).await?;
        let v1_paths = db_v1.snapshot_paths(&collection).ok_or_else(|| {
            std::io::Error::other("expected snapshot paths for custom storage mode")
        })?;
        let _ = std::fs::remove_file(v1_paths.generation_layout.active_file());
        // Open the same directory as V2 and trigger a load so the V2 companion
        // files (ids.json, vectors.u8.bin, meta.json, records.meta.jsonl) are
        // built from the V1 JSON.
        let db_v2_init = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .build()?;
        // A search triggers ensure_loaded → load_via_v1_json → rebuild V2 bundle.
        let _ = db_v2_init
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection.clone(),
                    query_vector: Arc::from(vec![0.1, 0.2, 0.3]),
                    options: VectorSearchOptions {
                        top_k: Some(1),
                        filter_expr: None,
                        threshold: None,
                    },
                },
            )
            .await?;
        drop(db_v2_init);
        let db = db_v1;

        let paths = db.snapshot_paths(&collection).ok_or_else(|| {
            std::io::Error::other("expected snapshot paths for custom storage mode")
        })?;
        let snapshot_bytes = tokio::fs::read(paths.v1_json.as_path())
            .await
            .map_err(ErrorEnvelope::from)?;
        let mut snapshot: CollectionSnapshot =
            serde_json::from_slice(&snapshot_bytes).map_err(|error| {
                ErrorEnvelope::expected(
                    ErrorCode::new("vector", "snapshot_parse_failed"),
                    error.to_string(),
                )
            })?;
        let mut synthetic = snapshot
            .records
            .first()
            .cloned()
            .ok_or_else(|| std::io::Error::other("expected at least one snapshot record"))
            .map_err(ErrorEnvelope::from)?;
        synthetic.id = "doc_missing".into();
        snapshot.records.push(synthetic);
        let mutated_payload = serde_json::to_vec(&snapshot).map_err(|error| {
            ErrorEnvelope::expected(
                ErrorCode::new("vector", "snapshot_serialize_failed"),
                error.to_string(),
            )
        })?;
        tokio::fs::write(paths.v1_json.as_path(), mutated_payload)
            .await
            .map_err(ErrorEnvelope::from)?;
        // Remove the metadata sidecar so the legacy v1→v2 repair path is exercised.
        let _ = tokio::fs::remove_file(paths.v2_records_meta.as_path()).await;

        let pre_repair_ids = tokio::fs::read(paths.v2_ids.as_path())
            .await
            .map_err(ErrorEnvelope::from)?;
        let pre_repair_ids: Vec<Box<str>> =
            serde_json::from_slice(&pre_repair_ids).map_err(|error| {
                ErrorEnvelope::expected(
                    ErrorCode::new("vector", "snapshot_ids_parse_failed"),
                    error.to_string(),
                )
            })?;
        assert!(
            !pre_repair_ids.iter().any(|id| id.as_ref() == "doc_missing"),
            "test setup failed: stale companion should not contain synthetic id"
        );
        let _ = std::fs::remove_file(paths.generation_layout.active_file());

        let restarted = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .build()?;
        let response = restarted
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection.clone(),
                    query_vector: Arc::from(vec![0.1, 0.2, 0.3]),
                    options: VectorSearchOptions {
                        top_k: Some(2),
                        filter_expr: None,
                        threshold: None,
                    },
                },
            )
            .await?;
        assert!(!response.results.is_empty());

        let repaired_ids = tokio::fs::read(paths.v2_ids.as_path())
            .await
            .map_err(ErrorEnvelope::from)?;
        let repaired_ids: Vec<Box<str>> =
            serde_json::from_slice(&repaired_ids).map_err(|error| {
                ErrorEnvelope::expected(
                    ErrorCode::new("vector", "snapshot_ids_parse_failed"),
                    error.to_string(),
                )
            })?;
        assert!(
            repaired_ids.iter().any(|id| id.as_ref() == "doc_missing"),
            "expected companion rewrite to include synthetic id from v1 snapshot"
        );
        assert_eq!(repaired_ids.len(), snapshot.records.len());
        Ok(())
    }

    #[tokio::test]
    async fn v2_checkpoint_write_path_stays_on_active_collection_index() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-v2-checkpoint-idx-reuse-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let collection = CollectionName::parse("checkpoint_active_restart")?;
        let ctx = RequestContext::new_request();

        let seed = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .build()?;
        seed.create_collection(&ctx, collection.clone(), 3, None)
            .await?;
        seed.insert(
            &ctx,
            collection.clone(),
            vec![VectorDocumentForInsert {
                id: "doc1".into(),
                vector: Arc::from(vec![0.1, 0.2, 0.3]),
                content: "hello".into(),
                metadata: sample_metadata("src/lib.rs")?,
            }],
        )
        .await?;
        seed.flush(&ctx, collection.clone()).await?;
        drop(seed);

        let db = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .build()?;
        let paths = db.snapshot_paths(&collection).ok_or_else(|| {
            std::io::Error::other("expected snapshot paths for custom storage mode")
        })?;
        let response = db
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection.clone(),
                    query_vector: Arc::from(vec![0.1, 0.2, 0.3]),
                    options: VectorSearchOptions {
                        top_k: Some(1),
                        filter_expr: None,
                        threshold: None,
                    },
                },
            )
            .await?;
        assert_eq!(response.results.len(), 1);

        // Corrupt the legacy V1 snapshot to ensure this path cannot fall back
        // during checkpoint writes.
        tokio::fs::write(paths.v1_json.as_path(), b"{this is not valid json")
            .await
            .map_err(ErrorEnvelope::from)?;

        db.reset_v2_write_path_counters();
        db.insert(
            &ctx,
            collection.clone(),
            vec![VectorDocumentForInsert {
                id: "doc2".into(),
                vector: Arc::from(vec![0.4, 0.5, 0.6]),
                content: "second".into(),
                metadata: sample_metadata("src/lib2.rs")?,
            }],
        )
        .await?;
        db.flush(&ctx, collection.clone()).await?;

        let (from_collection_calls, from_bundle_calls) = db.v2_write_path_calls();
        assert_eq!(
            from_collection_calls, 1,
            "expected V2 checkpoint write from active index"
        );
        assert_eq!(
            from_bundle_calls, 0,
            "unexpected V2 checkpoint rebuild from serialized bundle"
        );

        let response = db
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection,
                    query_vector: Arc::from(vec![0.4, 0.5, 0.6]),
                    options: VectorSearchOptions {
                        top_k: Some(1),
                        filter_expr: None,
                        threshold: None,
                    },
                },
            )
            .await?;
        assert_eq!(response.results[0].document.id, "doc2".into());
        Ok(())
    }

    #[tokio::test]
    async fn v2_checkpoint_flush_preserves_insertion_order_for_ids_and_graph() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-v2-checkpoint-order-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let collection = CollectionName::parse("checkpoint_order")?;
        let ctx = RequestContext::new_request();
        let db = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .build()?;
        db.create_collection(&ctx, collection.clone(), 72, None)
            .await?;

        let batch_orders: [[usize; 12]; 6] = [
            [41, 3, 57, 8, 66, 1, 52, 11, 70, 5, 61, 14],
            [23, 48, 19, 35, 27, 44, 16, 31, 21, 39, 18, 29],
            [62, 7, 55, 13, 68, 2, 50, 10, 64, 4, 59, 12],
            [24, 47, 20, 34, 26, 45, 17, 30, 22, 38, 15, 28],
            [63, 6, 56, 9, 69, 0, 51, 25, 65, 32, 60, 36],
            [71, 37, 58, 33, 67, 40, 53, 42, 54, 43, 46, 49],
        ];

        let mut expected_ids = Vec::with_capacity(72);
        for batch in batch_orders {
            let mut documents = Vec::with_capacity(batch.len());
            for position in batch {
                let id: Box<str> = format!("chunk_{position:04}").into_boxed_str();
                expected_ids.push(id.clone());
                documents.push(VectorDocumentForInsert {
                    id,
                    vector: Arc::from(deterministic_dense_unit_vector(position, 72)),
                    content: format!("content-{position}").into_boxed_str(),
                    metadata: sample_metadata(&format!("src/doc_{position:04}.rs"))?,
                });
            }
            db.insert(&ctx, collection.clone(), documents).await?;
        }
        db.flush(&ctx, collection.clone()).await?;

        let paths = db.snapshot_paths(&collection).ok_or_else(|| {
            std::io::Error::other("expected snapshot paths for custom storage mode")
        })?;
        let persisted_ids = tokio::fs::read(paths.v2_ids.as_path())
            .await
            .map_err(ErrorEnvelope::from)?;
        let persisted_ids: Vec<Box<str>> =
            serde_json::from_slice(&persisted_ids).map_err(|error| {
                ErrorEnvelope::expected(
                    ErrorCode::new("vector", "snapshot_ids_parse_failed"),
                    error.to_string(),
                )
            })?;

        assert_eq!(
            persisted_ids, expected_ids,
            "fresh V2 checkpoints must preserve insertion order in ids.json when graph reuse is enabled"
        );

        assert!(
            paths
                .v2_dir
                .join(format!("{SNAPSHOT_V2_HNSW_GRAPH_BASENAME}.hnsw.graph"))
                .is_file(),
            "expected persisted HNSW graph for graph-safe snapshot"
        );

        drop(db);
        let restarted = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .build()?;
        for probe in [41_usize, 48, 62, 24, 63, 71] {
            let query = deterministic_dense_unit_vector(probe, 72);
            let response = restarted
                .search(
                    &ctx,
                    VectorSearchRequest {
                        collection_name: collection.clone(),
                        query_vector: Arc::from(query),
                        options: VectorSearchOptions {
                            top_k: Some(1),
                            filter_expr: None,
                            threshold: None,
                        },
                    },
                )
                .await?;
            assert_eq!(
                response.results[0].document.id,
                format!("chunk_{probe:04}").into_boxed_str(),
                "persisted-graph reload must preserve top-1 identity for exact self queries"
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn snapshot_roundtrip_persists_records() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let db = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .snapshot_format(VectorSnapshotFormat::V1)
        .build()?;
        let collection = CollectionName::parse("local_snapshot")?;
        let ctx = RequestContext::new_request();
        db.create_collection(&ctx, collection.clone(), 3, None)
            .await?;
        db.insert(
            &ctx,
            collection.clone(),
            vec![VectorDocumentForInsert {
                id: "doc1".into(),
                vector: Arc::from(vec![0.1, 0.2, 0.3]),
                content: "hello".into(),
                metadata: sample_metadata("src/lib.rs")?,
            }],
        )
        .await?;
        db.flush(&ctx, collection.clone()).await?;

        let restored = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .snapshot_format(VectorSnapshotFormat::V1)
        .build()?;
        let response = restored
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection,
                    query_vector: Arc::from(vec![0.1, 0.2, 0.3]),
                    options: VectorSearchOptions {
                        top_k: Some(1),
                        filter_expr: None,
                        threshold: None,
                    },
                },
            )
            .await?;

        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].document.id, "doc1".into());
        assert!(response.stats.is_some());
        Ok(())
    }

    #[tokio::test]
    async fn v2_load_rejects_kernel_mismatch_without_force_reindex() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-kernel-mismatch-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let collection = CollectionName::parse("kernel_mismatch")?;
        let ctx = RequestContext::new_request();

        let db = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .build()?;
        db.create_collection(&ctx, collection.clone(), 3, None)
            .await?;

        let paths = db.snapshot_paths(&collection).ok_or_else(|| {
            std::io::Error::other("expected snapshot paths for custom storage mode")
        })?;
        let mut meta = read_metadata(paths.v2_meta.as_path()).map_err(|error| {
            ErrorEnvelope::expected(
                ErrorCode::new("vector", "snapshot_invalid"),
                error.to_string(),
            )
        })?;
        meta.kernel = VectorKernelKind::Dfrr;
        semantic_code_vector::write_metadata(paths.v2_meta.as_path(), &meta).map_err(|error| {
            ErrorEnvelope::expected(
                ErrorCode::new("vector", "snapshot_invalid"),
                error.to_string(),
            )
        })?;

        let restarted = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .build()?;
        let error = restarted
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection,
                    query_vector: Arc::from(vec![0.1, 0.2, 0.3]),
                    options: VectorSearchOptions {
                        top_k: Some(1),
                        filter_expr: None,
                        threshold: None,
                    },
                },
            )
            .await
            .err()
            .ok_or_else(|| std::io::Error::other("expected kernel mismatch error"))?;
        assert_eq!(
            error.code,
            ErrorCode::new("vector", "snapshot_kernel_mismatch")
        );
        assert_eq!(
            error.metadata.get("collection").map(String::as_str),
            Some("kernel_mismatch")
        );
        assert_eq!(
            error.metadata.get("snapshotKernel").map(String::as_str),
            Some("dfrr")
        );
        assert_eq!(
            error.metadata.get("configuredKernel").map(String::as_str),
            Some("hnsw-rs")
        );
        let snapshot_dir = error
            .metadata
            .get("snapshotDir")
            .ok_or_else(|| std::io::Error::other("expected snapshotDir metadata"))?;
        assert!(
            snapshot_dir.contains("kernel_mismatch.v2"),
            "unexpected snapshotDir metadata: {snapshot_dir}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn v2_load_force_reindex_rewrites_kernel_metadata() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-kernel-force-reindex-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let collection = CollectionName::parse("kernel_force_reindex")?;
        let ctx = RequestContext::new_request();

        // Use V1 format to persist float32 vectors in v1 JSON.  The
        // force_reindex rebuild path requires those vectors to re-build the HNSW
        // graph; in pure V2 mode vectors are only stored in quantized form.
        let db_v1 = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .snapshot_format(VectorSnapshotFormat::V1)
        .build()?;
        db_v1
            .create_collection(&ctx, collection.clone(), 3, None)
            .await?;
        db_v1
            .insert(
                &ctx,
                collection.clone(),
                vec![VectorDocumentForInsert {
                    id: "doc1".into(),
                    vector: Arc::from(vec![0.1, 0.2, 0.3]),
                    content: "hello".into(),
                    metadata: sample_metadata("src/lib.rs")?,
                }],
            )
            .await?;
        db_v1.flush(&ctx, collection.clone()).await?;
        let v1_paths = db_v1.snapshot_paths(&collection).ok_or_else(|| {
            std::io::Error::other("expected snapshot paths for custom storage mode")
        })?;
        let _ = std::fs::remove_file(v1_paths.generation_layout.active_file());
        drop(db_v1);
        // Open as V2 and search to build the V2 companion files.
        let db_v2_init = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .build()?;
        let _ = db_v2_init
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection.clone(),
                    query_vector: Arc::from(vec![0.1, 0.2, 0.3]),
                    options: VectorSearchOptions {
                        top_k: Some(1),
                        filter_expr: None,
                        threshold: None,
                    },
                },
            )
            .await?;
        let paths = db_v2_init.snapshot_paths(&collection).ok_or_else(|| {
            std::io::Error::other("expected snapshot paths for custom storage mode")
        })?;
        drop(db_v2_init);
        let original_meta = read_metadata(paths.v2_meta.as_path()).map_err(|error| {
            ErrorEnvelope::expected(
                ErrorCode::new("vector", "snapshot_invalid"),
                error.to_string(),
            )
        })?;
        let mut meta = original_meta.clone();
        meta.kernel = VectorKernelKind::Dfrr;
        semantic_code_vector::write_metadata(paths.v2_meta.as_path(), &meta).map_err(|error| {
            ErrorEnvelope::expected(
                ErrorCode::new("vector", "snapshot_invalid"),
                error.to_string(),
            )
        })?;
        let _ = std::fs::remove_file(paths.generation_layout.active_file());

        let restarted = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .force_reindex_on_kernel_change(true)
        .build()?;
        let response = restarted
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection.clone(),
                    query_vector: Arc::from(vec![0.1, 0.2, 0.3]),
                    options: VectorSearchOptions {
                        top_k: Some(1),
                        filter_expr: None,
                        threshold: None,
                    },
                },
            )
            .await?;
        assert_eq!(response.results.len(), 1);
        assert!(response.stats.is_some());

        let rewritten_meta = read_metadata(paths.v2_meta.as_path()).map_err(|error| {
            ErrorEnvelope::expected(
                ErrorCode::new("vector", "snapshot_invalid"),
                error.to_string(),
            )
        })?;
        assert_eq!(rewritten_meta.kernel, VectorKernelKind::HnswRs);
        assert_eq!(rewritten_meta.version, original_meta.version);
        assert_eq!(rewritten_meta.dimension, original_meta.dimension);
        assert_eq!(rewritten_meta.count, original_meta.count);
        assert_eq!(rewritten_meta.params, original_meta.params);
        Ok(())
    }

    #[tokio::test]
    async fn v2_load_tolerates_dfrr_hnsw_snapshot_mismatch_and_warms_before_search() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-dfrr-tolerant-load-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let collection = CollectionName::parse("dfrr_tolerant_load")?;
        let ctx = RequestContext::new_request();

        let seed = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .build()?;
        seed.create_collection(&ctx, collection.clone(), 3, None)
            .await?;
        seed.insert(
            &ctx,
            collection.clone(),
            vec![VectorDocumentForInsert {
                id: "doc1".into(),
                vector: Arc::from(vec![0.1, 0.2, 0.3]),
                content: "hello".into(),
                metadata: sample_metadata("src/lib.rs")?,
            }],
        )
        .await?;
        seed.flush(&ctx, collection.clone()).await?;
        let paths = seed.snapshot_paths(&collection).ok_or_else(|| {
            std::io::Error::other("expected snapshot paths for tolerant-load test")
        })?;
        let _active_generation = paths
            .generation_layout
            .read_active_generation_id()?
            .ok_or_else(|| std::io::Error::other("expected active generation after flush"))?;
        let _ = std::fs::remove_file(paths.generation_layout.active_file());
        drop(seed);

        let kernel = Arc::new(TestDfrrKernel::default());
        let restarted =
            LocalVectorDbBuilder::new(tmp.clone(), kernel.clone(), CancellationToken::new())
                .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
                .build()?;

        restarted.ensure_loaded(&collection).await?;
        assert_eq!(
            kernel.warm_calls(),
            1,
            "DFRR kernel should be warmed during load before first search"
        );
        assert_eq!(
            kernel.search_calls(),
            0,
            "load-time warm should happen before any explicit search"
        );

        let rewritten_meta = read_metadata(paths.v2_meta.as_path()).map_err(|error| {
            ErrorEnvelope::expected(
                ErrorCode::new("vector", "snapshot_invalid"),
                error.to_string(),
            )
        })?;
        assert_eq!(
            rewritten_meta.kernel,
            VectorKernelKind::Dfrr,
            "tolerant DFRR loads should rewrite only the runtime kernel metadata"
        );
        let configured_snapshot_dir = kernel.snapshot_dir().ok_or_else(|| {
            std::io::Error::other("expected DFRR test kernel to receive a snapshot dir")
        })?;
        assert_eq!(
            configured_snapshot_dir,
            paths.v2_dir.clone(),
            "legacy tolerant-load path should continue using the v2 root when ACTIVE is intentionally removed"
        );

        let response = restarted
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection.clone(),
                    query_vector: Arc::from(vec![0.1, 0.2, 0.3]),
                    options: VectorSearchOptions {
                        top_k: Some(1),
                        filter_expr: None,
                        threshold: None,
                    },
                },
            )
            .await?;
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].document.id, "doc1".into());
        assert_eq!(
            kernel.warm_calls(),
            1,
            "first search should not trigger a hidden extra warmup"
        );
        assert_eq!(kernel.search_calls(), 1);
        Ok(())
    }

    #[test]
    fn resolve_kernel_snapshot_dir_prefers_generation_scoped_kernel_roots() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-kernel-snapshot-dir-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let collection = CollectionName::parse("kernel_snapshot_dir")?;
        let loader = CollectionLoaderContext {
            codebase_root: tmp.clone(),
            storage_mode: SnapshotStorageMode::Custom(tmp),
            snapshot_format: VectorSnapshotFormat::V2,
            snapshot_max_bytes: None,
            kernel: Arc::new(TestDfrrKernel::default()),
            runtime_dfrr_ready_state: None,
            dfrr_prewarm_requests: Vec::new(),
            force_reindex_on_kernel_change: false,
            search_backend: VectorSearchBackend::F32Hnsw,
            hnsw_params: HnswParams::default(),
        };
        let paths = loader.snapshot_paths(&collection).ok_or_else(|| {
            std::io::Error::other("expected snapshot paths for custom storage mode")
        })?;

        let legacy = resolve_kernel_snapshot_dir(&loader, Some(&paths))?;
        assert_eq!(legacy, Some(paths.v2_dir.clone()));

        std::fs::create_dir_all(paths.generation_layout.root()).map_err(ErrorEnvelope::from)?;
        std::fs::write(paths.generation_layout.active_file(), "gen-dfrr")
            .map_err(ErrorEnvelope::from)?;
        let resolved = resolve_kernel_snapshot_dir(&loader, Some(&paths))?;
        assert_eq!(
            resolved,
            Some(
                paths
                    .generation_layout
                    .generation(&semantic_code_vector::GenerationId::new("gen-dfrr")?)
                    .kernels_dir()
                    .to_path_buf()
            )
        );
        Ok(())
    }

    #[tokio::test]
    async fn legacy_v1_dfrr_load_uses_records_only_host_index() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-dfrr-v1-records-only-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let collection = CollectionName::parse("dfrr_v1_records_only")?;
        let ctx = RequestContext::new_request();

        let seed = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .snapshot_format(VectorSnapshotFormat::V1)
        .build()?;
        seed.create_collection(&ctx, collection.clone(), 3, None)
            .await?;
        seed.insert(
            &ctx,
            collection.clone(),
            vec![VectorDocumentForInsert {
                id: "doc1".into(),
                vector: Arc::from(vec![0.1, 0.2, 0.3]),
                content: "hello".into(),
                metadata: sample_metadata("src/lib.rs")?,
            }],
        )
        .await?;
        seed.flush(&ctx, collection.clone()).await?;
        drop(seed);

        let kernel = Arc::new(TestDfrrKernel::default());
        let restarted =
            LocalVectorDbBuilder::new(tmp.clone(), kernel.clone(), CancellationToken::new())
                .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
                .snapshot_format(VectorSnapshotFormat::V1)
                .build()?;
        restarted.ensure_loaded(&collection).await?;

        let guard = restarted.collections.read().await;
        let loaded = guard
            .get(&collection)
            .ok_or_else(|| std::io::Error::other("expected loaded collection"))?;
        let index = loaded.read_index()?;
        let host_hnsw_count = index.host_hnsw_count();
        drop(index);
        drop(guard);
        assert!(
            host_hnsw_count == 0,
            "legacy DFRR load should keep the host HNSW graph empty and rely on records-only load"
        );

        let response = restarted
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection.clone(),
                    query_vector: Arc::from(vec![0.1, 0.2, 0.3]),
                    options: VectorSearchOptions {
                        top_k: Some(1),
                        filter_expr: None,
                        threshold: None,
                    },
                },
            )
            .await?;
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].document.id, "doc1".into());
        assert_eq!(kernel.warm_calls(), 1);
        assert_eq!(kernel.search_calls(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn warm_collection_kernel_state_propagates_cancellation() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-dfrr-warm-cancel-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let collection_name = CollectionName::parse("dfrr_warm_cancel")?;
        let kernel = Arc::new(TestDfrrKernel::default());
        let loader = CollectionLoaderContext {
            codebase_root: tmp.clone(),
            storage_mode: SnapshotStorageMode::Custom(tmp.clone()),
            snapshot_format: VectorSnapshotFormat::V2,
            snapshot_max_bytes: None,
            kernel,
            runtime_dfrr_ready_state: None,
            dfrr_prewarm_requests: Vec::new(),
            force_reindex_on_kernel_change: false,
            search_backend: VectorSearchBackend::F32Hnsw,
            hnsw_params: HnswParams::default(),
        };
        let collection = LocalCollection::new(3, IndexMode::Dense, HnswParams::default())?;
        let cancellation = CancellationToken::new();
        cancellation.cancel();

        let error = warm_collection_kernel_state(
            &loader,
            &collection_name,
            Arc::clone(&collection.index),
            None,
            true,
            Some(cancellation),
        )
        .await
        .err()
        .ok_or_else(|| std::io::Error::other("expected warm cancellation error"))?;
        assert!(
            error.is_cancelled(),
            "expected cancellation error, got: {error}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn flush_prewarms_dfrr_ready_state_and_records_catalog_entry() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-dfrr-prewarm-flush-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let collection = CollectionName::parse("dfrr_prewarm_flush")?;
        let ctx = RequestContext::new_request();
        let prewarm_kernel = Arc::new(TestDfrrKernel::default());
        let requirement = DfrrReadyStateRequirement {
            ready_state_fingerprint: "dfrr-fingerprint-a".into(),
            config_json: "{\"efSearch\":64,\"clusterCount\":8}".into(),
        };
        let db = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .dfrr_prewarm_requests(vec![DfrrReadyStatePrewarmRequest {
            requirement: requirement.clone(),
            kernel: prewarm_kernel.clone(),
        }])
        .build()?;

        db.create_collection(&ctx, collection.clone(), 3, None)
            .await?;
        db.insert(
            &ctx,
            collection.clone(),
            vec![VectorDocumentForInsert {
                id: "doc1".into(),
                vector: Arc::from(vec![0.1, 0.2, 0.3]),
                content: "hello".into(),
                metadata: sample_metadata("src/lib.rs")?,
            }],
        )
        .await?;
        db.flush(&ctx, collection.clone()).await?;

        assert_eq!(prewarm_kernel.warm_calls(), 1);
        let paths = db.snapshot_paths(&collection).ok_or_else(|| {
            std::io::Error::other("expected snapshot paths after DFRR prewarm flush")
        })?;
        let active_generation = paths
            .generation_layout
            .read_active_generation_id()?
            .ok_or_else(|| std::io::Error::other("expected active generation after flush"))?;
        assert!(has_ready_dfrr_state(
            &collection,
            &paths.generation_layout,
            &active_generation,
            requirement.ready_state_fingerprint.as_ref(),
        )?);
        Ok(())
    }

    #[tokio::test]
    async fn load_fails_when_runtime_dfrr_ready_state_is_missing() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-dfrr-missing-ready-state-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let collection = CollectionName::parse("dfrr_missing_ready_state")?;
        let ctx = RequestContext::new_request();

        let seed = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .build()?;
        seed.create_collection(&ctx, collection.clone(), 3, None)
            .await?;
        seed.insert(
            &ctx,
            collection.clone(),
            vec![VectorDocumentForInsert {
                id: "doc1".into(),
                vector: Arc::from(vec![0.1, 0.2, 0.3]),
                content: "hello".into(),
                metadata: sample_metadata("src/lib.rs")?,
            }],
        )
        .await?;
        seed.flush(&ctx, collection.clone()).await?;
        drop(seed);

        let kernel = Arc::new(TestDfrrKernel::default());
        let restarted =
            LocalVectorDbBuilder::new(tmp.clone(), kernel.clone(), CancellationToken::new())
                .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
                .runtime_dfrr_ready_state(DfrrReadyStateRequirement {
                    ready_state_fingerprint: "missing-dfrr-ready-state".into(),
                    config_json: "{\"efSearch\":64,\"clusterCount\":8}".into(),
                })
                .build()?;

        let error = restarted
            .ensure_loaded(&collection)
            .await
            .err()
            .ok_or_else(|| std::io::Error::other("expected DFRR ready-state load failure"))?;
        assert_eq!(error.code.to_string(), "vector:dfrr_ready_state_missing");
        assert_eq!(kernel.warm_calls(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn load_succeeds_when_runtime_dfrr_ready_state_was_prewarmed() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-dfrr-ready-state-reuse-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let collection = CollectionName::parse("dfrr_ready_state_reuse")?;
        let ctx = RequestContext::new_request();
        let requirement = DfrrReadyStateRequirement {
            ready_state_fingerprint: "dfrr-ready-state-reuse".into(),
            config_json: "{\"efSearch\":64,\"clusterCount\":8}".into(),
        };
        let seed_prewarm_kernel = Arc::new(TestDfrrKernel::default());
        let seed = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .dfrr_prewarm_requests(vec![DfrrReadyStatePrewarmRequest {
            requirement: requirement.clone(),
            kernel: seed_prewarm_kernel.clone(),
        }])
        .build()?;
        seed.create_collection(&ctx, collection.clone(), 3, None)
            .await?;
        seed.insert(
            &ctx,
            collection.clone(),
            vec![VectorDocumentForInsert {
                id: "doc1".into(),
                vector: Arc::from(vec![0.1, 0.2, 0.3]),
                content: "hello".into(),
                metadata: sample_metadata("src/lib.rs")?,
            }],
        )
        .await?;
        seed.flush(&ctx, collection.clone()).await?;
        assert_eq!(seed_prewarm_kernel.warm_calls(), 1);
        drop(seed);

        let runtime_kernel = Arc::new(TestDfrrKernel::default());
        let restarted = LocalVectorDbBuilder::new(
            tmp.clone(),
            runtime_kernel.clone(),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .runtime_dfrr_ready_state(requirement)
        .build()?;

        restarted.ensure_loaded(&collection).await?;
        assert_eq!(runtime_kernel.warm_calls(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn concurrent_searches_on_loaded_v2_collection_return_stable_results() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-v2-concurrent-search-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let collection = CollectionName::parse("local_snapshot_concurrent_search")?;
        let ctx = RequestContext::new_request();
        let db = LocalVectorDbBuilder::new(
            tmp.clone(),
            Arc::new(HnswKernel::new()),
            CancellationToken::new(),
        )
        .storage_mode(SnapshotStorageMode::Custom(tmp.clone()))
        .build()?;

        db.create_collection(&ctx, collection.clone(), 3, None)
            .await?;
        db.insert(
            &ctx,
            collection.clone(),
            vec![
                VectorDocumentForInsert {
                    id: "doc_a".into(),
                    vector: Arc::from(vec![1.0, 0.0, 0.0]),
                    content: "document a".into(),
                    metadata: sample_metadata("src/a.rs")?,
                },
                VectorDocumentForInsert {
                    id: "doc_b".into(),
                    vector: Arc::from(vec![0.0, 1.0, 0.0]),
                    content: "document b".into(),
                    metadata: sample_metadata("src/b.rs")?,
                },
                VectorDocumentForInsert {
                    id: "doc_c".into(),
                    vector: Arc::from(vec![0.0, 0.0, 1.0]),
                    content: "document c".into(),
                    metadata: sample_metadata("src/c.rs")?,
                },
            ],
        )
        .await?;
        db.flush(&ctx, collection.clone()).await?;

        let db = Arc::new(db);

        let baseline = db
            .search(
                &ctx,
                VectorSearchRequest {
                    collection_name: collection.clone(),
                    query_vector: Arc::from(vec![1.0, 0.0, 0.0]),
                    options: VectorSearchOptions {
                        top_k: Some(1),
                        filter_expr: None,
                        threshold: None,
                    },
                },
            )
            .await?;
        assert_eq!(baseline.results[0].document.id, "doc_a".into());

        let mut handles = Vec::with_capacity(16);
        for _ in 0..16 {
            let collection = collection.clone();
            let db = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                db.search(
                    &RequestContext::new_request(),
                    VectorSearchRequest {
                        collection_name: collection.clone(),
                        query_vector: Arc::from(vec![1.0, 0.0, 0.0]),
                        options: VectorSearchOptions {
                            top_k: Some(1),
                            filter_expr: None,
                            threshold: None,
                        },
                    },
                )
                .await
            }));
        }

        for handle in handles {
            let response = handle.await.map_err(|error| {
                ErrorEnvelope::unexpected(
                    ErrorCode::new("vector", "search_task_failed"),
                    error.to_string(),
                    ErrorClass::NonRetriable,
                )
            })??;
            assert_eq!(response.results.len(), 1);
            assert_eq!(response.results[0].document.id, "doc_a".into());
        }

        Ok(())
    }

    #[tokio::test]
    async fn loader_actor_load_evict_shutdown() {
        use semantic_code_vector::HnswKernel;

        // Minimal LoaderContext pointing at a temp dir with NO snapshot files —
        // load_collection will fail because there is nothing on disk.  We are
        // testing the actor lifecycle (commands flow, shutdown is clean), not
        // the loading logic.
        let tmp = std::env::temp_dir().join(format!(
            "sca-actor-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let loader = CollectionLoaderContext {
            codebase_root: tmp.clone(),
            storage_mode: SnapshotStorageMode::Custom(tmp.clone()),
            snapshot_format: VectorSnapshotFormat::V1,
            snapshot_max_bytes: None,
            kernel: Arc::new(HnswKernel::new()),
            runtime_dfrr_ready_state: None,
            dfrr_prewarm_requests: Vec::new(),
            force_reindex_on_kernel_change: false,
            search_backend: VectorSearchBackend::F32Hnsw,
            hnsw_params: HnswParams::default(),
        };
        let collections: Arc<RwLock<HashMap<CollectionName, LocalCollection>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let checkpoint_states: Arc<
            RwLock<HashMap<CollectionName, Arc<CollectionCheckpointState>>>,
        > = Arc::new(RwLock::new(HashMap::new()));
        let cancellation = CancellationToken::new();

        let (handle, join) = CollectionLoaderActor::spawn(
            loader,
            collections.clone(),
            checkpoint_states,
            cancellation.clone(),
        );

        let name = CollectionName::parse("test_actor").expect("valid name in test");

        // Load will fail because there is no snapshot on disk — that is OK.
        let _ = handle.load(name.clone()).await;

        handle
            .evict(name.clone())
            .await
            .expect("evict should succeed");
        assert!(
            !collections.read().await.contains_key(&name),
            "collection should be absent after evict"
        );

        cancellation.cancel();
        join.await.expect("actor should not panic");
    }

    #[test]
    fn local_collection_v1_snapshot_and_sidecar_use_active_origin_order() -> Result<()> {
        let mut collection = LocalCollection::new(3, IndexMode::Dense, HnswParams::default())?;
        collection.insert(vec![
            VectorDocumentForInsert {
                id: "gamma".into(),
                vector: Arc::from(vec![0.0, 0.0, 1.0]),
                content: "gamma".into(),
                metadata: sample_metadata("src/gamma.rs")?,
            },
            VectorDocumentForInsert {
                id: "alpha".into(),
                vector: Arc::from(vec![1.0, 0.0, 0.0]),
                content: "alpha-old".into(),
                metadata: sample_metadata("src/alpha_old.rs")?,
            },
            VectorDocumentForInsert {
                id: "beta".into(),
                vector: Arc::from(vec![0.0, 1.0, 0.0]),
                content: "beta".into(),
                metadata: sample_metadata("src/beta.rs")?,
            },
        ])?;
        collection.insert(vec![VectorDocumentForInsert {
            id: "alpha".into(),
            vector: Arc::from(vec![0.9, 0.1, 0.0]),
            content: "alpha-new".into(),
            metadata: sample_metadata("src/alpha_new.rs")?,
        }])?;

        let snapshot = collection.snapshot()?;
        let snapshot_ids = snapshot
            .records
            .iter()
            .map(|record| record.id.as_ref())
            .collect::<Vec<_>>();
        assert_eq!(
            snapshot_ids,
            vec!["gamma", "beta", "alpha"],
            "legacy v1 snapshots must preserve canonical active-origin ordering"
        );

        let temp_root = std::env::temp_dir().join(format!(
            "sca-local-collection-snapshot-order-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        std::fs::create_dir_all(&temp_root).map_err(ErrorEnvelope::from)?;
        let generation_layout = CollectionGenerationPaths::new(temp_root.join("generation"));
        let build_dir = generation_layout.root().join(LOCAL_BUILD_JOURNAL_DIR_NAME);
        let paths = CollectionSnapshotPaths {
            v1_json: temp_root.join("snapshot.v1.json"),
            v2_dir: temp_root.clone(),
            v2_meta: temp_root.join("snapshot.meta"),
            v2_vectors: temp_root.join("vectors.u8.bin"),
            v2_ids: temp_root.join("ids.json"),
            v2_records_meta: temp_root.join("records.meta.jsonl"),
            insert_wal: temp_root.join("insert.wal.jsonl"),
            generation_layout,
            build_meta: build_dir.join(LOCAL_BUILD_JOURNAL_META_FILE_NAME),
            build_rows: build_dir.join(LOCAL_BUILD_JOURNAL_ROWS_FILE_NAME),
            build_vectors: build_dir.join(LOCAL_BUILD_JOURNAL_VECTORS_FILE_NAME),
            build_sealed: build_dir.join(LOCAL_BUILD_JOURNAL_SEALED_FILE_NAME),
        };

        write_records_meta_sidecar(&paths, &snapshot)?;
        let payload =
            std::fs::read_to_string(&paths.v2_records_meta).map_err(ErrorEnvelope::from)?;
        let sidecar_ids = payload
            .lines()
            .skip(1)
            .map(|line| {
                serde_json::from_str::<RecordMetadataEntry>(line)
                    .map(|entry| entry.id)
                    .map_err(|error| {
                        ErrorEnvelope::expected(
                            ErrorCode::new("vector", "sidecar_parse_failed"),
                            error.to_string(),
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let sidecar_ids = sidecar_ids.iter().map(AsRef::as_ref).collect::<Vec<_>>();
        assert_eq!(
            sidecar_ids, snapshot_ids,
            "v1 auto-migration sidecar must preserve the same canonical record order"
        );

        let _ = std::fs::remove_dir_all(&temp_root);
        Ok(())
    }

    #[test]
    fn read_records_meta_sidecar_rejects_duplicate_ids_with_line_numbers() -> Result<()> {
        let temp_root = std::env::temp_dir().join(format!(
            "sca-sidecar-duplicate-id-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        std::fs::create_dir_all(&temp_root).map_err(ErrorEnvelope::from)?;
        let path = temp_root.join("records.meta.jsonl");

        let header = RecordMetadataHeader {
            version: LOCAL_SNAPSHOT_VERSION,
            dimension: 3,
            index_mode: IndexMode::Dense,
            count: 2,
            checkpoint_sequence: None,
        };
        let payload = serialize_records_meta_sidecar(
            &header,
            [
                RecordMetadataEntry {
                    id: "chunk_duplicate".into(),
                    content: "alpha".into(),
                    metadata: sample_metadata("src/a.rs")?,
                },
                RecordMetadataEntry {
                    id: "chunk_duplicate".into(),
                    content: "beta".into(),
                    metadata: sample_metadata("src/b.rs")?,
                },
            ],
        )?;
        write_records_meta_payload(&path, payload.as_slice())?;

        let error = match read_records_meta_sidecar(&path) {
            Ok(_) => {
                return Err(ErrorEnvelope::invariant(
                    ErrorCode::new("vector", "duplicate_sidecar_test_failed"),
                    "duplicate ids must fail",
                ));
            },
            Err(error) => error,
        };
        assert_eq!(
            error.code,
            ErrorCode::new("vector", "duplicate_record_id_in_sidecar")
        );
        assert_eq!(
            error.metadata.get("id").map(String::as_str),
            Some("chunk_duplicate")
        );
        assert_eq!(
            error.metadata.get("first_line").map(String::as_str),
            Some("2")
        );
        assert_eq!(
            error.metadata.get("duplicate_line").map(String::as_str),
            Some("3")
        );

        let _ = std::fs::remove_dir_all(&temp_root);
        Ok(())
    }
}
