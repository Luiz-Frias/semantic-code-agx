//! Local vector database adapter backed by HNSW.

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
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use semantic_code_vector::{
    HnswParams, SNAPSHOT_V2_META_FILE_NAME, SNAPSHOT_V2_VECTORS_FILE_NAME, SnapshotStats,
    VectorIndex, VectorKernel, VectorKernelKind, VectorRecord, VectorSearchBackend,
    VectorSnapshotWriteVersion, read_metadata,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::AsyncWriteExt;
use tokio::sync::{Mutex, Notify, RwLock};
use tokio::task::{JoinHandle, spawn_blocking};
use tracing::Instrument;

const LOCAL_SNAPSHOT_VERSION: u32 = 1;
const LOCAL_SNAPSHOT_DIR: &str = "vector";
const LOCAL_COLLECTIONS_DIR: &str = "collections";
const LOCAL_SNAPSHOT_V2_DIR_SUFFIX: &str = ".v2";
const LOCAL_SNAPSHOT_V2_IDS_FILE_NAME: &str = "ids.json";
const LOCAL_SNAPSHOT_V2_RECORDS_META_FILE_NAME: &str = "records.meta.jsonl";
const LOCAL_INSERT_WAL_FILE_SUFFIX: &str = ".wal.jsonl";

#[derive(Debug, Clone)]
struct CollectionSnapshotPaths {
    v1_json: PathBuf,
    v2_dir: PathBuf,
    v2_meta: PathBuf,
    v2_vectors: PathBuf,
    v2_ids: PathBuf,
    v2_records_meta: PathBuf,
    insert_wal: PathBuf,
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
}

impl CollectionCheckpointState {
    fn new() -> Self {
        Self {
            progress: Mutex::new(CollectionCheckpointProgress::default()),
            wal_io: Mutex::new(()),
            notify: Notify::new(),
        }
    }
}

#[derive(Default)]
struct CollectionCheckpointProgress {
    scheduled_sequence: u64,
    durable_sequence: u64,
    last_error: Option<ErrorEnvelope>,
    worker: Option<JoinHandle<()>>,
}

/// Local vector DB backed by an HNSW index.
pub struct LocalVectorDb {
    provider: VectorDbProviderInfo,
    codebase_root: PathBuf,
    storage_mode: SnapshotStorageMode,
    snapshot_format: VectorSnapshotFormat,
    snapshot_max_bytes: Option<u64>,
    kernel: Arc<dyn VectorKernel + Send + Sync>,
    force_reindex_on_kernel_change: bool,
    search_backend: VectorSearchBackend,
    collections: Arc<RwLock<HashMap<CollectionName, LocalCollection>>>,
    checkpoint_states: Arc<RwLock<HashMap<CollectionName, Arc<CollectionCheckpointState>>>>,
    #[cfg(test)]
    checkpoint_delay_ms: Arc<AtomicU64>,
}

impl LocalVectorDb {
    /// Create a local vector DB adapter scoped to a codebase root.
    ///
    /// The `kernel` is a pre-built concrete kernel (created by the factory).
    pub fn new(
        codebase_root: PathBuf,
        storage_mode: SnapshotStorageMode,
        snapshot_format: VectorSnapshotFormat,
        snapshot_max_bytes: Option<u64>,
        kernel: Arc<dyn VectorKernel + Send + Sync>,
        force_reindex_on_kernel_change: bool,
        search_strategy: VectorSearchStrategy,
    ) -> Result<Self> {
        let provider = VectorDbProviderInfo {
            id: VectorDbProviderId::parse("local").map_err(ErrorEnvelope::from)?,
            name: "Local".into(),
        };
        Ok(Self {
            provider,
            codebase_root,
            storage_mode,
            snapshot_format,
            snapshot_max_bytes,
            kernel,
            force_reindex_on_kernel_change,
            search_backend: resolve_search_backend(search_strategy),
            collections: Arc::new(RwLock::new(HashMap::new())),
            checkpoint_states: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(test)]
            checkpoint_delay_ms: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Return whether a local vector kernel is available in this build.
    pub const fn is_kernel_supported(kernel: ConfigVectorKernelKind) -> bool {
        match kernel {
            ConfigVectorKernelKind::Dfrr => cfg!(feature = "experimental-dfrr-kernel"),
            ConfigVectorKernelKind::HnswRs | ConfigVectorKernelKind::FlatScan => true,
        }
    }

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
        Some(CollectionSnapshotPaths {
            v1_json,
            v2_dir,
            v2_meta,
            v2_vectors,
            v2_ids,
            v2_records_meta,
            insert_wal,
        })
    }

    async fn ensure_loaded(&self, collection_name: &CollectionName) -> Result<()> {
        {
            let collections = self.collections.read().await;
            if collections.contains_key(collection_name) {
                return Ok(());
            }
        }

        let (mut collection, checkpoint_sequence) = match self.snapshot_format {
            VectorSnapshotFormat::V2 => {
                match self.try_load_v2_with_sidecar(collection_name).await? {
                    Some(result) => result,
                    None => {
                        // Sidecar missing — fall back to v1 JSON + v2 index (legacy path).
                        self.load_via_v1_json(collection_name).await?
                    },
                }
            },
            VectorSnapshotFormat::V1 => self.load_via_v1_json(collection_name).await?,
        };

        self.replay_collection_from_wal(collection_name, &mut collection, checkpoint_sequence)
            .await?;
        self.collections
            .write()
            .await
            .entry(collection_name.clone())
            .or_insert(collection);
        self.set_checkpoint_durable_hint(collection_name, checkpoint_sequence)
            .await;
        Ok(())
    }

    /// v2 fast path: load the index from the v2 binary bundle and metadata
    /// from the JSONL sidecar.  Skips the v1 JSON entirely.
    ///
    /// Returns `None` if the sidecar doesn't exist (caller should fall back).
    async fn try_load_v2_with_sidecar(
        &self,
        collection_name: &CollectionName,
    ) -> Result<Option<(LocalCollection, u64)>> {
        let Some(paths) = self.snapshot_paths(collection_name) else {
            return Ok(None);
        };
        if !path_exists(&paths.v2_records_meta).await? {
            return Ok(None);
        }
        // Verify the v2 binary companion files are present.
        match self
            .detect_v2_companion_state(collection_name, &paths)
            .await?
        {
            V2CompanionState::Missing => return Ok(None),
            V2CompanionState::Present => {},
        }

        let snapshot_kernel = self
            .read_v2_kernel_metadata(collection_name, paths.v2_meta.clone())
            .await?;
        let is_graph_agnostic = matches!(self.kernel.kind(), VectorKernelKind::FlatScan);
        if snapshot_kernel != self.kernel.kind() && !is_graph_agnostic {
            if self.force_reindex_on_kernel_change {
                // Need v1 snapshot for rebuild — fall back to legacy path.
                tracing::info!(
                    collection = %collection_name,
                    snapshot_kernel = ?snapshot_kernel,
                    requested_kernel = ?self.kernel.kind(),
                    "kernel mismatch with v2 sidecar; falling back to v1 JSON for rebuild"
                );
                return Ok(None);
            }
            return Err(snapshot_kernel_mismatch_error(
                collection_name,
                &paths.v2_dir,
                snapshot_kernel,
                self.kernel.kind(),
            ));
        }

        let index = match self
            .load_index_from_v2(collection_name, paths.v2_dir.clone())
            .await
        {
            Ok(index) => index,
            Err(error) if is_v2_companion_repairable_error(&error) => {
                tracing::warn!(
                    collection = %collection_name,
                    error_code = %error.code,
                    "v2 companion load failed; falling back to v1 JSON"
                );
                return Ok(None);
            },
            Err(error) => return Err(error),
        };

        let sidecar_path = paths.v2_records_meta.clone();
        let collection = spawn_blocking(move || {
            LocalCollection::from_v2_index_and_sidecar(index, &sidecar_path)
        })
        .await
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "sidecar_load_task_failed"),
                "metadata sidecar load task failed",
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata("source", error.to_string())
        })??;

        let checkpoint_sequence = collection.last_insert_sequence;
        tracing::info!(
            collection = %collection_name,
            documents = collection.documents.len(),
            "loaded collection from v2 bundle + metadata sidecar (skipped v1 JSON)"
        );
        Ok(Some((collection, checkpoint_sequence)))
    }

    /// Legacy load path: read v1 JSON, then optionally use v2 index.
    ///
    /// When running in v2 mode and the metadata sidecar is missing, this also
    /// writes the sidecar as a one-time migration so subsequent loads skip the
    /// v1 JSON.
    async fn load_via_v1_json(
        &self,
        collection_name: &CollectionName,
    ) -> Result<(LocalCollection, u64)> {
        let snapshot = self.read_snapshot_json(collection_name).await?;
        let Some(snapshot) = snapshot else {
            let collection = LocalCollection::new(0, IndexMode::Dense)?;
            return Ok((collection, 0));
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

        let collection = match self.snapshot_format {
            VectorSnapshotFormat::V1 => LocalCollection::from_snapshot(snapshot)?,
            VectorSnapshotFormat::V2 => self.load_collection_v2(collection_name, snapshot).await?,
        };
        Ok((collection, checkpoint_sequence))
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

    async fn load_collection_v2(
        &self,
        collection_name: &CollectionName,
        snapshot: CollectionSnapshot,
    ) -> Result<LocalCollection> {
        let Some(paths) = self.snapshot_paths(collection_name) else {
            return LocalCollection::from_snapshot(snapshot);
        };

        match self
            .detect_v2_companion_state(collection_name, &paths)
            .await?
        {
            V2CompanionState::Present => {
                let snapshot_kernel = self
                    .read_v2_kernel_metadata(collection_name, paths.v2_meta.clone())
                    .await?;
                // TODO(bench): Make kernel mismatch check composable — e.g. a
                // whitelist of "graph-agnostic" kernels that can load any snapshot
                // without requiring a kernel match (flat-scan, future brute-force
                // variants). Currently hard-coded to skip for FlatScan only.
                let is_graph_agnostic = matches!(self.kernel.kind(), VectorKernelKind::FlatScan);
                if snapshot_kernel != self.kernel.kind() && !is_graph_agnostic {
                    if self.force_reindex_on_kernel_change {
                        return self.rebuild_collection_with_v2_bundle(
                            collection_name,
                            &paths,
                            snapshot,
                        );
                    }
                    return Err(snapshot_kernel_mismatch_error(
                        collection_name,
                        &paths.v2_dir,
                        snapshot_kernel,
                        self.kernel.kind(),
                    ));
                }
                let index = match self
                    .load_index_from_v2(collection_name, paths.v2_dir.clone())
                    .await
                {
                    Ok(index) => index,
                    Err(error) if is_v2_companion_repairable_error(&error) => {
                        tracing::warn!(
                            collection = %collection_name,
                            error_code = %error.code,
                            "v2 companion load failed; rebuilding from v1 snapshot"
                        );
                        return self.rebuild_collection_with_v2_bundle(
                            collection_name,
                            &paths,
                            snapshot,
                        );
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
                    return self.rebuild_collection_with_v2_bundle(
                        collection_name,
                        &paths,
                        snapshot,
                    );
                }
                LocalCollection::from_snapshot_with_index(snapshot, index)
            },
            V2CompanionState::Missing => {
                self.rebuild_collection_with_v2_bundle(collection_name, &paths, snapshot)
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
            .map_err(|error| {
                ErrorEnvelope::unexpected(
                    ErrorCode::new("vector", "snapshot_load_task_failed"),
                    "snapshot v2 metadata task failed",
                    ErrorClass::NonRetriable,
                )
                .with_metadata("collection", collection_name.as_str().to_string())
                .with_metadata("path", meta_path.display().to_string())
                .with_metadata("source", error.to_string())
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

    async fn load_index_from_v2(
        &self,
        collection_name: &CollectionName,
        snapshot_dir: PathBuf,
    ) -> Result<VectorIndex> {
        let snapshot_dir_for_task = snapshot_dir.clone();
        spawn_blocking(move || VectorIndex::from_snapshot_v2(snapshot_dir_for_task))
            .await
            .map_err(|error| {
                ErrorEnvelope::unexpected(
                    ErrorCode::new("vector", "snapshot_load_task_failed"),
                    "snapshot v2 load task failed",
                    ErrorClass::NonRetriable,
                )
                .with_metadata("collection", collection_name.as_str().to_string())
                .with_metadata("snapshotDir", snapshot_dir.display().to_string())
                .with_metadata("source", error.to_string())
            })?
    }

    async fn write_snapshot(
        &self,
        collection_name: &CollectionName,
        snapshot: &CollectionSnapshot,
    ) -> Result<()> {
        let Some(paths) = self.snapshot_paths(collection_name) else {
            return Ok(());
        };

        let payload = serialize_snapshot_json(snapshot)?;
        if self.snapshot_format == VectorSnapshotFormat::V1 {
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
                self.snapshot_max_bytes,
            )?;
            log_json_snapshot_stats(collection_name, &paths.v1_json, snapshot, payload_bytes);
        }
        Self::write_snapshot_json(paths.v1_json.as_path(), payload.as_slice()).await?;
        if self.snapshot_format == VectorSnapshotFormat::V2 {
            Self::write_v2_bundle(
                collection_name,
                &paths,
                snapshot,
                self.kernel.kind(),
                self.snapshot_max_bytes,
            )?;
        }
        Ok(())
    }

    fn rebuild_collection_with_v2_bundle(
        &self,
        collection_name: &CollectionName,
        paths: &CollectionSnapshotPaths,
        snapshot: CollectionSnapshot,
    ) -> Result<LocalCollection> {
        let collection = LocalCollection::from_snapshot(snapshot)?;
        let migrated = collection.snapshot();
        Self::write_v2_bundle(
            collection_name,
            paths,
            &migrated,
            self.kernel.kind(),
            self.snapshot_max_bytes,
        )?;
        Ok(collection)
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

    async fn replay_collection_from_wal(
        &self,
        collection_name: &CollectionName,
        collection: &mut LocalCollection,
        checkpoint_sequence: u64,
    ) -> Result<()> {
        let Some(paths) = self.snapshot_paths(collection_name) else {
            return Ok(());
        };
        let checkpoint_state = self.checkpoint_state_for_collection(collection_name).await;
        let _wal_io = checkpoint_state.wal_io.lock().await;
        let records = read_insert_wal_records(paths.insert_wal.as_path()).await?;
        replay_insert_wal_records(
            paths.insert_wal.as_path(),
            collection,
            checkpoint_sequence,
            records.as_slice(),
        )
    }

    async fn set_checkpoint_durable_hint(
        &self,
        collection_name: &CollectionName,
        durable_sequence: u64,
    ) {
        let state = self.checkpoint_state_for_collection(collection_name).await;
        let mut progress = state.progress.lock().await;
        progress.durable_sequence = progress.durable_sequence.max(durable_sequence);
        progress.scheduled_sequence = progress.scheduled_sequence.max(progress.durable_sequence);
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

    async fn schedule_checkpoint(
        &self,
        collection_name: &CollectionName,
        sequence: u64,
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

        if progress.worker.is_none() && progress.scheduled_sequence > progress.durable_sequence {
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
                .write_checkpoint_and_compact_wal(&collection_name, target_sequence)
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
    ) -> Result<u64> {
        let snapshot = {
            let collections = self.collections.read().await;
            let snapshot = collections
                .get(collection_name)
                .map(LocalCollection::snapshot);
            drop(collections);
            let Some(snapshot) = snapshot else {
                return Err(ErrorEnvelope::expected(
                    ErrorCode::not_found(),
                    "collection not found",
                ));
            };
            snapshot
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

        #[cfg(test)]
        self.maybe_delay_checkpoint_for_tests().await;

        self.write_snapshot(collection_name, &snapshot).await?;
        if let Some(paths) = self.snapshot_paths(collection_name) {
            let checkpoint_state = self.checkpoint_state_for_collection(collection_name).await;
            let _wal_io = checkpoint_state.wal_io.lock().await;
            compact_insert_wal_records(paths.insert_wal.as_path(), checkpoint_sequence).await?;
        }
        Ok(checkpoint_sequence)
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
    fn set_checkpoint_delay_for_tests(&self, delay: std::time::Duration) {
        let delay_ms_u128 = delay.as_millis();
        let delay_ms = u64::try_from(delay_ms_u128).unwrap_or(u64::MAX);
        self.checkpoint_delay_ms.store(delay_ms, Ordering::Relaxed);
    }

    #[cfg(test)]
    async fn maybe_delay_checkpoint_for_tests(&self) {
        let delay_ms = self.checkpoint_delay_ms.load(Ordering::Relaxed);
        if delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
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

    fn write_v2_bundle(
        collection_name: &CollectionName,
        paths: &CollectionSnapshotPaths,
        snapshot: &CollectionSnapshot,
        kernel: VectorKernelKind,
        snapshot_max_bytes: Option<u64>,
    ) -> Result<()> {
        let collection = LocalCollection::from_snapshot(snapshot.clone())?;
        let stats = collection
            .index
            .snapshot_stats(VectorSnapshotWriteVersion::V2)
            .map_err(|error| map_snapshot_write_error(error, collection_name, &paths.v2_dir))?;
        enforce_snapshot_limit(
            collection_name,
            &paths.v2_dir,
            VectorSnapshotWriteVersion::V2,
            stats.bytes,
            snapshot_max_bytes,
        )?;
        log_v2_snapshot_stats(collection_name, &paths.v2_dir, &stats);
        collection
            .index
            .snapshot_v2_for_kernel(paths.v2_dir.as_path(), kernel)
            .map(|_| ())
            .map_err(|error| map_snapshot_write_error(error, collection_name, &paths.v2_dir))?;
        write_records_meta_sidecar(paths, snapshot)
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
                let collection = LocalCollection::new(dimension, IndexMode::Dense)?;
                let mut guard = db.collections.write().await;
                guard.insert(collection_name.clone(), collection);
                let snapshot = guard.get(&collection_name).map(LocalCollection::snapshot);
                drop(guard);
                let Some(snapshot) = snapshot else {
                    return Ok(());
                };
                db.write_snapshot(&collection_name, &snapshot).await
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
                let collection = LocalCollection::new(dimension, IndexMode::Hybrid)?;
                let mut guard = db.collections.write().await;
                guard.insert(collection_name.clone(), collection);
                let snapshot = guard.get(&collection_name).map(LocalCollection::snapshot);
                drop(guard);
                let Some(snapshot) = snapshot else {
                    return Ok(());
                };
                db.write_snapshot(&collection_name, &snapshot).await
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

                if let Some(state) = db.remove_checkpoint_state(&collection_name).await {
                    Self::stop_checkpoint_worker_for_drop(&collection_name, state.as_ref()).await?;
                }

                if let Some(paths) = snapshot {
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

                match tokio::fs::metadata(paths.v1_json.as_path()).await {
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
                let mut guard = db.collections.write().await;
                let Some(collection) = guard.get_mut(&collection_name) else {
                    return Err(ErrorEnvelope::expected(
                        ErrorCode::not_found(),
                        "collection not found",
                    ));
                };

                let wal_record = collection.insert(documents)?;
                drop(guard);
                db.append_insert_wal(&collection_name, &wal_record).await?;
                db.schedule_checkpoint(&collection_name, wal_record.sequence)
                    .await;
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
                let target_sequence = db.collection_insert_sequence(&collection_name).await?;
                if target_sequence == 0 {
                    return Ok(());
                }

                let checkpoint_state = db
                    .schedule_checkpoint(&collection_name, target_sequence)
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

                let response = {
                    let guard = db.collections.read().await;
                    let Some(collection) = guard.get(&collection_name) else {
                        return Err(ErrorEnvelope::expected(
                            ErrorCode::not_found(),
                            "collection not found",
                        ));
                    };

                    let search_output = collection.index.search_with_kernel(
                        query_vector.as_ref(),
                        top_k.saturating_mul(5),
                        &*db.kernel,
                        db.search_backend,
                    )?;

                    tracing::debug!(
                        kernel = ?db.kernel.kind(),
                        backend = ?db.search_backend,
                        top_k,
                        match_count = search_output.matches.len(),
                        "adapter.vectordb.local.search_completed"
                    );

                    let index_size = u64::try_from(collection.index.active_count()).ok();
                    let stats = Some(SearchStats {
                        expansions: search_output.stats.expansions,
                        kernel: kernel_kind_name(search_output.stats.kernel).into(),
                        extra: search_output.stats.extra,
                        kernel_search_duration_ns: search_output.stats.kernel_search_duration_ns,
                        index_size,
                    });

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
                        let search_output = collection.index.search_with_kernel(
                            query.as_ref(),
                            limit.saturating_mul(5),
                            &*db.kernel,
                            db.search_backend,
                        )?;

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
                let snapshot = collection.snapshot();
                drop(guard);
                db.write_snapshot(&collection_name, &snapshot).await
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

impl Clone for LocalVectorDb {
    fn clone(&self) -> Self {
        Self {
            provider: self.provider.clone(),
            codebase_root: self.codebase_root.clone(),
            storage_mode: self.storage_mode.clone(),
            snapshot_format: self.snapshot_format,
            snapshot_max_bytes: self.snapshot_max_bytes,
            kernel: Arc::clone(&self.kernel),
            force_reindex_on_kernel_change: self.force_reindex_on_kernel_change,
            search_backend: self.search_backend,
            collections: Arc::clone(&self.collections),
            checkpoint_states: Arc::clone(&self.checkpoint_states),
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

const fn kernel_kind_name(kernel: VectorKernelKind) -> &'static str {
    match kernel {
        VectorKernelKind::HnswRs => "hnsw-rs",
        VectorKernelKind::Dfrr => "dfrr",
        VectorKernelKind::FlatScan => "flat-scan",
    }
}

struct LocalCollection {
    dimension: u32,
    index_mode: IndexMode,
    index: VectorIndex,
    documents: BTreeMap<Box<str>, StoredDocument>,
    last_insert_sequence: u64,
}

impl LocalCollection {
    fn new(dimension: u32, index_mode: IndexMode) -> Result<Self> {
        let params = HnswParams::default();
        let index = VectorIndex::new(dimension, params)?;
        Ok(Self {
            dimension,
            index_mode,
            index,
            documents: BTreeMap::new(),
            last_insert_sequence: 0,
        })
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

        self.index.insert(records)?;
        for (id, doc) in docs {
            self.documents.insert(id, doc);
        }
        self.last_insert_sequence = sequence;

        Ok(InsertWalRecord {
            sequence,
            documents: wal_documents,
        })
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

        self.index.insert(index_records)?;
        for (id, document) in documents {
            self.documents.insert(id, document);
        }
        self.last_insert_sequence = record.sequence;
        Ok(())
    }

    fn delete(&mut self, ids: &[Box<str>]) -> Result<()> {
        self.index.delete(ids)?;
        for id in ids {
            self.documents.remove(id.as_ref());
        }
        Ok(())
    }

    fn snapshot(&self) -> CollectionSnapshot {
        let mut records = Vec::new();
        for (id, doc) in &self.documents {
            if let Some(record) = self.index.record_for_id(id.as_ref()) {
                records.push(CollectionRecord {
                    id: id.clone(),
                    vector: record.vector.clone(),
                    content: doc.content.clone(),
                    metadata: doc.metadata.clone(),
                });
            }
        }

        CollectionSnapshot {
            version: LOCAL_SNAPSHOT_VERSION,
            dimension: self.dimension,
            index_mode: self.index_mode,
            records,
            checkpoint_sequence: (self.last_insert_sequence > 0)
                .then_some(self.last_insert_sequence),
        }
    }

    fn from_snapshot(snapshot: CollectionSnapshot) -> Result<Self> {
        let CollectionSnapshot {
            version,
            dimension,
            index_mode,
            records,
            checkpoint_sequence,
        } = snapshot;
        validate_local_snapshot_version(version)?;
        let params = HnswParams::default();
        let mut index = VectorIndex::new(dimension, params)?;
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
            index,
            documents,
            last_insert_sequence: checkpoint_sequence.unwrap_or(0),
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
            index,
            documents,
            last_insert_sequence: checkpoint_sequence.unwrap_or(0),
        })
    }

    /// Construct from a pre-built v2 index and the metadata sidecar — no v1 JSON needed.
    fn from_v2_index_and_sidecar(index: VectorIndex, sidecar_path: &Path) -> Result<Self> {
        let sidecar = read_records_meta_sidecar(sidecar_path)?;
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
            index,
            documents: sidecar.documents,
            last_insert_sequence: sidecar.checkpoint_sequence.unwrap_or(0),
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
/// `ids.json` and `vectors.u8.bin`.  Records are emitted in the same
/// BTreeMap-by-ID order used by `CollectionSnapshot::records` (which matches
/// the order `snapshot_v2_for_kernel` uses via `ordered_records()`).
fn write_records_meta_sidecar(
    paths: &CollectionSnapshotPaths,
    snapshot: &CollectionSnapshot,
) -> Result<()> {
    use std::io::Write;

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

    let mut buf = Vec::with_capacity(snapshot.records.len() * 256);
    serde_json::to_writer(&mut buf, &header).map_err(|error| {
        snapshot_error(
            "sidecar_serialize_failed",
            "failed to serialize metadata sidecar header",
            error,
        )
    })?;
    buf.push(b'\n');

    for record in &snapshot.records {
        let entry = RecordMetadataEntry {
            id: record.id.clone(),
            content: record.content.clone(),
            metadata: record.metadata.clone(),
        };
        serde_json::to_writer(&mut buf, &entry).map_err(|error| {
            snapshot_error(
                "sidecar_serialize_failed",
                "failed to serialize metadata sidecar entry",
                error,
            )
        })?;
        buf.push(b'\n');
    }

    let path = &paths.v2_records_meta;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(ErrorEnvelope::from)?;
    }
    let mut file = std::fs::File::create(path).map_err(ErrorEnvelope::from)?;
    file.write_all(&buf).map_err(ErrorEnvelope::from)?;
    file.flush().map_err(ErrorEnvelope::from)?;

    tracing::debug!(
        path = %path.display(),
        records = count,
        bytes = buf.len(),
        "wrote records.meta.jsonl sidecar"
    );
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
    for (line_idx, line) in lines.enumerate() {
        let entry: RecordMetadataEntry = serde_json::from_slice(line).map_err(|error| {
            snapshot_error(
                "sidecar_parse_failed",
                &format!(
                    "failed to parse metadata sidecar record at line {}",
                    line_idx + 1
                ),
                error,
            )
        })?;
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

fn collection_name_from_filename(filename: &str) -> Option<CollectionName> {
    let trimmed = filename.strip_suffix(".json")?;
    CollectionName::parse(trimmed).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_domain::LineSpan;
    use semantic_code_ports::VectorSearchOptions;
    use semantic_code_vector::HnswKernel;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    use tokio::time::timeout;

    fn sample_metadata(path: &str) -> Result<VectorDocumentMetadata> {
        Ok(VectorDocumentMetadata {
            relative_path: path.into(),
            language: None,
            file_extension: Some("rs".into()),
            span: LineSpan::new(1, 1)?,
            node_kind: None,
        })
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

    #[test]
    fn snapshot_paths_resolve_v2_bundle() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-paths-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let db = LocalVectorDb::new(
            tmp.clone(),
            SnapshotStorageMode::Custom(tmp.clone()),
            VectorSnapshotFormat::V2,
            None,
            Arc::new(HnswKernel::new()),
            false,
            VectorSearchStrategy::F32Hnsw,
        )?;
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
        Ok(())
    }

    #[tokio::test]
    async fn insert_and_flush_persists_checkpoint_sequence_and_compacts_wal() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-wal-sequence-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let collection = CollectionName::parse("wal_sequence")?;
        let ctx = RequestContext::new_request();

        let db = LocalVectorDb::new(
            tmp.clone(),
            SnapshotStorageMode::Custom(tmp.clone()),
            VectorSnapshotFormat::V1,
            None,
            Arc::new(HnswKernel::new()),
            false,
            VectorSearchStrategy::F32Hnsw,
        )?;

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
    async fn restart_replays_wal_records_before_flush_and_flush_persists_state() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-wal-replay-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let collection = CollectionName::parse("wal_replay")?;
        let ctx = RequestContext::new_request();

        let initial = LocalVectorDb::new(
            tmp.clone(),
            SnapshotStorageMode::Custom(tmp.clone()),
            VectorSnapshotFormat::V1,
            None,
            Arc::new(HnswKernel::new()),
            false,
            VectorSearchStrategy::F32Hnsw,
        )?;
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

        let restarted = LocalVectorDb::new(
            tmp.clone(),
            SnapshotStorageMode::Custom(tmp.clone()),
            VectorSnapshotFormat::V1,
            None,
            Arc::new(HnswKernel::new()),
            false,
            VectorSearchStrategy::F32Hnsw,
        )?;
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

        let reopened = LocalVectorDb::new(
            tmp.clone(),
            SnapshotStorageMode::Custom(tmp.clone()),
            VectorSnapshotFormat::V1,
            None,
            Arc::new(HnswKernel::new()),
            false,
            VectorSearchStrategy::F32Hnsw,
        )?;
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
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let collection = CollectionName::parse("wal_gap")?;
        let ctx = RequestContext::new_request();

        let db = LocalVectorDb::new(
            tmp.clone(),
            SnapshotStorageMode::Custom(tmp.clone()),
            VectorSnapshotFormat::V1,
            None,
            Arc::new(HnswKernel::new()),
            false,
            VectorSearchStrategy::F32Hnsw,
        )?;
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

        let restarted = LocalVectorDb::new(
            tmp.clone(),
            SnapshotStorageMode::Custom(tmp.clone()),
            VectorSnapshotFormat::V1,
            None,
            Arc::new(HnswKernel::new()),
            false,
            VectorSearchStrategy::F32Hnsw,
        )?;
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
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let collection = CollectionName::parse("flush_waits")?;
        let ctx = RequestContext::new_request();

        let db = LocalVectorDb::new(
            tmp.clone(),
            SnapshotStorageMode::Custom(tmp.clone()),
            VectorSnapshotFormat::V1,
            None,
            Arc::new(HnswKernel::new()),
            false,
            VectorSearchStrategy::F32Hnsw,
        )?;
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
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let collection = CollectionName::parse("local_snapshot")?;
        let ctx = RequestContext::new_request();
        let db_v1 = LocalVectorDb::new(
            tmp.clone(),
            SnapshotStorageMode::Custom(tmp.clone()),
            VectorSnapshotFormat::V1,
            None,
            Arc::new(HnswKernel::new()),
            false,
            VectorSearchStrategy::F32Hnsw,
        )?;

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

        let db_v2 = LocalVectorDb::new(
            tmp.clone(),
            SnapshotStorageMode::Custom(tmp.clone()),
            VectorSnapshotFormat::V2,
            None,
            Arc::new(HnswKernel::new()),
            false,
            VectorSearchStrategy::F32Hnsw,
        )?;
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
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let collection = CollectionName::parse("local_snapshot_repair")?;
        let ctx = RequestContext::new_request();
        let db = LocalVectorDb::new(
            tmp.clone(),
            SnapshotStorageMode::Custom(tmp.clone()),
            VectorSnapshotFormat::V2,
            None,
            Arc::new(HnswKernel::new()),
            false,
            VectorSearchStrategy::F32Hnsw,
        )?;
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

        let restarted = LocalVectorDb::new(
            tmp.clone(),
            SnapshotStorageMode::Custom(tmp.clone()),
            VectorSnapshotFormat::V2,
            None,
            Arc::new(HnswKernel::new()),
            false,
            VectorSearchStrategy::F32Hnsw,
        )?;
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
    async fn snapshot_roundtrip_persists_records() -> Result<()> {
        let tmp = std::env::temp_dir().join(format!(
            "sca-localdb-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let db = LocalVectorDb::new(
            tmp.clone(),
            SnapshotStorageMode::Custom(tmp.clone()),
            VectorSnapshotFormat::V1,
            None,
            Arc::new(HnswKernel::new()),
            false,
            VectorSearchStrategy::F32Hnsw,
        )?;
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

        let restored = LocalVectorDb::new(
            tmp.clone(),
            SnapshotStorageMode::Custom(tmp.clone()),
            VectorSnapshotFormat::V1,
            None,
            Arc::new(HnswKernel::new()),
            false,
            VectorSearchStrategy::F32Hnsw,
        )?;
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
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let collection = CollectionName::parse("kernel_mismatch")?;
        let ctx = RequestContext::new_request();

        let db = LocalVectorDb::new(
            tmp.clone(),
            SnapshotStorageMode::Custom(tmp.clone()),
            VectorSnapshotFormat::V2,
            None,
            Arc::new(HnswKernel::new()),
            false,
            VectorSearchStrategy::F32Hnsw,
        )?;
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

        let restarted = LocalVectorDb::new(
            tmp.clone(),
            SnapshotStorageMode::Custom(tmp.clone()),
            VectorSnapshotFormat::V2,
            None,
            Arc::new(HnswKernel::new()),
            false,
            VectorSearchStrategy::F32Hnsw,
        )?;
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
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let collection = CollectionName::parse("kernel_force_reindex")?;
        let ctx = RequestContext::new_request();

        let db = LocalVectorDb::new(
            tmp.clone(),
            SnapshotStorageMode::Custom(tmp.clone()),
            VectorSnapshotFormat::V2,
            None,
            Arc::new(HnswKernel::new()),
            false,
            VectorSearchStrategy::F32Hnsw,
        )?;
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

        let paths = db.snapshot_paths(&collection).ok_or_else(|| {
            std::io::Error::other("expected snapshot paths for custom storage mode")
        })?;
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

        let restarted = LocalVectorDb::new(
            tmp.clone(),
            SnapshotStorageMode::Custom(tmp.clone()),
            VectorSnapshotFormat::V2,
            None,
            Arc::new(HnswKernel::new()),
            true,
            VectorSearchStrategy::F32Hnsw,
        )?;
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
}
