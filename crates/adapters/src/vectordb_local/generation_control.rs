use super::{CollectionSnapshot, write_published_records_meta_sidecar};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior};
use semantic_code_domain::CollectionName;
use semantic_code_shared::{CancellationToken, ErrorClass, ErrorCode, ErrorEnvelope, Result};
use semantic_code_vector::{
    CollectionGenerationPaths, ExactVectorRowSource, ExactVectorRowView, ExactVectorRows,
    GenerationId, write_exact_generation,
};
use serde::Serialize;
use std::collections::{BTreeSet, HashMap};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio_util::task::TaskTracker;
use tokio_util::task::task_tracker::TaskTrackerToken;

const BUILD_COORDINATOR_CHANNEL_CAPACITY: usize = 8;
const SQLITE_BUSY_TIMEOUT: Duration = Duration::from_secs(2);
const DUPLICATE_ID_REPORT_FILE_NAME: &str = "duplicate_ids.json";

#[derive(Debug)]
enum CollectionBuildCommand {
    EnsureScaffold {
        name: CollectionName,
        layout: CollectionGenerationPaths,
        reply: oneshot::Sender<Result<()>>,
    },
    DropCollection {
        name: CollectionName,
        layout: CollectionGenerationPaths,
        reply: oneshot::Sender<Result<()>>,
    },
    StageBaseGeneration {
        name: CollectionName,
        layout: CollectionGenerationPaths,
        rows: ExactVectorRows,
        snapshot: CollectionSnapshot,
        reply: oneshot::Sender<Result<GenerationId>>,
    },
    ActivateGeneration {
        name: CollectionName,
        layout: CollectionGenerationPaths,
        generation_id: GenerationId,
        reply: oneshot::Sender<Result<()>>,
    },
    BeginJournalAppend {
        name: CollectionName,
        reply: oneshot::Sender<Result<BuildJournalAdmission>>,
    },
    CloseSession {
        name: CollectionName,
        reply: oneshot::Sender<Result<BuildSessionCloseHandle>>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BuildSessionState {
    Open,
    Closing,
    Sealed { generation_id: GenerationId },
    Publishing { generation_id: GenerationId },
    Published { generation_id: GenerationId },
    Failed,
    Cancelled,
}

#[derive(Clone)]
struct BuildSessionRecord {
    state: BuildSessionState,
    tracker: TaskTracker,
}

impl BuildSessionRecord {
    fn open() -> Self {
        Self {
            state: BuildSessionState::Open,
            tracker: TaskTracker::new(),
        }
    }

    const fn state_name(&self) -> BuildSessionStateName {
        match self.state {
            BuildSessionState::Open => BuildSessionStateName::Open,
            BuildSessionState::Closing => BuildSessionStateName::Closing,
            BuildSessionState::Sealed { .. } => BuildSessionStateName::Sealed,
            BuildSessionState::Publishing { .. } => BuildSessionStateName::Publishing,
            BuildSessionState::Published { .. } => BuildSessionStateName::Published,
            BuildSessionState::Failed => BuildSessionStateName::Failed,
            BuildSessionState::Cancelled => BuildSessionStateName::Cancelled,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BuildSessionStateName {
    Open,
    Closing,
    Sealed,
    Publishing,
    Published,
    Failed,
    Cancelled,
}

impl BuildSessionStateName {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Closing => "closing",
            Self::Sealed => "sealed",
            Self::Publishing => "publishing",
            Self::Published => "published",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug)]
pub(super) struct BuildJournalAdmission {
    _token: TaskTrackerToken,
}

#[derive(Debug)]
pub(super) struct BuildSessionCloseHandle {
    tracker: TaskTracker,
}

impl BuildSessionCloseHandle {
    pub(super) async fn wait(self) {
        self.tracker.wait().await;
    }
}

fn invalid_build_session_state_error(
    name: &CollectionName,
    operation: &'static str,
    current: BuildSessionStateName,
    allowed: &[BuildSessionStateName],
) -> ErrorEnvelope {
    let allowed = allowed
        .iter()
        .map(|state| state.as_str())
        .collect::<Vec<_>>()
        .join(",");
    ErrorEnvelope::expected(
        ErrorCode::new("vector", "build_session_state_invalid"),
        "build session state does not allow this operation",
    )
    .with_metadata("collection", name.as_str().to_string())
    .with_metadata("operation", operation.to_string())
    .with_metadata("current", current.as_str().to_string())
    .with_metadata("allowed", allowed)
}

#[derive(Clone)]
pub(super) struct CollectionBuildCoordinatorHandle {
    tx: mpsc::Sender<CollectionBuildCommand>,
}

impl CollectionBuildCoordinatorHandle {
    pub(super) async fn ensure_scaffold(
        &self,
        name: CollectionName,
        layout: CollectionGenerationPaths,
    ) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(CollectionBuildCommand::EnsureScaffold {
                name: name.clone(),
                layout,
                reply: reply_tx,
            })
            .await
            .map_err(|_| build_coordinator_channel_closed_error(&name))?;
        reply_rx.await.map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "build_coordinator_reply_dropped"),
                "build coordinator reply channel dropped",
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", name.as_str().to_string())
        })?
    }

    pub(super) async fn drop_collection(
        &self,
        name: CollectionName,
        layout: CollectionGenerationPaths,
    ) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(CollectionBuildCommand::DropCollection {
                name: name.clone(),
                layout,
                reply: reply_tx,
            })
            .await
            .map_err(|_| build_coordinator_channel_closed_error(&name))?;
        reply_rx.await.map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "build_coordinator_reply_dropped"),
                "build coordinator reply channel dropped",
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", name.as_str().to_string())
        })?
    }

    pub(super) async fn stage_base_generation(
        &self,
        name: CollectionName,
        layout: CollectionGenerationPaths,
        rows: ExactVectorRows,
        snapshot: CollectionSnapshot,
    ) -> Result<GenerationId> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(CollectionBuildCommand::StageBaseGeneration {
                name: name.clone(),
                layout,
                rows,
                snapshot,
                reply: reply_tx,
            })
            .await
            .map_err(|_| build_coordinator_channel_closed_error(&name))?;
        reply_rx.await.map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "build_coordinator_reply_dropped"),
                "build coordinator reply channel dropped",
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", name.as_str().to_string())
        })?
    }

    pub(super) async fn activate_generation(
        &self,
        name: CollectionName,
        layout: CollectionGenerationPaths,
        generation_id: GenerationId,
    ) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(CollectionBuildCommand::ActivateGeneration {
                name: name.clone(),
                layout,
                generation_id,
                reply: reply_tx,
            })
            .await
            .map_err(|_| build_coordinator_channel_closed_error(&name))?;
        reply_rx.await.map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "build_coordinator_reply_dropped"),
                "build coordinator reply channel dropped",
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", name.as_str().to_string())
        })?
    }

    pub(super) async fn begin_journal_append(
        &self,
        name: CollectionName,
    ) -> Result<BuildJournalAdmission> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(CollectionBuildCommand::BeginJournalAppend {
                name: name.clone(),
                reply: reply_tx,
            })
            .await
            .map_err(|_| build_coordinator_channel_closed_error(&name))?;
        reply_rx.await.map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "build_coordinator_reply_dropped"),
                "build coordinator reply channel dropped",
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", name.as_str().to_string())
        })?
    }

    pub(super) async fn close_session(
        &self,
        name: CollectionName,
    ) -> Result<BuildSessionCloseHandle> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(CollectionBuildCommand::CloseSession {
                name: name.clone(),
                reply: reply_tx,
            })
            .await
            .map_err(|_| build_coordinator_channel_closed_error(&name))?;
        reply_rx.await.map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "build_coordinator_reply_dropped"),
                "build coordinator reply channel dropped",
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", name.as_str().to_string())
        })?
    }
}

pub(super) struct CollectionBuildCoordinatorActor {
    sessions: HashMap<CollectionName, BuildSessionRecord>,
    rx: mpsc::Receiver<CollectionBuildCommand>,
    cancellation: CancellationToken,
}

impl CollectionBuildCoordinatorActor {
    pub(super) fn spawn(
        cancellation: CancellationToken,
    ) -> (CollectionBuildCoordinatorHandle, JoinHandle<()>) {
        let (tx, rx) = mpsc::channel(BUILD_COORDINATOR_CHANNEL_CAPACITY);
        let actor = Self {
            sessions: HashMap::new(),
            rx,
            cancellation,
        };
        let join = tokio::spawn(actor.run());
        (CollectionBuildCoordinatorHandle { tx }, join)
    }

    async fn run(mut self) {
        loop {
            tokio::select! {
                () = self.cancellation.cancelled() => {
                    for session in self.sessions.values_mut() {
                        session.state = BuildSessionState::Cancelled;
                        session.tracker.close();
                    }
                    tracing::info!("collection build coordinator shutting down (cancellation)");
                    break;
                }
                cmd = self.rx.recv() => {
                    match cmd {
                        Some(CollectionBuildCommand::EnsureScaffold { name, layout, reply }) => {
                            let result = self.handle_ensure_scaffold(&name, &layout).await;
                            let _ = reply.send(result);
                        }
                        Some(CollectionBuildCommand::DropCollection { name, layout, reply }) => {
                            let result = self.handle_drop_collection(&name, &layout).await;
                            let _ = reply.send(result);
                        }
                        Some(CollectionBuildCommand::StageBaseGeneration { name, layout, rows, snapshot, reply }) => {
                            let result = self
                                .handle_stage_base_generation(&name, &layout, rows, snapshot)
                                .await;
                            let _ = reply.send(result);
                        }
                        Some(CollectionBuildCommand::ActivateGeneration { name, layout, generation_id, reply }) => {
                            let result = self
                                .handle_activate_generation(&name, &layout, &generation_id)
                                .await;
                            let _ = reply.send(result);
                        }
                        Some(CollectionBuildCommand::BeginJournalAppend { name, reply }) => {
                            let result = self.handle_begin_journal_append(&name);
                            let _ = reply.send(result);
                        }
                        Some(CollectionBuildCommand::CloseSession { name, reply }) => {
                            let result = self.handle_close_session(&name);
                            let _ = reply.send(result);
                        }
                        None => {
                            tracing::info!("collection build coordinator shutting down (channel closed)");
                            break;
                        }
                    }
                }
            }
        }
    }

    async fn handle_ensure_scaffold(
        &mut self,
        name: &CollectionName,
        layout: &CollectionGenerationPaths,
    ) -> Result<()> {
        let name_for_task = name.clone();
        let layout_for_task = layout.clone();
        tokio::task::spawn_blocking(move || {
            initialize_generation_scaffold(&name_for_task, &layout_for_task)
        })
        .await
        .map_err(|join_error| {
            map_join_error(&join_error, "initialize_generation_scaffold", name)
        })??;

        self.sessions
            .entry(name.clone())
            .or_insert_with(BuildSessionRecord::open);
        Ok(())
    }

    async fn handle_drop_collection(
        &mut self,
        name: &CollectionName,
        layout: &CollectionGenerationPaths,
    ) -> Result<()> {
        let name_for_task = name.clone();
        let layout_for_task = layout.clone();
        tokio::task::spawn_blocking(move || {
            remove_generation_scaffold(&name_for_task, &layout_for_task)
        })
        .await
        .map_err(|join_error| map_join_error(&join_error, "remove_generation_scaffold", name))??;

        self.sessions.remove(name);
        Ok(())
    }

    async fn handle_stage_base_generation(
        &mut self,
        name: &CollectionName,
        layout: &CollectionGenerationPaths,
        rows: ExactVectorRows,
        snapshot: CollectionSnapshot,
    ) -> Result<GenerationId> {
        self.require_state(
            name,
            "stage_base_generation",
            &[BuildSessionStateName::Closing],
        )?;
        let name_for_task = name.clone();
        let layout_for_task = layout.clone();
        let generation = tokio::task::spawn_blocking(move || {
            stage_base_generation(&name_for_task, &layout_for_task, &rows, &snapshot)
        })
        .await
        .map_err(|join_error| map_join_error(&join_error, "stage_base_generation", name))?;
        let generation = match generation {
            Ok(generation) => generation,
            Err(error) => {
                self.mark_failed(name);
                return Err(error);
            },
        };
        if let Some(session) = self.sessions.get_mut(name) {
            session.state = BuildSessionState::Sealed {
                generation_id: generation.clone(),
            };
        }
        Ok(generation)
    }

    async fn handle_activate_generation(
        &mut self,
        name: &CollectionName,
        layout: &CollectionGenerationPaths,
        generation_id: &GenerationId,
    ) -> Result<()> {
        let Some(expected_generation) = self.sessions.get(name).and_then(|session| match &session
            .state
        {
            BuildSessionState::Sealed { generation_id } => Some(generation_id.clone()),
            _ => None,
        }) else {
            self.require_state(
                name,
                "activate_generation",
                &[BuildSessionStateName::Sealed],
            )?;
            unreachable!("require_state must fail for non-sealed sessions");
        };
        if expected_generation != *generation_id {
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("vector", "build_session_generation_mismatch"),
                "build session generation does not match activation request",
            )
            .with_metadata("collection", name.as_str().to_string())
            .with_metadata(
                "expected_generation",
                expected_generation.as_str().to_string(),
            )
            .with_metadata("requested_generation", generation_id.as_str().to_string()));
        }
        if let Some(session) = self.sessions.get_mut(name) {
            session.state = BuildSessionState::Publishing {
                generation_id: generation_id.clone(),
            };
        }
        let name_for_task = name.clone();
        let layout_for_task = layout.clone();
        let generation_id_for_task = generation_id.clone();
        tokio::task::spawn_blocking(move || {
            activate_generation(&name_for_task, &layout_for_task, &generation_id_for_task)
        })
        .await
        .map_err(|join_error| map_join_error(&join_error, "activate_generation", name))?
        .inspect_err(|_| {
            self.mark_failed(name);
        })?;
        if let Some(session) = self.sessions.get_mut(name) {
            session.state = BuildSessionState::Published {
                generation_id: generation_id.clone(),
            };
        }
        Ok(())
    }

    fn handle_begin_journal_append(
        &mut self,
        name: &CollectionName,
    ) -> Result<BuildJournalAdmission> {
        let session = self.session_mut(name)?;
        if !matches!(session.state, BuildSessionState::Open) {
            return Err(invalid_build_session_state_error(
                name,
                "begin_journal_append",
                session.state_name(),
                &[BuildSessionStateName::Open],
            ));
        }
        Ok(BuildJournalAdmission {
            _token: session.tracker.token(),
        })
    }

    fn handle_close_session(&mut self, name: &CollectionName) -> Result<BuildSessionCloseHandle> {
        let session = self.session_mut(name)?;
        match session.state {
            BuildSessionState::Open => {
                session.state = BuildSessionState::Closing;
                session.tracker.close();
                Ok(BuildSessionCloseHandle {
                    tracker: session.tracker.clone(),
                })
            },
            BuildSessionState::Closing => Ok(BuildSessionCloseHandle {
                tracker: session.tracker.clone(),
            }),
            _ => Err(invalid_build_session_state_error(
                name,
                "close_session",
                session.state_name(),
                &[BuildSessionStateName::Open, BuildSessionStateName::Closing],
            )),
        }
    }

    fn session_mut(&mut self, name: &CollectionName) -> Result<&mut BuildSessionRecord> {
        self.sessions.get_mut(name).ok_or_else(|| {
            ErrorEnvelope::expected(
                ErrorCode::new("vector", "build_session_missing"),
                "build session missing for collection",
            )
            .with_metadata("collection", name.as_str().to_string())
        })
    }

    fn require_state(
        &mut self,
        name: &CollectionName,
        operation: &'static str,
        allowed: &[BuildSessionStateName],
    ) -> Result<()> {
        let session = self.session_mut(name)?;
        let current = session.state_name();
        if allowed.contains(&current) {
            return Ok(());
        }
        Err(invalid_build_session_state_error(
            name, operation, current, allowed,
        ))
    }

    fn mark_failed(&mut self, name: &CollectionName) {
        if let Some(session) = self.sessions.get_mut(name) {
            session.state = BuildSessionState::Failed;
        }
    }
}

fn initialize_generation_scaffold(
    collection_name: &CollectionName,
    layout: &CollectionGenerationPaths,
) -> Result<()> {
    std::fs::create_dir_all(layout.generations_dir()).map_err(|source| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "generation_root_create_failed"),
            "failed to create generation root directory",
            ErrorClass::NonRetriable,
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("path", layout.generations_dir().display().to_string())
        .with_metadata("source", source.to_string())
    })?;

    let mut connection = open_catalog(layout.catalog_db().as_path(), collection_name)?;
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "generation_catalog_transaction_failed"),
                format!("generation catalog transaction start failed: {error}"),
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata("path", layout.catalog_db().display().to_string())
        })?;

    transaction
        .execute_batch(
            "CREATE TABLE IF NOT EXISTS generations (
                generation_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS active_generation (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                generation_id TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS build_lease (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                generation_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                updated_at_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS kernel_states (
                generation_id TEXT NOT NULL,
                kernel TEXT NOT NULL,
                state TEXT NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                PRIMARY KEY (generation_id, kernel)
            );
            CREATE TABLE IF NOT EXISTS dfrr_ready_states (
                generation_id TEXT NOT NULL,
                ready_state_fingerprint TEXT NOT NULL,
                state TEXT NOT NULL,
                artifact_root TEXT NOT NULL,
                config_json TEXT NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                PRIMARY KEY (generation_id, ready_state_fingerprint)
            );",
        )
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "generation_catalog_schema_failed"),
                format!("generation catalog schema setup failed: {error}"),
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata("path", layout.catalog_db().display().to_string())
        })?;

    transaction.commit().map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "generation_catalog_commit_failed"),
            format!("generation catalog commit failed: {error}"),
            ErrorClass::NonRetriable,
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("path", layout.catalog_db().display().to_string())
    })?;

    Ok(())
}

fn remove_generation_scaffold(
    collection_name: &CollectionName,
    layout: &CollectionGenerationPaths,
) -> Result<()> {
    match std::fs::remove_dir_all(layout.root()) {
        Ok(()) => Ok(()),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "generation_root_remove_failed"),
            "failed to remove generation root directory",
            ErrorClass::NonRetriable,
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("path", layout.root().display().to_string())
        .with_metadata("source", source.to_string())),
    }
}

fn stage_base_generation(
    collection_name: &CollectionName,
    layout: &CollectionGenerationPaths,
    rows: &ExactVectorRows,
    snapshot: &CollectionSnapshot,
) -> Result<GenerationId> {
    initialize_generation_scaffold(collection_name, layout)?;
    let generation_id = next_generation_id()?;
    let generation = layout.generation(&generation_id);

    std::fs::create_dir_all(generation.base_dir()).map_err(|source| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "generation_base_dir_create_failed"),
            "failed to create generation base directory",
            ErrorClass::NonRetriable,
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("path", generation.base_dir().display().to_string())
        .with_metadata("source", source.to_string())
    })?;
    std::fs::create_dir_all(generation.kernels_dir()).map_err(|source| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "generation_kernels_dir_create_failed"),
            "failed to create generation kernels directory",
            ErrorClass::NonRetriable,
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("path", generation.kernels_dir().display().to_string())
        .with_metadata("source", source.to_string())
    })?;
    std::fs::create_dir_all(generation.derived_dir()).map_err(|source| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "generation_derived_dir_create_failed"),
            "failed to create generation derived directory",
            ErrorClass::NonRetriable,
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("path", generation.derived_dir().display().to_string())
        .with_metadata("source", source.to_string())
    })?;

    let duplicate_report_path = generation.derived_dir().join(DUPLICATE_ID_REPORT_FILE_NAME);
    if let Err(error) =
        validate_generation_publish_inputs(collection_name, rows, snapshot, &duplicate_report_path)
    {
        let _ = upsert_generation_state(collection_name, layout, &generation_id, "failed");
        return Err(error.with_metadata("generationId", generation_id.as_str().to_string()));
    }

    write_exact_generation(generation.base_dir(), rows)?;
    write_published_records_meta_sidecar(generation.base_dir(), snapshot)?;
    upsert_generation_state(collection_name, layout, &generation_id, "sealed")?;

    Ok(generation_id)
}

fn validate_generation_publish_inputs(
    collection_name: &CollectionName,
    rows: &ExactVectorRows,
    snapshot: &CollectionSnapshot,
    duplicate_report_path: &Path,
) -> Result<()> {
    let exact_duplicates = collect_exact_row_duplicates(rows);
    let sidecar_duplicates = collect_sidecar_duplicates(snapshot);
    if !exact_duplicates.is_empty() || !sidecar_duplicates.is_empty() {
        write_duplicate_id_report(
            collection_name,
            rows,
            snapshot,
            exact_duplicates.as_slice(),
            sidecar_duplicates.as_slice(),
            duplicate_report_path,
        )?;
        let code = if exact_duplicates.is_empty() {
            ErrorCode::new("vector", "generation_publish_duplicate_sidecar_id")
        } else {
            ErrorCode::new("vector", "generation_publish_duplicate_exact_row_id")
        };
        let message = if exact_duplicates.is_empty() {
            "sidecar rows contain a duplicate logical id during generation publish"
        } else {
            "exact rows contain a duplicate logical id during generation publish"
        };
        let first_duplicate = exact_duplicates
            .first()
            .map(|duplicate| duplicate.id.as_ref())
            .or_else(|| {
                sidecar_duplicates
                    .first()
                    .map(|duplicate| duplicate.id.as_ref())
            })
            .unwrap_or_default();
        return Err(ErrorEnvelope::expected(code, message)
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata("duplicateId", first_duplicate.to_string())
            .with_metadata("exactDuplicateCount", exact_duplicates.len().to_string())
            .with_metadata(
                "sidecarDuplicateCount",
                sidecar_duplicates.len().to_string(),
            )
            .with_metadata(
                "duplicateReportPath",
                duplicate_report_path.display().to_string(),
            ));
    }

    if rows.row_count() != snapshot.records.len() {
        return Err(ErrorEnvelope::expected(
            ErrorCode::new("vector", "generation_publish_row_count_mismatch"),
            "exact rows and sidecar rows differ in count during generation publish",
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("exact_row_count", rows.row_count().to_string())
        .with_metadata("sidecar_row_count", snapshot.records.len().to_string())
        .with_metadata(
            "duplicateReportPath",
            duplicate_report_path.display().to_string(),
        ));
    }

    let exact_ids = rows
        .rows()
        .map(|row| Box::<str>::from(row.id()))
        .collect::<Vec<_>>();
    let sidecar_ids = snapshot
        .records
        .iter()
        .map(|record| record.id.clone())
        .collect::<Vec<_>>();

    if exact_ids != sidecar_ids {
        let mismatch_index = exact_ids
            .iter()
            .zip(sidecar_ids.iter())
            .position(|(exact_id, sidecar_id)| exact_id != sidecar_id)
            .unwrap_or(0);
        return Err(ErrorEnvelope::expected(
            ErrorCode::new("vector", "generation_publish_row_id_mismatch"),
            "exact rows and sidecar rows differ in canonical id order during generation publish",
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("row_index", mismatch_index.to_string())
        .with_metadata(
            "exact_id",
            exact_ids
                .get(mismatch_index)
                .map_or_else(String::new, |id| id.as_ref().to_string()),
        )
        .with_metadata(
            "sidecar_id",
            sidecar_ids
                .get(mismatch_index)
                .map_or_else(String::new, |id| id.as_ref().to_string()),
        )
        .with_metadata(
            "duplicateReportPath",
            duplicate_report_path.display().to_string(),
        ));
    }

    Ok(())
}

#[derive(Debug, Serialize)]
struct DuplicateIdReport {
    collection: String,
    exact_row_count: usize,
    exact_unique_id_count: usize,
    sidecar_row_count: usize,
    sidecar_unique_id_count: usize,
    exact_row_duplicates: Vec<ExactRowDuplicateReport>,
    sidecar_duplicates: Vec<SidecarDuplicateReport>,
}

#[derive(Debug, Clone, Serialize)]
struct ExactRowDuplicateReport {
    id: Box<str>,
    count: usize,
    origins: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct SidecarDuplicateReport {
    id: Box<str>,
    count: usize,
    samples: Vec<DuplicateRowSample>,
}

#[derive(Debug, Clone, Serialize)]
struct DuplicateRowSample {
    row_index: usize,
    relative_path: Box<str>,
    start_line: u32,
    end_line: u32,
    fragment_start_byte: Option<u32>,
    fragment_end_byte: Option<u32>,
    content_prefix: Box<str>,
}

fn collect_exact_row_duplicates(rows: &ExactVectorRows) -> Vec<ExactRowDuplicateReport> {
    let mut by_id: HashMap<Box<str>, Vec<usize>> = HashMap::new();
    for row in rows.rows() {
        by_id
            .entry(Box::<str>::from(row.id()))
            .or_default()
            .push(row.origin().as_usize());
    }

    let mut duplicates = by_id
        .into_iter()
        .filter_map(|(id, origins)| {
            (origins.len() > 1).then_some(ExactRowDuplicateReport {
                count: origins.len(),
                id,
                origins,
            })
        })
        .collect::<Vec<_>>();
    duplicates.sort_by(|lhs, rhs| lhs.id.cmp(&rhs.id));
    duplicates
}

fn collect_sidecar_duplicates(snapshot: &CollectionSnapshot) -> Vec<SidecarDuplicateReport> {
    let mut by_id: HashMap<Box<str>, Vec<DuplicateRowSample>> = HashMap::new();
    for (row_index, record) in snapshot.records.iter().enumerate() {
        by_id
            .entry(record.id.clone())
            .or_default()
            .push(DuplicateRowSample {
                row_index,
                relative_path: record.metadata.relative_path.clone(),
                start_line: record.metadata.span.start_line(),
                end_line: record.metadata.span.end_line(),
                fragment_start_byte: record.metadata.fragment_start_byte,
                fragment_end_byte: record.metadata.fragment_end_byte,
                content_prefix: record
                    .content
                    .chars()
                    .take(80)
                    .collect::<String>()
                    .into_boxed_str(),
            });
    }

    let mut duplicates = by_id
        .into_iter()
        .filter_map(|(id, samples)| {
            (samples.len() > 1).then_some(SidecarDuplicateReport {
                count: samples.len(),
                id,
                samples,
            })
        })
        .collect::<Vec<_>>();
    duplicates.sort_by(|lhs, rhs| lhs.id.cmp(&rhs.id));
    duplicates
}

fn write_duplicate_id_report(
    collection_name: &CollectionName,
    rows: &ExactVectorRows,
    snapshot: &CollectionSnapshot,
    exact_duplicates: &[ExactRowDuplicateReport],
    sidecar_duplicates: &[SidecarDuplicateReport],
    path: &Path,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(ErrorEnvelope::from)?;
    }
    let exact_unique_id_count = rows
        .rows()
        .map(|row| row.id().to_string())
        .collect::<BTreeSet<_>>()
        .len();
    let sidecar_unique_id_count = snapshot
        .records
        .iter()
        .map(|record| record.id.clone())
        .collect::<BTreeSet<_>>()
        .len();
    let payload = serde_json::to_vec_pretty(&DuplicateIdReport {
        collection: collection_name.as_str().to_string(),
        exact_row_count: rows.row_count(),
        exact_unique_id_count,
        sidecar_row_count: snapshot.records.len(),
        sidecar_unique_id_count,
        exact_row_duplicates: exact_duplicates.to_vec(),
        sidecar_duplicates: sidecar_duplicates.to_vec(),
    })
    .map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "duplicate_report_serialize_failed"),
            format!("failed to serialize duplicate id report: {error}"),
            ErrorClass::NonRetriable,
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("path", path.display().to_string())
    })?;
    std::fs::write(path, payload).map_err(|source| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "duplicate_report_write_failed"),
            "failed to write duplicate id report",
            ErrorClass::NonRetriable,
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("path", path.display().to_string())
        .with_metadata("source", source.to_string())
    })
}

fn activate_generation(
    collection_name: &CollectionName,
    layout: &CollectionGenerationPaths,
    generation_id: &GenerationId,
) -> Result<()> {
    upsert_generation_state(collection_name, layout, generation_id, "published")?;
    write_active_generation_catalog(collection_name, layout, generation_id)?;
    write_active_generation_pointer(
        collection_name,
        layout.active_file().as_path(),
        generation_id,
    )?;

    Ok(())
}

fn write_active_generation_pointer(
    collection_name: &CollectionName,
    path: &Path,
    generation_id: &GenerationId,
) -> Result<()> {
    static NEXT_ACTIVE_POINTER_WRITE_ID: AtomicU64 = AtomicU64::new(0);

    let Some(parent) = path.parent() else {
        return Err(ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "generation_active_file_parent_missing"),
            "ACTIVE generation pointer parent directory is missing",
            ErrorClass::NonRetriable,
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("path", path.display().to_string()));
    };

    std::fs::create_dir_all(parent).map_err(|source| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "generation_active_file_parent_create_failed"),
            "failed to create ACTIVE generation pointer parent directory",
            ErrorClass::NonRetriable,
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("path", parent.display().to_string())
        .with_metadata("source", source.to_string())
    })?;

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("ACTIVE");
    let temp_path = parent.join(format!(
        ".{file_name}.tmp-{}-{}",
        std::process::id(),
        NEXT_ACTIVE_POINTER_WRITE_ID.fetch_add(1, Ordering::Relaxed)
    ));

    let write_result = (|| -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp_path)
            .map_err(|source| {
                ErrorEnvelope::unexpected(
                    ErrorCode::new("vector", "generation_active_file_temp_open_failed"),
                    "failed to open temporary ACTIVE generation pointer file",
                    ErrorClass::NonRetriable,
                )
                .with_metadata("collection", collection_name.as_str().to_string())
                .with_metadata("path", temp_path.display().to_string())
                .with_metadata("source", source.to_string())
            })?;
        file.write_all(generation_id.as_str().as_bytes())
            .map_err(ErrorEnvelope::from)?;
        file.sync_all().map_err(ErrorEnvelope::from)?;
        std::fs::rename(&temp_path, path).map_err(|source| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "generation_active_file_write_failed"),
                "failed to atomically replace ACTIVE generation pointer",
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata("path", path.display().to_string())
            .with_metadata("tempPath", temp_path.display().to_string())
            .with_metadata("source", source.to_string())
        })?;
        sync_directory(parent, collection_name)?;
        Ok(())
    })();

    if write_result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }

    write_result
}

fn sync_directory(path: &Path, collection_name: &CollectionName) -> Result<()> {
    let directory = std::fs::File::open(path).map_err(|source| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "generation_directory_open_failed"),
            "failed to open generation directory for sync",
            ErrorClass::NonRetriable,
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("path", path.display().to_string())
        .with_metadata("source", source.to_string())
    })?;
    directory.sync_all().map_err(|source| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "generation_directory_sync_failed"),
            "failed to sync generation directory",
            ErrorClass::NonRetriable,
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("path", path.display().to_string())
        .with_metadata("source", source.to_string())
    })
}

fn write_active_generation_catalog(
    collection_name: &CollectionName,
    layout: &CollectionGenerationPaths,
    generation_id: &GenerationId,
) -> Result<()> {
    let mut connection = open_catalog(layout.catalog_db().as_path(), collection_name)?;
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "generation_catalog_transaction_failed"),
                format!("generation catalog transaction start failed: {error}"),
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata("path", layout.catalog_db().display().to_string())
        })?;

    let now_ms = now_epoch_ms()?;
    transaction
        .execute(
            "INSERT OR REPLACE INTO generations (generation_id, state, created_at_ms, updated_at_ms)
             VALUES (?1, ?2, COALESCE((SELECT created_at_ms FROM generations WHERE generation_id = ?1), ?3), ?4)",
            (
                generation_id.as_str(),
                "published",
                now_ms.cast_signed(),
                now_ms.cast_signed(),
            ),
        )
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "generation_catalog_write_failed"),
                format!("generation catalog write failed: {error}"),
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata("path", layout.catalog_db().display().to_string())
        })?;
    transaction
        .execute(
            "INSERT INTO active_generation (singleton, generation_id)
             VALUES (1, ?1)
             ON CONFLICT(singleton) DO UPDATE SET generation_id = excluded.generation_id",
            [generation_id.as_str()],
        )
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "generation_catalog_write_failed"),
                format!("active generation catalog write failed: {error}"),
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata("path", layout.catalog_db().display().to_string())
        })?;

    transaction.commit().map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "generation_catalog_commit_failed"),
            format!("generation catalog commit failed: {error}"),
            ErrorClass::NonRetriable,
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("path", layout.catalog_db().display().to_string())
    })?;

    Ok(())
}

fn upsert_generation_state(
    collection_name: &CollectionName,
    layout: &CollectionGenerationPaths,
    generation_id: &GenerationId,
    state: &str,
) -> Result<()> {
    let mut connection = open_catalog(layout.catalog_db().as_path(), collection_name)?;
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "generation_catalog_transaction_failed"),
                format!("generation catalog transaction start failed: {error}"),
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata("path", layout.catalog_db().display().to_string())
        })?;
    let now_ms = now_epoch_ms()?;
    transaction
        .execute(
            "INSERT OR REPLACE INTO generations (generation_id, state, created_at_ms, updated_at_ms)
             VALUES (?1, ?2, COALESCE((SELECT created_at_ms FROM generations WHERE generation_id = ?1), ?3), ?4)",
            (
                generation_id.as_str(),
                state,
                now_ms.cast_signed(),
                now_ms.cast_signed(),
            ),
        )
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "generation_catalog_write_failed"),
                format!("generation catalog write failed: {error}"),
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata("path", layout.catalog_db().display().to_string())
        })?;
    transaction.commit().map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "generation_catalog_commit_failed"),
            format!("generation catalog commit failed: {error}"),
            ErrorClass::NonRetriable,
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("path", layout.catalog_db().display().to_string())
    })?;
    Ok(())
}

pub(super) fn upsert_dfrr_ready_state(
    collection_name: &CollectionName,
    layout: &CollectionGenerationPaths,
    generation_id: &GenerationId,
    ready_state_fingerprint: &str,
    state: &str,
    artifact_root: &Path,
    config_json: &str,
) -> Result<()> {
    let mut connection = open_catalog(layout.catalog_db().as_path(), collection_name)?;
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "generation_catalog_transaction_failed"),
                format!("generation catalog transaction start failed: {error}"),
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata("path", layout.catalog_db().display().to_string())
        })?;
    let now_ms = now_epoch_ms()?;
    transaction
        .execute(
            "INSERT OR REPLACE INTO dfrr_ready_states (
                generation_id,
                ready_state_fingerprint,
                state,
                artifact_root,
                config_json,
                updated_at_ms
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            (
                generation_id.as_str(),
                ready_state_fingerprint,
                state,
                artifact_root.display().to_string(),
                config_json,
                now_ms.cast_signed(),
            ),
        )
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "generation_catalog_write_failed"),
                format!("DFRR ready-state catalog write failed: {error}"),
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata("path", layout.catalog_db().display().to_string())
            .with_metadata("generationId", generation_id.as_str().to_string())
            .with_metadata("readyStateFingerprint", ready_state_fingerprint.to_string())
        })?;
    transaction.commit().map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "generation_catalog_commit_failed"),
            format!("generation catalog commit failed: {error}"),
            ErrorClass::NonRetriable,
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("path", layout.catalog_db().display().to_string())
    })?;
    Ok(())
}

pub(super) fn has_ready_dfrr_state(
    collection_name: &CollectionName,
    layout: &CollectionGenerationPaths,
    generation_id: &GenerationId,
    ready_state_fingerprint: &str,
) -> Result<bool> {
    let connection = open_catalog(layout.catalog_db().as_path(), collection_name)?;
    connection
        .execute_batch(
            "CREATE TABLE IF NOT EXISTS dfrr_ready_states (
                generation_id TEXT NOT NULL,
                ready_state_fingerprint TEXT NOT NULL,
                state TEXT NOT NULL,
                artifact_root TEXT NOT NULL,
                config_json TEXT NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                PRIMARY KEY (generation_id, ready_state_fingerprint)
            );",
        )
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "generation_catalog_schema_failed"),
                format!("DFRR ready-state catalog schema setup failed: {error}"),
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata("path", layout.catalog_db().display().to_string())
        })?;
    let state = connection
        .query_row(
            "SELECT state FROM dfrr_ready_states
             WHERE generation_id = ?1 AND ready_state_fingerprint = ?2",
            (generation_id.as_str(), ready_state_fingerprint),
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "generation_catalog_query_failed"),
                format!("DFRR ready-state catalog query failed: {error}"),
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata("path", layout.catalog_db().display().to_string())
            .with_metadata("generationId", generation_id.as_str().to_string())
            .with_metadata("readyStateFingerprint", ready_state_fingerprint.to_string())
        })?;
    Ok(state.as_deref() == Some("ready"))
}

fn open_catalog(path: &Path, collection_name: &CollectionName) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "generation_catalog_dir_create_failed"),
                "failed to create generation catalog directory",
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata("path", parent.display().to_string())
            .with_metadata("source", source.to_string())
        })?;
    }

    let connection = Connection::open(path).map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "generation_catalog_open_failed"),
            format!("generation catalog open failed: {error}"),
            ErrorClass::NonRetriable,
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("path", path.display().to_string())
    })?;
    connection
        .busy_timeout(SQLITE_BUSY_TIMEOUT)
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "generation_catalog_busy_timeout_failed"),
                format!("generation catalog busy-timeout failed: {error}"),
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata("path", path.display().to_string())
        })?;
    connection
        .pragma_update(None, "journal_mode", "WAL")
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "generation_catalog_wal_failed"),
                format!("generation catalog WAL setup failed: {error}"),
                ErrorClass::NonRetriable,
            )
            .with_metadata("collection", collection_name.as_str().to_string())
            .with_metadata("path", path.display().to_string())
        })?;
    Ok(connection)
}

fn build_coordinator_channel_closed_error(collection_name: &CollectionName) -> ErrorEnvelope {
    ErrorEnvelope::unexpected(
        ErrorCode::new("vector", "build_coordinator_channel_closed"),
        "collection build coordinator channel closed",
        ErrorClass::NonRetriable,
    )
    .with_metadata("collection", collection_name.as_str().to_string())
}

fn map_join_error(
    join_error: &tokio::task::JoinError,
    operation: &str,
    collection_name: &CollectionName,
) -> ErrorEnvelope {
    if join_error.is_panic() {
        ErrorEnvelope::invariant(
            ErrorCode::new("vector", "build_coordinator_task_panicked"),
            format!("{operation} task panicked"),
        )
        .with_metadata("collection", collection_name.as_str().to_string())
        .with_metadata("source", join_error.to_string())
    } else {
        ErrorEnvelope::cancelled(format!("{operation} task cancelled"))
            .with_metadata("collection", collection_name.as_str().to_string())
    }
}

fn next_generation_id() -> Result<GenerationId> {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let millis = now_epoch_ms()?;
    let seq = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    GenerationId::new(format!("gen-{millis}-{seq:06}"))
}

fn now_epoch_ms() -> Result<u64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "generation_time_failed"),
                format!("failed to compute generation timestamp: {error}"),
                ErrorClass::NonRetriable,
            )
        })?;
    u64::try_from(duration.as_millis()).map_err(|_| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "generation_time_overflow"),
            "generation timestamp overflow",
            ErrorClass::NonRetriable,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::{
        CollectionBuildCoordinatorActor, CollectionSnapshot, validate_generation_publish_inputs,
        write_active_generation_pointer,
    };
    use crate::vectordb_local::CollectionRecord;
    use semantic_code_domain::{CollectionName, LineSpan};
    use semantic_code_ports::VectorDocumentMetadata;
    use semantic_code_shared::{
        CancellationToken, ErrorClass, ErrorCode, ErrorEnvelope, Result as SharedResult,
    };
    use semantic_code_vector::{CollectionGenerationPaths, GenerationId};
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

    fn sample_metadata(path: &str) -> SharedResult<VectorDocumentMetadata> {
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

    #[tokio::test]
    async fn build_coordinator_initializes_generation_catalog_and_root() -> SharedResult<()> {
        let temp = TempDir::create("build-coordinator-scaffold").map_err(ErrorEnvelope::from)?;
        let collection = CollectionName::parse("build_registry")?;
        let layout = CollectionGenerationPaths::new(temp.path().join(collection.as_str()));
        let (handle, join) = CollectionBuildCoordinatorActor::spawn(CancellationToken::new());

        handle
            .ensure_scaffold(collection.clone(), layout.clone())
            .await?;

        assert!(layout.catalog_db().is_file());
        assert!(layout.generations_dir().is_dir());

        drop(handle);
        join.abort();
        let _ = join.await;
        Ok(())
    }

    #[tokio::test]
    async fn build_coordinator_drop_removes_generation_root() -> SharedResult<()> {
        let temp = TempDir::create("build-coordinator-drop").map_err(ErrorEnvelope::from)?;
        let collection = CollectionName::parse("build_registry_drop")?;
        let layout = CollectionGenerationPaths::new(temp.path().join(collection.as_str()));
        let (handle, join) = CollectionBuildCoordinatorActor::spawn(CancellationToken::new());

        handle
            .ensure_scaffold(collection.clone(), layout.clone())
            .await?;
        handle
            .drop_collection(collection.clone(), layout.clone())
            .await?;

        assert!(!layout.root().exists());

        drop(handle);
        join.abort();
        let _ = join.await;
        Ok(())
    }

    #[tokio::test]
    async fn build_coordinator_close_session_blocks_until_admissions_drop() -> SharedResult<()> {
        let temp = TempDir::create("build-coordinator-close").map_err(ErrorEnvelope::from)?;
        let collection = CollectionName::parse("build_registry_close")?;
        let layout = CollectionGenerationPaths::new(temp.path().join(collection.as_str()));
        let (handle, join) = CollectionBuildCoordinatorActor::spawn(CancellationToken::new());

        handle
            .ensure_scaffold(collection.clone(), layout.clone())
            .await?;
        let admission = handle.begin_journal_append(collection.clone()).await?;
        let close = handle.close_session(collection.clone()).await?;

        let mut wait = tokio::spawn(async move {
            close.wait().await;
        });
        let timed = tokio::time::timeout(std::time::Duration::from_millis(50), &mut wait).await;
        assert!(
            timed.is_err(),
            "close wait should block while admissions are live"
        );

        drop(admission);
        wait.await
            .expect("close wait should complete once admissions drop");

        drop(handle);
        join.abort();
        let _ = join.await;
        Ok(())
    }

    #[tokio::test]
    async fn build_coordinator_rejects_new_admissions_after_close() -> SharedResult<()> {
        let temp =
            TempDir::create("build-coordinator-admission-close").map_err(ErrorEnvelope::from)?;
        let collection = CollectionName::parse("build_registry_admission_close")?;
        let layout = CollectionGenerationPaths::new(temp.path().join(collection.as_str()));
        let (handle, join) = CollectionBuildCoordinatorActor::spawn(CancellationToken::new());

        handle
            .ensure_scaffold(collection.clone(), layout.clone())
            .await?;
        handle.close_session(collection.clone()).await?.wait().await;

        let error = handle
            .begin_journal_append(collection.clone())
            .await
            .expect_err("admission after close must fail");
        assert_eq!(
            error.code,
            ErrorCode::new("vector", "build_session_state_invalid")
        );

        drop(handle);
        join.abort();
        let _ = join.await;
        Ok(())
    }

    #[tokio::test]
    async fn build_coordinator_publish_base_generation_updates_active_catalog() -> SharedResult<()>
    {
        let temp = TempDir::create("build-coordinator-publish").map_err(ErrorEnvelope::from)?;
        let collection = CollectionName::parse("build_registry_publish")?;
        let layout = CollectionGenerationPaths::new(temp.path().join(collection.as_str()));
        let (handle, join) = CollectionBuildCoordinatorActor::spawn(CancellationToken::new());

        handle
            .ensure_scaffold(collection.clone(), layout.clone())
            .await?;
        handle.close_session(collection.clone()).await?.wait().await;
        let generation_id = handle
            .stage_base_generation(
                collection.clone(),
                layout.clone(),
                semantic_code_vector::ExactVectorRows::new(
                    2,
                    vec![
                        semantic_code_vector::ExactVectorRow::new(
                            "a",
                            semantic_code_vector::OriginId::from_usize(0),
                            vec![1.0, 0.0],
                        ),
                        semantic_code_vector::ExactVectorRow::new(
                            "b",
                            semantic_code_vector::OriginId::from_usize(1),
                            vec![0.0, 1.0],
                        ),
                    ],
                )?,
                CollectionSnapshot {
                    version: 1,
                    dimension: 2,
                    index_mode: semantic_code_domain::IndexMode::Dense,
                    records: vec![
                        CollectionRecord {
                            id: "a".into(),
                            vector: vec![1.0, 0.0],
                            content: "alpha".into(),
                            metadata: sample_metadata("src/a.rs")?,
                        },
                        CollectionRecord {
                            id: "b".into(),
                            vector: vec![0.0, 1.0],
                            content: "beta".into(),
                            metadata: sample_metadata("src/b.rs")?,
                        },
                    ],
                    checkpoint_sequence: Some(2),
                },
            )
            .await?;
        assert!(
            !layout.active_file().exists(),
            "staging should not flip ACTIVE before activation"
        );
        handle
            .activate_generation(collection.clone(), layout.clone(), generation_id.clone())
            .await?;

        assert_eq!(
            std::fs::read_to_string(layout.active_file()).map_err(ErrorEnvelope::from)?,
            generation_id.as_str()
        );
        assert!(
            layout
                .generation(&generation_id)
                .base_dir()
                .join(semantic_code_vector::EXACT_GENERATION_META_FILE_NAME)
                .is_file()
        );

        let connection = super::open_catalog(layout.catalog_db().as_path(), &collection)?;
        let active_generation: String = connection
            .query_row(
                "SELECT generation_id FROM active_generation WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .map_err(|error| {
                ErrorEnvelope::unexpected(
                    ErrorCode::new("vector", "generation_catalog_query_failed"),
                    format!("generation catalog query failed: {error}"),
                    ErrorClass::NonRetriable,
                )
                .with_metadata("collection", collection.as_str().to_string())
                .with_metadata("path", layout.catalog_db().display().to_string())
            })?;
        assert_eq!(active_generation, generation_id.as_str());

        drop(handle);
        join.abort();
        let _ = join.await;
        Ok(())
    }

    #[test]
    fn write_active_generation_pointer_replaces_file_without_temp_leaks() -> SharedResult<()> {
        let temp = TempDir::create("active-generation-pointer").map_err(ErrorEnvelope::from)?;
        let collection = CollectionName::parse("active_pointer")?;
        let layout = CollectionGenerationPaths::new(temp.path().join(collection.as_str()));
        let first = GenerationId::new("gen-1")?;
        let second = GenerationId::new("gen-2")?;

        write_active_generation_pointer(&collection, layout.active_file().as_path(), &first)?;
        write_active_generation_pointer(&collection, layout.active_file().as_path(), &second)?;

        assert_eq!(
            std::fs::read_to_string(layout.active_file()).map_err(ErrorEnvelope::from)?,
            second.as_str()
        );
        let temp_entries = std::fs::read_dir(layout.root())
            .map_err(ErrorEnvelope::from)?
            .map(|entry| entry.map(|entry| entry.file_name()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(ErrorEnvelope::from)?;
        assert!(
            temp_entries
                .iter()
                .all(|name| !name.to_string_lossy().starts_with(".ACTIVE.tmp-")),
            "temporary ACTIVE pointer files should not be left behind"
        );
        Ok(())
    }

    #[tokio::test]
    async fn build_coordinator_rejects_duplicate_sidecar_ids_before_activation() -> SharedResult<()>
    {
        let temp =
            TempDir::create("build-coordinator-duplicate-sidecar").map_err(ErrorEnvelope::from)?;
        let collection = CollectionName::parse("build_registry_duplicate_sidecar")?;
        let layout = CollectionGenerationPaths::new(temp.path().join(collection.as_str()));
        let (handle, join) = CollectionBuildCoordinatorActor::spawn(CancellationToken::new());

        handle
            .ensure_scaffold(collection.clone(), layout.clone())
            .await?;
        handle.close_session(collection.clone()).await?.wait().await;
        let error = handle
            .stage_base_generation(
                collection.clone(),
                layout.clone(),
                semantic_code_vector::ExactVectorRows::new(
                    2,
                    vec![
                        semantic_code_vector::ExactVectorRow::new(
                            "a",
                            semantic_code_vector::OriginId::from_usize(0),
                            vec![1.0, 0.0],
                        ),
                        semantic_code_vector::ExactVectorRow::new(
                            "b",
                            semantic_code_vector::OriginId::from_usize(1),
                            vec![0.0, 1.0],
                        ),
                    ],
                )?,
                CollectionSnapshot {
                    version: 1,
                    dimension: 2,
                    index_mode: semantic_code_domain::IndexMode::Dense,
                    records: vec![
                        CollectionRecord {
                            id: "dup".into(),
                            vector: vec![1.0, 0.0],
                            content: "alpha".into(),
                            metadata: sample_metadata("src/a.rs")?,
                        },
                        CollectionRecord {
                            id: "dup".into(),
                            vector: vec![0.0, 1.0],
                            content: "beta".into(),
                            metadata: sample_metadata("src/b.rs")?,
                        },
                    ],
                    checkpoint_sequence: Some(2),
                },
            )
            .await
            .expect_err("duplicate sidecar ids must fail before activation");

        assert_eq!(
            error.code,
            ErrorCode::new("vector", "generation_publish_duplicate_sidecar_id")
        );
        let report_path = error
            .metadata
            .get("duplicateReportPath")
            .expect("duplicate report path metadata should be present");
        assert!(
            Path::new(report_path).is_file(),
            "duplicate report should be written for duplicate sidecar ids"
        );
        assert!(
            !layout.active_file().exists(),
            "failed stage must not write ACTIVE"
        );

        drop(handle);
        join.abort();
        let _ = join.await;
        Ok(())
    }

    #[test]
    fn validate_generation_publish_inputs_rejects_duplicate_exact_row_ids() -> SharedResult<()> {
        let collection = CollectionName::parse("duplicate_exact_rows")?;
        let rows = semantic_code_vector::ExactVectorRows::new(
            2,
            vec![
                semantic_code_vector::ExactVectorRow::new(
                    "dup",
                    semantic_code_vector::OriginId::from_usize(0),
                    vec![1.0, 0.0],
                ),
                semantic_code_vector::ExactVectorRow::new(
                    "dup",
                    semantic_code_vector::OriginId::from_usize(1),
                    vec![0.0, 1.0],
                ),
            ],
        )?;
        let snapshot = CollectionSnapshot {
            version: 1,
            dimension: 2,
            index_mode: semantic_code_domain::IndexMode::Dense,
            records: vec![
                CollectionRecord {
                    id: "dup".into(),
                    vector: vec![1.0, 0.0],
                    content: "alpha".into(),
                    metadata: sample_metadata("src/a.rs")?,
                },
                CollectionRecord {
                    id: "other".into(),
                    vector: vec![0.0, 1.0],
                    content: "beta".into(),
                    metadata: sample_metadata("src/b.rs")?,
                },
            ],
            checkpoint_sequence: None,
        };
        let report_path = std::env::temp_dir().join(format!(
            "duplicate-exact-report-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));

        let error = validate_generation_publish_inputs(&collection, &rows, &snapshot, &report_path)
            .expect_err("duplicate exact ids must fail");
        assert_eq!(
            error.code,
            ErrorCode::new("vector", "generation_publish_duplicate_exact_row_id")
        );
        assert!(
            report_path.is_file(),
            "duplicate exact-row report should be written"
        );
        let _ = std::fs::remove_file(report_path);
        Ok(())
    }
}
