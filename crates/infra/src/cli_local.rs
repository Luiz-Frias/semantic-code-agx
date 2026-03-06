//! Local CLI orchestration helpers.

use crate::calibration_persist::{apply_ema_observation, clear_ema_sidecars};
use crate::cli_calibration::{read_calibration, write_calibration};
use crate::cli_manifest::{
    CliManifest, append_context_gitignore, config_path as context_config_path,
    ensure_default_config, read_manifest, touch_manifest, write_manifest,
};
use crate::embedding_factory::build_embedding_port_with_telemetry;
use crate::vectordb_factory::{build_local_kernel, build_vectordb_port};
use crate::{InfraError, InfraResult};
use semantic_code_adapters::{
    IgnoreMatcher, JsonLogger, JsonTelemetry, LocalCalibrationAdapter, LocalFileSync,
    LocalFileSystem, LocalPathPolicy, StderrLogSink, TaggedTelemetry, TreeSitterSplitter,
};
use semantic_code_app::{
    CalibrateBq1Deps, CalibrateBq1Input, ClearIndexDeps, ClearIndexInput, IndexCodebaseDeps,
    IndexCodebaseInput, IndexCodebaseOutput, IndexProgress, ReindexByChangeDeps,
    ReindexByChangeInput, ReindexByChangeOutput, SemanticSearchDeps, SemanticSearchInput,
    SemanticSearchOutput, calibrate_bq1, clear_index, index_codebase, reindex_by_change,
    semantic_search,
};
use semantic_code_config::{
    BackendConfig, RuntimeEnv, SnapshotStorageMode, ValidatedBackendConfig,
    ValidatedClearIndexRequest, ValidatedIndexRequest, ValidatedReindexByChangeRequest,
    ValidatedSearchRequest, VectorSearchStrategy, load_backend_config_from_path,
    load_backend_config_std_env, load_runtime_env_std_env, to_pretty_toml,
};
use semantic_code_domain::{
    CalibrationParams, CalibrationState, CollectionName, CollectionNamingInput, IndexMode,
    derive_collection_name,
};
use semantic_code_ports::{LogFields, LogLevel, LoggerPort, TelemetryPort, TelemetryTags};
use semantic_code_shared::{
    BoundedU32, ErrorClass, ErrorCode, ErrorEnvelope, REDACTED_VALUE, RequestContext,
};
use semantic_code_vector::VectorSearchBackend;
use serde_json::Value;
use std::collections::BTreeMap;
use std::future::Future;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

const DEFAULT_CHUNK_LIMIT: NonZeroUsize = NonZeroUsize::MAX;
const VECTOR_SNAPSHOT_DIR: &str = "vector";
const VECTOR_COLLECTIONS_DIR: &str = "collections";
const SYNC_SNAPSHOT_DIR: &str = "sync";
const SNAPSHOT_FILE_EXT: &str = "json";

/// Summary of local CLI status information.
#[derive(Debug, Clone)]
pub struct CliStatus {
    /// Manifest associated with the local codebase.
    pub manifest: CliManifest,
    /// Vector snapshot status.
    pub vector_snapshot: SnapshotStatus,
    /// Sync snapshot status.
    pub sync_snapshot: SnapshotStatus,
    /// Effective config summary.
    pub config: CliConfigSummary,
}

/// Summary of init command results.
#[derive(Debug, Clone)]
pub struct CliInitStatus {
    /// Config path used for initialization.
    pub config_path: PathBuf,
    /// Manifest path used for initialization.
    pub manifest_path: PathBuf,
    /// Whether the config file was created or overwritten.
    pub created_config: bool,
    /// Whether the manifest was created or overwritten.
    pub created_manifest: bool,
}

/// Snapshot information for CLI status output.
#[derive(Debug, Clone)]
pub struct SnapshotStatus {
    /// Snapshot path if storage is enabled.
    pub path: Option<PathBuf>,
    /// Whether the snapshot file exists.
    pub exists: bool,
    /// Last modified timestamp (ms since epoch) when available.
    pub updated_at_ms: Option<u64>,
    /// Record count when available (vector snapshots only).
    pub record_count: Option<usize>,
}

/// Minimal config summary for CLI status output.
#[derive(Debug, Clone)]
pub struct CliConfigSummary {
    /// Index mode from config.
    pub index_mode: IndexMode,
    /// Snapshot storage mode from config.
    pub snapshot_storage: SnapshotStorageMode,
    /// Embedding dimension override, if provided.
    pub embedding_dimension: Option<u32>,
    /// Embedding cache enabled.
    pub embedding_cache_enabled: bool,
    /// Disk cache enabled.
    pub embedding_cache_disk_enabled: bool,
    /// Cache max entries (memory).
    pub embedding_cache_max_entries: u32,
    /// Cache max bytes (memory).
    pub embedding_cache_max_bytes: u64,
    /// Cache disk path, if configured.
    pub embedding_cache_disk_path: Option<Box<str>>,
    /// Cache disk provider.
    pub embedding_cache_disk_provider: Option<Box<str>>,
    /// Cache disk connection (redacted when present).
    pub embedding_cache_disk_connection: Option<Box<str>>,
    /// Cache disk table name.
    pub embedding_cache_disk_table: Option<Box<str>>,
    /// Cache disk max bytes.
    pub embedding_cache_disk_max_bytes: u64,
    /// Retry max attempts.
    pub retry_max_attempts: u32,
    /// Retry base delay (ms).
    pub retry_base_delay_ms: u64,
    /// Retry max delay (ms).
    pub retry_max_delay_ms: u64,
    /// Retry jitter ratio (percent).
    pub retry_jitter_ratio_pct: u32,
    /// Max in-flight file tasks.
    pub max_in_flight_files: Option<u32>,
    /// Max in-flight embedding batches.
    pub max_in_flight_embedding_batches: Option<u32>,
    /// Max in-flight insert batches.
    pub max_in_flight_inserts: Option<u32>,
    /// Max buffered chunks.
    pub max_buffered_chunks: Option<u32>,
    /// Max buffered embeddings.
    pub max_buffered_embeddings: Option<u32>,
}

/// Run a local index operation.
pub fn run_index_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    request: &ValidatedIndexRequest,
    init_if_missing: bool,
) -> InfraResult<IndexCodebaseOutput> {
    run_index_local_with_progress(
        config_path,
        overrides_json,
        request,
        init_if_missing,
        None,
        None,
    )
}

fn build_index_input(
    config: &ValidatedBackendConfig,
    manifest: &CliManifest,
    request: &ValidatedIndexRequest,
    on_progress: Option<Arc<dyn Fn(IndexProgress) + Send + Sync>>,
) -> InfraResult<IndexCodebaseInput> {
    let request = request.as_ref();
    Ok(IndexCodebaseInput {
        codebase_root: request.codebase_root.clone(),
        collection_name: request
            .collection_name
            .clone()
            .unwrap_or_else(|| manifest.collection_name.clone()),
        index_mode: manifest.index_mode,
        supported_extensions: Some(config.sync.allowed_extensions.clone()),
        ignore_patterns: Some(config.sync.ignore_patterns.clone()),
        file_list: None,
        force_reindex: request.force_reindex,
        on_progress,
        embedding_batch_size: nonzero_usize_from_u32(
            config.limits().embedding_batch_size.get(),
            "embedding batch size",
        )?,
        chunk_limit: DEFAULT_CHUNK_LIMIT,
        max_files: Some(nonzero_usize_from_u32(
            config.limits().sync_max_files.get(),
            "sync max files",
        )?),
        max_file_size_bytes: Some(config.limits().sync_max_file_size_bytes.get()),
        max_buffered_chunks: opt_nonzero_usize_from_u32(
            config
                .limits()
                .core_max_buffered_chunks
                .map(BoundedU32::get),
            "core max buffered chunks",
        )?,
        max_buffered_embeddings: opt_nonzero_usize_from_u32(
            config
                .limits()
                .core_max_buffered_embeddings
                .map(BoundedU32::get),
            "core max buffered embeddings",
        )?,
        max_in_flight_files: opt_nonzero_usize_from_u32(
            config
                .limits()
                .core_max_in_flight_files
                .map(BoundedU32::get),
            "core max in-flight files",
        )?,
        max_in_flight_embedding_batches: opt_nonzero_usize_from_u32(
            config
                .limits()
                .core_max_in_flight_embedding_batches
                .map(BoundedU32::get),
            "core max in-flight embedding batches",
        )?,
        max_in_flight_inserts: opt_nonzero_usize_from_u32(
            config
                .limits()
                .core_max_in_flight_inserts
                .map(BoundedU32::get),
            "core max in-flight inserts",
        )?,
    })
}

fn build_splitter(config: &ValidatedBackendConfig) -> InfraResult<TreeSitterSplitter> {
    let splitter = TreeSitterSplitter::default();
    splitter.set_max_chunk_chars(usize_from_u32(
        config.limits().core_max_chunk_chars.get(),
        "core max chunk chars",
    )?);
    Ok(splitter)
}

/// Run a local index operation with optional progress and cancellation.
#[tracing::instrument(
    name = "cli.index.local_with_progress",
    skip_all,
    fields(
        correlation_id = tracing::field::Empty,
        has_config_path = config_path.is_some(),
        has_overrides = overrides_json.is_some(),
        init_if_missing = init_if_missing,
        has_progress_callback = on_progress.is_some(),
        has_cancel_path = cancel_path.is_some(),
    )
)]
pub fn run_index_local_with_progress(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    request: &ValidatedIndexRequest,
    init_if_missing: bool,
    on_progress: Option<Arc<dyn Fn(IndexProgress) + Send + Sync>>,
    cancel_path: Option<PathBuf>,
) -> InfraResult<IndexCodebaseOutput> {
    let codebase_root = request.as_ref().codebase_root.as_path();
    let config_path = resolve_config_path(config_path, codebase_root);
    let (config, env) = load_config_with_env(config_path.as_deref(), overrides_json)?;
    let manifest = ensure_manifest(codebase_root, &config, init_if_missing)?;
    let observability = observability_from_env();
    let ctx = RequestContext::new_request();
    tracing::Span::current().record("correlation_id", ctx.correlation_id().as_str());
    let scoped_logger = scope_logger(observability.logger.as_ref(), &ctx);
    let scoped_telemetry = scope_telemetry(observability.telemetry.as_ref(), &ctx);
    let embedding = build_embedding_port_with_telemetry(
        &config,
        &env,
        codebase_root,
        scoped_telemetry.clone(),
    )?;
    let input = build_index_input(&config, &manifest, request, on_progress)?;

    // Capture auto-calibrate flag before config is moved into the async closure.
    let should_auto_calibrate = config.vector_db.effective_vector_kernel()
        == semantic_code_config::VectorKernelKind::Dfrr
        && config
            .vector_db
            .dfrr_search
            .as_ref()
            .is_some_and(|d| d.auto_calibrate);

    let snapshot_storage = manifest.snapshot_storage.clone();
    let codebase_root = request.as_ref().codebase_root.clone();
    let codebase_root_async = codebase_root.clone();
    let output = run_async_with_ctx(ctx, move |ctx| async move {
        let cancel_handle = spawn_cancel_watcher(
            &ctx,
            cancel_path,
            config.embedding.jobs.cancel_poll_interval_ms,
        );
        let vectordb = build_vectordb_port(&config, &codebase_root_async, snapshot_storage).await?;
        let splitter = build_splitter(&config)?;
        let deps = IndexCodebaseDeps {
            embedding,
            vectordb,
            splitter: Arc::new(splitter),
            filesystem: Arc::new(LocalFileSystem::new(Some(config.sync.max_file_size_bytes))),
            path_policy: Arc::new(LocalPathPolicy::new()),
            ignore: Arc::new(IgnoreMatcher::new()),
            logger: scoped_logger,
            telemetry: scoped_telemetry,
        };
        let collection_name = input.collection_name.clone();
        let result = index_codebase(&ctx, &deps, input).await;
        let result = match result {
            Ok(output) => {
                deps.vectordb.flush(&ctx, collection_name).await?;
                Ok(output)
            },
            Err(err) => Err(err),
        };
        finalize_cancel_watcher(cancel_handle).await?;
        result
    })?;
    touch_manifest(&codebase_root, &manifest)?;

    // Auto-calibrate BQ1 threshold if configured and no calibration file exists.
    if should_auto_calibrate {
        auto_calibrate_after_index(config_path.as_deref(), overrides_json, &codebase_root);
    }

    Ok(output)
}

/// Run a local semantic search.
#[tracing::instrument(
    name = "cli.search.local",
    skip_all,
    fields(
        correlation_id = tracing::field::Empty,
        has_config_path = config_path.is_some(),
        has_overrides = overrides_json.is_some(),
    )
)]
pub fn run_search_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    request: &ValidatedSearchRequest,
) -> InfraResult<SemanticSearchOutput> {
    let request = request.as_ref();
    let codebase_root = request.codebase_root.as_path();
    let config_path = resolve_config_path(config_path, codebase_root);
    let (config, env) = load_config_with_env(config_path.as_deref(), overrides_json)?;
    let manifest = ensure_manifest(codebase_root, &config, false)?;
    let observability = observability_from_env();
    let ctx = RequestContext::new_request();
    tracing::Span::current().record("correlation_id", ctx.correlation_id().as_str());
    let scoped_logger = scope_logger(observability.logger.as_ref(), &ctx);
    let scoped_telemetry = scope_telemetry(observability.telemetry.as_ref(), &ctx);
    let embedding = build_embedding_port_with_telemetry(
        &config,
        &env,
        codebase_root,
        scoped_telemetry.clone(),
    )?;
    let input = SemanticSearchInput {
        codebase_root: codebase_root.to_string_lossy().to_string().into_boxed_str(),
        collection_name: manifest.collection_name.clone(),
        index_mode: manifest.index_mode,
        query: request.query.clone(),
        top_k: request.top_k,
        threshold: request
            .threshold
            .map(|value| f32_from_f64(value, "threshold"))
            .transpose()?,
        query_vector: None,
    };

    let snapshot_storage = manifest.snapshot_storage;
    let codebase_root = codebase_root.to_path_buf();
    run_async_with_ctx(ctx, move |ctx| async move {
        let vectordb = build_vectordb_port(&config, &codebase_root, snapshot_storage).await?;
        let deps = SemanticSearchDeps {
            embedding,
            vectordb,
            logger: scoped_logger,
            telemetry: scoped_telemetry,
        };
        let output = semantic_search(&ctx, &deps, input).await?;

        // Observe BQ1 EMA if calibration exists and search returned DFRR stats.
        if let Some(ref stats) = output.stats {
            observe_ema_if_calibrated(&codebase_root, stats);
        }

        Ok(output)
    })
}

// ── Warm search session for stdin-batch mode ─────────────────────────────────

/// A pre-warmed search session that keeps config, embedding, and vectordb loaded.
///
/// Created via [`open_search_session`]. Each [`search`](LocalSearchSession::search)
/// call reuses the warm context — only query embedding and vector lookup run per call.
pub struct LocalSearchSession {
    deps: SemanticSearchDeps,
    collection_name: CollectionName,
    index_mode: IndexMode,
    codebase_root: Box<str>,
    runtime: tokio::runtime::Runtime,
}

impl LocalSearchSession {
    /// Execute a single search against the warm session.
    pub fn search(
        &self,
        query: &str,
        top_k: Option<u32>,
        threshold: Option<f32>,
    ) -> InfraResult<SemanticSearchOutput> {
        let ctx = RequestContext::new_request();
        let deps = self.deps.clone();
        let input = SemanticSearchInput {
            codebase_root: self.codebase_root.clone(),
            collection_name: self.collection_name.clone(),
            index_mode: self.index_mode,
            query: query.into(),
            top_k,
            threshold,
            query_vector: None,
        };
        self.runtime
            .block_on(async { semantic_search(&ctx, &deps, input).await })
    }

    /// Execute a search using a pre-computed query embedding vector.
    pub fn search_with_vector(
        &self,
        query_label: &str,
        vector: semantic_code_ports::EmbeddingVector,
        top_k: Option<u32>,
        threshold: Option<f32>,
    ) -> InfraResult<SemanticSearchOutput> {
        let ctx = RequestContext::new_request();
        let deps = self.deps.clone();
        let input = SemanticSearchInput {
            codebase_root: self.codebase_root.clone(),
            collection_name: self.collection_name.clone(),
            index_mode: self.index_mode,
            query: query_label.into(),
            top_k,
            threshold,
            query_vector: Some(vector),
        };
        self.runtime
            .block_on(async { semantic_search(&ctx, &deps, input).await })
    }

    /// Embed a query string and return the raw vector.
    pub fn embed(&self, query: &str) -> InfraResult<semantic_code_ports::EmbeddingVector> {
        let ctx = RequestContext::new_request();
        let deps = self.deps.clone();
        self.runtime.block_on(async {
            ctx.ensure_not_cancelled("embed")?;
            deps.embedding.embed(&ctx, query.into()).await
        })
    }
}

/// Open a warm search session by loading config, embedding, and vectordb once.
///
/// The returned [`LocalSearchSession`] can be used to run multiple searches
/// without re-loading the index or rebuilding the HNSW graph.
#[tracing::instrument(
    name = "infra.open_search_session",
    skip_all,
    fields(
        has_config_path = config_path.is_some(),
        has_overrides_json = overrides_json.is_some()
    )
)]
pub fn open_search_session(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    codebase_root: &Path,
) -> InfraResult<LocalSearchSession> {
    let config_path = resolve_config_path(config_path, codebase_root);
    let (config, env) = load_config_with_env(config_path.as_deref(), overrides_json)?;
    let manifest = ensure_manifest(codebase_root, &config, false)?;
    let observability = observability_from_env();
    let ctx = RequestContext::new_request();
    let scoped_logger = scope_logger(observability.logger.as_ref(), &ctx);
    let scoped_telemetry = scope_telemetry(observability.telemetry.as_ref(), &ctx);
    let embedding = build_embedding_port_with_telemetry(
        &config,
        &env,
        codebase_root,
        scoped_telemetry.clone(),
    )?;

    let snapshot_storage = manifest.snapshot_storage;
    let codebase_root_buf = codebase_root.to_path_buf();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(InfraError::from)?;

    let vectordb = runtime.block_on(async {
        build_vectordb_port(&config, &codebase_root_buf, snapshot_storage).await
    })?;

    let deps = SemanticSearchDeps {
        embedding,
        vectordb,
        logger: scoped_logger,
        telemetry: scoped_telemetry,
    };

    tracing::info!("search session opened; index loaded and warm");

    Ok(LocalSearchSession {
        deps,
        collection_name: manifest.collection_name,
        index_mode: manifest.index_mode,
        codebase_root: codebase_root.to_string_lossy().to_string().into_boxed_str(),
        runtime,
    })
}

/// Observe BQ1 EMA if a calibration sidecar exists and the stats contain DFRR metrics.
///
/// Best-effort: silently ignores read/write errors (non-critical path).
fn observe_ema_if_calibrated(
    codebase_root: &Path,
    search_stats: &semantic_code_domain::SearchStats,
) {
    let Ok(Some(observation)) = apply_ema_observation(codebase_root, search_stats) else {
        return;
    };

    tracing::debug!(
        observed_skip_rate = format_args!("{:.4}", observation.observed_skip_rate),
        ema_skip_rate = format_args!("{:.4}", observation.ema_skip_rate),
        samples_seen = observation.samples_seen,
        "BQ1 EMA observation"
    );

    if observation.drift_detected {
        tracing::warn!(
            calibrated_skip_rate = format_args!("{:.4}", observation.calibrated_skip_rate),
            ema_skip_rate = format_args!("{:.4}", observation.ema_skip_rate),
            "BQ1 skip rate drift detected \u{2014} consider re-running `sca calibrate`"
        );
    }
}

/// Run BQ1 calibration automatically after indexing, if no calibration file exists.
///
/// Best-effort: logs a warning on failure but never propagates errors.
fn auto_calibrate_after_index(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    codebase_root: &Path,
) {
    // Skip if calibration already exists.
    match read_calibration(codebase_root) {
        Ok(Some(_)) => {
            tracing::debug!("auto-calibrate skipped: calibration file already exists");
            return;
        },
        Ok(None) => {},
        Err(error) => {
            tracing::warn!(error = %error, "auto-calibrate: failed to check calibration file");
            return;
        },
    }

    tracing::info!("auto-calibrating BQ1 threshold after index (autoCalibrate = true)");

    let params = CalibrationParams::default();
    match run_calibrate_local(config_path, overrides_json, codebase_root, &params) {
        Ok(state) => {
            tracing::info!(
                threshold = format_args!("{:.4}", state.threshold),
                recall = format_args!("{:.4}", state.recall_at_threshold),
                skip_rate = format_args!("{:.4}", state.skip_rate),
                "auto-calibration complete"
            );
        },
        Err(error) => {
            tracing::warn!(
                error = %error,
                "auto-calibration failed \u{2014} run `sca calibrate` manually"
            );
        },
    }
}

/// Clear the local index and sync snapshot.
#[tracing::instrument(
    name = "cli.clear.local",
    skip_all,
    fields(
        correlation_id = tracing::field::Empty,
        has_config_path = config_path.is_some(),
        has_overrides = overrides_json.is_some(),
    )
)]
pub fn run_clear_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    request: &ValidatedClearIndexRequest,
) -> InfraResult<()> {
    let request = request.as_ref();
    let codebase_root = request.codebase_root.as_path();
    let config_path = resolve_config_path(config_path, codebase_root);
    let config = load_config(config_path.as_deref(), overrides_json)?;
    let manifest = ensure_manifest(codebase_root, &config, false)?;
    let snapshot_storage = manifest.snapshot_storage.clone();
    let file_sync = Arc::new(LocalFileSync::new(
        codebase_root.to_path_buf(),
        snapshot_storage.clone(),
    ));
    let observability = observability_from_env();
    let ctx = RequestContext::new_request();
    tracing::Span::current().record("correlation_id", ctx.correlation_id().as_str());
    let scoped_logger = scope_logger(observability.logger.as_ref(), &ctx);
    let scoped_telemetry = scope_telemetry(observability.telemetry.as_ref(), &ctx);
    let input = ClearIndexInput {
        codebase_root: codebase_root.to_path_buf(),
        collection_name: manifest.collection_name,
    };

    let codebase_root = codebase_root.to_path_buf();
    run_async_with_ctx(ctx, move |ctx| async move {
        let vectordb = build_vectordb_port(&config, &codebase_root, snapshot_storage).await?;
        let deps = ClearIndexDeps {
            vectordb,
            file_sync,
            logger: scoped_logger,
            telemetry: scoped_telemetry,
        };
        clear_index(&ctx, &deps, input).await
    })
}

/// Run BQ1 threshold calibration against the local vector index.
///
/// Loads the kernel and snapshot independently (bypassing `LocalVectorDb`)
/// to build a `LocalCalibrationAdapter`, then runs the binary search
/// calibration and persists the result to `.context/calibration.json`.
#[tracing::instrument(
    name = "cli.calibrate.local",
    skip_all,
    fields(
        correlation_id = tracing::field::Empty,
        has_config_path = config_path.is_some(),
        has_overrides = overrides_json.is_some(),
        target_recall = %params.target_recall,
        num_queries = params.num_queries.value(),
    )
)]
pub fn run_calibrate_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    codebase_root: &Path,
    params: &CalibrationParams,
) -> InfraResult<CalibrationState> {
    let config_path = resolve_config_path(config_path, codebase_root);
    let config = load_config(config_path.as_deref(), overrides_json)?;
    let manifest = ensure_manifest(codebase_root, &config, false)?;
    let observability = observability_from_env();
    let ctx = RequestContext::new_request();
    tracing::Span::current().record("correlation_id", ctx.correlation_id().as_str());
    let scoped_logger = scope_logger(observability.logger.as_ref(), &ctx);
    let scoped_telemetry = scope_telemetry(observability.telemetry.as_ref(), &ctx);

    // Build kernel and resolve snapshot path.
    let kernel_kind = config.vector_db.effective_vector_kernel();
    let kernel = build_local_kernel(
        kernel_kind,
        config.vector_db.hnsw_search.as_ref(),
        config.vector_db.dfrr_search.as_ref(),
    )?;
    let search_backend =
        resolve_vector_search_backend(config.vector_db.effective_search_strategy());
    let snapshot_path = vector_snapshot_path(
        codebase_root,
        &manifest.collection_name,
        &manifest.snapshot_storage,
    )
    .ok_or_else(|| {
        ErrorEnvelope::expected(
            ErrorCode::new("calibration", "no_snapshot_storage"),
            "snapshot storage is disabled; cannot calibrate without a local snapshot",
        )
    })?;

    let params = params.clone();
    let codebase_root = codebase_root.to_path_buf();
    let state = run_async_with_ctx(ctx, move |ctx| async move {
        // Load the snapshot into a VectorIndex directly for calibration.
        let adapter = LocalCalibrationAdapter::load_from_snapshot_path(
            kernel,
            &snapshot_path,
            search_backend,
        )?;

        let input = CalibrateBq1Input { params };
        let deps = CalibrateBq1Deps {
            calibration: Arc::new(adapter),
            logger: scoped_logger,
            telemetry: scoped_telemetry,
        };
        let output = calibrate_bq1(&ctx, &deps, &input).await?;
        Ok(output.state)
    })?;

    // Persist the calibration state to `.context/calibration.json`.
    write_calibration(&codebase_root, &state)?;
    // Calibration invalidates prior EMA sidecars.
    clear_ema_sidecars(&codebase_root)?;
    let calibration_file = crate::cli_calibration::calibration_path(&codebase_root);
    tracing::info!(
        path = %calibration_file.display(),
        threshold = format_args!("{:.4}", state.threshold),
        "calibration state persisted"
    );

    Ok(state)
}

/// Map `VectorSearchStrategy` to the vector-crate `VectorSearchBackend`.
const fn resolve_vector_search_backend(strategy: VectorSearchStrategy) -> VectorSearchBackend {
    match strategy {
        VectorSearchStrategy::F32Hnsw => VectorSearchBackend::F32Hnsw,
        VectorSearchStrategy::U8Exact => VectorSearchBackend::ExperimentalU8Quantized,
        VectorSearchStrategy::U8ThenF32Rerank => VectorSearchBackend::ExperimentalU8ThenF32Rerank,
    }
}

fn build_reindex_input(
    config: &ValidatedBackendConfig,
    manifest: &CliManifest,
    request: &ValidatedReindexByChangeRequest,
    on_progress: Option<Arc<dyn Fn(IndexProgress) + Send + Sync>>,
) -> InfraResult<ReindexByChangeInput> {
    let request = request.as_ref();
    Ok(ReindexByChangeInput {
        codebase_root: request.codebase_root.clone(),
        collection_name: manifest.collection_name.clone(),
        index_mode: manifest.index_mode,
        supported_extensions: Some(config.sync.allowed_extensions.clone()),
        ignore_patterns: Some(config.sync.ignore_patterns.clone()),
        embedding_batch_size: nonzero_usize_from_u32(
            config.limits().embedding_batch_size.get(),
            "embedding batch size",
        )?,
        chunk_limit: DEFAULT_CHUNK_LIMIT,
        max_files: Some(nonzero_usize_from_u32(
            config.limits().sync_max_files.get(),
            "sync max files",
        )?),
        max_file_size_bytes: Some(config.limits().sync_max_file_size_bytes.get()),
        max_buffered_chunks: opt_nonzero_usize_from_u32(
            config
                .limits()
                .core_max_buffered_chunks
                .map(BoundedU32::get),
            "core max buffered chunks",
        )?,
        max_buffered_embeddings: opt_nonzero_usize_from_u32(
            config
                .limits()
                .core_max_buffered_embeddings
                .map(BoundedU32::get),
            "core max buffered embeddings",
        )?,
        max_in_flight_files: opt_nonzero_usize_from_u32(
            config
                .limits()
                .core_max_in_flight_files
                .map(BoundedU32::get),
            "core max in-flight files",
        )?,
        max_in_flight_embedding_batches: opt_nonzero_usize_from_u32(
            config
                .limits()
                .core_max_in_flight_embedding_batches
                .map(BoundedU32::get),
            "core max in-flight embedding batches",
        )?,
        max_in_flight_inserts: opt_nonzero_usize_from_u32(
            config
                .limits()
                .core_max_in_flight_inserts
                .map(BoundedU32::get),
            "core max in-flight inserts",
        )?,
        on_progress,
    })
}

/// Run a local reindex-by-change operation.
pub fn run_reindex_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    request: &ValidatedReindexByChangeRequest,
) -> InfraResult<ReindexByChangeOutput> {
    run_reindex_local_with_progress(config_path, overrides_json, request, None, None)
}

/// Run a local reindex-by-change operation with optional progress and cancellation.
#[tracing::instrument(
    name = "cli.reindex.local_with_progress",
    skip_all,
    fields(
        correlation_id = tracing::field::Empty,
        has_config_path = config_path.is_some(),
        has_overrides = overrides_json.is_some(),
        has_progress_callback = on_progress.is_some(),
        has_cancel_path = cancel_path.is_some(),
    )
)]
pub fn run_reindex_local_with_progress(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    request: &ValidatedReindexByChangeRequest,
    on_progress: Option<Arc<dyn Fn(IndexProgress) + Send + Sync>>,
    cancel_path: Option<PathBuf>,
) -> InfraResult<ReindexByChangeOutput> {
    let codebase_root = request.as_ref().codebase_root.as_path();
    let config_path = resolve_config_path(config_path, codebase_root);
    let (config, env) = load_config_with_env(config_path.as_deref(), overrides_json)?;
    let manifest = ensure_manifest(codebase_root, &config, false)?;
    let observability = observability_from_env();
    let ctx = RequestContext::new_request();
    tracing::Span::current().record("correlation_id", ctx.correlation_id().as_str());
    let scoped_logger = scope_logger(observability.logger.as_ref(), &ctx);
    let scoped_telemetry = scope_telemetry(observability.telemetry.as_ref(), &ctx);
    let embedding = build_embedding_port_with_telemetry(
        &config,
        &env,
        codebase_root,
        scoped_telemetry.clone(),
    )?;
    let input = build_reindex_input(&config, &manifest, request, on_progress)?;

    let snapshot_storage = manifest.snapshot_storage;
    let codebase_root = request.as_ref().codebase_root.clone();
    run_async_with_ctx(ctx, move |ctx| async move {
        let cancel_handle = spawn_cancel_watcher(
            &ctx,
            cancel_path,
            config.embedding.jobs.cancel_poll_interval_ms,
        );
        let vectordb =
            build_vectordb_port(&config, &codebase_root, snapshot_storage.clone()).await?;
        let splitter = build_splitter(&config)?;
        let deps = ReindexByChangeDeps {
            file_sync: Arc::new(LocalFileSync::new(codebase_root, snapshot_storage)),
            vectordb,
            embedding,
            splitter: Arc::new(splitter),
            filesystem: Arc::new(LocalFileSystem::new(Some(config.sync.max_file_size_bytes))),
            path_policy: Arc::new(LocalPathPolicy::new()),
            ignore: Arc::new(IgnoreMatcher::new()),
            logger: scoped_logger,
            telemetry: scoped_telemetry,
        };
        let collection_name = input.collection_name.clone();
        let result = reindex_by_change(&ctx, &deps, input).await;
        let result = match result {
            Ok(output) => {
                deps.vectordb.flush(&ctx, collection_name).await?;
                Ok(output)
            },
            Err(err) => Err(err),
        };
        finalize_cancel_watcher(cancel_handle).await?;
        result
    })
}

/// Initialize config and manifest for a codebase.
pub fn run_init_local(
    config_path: Option<&Path>,
    codebase_root: &Path,
    storage_mode: Option<SnapshotStorageMode>,
    force: bool,
) -> InfraResult<CliInitStatus> {
    let config_path =
        config_path.map_or_else(|| context_config_path(codebase_root), Path::to_path_buf);
    let manifest_path = crate::cli_manifest::manifest_path(codebase_root);

    let created_config = if force || !config_path.is_file() {
        let mut config = BackendConfig::default();
        if let Some(storage_mode) = storage_mode {
            config.vector_db.snapshot_storage = storage_mode;
        }
        let payload = to_pretty_toml(&config)?;
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&config_path, payload)?;
        true
    } else {
        false
    };

    let validated = load_backend_config_from_path(Some(&config_path), None, &BTreeMap::new())?;

    if let Some(existing) = read_manifest(codebase_root)? {
        validate_manifest_root(codebase_root, &existing)?;
        if !force {
            append_context_gitignore(codebase_root)?;
            return Ok(CliInitStatus {
                config_path,
                manifest_path,
                created_config,
                created_manifest: false,
            });
        }
    }

    let collection_name = derive_collection_name(&CollectionNamingInput::new(
        codebase_root.to_path_buf(),
        validated.vector_db.index_mode,
    ))
    .map_err(ErrorEnvelope::from)?;
    let manifest = CliManifest::new(
        codebase_root,
        collection_name.as_str(),
        validated.vector_db.index_mode,
        validated.vector_db.snapshot_storage.clone(),
    )?;
    write_manifest(codebase_root, &manifest)?;
    append_context_gitignore(codebase_root)?;

    Ok(CliInitStatus {
        config_path,
        manifest_path,
        created_config,
        created_manifest: true,
    })
}

/// Read local CLI status information.
pub fn read_status_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    codebase_root: &Path,
) -> InfraResult<CliStatus> {
    let config_path = resolve_config_path(config_path, codebase_root);
    let config = load_config(config_path.as_deref(), overrides_json)?;
    let manifest = ensure_manifest(codebase_root, &config, false)?;
    let vector_snapshot = vector_snapshot_status(codebase_root, &manifest)?;
    let sync_snapshot = sync_snapshot_status(codebase_root, &manifest.snapshot_storage)?;
    let cache_provider = config.embedding.cache.disk_provider.map(cache_provider_str);
    let cache_connection = config
        .embedding
        .cache
        .disk_connection
        .as_ref()
        .map(|_| REDACTED_VALUE.to_string().into_boxed_str());
    let config_summary = CliConfigSummary {
        index_mode: config.vector_db.index_mode,
        snapshot_storage: config.vector_db.snapshot_storage.clone(),
        embedding_dimension: config.embedding.dimension,
        embedding_cache_enabled: config.embedding.cache.enabled,
        embedding_cache_disk_enabled: config.embedding.cache.disk_enabled,
        embedding_cache_max_entries: config.embedding.cache.max_entries,
        embedding_cache_max_bytes: config.embedding.cache.max_bytes,
        embedding_cache_disk_path: config.embedding.cache.disk_path.clone(),
        embedding_cache_disk_provider: cache_provider,
        embedding_cache_disk_connection: cache_connection,
        embedding_cache_disk_table: config.embedding.cache.disk_table.clone(),
        embedding_cache_disk_max_bytes: config.embedding.cache.disk_max_bytes,
        retry_max_attempts: config.core.retry.max_attempts,
        retry_base_delay_ms: config.core.retry.base_delay_ms,
        retry_max_delay_ms: config.core.retry.max_delay_ms,
        retry_jitter_ratio_pct: config.core.retry.jitter_ratio_pct,
        max_in_flight_files: config.core.max_in_flight_files,
        max_in_flight_embedding_batches: config.core.max_in_flight_embedding_batches,
        max_in_flight_inserts: config.core.max_in_flight_inserts,
        max_buffered_chunks: config.core.max_buffered_chunks,
        max_buffered_embeddings: config.core.max_buffered_embeddings,
    };

    Ok(CliStatus {
        manifest,
        vector_snapshot,
        sync_snapshot,
        config: config_summary,
    })
}

fn cache_provider_str(provider: semantic_code_config::EmbeddingCacheDiskProvider) -> Box<str> {
    let value = match provider {
        semantic_code_config::EmbeddingCacheDiskProvider::Sqlite => "sqlite",
        semantic_code_config::EmbeddingCacheDiskProvider::Postgres => "postgres",
        semantic_code_config::EmbeddingCacheDiskProvider::Mysql => "mysql",
        semantic_code_config::EmbeddingCacheDiskProvider::Mssql => "mssql",
    };
    value.to_string().into_boxed_str()
}

fn load_config(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
) -> InfraResult<ValidatedBackendConfig> {
    tracing::debug!(
        has_config_path = config_path.is_some(),
        has_overrides = overrides_json.is_some(),
        "loading backend config"
    );
    load_backend_config_std_env(config_path, overrides_json)
}

fn load_config_with_env(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
) -> InfraResult<(ValidatedBackendConfig, RuntimeEnv)> {
    tracing::debug!(
        has_config_path = config_path.is_some(),
        has_overrides = overrides_json.is_some(),
        "loading backend config with environment overrides"
    );
    let env = load_runtime_env_std_env()?;
    tracing::debug!(
        has_embedding_provider_override = env.embedding_provider.is_some(),
        has_vector_db_provider_override = env.vector_db_provider.is_some(),
        has_embedding_api_key = env.embedding_api_key.is_some(),
        has_openai_api_key = env.openai_api_key.is_some(),
        has_gemini_api_key = env.gemini_api_key.is_some(),
        has_voyage_api_key = env.voyage_api_key.is_some(),
        has_vector_db_token = env.vector_db_token.is_some(),
        has_vector_db_password = env.vector_db_password.is_some(),
        "resolved backend environment override flags"
    );
    let config = load_backend_config_std_env(config_path, overrides_json)?;
    Ok((config, env))
}

fn ensure_manifest(
    codebase_root: &Path,
    config: &ValidatedBackendConfig,
    init_if_missing: bool,
) -> InfraResult<CliManifest> {
    if let Some(manifest) = read_manifest(codebase_root)? {
        validate_manifest_root(codebase_root, &manifest)?;
        return Ok(manifest);
    }

    if !init_if_missing {
        return Err(missing_manifest_error());
    }

    let collection_name = derive_collection_name(&CollectionNamingInput::new(
        codebase_root.to_path_buf(),
        config.vector_db.index_mode,
    ))
    .map_err(ErrorEnvelope::from)?;
    let manifest = CliManifest::new(
        codebase_root,
        collection_name.as_str(),
        config.vector_db.index_mode,
        config.vector_db.snapshot_storage.clone(),
    )?;
    write_manifest(codebase_root, &manifest)?;
    ensure_default_config(codebase_root)?;
    append_context_gitignore(codebase_root)?;
    Ok(manifest)
}

fn resolve_config_path(config_path: Option<&Path>, codebase_root: &Path) -> Option<PathBuf> {
    config_path.map_or_else(
        || {
            let default_path = context_config_path(codebase_root);
            if default_path.exists() {
                Some(default_path)
            } else {
                None
            }
        },
        |path| Some(path.to_path_buf()),
    )
}

fn validate_manifest_root(codebase_root: &Path, manifest: &CliManifest) -> InfraResult<()> {
    let normalized_root = normalize_root(codebase_root);
    if normalized_root != manifest.codebase_root {
        return Err(
            ErrorEnvelope::expected(ErrorCode::invalid_input(), "manifest root mismatch")
                .with_metadata(
                    "expected",
                    manifest.codebase_root.to_string_lossy().to_string(),
                )
                .with_metadata("provided", normalized_root.to_string_lossy().to_string()),
        );
    }
    Ok(())
}

fn missing_manifest_error() -> InfraError {
    ErrorEnvelope::expected(
        ErrorCode::invalid_input(),
        "manifest missing; run `sca index --init`",
    )
}

fn normalize_root(path: &Path) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}

fn vector_snapshot_status(
    codebase_root: &Path,
    manifest: &CliManifest,
) -> InfraResult<SnapshotStatus> {
    let path = vector_snapshot_path(
        codebase_root,
        &manifest.collection_name,
        &manifest.snapshot_storage,
    );
    let Some(path) = path else {
        return Ok(SnapshotStatus {
            path: None,
            exists: false,
            updated_at_ms: None,
            record_count: None,
        });
    };

    if !path.is_file() {
        return Ok(SnapshotStatus {
            path: Some(path),
            exists: false,
            updated_at_ms: None,
            record_count: None,
        });
    }

    Ok(SnapshotStatus {
        path: Some(path.clone()),
        exists: true,
        updated_at_ms: Some(file_mtime_ms(&path)?),
        record_count: Some(read_vector_record_count(&path)?),
    })
}

fn sync_snapshot_status(
    codebase_root: &Path,
    storage_mode: &SnapshotStorageMode,
) -> InfraResult<SnapshotStatus> {
    let path = sync_snapshot_path(codebase_root, storage_mode);
    let Some(path) = path else {
        return Ok(SnapshotStatus {
            path: None,
            exists: false,
            updated_at_ms: None,
            record_count: None,
        });
    };

    if !path.is_file() {
        return Ok(SnapshotStatus {
            path: Some(path),
            exists: false,
            updated_at_ms: None,
            record_count: None,
        });
    }

    Ok(SnapshotStatus {
        path: Some(path.clone()),
        exists: true,
        updated_at_ms: Some(file_mtime_ms(&path)?),
        record_count: None,
    })
}

fn vector_snapshot_path(
    codebase_root: &Path,
    collection_name: &CollectionName,
    storage_mode: &SnapshotStorageMode,
) -> Option<PathBuf> {
    storage_mode.resolve_root(codebase_root).map(|root| {
        root.join(VECTOR_SNAPSHOT_DIR)
            .join(VECTOR_COLLECTIONS_DIR)
            .join(format!(
                "{}.{}",
                collection_name.as_str(),
                SNAPSHOT_FILE_EXT
            ))
    })
}

fn sync_snapshot_path(codebase_root: &Path, storage_mode: &SnapshotStorageMode) -> Option<PathBuf> {
    let root = storage_mode.resolve_root(codebase_root)?;
    let normalized = normalize_root(codebase_root);
    let digest = md5::compute(normalized.to_string_lossy().as_bytes());
    let hash = format!("{digest:x}");
    Some(
        root.join(SYNC_SNAPSHOT_DIR)
            .join(format!("{hash}.{SNAPSHOT_FILE_EXT}")),
    )
}

fn read_vector_record_count(path: &Path) -> InfraResult<usize> {
    let contents = std::fs::read_to_string(path)?;
    let value: Value = serde_json::from_str(&contents).map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::invalid_input(),
            format!("vector snapshot parse failed: {error}"),
            ErrorClass::NonRetriable,
        )
    })?;
    Ok(value
        .get("records")
        .and_then(|records| records.as_array())
        .map_or(0, Vec::len))
}

fn file_mtime_ms(path: &Path) -> InfraResult<u64> {
    let metadata = std::fs::metadata(path)?;
    let modified = metadata.modified().map_err(InfraError::from)?;
    let duration = modified.duration_since(UNIX_EPOCH).map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            format!("mtime error: {error}"),
            ErrorClass::NonRetriable,
        )
    })?;
    u64::try_from(duration.as_millis()).map_err(|_| {
        ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            "mtime overflow",
            ErrorClass::NonRetriable,
        )
    })
}

fn usize_from_u32(value: u32, label: &str) -> InfraResult<usize> {
    usize::try_from(value).map_err(|_| {
        ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            format!("{label} overflow"),
            ErrorClass::NonRetriable,
        )
    })
}

fn nonzero_usize_from_u32(value: u32, label: &str) -> InfraResult<NonZeroUsize> {
    let value = usize_from_u32(value, label)?;
    NonZeroUsize::new(value).ok_or_else(|| {
        ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            format!("{label} must be greater than zero"),
        )
    })
}

fn opt_nonzero_usize_from_u32(
    value: Option<u32>,
    label: &str,
) -> InfraResult<Option<NonZeroUsize>> {
    value
        .map(|value| nonzero_usize_from_u32(value, label))
        .transpose()
}

fn f32_from_f64(value: f64, field: &str) -> InfraResult<f32> {
    if !value.is_finite() {
        return Err(ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            format!("{field} must be finite"),
        ));
    }
    let parsed = value.to_string().parse::<f32>().map_err(|_| {
        ErrorEnvelope::expected(ErrorCode::invalid_input(), format!("{field} out of range"))
    })?;
    if !parsed.is_finite() {
        return Err(ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            format!("{field} out of range"),
        ));
    }
    Ok(parsed)
}

#[derive(Clone)]
struct Observability {
    logger: Option<Arc<dyn LoggerPort>>,
    telemetry: Option<Arc<dyn TelemetryPort>>,
}

const LOG_FORMAT_ENV: &str = "SCA_LOG_FORMAT";
const LOG_LEVEL_ENV: &str = "SCA_LOG_LEVEL";
const TELEMETRY_FORMAT_ENV: &str = "SCA_TELEMETRY_FORMAT";
const TRACE_SAMPLE_RATE_ENV: &str = "SCA_TRACE_SAMPLE_RATE";

fn observability_from_env() -> Observability {
    let log_enabled = env_is_json(LOG_FORMAT_ENV);
    let telemetry_enabled = std::env::var(TELEMETRY_FORMAT_ENV)
        .ok()
        .map_or(log_enabled, |value| value.eq_ignore_ascii_case("json"));
    tracing::debug!(
        log_enabled = log_enabled,
        telemetry_enabled = telemetry_enabled,
        "resolved local observability flags from environment"
    );

    if !log_enabled && !telemetry_enabled {
        return Observability {
            logger: None,
            telemetry: None,
        };
    }

    let min_log_level = parse_log_level();
    let sample_rate = parse_sample_rate();
    let sink: Arc<dyn semantic_code_adapters::LogSink> = Arc::new(StderrLogSink);
    let logger: Option<Arc<dyn LoggerPort>> = if log_enabled {
        Some(Arc::new(
            JsonLogger::new(Arc::clone(&sink)).with_min_level(min_log_level),
        ))
    } else {
        None
    };
    let telemetry: Option<Arc<dyn TelemetryPort>> = if telemetry_enabled {
        Some(Arc::new(
            JsonTelemetry::new(Arc::clone(&sink)).with_span_sample_rate(sample_rate),
        ))
    } else {
        None
    };
    tracing::debug!(
        log_enabled = logger.is_some(),
        telemetry_enabled = telemetry.is_some(),
        ?min_log_level,
        sample_rate = sample_rate,
        "initialized local observability sinks"
    );

    Observability { logger, telemetry }
}

fn env_is_json(key: &str) -> bool {
    std::env::var(key)
        .ok()
        .is_some_and(|value| value.eq_ignore_ascii_case("json"))
}

fn parse_log_level() -> LogLevel {
    let value = std::env::var(LOG_LEVEL_ENV)
        .ok()
        .map(|value| value.to_ascii_lowercase());
    match value.as_deref() {
        Some("debug") => LogLevel::Debug,
        Some("warn") => LogLevel::Warn,
        Some("error") => LogLevel::Error,
        _ => LogLevel::Info,
    }
}

fn parse_sample_rate() -> f64 {
    std::env::var(TRACE_SAMPLE_RATE_ENV)
        .ok()
        .map_or(1.0, |value| {
            value.parse::<f64>().ok().unwrap_or(1.0).clamp(0.0, 1.0)
        })
}

fn scope_logger(
    logger: Option<&Arc<dyn LoggerPort>>,
    ctx: &RequestContext,
) -> Option<Arc<dyn LoggerPort>> {
    let logger = logger?;
    let mut fields = LogFields::new();
    fields.insert(
        "correlationId".to_owned().into_boxed_str(),
        Value::String(ctx.correlation_id().as_str().to_string()),
    );
    Some(Arc::from(logger.child(fields)))
}

fn scope_telemetry(
    telemetry: Option<&Arc<dyn TelemetryPort>>,
    ctx: &RequestContext,
) -> Option<Arc<dyn TelemetryPort>> {
    let telemetry = telemetry?;
    let mut tags = TelemetryTags::new();
    tags.insert(
        "correlationId".to_owned().into_boxed_str(),
        ctx.correlation_id().as_str().to_string().into_boxed_str(),
    );
    Some(Arc::new(TaggedTelemetry::new(Arc::clone(telemetry), tags)))
}

fn run_async_with_ctx<F, T>(
    ctx: RequestContext,
    op: impl FnOnce(RequestContext) -> F,
) -> InfraResult<T>
where
    F: Future<Output = Result<T, ErrorEnvelope>>,
{
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(InfraError::from)?;
    runtime.block_on(async { op(ctx).await })
}

fn spawn_cancel_watcher(
    ctx: &RequestContext,
    cancel_path: Option<PathBuf>,
    poll_interval_ms: u64,
) -> Option<tokio::task::JoinHandle<()>> {
    let cancel_path = cancel_path?;
    let interval = Duration::from_millis(poll_interval_ms.max(1));
    let token = ctx.cancellation_token();
    let handle = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        loop {
            ticker.tick().await;
            if token.is_cancelled() {
                break;
            }
            if matches!(tokio::fs::try_exists(&cancel_path).await, Ok(true)) {
                token.cancel();
                break;
            }
        }
    });
    Some(handle)
}

async fn finalize_cancel_watcher(handle: Option<tokio::task::JoinHandle<()>>) -> InfraResult<()> {
    let Some(handle) = handle else {
        return Ok(());
    };
    handle.abort();
    if let Err(error) = handle.await
        && error.is_panic()
    {
        return Err(ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            format!("cancel watcher failed: {error}"),
            ErrorClass::NonRetriable,
        ));
    }
    Ok(())
}
