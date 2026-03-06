//! Facade-owned DTOs for CLI-facing contracts.

use semantic_code_domain::{
    CollectionName, IndexMode, Language, LineSpan, SearchStats as DomainSearchStats,
};
use semantic_code_shared::ErrorEnvelope;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

/// Build metadata payload used by CLI-facing surfaces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildInfo {
    /// Package name from Cargo metadata.
    pub name: &'static str,
    /// Package version from Cargo metadata.
    pub version: &'static str,
    /// Rust compiler version used to build the binary.
    pub rustc_version: &'static str,
    /// Target triple the binary was compiled for.
    pub target: &'static str,
    /// Build profile (`debug` or `release`).
    pub profile: &'static str,
    /// Optional git hash captured at build time.
    pub git_hash: Option<&'static str>,
    /// Whether the source tree had local changes at build time.
    pub git_dirty: bool,
}

impl From<semantic_code_core::BuildInfo> for BuildInfo {
    fn from(value: semantic_code_core::BuildInfo) -> Self {
        Self {
            name: value.name,
            version: value.version,
            rustc_version: value.rustc_version,
            target: value.target,
            profile: value.profile,
            git_hash: value.git_hash,
            git_dirty: value.git_dirty,
        }
    }
}

/// Facade-owned runtime error envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InfraError(ErrorEnvelope);

impl InfraError {
    /// Borrow the underlying shared error envelope.
    #[must_use]
    pub const fn as_envelope(&self) -> &ErrorEnvelope {
        &self.0
    }
}

impl From<ErrorEnvelope> for InfraError {
    fn from(value: ErrorEnvelope) -> Self {
        Self(value)
    }
}

impl From<InfraError> for ErrorEnvelope {
    fn from(value: InfraError) -> Self {
        value.0
    }
}

impl AsRef<ErrorEnvelope> for InfraError {
    fn as_ref(&self) -> &ErrorEnvelope {
        self.as_envelope()
    }
}

impl fmt::Display for InfraError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::error::Error for InfraError {}

/// Supported request kinds for JSON validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestKind {
    /// Index request.
    Index,
    /// Search request.
    Search,
    /// Reindex-by-change request.
    ReindexByChange,
    /// Clear-index request.
    ClearIndex,
}

impl RequestKind {
    /// Canonical string representation (for CLI/UI).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Index => "index",
            Self::Search => "search",
            Self::ReindexByChange => "reindexByChange",
            Self::ClearIndex => "clearIndex",
        }
    }
}

impl fmt::Display for RequestKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl From<RequestKind> for semantic_code_infra::RequestKind {
    fn from(value: RequestKind) -> Self {
        match value {
            RequestKind::Index => Self::Index,
            RequestKind::Search => Self::Search,
            RequestKind::ReindexByChange => Self::ReindexByChange,
            RequestKind::ClearIndex => Self::ClearIndex,
        }
    }
}

/// Facade-owned snapshot storage mode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum SnapshotStorageMode {
    /// Disable persistence (in-memory only).
    Disabled,
    /// Store under `.context/` inside the codebase root.
    #[default]
    Project,
    /// Store under a custom absolute path.
    Custom(PathBuf),
}

impl SnapshotStorageMode {
    /// Resolve the snapshot root directory for a codebase.
    #[must_use]
    pub fn resolve_root(&self, codebase_root: &Path) -> Option<PathBuf> {
        match self {
            Self::Disabled => None,
            Self::Project => Some(codebase_root.join(".context")),
            Self::Custom(path) => Some(path.clone()),
        }
    }
}

impl From<semantic_code_config::SnapshotStorageMode> for SnapshotStorageMode {
    fn from(value: semantic_code_config::SnapshotStorageMode) -> Self {
        match value {
            semantic_code_config::SnapshotStorageMode::Disabled => Self::Disabled,
            semantic_code_config::SnapshotStorageMode::Project => Self::Project,
            semantic_code_config::SnapshotStorageMode::Custom(path) => Self::Custom(path),
        }
    }
}

impl From<SnapshotStorageMode> for semantic_code_config::SnapshotStorageMode {
    fn from(value: SnapshotStorageMode) -> Self {
        match value {
            SnapshotStorageMode::Disabled => Self::Disabled,
            SnapshotStorageMode::Project => Self::Project,
            SnapshotStorageMode::Custom(path) => Self::Custom(path),
        }
    }
}

/// Deterministic result key used for ordering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchResultKey {
    /// Stable logical path identifier.
    pub relative_path: Box<str>,
    /// Line span of the result.
    pub span: LineSpan,
}

impl From<semantic_code_domain::SearchResultKey> for SearchResultKey {
    fn from(value: semantic_code_domain::SearchResultKey) -> Self {
        Self {
            relative_path: value.relative_path,
            span: value.span,
        }
    }
}

/// Search result payload returned by facade search APIs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchResult {
    /// Ordering key.
    pub key: SearchResultKey,
    /// Optional chunk content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Box<str>>,
    /// Optional language hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<Language>,
    /// Similarity score.
    pub score: f32,
}

impl From<semantic_code_domain::SearchResult> for SearchResult {
    fn from(value: semantic_code_domain::SearchResult) -> Self {
        Self {
            key: value.key.into(),
            content: value.content,
            language: value.language,
            score: value.score,
        }
    }
}

/// Search diagnostics payload returned by facade search APIs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchStats {
    /// Kernel-specific expansion count, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expansions: Option<u64>,
    /// Effective kernel used for search.
    pub kernel: Box<str>,
    /// Kernel-specific extended metrics (e.g. DFRR pulls, splits, bucket utilization).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<Box<str>, f64>,
    /// Wall-clock nanoseconds spent in the kernel search algorithm.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_search_duration_ns: Option<u64>,
    /// Number of active vectors in the index at search time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_size: Option<u64>,
}

impl From<DomainSearchStats> for SearchStats {
    fn from(value: DomainSearchStats) -> Self {
        Self {
            expansions: value.expansions,
            kernel: value.kernel,
            extra: value.extra,
            kernel_search_duration_ns: value.kernel_search_duration_ns,
            index_size: value.index_size,
        }
    }
}

/// Search output payload returned by facade APIs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchOutput {
    /// Ordered search results.
    pub results: Vec<SearchResult>,
    /// Optional vector-search diagnostics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<SearchStats>,
}

impl From<semantic_code_app::SemanticSearchOutput> for SearchOutput {
    fn from(value: semantic_code_app::SemanticSearchOutput) -> Self {
        Self {
            results: value.results.into_iter().map(Into::into).collect(),
            stats: value.stats.map(Into::into),
        }
    }
}

/// API error kind (expected vs invariant).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ApiV1ErrorKind {
    /// Caller-provided input or environment issue.
    Expected,
    /// Internal invariant violation.
    Invariant,
}

impl From<semantic_code_api::v1::ApiV1ErrorKind> for ApiV1ErrorKind {
    fn from(value: semantic_code_api::v1::ApiV1ErrorKind) -> Self {
        match value {
            semantic_code_api::v1::ApiV1ErrorKind::Expected => Self::Expected,
            semantic_code_api::v1::ApiV1ErrorKind::Invariant => Self::Invariant,
        }
    }
}

/// Stable API error code string.
pub type ApiV1ErrorCode = String;
/// Optional API error metadata.
pub type ApiV1ErrorMeta = BTreeMap<String, String>;

/// API error payload emitted by facade error formatting helpers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiV1ErrorDto {
    /// Stable machine-readable error code.
    pub code: ApiV1ErrorCode,
    /// Human-readable error message.
    pub message: String,
    /// Error kind (expected vs invariant).
    pub kind: ApiV1ErrorKind,
    /// Optional metadata map.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<ApiV1ErrorMeta>,
}

impl From<semantic_code_api::v1::ApiV1ErrorDto> for ApiV1ErrorDto {
    fn from(value: semantic_code_api::v1::ApiV1ErrorDto) -> Self {
        Self {
            code: value.code,
            message: value.message,
            kind: value.kind.into(),
            meta: value.meta,
        }
    }
}

/// Facade-owned validated index request.
#[derive(Debug, Clone)]
pub struct IndexRequest(semantic_code_config::ValidatedIndexRequest);

impl IndexRequest {
    pub(crate) const fn as_validated(&self) -> &semantic_code_config::ValidatedIndexRequest {
        &self.0
    }
}

impl From<semantic_code_config::ValidatedIndexRequest> for IndexRequest {
    fn from(value: semantic_code_config::ValidatedIndexRequest) -> Self {
        Self(value)
    }
}

/// Facade-owned validated search request.
#[derive(Debug, Clone)]
pub struct SearchRequest(semantic_code_config::ValidatedSearchRequest);

impl SearchRequest {
    pub(crate) const fn as_validated(&self) -> &semantic_code_config::ValidatedSearchRequest {
        &self.0
    }
}

impl From<semantic_code_config::ValidatedSearchRequest> for SearchRequest {
    fn from(value: semantic_code_config::ValidatedSearchRequest) -> Self {
        Self(value)
    }
}

/// Facade-owned validated reindex-by-change request.
#[derive(Debug, Clone)]
pub struct ReindexByChangeRequest(semantic_code_config::ValidatedReindexByChangeRequest);

impl ReindexByChangeRequest {
    pub(crate) const fn as_validated(
        &self,
    ) -> &semantic_code_config::ValidatedReindexByChangeRequest {
        &self.0
    }
}

impl From<semantic_code_config::ValidatedReindexByChangeRequest> for ReindexByChangeRequest {
    fn from(value: semantic_code_config::ValidatedReindexByChangeRequest) -> Self {
        Self(value)
    }
}

/// Facade-owned validated clear-index request.
#[derive(Debug, Clone)]
pub struct ClearIndexRequest(semantic_code_config::ValidatedClearIndexRequest);

impl ClearIndexRequest {
    pub(crate) const fn as_validated(&self) -> &semantic_code_config::ValidatedClearIndexRequest {
        &self.0
    }
}

impl From<semantic_code_config::ValidatedClearIndexRequest> for ClearIndexRequest {
    fn from(value: semantic_code_config::ValidatedClearIndexRequest) -> Self {
        Self(value)
    }
}

/// Completion status for indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum IndexCodebaseStatus {
    /// Completed successfully.
    Completed,
    /// Stopped because the chunk limit was reached.
    LimitReached,
}

impl From<semantic_code_app::IndexCodebaseStatus> for IndexCodebaseStatus {
    fn from(value: semantic_code_app::IndexCodebaseStatus) -> Self {
        match value {
            semantic_code_app::IndexCodebaseStatus::Completed => Self::Completed,
            semantic_code_app::IndexCodebaseStatus::LimitReached => Self::LimitReached,
        }
    }
}

/// Scan stage stats for indexing output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexScanStats {
    /// Files discovered for indexing.
    pub files: u64,
    /// Elapsed time in milliseconds.
    pub duration_ms: u64,
}

impl From<semantic_code_app::ScanStageStats> for IndexScanStats {
    fn from(value: semantic_code_app::ScanStageStats) -> Self {
        Self {
            files: value.files,
            duration_ms: value.duration_ms,
        }
    }
}

/// Split stage stats for indexing output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexSplitStats {
    /// Files processed by the splitter.
    pub files: u64,
    /// Chunks produced by the splitter.
    pub chunks: u64,
    /// Elapsed time in milliseconds.
    pub duration_ms: u64,
}

impl From<semantic_code_app::SplitStageStats> for IndexSplitStats {
    fn from(value: semantic_code_app::SplitStageStats) -> Self {
        Self {
            files: value.files,
            chunks: value.chunks,
            duration_ms: value.duration_ms,
        }
    }
}

/// Embedding stage stats for indexing output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexEmbedStats {
    /// Embedding batches executed.
    pub batches: u64,
    /// Chunks embedded.
    pub chunks: u64,
    /// Elapsed time in milliseconds.
    pub duration_ms: u64,
}

impl From<semantic_code_app::EmbedStageStats> for IndexEmbedStats {
    fn from(value: semantic_code_app::EmbedStageStats) -> Self {
        Self {
            batches: value.batches,
            chunks: value.chunks,
            duration_ms: value.duration_ms,
        }
    }
}

/// Insert stage stats for indexing output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexInsertStats {
    /// Insert batches executed.
    pub batches: u64,
    /// Chunks inserted.
    pub chunks: u64,
    /// Elapsed time in milliseconds.
    pub duration_ms: u64,
}

impl From<semantic_code_app::InsertStageStats> for IndexInsertStats {
    fn from(value: semantic_code_app::InsertStageStats) -> Self {
        Self {
            batches: value.batches,
            chunks: value.chunks,
            duration_ms: value.duration_ms,
        }
    }
}

/// Aggregated ingestion stage stats for indexing output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexStageStats {
    /// Scan stage stats.
    pub scan: IndexScanStats,
    /// Split stage stats.
    pub split: IndexSplitStats,
    /// Embedding stage stats.
    pub embed: IndexEmbedStats,
    /// Insert stage stats.
    pub insert: IndexInsertStats,
}

impl From<semantic_code_app::IndexStageStats> for IndexStageStats {
    fn from(value: semantic_code_app::IndexStageStats) -> Self {
        Self {
            scan: value.scan.into(),
            split: value.split.into(),
            embed: value.embed.into(),
            insert: value.insert.into(),
        }
    }
}

/// Output returned by the index use-case through facade APIs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexCodebaseOutput {
    /// Number of files indexed.
    pub indexed_files: usize,
    /// Number of chunks indexed.
    pub total_chunks: usize,
    /// Completion status.
    pub status: IndexCodebaseStatus,
    /// Stage-level ingestion stats.
    pub stage_stats: IndexStageStats,
}

impl From<semantic_code_app::IndexCodebaseOutput> for IndexCodebaseOutput {
    fn from(value: semantic_code_app::IndexCodebaseOutput) -> Self {
        Self {
            indexed_files: value.indexed_files,
            total_chunks: value.total_chunks,
            status: value.status.into(),
            stage_stats: value.stage_stats.into(),
        }
    }
}

/// Output returned by reindex-by-change through facade APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReindexByChangeOutput {
    /// Added files count.
    pub added: usize,
    /// Removed files count.
    pub removed: usize,
    /// Modified files count.
    pub modified: usize,
}

impl From<semantic_code_app::ReindexByChangeOutput> for ReindexByChangeOutput {
    fn from(value: semantic_code_app::ReindexByChangeOutput) -> Self {
        Self {
            added: value.added,
            removed: value.removed,
            modified: value.modified,
        }
    }
}

/// Manifest persisted for local CLI operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CliManifestStatus {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Codebase root path for this manifest.
    pub codebase_root: PathBuf,
    /// Collection name used for indexing.
    pub collection_name: CollectionName,
    /// Index mode used for collection naming.
    pub index_mode: IndexMode,
    /// Snapshot storage mode for local persistence.
    pub snapshot_storage: SnapshotStorageMode,
    /// Manifest creation timestamp (milliseconds since epoch).
    pub created_at_ms: u64,
    /// Manifest last updated timestamp (milliseconds since epoch).
    pub updated_at_ms: u64,
}

/// Snapshot information for CLI status output.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
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

impl From<semantic_code_infra::SnapshotStatus> for SnapshotStatus {
    fn from(value: semantic_code_infra::SnapshotStatus) -> Self {
        Self {
            path: value.path,
            exists: value.exists,
            updated_at_ms: value.updated_at_ms,
            record_count: value.record_count,
        }
    }
}

/// Minimal config summary for CLI status output.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
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

impl From<semantic_code_infra::CliConfigSummary> for CliConfigSummary {
    fn from(value: semantic_code_infra::CliConfigSummary) -> Self {
        Self {
            index_mode: value.index_mode,
            snapshot_storage: value.snapshot_storage.into(),
            embedding_dimension: value.embedding_dimension,
            embedding_cache_enabled: value.embedding_cache_enabled,
            embedding_cache_disk_enabled: value.embedding_cache_disk_enabled,
            embedding_cache_max_entries: value.embedding_cache_max_entries,
            embedding_cache_max_bytes: value.embedding_cache_max_bytes,
            embedding_cache_disk_path: value.embedding_cache_disk_path,
            embedding_cache_disk_provider: value.embedding_cache_disk_provider,
            embedding_cache_disk_connection: value.embedding_cache_disk_connection,
            embedding_cache_disk_table: value.embedding_cache_disk_table,
            embedding_cache_disk_max_bytes: value.embedding_cache_disk_max_bytes,
            retry_max_attempts: value.retry_max_attempts,
            retry_base_delay_ms: value.retry_base_delay_ms,
            retry_max_delay_ms: value.retry_max_delay_ms,
            retry_jitter_ratio_pct: value.retry_jitter_ratio_pct,
            max_in_flight_files: value.max_in_flight_files,
            max_in_flight_embedding_batches: value.max_in_flight_embedding_batches,
            max_in_flight_inserts: value.max_in_flight_inserts,
            max_buffered_chunks: value.max_buffered_chunks,
            max_buffered_embeddings: value.max_buffered_embeddings,
        }
    }
}

/// Summary of local CLI status information.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CliStatus {
    /// Manifest associated with the local codebase.
    pub manifest: CliManifestStatus,
    /// Vector snapshot status.
    pub vector_snapshot: SnapshotStatus,
    /// Sync snapshot status.
    pub sync_snapshot: SnapshotStatus,
    /// Effective config summary.
    pub config: CliConfigSummary,
}

impl From<semantic_code_infra::CliStatus> for CliStatus {
    fn from(value: semantic_code_infra::CliStatus) -> Self {
        let manifest = value.manifest;
        Self {
            manifest: CliManifestStatus {
                schema_version: manifest.schema_version,
                codebase_root: manifest.codebase_root,
                collection_name: manifest.collection_name,
                index_mode: manifest.index_mode,
                snapshot_storage: manifest.snapshot_storage.into(),
                created_at_ms: manifest.created_at_ms,
                updated_at_ms: manifest.updated_at_ms,
            },
            vector_snapshot: value.vector_snapshot.into(),
            sync_snapshot: value.sync_snapshot.into(),
            config: value.config.into(),
        }
    }
}

/// Summary of init command results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
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

impl From<semantic_code_infra::CliInitStatus> for CliInitStatus {
    fn from(value: semantic_code_infra::CliInitStatus) -> Self {
        Self {
            config_path: value.config_path,
            manifest_path: value.manifest_path,
            created_config: value.created_config,
            created_manifest: value.created_manifest,
        }
    }
}

/// Storage threshold status when free-space data is available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StorageThresholdStatus {
    /// Available free space is at or above required bytes.
    Pass,
    /// Available free space is below required bytes.
    Fail,
    /// Free-space information could not be resolved.
    Unknown,
}

impl From<semantic_code_infra::StorageThresholdStatus> for StorageThresholdStatus {
    fn from(value: semantic_code_infra::StorageThresholdStatus) -> Self {
        match value {
            semantic_code_infra::StorageThresholdStatus::Pass => Self::Pass,
            semantic_code_infra::StorageThresholdStatus::Fail => Self::Fail,
            semantic_code_infra::StorageThresholdStatus::Unknown => Self::Unknown,
        }
    }
}

/// Storage estimate summary used by CLI output and index preflight.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CliStorageEstimate {
    /// Codebase root used for scanning.
    pub codebase_root: PathBuf,
    /// Effective vector provider identifier.
    pub vector_provider: Box<str>,
    /// Whether local snapshot storage is the active write target.
    pub local_storage_enforced: bool,
    /// Local storage root path when applicable.
    pub local_storage_root: Option<PathBuf>,
    /// Effective index mode.
    pub index_mode: IndexMode,
    /// Number of regular files scanned after ignore checks.
    pub files_scanned: u64,
    /// Number of files considered for indexing (after extension + size + UTF-8 checks).
    pub files_indexable: u64,
    /// Aggregate UTF-8 byte count across indexable files.
    pub bytes_indexable: u64,
    /// Aggregate character count across indexable files.
    pub chars_indexable: u64,
    /// Estimated chunk count.
    pub estimated_chunks: u64,
    /// Lower-bound embedding dimension used for estimate ranges.
    pub dimension_low: u32,
    /// Upper-bound embedding dimension used for estimate ranges.
    pub dimension_high: u32,
    /// Lower-bound estimated index bytes.
    pub estimated_bytes_low: u64,
    /// Upper-bound estimated index bytes.
    pub estimated_bytes_high: u64,
    /// Required free-space bytes using selected safety factor.
    pub required_free_bytes: u64,
    /// Safety factor numerator.
    pub safety_factor_num: u64,
    /// Safety factor denominator.
    pub safety_factor_den: u64,
    /// Available free-space bytes at the resolved storage root, when known.
    pub available_bytes: Option<u64>,
    /// Pass/fail/unknown threshold status.
    pub threshold_status: StorageThresholdStatus,
}

impl From<semantic_code_infra::CliStorageEstimate> for CliStorageEstimate {
    fn from(value: semantic_code_infra::CliStorageEstimate) -> Self {
        Self {
            codebase_root: value.codebase_root,
            vector_provider: value.vector_provider,
            local_storage_enforced: value.local_storage_enforced,
            local_storage_root: value.local_storage_root,
            index_mode: value.index_mode,
            files_scanned: value.files_scanned,
            files_indexable: value.files_indexable,
            bytes_indexable: value.bytes_indexable,
            chars_indexable: value.chars_indexable,
            estimated_chunks: value.estimated_chunks,
            dimension_low: value.dimension_low,
            dimension_high: value.dimension_high,
            estimated_bytes_low: value.estimated_bytes_low,
            estimated_bytes_high: value.estimated_bytes_high,
            required_free_bytes: value.required_free_bytes,
            safety_factor_num: value.safety_factor_num,
            safety_factor_den: value.safety_factor_den,
            available_bytes: value.available_bytes,
            threshold_status: value.threshold_status.into(),
        }
    }
}

/// Background job kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum JobKind {
    /// Index a codebase.
    Index,
    /// Reindex by change.
    Reindex,
}

impl From<JobKind> for semantic_code_infra::JobKind {
    fn from(value: JobKind) -> Self {
        match value {
            JobKind::Index => Self::Index,
            JobKind::Reindex => Self::Reindex,
        }
    }
}

impl From<semantic_code_infra::JobKind> for JobKind {
    fn from(value: semantic_code_infra::JobKind) -> Self {
        match value {
            semantic_code_infra::JobKind::Index => Self::Index,
            semantic_code_infra::JobKind::Reindex => Self::Reindex,
        }
    }
}

/// Background job state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum JobState {
    /// Job is queued for execution.
    Queued,
    /// Job is running.
    Running,
    /// Job completed successfully.
    Completed,
    /// Job failed.
    Failed,
    /// Job was cancelled.
    Cancelled,
}

impl From<semantic_code_infra::JobState> for JobState {
    fn from(value: semantic_code_infra::JobState) -> Self {
        match value {
            semantic_code_infra::JobState::Queued => Self::Queued,
            semantic_code_infra::JobState::Running => Self::Running,
            semantic_code_infra::JobState::Completed => Self::Completed,
            semantic_code_infra::JobState::Failed => Self::Failed,
            semantic_code_infra::JobState::Cancelled => Self::Cancelled,
        }
    }
}

/// Background job request payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobRequest {
    /// Job identifier.
    pub id: Box<str>,
    /// Job kind.
    pub kind: JobKind,
    /// Codebase root for this job.
    pub codebase_root: PathBuf,
    /// Optional config path override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_path: Option<PathBuf>,
    /// Optional config overrides JSON.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overrides_json: Option<String>,
    /// Whether to auto-init manifests for index jobs.
    #[serde(default)]
    pub init_if_missing: bool,
    /// Creation timestamp (ms since epoch).
    pub created_at_ms: u64,
}

impl JobRequest {
    /// Create a new job request.
    pub fn new(
        kind: JobKind,
        codebase_root: &Path,
        config_path: Option<&Path>,
        overrides_json: Option<String>,
        init_if_missing: bool,
    ) -> Result<Self, InfraError> {
        semantic_code_infra::JobRequest::new(
            kind.into(),
            codebase_root,
            config_path,
            overrides_json,
            init_if_missing,
        )
        .map_err(Into::into)
        .map(Self::from)
    }
}

impl From<semantic_code_infra::JobRequest> for JobRequest {
    fn from(value: semantic_code_infra::JobRequest) -> Self {
        Self {
            id: value.id,
            kind: value.kind.into(),
            codebase_root: value.codebase_root,
            config_path: value.config_path,
            overrides_json: value.overrides_json,
            init_if_missing: value.init_if_missing,
            created_at_ms: value.created_at_ms,
        }
    }
}

impl From<JobRequest> for semantic_code_infra::JobRequest {
    fn from(value: JobRequest) -> Self {
        Self {
            id: value.id,
            kind: value.kind.into(),
            codebase_root: value.codebase_root,
            config_path: value.config_path,
            overrides_json: value.overrides_json,
            init_if_missing: value.init_if_missing,
            created_at_ms: value.created_at_ms,
        }
    }
}

/// Job progress snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobProgress {
    /// Stage label for quick filtering.
    pub stage: Box<str>,
    /// Human-friendly phase label.
    pub phase: Box<str>,
    /// Current count in this phase.
    pub current: u64,
    /// Total count in this phase.
    pub total: u64,
    /// Completion percentage (0-100).
    pub percentage: u8,
}

impl From<semantic_code_infra::JobProgress> for JobProgress {
    fn from(value: semantic_code_infra::JobProgress) -> Self {
        Self {
            stage: value.stage,
            phase: value.phase,
            current: value.current,
            total: value.total,
            percentage: value.percentage,
        }
    }
}

/// Job-friendly scan stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobScanStats {
    /// Files discovered for indexing.
    pub files: u64,
    /// Elapsed time in milliseconds.
    pub duration_ms: u64,
}

/// Job-friendly split stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobSplitStats {
    /// Files processed by the splitter.
    pub files: u64,
    /// Chunks produced by the splitter.
    pub chunks: u64,
    /// Elapsed time in milliseconds.
    pub duration_ms: u64,
}

/// Job-friendly embed stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobEmbedStats {
    /// Embedding batches executed.
    pub batches: u64,
    /// Chunks embedded.
    pub chunks: u64,
    /// Elapsed time in milliseconds.
    pub duration_ms: u64,
}

/// Job-friendly insert stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobInsertStats {
    /// Insert batches executed.
    pub batches: u64,
    /// Chunks inserted.
    pub chunks: u64,
    /// Elapsed time in milliseconds.
    pub duration_ms: u64,
}

/// Job-friendly stage stats for indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobStageStats {
    /// Scan stage stats.
    pub scan: JobScanStats,
    /// Split stage stats.
    pub split: JobSplitStats,
    /// Embedding stage stats.
    pub embed: JobEmbedStats,
    /// Insert stage stats.
    pub insert: JobInsertStats,
}

/// Job result payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum JobResult {
    /// Index job result.
    Index {
        /// Indexed files count.
        indexed_files: usize,
        /// Total chunks indexed.
        total_chunks: usize,
        /// Index completion status.
        index_status: Box<str>,
        /// Stage-level ingestion stats.
        stage_stats: JobStageStats,
    },
    /// Reindex job result.
    Reindex {
        /// Added files count.
        added: usize,
        /// Removed files count.
        removed: usize,
        /// Modified files count.
        modified: usize,
    },
}

impl From<semantic_code_infra::JobResult> for JobResult {
    fn from(value: semantic_code_infra::JobResult) -> Self {
        match value {
            semantic_code_infra::JobResult::Index {
                indexed_files,
                total_chunks,
                index_status,
                stage_stats,
            } => Self::Index {
                indexed_files,
                total_chunks,
                index_status,
                stage_stats: JobStageStats {
                    scan: JobScanStats {
                        files: stage_stats.scan.files,
                        duration_ms: stage_stats.scan.duration_ms,
                    },
                    split: JobSplitStats {
                        files: stage_stats.split.files,
                        chunks: stage_stats.split.chunks,
                        duration_ms: stage_stats.split.duration_ms,
                    },
                    embed: JobEmbedStats {
                        batches: stage_stats.embed.batches,
                        chunks: stage_stats.embed.chunks,
                        duration_ms: stage_stats.embed.duration_ms,
                    },
                    insert: JobInsertStats {
                        batches: stage_stats.insert.batches,
                        chunks: stage_stats.insert.chunks,
                        duration_ms: stage_stats.insert.duration_ms,
                    },
                },
            },
            semantic_code_infra::JobResult::Reindex {
                added,
                removed,
                modified,
            } => Self::Reindex {
                added,
                removed,
                modified,
            },
        }
    }
}

/// Job error payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobError {
    /// Stable error code.
    pub code: Box<str>,
    /// Human-readable message.
    pub message: Box<str>,
    /// Error class.
    pub class: Box<str>,
}

impl From<semantic_code_infra::JobError> for JobError {
    fn from(value: semantic_code_infra::JobError) -> Self {
        Self {
            code: value.code,
            message: value.message,
            class: value.class,
        }
    }
}

/// Current job status.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobStatus {
    /// Job identifier.
    pub id: Box<str>,
    /// Job kind.
    pub kind: JobKind,
    /// Job state.
    pub state: JobState,
    /// Job creation time (ms since epoch).
    pub created_at_ms: u64,
    /// Job last update time (ms since epoch).
    pub updated_at_ms: u64,
    /// Latest progress snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<JobProgress>,
    /// Job result payload (when completed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<JobResult>,
    /// Error payload (when failed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JobError>,
    /// Whether cancel was requested.
    #[serde(default)]
    pub cancel_requested: bool,
    /// Non-fatal warnings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<JobError>,
}

impl From<semantic_code_infra::JobStatus> for JobStatus {
    fn from(value: semantic_code_infra::JobStatus) -> Self {
        Self {
            id: value.id,
            kind: value.kind.into(),
            state: value.state.into(),
            created_at_ms: value.created_at_ms,
            updated_at_ms: value.updated_at_ms,
            progress: value.progress.map(Into::into),
            result: value.result.map(Into::into),
            error: value.error.map(Into::into),
            cancel_requested: value.cancel_requested,
            warnings: value.warnings.into_iter().map(Into::into).collect(),
        }
    }
}
