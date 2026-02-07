//! Backend configuration schema, defaults, validation, and normalization.
//!
//! Phase 03 focuses on making configuration parsing deterministic and safe:
//! - Deserialization uses `serde` (JSON for now).
//! - Validation is manual and returns typed errors mapped to `ErrorEnvelope`.
//! - Normalization enforces stable ordering for list fields.

use crate::storage::SnapshotStorageMode;
use semantic_code_domain::IndexMode;
use semantic_code_shared::{BoundedU32, BoundedU64, ErrorCode, ErrorEnvelope};
use serde::{Deserialize, Serialize, de};
use std::collections::BTreeMap;
use std::fmt;
use url::Url;

/// Sanitizes a URL for error messages by stripping credentials.
///
/// This prevents accidental exposure of passwords or tokens that may be
/// embedded in URLs (e.g., `https://user:pass@host.com`).  // pragma: allowlist secret
fn sanitize_url_for_error(url: &str) -> String {
    match Url::parse(url) {
        Ok(mut parsed) => {
            if parsed.password().is_some() || !parsed.username().is_empty() {
                // Clear credentials before logging.
                if parsed.set_username("").is_err() {
                    return "[invalid url: invalid username]".to_string();
                }
                if parsed.set_password(None).is_err() {
                    return "[invalid url: invalid password]".to_string();
                }
            }
            parsed.to_string()
        },
        Err(error) => format!("[invalid url: {error}]"),
    }
}

/// Current supported configuration schema version.
pub const CURRENT_CONFIG_VERSION: u32 = 1;

const CORE_TIMEOUT_MIN_MS: u64 = 1_000;
const CORE_TIMEOUT_MAX_MS: u64 = 600_000;
const CORE_MAX_CONCURRENCY_MIN: u32 = 1;
const CORE_MAX_CONCURRENCY_MAX: u32 = 256;
const CORE_MAX_IN_FLIGHT_MIN: u32 = 1;
const CORE_MAX_IN_FLIGHT_MAX: u32 = 256;
const CORE_MAX_BUFFERED_MIN: u32 = 1;
const CORE_MAX_BUFFERED_MAX: u32 = 1_000_000;
const CORE_MAX_CHUNK_CHARS_MIN: u32 = 1;
const CORE_MAX_CHUNK_CHARS_MAX: u32 = 20_000;
const CORE_MAX_CHUNK_CHARS_DEFAULT: u32 = 2_500;

const RETRY_MAX_ATTEMPTS_MIN: u32 = 1;
const RETRY_MAX_ATTEMPTS_MAX: u32 = 10;
const RETRY_BASE_DELAY_MIN_MS: u64 = 1;
const RETRY_BASE_DELAY_MAX_MS: u64 = 60_000;
const RETRY_MAX_DELAY_MIN_MS: u64 = 1;
const RETRY_MAX_DELAY_MAX_MS: u64 = 600_000;
const RETRY_JITTER_RATIO_PCT_MIN: u32 = 0;
const RETRY_JITTER_RATIO_PCT_MAX: u32 = 100;

const EMBEDDING_TIMEOUT_MIN_MS: u64 = 1_000;
const EMBEDDING_TIMEOUT_MAX_MS: u64 = 1_200_000;
const EMBEDDING_BATCH_SIZE_MIN: u32 = 1;
const EMBEDDING_BATCH_SIZE_MAX: u32 = 8_192;
const EMBEDDING_DIMENSION_MIN: u32 = 1;
const EMBEDDING_DIMENSION_MAX: u32 = 65_536;
const EMBEDDING_ONNX_SESSION_POOL_MIN: u32 = 1;
const EMBEDDING_ONNX_SESSION_POOL_MAX: u32 = 64;
const EMBEDDING_ROUTING_REMOTE_BATCHES_MIN: u32 = 1;
const EMBEDDING_ROUTING_REMOTE_BATCHES_MAX: u32 = 1_000_000;
const EMBEDDING_JOB_PROGRESS_MIN_MS: u64 = 50;
const EMBEDDING_JOB_PROGRESS_MAX_MS: u64 = 60_000;
const EMBEDDING_JOB_CANCEL_POLL_MIN_MS: u64 = 50;
const EMBEDDING_JOB_CANCEL_POLL_MAX_MS: u64 = 60_000;
const EMBEDDING_CACHE_MAX_ENTRIES_MIN: u32 = 1;
const EMBEDDING_CACHE_MAX_ENTRIES_MAX: u32 = 100_000;
const EMBEDDING_CACHE_MAX_BYTES_MIN: u64 = 1;
const EMBEDDING_CACHE_MAX_BYTES_MAX: u64 = 10_000_000_000;
const EMBEDDING_CACHE_DISK_MAX_BYTES_MIN: u64 = 1;
const EMBEDDING_CACHE_DISK_MAX_BYTES_MAX: u64 = 100_000_000_000;

const VECTOR_DB_TIMEOUT_MIN_MS: u64 = 1_000;
const VECTOR_DB_TIMEOUT_MAX_MS: u64 = 1_200_000;
const VECTOR_DB_INDEX_TIMEOUT_MIN_MS: u64 = 1_000;
const VECTOR_DB_INDEX_TIMEOUT_MAX_MS: u64 = 3_600_000;
const VECTOR_DB_BATCH_SIZE_MIN: u32 = 1;
const VECTOR_DB_BATCH_SIZE_MAX: u32 = 16_384;
const VECTOR_DB_INDEX_PARAMS_MAX: usize = 128;

const SYNC_MAX_FILES_MIN: u32 = 1;
const SYNC_MAX_FILES_MAX: u32 = 10_000_000;
const SYNC_MAX_FILE_SIZE_MIN_BYTES: u64 = 1;
const SYNC_MAX_FILE_SIZE_MAX_BYTES: u64 = 100_000_000;

const SYNC_ALLOWED_EXTENSIONS_MAX: usize = 128;
const SYNC_IGNORE_PATTERNS_MAX: usize = 512;

/// Top-level backend configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
pub struct BackendConfig {
    /// Schema version for forward-compatible migrations.
    pub version: u32,
    /// Core runtime settings.
    pub core: CoreConfig,
    /// Embedding adapter settings.
    pub embedding: EmbeddingConfig,
    /// Vector DB adapter settings.
    pub vector_db: VectorDbConfig,
    /// File sync/scanning settings.
    pub sync: SyncConfig,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            version: CURRENT_CONFIG_VERSION,
            core: CoreConfig::default(),
            embedding: EmbeddingConfig::default(),
            vector_db: VectorDbConfig::default(),
            sync: SyncConfig::default(),
        }
    }
}

impl BackendConfig {
    /// Validate and normalize the config.
    pub fn validate_and_normalize(mut self) -> Result<ValidatedBackendConfig, ConfigSchemaError> {
        self.validate_version()?;

        self.core.validate()?;
        self.embedding.normalize();
        self.embedding.validate()?;
        self.vector_db.normalize();
        self.vector_db.validate()?;
        self.sync.normalize_and_validate()?;

        let limits = ConfigLimits::new(&self)?;
        Ok(ValidatedBackendConfig { raw: self, limits })
    }

    const fn validate_version(&self) -> Result<(), ConfigSchemaError> {
        if self.version != CURRENT_CONFIG_VERSION {
            return Err(ConfigSchemaError::UnsupportedVersion {
                found: self.version,
                supported: CURRENT_CONFIG_VERSION,
            });
        }
        Ok(())
    }
}

/// Validated config wrapper carrying bounded numeric values.
#[derive(Debug, Clone)]
pub struct ValidatedBackendConfig {
    raw: BackendConfig,
    limits: ConfigLimits,
}

impl ValidatedBackendConfig {
    /// Access validated numeric bounds.
    #[must_use]
    pub const fn limits(&self) -> &ConfigLimits {
        &self.limits
    }

    /// Borrow the raw config.
    #[must_use]
    pub const fn as_ref(&self) -> &BackendConfig {
        &self.raw
    }

    /// Consume the wrapper and return the raw config.
    #[must_use]
    pub fn into_inner(self) -> BackendConfig {
        self.raw
    }
}

impl AsRef<BackendConfig> for ValidatedBackendConfig {
    fn as_ref(&self) -> &BackendConfig {
        &self.raw
    }
}

impl std::ops::Deref for ValidatedBackendConfig {
    type Target = BackendConfig;

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

/// Validated numeric limits derived from the config.
#[derive(Debug, Clone, Copy)]
pub struct ConfigLimits {
    /// Core timeout (ms).
    pub core_timeout_ms: BoundedU64<CORE_TIMEOUT_MIN_MS, CORE_TIMEOUT_MAX_MS>,
    /// Core max concurrency.
    pub core_max_concurrency: BoundedU32<CORE_MAX_CONCURRENCY_MIN, CORE_MAX_CONCURRENCY_MAX>,
    /// Core max chunk chars.
    pub core_max_chunk_chars: BoundedU32<CORE_MAX_CHUNK_CHARS_MIN, CORE_MAX_CHUNK_CHARS_MAX>,
    /// Embedding batch size.
    pub embedding_batch_size: BoundedU32<EMBEDDING_BATCH_SIZE_MIN, EMBEDDING_BATCH_SIZE_MAX>,
    /// Vector DB batch size.
    pub vector_db_batch_size: BoundedU32<VECTOR_DB_BATCH_SIZE_MIN, VECTOR_DB_BATCH_SIZE_MAX>,
    /// Sync max files.
    pub sync_max_files: BoundedU32<SYNC_MAX_FILES_MIN, SYNC_MAX_FILES_MAX>,
    /// Sync max file size (bytes).
    pub sync_max_file_size_bytes:
        BoundedU64<SYNC_MAX_FILE_SIZE_MIN_BYTES, SYNC_MAX_FILE_SIZE_MAX_BYTES>,
    /// Optional caps (in-flight files).
    pub core_max_in_flight_files:
        Option<BoundedU32<CORE_MAX_IN_FLIGHT_MIN, CORE_MAX_IN_FLIGHT_MAX>>,
    /// Optional caps (in-flight embedding batches).
    pub core_max_in_flight_embedding_batches:
        Option<BoundedU32<CORE_MAX_IN_FLIGHT_MIN, CORE_MAX_IN_FLIGHT_MAX>>,
    /// Optional caps (in-flight inserts).
    pub core_max_in_flight_inserts:
        Option<BoundedU32<CORE_MAX_IN_FLIGHT_MIN, CORE_MAX_IN_FLIGHT_MAX>>,
    /// Optional caps (buffered chunks).
    pub core_max_buffered_chunks: Option<BoundedU32<CORE_MAX_BUFFERED_MIN, CORE_MAX_BUFFERED_MAX>>,
    /// Optional caps (buffered embeddings).
    pub core_max_buffered_embeddings:
        Option<BoundedU32<CORE_MAX_BUFFERED_MIN, CORE_MAX_BUFFERED_MAX>>,
}

impl ConfigLimits {
    fn new(config: &BackendConfig) -> Result<Self, ConfigSchemaError> {
        Ok(Self {
            core_timeout_ms: bounded_u64(
                "core",
                "timeoutMs",
                config.core.timeout_ms,
                CORE_TIMEOUT_MIN_MS,
                CORE_TIMEOUT_MAX_MS,
            )?,
            core_max_concurrency: bounded_u32(
                "core",
                "maxConcurrency",
                config.core.max_concurrency,
                CORE_MAX_CONCURRENCY_MIN,
                CORE_MAX_CONCURRENCY_MAX,
            )?,
            core_max_chunk_chars: bounded_u32(
                "core",
                "maxChunkChars",
                config.core.max_chunk_chars,
                CORE_MAX_CHUNK_CHARS_MIN,
                CORE_MAX_CHUNK_CHARS_MAX,
            )?,
            embedding_batch_size: bounded_u32(
                "embedding",
                "batchSize",
                config.embedding.batch_size,
                EMBEDDING_BATCH_SIZE_MIN,
                EMBEDDING_BATCH_SIZE_MAX,
            )?,
            vector_db_batch_size: bounded_u32(
                "vectorDb",
                "batchSize",
                config.vector_db.batch_size,
                VECTOR_DB_BATCH_SIZE_MIN,
                VECTOR_DB_BATCH_SIZE_MAX,
            )?,
            sync_max_files: bounded_u32(
                "sync",
                "maxFiles",
                config.sync.max_files,
                SYNC_MAX_FILES_MIN,
                SYNC_MAX_FILES_MAX,
            )?,
            sync_max_file_size_bytes: bounded_u64(
                "sync",
                "maxFileSizeBytes",
                config.sync.max_file_size_bytes,
                SYNC_MAX_FILE_SIZE_MIN_BYTES,
                SYNC_MAX_FILE_SIZE_MAX_BYTES,
            )?,
            core_max_in_flight_files: bounded_opt_u32(
                "core",
                "maxInFlightFiles",
                config.core.max_in_flight_files,
                CORE_MAX_IN_FLIGHT_MIN,
                CORE_MAX_IN_FLIGHT_MAX,
            )?,
            core_max_in_flight_embedding_batches: bounded_opt_u32(
                "core",
                "maxInFlightEmbeddingBatches",
                config.core.max_in_flight_embedding_batches,
                CORE_MAX_IN_FLIGHT_MIN,
                CORE_MAX_IN_FLIGHT_MAX,
            )?,
            core_max_in_flight_inserts: bounded_opt_u32(
                "core",
                "maxInFlightInserts",
                config.core.max_in_flight_inserts,
                CORE_MAX_IN_FLIGHT_MIN,
                CORE_MAX_IN_FLIGHT_MAX,
            )?,
            core_max_buffered_chunks: bounded_opt_u32(
                "core",
                "maxBufferedChunks",
                config.core.max_buffered_chunks,
                CORE_MAX_BUFFERED_MIN,
                CORE_MAX_BUFFERED_MAX,
            )?,
            core_max_buffered_embeddings: bounded_opt_u32(
                "core",
                "maxBufferedEmbeddings",
                config.core.max_buffered_embeddings,
                CORE_MAX_BUFFERED_MIN,
                CORE_MAX_BUFFERED_MAX,
            )?,
        })
    }
}

/// Parse a backend config from a JSON string, applying validation and normalization.
pub fn parse_backend_config_json(input: &str) -> Result<ValidatedBackendConfig, ErrorEnvelope> {
    let config: BackendConfig = serde_json::from_str(input).map_err(|error| {
        ErrorEnvelope::expected(
            ErrorCode::new("config", "invalid_json"),
            format!("invalid config JSON: {error}"),
        )
    })?;

    config.validate_and_normalize().map_err(Into::into)
}

/// Parse a backend config from a TOML string, applying validation and normalization.
pub fn parse_backend_config_toml(input: &str) -> Result<ValidatedBackendConfig, ErrorEnvelope> {
    let config: BackendConfig = toml::from_str(input).map_err(|error| {
        ErrorEnvelope::expected(
            ErrorCode::new("config", "invalid_toml"),
            format!("invalid config TOML: {error}"),
        )
    })?;

    config.validate_and_normalize().map_err(Into::into)
}

/// Core runtime configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
pub struct CoreConfig {
    /// Maximum time (in ms) allowed for a single request/operation at the boundary.
    pub timeout_ms: u64,
    /// Maximum concurrent in-flight work items.
    pub max_concurrency: u32,
    /// Optional cap on in-flight file tasks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_in_flight_files: Option<u32>,
    /// Optional cap on in-flight embedding batches.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_in_flight_embedding_batches: Option<u32>,
    /// Optional cap on in-flight insert batches.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_in_flight_inserts: Option<u32>,
    /// Optional cap on buffered chunks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_buffered_chunks: Option<u32>,
    /// Optional cap on buffered embeddings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_buffered_embeddings: Option<u32>,
    /// Maximum characters allowed per chunk (best-effort).
    pub max_chunk_chars: u32,
    /// Retry policy for transient failures.
    #[serde(default)]
    pub retry: RetryConfig,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 30_000,
            max_concurrency: 8,
            max_in_flight_files: None,
            max_in_flight_embedding_batches: None,
            max_in_flight_inserts: None,
            max_buffered_chunks: None,
            max_buffered_embeddings: None,
            max_chunk_chars: CORE_MAX_CHUNK_CHARS_DEFAULT,
            retry: RetryConfig::default(),
        }
    }
}

impl CoreConfig {
    fn validate(&self) -> Result<(), ConfigSchemaError> {
        validate_timeout_ms(
            "core",
            "timeoutMs",
            self.timeout_ms,
            CORE_TIMEOUT_MIN_MS,
            CORE_TIMEOUT_MAX_MS,
        )?;
        validate_limit_u32(
            "core",
            "maxConcurrency",
            self.max_concurrency,
            CORE_MAX_CONCURRENCY_MIN,
            CORE_MAX_CONCURRENCY_MAX,
        )?;
        validate_optional_limit_u32(
            "core",
            "maxInFlightFiles",
            self.max_in_flight_files,
            CORE_MAX_IN_FLIGHT_MIN,
            CORE_MAX_IN_FLIGHT_MAX,
        )?;
        validate_optional_limit_u32(
            "core",
            "maxInFlightEmbeddingBatches",
            self.max_in_flight_embedding_batches,
            CORE_MAX_IN_FLIGHT_MIN,
            CORE_MAX_IN_FLIGHT_MAX,
        )?;
        validate_optional_limit_u32(
            "core",
            "maxInFlightInserts",
            self.max_in_flight_inserts,
            CORE_MAX_IN_FLIGHT_MIN,
            CORE_MAX_IN_FLIGHT_MAX,
        )?;
        validate_optional_limit_u32(
            "core",
            "maxBufferedChunks",
            self.max_buffered_chunks,
            CORE_MAX_BUFFERED_MIN,
            CORE_MAX_BUFFERED_MAX,
        )?;
        validate_optional_limit_u32(
            "core",
            "maxBufferedEmbeddings",
            self.max_buffered_embeddings,
            CORE_MAX_BUFFERED_MIN,
            CORE_MAX_BUFFERED_MAX,
        )?;
        validate_limit_u32(
            "core",
            "maxChunkChars",
            self.max_chunk_chars,
            CORE_MAX_CHUNK_CHARS_MIN,
            CORE_MAX_CHUNK_CHARS_MAX,
        )?;
        self.retry.validate()?;
        Ok(())
    }
}

/// Retry policy configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
pub struct RetryConfig {
    /// Maximum attempts (including the first attempt).
    pub max_attempts: u32,
    /// Base delay for exponential backoff (ms).
    pub base_delay_ms: u64,
    /// Maximum delay cap for backoff (ms).
    pub max_delay_ms: u64,
    /// Jitter ratio as a percentage (0..=100).
    pub jitter_ratio_pct: u32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay_ms: 250,
            max_delay_ms: 5_000,
            jitter_ratio_pct: 20,
        }
    }
}

impl RetryConfig {
    fn validate(&self) -> Result<(), ConfigSchemaError> {
        validate_limit_u32(
            "core.retry",
            "maxAttempts",
            self.max_attempts,
            RETRY_MAX_ATTEMPTS_MIN,
            RETRY_MAX_ATTEMPTS_MAX,
        )?;
        validate_timeout_ms(
            "core.retry",
            "baseDelayMs",
            self.base_delay_ms,
            RETRY_BASE_DELAY_MIN_MS,
            RETRY_BASE_DELAY_MAX_MS,
        )?;
        validate_timeout_ms(
            "core.retry",
            "maxDelayMs",
            self.max_delay_ms,
            RETRY_MAX_DELAY_MIN_MS,
            RETRY_MAX_DELAY_MAX_MS,
        )?;
        if self.max_delay_ms < self.base_delay_ms {
            return Err(ConfigSchemaError::LimitOutOfRange {
                section: "core.retry",
                field: "maxDelayMs",
                value: self.max_delay_ms,
                min: self.base_delay_ms,
                max: RETRY_MAX_DELAY_MAX_MS,
            });
        }
        validate_limit_u32(
            "core.retry",
            "jitterRatioPct",
            self.jitter_ratio_pct,
            RETRY_JITTER_RATIO_PCT_MIN,
            RETRY_JITTER_RATIO_PCT_MAX,
        )?;
        Ok(())
    }
}

/// ONNX-specific configuration for local embeddings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
pub struct OnnxEmbeddingConfig {
    /// Optional model directory override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_dir: Option<Box<str>>,
    /// Optional model filename override (defaults to `onnx/model.onnx` if present).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_filename: Option<Box<str>>,
    /// Optional tokenizer filename override (defaults to `tokenizer.json`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_filename: Option<Box<str>>,
    /// Optional Hugging Face repo ID for download-on-missing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo: Option<Box<str>>,
    /// Download missing ONNX assets on first use.
    pub download_on_missing: bool,
    /// Number of ONNX sessions to keep in the pool.
    pub session_pool_size: u32,
}

impl Default for OnnxEmbeddingConfig {
    fn default() -> Self {
        Self {
            model_dir: None,
            model_filename: None,
            tokenizer_filename: None,
            repo: None,
            download_on_missing: true,
            session_pool_size: 1,
        }
    }
}

impl OnnxEmbeddingConfig {
    fn normalize(&mut self) {
        normalize_optional_trimmed(&mut self.model_dir);
        normalize_optional_trimmed(&mut self.model_filename);
        normalize_optional_trimmed(&mut self.tokenizer_filename);
        normalize_optional_trimmed(&mut self.repo);
    }
}

/// Embedding routing mode for hybrid/local selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EmbeddingRoutingMode {
    /// Prefer local embeddings, fall back to remote.
    LocalFirst,
    /// Prefer remote embeddings, fall back to local.
    RemoteFirst,
    /// Split batches between local and remote according to budget.
    Split,
}

impl EmbeddingRoutingMode {
    /// Return the canonical config string for this routing mode.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LocalFirst => "localFirst",
            Self::RemoteFirst => "remoteFirst",
            Self::Split => "split",
        }
    }

    /// Parse a routing mode from user or env input.
    pub fn parse(input: &str) -> Option<Self> {
        let normalized = input.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "localfirst" | "local_first" | "local-first" => Some(Self::LocalFirst),
            "remotefirst" | "remote_first" | "remote-first" => Some(Self::RemoteFirst),
            "split" => Some(Self::Split),
            _ => None,
        }
    }
}

impl fmt::Display for EmbeddingRoutingMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Split routing configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
pub struct EmbeddingSplitConfig {
    /// Maximum remote embedding batches per run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_remote_batches: Option<u32>,
}

impl EmbeddingSplitConfig {
    fn validate(&self) -> Result<(), ConfigSchemaError> {
        if let Some(value) = self.max_remote_batches {
            validate_limit_u32(
                "embedding.routing.split",
                "maxRemoteBatches",
                value,
                EMBEDDING_ROUTING_REMOTE_BATCHES_MIN,
                EMBEDDING_ROUTING_REMOTE_BATCHES_MAX,
            )?;
        }
        Ok(())
    }
}

/// Embedding routing configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
pub struct EmbeddingRoutingConfig {
    /// Routing mode override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<EmbeddingRoutingMode>,
    /// Split routing settings.
    #[serde(default)]
    pub split: EmbeddingSplitConfig,
}

impl EmbeddingRoutingConfig {
    fn validate(&self) -> Result<(), ConfigSchemaError> {
        self.split.validate()?;
        Ok(())
    }
}

/// Embedding job runtime configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
pub struct EmbeddingJobsConfig {
    /// Progress update interval for background jobs (ms).
    pub progress_interval_ms: u64,
    /// Cancel polling interval for background jobs (ms).
    pub cancel_poll_interval_ms: u64,
}

impl Default for EmbeddingJobsConfig {
    fn default() -> Self {
        Self {
            progress_interval_ms: 250,
            cancel_poll_interval_ms: 250,
        }
    }
}

impl EmbeddingJobsConfig {
    fn validate(&self) -> Result<(), ConfigSchemaError> {
        validate_timeout_ms(
            "embedding.jobs",
            "progressIntervalMs",
            self.progress_interval_ms,
            EMBEDDING_JOB_PROGRESS_MIN_MS,
            EMBEDDING_JOB_PROGRESS_MAX_MS,
        )?;
        validate_timeout_ms(
            "embedding.jobs",
            "cancelPollIntervalMs",
            self.cancel_poll_interval_ms,
            EMBEDDING_JOB_CANCEL_POLL_MIN_MS,
            EMBEDDING_JOB_CANCEL_POLL_MAX_MS,
        )?;
        Ok(())
    }
}

/// Embedding provider configuration (Phase 06+ adds provider-specific fields).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
pub struct EmbeddingConfig {
    /// Optional provider identifier (e.g. `openai`, `gemini`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<Box<str>>,
    /// Optional embedding model override (provider-specific).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<Box<str>>,
    /// Optional base URL for the embedding provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<Box<str>>,
    /// Optional embedding dimension override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimension: Option<u32>,
    /// Timeout for embedding requests (ms).
    pub timeout_ms: u64,
    /// Batch size for embedding calls.
    pub batch_size: u32,
    /// Prefer local ONNX embeddings over remote providers.
    pub local_first: bool,
    /// Force local ONNX embeddings only.
    pub local_only: bool,
    /// Local ONNX configuration.
    #[serde(default)]
    pub onnx: OnnxEmbeddingConfig,
    /// Embedding routing configuration.
    #[serde(default)]
    pub routing: EmbeddingRoutingConfig,
    /// Background job configuration.
    #[serde(default)]
    pub jobs: EmbeddingJobsConfig,
    /// Embedding cache configuration.
    #[serde(default)]
    pub cache: EmbeddingCacheConfig,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: None,
            model: None,
            base_url: None,
            dimension: None,
            timeout_ms: 60_000,
            batch_size: 32,
            local_first: false,
            local_only: false,
            onnx: OnnxEmbeddingConfig::default(),
            routing: EmbeddingRoutingConfig::default(),
            jobs: EmbeddingJobsConfig::default(),
            cache: EmbeddingCacheConfig::default(),
        }
    }
}

impl EmbeddingConfig {
    fn normalize(&mut self) {
        normalize_optional_trimmed(&mut self.provider);
        normalize_optional_trimmed(&mut self.model);
        normalize_optional_trimmed(&mut self.base_url);
        self.onnx.normalize();
        self.cache.normalize();
    }

    fn validate(&self) -> Result<(), ConfigSchemaError> {
        if let Some(url) = self.base_url.as_deref() {
            validate_http_url("embedding", "baseUrl", url)?;
        }
        if let Some(dimension) = self.dimension {
            validate_limit_u32(
                "embedding",
                "dimension",
                dimension,
                EMBEDDING_DIMENSION_MIN,
                EMBEDDING_DIMENSION_MAX,
            )?;
        }
        validate_timeout_ms(
            "embedding",
            "timeoutMs",
            self.timeout_ms,
            EMBEDDING_TIMEOUT_MIN_MS,
            EMBEDDING_TIMEOUT_MAX_MS,
        )?;
        validate_limit_u32(
            "embedding",
            "batchSize",
            self.batch_size,
            EMBEDDING_BATCH_SIZE_MIN,
            EMBEDDING_BATCH_SIZE_MAX,
        )?;
        validate_limit_u32(
            "embedding.onnx",
            "sessionPoolSize",
            self.onnx.session_pool_size,
            EMBEDDING_ONNX_SESSION_POOL_MIN,
            EMBEDDING_ONNX_SESSION_POOL_MAX,
        )?;
        self.routing.validate()?;
        self.jobs.validate()?;
        self.cache.validate()?;
        Ok(())
    }
}

/// Embedding cache configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
pub struct EmbeddingCacheConfig {
    /// Enable in-memory cache.
    pub enabled: bool,
    /// Maximum in-memory entries.
    pub max_entries: u32,
    /// Maximum in-memory bytes.
    pub max_bytes: u64,
    /// Enable disk cache.
    pub disk_enabled: bool,
    /// Disk cache provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disk_provider: Option<EmbeddingCacheDiskProvider>,
    /// Optional disk cache path (`SQLite` file).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disk_path: Option<Box<str>>,
    /// Optional disk cache connection string (Postgres/MySQL/MSSQL).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disk_connection: Option<Box<str>>,
    /// Optional disk cache table name override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disk_table: Option<Box<str>>,
    /// Maximum disk cache bytes.
    pub disk_max_bytes: u64,
}

impl Default for EmbeddingCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_entries: 2_048,
            max_bytes: 128 * 1024 * 1024,
            disk_enabled: false,
            disk_provider: None,
            disk_path: None,
            disk_connection: None,
            disk_table: None,
            disk_max_bytes: 1024 * 1024 * 1024,
        }
    }
}

impl EmbeddingCacheConfig {
    fn normalize(&mut self) {
        normalize_optional_trimmed(&mut self.disk_path);
        normalize_optional_trimmed(&mut self.disk_connection);
        normalize_optional_trimmed(&mut self.disk_table);
    }

    fn validate(&self) -> Result<(), ConfigSchemaError> {
        if self.enabled {
            validate_limit_u32(
                "embedding.cache",
                "maxEntries",
                self.max_entries,
                EMBEDDING_CACHE_MAX_ENTRIES_MIN,
                EMBEDDING_CACHE_MAX_ENTRIES_MAX,
            )?;
            validate_limit_u64(
                "embedding.cache",
                "maxBytes",
                self.max_bytes,
                EMBEDDING_CACHE_MAX_BYTES_MIN,
                EMBEDDING_CACHE_MAX_BYTES_MAX,
            )?;
        }
        if self.disk_enabled {
            validate_limit_u64(
                "embedding.cache",
                "diskMaxBytes",
                self.disk_max_bytes,
                EMBEDDING_CACHE_DISK_MAX_BYTES_MIN,
                EMBEDDING_CACHE_DISK_MAX_BYTES_MAX,
            )?;
            let provider = self
                .disk_provider
                .unwrap_or(EmbeddingCacheDiskProvider::Sqlite);
            if provider == EmbeddingCacheDiskProvider::Sqlite {
                if let Some(path) = self.disk_path.as_deref()
                    && path.is_empty()
                {
                    return Err(ConfigSchemaError::InvalidSnapshotStoragePath {
                        section: "embedding.cache",
                        field: "diskPath",
                        path: path.to_string(),
                    });
                }
                if let Some(connection) = self.disk_connection.as_deref()
                    && !connection.is_empty()
                {
                    return Err(ConfigSchemaError::InvalidCacheConfig {
                        section: "embedding.cache",
                        field: "diskConnection",
                        reason: "diskConnection is not valid for sqlite provider".to_string(),
                    });
                }
            } else {
                if let Some(path) = self.disk_path.as_deref()
                    && !path.is_empty()
                {
                    return Err(ConfigSchemaError::InvalidCacheConfig {
                        section: "embedding.cache",
                        field: "diskPath",
                        reason: "diskPath is only valid for sqlite provider".to_string(),
                    });
                }
                if let Some(connection) = self.disk_connection.as_deref()
                    && connection.is_empty()
                {
                    return Err(ConfigSchemaError::InvalidCacheConfig {
                        section: "embedding.cache",
                        field: "diskConnection",
                        reason: "diskConnection must be set for non-sqlite providers".to_string(),
                    });
                }
                if self.disk_connection.is_none() {
                    return Err(ConfigSchemaError::InvalidCacheConfig {
                        section: "embedding.cache",
                        field: "diskConnection",
                        reason: "diskConnection must be set for non-sqlite providers".to_string(),
                    });
                }
            }
            if let Some(table) = self.disk_table.as_deref()
                && table.is_empty()
            {
                return Err(ConfigSchemaError::InvalidCacheConfig {
                    section: "embedding.cache",
                    field: "diskTable",
                    reason: "diskTable must not be empty".to_string(),
                });
            }
            if let Some(table) = self.disk_table.as_deref()
                && !is_valid_identifier(table)
            {
                return Err(ConfigSchemaError::InvalidCacheConfig {
                    section: "embedding.cache",
                    field: "diskTable",
                    reason: "diskTable must contain only [A-Za-z0-9_]".to_string(),
                });
            }
        }
        Ok(())
    }
}

/// Disk cache provider options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EmbeddingCacheDiskProvider {
    /// `SQLite` file-based cache.
    Sqlite,
    /// Postgres-backed cache.
    Postgres,
    /// MySQL-backed cache.
    Mysql,
    /// Microsoft SQL Server-backed cache.
    Mssql,
}

/// Vector DB configuration (local/external selection is handled in later milestones).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
pub struct VectorDbConfig {
    /// Optional provider identifier (e.g. `local`, `milvus`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<Box<str>>,
    /// Optional address/host for the vector DB provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub address: Option<Box<str>>,
    /// Optional base URL for the vector DB provider (HTTP/REST).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<Box<str>>,
    /// Optional database name for the vector DB provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub database: Option<Box<str>>,
    /// Optional auth token (kept in memory; not serialized).
    #[serde(skip_serializing)]
    pub token: Option<Box<str>>,
    /// Optional auth username.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub username: Option<Box<str>>,
    /// Optional auth password (kept in memory; not serialized).
    #[serde(skip_serializing)]
    pub password: Option<Box<str>>,
    /// Whether to enforce TLS/SSL for gRPC connections.
    pub ssl: bool,
    /// Indexing mode used for collection naming decisions.
    pub index_mode: IndexMode,
    /// Timeout for vector DB operations (ms).
    pub timeout_ms: u64,
    /// Timeout for vector DB index builds (ms).
    pub index_timeout_ms: u64,
    /// Index configuration for dense and sparse vector fields.
    pub index: VectorDbIndexConfig,
    /// Batch size for inserts/deletes.
    pub batch_size: u32,
    /// Snapshot persistence mode for local vector DBs.
    pub snapshot_storage: SnapshotStorageMode,
}

impl Default for VectorDbConfig {
    fn default() -> Self {
        Self {
            provider: None,
            address: None,
            base_url: None,
            database: None,
            token: None,
            username: None,
            password: None,
            ssl: false,
            index_mode: IndexMode::Dense,
            timeout_ms: 60_000,
            index_timeout_ms: 60_000,
            index: VectorDbIndexConfig::default(),
            batch_size: 128,
            snapshot_storage: SnapshotStorageMode::default(),
        }
    }
}

/// Milvus index configuration (dense + sparse).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VectorDbIndexConfig {
    /// Dense vector index configuration.
    #[serde(default = "VectorDbIndexSpec::dense_default")]
    pub dense: VectorDbIndexSpec,
    /// Sparse vector index configuration (hybrid mode).
    #[serde(default = "VectorDbIndexSpec::sparse_default")]
    pub sparse: VectorDbIndexSpec,
}

impl Default for VectorDbIndexConfig {
    fn default() -> Self {
        Self {
            dense: VectorDbIndexSpec::dense_default(),
            sparse: VectorDbIndexSpec::sparse_default(),
        }
    }
}

impl VectorDbIndexConfig {
    fn normalize(&mut self) {
        self.dense.normalize();
        self.sparse.normalize();
    }

    fn validate(&self) -> Result<(), ConfigSchemaError> {
        validate_index_spec("vectorDb", "index.dense", &self.dense)?;
        validate_index_spec("vectorDb", "index.sparse", &self.sparse)?;
        Ok(())
    }
}

/// Vector DB index spec for a single field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VectorDbIndexSpec {
    /// Milvus index type (e.g. `AUTOINDEX`, `HNSW`).
    pub index_type: Box<str>,
    /// Milvus metric type (e.g. `COSINE`, `IP`, `BM25`).
    pub metric_type: Box<str>,
    /// Index build parameters.
    #[serde(
        default,
        deserialize_with = "deserialize_index_params",
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub params: BTreeMap<Box<str>, Box<str>>,
}

impl VectorDbIndexSpec {
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

    fn normalize(&mut self) {
        normalize_boxed_str(&mut self.index_type);
        normalize_boxed_str(&mut self.metric_type);
        normalize_index_params(&mut self.params);
    }
}

impl VectorDbConfig {
    fn normalize(&mut self) {
        normalize_optional_trimmed(&mut self.provider);
        normalize_optional_trimmed(&mut self.address);
        normalize_optional_trimmed(&mut self.base_url);
        normalize_optional_trimmed(&mut self.database);
        normalize_optional_trimmed(&mut self.token);
        normalize_optional_trimmed(&mut self.username);
        normalize_optional_trimmed(&mut self.password);
        self.index.normalize();
    }

    fn validate(&self) -> Result<(), ConfigSchemaError> {
        if let Some(url) = self.base_url.as_deref() {
            validate_http_url("vectorDb", "baseUrl", url)?;
        }
        validate_timeout_ms(
            "vectorDb",
            "timeoutMs",
            self.timeout_ms,
            VECTOR_DB_TIMEOUT_MIN_MS,
            VECTOR_DB_TIMEOUT_MAX_MS,
        )?;
        validate_timeout_ms(
            "vectorDb",
            "indexTimeoutMs",
            self.index_timeout_ms,
            VECTOR_DB_INDEX_TIMEOUT_MIN_MS,
            VECTOR_DB_INDEX_TIMEOUT_MAX_MS,
        )?;
        validate_limit_u32(
            "vectorDb",
            "batchSize",
            self.batch_size,
            VECTOR_DB_BATCH_SIZE_MIN,
            VECTOR_DB_BATCH_SIZE_MAX,
        )?;
        validate_snapshot_storage("vectorDb", "snapshotStorage", &self.snapshot_storage)?;
        self.index.validate()?;
        Ok(())
    }
}

/// File sync and scanning configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
pub struct SyncConfig {
    /// Allowlist of file extensions (no leading dot).
    pub allowed_extensions: Vec<Box<str>>,
    /// Ignore patterns applied during scan.
    pub ignore_patterns: Vec<Box<str>>,
    /// Maximum number of files considered during a scan.
    pub max_files: u32,
    /// Maximum file size (bytes) for reading contents.
    pub max_file_size_bytes: u64,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            allowed_extensions: default_allowed_extensions(),
            ignore_patterns: default_ignore_patterns(),
            max_files: 250_000,
            max_file_size_bytes: 2_000_000,
        }
    }
}

impl SyncConfig {
    fn normalize_and_validate(&mut self) -> Result<(), ConfigSchemaError> {
        self.allowed_extensions = normalize_extensions(&self.allowed_extensions)?;
        self.ignore_patterns = normalize_ignore_patterns(&self.ignore_patterns)?;

        if self.allowed_extensions.len() > SYNC_ALLOWED_EXTENSIONS_MAX {
            return Err(ConfigSchemaError::ListTooLarge {
                section: "sync",
                field: "allowedExtensions",
                len: self.allowed_extensions.len(),
                max: SYNC_ALLOWED_EXTENSIONS_MAX,
            });
        }

        if self.ignore_patterns.len() > SYNC_IGNORE_PATTERNS_MAX {
            return Err(ConfigSchemaError::ListTooLarge {
                section: "sync",
                field: "ignorePatterns",
                len: self.ignore_patterns.len(),
                max: SYNC_IGNORE_PATTERNS_MAX,
            });
        }

        validate_limit_u32(
            "sync",
            "maxFiles",
            self.max_files,
            SYNC_MAX_FILES_MIN,
            SYNC_MAX_FILES_MAX,
        )?;
        validate_limit_u64(
            "sync",
            "maxFileSizeBytes",
            self.max_file_size_bytes,
            SYNC_MAX_FILE_SIZE_MIN_BYTES,
            SYNC_MAX_FILE_SIZE_MAX_BYTES,
        )?;

        Ok(())
    }
}

/// Typed validation errors for the configuration schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigSchemaError {
    /// The config version is not supported by this binary.
    UnsupportedVersion {
        /// Version found in the config.
        found: u32,
        /// Version supported by this crate.
        supported: u32,
    },
    /// A timeout value is out of bounds.
    TimeoutOutOfRange {
        /// Schema section (e.g. `core`).
        section: &'static str,
        /// Field name in the config file (e.g. `timeoutMs`).
        field: &'static str,
        /// Value provided (ms).
        value_ms: u64,
        /// Minimum allowed value (ms).
        min_ms: u64,
        /// Maximum allowed value (ms).
        max_ms: u64,
    },
    /// A numeric limit is out of bounds.
    LimitOutOfRange {
        /// Schema section (e.g. `sync`).
        section: &'static str,
        /// Field name in the config file (e.g. `maxFiles`).
        field: &'static str,
        /// Value provided.
        value: u64,
        /// Minimum allowed value.
        min: u64,
        /// Maximum allowed value.
        max: u64,
    },
    /// A list field exceeds the maximum allowed size.
    ListTooLarge {
        /// Schema section (e.g. `sync`).
        section: &'static str,
        /// Field name in the config file (e.g. `ignorePatterns`).
        field: &'static str,
        /// Number of entries after normalization/deduplication.
        len: usize,
        /// Maximum allowed number of entries.
        max: usize,
    },
    /// A file extension entry is invalid.
    InvalidExtension {
        /// Invalid extension value.
        extension: String,
    },
    /// An ignore pattern entry is invalid.
    InvalidIgnorePattern {
        /// Invalid ignore pattern value.
        pattern: String,
    },
    /// A URL entry is invalid.
    InvalidUrl {
        /// Schema section (e.g. `embedding`).
        section: &'static str,
        /// Field name in the config file (e.g. `baseUrl`).
        field: &'static str,
        /// Invalid URL value.
        url: String,
    },
    /// Snapshot storage path is invalid.
    InvalidSnapshotStoragePath {
        /// Schema section (e.g. `vectorDb`).
        section: &'static str,
        /// Field name in the config file (e.g. `snapshotStorage`).
        field: &'static str,
        /// Invalid path value.
        path: String,
    },
    /// A cache config value is invalid.
    InvalidCacheConfig {
        /// Schema section (e.g. `embedding.cache`).
        section: &'static str,
        /// Field name in the config file.
        field: &'static str,
        /// Human readable reason.
        reason: String,
    },
    /// An index config value is invalid.
    InvalidIndexConfig {
        /// Schema section (e.g. `vectorDb`).
        section: &'static str,
        /// Field name in the config file.
        field: &'static str,
        /// Human readable reason.
        reason: String,
    },
}

impl ConfigSchemaError {
    fn error_code(&self) -> ErrorCode {
        match self {
            Self::UnsupportedVersion { .. } => ErrorCode::new("config", "unsupported_version"),
            Self::TimeoutOutOfRange { .. } => ErrorCode::new("config", "invalid_timeout"),
            Self::LimitOutOfRange { .. } => ErrorCode::new("config", "invalid_limit"),
            Self::ListTooLarge { .. } => ErrorCode::new("config", "list_too_large"),
            Self::InvalidExtension { .. } => ErrorCode::new("config", "invalid_extension"),
            Self::InvalidIgnorePattern { .. } => ErrorCode::new("config", "invalid_ignore_pattern"),
            Self::InvalidUrl { .. } => ErrorCode::new("config", "invalid_url"),
            Self::InvalidSnapshotStoragePath { .. } => {
                ErrorCode::new("config", "invalid_snapshot_storage")
            },
            Self::InvalidCacheConfig { .. } => ErrorCode::new("config", "invalid_cache_config"),
            Self::InvalidIndexConfig { .. } => ErrorCode::new("config", "invalid_index_config"),
        }
    }
}

impl fmt::Display for ConfigSchemaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion { found, supported } => {
                write!(
                    formatter,
                    "unsupported config version: {found} (supported: {supported})"
                )
            },
            Self::TimeoutOutOfRange {
                section,
                field,
                value_ms,
                min_ms,
                max_ms,
            } => write!(
                formatter,
                "{section}.{field} must be within [{min_ms}, {max_ms}] ms (got {value_ms})"
            ),
            Self::LimitOutOfRange {
                section,
                field,
                value,
                min,
                max,
            } => write!(
                formatter,
                "{section}.{field} must be within [{min}, {max}] (got {value})"
            ),
            Self::ListTooLarge {
                section,
                field,
                len,
                max,
            } => write!(
                formatter,
                "{section}.{field} must have at most {max} entries (got {len})"
            ),
            Self::InvalidExtension { extension } => {
                write!(formatter, "invalid extension entry: {extension}")
            },
            Self::InvalidIgnorePattern { pattern } => {
                write!(formatter, "invalid ignore pattern entry: {pattern}")
            },
            Self::InvalidUrl { section, field, .. } => {
                write!(formatter, "invalid URL for {section}.{field}")
            },
            Self::InvalidSnapshotStoragePath { section, field, .. } => {
                write!(
                    formatter,
                    "invalid snapshot storage path for {section}.{field}"
                )
            },
            Self::InvalidCacheConfig {
                section,
                field,
                reason,
            } => write!(
                formatter,
                "invalid cache config for {section}.{field}: {reason}"
            ),
            Self::InvalidIndexConfig {
                section,
                field,
                reason,
            } => write!(
                formatter,
                "invalid index config for {section}.{field}: {reason}"
            ),
        }
    }
}

impl std::error::Error for ConfigSchemaError {}

impl From<ConfigSchemaError> for ErrorEnvelope {
    fn from(error: ConfigSchemaError) -> Self {
        let code = error.error_code();
        let message = error.to_string();
        let mut envelope = Self::expected(code, message);

        match error {
            ConfigSchemaError::UnsupportedVersion { found, supported } => {
                envelope = envelope
                    .with_metadata("found", found.to_string())
                    .with_metadata("supported", supported.to_string());
            },
            ConfigSchemaError::TimeoutOutOfRange {
                section,
                field,
                value_ms,
                min_ms,
                max_ms,
            } => {
                envelope = envelope
                    .with_metadata("section", section)
                    .with_metadata("field", field)
                    .with_metadata("value_ms", value_ms.to_string())
                    .with_metadata("min_ms", min_ms.to_string())
                    .with_metadata("max_ms", max_ms.to_string());
            },
            ConfigSchemaError::LimitOutOfRange {
                section,
                field,
                value,
                min,
                max,
            } => {
                envelope = envelope
                    .with_metadata("section", section)
                    .with_metadata("field", field)
                    .with_metadata("value", value.to_string())
                    .with_metadata("min", min.to_string())
                    .with_metadata("max", max.to_string());
            },
            ConfigSchemaError::ListTooLarge {
                section,
                field,
                len,
                max,
            } => {
                envelope = envelope
                    .with_metadata("section", section)
                    .with_metadata("field", field)
                    .with_metadata("len", len.to_string())
                    .with_metadata("max", max.to_string());
            },
            ConfigSchemaError::InvalidExtension { extension } => {
                envelope = envelope.with_metadata("extension", extension);
            },
            ConfigSchemaError::InvalidIgnorePattern { pattern } => {
                envelope = envelope.with_metadata("pattern", pattern);
            },
            ConfigSchemaError::InvalidUrl {
                section,
                field,
                url,
            } => {
                envelope = envelope
                    .with_metadata("section", section)
                    .with_metadata("field", field)
                    .with_metadata("url", sanitize_url_for_error(&url));
            },
            ConfigSchemaError::InvalidSnapshotStoragePath {
                section,
                field,
                path,
            } => {
                envelope = envelope
                    .with_metadata("section", section)
                    .with_metadata("field", field)
                    .with_metadata("path", path);
            },
            ConfigSchemaError::InvalidCacheConfig {
                section,
                field,
                reason,
            }
            | ConfigSchemaError::InvalidIndexConfig {
                section,
                field,
                reason,
            } => {
                envelope = envelope
                    .with_metadata("section", section)
                    .with_metadata("field", field)
                    .with_metadata("reason", reason);
            },
        }

        envelope
    }
}

fn validate_index_spec(
    section: &'static str,
    field_prefix: &'static str,
    spec: &VectorDbIndexSpec,
) -> Result<(), ConfigSchemaError> {
    if spec.index_type.trim().is_empty() {
        return Err(ConfigSchemaError::InvalidIndexConfig {
            section,
            field: "indexType",
            reason: format!("{field_prefix}.indexType is required"),
        });
    }
    if spec.metric_type.trim().is_empty() {
        return Err(ConfigSchemaError::InvalidIndexConfig {
            section,
            field: "metricType",
            reason: format!("{field_prefix}.metricType is required"),
        });
    }
    if spec.params.len() > VECTOR_DB_INDEX_PARAMS_MAX {
        return Err(ConfigSchemaError::InvalidIndexConfig {
            section,
            field: "params",
            reason: format!(
                "{field_prefix}.params exceeds max entries ({VECTOR_DB_INDEX_PARAMS_MAX})"
            ),
        });
    }
    for (key, value) in &spec.params {
        if key.trim().is_empty() {
            return Err(ConfigSchemaError::InvalidIndexConfig {
                section,
                field: "params",
                reason: format!("{field_prefix}.params has empty key"),
            });
        }
        if value.trim().is_empty() {
            return Err(ConfigSchemaError::InvalidIndexConfig {
                section,
                field: "params",
                reason: format!("{field_prefix}.params has empty value for key {key}"),
            });
        }
    }
    Ok(())
}

const fn validate_timeout_ms(
    section: &'static str,
    field: &'static str,
    value_ms: u64,
    min_ms: u64,
    max_ms: u64,
) -> Result<(), ConfigSchemaError> {
    if value_ms < min_ms || value_ms > max_ms {
        return Err(ConfigSchemaError::TimeoutOutOfRange {
            section,
            field,
            value_ms,
            min_ms,
            max_ms,
        });
    }
    Ok(())
}

fn bounded_u32<const MIN: u32, const MAX: u32>(
    section: &'static str,
    field: &'static str,
    value: u32,
    min: u32,
    max: u32,
) -> Result<BoundedU32<MIN, MAX>, ConfigSchemaError> {
    BoundedU32::try_new(value).map_err(|_| ConfigSchemaError::LimitOutOfRange {
        section,
        field,
        value: u64::from(value),
        min: u64::from(min),
        max: u64::from(max),
    })
}

fn bounded_opt_u32<const MIN: u32, const MAX: u32>(
    section: &'static str,
    field: &'static str,
    value: Option<u32>,
    min: u32,
    max: u32,
) -> Result<Option<BoundedU32<MIN, MAX>>, ConfigSchemaError> {
    value
        .map(|value| bounded_u32::<MIN, MAX>(section, field, value, min, max))
        .transpose()
}

fn bounded_u64<const MIN: u64, const MAX: u64>(
    section: &'static str,
    field: &'static str,
    value: u64,
    min: u64,
    max: u64,
) -> Result<BoundedU64<MIN, MAX>, ConfigSchemaError> {
    BoundedU64::try_new(value).map_err(|_| ConfigSchemaError::LimitOutOfRange {
        section,
        field,
        value,
        min,
        max,
    })
}

fn validate_http_url(
    section: &'static str,
    field: &'static str,
    value: &str,
) -> Result<(), ConfigSchemaError> {
    let parsed = Url::parse(value).map_err(|_| ConfigSchemaError::InvalidUrl {
        section,
        field,
        url: value.to_owned(),
    })?;

    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        return Err(ConfigSchemaError::InvalidUrl {
            section,
            field,
            url: value.to_owned(),
        });
    }

    Ok(())
}

fn validate_limit_u32(
    section: &'static str,
    field: &'static str,
    value: u32,
    min: u32,
    max: u32,
) -> Result<(), ConfigSchemaError> {
    if value < min || value > max {
        return Err(ConfigSchemaError::LimitOutOfRange {
            section,
            field,
            value: u64::from(value),
            min: u64::from(min),
            max: u64::from(max),
        });
    }
    Ok(())
}

fn validate_optional_limit_u32(
    section: &'static str,
    field: &'static str,
    value: Option<u32>,
    min: u32,
    max: u32,
) -> Result<(), ConfigSchemaError> {
    if let Some(value) = value {
        validate_limit_u32(section, field, value, min, max)?;
    }
    Ok(())
}

const fn validate_limit_u64(
    section: &'static str,
    field: &'static str,
    value: u64,
    min: u64,
    max: u64,
) -> Result<(), ConfigSchemaError> {
    if value < min || value > max {
        return Err(ConfigSchemaError::LimitOutOfRange {
            section,
            field,
            value,
            min,
            max,
        });
    }
    Ok(())
}

fn validate_snapshot_storage(
    section: &'static str,
    field: &'static str,
    mode: &SnapshotStorageMode,
) -> Result<(), ConfigSchemaError> {
    let SnapshotStorageMode::Custom(path) = mode else {
        return Ok(());
    };

    if path.as_os_str().is_empty() || !path.is_absolute() {
        return Err(ConfigSchemaError::InvalidSnapshotStoragePath {
            section,
            field,
            path: path.display().to_string(),
        });
    }

    Ok(())
}

fn normalize_optional_trimmed(value: &mut Option<Box<str>>) {
    let Some(raw) = value.take() else {
        return;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        *value = None;
    } else {
        *value = Some(trimmed.to_owned().into_boxed_str());
    }
}

fn normalize_boxed_str(value: &mut Box<str>) {
    let trimmed = value.trim();
    if trimmed == value.as_ref() {
        return;
    }
    *value = trimmed.to_owned().into_boxed_str();
}

fn normalize_index_params(params: &mut BTreeMap<Box<str>, Box<str>>) {
    if params.is_empty() {
        return;
    }
    let mut normalized = BTreeMap::new();
    for (key, value) in std::mem::take(params) {
        let key_trimmed = key.trim();
        let value_trimmed = value.trim();
        normalized.insert(
            key_trimmed.to_owned().into_boxed_str(),
            value_trimmed.to_owned().into_boxed_str(),
        );
    }
    *params = normalized;
}

fn deserialize_index_params<'de, D>(
    deserializer: D,
) -> Result<BTreeMap<Box<str>, Box<str>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw: BTreeMap<String, serde_json::Value> = BTreeMap::deserialize(deserializer)?;
    let mut out = BTreeMap::new();
    for (key, value) in raw {
        let string_value = match value {
            serde_json::Value::String(value) => value,
            serde_json::Value::Number(value) => value.to_string(),
            serde_json::Value::Bool(value) => value.to_string(),
            serde_json::Value::Null => {
                return Err(de::Error::custom("index param values cannot be null"));
            },
            serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                return Err(de::Error::custom(
                    "index param values must be string/number/bool",
                ));
            },
        };
        out.insert(
            key.trim().to_owned().into_boxed_str(),
            string_value.trim().to_owned().into_boxed_str(),
        );
    }
    Ok(out)
}

fn is_valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn normalize_extensions(input: &[Box<str>]) -> Result<Vec<Box<str>>, ConfigSchemaError> {
    let mut normalized = Vec::with_capacity(input.len());
    for ext in input {
        let raw = ext.as_ref();
        let trimmed = raw.trim();
        let trimmed = trimmed.strip_prefix("*.").unwrap_or(trimmed);
        let trimmed = trimmed.strip_prefix('.').unwrap_or(trimmed);
        let candidate = trimmed.to_ascii_lowercase();

        if candidate.is_empty() || !candidate.chars().all(|ch| ch.is_ascii_alphanumeric()) {
            return Err(ConfigSchemaError::InvalidExtension {
                extension: trimmed.to_owned(),
            });
        }

        normalized.push(candidate.into_boxed_str());
    }

    normalized.sort_unstable();
    normalized.dedup();
    Ok(normalized)
}

fn normalize_ignore_patterns(input: &[Box<str>]) -> Result<Vec<Box<str>>, ConfigSchemaError> {
    let mut normalized = Vec::with_capacity(input.len());
    for pattern in input {
        let raw = pattern.as_ref().trim();
        if raw.is_empty() {
            return Err(ConfigSchemaError::InvalidIgnorePattern {
                pattern: pattern.as_ref().to_owned(),
            });
        }
        let replaced = raw.replace('\\', "/");
        let collapsed = collapse_forward_slashes(&replaced);
        normalized.push(collapsed.into_boxed_str());
    }

    if !normalized
        .iter()
        .any(|pattern| pattern.as_ref() == ".context/")
    {
        normalized.push(".context/".into());
    }

    normalized.sort_unstable();
    normalized.dedup();
    Ok(normalized)
}

fn collapse_forward_slashes(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut previous_was_slash = false;

    for ch in input.chars() {
        if ch == '/' {
            if previous_was_slash {
                continue;
            }
            previous_was_slash = true;
        } else {
            previous_was_slash = false;
        }
        output.push(ch);
    }

    output
}

fn default_allowed_extensions() -> Vec<Box<str>> {
    // Keep this list small and stable; Phase 05 expands language/splitter support.
    vec![
        "c".into(),
        "cpp".into(),
        "go".into(),
        "java".into(),
        "js".into(),
        "jsx".into(),
        "md".into(),
        "py".into(),
        "rs".into(),
        "ts".into(),
        "tsx".into(),
    ]
}

fn default_ignore_patterns() -> Vec<Box<str>> {
    vec![
        ".context/".into(),
        ".git/".into(),
        "node_modules/".into(),
        "target/".into(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn defaults_are_applied() -> Result<(), Box<dyn Error>> {
        let config = parse_backend_config_json("{}")?;

        assert_eq!(config.version, CURRENT_CONFIG_VERSION);
        assert_eq!(config.core, CoreConfig::default());
        assert_eq!(config.embedding, EmbeddingConfig::default());
        assert_eq!(config.vector_db, VectorDbConfig::default());
        assert_eq!(config.sync, SyncConfig::default());

        Ok(())
    }

    #[test]
    fn invalid_timeout_returns_error_code() -> Result<(), Box<dyn Error>> {
        let payload = serde_json::json!({
            "version": 1,
            "core": { "timeoutMs": 0 }
        });

        let result = parse_backend_config_json(&payload.to_string());
        assert!(result.is_err());

        let error = result
            .err()
            .ok_or_else(|| std::io::Error::other("expected validation error"))?;
        assert_eq!(error.code, ErrorCode::new("config", "invalid_timeout"));
        assert_eq!(
            error.metadata.get("section").map(String::as_str),
            Some("core")
        );
        assert_eq!(
            error.metadata.get("field").map(String::as_str),
            Some("timeoutMs")
        );

        Ok(())
    }

    #[test]
    fn normalization_is_deterministic() -> Result<(), Box<dyn Error>> {
        let payload = serde_json::json!({
            "version": 1,
            "sync": {
                "allowedExtensions": [" TSX ", ".rs", "ts", "*.RS", "tsx"],
                "ignorePatterns": [" target/", "node_modules/", "dist\\\\", "node_modules/"]
            }
        });
        let config = parse_backend_config_json(&payload.to_string())?;

        let extensions: Vec<&str> = config
            .sync
            .allowed_extensions
            .iter()
            .map(AsRef::as_ref)
            .collect();
        assert_eq!(extensions, vec!["rs", "ts", "tsx"]);

        let patterns: Vec<&str> = config
            .sync
            .ignore_patterns
            .iter()
            .map(AsRef::as_ref)
            .collect();
        assert_eq!(
            patterns,
            vec![".context/", "dist/", "node_modules/", "target/"]
        );

        Ok(())
    }

    #[test]
    fn max_list_sizes_are_enforced() -> Result<(), Box<dyn Error>> {
        let extensions: Vec<String> = (0..=SYNC_ALLOWED_EXTENSIONS_MAX)
            .map(|idx| format!("ext{idx}"))
            .collect();

        let payload = serde_json::json!({
            "version": 1,
            "sync": { "allowedExtensions": extensions }
        });

        let result = parse_backend_config_json(&payload.to_string());
        assert!(result.is_err());

        let error = result
            .err()
            .ok_or_else(|| std::io::Error::other("expected list size error"))?;
        assert_eq!(error.code, ErrorCode::new("config", "list_too_large"));
        assert_eq!(
            error.metadata.get("field").map(String::as_str),
            Some("allowedExtensions")
        );

        Ok(())
    }

    #[test]
    fn sanitize_url_strips_credentials() {
        // URL with username and password
        let sanitized = sanitize_url_for_error("https://user:secret@example.com/api");
        assert_eq!(sanitized, "https://example.com/api");

        // URL with only username
        let sanitized = sanitize_url_for_error("https://apikey@example.com/api");
        assert_eq!(sanitized, "https://example.com/api");

        // URL without credentials (unchanged)
        let sanitized = sanitize_url_for_error("https://example.com/api");
        assert_eq!(sanitized, "https://example.com/api");

        // Invalid URL returns placeholder with reason
        let sanitized = sanitize_url_for_error("not a valid url");
        assert!(
            sanitized.starts_with("[invalid url:"),
            "invalid url should include reason"
        );
    }
}
