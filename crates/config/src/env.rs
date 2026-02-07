//! Environment variable parsing and env-to-config merging.
//!
//! This module keeps env parsing:
//! - strict (invalid values fail fast)
//! - deterministic (CSV lists normalize to sorted/deduped values)
//! - safe (secret values are redacted in error metadata)

use crate::schema::{
    BackendConfig, EmbeddingCacheDiskProvider, EmbeddingRoutingMode, ValidatedBackendConfig,
};
use semantic_code_domain::IndexMode;
use semantic_code_shared::{ErrorCode, ErrorEnvelope, REDACTED_VALUE, SecretString, is_secret_key};
use std::collections::BTreeMap;
use std::fmt;
use url::Url;

/// Env var: core timeout in milliseconds.
pub const ENV_CORE_TIMEOUT_MS: &str = "SCA_CORE_TIMEOUT_MS";
/// Env var: core max concurrency.
pub const ENV_CORE_MAX_CONCURRENCY: &str = "SCA_CORE_MAX_CONCURRENCY";
/// Env var: max in-flight files.
pub const ENV_CORE_MAX_IN_FLIGHT_FILES: &str = "SCA_CORE_MAX_IN_FLIGHT_FILES";
/// Env var: max in-flight embedding batches.
pub const ENV_CORE_MAX_IN_FLIGHT_EMBEDDING_BATCHES: &str =
    "SCA_CORE_MAX_IN_FLIGHT_EMBEDDING_BATCHES";
/// Env var: max in-flight inserts.
pub const ENV_CORE_MAX_IN_FLIGHT_INSERTS: &str = "SCA_CORE_MAX_IN_FLIGHT_INSERTS";
/// Env var: max buffered chunks.
pub const ENV_CORE_MAX_BUFFERED_CHUNKS: &str = "SCA_CORE_MAX_BUFFERED_CHUNKS";
/// Env var: max buffered embeddings.
pub const ENV_CORE_MAX_BUFFERED_EMBEDDINGS: &str = "SCA_CORE_MAX_BUFFERED_EMBEDDINGS";
/// Env var: max chunk chars.
pub const ENV_CORE_MAX_CHUNK_CHARS: &str = "SCA_CORE_MAX_CHUNK_CHARS";
/// Env var: retry max attempts.
pub const ENV_CORE_RETRY_MAX_ATTEMPTS: &str = "SCA_CORE_RETRY_MAX_ATTEMPTS";
/// Env var: retry base delay in ms.
pub const ENV_CORE_RETRY_BASE_DELAY_MS: &str = "SCA_CORE_RETRY_BASE_DELAY_MS";
/// Env var: retry max delay in ms.
pub const ENV_CORE_RETRY_MAX_DELAY_MS: &str = "SCA_CORE_RETRY_MAX_DELAY_MS";
/// Env var: retry jitter ratio percent.
pub const ENV_CORE_RETRY_JITTER_RATIO_PCT: &str = "SCA_CORE_RETRY_JITTER_RATIO_PCT";

/// Env var: embedding provider identifier.
pub const ENV_EMBEDDING_PROVIDER: &str = "SCA_EMBEDDING_PROVIDER";
/// Env var: embedding provider identifier (alias).
pub const ENV_EMBEDDING_PROVIDER_ALIAS: &str = "EMBEDDING_PROVIDER";
/// Env var: embedding model override.
pub const ENV_EMBEDDING_MODEL: &str = "SCA_EMBEDDING_MODEL";
/// Env var: embedding model override (alias).
pub const ENV_EMBEDDING_MODEL_ALIAS: &str = "EMBEDDING_MODEL";
/// Env var: embedding timeout in milliseconds.
pub const ENV_EMBEDDING_TIMEOUT_MS: &str = "SCA_EMBEDDING_TIMEOUT_MS";
/// Env var: embedding timeout in milliseconds (alias).
pub const ENV_EMBEDDING_TIMEOUT_MS_ALIAS: &str = "EMBEDDING_TIMEOUT_MS";
/// Env var: embedding batch size.
pub const ENV_EMBEDDING_BATCH_SIZE: &str = "SCA_EMBEDDING_BATCH_SIZE";
/// Env var: embedding batch size (alias).
pub const ENV_EMBEDDING_BATCH_SIZE_ALIAS: &str = "EMBEDDING_BATCH_SIZE";
/// Env var: embedding dimension override.
pub const ENV_EMBEDDING_DIMENSION: &str = "SCA_EMBEDDING_DIMENSION";
/// Env var: embedding dimension override (alias).
pub const ENV_EMBEDDING_DIMENSION_ALIAS: &str = "EMBEDDING_DIMENSION";
/// Env var: embedding base URL.
pub const ENV_EMBEDDING_BASE_URL: &str = "SCA_EMBEDDING_BASE_URL";
/// Env var: embedding base URL (alias).
pub const ENV_EMBEDDING_BASE_URL_ALIAS: &str = "EMBEDDING_BASE_URL";
/// Env var: embedding API key (secret).
pub const ENV_EMBEDDING_API_AUTH: &str = "SCA_EMBEDDING_API_KEY";
/// Env var: embedding API key (alias).
pub const ENV_EMBEDDING_API_AUTH_ALIAS: &str = "EMBEDDING_API_KEY";
/// Env var: prefer local ONNX embeddings.
pub const ENV_EMBEDDING_LOCAL_FIRST: &str = "SCA_EMBEDDING_LOCAL_FIRST";
/// Env var: prefer local ONNX embeddings (alias).
pub const ENV_EMBEDDING_LOCAL_FIRST_ALIAS: &str = "EMBEDDING_LOCAL_FIRST";
/// Env var: force local ONNX embeddings only.
pub const ENV_EMBEDDING_LOCAL_ONLY: &str = "SCA_EMBEDDING_LOCAL_ONLY";
/// Env var: force local ONNX embeddings only (alias).
pub const ENV_EMBEDDING_LOCAL_ONLY_ALIAS: &str = "EMBEDDING_LOCAL_ONLY";
/// Env var: embedding routing mode.
pub const ENV_EMBEDDING_ROUTING_MODE: &str = "SCA_EMBEDDING_ROUTING_MODE";
/// Env var: embedding routing mode (alias).
pub const ENV_EMBEDDING_ROUTING_MODE_ALIAS: &str = "EMBEDDING_ROUTING_MODE";
/// Env var: max remote embedding batches for split routing.
pub const ENV_EMBEDDING_SPLIT_MAX_REMOTE_BATCHES: &str = "SCA_EMBEDDING_SPLIT_MAX_REMOTE_BATCHES";
/// Env var: max remote embedding batches for split routing (alias).
pub const ENV_EMBEDDING_SPLIT_MAX_REMOTE_BATCHES_ALIAS: &str = "EMBEDDING_SPLIT_MAX_REMOTE_BATCHES";
/// Env var: background job progress interval.
pub const ENV_EMBEDDING_JOBS_PROGRESS_INTERVAL_MS: &str = "SCA_EMBEDDING_JOBS_PROGRESS_INTERVAL_MS";
/// Env var: background job progress interval (alias).
pub const ENV_EMBEDDING_JOBS_PROGRESS_INTERVAL_MS_ALIAS: &str =
    "EMBEDDING_JOBS_PROGRESS_INTERVAL_MS";
/// Env var: background job cancel poll interval.
pub const ENV_EMBEDDING_JOBS_CANCEL_POLL_INTERVAL_MS: &str =
    "SCA_EMBEDDING_JOBS_CANCEL_POLL_INTERVAL_MS";
/// Env var: background job cancel poll interval (alias).
pub const ENV_EMBEDDING_JOBS_CANCEL_POLL_INTERVAL_MS_ALIAS: &str =
    "EMBEDDING_JOBS_CANCEL_POLL_INTERVAL_MS";
/// Env var: allow fallback to test embeddings when ONNX assets are missing.
pub const ENV_EMBEDDING_TEST_FALLBACK: &str = "SCA_EMBEDDING_TEST_FALLBACK";
/// Env var: allow fallback to test embeddings when ONNX assets are missing (alias).
pub const ENV_EMBEDDING_TEST_FALLBACK_ALIAS: &str = "EMBEDDING_TEST_FALLBACK";
/// Env var: ONNX model directory override.
pub const ENV_EMBEDDING_ONNX_MODEL_DIR: &str = "SCA_EMBEDDING_ONNX_MODEL_DIR";
/// Env var: ONNX model directory override (alias).
pub const ENV_EMBEDDING_ONNX_MODEL_DIR_ALIAS: &str = "EMBEDDING_ONNX_MODEL_DIR";
/// Env var: ONNX model filename override.
pub const ENV_EMBEDDING_ONNX_MODEL_FILENAME: &str = "SCA_EMBEDDING_ONNX_MODEL_FILENAME";
/// Env var: ONNX model filename override (alias).
pub const ENV_EMBEDDING_ONNX_MODEL_FILENAME_ALIAS: &str = "EMBEDDING_ONNX_MODEL_FILENAME";
/// Env var: ONNX tokenizer filename override.
pub const ENV_EMBEDDING_ONNX_TOK_FILENAME: &str = "SCA_EMBEDDING_ONNX_TOKENIZER_FILENAME";
/// Env var: ONNX tokenizer filename override (alias).
pub const ENV_EMBEDDING_ONNX_TOK_FILENAME_ALIAS: &str = "EMBEDDING_ONNX_TOKENIZER_FILENAME";
/// Env var: ONNX repo ID override.
pub const ENV_EMBEDDING_ONNX_REPO: &str = "SCA_EMBEDDING_ONNX_REPO";
/// Env var: ONNX repo ID override (alias).
pub const ENV_EMBEDDING_ONNX_REPO_ALIAS: &str = "EMBEDDING_ONNX_REPO";
/// Env var: download missing ONNX assets.
pub const ENV_EMBEDDING_ONNX_DOWNLOAD: &str = "SCA_EMBEDDING_ONNX_DOWNLOAD";
/// Env var: download missing ONNX assets (alias).
pub const ENV_EMBEDDING_ONNX_DOWNLOAD_ALIAS: &str = "EMBEDDING_ONNX_DOWNLOAD";
/// Env var: ONNX session pool size.
pub const ENV_EMBEDDING_ONNX_SESSION_POOL_SIZE: &str = "SCA_EMBEDDING_ONNX_SESSION_POOL_SIZE";
/// Env var: ONNX session pool size (alias).
pub const ENV_EMBEDDING_ONNX_SESSION_POOL_SIZE_ALIAS: &str = "EMBEDDING_ONNX_SESSION_POOL_SIZE";
/// Env var: embedding cache enabled.
pub const ENV_EMBEDDING_CACHE_ENABLED: &str = "SCA_EMBEDDING_CACHE_ENABLED";
/// Env var: embedding cache max entries.
pub const ENV_EMBEDDING_CACHE_MAX_ENTRIES: &str = "SCA_EMBEDDING_CACHE_MAX_ENTRIES";
/// Env var: embedding cache max bytes.
pub const ENV_EMBEDDING_CACHE_MAX_BYTES: &str = "SCA_EMBEDDING_CACHE_MAX_BYTES";
/// Env var: embedding disk cache enabled.
pub const ENV_EMBEDDING_CACHE_DISK_ENABLED: &str = "SCA_EMBEDDING_CACHE_DISK_ENABLED";
/// Env var: embedding disk cache path.
pub const ENV_EMBEDDING_CACHE_DISK_PATH: &str = "SCA_EMBEDDING_CACHE_DISK_PATH";
/// Env var: embedding disk cache max bytes.
pub const ENV_EMBEDDING_CACHE_DISK_MAX_BYTES: &str = "SCA_EMBEDDING_CACHE_DISK_MAX_BYTES";
/// Env var: embedding disk cache provider.
pub const ENV_EMBEDDING_CACHE_DISK_PROVIDER: &str = "SCA_EMBEDDING_CACHE_DISK_PROVIDER";
/// Env var: embedding disk cache connection string.
pub const ENV_EMBEDDING_CACHE_DISK_CONNECTION: &str = "SCA_EMBEDDING_CACHE_DISK_CONNECTION";
/// Env var: embedding disk cache table name.
pub const ENV_EMBEDDING_CACHE_DISK_TABLE: &str = "SCA_EMBEDDING_CACHE_DISK_TABLE";

/// Env var: OpenAI API key (secret).
pub const ENV_OPENAI_API_AUTH: &str = "OPENAI_API_KEY";
/// Env var: OpenAI base URL.
pub const ENV_OPENAI_BASE_URL: &str = "OPENAI_BASE_URL";
/// Env var: OpenAI model override.
pub const ENV_OPENAI_MODEL: &str = "OPENAI_MODEL";
/// Env var: Gemini API key (secret).
pub const ENV_GEMINI_API_AUTH: &str = "GEMINI_API_KEY";
/// Env var: Gemini base URL.
pub const ENV_GEMINI_BASE_URL: &str = "GEMINI_BASE_URL";
/// Env var: Gemini model override.
pub const ENV_GEMINI_MODEL: &str = "GEMINI_MODEL";
/// Env var: Voyage API key (secret).
pub const ENV_VOYAGE_API_AUTH: &str = "VOYAGEAI_API_KEY";
/// Env var: Voyage base URL.
pub const ENV_VOYAGE_BASE_URL: &str = "VOYAGEAI_BASE_URL";
/// Env var: Voyage model override.
pub const ENV_VOYAGE_MODEL: &str = "VOYAGEAI_MODEL";
/// Env var: Ollama model name.
pub const ENV_OLLAMA_MODEL: &str = "OLLAMA_MODEL";
/// Env var: Ollama host URL.
pub const ENV_OLLAMA_HOST: &str = "OLLAMA_HOST";

/// Env var: vector DB provider identifier.
pub const ENV_VECTOR_DB_PROVIDER: &str = "SCA_VECTOR_DB_PROVIDER";
/// Env var: vector DB index mode (`dense` | `hybrid`).
pub const ENV_VECTOR_DB_INDEX_MODE: &str = "SCA_VECTOR_DB_INDEX_MODE";
/// Env var: vector DB timeout in milliseconds.
pub const ENV_VECTOR_DB_TIMEOUT_MS: &str = "SCA_VECTOR_DB_TIMEOUT_MS";
/// Env var: vector DB index build timeout in milliseconds.
pub const ENV_VECTOR_DB_INDEX_TIMEOUT_MS: &str = "SCA_VECTOR_DB_INDEX_TIMEOUT_MS";
/// Env var: vector DB batch size.
pub const ENV_VECTOR_DB_BATCH_SIZE: &str = "SCA_VECTOR_DB_BATCH_SIZE";
/// Env var: vector DB base URL.
pub const ENV_VECTOR_DB_BASE_URL: &str = "SCA_VECTOR_DB_BASE_URL";
/// Env var: vector DB address/host.
pub const ENV_VECTOR_DB_ADDRESS: &str = "SCA_VECTOR_DB_ADDRESS";
/// Env var: vector DB database name.
pub const ENV_VECTOR_DB_DATABASE: &str = "SCA_VECTOR_DB_DATABASE";
/// Env var: vector DB SSL enablement (true/false).
pub const ENV_VECTOR_DB_SSL: &str = "SCA_VECTOR_DB_SSL";
/// Env var: vector DB auth token (secret).
// gitleaks:allow
pub const ENV_VECTOR_DB_TOKEN: &str = "SCA_VECTOR_DB_TOKEN";
/// Env var: vector DB auth username.
pub const ENV_VECTOR_DB_USERNAME: &str = "SCA_VECTOR_DB_USERNAME";
/// Env var: vector DB auth password (secret).
// gitleaks:allow
pub const ENV_VECTOR_DB_PASSWORD: &str = "SCA_VECTOR_DB_PASSWORD";

/// Env var: sync allowed extensions as CSV.
pub const ENV_SYNC_ALLOWED_EXTENSIONS: &str = "SCA_SYNC_ALLOWED_EXTENSIONS";
/// Env var: sync ignore patterns as CSV.
pub const ENV_SYNC_IGNORE_PATTERNS: &str = "SCA_SYNC_IGNORE_PATTERNS";
/// Env var: sync max files.
pub const ENV_SYNC_MAX_FILES: &str = "SCA_SYNC_MAX_FILES";
/// Env var: sync max file size in bytes.
pub const ENV_SYNC_MAX_FILE_SIZE_BYTES: &str = "SCA_SYNC_MAX_FILE_SIZE_BYTES";

const MAX_CSV_ITEMS: usize = 10_000;

/// Typed env-derived overrides for `BackendConfig`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BackendEnv {
    /// Override for `core.timeoutMs`.
    pub core_timeout_ms: Option<u64>,
    /// Override for `core.maxConcurrency`.
    pub core_max_concurrency: Option<u32>,
    /// Override for `core.maxInFlightFiles`.
    pub core_max_in_flight_files: Option<u32>,
    /// Override for `core.maxInFlightEmbeddingBatches`.
    pub core_max_in_flight_embedding_batches: Option<u32>,
    /// Override for `core.maxInFlightInserts`.
    pub core_max_in_flight_inserts: Option<u32>,
    /// Override for `core.maxBufferedChunks`.
    pub core_max_buffered_chunks: Option<u32>,
    /// Override for `core.maxBufferedEmbeddings`.
    pub core_max_buffered_embeddings: Option<u32>,
    /// Override for `core.maxChunkChars`.
    pub core_max_chunk_chars: Option<u32>,
    /// Override for `core.retry.maxAttempts`.
    pub core_retry_max_attempts: Option<u32>,
    /// Override for `core.retry.baseDelayMs`.
    pub core_retry_base_delay_ms: Option<u64>,
    /// Override for `core.retry.maxDelayMs`.
    pub core_retry_max_delay_ms: Option<u64>,
    /// Override for `core.retry.jitterRatioPct`.
    pub core_retry_jitter_ratio_pct: Option<u32>,

    /// Override for `embedding.provider`.
    pub embedding_provider: Option<Box<str>>,
    /// Override for `embedding.model`.
    pub embedding_model: Option<Box<str>>,
    /// Override for `embedding.timeoutMs`.
    pub embedding_timeout_ms: Option<u64>,
    /// Override for `embedding.batchSize`.
    pub embedding_batch_size: Option<u32>,
    /// Override for `embedding.dimension`.
    pub embedding_dimension: Option<u32>,
    /// Override for `embedding.baseUrl`.
    pub embedding_base_url: Option<Box<str>>,
    /// Override for `embedding.localFirst`.
    pub embedding_local_first: Option<bool>,
    /// Override for `embedding.localOnly`.
    pub embedding_local_only: Option<bool>,
    /// Override for `embedding.routing.mode`.
    pub embedding_routing_mode: Option<EmbeddingRoutingMode>,
    /// Override for `embedding.routing.split.maxRemoteBatches`.
    pub embedding_split_max_remote_batches: Option<u32>,
    /// Override for `embedding.jobs.progressIntervalMs`.
    pub embedding_jobs_progress_interval_ms: Option<u64>,
    /// Override for `embedding.jobs.cancelPollIntervalMs`.
    pub embedding_jobs_cancel_poll_interval_ms: Option<u64>,
    /// Allow fallback to test embeddings when ONNX assets are missing.
    pub embedding_test_fallback: Option<bool>,
    /// Override for `embedding.onnx.modelDir`.
    pub embedding_onnx_model_dir: Option<Box<str>>,
    /// Override for `embedding.onnx.modelFilename`.
    pub embedding_onnx_model_filename: Option<Box<str>>,
    /// Override for `embedding.onnx.tokenizerFilename`.
    pub embedding_onnx_tokenizer_filename: Option<Box<str>>,
    /// Override for `embedding.onnx.repo`.
    pub embedding_onnx_repo: Option<Box<str>>,
    /// Override for `embedding.onnx.downloadOnMissing`.
    pub embedding_onnx_download_on_missing: Option<bool>,
    /// Override for `embedding.onnx.sessionPoolSize`.
    pub embedding_onnx_session_pool_size: Option<u32>,
    /// Override for `embedding.cache.enabled`.
    pub embedding_cache_enabled: Option<bool>,
    /// Override for `embedding.cache.maxEntries`.
    pub embedding_cache_max_entries: Option<u32>,
    /// Override for `embedding.cache.maxBytes`.
    pub embedding_cache_max_bytes: Option<u64>,
    /// Override for `embedding.cache.diskEnabled`.
    pub embedding_cache_disk_enabled: Option<bool>,
    /// Override for `embedding.cache.diskPath`.
    pub embedding_cache_disk_path: Option<Box<str>>,
    /// Override for `embedding.cache.diskProvider`.
    pub embedding_cache_disk_provider: Option<EmbeddingCacheDiskProvider>,
    /// Override for `embedding.cache.diskConnection`.
    pub embedding_cache_disk_connection: Option<Box<str>>,
    /// Override for `embedding.cache.diskTable`.
    pub embedding_cache_disk_table: Option<Box<str>>,
    /// Override for `embedding.cache.diskMaxBytes`.
    pub embedding_cache_disk_max_bytes: Option<u64>,
    /// Secret: embedding API key (not persisted in config).
    pub embedding_api_key: Option<SecretString>,
    /// Provider-specific OpenAI API key (secret).
    pub openai_api_key: Option<SecretString>,
    /// Provider-specific OpenAI base URL.
    pub openai_base_url: Option<Box<str>>,
    /// Provider-specific OpenAI model override.
    pub openai_model: Option<Box<str>>,
    /// Provider-specific Gemini API key (secret).
    pub gemini_api_key: Option<SecretString>,
    /// Provider-specific Gemini base URL.
    pub gemini_base_url: Option<Box<str>>,
    /// Provider-specific Gemini model override.
    pub gemini_model: Option<Box<str>>,
    /// Provider-specific Voyage API key (secret).
    pub voyage_api_key: Option<SecretString>,
    /// Provider-specific Voyage base URL.
    pub voyage_base_url: Option<Box<str>>,
    /// Provider-specific Voyage model override.
    pub voyage_model: Option<Box<str>>,
    /// Provider-specific Ollama model name.
    pub ollama_model: Option<Box<str>>,
    /// Provider-specific Ollama host URL.
    pub ollama_host: Option<Box<str>>,

    /// Override for `vectorDb.provider`.
    pub vector_db_provider: Option<Box<str>>,
    /// Override for `vectorDb.indexMode`.
    pub vector_db_index_mode: Option<IndexMode>,
    /// Override for `vectorDb.timeoutMs`.
    pub vector_db_timeout_ms: Option<u64>,
    /// Override for `vectorDb.indexTimeoutMs`.
    pub vector_db_index_timeout_ms: Option<u64>,
    /// Override for `vectorDb.batchSize`.
    pub vector_db_batch_size: Option<u32>,
    /// Override for `vectorDb.baseUrl`.
    pub vector_db_base_url: Option<Box<str>>,
    /// Override for `vectorDb.address`.
    pub vector_db_address: Option<Box<str>>,
    /// Override for `vectorDb.database`.
    pub vector_db_database: Option<Box<str>>,
    /// Override for `vectorDb.ssl`.
    pub vector_db_ssl: Option<bool>,
    /// Secret: vector DB token (not persisted in config).
    pub vector_db_token: Option<SecretString>,
    /// Override for `vectorDb.username`.
    pub vector_db_username: Option<Box<str>>,
    /// Secret: vector DB password (not persisted in config).
    pub vector_db_password: Option<SecretString>,

    /// Override for `sync.allowedExtensions` (full replacement).
    pub sync_allowed_extensions: Option<Vec<Box<str>>>,
    /// Override for `sync.ignorePatterns` (full replacement).
    pub sync_ignore_patterns: Option<Vec<Box<str>>>,
    /// Override for `sync.maxFiles`.
    pub sync_max_files: Option<u32>,
    /// Override for `sync.maxFileSizeBytes`.
    pub sync_max_file_size_bytes: Option<u64>,
}

#[expect(
    clippy::struct_field_names,
    reason = "env override fields mirror config field names for clarity"
)]
struct CoreEnvOverrides {
    core_timeout_ms: Option<u64>,
    core_max_concurrency: Option<u32>,
    core_max_in_flight_files: Option<u32>,
    core_max_in_flight_embedding_batches: Option<u32>,
    core_max_in_flight_inserts: Option<u32>,
    core_max_buffered_chunks: Option<u32>,
    core_max_buffered_embeddings: Option<u32>,
    core_max_chunk_chars: Option<u32>,
    core_retry_max_attempts: Option<u32>,
    core_retry_base_delay_ms: Option<u64>,
    core_retry_max_delay_ms: Option<u64>,
    core_retry_jitter_ratio_pct: Option<u32>,
}

struct EmbeddingEnvOverrides {
    provider: Option<Box<str>>,
    model: Option<Box<str>>,
    timeout_ms: Option<u64>,
    batch_size: Option<u32>,
    dimension: Option<u32>,
    base_url: Option<Box<str>>,
    local_first: Option<bool>,
    local_only: Option<bool>,
    routing_mode: Option<EmbeddingRoutingMode>,
    split_max_remote_batches: Option<u32>,
    jobs_progress_interval_ms: Option<u64>,
    jobs_cancel_poll_interval_ms: Option<u64>,
    test_fallback: Option<bool>,
    onnx_model_dir: Option<Box<str>>,
    onnx_model_filename: Option<Box<str>>,
    onnx_tokenizer_filename: Option<Box<str>>,
    onnx_repo: Option<Box<str>>,
    onnx_download_on_missing: Option<bool>,
    onnx_session_pool_size: Option<u32>,
    cache_enabled: Option<bool>,
    cache_max_entries: Option<u32>,
    cache_max_bytes: Option<u64>,
    cache_disk_enabled: Option<bool>,
    cache_disk_path: Option<Box<str>>,
    cache_disk_provider: Option<EmbeddingCacheDiskProvider>,
    cache_disk_connection: Option<Box<str>>,
    cache_disk_table: Option<Box<str>>,
    cache_disk_max_bytes: Option<u64>,
    api_key: Option<SecretString>,
}

struct ProviderEnvOverrides {
    openai_api_key: Option<SecretString>,
    openai_base_url: Option<Box<str>>,
    openai_model: Option<Box<str>>,
    gemini_api_key: Option<SecretString>,
    gemini_base_url: Option<Box<str>>,
    gemini_model: Option<Box<str>>,
    voyage_api_key: Option<SecretString>,
    voyage_base_url: Option<Box<str>>,
    voyage_model: Option<Box<str>>,
    ollama_model: Option<Box<str>>,
    ollama_host: Option<Box<str>>,
}

struct VectorDbEnvOverrides {
    provider: Option<Box<str>>,
    index_mode: Option<IndexMode>,
    timeout_ms: Option<u64>,
    index_timeout_ms: Option<u64>,
    batch_size: Option<u32>,
    base_url: Option<Box<str>>,
    address: Option<Box<str>>,
    database: Option<Box<str>>,
    ssl: Option<bool>,
    token: Option<SecretString>,
    username: Option<Box<str>>,
    password: Option<SecretString>,
}

struct SyncEnvOverrides {
    allowed_extensions: Option<Vec<Box<str>>>,
    ignore_patterns: Option<Vec<Box<str>>>,
    max_files: Option<u32>,
    max_file_size_bytes: Option<u64>,
}

struct EmbeddingCoreEnvOverrides {
    provider: Option<Box<str>>,
    model: Option<Box<str>>,
    timeout_ms: Option<u64>,
    batch_size: Option<u32>,
    dimension: Option<u32>,
    base_url: Option<Box<str>>,
    local_first: Option<bool>,
    local_only: Option<bool>,
    routing_mode: Option<EmbeddingRoutingMode>,
    split_max_remote_batches: Option<u32>,
    jobs_progress_interval_ms: Option<u64>,
    jobs_cancel_poll_interval_ms: Option<u64>,
    test_fallback: Option<bool>,
}

struct EmbeddingOnnxEnvOverrides {
    model_dir: Option<Box<str>>,
    model_filename: Option<Box<str>>,
    tokenizer_filename: Option<Box<str>>,
    repo: Option<Box<str>>,
    download_on_missing: Option<bool>,
    session_pool_size: Option<u32>,
}

struct EmbeddingCacheEnvOverrides {
    enabled: Option<bool>,
    max_entries: Option<u32>,
    max_bytes: Option<u64>,
    disk_enabled: Option<bool>,
    disk_path: Option<Box<str>>,
    disk_provider: Option<EmbeddingCacheDiskProvider>,
    disk_connection: Option<Box<str>>,
    disk_table: Option<Box<str>>,
    disk_max_bytes: Option<u64>,
}

fn parse_core_env(map: &BTreeMap<String, String>) -> Result<CoreEnvOverrides, EnvParseError> {
    Ok(CoreEnvOverrides {
        core_timeout_ms: parse_optional_u64(map, ENV_CORE_TIMEOUT_MS)?,
        core_max_concurrency: parse_optional_u32(map, ENV_CORE_MAX_CONCURRENCY)?,
        core_max_in_flight_files: parse_optional_u32(map, ENV_CORE_MAX_IN_FLIGHT_FILES)?,
        core_max_in_flight_embedding_batches: parse_optional_u32(
            map,
            ENV_CORE_MAX_IN_FLIGHT_EMBEDDING_BATCHES,
        )?,
        core_max_in_flight_inserts: parse_optional_u32(map, ENV_CORE_MAX_IN_FLIGHT_INSERTS)?,
        core_max_buffered_chunks: parse_optional_u32(map, ENV_CORE_MAX_BUFFERED_CHUNKS)?,
        core_max_buffered_embeddings: parse_optional_u32(map, ENV_CORE_MAX_BUFFERED_EMBEDDINGS)?,
        core_max_chunk_chars: parse_optional_u32(map, ENV_CORE_MAX_CHUNK_CHARS)?,
        core_retry_max_attempts: parse_optional_u32(map, ENV_CORE_RETRY_MAX_ATTEMPTS)?,
        core_retry_base_delay_ms: parse_optional_u64(map, ENV_CORE_RETRY_BASE_DELAY_MS)?,
        core_retry_max_delay_ms: parse_optional_u64(map, ENV_CORE_RETRY_MAX_DELAY_MS)?,
        core_retry_jitter_ratio_pct: parse_optional_u32(map, ENV_CORE_RETRY_JITTER_RATIO_PCT)?,
    })
}

fn parse_embedding_env(
    map: &BTreeMap<String, String>,
) -> Result<EmbeddingEnvOverrides, EnvParseError> {
    let core = parse_embedding_core_env(map)?;
    let onnx = parse_embedding_onnx_env(map)?;
    let cache = parse_embedding_cache_env(map)?;
    Ok(EmbeddingEnvOverrides {
        provider: core.provider,
        model: core.model,
        timeout_ms: core.timeout_ms,
        batch_size: core.batch_size,
        dimension: core.dimension,
        base_url: core.base_url,
        local_first: core.local_first,
        local_only: core.local_only,
        routing_mode: core.routing_mode,
        split_max_remote_batches: core.split_max_remote_batches,
        jobs_progress_interval_ms: core.jobs_progress_interval_ms,
        jobs_cancel_poll_interval_ms: core.jobs_cancel_poll_interval_ms,
        test_fallback: core.test_fallback,
        onnx_model_dir: onnx.model_dir,
        onnx_model_filename: onnx.model_filename,
        onnx_tokenizer_filename: onnx.tokenizer_filename,
        onnx_repo: onnx.repo,
        onnx_download_on_missing: onnx.download_on_missing,
        onnx_session_pool_size: onnx.session_pool_size,
        cache_enabled: cache.enabled,
        cache_max_entries: cache.max_entries,
        cache_max_bytes: cache.max_bytes,
        cache_disk_enabled: cache.disk_enabled,
        cache_disk_path: cache.disk_path,
        cache_disk_provider: cache.disk_provider,
        cache_disk_connection: cache.disk_connection,
        cache_disk_table: cache.disk_table,
        cache_disk_max_bytes: cache.disk_max_bytes,
        api_key: parse_optional_secret_any(
            map,
            &[ENV_EMBEDDING_API_AUTH, ENV_EMBEDDING_API_AUTH_ALIAS],
        )?,
    })
}

fn parse_embedding_core_env(
    map: &BTreeMap<String, String>,
) -> Result<EmbeddingCoreEnvOverrides, EnvParseError> {
    Ok(EmbeddingCoreEnvOverrides {
        provider: parse_optional_trimmed_string_any(
            map,
            &[ENV_EMBEDDING_PROVIDER, ENV_EMBEDDING_PROVIDER_ALIAS],
        )?,
        model: parse_optional_trimmed_string_any(
            map,
            &[ENV_EMBEDDING_MODEL, ENV_EMBEDDING_MODEL_ALIAS],
        )?,
        timeout_ms: parse_optional_u64_any(
            map,
            &[ENV_EMBEDDING_TIMEOUT_MS, ENV_EMBEDDING_TIMEOUT_MS_ALIAS],
        )?,
        batch_size: parse_optional_u32_any(
            map,
            &[ENV_EMBEDDING_BATCH_SIZE, ENV_EMBEDDING_BATCH_SIZE_ALIAS],
        )?,
        dimension: parse_optional_u32_any(
            map,
            &[ENV_EMBEDDING_DIMENSION, ENV_EMBEDDING_DIMENSION_ALIAS],
        )?,
        base_url: parse_optional_url_string_any(
            map,
            &[ENV_EMBEDDING_BASE_URL, ENV_EMBEDDING_BASE_URL_ALIAS],
        )?,
        local_first: parse_optional_bool_any(
            map,
            &[ENV_EMBEDDING_LOCAL_FIRST, ENV_EMBEDDING_LOCAL_FIRST_ALIAS],
        )?,
        local_only: parse_optional_bool_any(
            map,
            &[ENV_EMBEDDING_LOCAL_ONLY, ENV_EMBEDDING_LOCAL_ONLY_ALIAS],
        )?,
        routing_mode: parse_optional_routing_mode_any(
            map,
            &[ENV_EMBEDDING_ROUTING_MODE, ENV_EMBEDDING_ROUTING_MODE_ALIAS],
        )?,
        split_max_remote_batches: parse_optional_u32_any(
            map,
            &[
                ENV_EMBEDDING_SPLIT_MAX_REMOTE_BATCHES,
                ENV_EMBEDDING_SPLIT_MAX_REMOTE_BATCHES_ALIAS,
            ],
        )?,
        jobs_progress_interval_ms: parse_optional_u64_any(
            map,
            &[
                ENV_EMBEDDING_JOBS_PROGRESS_INTERVAL_MS,
                ENV_EMBEDDING_JOBS_PROGRESS_INTERVAL_MS_ALIAS,
            ],
        )?,
        jobs_cancel_poll_interval_ms: parse_optional_u64_any(
            map,
            &[
                ENV_EMBEDDING_JOBS_CANCEL_POLL_INTERVAL_MS,
                ENV_EMBEDDING_JOBS_CANCEL_POLL_INTERVAL_MS_ALIAS,
            ],
        )?,
        test_fallback: parse_optional_bool_any(
            map,
            &[
                ENV_EMBEDDING_TEST_FALLBACK,
                ENV_EMBEDDING_TEST_FALLBACK_ALIAS,
            ],
        )?,
    })
}

fn parse_embedding_onnx_env(
    map: &BTreeMap<String, String>,
) -> Result<EmbeddingOnnxEnvOverrides, EnvParseError> {
    Ok(EmbeddingOnnxEnvOverrides {
        model_dir: parse_optional_trimmed_string_any(
            map,
            &[
                ENV_EMBEDDING_ONNX_MODEL_DIR,
                ENV_EMBEDDING_ONNX_MODEL_DIR_ALIAS,
            ],
        )?,
        model_filename: parse_optional_trimmed_string_any(
            map,
            &[
                ENV_EMBEDDING_ONNX_MODEL_FILENAME,
                ENV_EMBEDDING_ONNX_MODEL_FILENAME_ALIAS,
            ],
        )?,
        tokenizer_filename: parse_optional_trimmed_string_any(
            map,
            &[
                ENV_EMBEDDING_ONNX_TOK_FILENAME,
                ENV_EMBEDDING_ONNX_TOK_FILENAME_ALIAS,
            ],
        )?,
        repo: parse_optional_trimmed_string_any(
            map,
            &[ENV_EMBEDDING_ONNX_REPO, ENV_EMBEDDING_ONNX_REPO_ALIAS],
        )?,
        download_on_missing: parse_optional_bool_any(
            map,
            &[
                ENV_EMBEDDING_ONNX_DOWNLOAD,
                ENV_EMBEDDING_ONNX_DOWNLOAD_ALIAS,
            ],
        )?,
        session_pool_size: parse_optional_u32_any(
            map,
            &[
                ENV_EMBEDDING_ONNX_SESSION_POOL_SIZE,
                ENV_EMBEDDING_ONNX_SESSION_POOL_SIZE_ALIAS,
            ],
        )?,
    })
}

fn parse_embedding_cache_env(
    map: &BTreeMap<String, String>,
) -> Result<EmbeddingCacheEnvOverrides, EnvParseError> {
    Ok(EmbeddingCacheEnvOverrides {
        enabled: parse_optional_bool(map, ENV_EMBEDDING_CACHE_ENABLED)?,
        max_entries: parse_optional_u32(map, ENV_EMBEDDING_CACHE_MAX_ENTRIES)?,
        max_bytes: parse_optional_u64(map, ENV_EMBEDDING_CACHE_MAX_BYTES)?,
        disk_enabled: parse_optional_bool(map, ENV_EMBEDDING_CACHE_DISK_ENABLED)?,
        disk_path: parse_optional_trimmed_string(map, ENV_EMBEDDING_CACHE_DISK_PATH)?,
        disk_provider: parse_optional_cache_disk_provider(map, ENV_EMBEDDING_CACHE_DISK_PROVIDER)?,
        disk_connection: parse_optional_trimmed_string(map, ENV_EMBEDDING_CACHE_DISK_CONNECTION)?,
        disk_table: parse_optional_trimmed_string(map, ENV_EMBEDDING_CACHE_DISK_TABLE)?,
        disk_max_bytes: parse_optional_u64(map, ENV_EMBEDDING_CACHE_DISK_MAX_BYTES)?,
    })
}

fn parse_provider_env(
    map: &BTreeMap<String, String>,
) -> Result<ProviderEnvOverrides, EnvParseError> {
    Ok(ProviderEnvOverrides {
        openai_api_key: parse_optional_secret(map, ENV_OPENAI_API_AUTH)?,
        openai_base_url: parse_optional_url_string(map, ENV_OPENAI_BASE_URL)?,
        openai_model: parse_optional_trimmed_string(map, ENV_OPENAI_MODEL)?,
        gemini_api_key: parse_optional_secret(map, ENV_GEMINI_API_AUTH)?,
        gemini_base_url: parse_optional_url_string(map, ENV_GEMINI_BASE_URL)?,
        gemini_model: parse_optional_trimmed_string(map, ENV_GEMINI_MODEL)?,
        voyage_api_key: parse_optional_secret(map, ENV_VOYAGE_API_AUTH)?,
        voyage_base_url: parse_optional_url_string(map, ENV_VOYAGE_BASE_URL)?,
        voyage_model: parse_optional_trimmed_string(map, ENV_VOYAGE_MODEL)?,
        ollama_model: parse_optional_trimmed_string(map, ENV_OLLAMA_MODEL)?,
        ollama_host: parse_optional_url_string(map, ENV_OLLAMA_HOST)?,
    })
}

fn parse_vectordb_env(
    map: &BTreeMap<String, String>,
) -> Result<VectorDbEnvOverrides, EnvParseError> {
    Ok(VectorDbEnvOverrides {
        provider: parse_optional_trimmed_string(map, ENV_VECTOR_DB_PROVIDER)?,
        index_mode: parse_optional_index_mode(map, ENV_VECTOR_DB_INDEX_MODE)?,
        timeout_ms: parse_optional_u64(map, ENV_VECTOR_DB_TIMEOUT_MS)?,
        index_timeout_ms: parse_optional_u64(map, ENV_VECTOR_DB_INDEX_TIMEOUT_MS)?,
        batch_size: parse_optional_u32(map, ENV_VECTOR_DB_BATCH_SIZE)?,
        base_url: parse_optional_url_string(map, ENV_VECTOR_DB_BASE_URL)?,
        address: parse_optional_trimmed_string(map, ENV_VECTOR_DB_ADDRESS)?,
        database: parse_optional_trimmed_string(map, ENV_VECTOR_DB_DATABASE)?,
        ssl: parse_optional_bool(map, ENV_VECTOR_DB_SSL)?,
        token: parse_optional_secret(map, ENV_VECTOR_DB_TOKEN)?,
        username: parse_optional_trimmed_string(map, ENV_VECTOR_DB_USERNAME)?,
        password: parse_optional_secret(map, ENV_VECTOR_DB_PASSWORD)?,
    })
}

fn parse_sync_env(map: &BTreeMap<String, String>) -> Result<SyncEnvOverrides, EnvParseError> {
    Ok(SyncEnvOverrides {
        allowed_extensions: parse_optional_csv_extensions(map, ENV_SYNC_ALLOWED_EXTENSIONS)?,
        ignore_patterns: parse_optional_csv_patterns(map, ENV_SYNC_IGNORE_PATTERNS)?,
        max_files: parse_optional_u32(map, ENV_SYNC_MAX_FILES)?,
        max_file_size_bytes: parse_optional_u64(map, ENV_SYNC_MAX_FILE_SIZE_BYTES)?,
    })
}

impl BackendEnv {
    /// Parse env overrides from a key/value map (useful for tests and fixtures).
    pub fn from_map(map: &BTreeMap<String, String>) -> Result<Self, EnvParseError> {
        let core = parse_core_env(map)?;
        let embedding = parse_embedding_env(map)?;
        let providers = parse_provider_env(map)?;
        let vectordb = parse_vectordb_env(map)?;
        let sync = parse_sync_env(map)?;

        Ok(Self {
            core_timeout_ms: core.core_timeout_ms,
            core_max_concurrency: core.core_max_concurrency,
            core_max_in_flight_files: core.core_max_in_flight_files,
            core_max_in_flight_embedding_batches: core.core_max_in_flight_embedding_batches,
            core_max_in_flight_inserts: core.core_max_in_flight_inserts,
            core_max_buffered_chunks: core.core_max_buffered_chunks,
            core_max_buffered_embeddings: core.core_max_buffered_embeddings,
            core_max_chunk_chars: core.core_max_chunk_chars,
            core_retry_max_attempts: core.core_retry_max_attempts,
            core_retry_base_delay_ms: core.core_retry_base_delay_ms,
            core_retry_max_delay_ms: core.core_retry_max_delay_ms,
            core_retry_jitter_ratio_pct: core.core_retry_jitter_ratio_pct,
            embedding_provider: embedding.provider,
            embedding_model: embedding.model,
            embedding_timeout_ms: embedding.timeout_ms,
            embedding_batch_size: embedding.batch_size,
            embedding_dimension: embedding.dimension,
            embedding_base_url: embedding.base_url,
            embedding_local_first: embedding.local_first,
            embedding_local_only: embedding.local_only,
            embedding_routing_mode: embedding.routing_mode,
            embedding_split_max_remote_batches: embedding.split_max_remote_batches,
            embedding_jobs_progress_interval_ms: embedding.jobs_progress_interval_ms,
            embedding_jobs_cancel_poll_interval_ms: embedding.jobs_cancel_poll_interval_ms,
            embedding_test_fallback: embedding.test_fallback,
            embedding_onnx_model_dir: embedding.onnx_model_dir,
            embedding_onnx_model_filename: embedding.onnx_model_filename,
            embedding_onnx_tokenizer_filename: embedding.onnx_tokenizer_filename,
            embedding_onnx_repo: embedding.onnx_repo,
            embedding_onnx_download_on_missing: embedding.onnx_download_on_missing,
            embedding_onnx_session_pool_size: embedding.onnx_session_pool_size,
            embedding_cache_enabled: embedding.cache_enabled,
            embedding_cache_max_entries: embedding.cache_max_entries,
            embedding_cache_max_bytes: embedding.cache_max_bytes,
            embedding_cache_disk_enabled: embedding.cache_disk_enabled,
            embedding_cache_disk_path: embedding.cache_disk_path,
            embedding_cache_disk_provider: embedding.cache_disk_provider,
            embedding_cache_disk_connection: embedding.cache_disk_connection,
            embedding_cache_disk_table: embedding.cache_disk_table,
            embedding_cache_disk_max_bytes: embedding.cache_disk_max_bytes,
            embedding_api_key: embedding.api_key,
            openai_api_key: providers.openai_api_key,
            openai_base_url: providers.openai_base_url,
            openai_model: providers.openai_model,
            gemini_api_key: providers.gemini_api_key,
            gemini_base_url: providers.gemini_base_url,
            gemini_model: providers.gemini_model,
            voyage_api_key: providers.voyage_api_key,
            voyage_base_url: providers.voyage_base_url,
            voyage_model: providers.voyage_model,
            ollama_model: providers.ollama_model,
            ollama_host: providers.ollama_host,
            vector_db_provider: vectordb.provider,
            vector_db_index_mode: vectordb.index_mode,
            vector_db_timeout_ms: vectordb.timeout_ms,
            vector_db_index_timeout_ms: vectordb.index_timeout_ms,
            vector_db_batch_size: vectordb.batch_size,
            vector_db_base_url: vectordb.base_url,
            vector_db_address: vectordb.address,
            vector_db_database: vectordb.database,
            vector_db_ssl: vectordb.ssl,
            vector_db_token: vectordb.token,
            vector_db_username: vectordb.username,
            vector_db_password: vectordb.password,
            sync_allowed_extensions: sync.allowed_extensions,
            sync_ignore_patterns: sync.ignore_patterns,
            sync_max_files: sync.max_files,
            sync_max_file_size_bytes: sync.max_file_size_bytes,
        })
    }

    /// Parse env overrides from the current process environment.
    pub fn from_std_env() -> Result<Self, EnvParseError> {
        let mut map = BTreeMap::new();
        for name in [
            ENV_CORE_TIMEOUT_MS,
            ENV_CORE_MAX_CONCURRENCY,
            ENV_CORE_MAX_IN_FLIGHT_FILES,
            ENV_CORE_MAX_IN_FLIGHT_EMBEDDING_BATCHES,
            ENV_CORE_MAX_IN_FLIGHT_INSERTS,
            ENV_CORE_MAX_BUFFERED_CHUNKS,
            ENV_CORE_MAX_BUFFERED_EMBEDDINGS,
            ENV_CORE_MAX_CHUNK_CHARS,
            ENV_CORE_RETRY_MAX_ATTEMPTS,
            ENV_CORE_RETRY_BASE_DELAY_MS,
            ENV_CORE_RETRY_MAX_DELAY_MS,
            ENV_CORE_RETRY_JITTER_RATIO_PCT,
            ENV_EMBEDDING_PROVIDER,
            ENV_EMBEDDING_PROVIDER_ALIAS,
            ENV_EMBEDDING_MODEL,
            ENV_EMBEDDING_MODEL_ALIAS,
            ENV_EMBEDDING_TIMEOUT_MS,
            ENV_EMBEDDING_TIMEOUT_MS_ALIAS,
            ENV_EMBEDDING_BATCH_SIZE,
            ENV_EMBEDDING_BATCH_SIZE_ALIAS,
            ENV_EMBEDDING_DIMENSION,
            ENV_EMBEDDING_DIMENSION_ALIAS,
            ENV_EMBEDDING_BASE_URL,
            ENV_EMBEDDING_BASE_URL_ALIAS,
            ENV_EMBEDDING_LOCAL_FIRST,
            ENV_EMBEDDING_LOCAL_FIRST_ALIAS,
            ENV_EMBEDDING_LOCAL_ONLY,
            ENV_EMBEDDING_LOCAL_ONLY_ALIAS,
            ENV_EMBEDDING_ROUTING_MODE,
            ENV_EMBEDDING_ROUTING_MODE_ALIAS,
            ENV_EMBEDDING_SPLIT_MAX_REMOTE_BATCHES,
            ENV_EMBEDDING_SPLIT_MAX_REMOTE_BATCHES_ALIAS,
            ENV_EMBEDDING_JOBS_PROGRESS_INTERVAL_MS,
            ENV_EMBEDDING_JOBS_PROGRESS_INTERVAL_MS_ALIAS,
            ENV_EMBEDDING_JOBS_CANCEL_POLL_INTERVAL_MS,
            ENV_EMBEDDING_JOBS_CANCEL_POLL_INTERVAL_MS_ALIAS,
            ENV_EMBEDDING_TEST_FALLBACK,
            ENV_EMBEDDING_TEST_FALLBACK_ALIAS,
            ENV_EMBEDDING_ONNX_MODEL_DIR,
            ENV_EMBEDDING_ONNX_MODEL_DIR_ALIAS,
            ENV_EMBEDDING_ONNX_MODEL_FILENAME,
            ENV_EMBEDDING_ONNX_MODEL_FILENAME_ALIAS,
            ENV_EMBEDDING_ONNX_TOK_FILENAME,
            ENV_EMBEDDING_ONNX_TOK_FILENAME_ALIAS,
            ENV_EMBEDDING_ONNX_REPO,
            ENV_EMBEDDING_ONNX_REPO_ALIAS,
            ENV_EMBEDDING_ONNX_DOWNLOAD,
            ENV_EMBEDDING_ONNX_DOWNLOAD_ALIAS,
            ENV_EMBEDDING_ONNX_SESSION_POOL_SIZE,
            ENV_EMBEDDING_ONNX_SESSION_POOL_SIZE_ALIAS,
            ENV_EMBEDDING_CACHE_ENABLED,
            ENV_EMBEDDING_CACHE_MAX_ENTRIES,
            ENV_EMBEDDING_CACHE_MAX_BYTES,
            ENV_EMBEDDING_CACHE_DISK_ENABLED,
            ENV_EMBEDDING_CACHE_DISK_PATH,
            ENV_EMBEDDING_CACHE_DISK_PROVIDER,
            ENV_EMBEDDING_CACHE_DISK_CONNECTION,
            ENV_EMBEDDING_CACHE_DISK_TABLE,
            ENV_EMBEDDING_CACHE_DISK_MAX_BYTES,
            ENV_EMBEDDING_API_AUTH,
            ENV_EMBEDDING_API_AUTH_ALIAS,
            ENV_OPENAI_API_AUTH,
            ENV_OPENAI_BASE_URL,
            ENV_OPENAI_MODEL,
            ENV_GEMINI_API_AUTH,
            ENV_GEMINI_BASE_URL,
            ENV_GEMINI_MODEL,
            ENV_VOYAGE_API_AUTH,
            ENV_VOYAGE_BASE_URL,
            ENV_VOYAGE_MODEL,
            ENV_OLLAMA_MODEL,
            ENV_OLLAMA_HOST,
            ENV_VECTOR_DB_PROVIDER,
            ENV_VECTOR_DB_INDEX_MODE,
            ENV_VECTOR_DB_TIMEOUT_MS,
            ENV_VECTOR_DB_INDEX_TIMEOUT_MS,
            ENV_VECTOR_DB_BATCH_SIZE,
            ENV_VECTOR_DB_BASE_URL,
            ENV_VECTOR_DB_ADDRESS,
            ENV_VECTOR_DB_DATABASE,
            ENV_VECTOR_DB_SSL,
            ENV_VECTOR_DB_TOKEN,
            ENV_VECTOR_DB_USERNAME,
            ENV_VECTOR_DB_PASSWORD,
            ENV_SYNC_ALLOWED_EXTENSIONS,
            ENV_SYNC_IGNORE_PATTERNS,
            ENV_SYNC_MAX_FILES,
            ENV_SYNC_MAX_FILE_SIZE_BYTES,
        ] {
            if let Ok(value) = std::env::var(name) {
                map.insert(name.to_string(), value);
            }
        }

        Self::from_map(&map)
    }
}

/// Apply env overrides to a base config (env wins over file/default values).
pub fn apply_env_overrides(
    base: BackendConfig,
    env: &BackendEnv,
) -> Result<ValidatedBackendConfig, ErrorEnvelope> {
    let mut config = base;
    apply_core_env_overrides(&mut config, env);
    apply_embedding_env_overrides(&mut config, env);
    apply_vector_db_env_overrides(&mut config, env);
    apply_sync_env_overrides(&mut config, env);

    config.validate_and_normalize().map_err(Into::into)
}

const fn apply_core_env_overrides(config: &mut BackendConfig, env: &BackendEnv) {
    let mapper = EnvConfigMapper::new(config);
    EnvConfigMapper::set_u64(&mut mapper.config.core.timeout_ms, env.core_timeout_ms);
    EnvConfigMapper::set_u32(
        &mut mapper.config.core.max_concurrency,
        env.core_max_concurrency,
    );
    EnvConfigMapper::set_opt_u32(
        &mut mapper.config.core.max_in_flight_files,
        env.core_max_in_flight_files,
    );
    EnvConfigMapper::set_opt_u32(
        &mut mapper.config.core.max_in_flight_embedding_batches,
        env.core_max_in_flight_embedding_batches,
    );
    EnvConfigMapper::set_opt_u32(
        &mut mapper.config.core.max_in_flight_inserts,
        env.core_max_in_flight_inserts,
    );
    EnvConfigMapper::set_opt_u32(
        &mut mapper.config.core.max_buffered_chunks,
        env.core_max_buffered_chunks,
    );
    EnvConfigMapper::set_opt_u32(
        &mut mapper.config.core.max_buffered_embeddings,
        env.core_max_buffered_embeddings,
    );
    EnvConfigMapper::set_u32(
        &mut mapper.config.core.max_chunk_chars,
        env.core_max_chunk_chars,
    );
    EnvConfigMapper::set_u32(
        &mut mapper.config.core.retry.max_attempts,
        env.core_retry_max_attempts,
    );
    EnvConfigMapper::set_u64(
        &mut mapper.config.core.retry.base_delay_ms,
        env.core_retry_base_delay_ms,
    );
    EnvConfigMapper::set_u64(
        &mut mapper.config.core.retry.max_delay_ms,
        env.core_retry_max_delay_ms,
    );
    EnvConfigMapper::set_u32(
        &mut mapper.config.core.retry.jitter_ratio_pct,
        env.core_retry_jitter_ratio_pct,
    );
}

fn apply_embedding_env_overrides(config: &mut BackendConfig, env: &BackendEnv) {
    let mapper = EnvConfigMapper::new(config);
    EnvConfigMapper::set_opt_box_str(
        &mut mapper.config.embedding.provider,
        env.embedding_provider.as_deref(),
    );
    EnvConfigMapper::set_opt_box_str(
        &mut mapper.config.embedding.model,
        env.embedding_model.as_deref(),
    );
    EnvConfigMapper::set_u64(
        &mut mapper.config.embedding.timeout_ms,
        env.embedding_timeout_ms,
    );
    EnvConfigMapper::set_u32(
        &mut mapper.config.embedding.batch_size,
        env.embedding_batch_size,
    );
    EnvConfigMapper::set_opt_u32(
        &mut mapper.config.embedding.dimension,
        env.embedding_dimension,
    );
    EnvConfigMapper::set_opt_box_str(
        &mut mapper.config.embedding.base_url,
        env.embedding_base_url.as_deref(),
    );
    EnvConfigMapper::set_bool(
        &mut mapper.config.embedding.local_first,
        env.embedding_local_first,
    );
    EnvConfigMapper::set_bool(
        &mut mapper.config.embedding.local_only,
        env.embedding_local_only,
    );
    if let Some(mode) = env.embedding_routing_mode {
        mapper.config.embedding.routing.mode = Some(mode);
    }
    if let Some(value) = env.embedding_split_max_remote_batches {
        mapper.config.embedding.routing.split.max_remote_batches = Some(value);
    }
    EnvConfigMapper::set_u64(
        &mut mapper.config.embedding.jobs.progress_interval_ms,
        env.embedding_jobs_progress_interval_ms,
    );
    EnvConfigMapper::set_u64(
        &mut mapper.config.embedding.jobs.cancel_poll_interval_ms,
        env.embedding_jobs_cancel_poll_interval_ms,
    );

    apply_embedding_onnx_env_overrides(config, env);
    apply_embedding_cache_env_overrides(config, env);
}

fn apply_embedding_onnx_env_overrides(config: &mut BackendConfig, env: &BackendEnv) {
    let mapper = EnvConfigMapper::new(config);
    EnvConfigMapper::set_opt_box_str(
        &mut mapper.config.embedding.onnx.model_dir,
        env.embedding_onnx_model_dir.as_deref(),
    );
    EnvConfigMapper::set_opt_box_str(
        &mut mapper.config.embedding.onnx.model_filename,
        env.embedding_onnx_model_filename.as_deref(),
    );
    EnvConfigMapper::set_opt_box_str(
        &mut mapper.config.embedding.onnx.tokenizer_filename,
        env.embedding_onnx_tokenizer_filename.as_deref(),
    );
    EnvConfigMapper::set_opt_box_str(
        &mut mapper.config.embedding.onnx.repo,
        env.embedding_onnx_repo.as_deref(),
    );
    EnvConfigMapper::set_bool(
        &mut mapper.config.embedding.onnx.download_on_missing,
        env.embedding_onnx_download_on_missing,
    );
    EnvConfigMapper::set_u32(
        &mut mapper.config.embedding.onnx.session_pool_size,
        env.embedding_onnx_session_pool_size,
    );
}

fn apply_embedding_cache_env_overrides(config: &mut BackendConfig, env: &BackendEnv) {
    let mapper = EnvConfigMapper::new(config);
    EnvConfigMapper::set_bool(
        &mut mapper.config.embedding.cache.enabled,
        env.embedding_cache_enabled,
    );
    EnvConfigMapper::set_u32(
        &mut mapper.config.embedding.cache.max_entries,
        env.embedding_cache_max_entries,
    );
    EnvConfigMapper::set_u64(
        &mut mapper.config.embedding.cache.max_bytes,
        env.embedding_cache_max_bytes,
    );
    EnvConfigMapper::set_bool(
        &mut mapper.config.embedding.cache.disk_enabled,
        env.embedding_cache_disk_enabled,
    );
    EnvConfigMapper::set_opt_box_str(
        &mut mapper.config.embedding.cache.disk_path,
        env.embedding_cache_disk_path.as_deref(),
    );
    if let Some(provider) = env.embedding_cache_disk_provider {
        mapper.config.embedding.cache.disk_provider = Some(provider);
    }
    EnvConfigMapper::set_opt_box_str(
        &mut mapper.config.embedding.cache.disk_connection,
        env.embedding_cache_disk_connection.as_deref(),
    );
    EnvConfigMapper::set_opt_box_str(
        &mut mapper.config.embedding.cache.disk_table,
        env.embedding_cache_disk_table.as_deref(),
    );
    EnvConfigMapper::set_u64(
        &mut mapper.config.embedding.cache.disk_max_bytes,
        env.embedding_cache_disk_max_bytes,
    );
}

fn apply_vector_db_env_overrides(config: &mut BackendConfig, env: &BackendEnv) {
    let mapper = EnvConfigMapper::new(config);
    EnvConfigMapper::set_opt_box_str(
        &mut mapper.config.vector_db.provider,
        env.vector_db_provider.as_deref(),
    );
    EnvConfigMapper::set_opt_index_mode(
        &mut mapper.config.vector_db.index_mode,
        env.vector_db_index_mode,
    );
    EnvConfigMapper::set_u64(
        &mut mapper.config.vector_db.timeout_ms,
        env.vector_db_timeout_ms,
    );
    EnvConfigMapper::set_u64(
        &mut mapper.config.vector_db.index_timeout_ms,
        env.vector_db_index_timeout_ms,
    );
    EnvConfigMapper::set_u32(
        &mut mapper.config.vector_db.batch_size,
        env.vector_db_batch_size,
    );
    EnvConfigMapper::set_opt_box_str(
        &mut mapper.config.vector_db.base_url,
        env.vector_db_base_url.as_deref(),
    );
    EnvConfigMapper::set_opt_box_str(
        &mut mapper.config.vector_db.address,
        env.vector_db_address.as_deref(),
    );
    EnvConfigMapper::set_opt_box_str(
        &mut mapper.config.vector_db.database,
        env.vector_db_database.as_deref(),
    );
    EnvConfigMapper::set_bool(&mut mapper.config.vector_db.ssl, env.vector_db_ssl);
    EnvConfigMapper::set_opt_box_str(
        &mut mapper.config.vector_db.token,
        env.vector_db_token.as_ref().map(SecretString::expose),
    );
    EnvConfigMapper::set_opt_box_str(
        &mut mapper.config.vector_db.username,
        env.vector_db_username.as_deref(),
    );
    EnvConfigMapper::set_opt_box_str(
        &mut mapper.config.vector_db.password,
        env.vector_db_password.as_ref().map(SecretString::expose),
    );
}

fn apply_sync_env_overrides(config: &mut BackendConfig, env: &BackendEnv) {
    let mapper = EnvConfigMapper::new(config);
    EnvConfigMapper::set_clone(
        &mut mapper.config.sync.allowed_extensions,
        env.sync_allowed_extensions.as_ref(),
    );
    EnvConfigMapper::set_clone(
        &mut mapper.config.sync.ignore_patterns,
        env.sync_ignore_patterns.as_ref(),
    );
    EnvConfigMapper::set_u32(&mut mapper.config.sync.max_files, env.sync_max_files);
    EnvConfigMapper::set_u64(
        &mut mapper.config.sync.max_file_size_bytes,
        env.sync_max_file_size_bytes,
    );
}

struct EnvConfigMapper<'a> {
    config: &'a mut BackendConfig,
}

impl<'a> EnvConfigMapper<'a> {
    const fn new(config: &'a mut BackendConfig) -> Self {
        Self { config }
    }

    const fn set_u64(field: &mut u64, value: Option<u64>) {
        if let Some(value) = value {
            *field = value;
        }
    }

    const fn set_u32(field: &mut u32, value: Option<u32>) {
        if let Some(value) = value {
            *field = value;
        }
    }

    const fn set_bool(field: &mut bool, value: Option<bool>) {
        if let Some(value) = value {
            *field = value;
        }
    }

    const fn set_opt_u32(field: &mut Option<u32>, value: Option<u32>) {
        if value.is_some() {
            *field = value;
        }
    }

    const fn set_opt_index_mode(field: &mut IndexMode, value: Option<IndexMode>) {
        if let Some(value) = value {
            *field = value;
        }
    }

    fn set_opt_box_str(field: &mut Option<Box<str>>, value: Option<&str>) {
        if let Some(value) = value {
            *field = Some(value.to_owned().into_boxed_str());
        }
    }

    fn set_clone<T: Clone>(field: &mut T, value: Option<&T>) {
        if let Some(value) = value {
            *field = value.clone();
        }
    }
}

/// Validation failures when parsing env variables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvParseError {
    /// An env var was present but empty after trimming.
    EmptyValue {
        /// Env var name.
        var: &'static str,
    },
    /// A secret env var was present but empty after trimming.
    EmptySecret {
        /// Env var name.
        var: &'static str,
    },
    /// Boolean env var had an invalid value.
    InvalidBool {
        /// Env var name.
        var: &'static str,
        /// Raw input value.
        value: String,
    },
    /// Integer env var had an invalid value.
    InvalidInt {
        /// Env var name.
        var: &'static str,
        /// Raw input value.
        value: String,
    },
    /// URL env var had an invalid value.
    InvalidUrl {
        /// Env var name.
        var: &'static str,
        /// Raw input value.
        value: String,
    },
    /// Enum env var had an invalid value.
    InvalidEnum {
        /// Env var name.
        var: &'static str,
        /// Raw input value.
        value: String,
    },
    /// CSV list exceeds a safety limit.
    CsvTooLarge {
        /// Env var name.
        var: &'static str,
        /// Number of parsed items.
        len: usize,
        /// Maximum allowed.
        max: usize,
    },
    /// CSV contained an invalid extension entry.
    InvalidExtensionEntry {
        /// Env var name.
        var: &'static str,
        /// Invalid entry.
        entry: String,
    },
    /// CSV contained an invalid ignore pattern entry.
    InvalidIgnorePatternEntry {
        /// Env var name.
        var: &'static str,
        /// Invalid entry.
        entry: String,
    },
}

impl EnvParseError {
    fn error_code(&self) -> ErrorCode {
        match self {
            Self::EmptyValue { .. } | Self::EmptySecret { .. } => {
                ErrorCode::new("config", "empty_env_var")
            },
            Self::InvalidBool { .. } => ErrorCode::new("config", "invalid_env_bool"),
            Self::InvalidInt { .. } => ErrorCode::new("config", "invalid_env_int"),
            Self::InvalidUrl { .. } => ErrorCode::new("config", "invalid_env_url"),
            Self::InvalidEnum { .. } => ErrorCode::new("config", "invalid_env_enum"),
            Self::CsvTooLarge { .. }
            | Self::InvalidExtensionEntry { .. }
            | Self::InvalidIgnorePatternEntry { .. } => ErrorCode::new("config", "invalid_env_csv"),
        }
    }
}

impl fmt::Display for EnvParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyValue { var } | Self::EmptySecret { var } => {
                write!(formatter, "{var} must be non-empty")
            },
            Self::InvalidBool { var, .. } => write!(formatter, "{var} must be a boolean"),
            Self::InvalidInt { var, .. } => write!(formatter, "{var} must be an integer"),
            Self::InvalidUrl { var, .. } => write!(formatter, "{var} must be a valid URL"),
            Self::InvalidEnum { var, .. } => write!(formatter, "{var} has an unsupported value"),
            Self::CsvTooLarge { var, len, max } => {
                write!(formatter, "{var} is too large ({len} items, max {max})")
            },
            Self::InvalidExtensionEntry { var, entry } => {
                write!(formatter, "{var} contains invalid extension entry: {entry}")
            },
            Self::InvalidIgnorePatternEntry { var, entry } => {
                write!(
                    formatter,
                    "{var} contains invalid ignore pattern entry: {entry}"
                )
            },
        }
    }
}

impl std::error::Error for EnvParseError {}

impl From<EnvParseError> for ErrorEnvelope {
    fn from(error: EnvParseError) -> Self {
        let code = error.error_code();
        let message = error.to_string();
        let mut envelope = Self::expected(code, message);

        match error {
            EnvParseError::EmptyValue { var } | EnvParseError::EmptySecret { var } => {
                envelope = envelope.with_metadata("env_var", var);
            },
            EnvParseError::InvalidBool { var, value }
            | EnvParseError::InvalidInt { var, value }
            | EnvParseError::InvalidUrl { var, value }
            | EnvParseError::InvalidEnum { var, value } => {
                envelope = envelope
                    .with_metadata("env_var", var)
                    .with_metadata("value", redact_value(var, &value));
            },
            EnvParseError::CsvTooLarge { var, len, max } => {
                envelope = envelope
                    .with_metadata("env_var", var)
                    .with_metadata("len", len.to_string())
                    .with_metadata("max", max.to_string());
            },
            EnvParseError::InvalidExtensionEntry { var, entry }
            | EnvParseError::InvalidIgnorePatternEntry { var, entry } => {
                envelope = envelope
                    .with_metadata("env_var", var)
                    .with_metadata("entry", entry);
            },
        }

        envelope
    }
}

fn parse_optional_trimmed_string(
    map: &BTreeMap<String, String>,
    var: &'static str,
) -> Result<Option<Box<str>>, EnvParseError> {
    let Some(raw) = map.get(var) else {
        return Ok(None);
    };

    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(EnvParseError::EmptyValue { var });
    }

    Ok(Some(trimmed.to_owned().into_boxed_str()))
}

fn parse_optional_trimmed_string_any(
    map: &BTreeMap<String, String>,
    vars: &[&'static str],
) -> Result<Option<Box<str>>, EnvParseError> {
    for var in vars {
        if map.contains_key(*var) {
            return parse_optional_trimmed_string(map, var);
        }
    }
    Ok(None)
}

fn parse_optional_secret(
    map: &BTreeMap<String, String>,
    var: &'static str,
) -> Result<Option<SecretString>, EnvParseError> {
    let Some(raw) = map.get(var) else {
        return Ok(None);
    };

    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(EnvParseError::EmptySecret { var });
    }

    Ok(Some(SecretString::new(trimmed.to_owned())))
}

fn parse_optional_secret_any(
    map: &BTreeMap<String, String>,
    vars: &[&'static str],
) -> Result<Option<SecretString>, EnvParseError> {
    for var in vars {
        if map.contains_key(*var) {
            return parse_optional_secret(map, var);
        }
    }
    Ok(None)
}

fn parse_optional_u64(
    map: &BTreeMap<String, String>,
    var: &'static str,
) -> Result<Option<u64>, EnvParseError> {
    let Some(raw) = map.get(var) else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(EnvParseError::EmptyValue { var });
    }

    trimmed
        .parse::<u64>()
        .map(Some)
        .map_err(|_| EnvParseError::InvalidInt {
            var,
            value: raw.clone(),
        })
}

fn parse_optional_u64_any(
    map: &BTreeMap<String, String>,
    vars: &[&'static str],
) -> Result<Option<u64>, EnvParseError> {
    for var in vars {
        if map.contains_key(*var) {
            return parse_optional_u64(map, var);
        }
    }
    Ok(None)
}

fn parse_optional_u32(
    map: &BTreeMap<String, String>,
    var: &'static str,
) -> Result<Option<u32>, EnvParseError> {
    let Some(raw) = map.get(var) else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(EnvParseError::EmptyValue { var });
    }

    trimmed
        .parse::<u32>()
        .map(Some)
        .map_err(|_| EnvParseError::InvalidInt {
            var,
            value: raw.clone(),
        })
}

fn parse_optional_u32_any(
    map: &BTreeMap<String, String>,
    vars: &[&'static str],
) -> Result<Option<u32>, EnvParseError> {
    for var in vars {
        if map.contains_key(*var) {
            return parse_optional_u32(map, var);
        }
    }
    Ok(None)
}

fn parse_optional_bool(
    map: &BTreeMap<String, String>,
    var: &'static str,
) -> Result<Option<bool>, EnvParseError> {
    let Some(raw) = map.get(var) else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(EnvParseError::EmptyValue { var });
    }

    match trimmed.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(Some(true)),
        "false" | "0" | "no" | "off" => Ok(Some(false)),
        _ => Err(EnvParseError::InvalidBool {
            var,
            value: raw.clone(),
        }),
    }
}

fn parse_optional_bool_any(
    map: &BTreeMap<String, String>,
    vars: &[&'static str],
) -> Result<Option<bool>, EnvParseError> {
    for var in vars {
        if map.contains_key(*var) {
            return parse_optional_bool(map, var);
        }
    }
    Ok(None)
}

fn parse_optional_index_mode(
    map: &BTreeMap<String, String>,
    var: &'static str,
) -> Result<Option<IndexMode>, EnvParseError> {
    let Some(raw) = map.get(var) else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(EnvParseError::EmptyValue { var });
    }

    match trimmed.to_ascii_lowercase().as_str() {
        "dense" => Ok(Some(IndexMode::Dense)),
        "hybrid" => Ok(Some(IndexMode::Hybrid)),
        _ => Err(EnvParseError::InvalidEnum {
            var,
            value: raw.clone(),
        }),
    }
}

fn parse_optional_routing_mode_any(
    map: &BTreeMap<String, String>,
    vars: &[&'static str],
) -> Result<Option<EmbeddingRoutingMode>, EnvParseError> {
    for var in vars {
        let Some(raw) = map.get(*var) else {
            continue;
        };
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(EnvParseError::EmptyValue { var });
        }
        let parsed =
            EmbeddingRoutingMode::parse(trimmed).ok_or_else(|| EnvParseError::InvalidEnum {
                var,
                value: trimmed.to_string(),
            })?;
        return Ok(Some(parsed));
    }
    Ok(None)
}

fn parse_optional_cache_disk_provider(
    map: &BTreeMap<String, String>,
    var: &'static str,
) -> Result<Option<EmbeddingCacheDiskProvider>, EnvParseError> {
    let Some(raw) = map.get(var) else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(EnvParseError::EmptyValue { var });
    }
    let normalized = trimmed.to_ascii_lowercase();
    let parsed = match normalized.as_str() {
        "sqlite" => EmbeddingCacheDiskProvider::Sqlite,
        "postgres" | "postgresql" => EmbeddingCacheDiskProvider::Postgres,
        "mysql" => EmbeddingCacheDiskProvider::Mysql,
        "mssql" | "sqlserver" | "sql-server" => EmbeddingCacheDiskProvider::Mssql,
        _ => {
            return Err(EnvParseError::InvalidEnum {
                var,
                value: trimmed.to_string(),
            });
        },
    };
    Ok(Some(parsed))
}

fn parse_optional_url_string(
    map: &BTreeMap<String, String>,
    var: &'static str,
) -> Result<Option<Box<str>>, EnvParseError> {
    let Some(raw) = map.get(var) else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(EnvParseError::EmptyValue { var });
    }

    let parsed = Url::parse(trimmed).map_err(|_| EnvParseError::InvalidUrl {
        var,
        value: raw.clone(),
    })?;
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        return Err(EnvParseError::InvalidUrl {
            var,
            value: raw.clone(),
        });
    }

    Ok(Some(parsed.to_string().into_boxed_str()))
}

fn parse_optional_url_string_any(
    map: &BTreeMap<String, String>,
    vars: &[&'static str],
) -> Result<Option<Box<str>>, EnvParseError> {
    for var in vars {
        if map.contains_key(*var) {
            return parse_optional_url_string(map, var);
        }
    }
    Ok(None)
}

fn parse_optional_csv_extensions(
    map: &BTreeMap<String, String>,
    var: &'static str,
) -> Result<Option<Vec<Box<str>>>, EnvParseError> {
    let Some(raw) = map.get(var) else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Some(Vec::new()));
    }

    let items = parse_csv(trimmed);
    if items.len() > MAX_CSV_ITEMS {
        return Err(EnvParseError::CsvTooLarge {
            var,
            len: items.len(),
            max: MAX_CSV_ITEMS,
        });
    }

    let mut normalized = Vec::with_capacity(items.len());
    for item in items {
        let trimmed = item.trim();
        let trimmed = trimmed.strip_prefix("*.").unwrap_or(trimmed);
        let trimmed = trimmed.strip_prefix('.').unwrap_or(trimmed);
        let candidate = trimmed.to_ascii_lowercase();
        if candidate.is_empty() || !candidate.chars().all(|ch| ch.is_ascii_alphanumeric()) {
            return Err(EnvParseError::InvalidExtensionEntry { var, entry: item });
        }
        normalized.push(candidate.into_boxed_str());
    }

    normalized.sort_unstable();
    normalized.dedup();
    Ok(Some(normalized))
}

fn parse_optional_csv_patterns(
    map: &BTreeMap<String, String>,
    var: &'static str,
) -> Result<Option<Vec<Box<str>>>, EnvParseError> {
    let Some(raw) = map.get(var) else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Some(Vec::new()));
    }

    let items = parse_csv(trimmed);
    if items.len() > MAX_CSV_ITEMS {
        return Err(EnvParseError::CsvTooLarge {
            var,
            len: items.len(),
            max: MAX_CSV_ITEMS,
        });
    }

    let mut normalized = Vec::with_capacity(items.len());
    for item in items {
        let entry = item.trim();
        if entry.is_empty() {
            return Err(EnvParseError::InvalidIgnorePatternEntry { var, entry: item });
        }
        let replaced = entry.replace('\\', "/");
        let collapsed = collapse_forward_slashes(&replaced);
        normalized.push(collapsed.into_boxed_str());
    }

    normalized.sort_unstable();
    normalized.dedup();
    Ok(Some(normalized))
}

fn parse_csv(input: &str) -> Vec<String> {
    input
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(ToOwned::to_owned)
        .collect()
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

fn redact_value(var: &str, value: &str) -> String {
    if is_secret_key(var) {
        REDACTED_VALUE.to_string()
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn int_parsing_with_missing_defaults_to_none() -> Result<(), Box<dyn Error>> {
        let map = BTreeMap::new();
        assert_eq!(parse_optional_u32(&map, "MISSING")?, None);
        assert_eq!(parse_optional_u64(&map, "MISSING")?, None);
        Ok(())
    }

    #[test]
    fn csv_extensions_are_normalized_deterministically() -> Result<(), Box<dyn Error>> {
        let mut map = BTreeMap::new();
        map.insert(
            ENV_SYNC_ALLOWED_EXTENSIONS.to_string(),
            " TSX , .rs,ts,*.RS,tsx".to_string(),
        );
        let env = BackendEnv::from_map(&map)?;

        let values = env
            .sync_allowed_extensions
            .as_ref()
            .ok_or_else(|| std::io::Error::other("missing extensions"))?;
        let as_str: Vec<&str> = values.iter().map(AsRef::as_ref).collect();
        assert_eq!(as_str, vec!["rs", "ts", "tsx"]);
        Ok(())
    }

    #[test]
    fn url_validation_accepts_http_and_https() -> Result<(), Box<dyn Error>> {
        let mut map = BTreeMap::new();
        map.insert(
            ENV_EMBEDDING_BASE_URL.to_string(),
            "https://example.com/v1".to_string(),
        );
        let env = BackendEnv::from_map(&map)?;
        assert_eq!(
            env.embedding_base_url.as_deref(),
            Some("https://example.com/v1")
        );

        map.insert(
            ENV_EMBEDDING_BASE_URL.to_string(),
            "ftp://example.com".to_string(),
        );
        let error = BackendEnv::from_map(&map).err();
        assert!(matches!(error, Some(EnvParseError::InvalidUrl { .. })));
        Ok(())
    }

    #[test]
    fn secret_values_are_redacted_in_error_metadata() -> Result<(), Box<dyn Error>> {
        let mut map = BTreeMap::new();
        map.insert(ENV_EMBEDDING_API_AUTH.to_string(), "   ".to_string());

        let error = BackendEnv::from_map(&map).err();
        let envelope: ErrorEnvelope = error
            .ok_or_else(|| std::io::Error::other("expected secret error"))?
            .into();

        assert_eq!(envelope.code, ErrorCode::new("config", "empty_env_var"));
        assert_eq!(
            envelope.metadata.get("env_var").map(String::as_str),
            Some(ENV_EMBEDDING_API_AUTH)
        );
        assert!(
            !envelope.metadata.contains_key("value"),
            "empty secrets should not echo value"
        );
        Ok(())
    }
}
