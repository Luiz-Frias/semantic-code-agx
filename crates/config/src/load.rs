//! Config loading helpers (env + file + overrides).
//!
//! The loader is responsible for deterministic merge order and surfacing
//! user-facing errors as typed `ErrorEnvelope`s.

use crate::{
    BackendConfig, BackendEnv, EmbeddingCacheDiskProvider, EmbeddingRoutingMode,
    ValidatedBackendConfig, VectorDbIndexConfig, apply_env_overrides,
};
use semantic_code_domain::IndexMode;
use semantic_code_shared::{ErrorCode, ErrorEnvelope};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConfigFormat {
    Json,
    Toml,
}

/// Load the backend config from sources using a deterministic precedence order.
///
/// Precedence (highest wins):
/// - env overrides (`BackendEnv`)
/// - overrides JSON (partial config)
/// - config JSON (file content)
/// - defaults (`BackendConfig::default()`)
pub fn load_backend_config_from_sources(
    config_json: Option<&str>,
    overrides_json: Option<&str>,
    env: &BackendEnv,
) -> Result<ValidatedBackendConfig, ErrorEnvelope> {
    let mut config = match config_json {
        None => BackendConfig::default(),
        Some(input) => parse_config_unvalidated(input, ConfigFormat::Json)?,
    };

    if let Some(input) = overrides_json {
        let overrides = parse_overrides_json(input)?;
        apply_overrides(&mut config, &overrides);
    }

    // env is applied last and also validates/normalizes the resulting config.
    apply_env_overrides(config, env)
}

/// Load the backend config from an optional file path.
pub fn load_backend_config_from_path(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    env: &BackendEnv,
) -> Result<ValidatedBackendConfig, ErrorEnvelope> {
    let mut config = match config_path {
        None => BackendConfig::default(),
        Some(path) => {
            let config_text = read_config_file(path)?;
            let format = detect_config_format(path)?;
            parse_config_unvalidated(&config_text, format)?
        },
    };

    if let Some(input) = overrides_json {
        let overrides = parse_overrides_json(input)?;
        apply_overrides(&mut config, &overrides);
    }

    // env is applied last and also validates/normalizes the resulting config.
    apply_env_overrides(config, env)
}

/// Load the backend config from std env and an optional file path.
pub fn load_backend_config_std_env(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
) -> Result<ValidatedBackendConfig, ErrorEnvelope> {
    let env = BackendEnv::from_std_env().map_err(ErrorEnvelope::from)?;
    load_backend_config_from_path(config_path, overrides_json, &env)
}

/// Serialize the config as deterministic pretty JSON (with trailing newline).
pub fn to_pretty_json(config: &BackendConfig) -> Result<String, ErrorEnvelope> {
    let mut output = serde_json::to_string_pretty(config).map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("core", "internal"),
            format!("failed to serialize config: {error}"),
            semantic_code_shared::ErrorClass::NonRetriable,
        )
    })?;
    output.push('\n');
    Ok(output)
}

/// Serialize the config as deterministic pretty TOML (with trailing newline).
pub fn to_pretty_toml(config: &BackendConfig) -> Result<String, ErrorEnvelope> {
    let mut output = toml::to_string_pretty(config).map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("config", "serialize_toml"),
            format!("failed to serialize config TOML: {error}"),
            semantic_code_shared::ErrorClass::NonRetriable,
        )
    })?;
    output.push('\n');
    Ok(output)
}

fn parse_config_unvalidated(
    input: &str,
    format: ConfigFormat,
) -> Result<BackendConfig, ErrorEnvelope> {
    match format {
        ConfigFormat::Json => serde_json::from_str(input).map_err(|error| {
            ErrorEnvelope::expected(
                ErrorCode::new("config", "invalid_json"),
                format!("invalid config JSON: {error}"),
            )
            .with_metadata("source", "config")
        }),
        ConfigFormat::Toml => toml::from_str(input).map_err(|error| {
            ErrorEnvelope::expected(
                ErrorCode::new("config", "invalid_toml"),
                format!("invalid config TOML: {error}"),
            )
            .with_metadata("source", "config")
        }),
    }
}

fn parse_overrides_json(input: &str) -> Result<BackendConfigOverrides, ErrorEnvelope> {
    serde_json::from_str(input).map_err(|error| {
        ErrorEnvelope::expected(
            ErrorCode::new("config", "invalid_json"),
            format!("invalid overrides JSON: {error}"),
        )
        .with_metadata("source", "overrides")
    })
}

fn read_config_file(path: &Path) -> Result<String, ErrorEnvelope> {
    std::fs::read_to_string(path).map_err(|error| {
        let code = match error.kind() {
            std::io::ErrorKind::NotFound => ErrorCode::new("config", "config_file_not_found"),
            std::io::ErrorKind::PermissionDenied => {
                ErrorCode::new("config", "config_file_permission_denied")
            },
            _ => ErrorCode::new("config", "config_file_io"),
        };

        ErrorEnvelope::expected(code, format!("failed to read config file: {error}"))
            .with_metadata("path", path.to_string_lossy().to_string())
    })
}

fn detect_config_format(path: &Path) -> Result<ConfigFormat, ErrorEnvelope> {
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase);
    match ext.as_deref() {
        None | Some("json") => Ok(ConfigFormat::Json),
        Some("toml") => Ok(ConfigFormat::Toml),
        Some(other) => Err(ErrorEnvelope::expected(
            ErrorCode::new("config", "unsupported_format"),
            "unsupported config format; use .json or .toml",
        )
        .with_metadata("extension", other.to_string())),
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
struct BackendConfigOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<u32>,
    #[serde(default)]
    core: CoreConfigOverrides,
    #[serde(default)]
    embedding: EmbeddingConfigOverrides,
    #[serde(default)]
    vector_db: VectorDbConfigOverrides,
    #[serde(default)]
    sync: SyncConfigOverrides,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
struct CoreConfigOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    timeout_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_concurrency: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_in_flight_files: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_in_flight_embedding_batches: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_in_flight_inserts: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_buffered_chunks: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_buffered_embeddings: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_chunk_chars: Option<u32>,
    #[serde(default)]
    retry: RetryConfigOverrides,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
struct RetryConfigOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_attempts: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    base_delay_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_delay_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    jitter_ratio_pct: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
struct EmbeddingConfigOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: Option<Box<str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<Box<str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    base_url: Option<Box<str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimension: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timeout_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    batch_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    local_first: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    local_only: Option<bool>,
    #[serde(default)]
    routing: EmbeddingRoutingOverrides,
    #[serde(default)]
    jobs: EmbeddingJobsOverrides,
    #[serde(default)]
    onnx: EmbeddingOnnxOverrides,
    #[serde(default)]
    cache: EmbeddingCacheOverrides,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
struct EmbeddingRoutingOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    mode: Option<EmbeddingRoutingMode>,
    #[serde(default)]
    split: EmbeddingSplitOverrides,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
struct EmbeddingSplitOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_remote_batches: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
struct EmbeddingJobsOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    progress_interval_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cancel_poll_interval_ms: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
struct EmbeddingOnnxOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    model_dir: Option<Box<str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_filename: Option<Box<str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokenizer_filename: Option<Box<str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repo: Option<Box<str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    download_on_missing: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    session_pool_size: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
struct EmbeddingCacheOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_entries: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    disk_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    disk_provider: Option<EmbeddingCacheDiskProvider>,
    #[serde(skip_serializing_if = "Option::is_none")]
    disk_path: Option<Box<str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    disk_connection: Option<Box<str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    disk_table: Option<Box<str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    disk_max_bytes: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
struct VectorDbConfigOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: Option<Box<str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    address: Option<Box<str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    index_mode: Option<IndexMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    base_url: Option<Box<str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    database: Option<Box<str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    token: Option<Box<str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    username: Option<Box<str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    password: Option<Box<str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ssl: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timeout_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    index_timeout_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    index: Option<VectorDbIndexConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    batch_size: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, default)]
struct SyncConfigOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_extensions: Option<Vec<Box<str>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ignore_patterns: Option<Vec<Box<str>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_files: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_file_size_bytes: Option<u64>,
}

fn apply_overrides(config: &mut BackendConfig, overrides: &BackendConfigOverrides) {
    if let Some(version) = overrides.version {
        config.version = version;
    }

    apply_core_overrides(config, &overrides.core);
    apply_embedding_overrides(config, &overrides.embedding);
    apply_vector_db_overrides(config, &overrides.vector_db);
    apply_sync_overrides(config, &overrides.sync);
}

const fn apply_core_overrides(config: &mut BackendConfig, overrides: &CoreConfigOverrides) {
    let mapper = OverrideMapper::new(config);
    OverrideMapper::set_u64(&mut mapper.config.core.timeout_ms, overrides.timeout_ms);
    OverrideMapper::set_u32(
        &mut mapper.config.core.max_concurrency,
        overrides.max_concurrency,
    );
    OverrideMapper::set_opt_u32(
        &mut mapper.config.core.max_in_flight_files,
        overrides.max_in_flight_files,
    );
    OverrideMapper::set_opt_u32(
        &mut mapper.config.core.max_in_flight_embedding_batches,
        overrides.max_in_flight_embedding_batches,
    );
    OverrideMapper::set_opt_u32(
        &mut mapper.config.core.max_in_flight_inserts,
        overrides.max_in_flight_inserts,
    );
    OverrideMapper::set_opt_u32(
        &mut mapper.config.core.max_buffered_chunks,
        overrides.max_buffered_chunks,
    );
    OverrideMapper::set_opt_u32(
        &mut mapper.config.core.max_buffered_embeddings,
        overrides.max_buffered_embeddings,
    );
    OverrideMapper::set_u32(
        &mut mapper.config.core.max_chunk_chars,
        overrides.max_chunk_chars,
    );
    apply_retry_overrides(config, &overrides.retry);
}

const fn apply_retry_overrides(config: &mut BackendConfig, overrides: &RetryConfigOverrides) {
    let mapper = OverrideMapper::new(config);
    OverrideMapper::set_u32(
        &mut mapper.config.core.retry.max_attempts,
        overrides.max_attempts,
    );
    OverrideMapper::set_u64(
        &mut mapper.config.core.retry.base_delay_ms,
        overrides.base_delay_ms,
    );
    OverrideMapper::set_u64(
        &mut mapper.config.core.retry.max_delay_ms,
        overrides.max_delay_ms,
    );
    OverrideMapper::set_u32(
        &mut mapper.config.core.retry.jitter_ratio_pct,
        overrides.jitter_ratio_pct,
    );
}

fn apply_embedding_overrides(config: &mut BackendConfig, overrides: &EmbeddingConfigOverrides) {
    let mapper = OverrideMapper::new(config);
    OverrideMapper::set_opt_box_str(
        &mut mapper.config.embedding.provider,
        overrides.provider.as_deref(),
    );
    OverrideMapper::set_opt_box_str(
        &mut mapper.config.embedding.model,
        overrides.model.as_deref(),
    );
    OverrideMapper::set_opt_box_str(
        &mut mapper.config.embedding.base_url,
        overrides.base_url.as_deref(),
    );
    OverrideMapper::set_opt_u32(&mut mapper.config.embedding.dimension, overrides.dimension);
    OverrideMapper::set_u64(
        &mut mapper.config.embedding.timeout_ms,
        overrides.timeout_ms,
    );
    OverrideMapper::set_u32(
        &mut mapper.config.embedding.batch_size,
        overrides.batch_size,
    );
    OverrideMapper::set_bool(
        &mut mapper.config.embedding.local_first,
        overrides.local_first,
    );
    OverrideMapper::set_bool(
        &mut mapper.config.embedding.local_only,
        overrides.local_only,
    );

    apply_embedding_routing_overrides(config, &overrides.routing);
    apply_embedding_jobs_overrides(config, &overrides.jobs);
    apply_embedding_onnx_overrides(config, &overrides.onnx);
    apply_embedding_cache_overrides(config, &overrides.cache);
}

const fn apply_embedding_routing_overrides(
    config: &mut BackendConfig,
    overrides: &EmbeddingRoutingOverrides,
) {
    if overrides.mode.is_some() {
        config.embedding.routing.mode = overrides.mode;
    }
    if overrides.split.max_remote_batches.is_some() {
        config.embedding.routing.split.max_remote_batches = overrides.split.max_remote_batches;
    }
}

const fn apply_embedding_jobs_overrides(
    config: &mut BackendConfig,
    overrides: &EmbeddingJobsOverrides,
) {
    let mapper = OverrideMapper::new(config);
    OverrideMapper::set_u64(
        &mut mapper.config.embedding.jobs.progress_interval_ms,
        overrides.progress_interval_ms,
    );
    OverrideMapper::set_u64(
        &mut mapper.config.embedding.jobs.cancel_poll_interval_ms,
        overrides.cancel_poll_interval_ms,
    );
}

fn apply_embedding_onnx_overrides(config: &mut BackendConfig, overrides: &EmbeddingOnnxOverrides) {
    let mapper = OverrideMapper::new(config);
    OverrideMapper::set_opt_box_str(
        &mut mapper.config.embedding.onnx.model_dir,
        overrides.model_dir.as_deref(),
    );
    OverrideMapper::set_opt_box_str(
        &mut mapper.config.embedding.onnx.model_filename,
        overrides.model_filename.as_deref(),
    );
    OverrideMapper::set_opt_box_str(
        &mut mapper.config.embedding.onnx.tokenizer_filename,
        overrides.tokenizer_filename.as_deref(),
    );
    OverrideMapper::set_opt_box_str(
        &mut mapper.config.embedding.onnx.repo,
        overrides.repo.as_deref(),
    );
    OverrideMapper::set_bool(
        &mut mapper.config.embedding.onnx.download_on_missing,
        overrides.download_on_missing,
    );
    OverrideMapper::set_u32(
        &mut mapper.config.embedding.onnx.session_pool_size,
        overrides.session_pool_size,
    );
}

fn apply_embedding_cache_overrides(
    config: &mut BackendConfig,
    overrides: &EmbeddingCacheOverrides,
) {
    let mapper = OverrideMapper::new(config);
    OverrideMapper::set_bool(
        &mut mapper.config.embedding.cache.enabled,
        overrides.enabled,
    );
    OverrideMapper::set_u32(
        &mut mapper.config.embedding.cache.max_entries,
        overrides.max_entries,
    );
    OverrideMapper::set_u64(
        &mut mapper.config.embedding.cache.max_bytes,
        overrides.max_bytes,
    );
    OverrideMapper::set_bool(
        &mut mapper.config.embedding.cache.disk_enabled,
        overrides.disk_enabled,
    );
    if let Some(provider) = overrides.disk_provider {
        mapper.config.embedding.cache.disk_provider = Some(provider);
    }
    OverrideMapper::set_opt_box_str(
        &mut mapper.config.embedding.cache.disk_path,
        overrides.disk_path.as_deref(),
    );
    OverrideMapper::set_opt_box_str(
        &mut mapper.config.embedding.cache.disk_connection,
        overrides.disk_connection.as_deref(),
    );
    OverrideMapper::set_opt_box_str(
        &mut mapper.config.embedding.cache.disk_table,
        overrides.disk_table.as_deref(),
    );
    OverrideMapper::set_u64(
        &mut mapper.config.embedding.cache.disk_max_bytes,
        overrides.disk_max_bytes,
    );
}

fn apply_vector_db_overrides(config: &mut BackendConfig, overrides: &VectorDbConfigOverrides) {
    let mapper = OverrideMapper::new(config);
    OverrideMapper::set_opt_box_str(
        &mut mapper.config.vector_db.provider,
        overrides.provider.as_deref(),
    );
    OverrideMapper::set_opt_box_str(
        &mut mapper.config.vector_db.address,
        overrides.address.as_deref(),
    );
    OverrideMapper::set_opt_index_mode(
        &mut mapper.config.vector_db.index_mode,
        overrides.index_mode,
    );
    OverrideMapper::set_opt_box_str(
        &mut mapper.config.vector_db.base_url,
        overrides.base_url.as_deref(),
    );
    OverrideMapper::set_opt_box_str(
        &mut mapper.config.vector_db.database,
        overrides.database.as_deref(),
    );
    OverrideMapper::set_opt_box_str(
        &mut mapper.config.vector_db.token,
        overrides.token.as_deref(),
    );
    OverrideMapper::set_opt_box_str(
        &mut mapper.config.vector_db.username,
        overrides.username.as_deref(),
    );
    OverrideMapper::set_opt_box_str(
        &mut mapper.config.vector_db.password,
        overrides.password.as_deref(),
    );
    OverrideMapper::set_bool(&mut mapper.config.vector_db.ssl, overrides.ssl);
    OverrideMapper::set_u64(
        &mut mapper.config.vector_db.timeout_ms,
        overrides.timeout_ms,
    );
    OverrideMapper::set_u64(
        &mut mapper.config.vector_db.index_timeout_ms,
        overrides.index_timeout_ms,
    );
    if let Some(index) = overrides.index.as_ref() {
        mapper.config.vector_db.index = index.clone();
    }
    OverrideMapper::set_u32(
        &mut mapper.config.vector_db.batch_size,
        overrides.batch_size,
    );
}

fn apply_sync_overrides(config: &mut BackendConfig, overrides: &SyncConfigOverrides) {
    let mapper = OverrideMapper::new(config);
    OverrideMapper::set_clone(
        &mut mapper.config.sync.allowed_extensions,
        overrides.allowed_extensions.as_ref(),
    );
    OverrideMapper::set_clone(
        &mut mapper.config.sync.ignore_patterns,
        overrides.ignore_patterns.as_ref(),
    );
    OverrideMapper::set_u32(&mut mapper.config.sync.max_files, overrides.max_files);
    OverrideMapper::set_u64(
        &mut mapper.config.sync.max_file_size_bytes,
        overrides.max_file_size_bytes,
    );
}

struct OverrideMapper<'a> {
    config: &'a mut BackendConfig,
}

impl<'a> OverrideMapper<'a> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_precedence_is_deterministic() -> Result<(), Box<dyn std::error::Error>> {
        let config_json = r#"{
          "version": 1,
          "core": { "timeoutMs": 45000 }
        }"#;

        let overrides_json = r#"{
          "core": { "timeoutMs": 50000 }
        }"#;

        let env = BackendEnv {
            core_timeout_ms: Some(60000),
            ..BackendEnv::default()
        };

        let config =
            load_backend_config_from_sources(Some(config_json), Some(overrides_json), &env)?;
        assert_eq!(config.core.timeout_ms, 60000);
        Ok(())
    }

    #[test]
    fn serialization_is_deterministic() -> Result<(), Box<dyn std::error::Error>> {
        let env = BackendEnv::default();
        let config = load_backend_config_from_sources(None, None, &env)?;
        let first = to_pretty_json(&config)?;
        let second = to_pretty_json(&config)?;
        assert_eq!(first, second);
        Ok(())
    }

    // =========================================================================
    // ERROR PRECEDENCE TESTS (Code Review Fix 5)
    // =========================================================================

    #[test]
    fn missing_config_with_valid_env_uses_defaults() -> Result<(), Box<dyn std::error::Error>> {
        // No config file, but valid env settings
        let env = BackendEnv {
            core_timeout_ms: Some(30000),
            core_max_concurrency: Some(8),
            ..BackendEnv::default()
        };

        let config = load_backend_config_from_sources(None, None, &env)?;

        // Env values should be applied
        assert_eq!(config.core.timeout_ms, 30000);
        assert_eq!(config.core.max_concurrency, 8);
        // Other fields should use defaults
        assert_eq!(config.version, 1);
        Ok(())
    }

    #[test]
    fn invalid_config_value_overridden_by_valid_env_succeeds()
    -> Result<(), Box<dyn std::error::Error>> {
        // Config has invalid timeout (0 is too low), but env overrides with valid value
        let config_json = r#"{
          "version": 1,
          "core": { "timeoutMs": 500 }
        }"#;
        // Note: 500ms is below the minimum of 1000ms

        // Env provides a valid timeout that will override the invalid one
        let env = BackendEnv {
            core_timeout_ms: Some(30000),
            ..BackendEnv::default()
        };

        // This should succeed because env overrides the invalid config value
        // before validation runs
        let config = load_backend_config_from_sources(Some(config_json), None, &env)?;
        assert_eq!(config.core.timeout_ms, 30000);
        Ok(())
    }

    #[test]
    fn valid_config_with_invalid_overrides_fails() -> Result<(), Box<dyn std::error::Error>> {
        // Valid base config
        let config_json = r#"{
          "version": 1,
          "core": { "timeoutMs": 30000 }
        }"#;

        // Invalid overrides JSON (malformed)
        let overrides_json = r#"{ "core": { "timeoutMs": }"#; // malformed JSON

        let env = BackendEnv::default();

        let result =
            load_backend_config_from_sources(Some(config_json), Some(overrides_json), &env);
        assert!(result.is_err());

        let error = result
            .err()
            .ok_or_else(|| std::io::Error::other("expected overrides error"))?;
        assert_eq!(error.code, ErrorCode::new("config", "invalid_json"));
        assert_eq!(
            error.metadata.get("source").map(String::as_str),
            Some("overrides")
        );
        Ok(())
    }

    #[test]
    fn env_validation_fails_with_invalid_env_value() -> Result<(), Box<dyn std::error::Error>> {
        // Valid config
        let config_json = r#"{
          "version": 1,
          "core": { "timeoutMs": 30000 }
        }"#;

        // Env has an invalid value (timeout too low)
        let env = BackendEnv {
            core_timeout_ms: Some(100), // Below minimum of 1000ms
            ..BackendEnv::default()
        };

        let result = load_backend_config_from_sources(Some(config_json), None, &env);
        assert!(result.is_err());

        let error = result
            .err()
            .ok_or_else(|| std::io::Error::other("expected env validation error"))?;
        assert_eq!(error.code, ErrorCode::new("config", "invalid_timeout"));
        Ok(())
    }
}
