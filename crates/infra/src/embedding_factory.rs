//! Embedding adapter selection and local fallback wiring.

use crate::InfraResult;
use crate::embedding_router::SplitEmbeddingRouter;
use semantic_code_adapters::cache::{
    CachingEmbedding, DiskCacheProvider, EmbeddingCache, EmbeddingCacheConfig,
};
use semantic_code_adapters::embedding::fixed::FixedDimensionEmbedding;
use semantic_code_adapters::embedding::gemini::{GeminiEmbedding, GeminiEmbeddingConfig};
use semantic_code_adapters::embedding::ollama::{OllamaEmbedding, OllamaEmbeddingConfig};
use semantic_code_adapters::embedding::onnx::{OnnxEmbedding, OnnxEmbeddingConfig};
use semantic_code_adapters::embedding::openai::{OpenAiEmbedding, OpenAiEmbeddingConfig};
use semantic_code_adapters::embedding::voyage::{VoyageEmbedding, VoyageEmbeddingConfig};
use semantic_code_adapters::embedding_test::TestEmbedding;
use semantic_code_config::{
    BackendEnv, EmbeddingCacheDiskProvider, EmbeddingConfig, EmbeddingRoutingMode,
    ValidatedBackendConfig,
};
use semantic_code_ports::{EmbeddingPort, TelemetryPort};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RetryPolicy, SecretString};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

const DEFAULT_TEST_EMBEDDING_DIMENSION: u32 = 8;
const FIXED_EMBEDDING_DIMENSIONS: &[u32] =
    &[DEFAULT_TEST_EMBEDDING_DIMENSION, 384, 768, 1024, 1536];
const DEFAULT_ONNX_REPO: &str = "Xenova/all-MiniLM-L6-v2";
const CONTEXT_DIR: &str = ".context";
const MODELS_DIR: &str = "models";
const ONNX_MODELS_DIR: &str = "onnx";
const ONNX_CACHE_DIR: &str = "onnx-cache";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderKind {
    Auto,
    Test,
    Onnx,
    OpenAi,
    Gemini,
    Voyage,
    Ollama,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalMode {
    Disabled,
    First,
    Only,
}

/// Build an embedding port using config and env overrides.
pub fn build_embedding_port(
    config: &ValidatedBackendConfig,
    env: &BackendEnv,
    codebase_root: &Path,
) -> InfraResult<Arc<dyn EmbeddingPort>> {
    build_embedding_port_with_telemetry(config, env, codebase_root, None)
}

/// Build an embedding port and attach telemetry hooks when provided.
pub fn build_embedding_port_with_telemetry(
    config: &ValidatedBackendConfig,
    env: &BackendEnv,
    codebase_root: &Path,
    telemetry: Option<Arc<dyn TelemetryPort>>,
) -> InfraResult<Arc<dyn EmbeddingPort>> {
    let provider = parse_provider(config.embedding.provider.as_deref())?;
    let allow_test_fallback = env.embedding_test_fallback.unwrap_or(false);
    let routing_mode = config.embedding.routing.mode;

    if routing_mode == Some(EmbeddingRoutingMode::Split) {
        return build_split_embedding_port(config, env, codebase_root, provider, telemetry);
    }

    let local_mode = match routing_mode {
        Some(EmbeddingRoutingMode::LocalFirst) => LocalMode::First,
        Some(EmbeddingRoutingMode::RemoteFirst) => LocalMode::Disabled,
        _ => resolve_local_mode(&config.embedding),
    };

    let port: Arc<dyn EmbeddingPort> = match provider {
        ProviderKind::Test => Ok(wrap_embedding_fixed(
            Some(embed_dimension(config)),
            TestEmbedding::new(embed_dimension(config))?,
        )),
        ProviderKind::Onnx => build_onnx(config, codebase_root, allow_test_fallback),
        ProviderKind::Auto => {
            build_auto(config, env, codebase_root, local_mode, allow_test_fallback)
        },
        ProviderKind::OpenAi => build_remote_with_fallback(
            config,
            env,
            codebase_root,
            local_mode,
            ProviderKind::OpenAi,
            allow_test_fallback,
        ),
        ProviderKind::Gemini => build_remote_with_fallback(
            config,
            env,
            codebase_root,
            local_mode,
            ProviderKind::Gemini,
            allow_test_fallback,
        ),
        ProviderKind::Voyage => build_remote_with_fallback(
            config,
            env,
            codebase_root,
            local_mode,
            ProviderKind::Voyage,
            allow_test_fallback,
        ),
        ProviderKind::Ollama => build_remote_with_fallback(
            config,
            env,
            codebase_root,
            local_mode,
            ProviderKind::Ollama,
            allow_test_fallback,
        ),
    }?;

    wrap_with_resilience(port, config, env, codebase_root, telemetry)
}

fn build_auto(
    config: &ValidatedBackendConfig,
    env: &BackendEnv,
    codebase_root: &Path,
    local_mode: LocalMode,
    allow_test_fallback: bool,
) -> InfraResult<Arc<dyn EmbeddingPort>> {
    if local_mode == LocalMode::Only {
        return build_onnx(config, codebase_root, allow_test_fallback);
    }

    let local_attempt = if local_mode == LocalMode::First {
        let attempt = try_build_onnx(config, codebase_root);
        if let Ok(Some(local)) = attempt {
            return Ok(local);
        }
        Some(attempt)
    } else {
        None
    };

    if let Some(provider) = auto_remote_provider(env) {
        return build_remote(config, env, provider);
    }

    let attempt = local_attempt.unwrap_or_else(|| try_build_onnx(config, codebase_root));
    match attempt {
        Ok(Some(local)) => Ok(local),
        Ok(None) => fallback_or_missing(config, allow_test_fallback, ProviderKind::Auto),
        Err(error) => Err(error),
    }
}

fn build_split_embedding_port(
    config: &ValidatedBackendConfig,
    env: &BackendEnv,
    codebase_root: &Path,
    provider: ProviderKind,
    telemetry: Option<Arc<dyn TelemetryPort>>,
) -> InfraResult<Arc<dyn EmbeddingPort>> {
    let max_remote_batches = config
        .embedding
        .routing
        .split
        .max_remote_batches
        .ok_or_else(|| {
            ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "split routing requires embedding.routing.split.maxRemoteBatches",
            )
        })?;
    let allow_test_fallback = env.embedding_test_fallback.unwrap_or(false);
    let local = build_onnx(config, codebase_root, allow_test_fallback)?;
    let remote_provider = resolve_split_remote_provider(provider, env)?;
    let remote = build_remote(config, env, remote_provider)?;

    let local = wrap_with_resilience(local, config, env, codebase_root, telemetry.clone())?;
    let remote = wrap_with_resilience(remote, config, env, codebase_root, telemetry)?;

    Ok(Arc::new(SplitEmbeddingRouter::new(
        local,
        remote,
        max_remote_batches,
    )?))
}

fn resolve_split_remote_provider(
    provider: ProviderKind,
    env: &BackendEnv,
) -> InfraResult<ProviderKind> {
    match provider {
        ProviderKind::OpenAi
        | ProviderKind::Gemini
        | ProviderKind::Voyage
        | ProviderKind::Ollama => Ok(provider),
        ProviderKind::Auto => auto_remote_provider(env).ok_or_else(|| {
            ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "split routing requires a configured remote provider",
            )
        }),
        ProviderKind::Onnx | ProviderKind::Test => Err(ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            "split routing requires a remote embedding provider",
        )),
    }
}

fn build_remote_with_fallback(
    config: &ValidatedBackendConfig,
    env: &BackendEnv,
    codebase_root: &Path,
    local_mode: LocalMode,
    provider: ProviderKind,
    allow_test_fallback: bool,
) -> InfraResult<Arc<dyn EmbeddingPort>> {
    if local_mode == LocalMode::Disabled && !remote_ready(provider, env) {
        return fallback_or_missing(config, allow_test_fallback, provider);
    }

    let mut local_error = None;
    if local_mode == LocalMode::First {
        match try_build_onnx(config, codebase_root) {
            Ok(Some(local)) => return Ok(local),
            Ok(None) => {},
            Err(error) => {
                local_error = Some(error);
            },
        }
    }

    if remote_ready(provider, env) {
        return build_remote(config, env, provider);
    }

    let local = try_build_onnx(config, codebase_root);
    if let Ok(Some(local)) = local.as_ref() {
        return Ok(Arc::clone(local));
    }
    if matches!(local.as_ref(), Ok(None)) {
        if let Some(error) = local_error {
            return Err(error);
        }
        return fallback_or_missing(config, allow_test_fallback, provider);
    }
    local?;
    if let Some(error) = local_error {
        return Err(error);
    }

    Err(missing_provider_error(provider))
}

fn build_remote(
    config: &ValidatedBackendConfig,
    env: &BackendEnv,
    provider: ProviderKind,
) -> InfraResult<Arc<dyn EmbeddingPort>> {
    match provider {
        ProviderKind::OpenAi => {
            let api_key = resolve_api_key(provider, env).ok_or_else(|| {
                ErrorEnvelope::expected(ErrorCode::invalid_input(), "OpenAI API key is required")
            })?;
            let api_key = api_key.into_inner();
            let mut adapter_config =
                OpenAiEmbeddingConfig::from_embedding_config(api_key, &config.embedding);
            adapter_config.model = resolve_model(provider, &config.embedding, env);
            adapter_config.base_url = resolve_base_url(provider, &config.embedding, env);
            Ok(wrap_embedding_fixed(
                config.embedding.dimension,
                OpenAiEmbedding::new(&adapter_config)?,
            ))
        },
        ProviderKind::Gemini => {
            let api_key = resolve_api_key(provider, env).ok_or_else(|| {
                ErrorEnvelope::expected(ErrorCode::invalid_input(), "Gemini API key is required")
            })?;
            let api_key = api_key.into_inner();
            let mut adapter_config =
                GeminiEmbeddingConfig::from_embedding_config(api_key, &config.embedding);
            adapter_config.model = resolve_model(provider, &config.embedding, env);
            adapter_config.base_url = resolve_base_url(provider, &config.embedding, env);
            Ok(wrap_embedding_fixed(
                config.embedding.dimension,
                GeminiEmbedding::new(&adapter_config)?,
            ))
        },
        ProviderKind::Voyage => {
            let api_key = resolve_api_key(provider, env).ok_or_else(|| {
                ErrorEnvelope::expected(ErrorCode::invalid_input(), "Voyage API key is required")
            })?;
            let api_key = api_key.into_inner();
            let mut adapter_config =
                VoyageEmbeddingConfig::from_embedding_config(api_key, &config.embedding);
            adapter_config.model = resolve_model(provider, &config.embedding, env);
            adapter_config.base_url = resolve_base_url(provider, &config.embedding, env);
            Ok(wrap_embedding_fixed(
                config.embedding.dimension,
                VoyageEmbedding::new(&adapter_config)?,
            ))
        },
        ProviderKind::Ollama => {
            let mut adapter_config =
                OllamaEmbeddingConfig::from_embedding_config(&config.embedding);
            adapter_config.model = resolve_model(provider, &config.embedding, env);
            adapter_config.base_url = resolve_base_url(provider, &config.embedding, env);
            Ok(wrap_embedding_fixed(
                config.embedding.dimension,
                OllamaEmbedding::new(&adapter_config)?,
            ))
        },
        _ => Err(missing_provider_error(provider)),
    }
}

fn wrap_with_resilience(
    port: Arc<dyn EmbeddingPort>,
    config: &ValidatedBackendConfig,
    env: &BackendEnv,
    codebase_root: &Path,
    telemetry: Option<Arc<dyn TelemetryPort>>,
) -> InfraResult<Arc<dyn EmbeddingPort>> {
    let cache_namespace = build_cache_namespace(&port, config, env);
    let cache_config = build_cache_config(config, codebase_root)?;
    let cache = EmbeddingCache::new(&cache_config)?;
    let retry_policy = RetryPolicy {
        max_attempts: config.core.retry.max_attempts,
        base_delay_ms: config.core.retry.base_delay_ms,
        max_delay_ms: config.core.retry.max_delay_ms,
        jitter_ratio_pct: config.core.retry.jitter_ratio_pct,
    };
    let timeout_ms = config.embedding.timeout_ms;
    let max_in_flight = max_in_flight_embedding_requests(config, &port);

    Ok(Arc::new(CachingEmbedding::new(
        port,
        cache,
        cache_namespace,
        retry_policy,
        timeout_ms,
        max_in_flight,
        telemetry,
    )))
}

fn build_cache_namespace(
    port: &Arc<dyn EmbeddingPort>,
    config: &ValidatedBackendConfig,
    env: &BackendEnv,
) -> Box<str> {
    let provider = port.provider().id.as_str();
    let provider_kind = match provider {
        "openai" => Some(ProviderKind::OpenAi),
        "gemini" => Some(ProviderKind::Gemini),
        "voyage" => Some(ProviderKind::Voyage),
        "ollama" => Some(ProviderKind::Ollama),
        "onnx" | "local" => Some(ProviderKind::Onnx),
        "test" => Some(ProviderKind::Test),
        _ => None,
    };
    let model = provider_kind
        .and_then(|kind| resolve_model(kind, &config.embedding, env))
        .or_else(|| config.embedding.model.clone());
    let base_url = provider_kind
        .and_then(|kind| resolve_base_url(kind, &config.embedding, env))
        .or_else(|| config.embedding.base_url.clone());
    let dimension = config.embedding.dimension.unwrap_or(0);

    let model_str = model.as_deref().unwrap_or("");
    let base_str = base_url.as_deref().unwrap_or("");
    format!("provider={provider};model={model_str};base_url={base_str};dimension={dimension}")
        .into_boxed_str()
}

fn max_in_flight_embedding_requests(
    config: &ValidatedBackendConfig,
    port: &Arc<dyn EmbeddingPort>,
) -> Option<usize> {
    let provider = port.provider().id.as_str();
    if !matches!(provider, "openai" | "gemini" | "voyage" | "ollama") {
        return None;
    }
    config
        .core
        .max_in_flight_embedding_batches
        .and_then(|value| usize::try_from(value).ok())
}

fn build_cache_config(
    config: &ValidatedBackendConfig,
    codebase_root: &Path,
) -> InfraResult<EmbeddingCacheConfig> {
    let cache_config = &config.embedding.cache;
    let disk_provider = map_disk_provider(cache_config.disk_provider);
    let disk_path = if cache_config.disk_enabled && disk_provider == DiskCacheProvider::Sqlite {
        Some(
            cache_config
                .disk_path
                .as_deref()
                .map_or_else(|| default_cache_path(codebase_root), PathBuf::from),
        )
    } else {
        None
    };

    let max_entries = usize::try_from(cache_config.max_entries).map_err(|_| {
        ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            "embedding cache maxEntries is too large",
        )
    })?;

    Ok(EmbeddingCacheConfig {
        enabled: cache_config.enabled,
        max_entries,
        max_bytes: cache_config.max_bytes,
        disk_enabled: cache_config.disk_enabled,
        disk_provider,
        disk_path,
        disk_connection: cache_config.disk_connection.clone(),
        disk_table: cache_config.disk_table.clone(),
        disk_max_bytes: Some(cache_config.disk_max_bytes),
    })
}

fn map_disk_provider(provider: Option<EmbeddingCacheDiskProvider>) -> DiskCacheProvider {
    match provider.unwrap_or(EmbeddingCacheDiskProvider::Sqlite) {
        EmbeddingCacheDiskProvider::Sqlite => DiskCacheProvider::Sqlite,
        EmbeddingCacheDiskProvider::Postgres => DiskCacheProvider::Postgres,
        EmbeddingCacheDiskProvider::Mysql => DiskCacheProvider::Mysql,
        EmbeddingCacheDiskProvider::Mssql => DiskCacheProvider::Mssql,
    }
}

fn default_cache_path(codebase_root: &Path) -> PathBuf {
    codebase_root
        .join(CONTEXT_DIR)
        .join("cache")
        .join("embeddings")
        .join("cache.db")
}

fn remote_ready(provider: ProviderKind, env: &BackendEnv) -> bool {
    match provider {
        ProviderKind::OpenAi | ProviderKind::Gemini | ProviderKind::Voyage => {
            resolve_api_key(provider, env).is_some()
        },
        ProviderKind::Ollama => true,
        _ => false,
    }
}

fn resolve_api_key(provider: ProviderKind, env: &BackendEnv) -> Option<SecretString> {
    match provider {
        ProviderKind::OpenAi => env
            .openai_api_key
            .clone()
            .or_else(|| env.embedding_api_key.clone()),
        ProviderKind::Gemini => env
            .gemini_api_key
            .clone()
            .or_else(|| env.embedding_api_key.clone()),
        ProviderKind::Voyage => env
            .voyage_api_key
            .clone()
            .or_else(|| env.embedding_api_key.clone()),
        _ => None,
    }
}

fn resolve_model(
    provider: ProviderKind,
    config: &EmbeddingConfig,
    env: &BackendEnv,
) -> Option<Box<str>> {
    match provider {
        ProviderKind::OpenAi => env.openai_model.clone().or_else(|| config.model.clone()),
        ProviderKind::Gemini => env.gemini_model.clone().or_else(|| config.model.clone()),
        ProviderKind::Voyage => env.voyage_model.clone().or_else(|| config.model.clone()),
        ProviderKind::Ollama => env.ollama_model.clone().or_else(|| config.model.clone()),
        _ => config.model.clone(),
    }
}

fn resolve_base_url(
    provider: ProviderKind,
    config: &EmbeddingConfig,
    env: &BackendEnv,
) -> Option<Box<str>> {
    match provider {
        ProviderKind::OpenAi => env
            .openai_base_url
            .clone()
            .or_else(|| config.base_url.clone()),
        ProviderKind::Gemini => env
            .gemini_base_url
            .clone()
            .or_else(|| config.base_url.clone()),
        ProviderKind::Voyage => env
            .voyage_base_url
            .clone()
            .or_else(|| config.base_url.clone()),
        ProviderKind::Ollama => env.ollama_host.clone().or_else(|| config.base_url.clone()),
        _ => config.base_url.clone(),
    }
}

const fn auto_remote_provider(env: &BackendEnv) -> Option<ProviderKind> {
    if env.openai_api_key.is_some() || env.embedding_api_key.is_some() {
        return Some(ProviderKind::OpenAi);
    }
    if env.gemini_api_key.is_some() {
        return Some(ProviderKind::Gemini);
    }
    if env.voyage_api_key.is_some() {
        return Some(ProviderKind::Voyage);
    }
    if env.ollama_model.is_some() || env.ollama_host.is_some() {
        return Some(ProviderKind::Ollama);
    }
    None
}

fn build_onnx(
    config: &ValidatedBackendConfig,
    codebase_root: &Path,
    allow_test_fallback: bool,
) -> InfraResult<Arc<dyn EmbeddingPort>> {
    match try_build_onnx(config, codebase_root) {
        Ok(Some(local)) => Ok(local),
        Ok(None) => fallback_or_missing(config, allow_test_fallback, ProviderKind::Onnx),
        Err(error) if allow_test_fallback && is_onnx_assets_missing(&error) => {
            Ok(wrap_embedding_fixed(
                Some(embed_dimension(config)),
                TestEmbedding::new(embed_dimension(config))?,
            ))
        },
        Err(error) => Err(error),
    }
}

fn try_build_onnx(
    config: &ValidatedBackendConfig,
    codebase_root: &Path,
) -> InfraResult<Option<Arc<dyn EmbeddingPort>>> {
    let (preferred_dir, legacy_dir) = resolve_onnx_model_dirs(codebase_root, &config.embedding);
    let model_filename = config.embedding.onnx.model_filename.as_deref();
    let tokenizer_filename = config.embedding.onnx.tokenizer_filename.as_deref();
    let mut model_dir = preferred_dir.clone();

    if !onnx_assets_present(&model_dir, model_filename, tokenizer_filename)
        && let Some(legacy) = legacy_dir.as_ref()
        && onnx_assets_present(legacy, model_filename, tokenizer_filename)
    {
        model_dir.clone_from(legacy);
    }

    if !onnx_assets_present(&model_dir, model_filename, tokenizer_filename) {
        if !config.embedding.onnx.download_on_missing {
            return Err(onnx_assets_missing_error(
                &model_dir,
                model_filename,
                tokenizer_filename,
            ));
        }
        if model_dir != preferred_dir {
            model_dir.clone_from(&preferred_dir);
        }
        let repo = config
            .embedding
            .onnx
            .repo
            .as_deref()
            .unwrap_or(DEFAULT_ONNX_REPO);
        let warnings = download_onnx_assets(repo, &model_dir)?;
        for warning in warnings {
            eprintln!("warning: {warning}");
        }
    }

    if !onnx_assets_present(&model_dir, model_filename, tokenizer_filename) {
        return Ok(None);
    }

    let session_pool_size = usize::try_from(config.embedding.onnx.session_pool_size)
        .unwrap_or(1)
        .max(1);
    let config = OnnxEmbeddingConfig {
        model_dir,
        model_filename: config.embedding.onnx.model_filename.clone(),
        tokenizer_filename: config.embedding.onnx.tokenizer_filename.clone(),
        dimension: config.embedding.dimension,
        session_pool_size,
    };
    Ok(Some(wrap_embedding_fixed(
        config.dimension,
        OnnxEmbedding::new(&config)?,
    )))
}

fn onnx_assets_present(
    model_dir: &Path,
    model_filename: Option<&str>,
    tokenizer_filename: Option<&str>,
) -> bool {
    let model_path = resolve_model_path(model_dir, model_filename);
    let tokenizer_path = resolve_tokenizer_path(model_dir, tokenizer_filename);
    model_path.exists() && tokenizer_path.exists()
}

fn onnx_assets_missing_error(
    model_dir: &Path,
    model_filename: Option<&str>,
    tokenizer_filename: Option<&str>,
) -> ErrorEnvelope {
    let tokenizer_path = resolve_tokenizer_path(model_dir, tokenizer_filename);
    let hint = "Set SCA_EMBEDDING_ONNX_MODEL_DIR or enable embedding.onnx.downloadOnMissing=true.";
    let message = model_filename.map_or_else(
        || {
            let nested = model_dir.join("onnx").join("model.onnx");
            let root = model_dir.join("model.onnx");
            format!(
                "ONNX assets not found. Expected model at {} or {} and tokenizer at {}. {hint}",
                nested.display(),
                root.display(),
                tokenizer_path.display()
            )
        },
        |filename| {
            let model_path = model_dir.join(filename);
            format!(
                "ONNX assets not found. Expected model at {} and tokenizer at {}. {hint}",
                model_path.display(),
                tokenizer_path.display()
            )
        },
    );
    ErrorEnvelope::expected(ErrorCode::new("embedding", "onnx_assets_missing"), message)
        .with_metadata("model_dir", model_dir.to_string_lossy().to_string())
        .with_metadata(
            "tokenizer_path",
            tokenizer_path.to_string_lossy().to_string(),
        )
}

fn resolve_model_path(model_dir: &Path, model_filename: Option<&str>) -> PathBuf {
    if let Some(filename) = model_filename {
        return model_dir.join(filename);
    }
    let nested = model_dir.join("onnx").join("model.onnx");
    if nested.exists() {
        return nested;
    }
    model_dir.join("model.onnx")
}

fn resolve_tokenizer_path(model_dir: &Path, tokenizer_filename: Option<&str>) -> PathBuf {
    let filename = tokenizer_filename.unwrap_or("tokenizer.json");
    model_dir.join(filename)
}

fn resolve_onnx_model_dirs(
    codebase_root: &Path,
    config: &EmbeddingConfig,
) -> (PathBuf, Option<PathBuf>) {
    // TODO: refactor repeated Option selection with a mapper helper.
    if let Some(model_dir) = config.onnx.model_dir.as_deref() {
        let path = PathBuf::from(model_dir);
        if path.is_absolute() {
            return (path, None);
        }
        return (codebase_root.join(path), None);
    }
    let repo = config.onnx.repo.as_deref().unwrap_or(DEFAULT_ONNX_REPO);
    let slug = repo.replace('/', "-");
    let models_root = codebase_root
        .join(CONTEXT_DIR)
        .join(MODELS_DIR)
        .join(ONNX_MODELS_DIR);
    let preferred = models_root.join(&slug);
    let legacy = codebase_root
        .join(CONTEXT_DIR)
        .join(ONNX_CACHE_DIR)
        .join(&slug);
    (preferred, Some(legacy))
}

fn download_onnx_assets(repo: &str, model_dir: &Path) -> InfraResult<Vec<String>> {
    std::fs::create_dir_all(model_dir).map_err(|error| {
        ErrorEnvelope::expected(
            ErrorCode::io(),
            format!("failed to create ONNX cache directory: {error}"),
        )
        .with_metadata("path", model_dir.to_string_lossy().to_string())
    })?;

    let required = ["onnx/model.onnx", "tokenizer.json"];
    let optional = [
        "tokenizer_config.json",
        "config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ];

    for filename in required {
        if let Err(error) = run_hf_download(repo, filename, model_dir) {
            eprintln!(
                "error: required ONNX asset download failed (repo={repo}, file={filename}): {error}"
            );
            return Err(error);
        }
    }
    let mut warnings = Vec::new();
    for filename in optional {
        if let Err(error) = run_hf_download(repo, filename, model_dir) {
            warnings.push(format!(
                "optional ONNX asset download failed (repo={repo}, file={filename}): {error}"
            ));
        }
    }

    Ok(warnings)
}

fn run_hf_download(repo: &str, filename: &str, model_dir: &Path) -> InfraResult<()> {
    match run_hf_command("hf", repo, filename, model_dir) {
        Ok(()) => Ok(()),
        Err(error) => {
            if error.kind == std::io::ErrorKind::NotFound {
                return Err(ErrorEnvelope::expected(
                    ErrorCode::invalid_input(),
                    "hf CLI is required for local ONNX downloads. Install `hf` or configure a remote embedding provider.",
                )
                .with_metadata("cli", "hf")
                .with_metadata("repo", repo.to_string())
                .with_metadata("file", filename.to_string()));
            }
            Err(error.into())
        },
    }
}

fn run_hf_command(
    cli_name: &str,
    repo: &str,
    filename: &str,
    model_dir: &Path,
) -> Result<(), HfCommandError> {
    let output = Command::new(cli_name)
        .arg("download")
        .arg(repo)
        .arg(filename)
        .arg("--local-dir")
        .arg(model_dir)
        .output()
        .map_err(|error| HfCommandError {
            kind: error.kind(),
            message: format!("failed to run {cli_name}: {error}"),
        })?;

    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(HfCommandError {
        kind: std::io::ErrorKind::Other,
        message: format!("{cli_name} download failed for {repo}/{filename}: {stderr}"),
    })
}

#[derive(Debug)]
struct HfCommandError {
    kind: std::io::ErrorKind,
    message: String,
}

impl From<HfCommandError> for ErrorEnvelope {
    fn from(error: HfCommandError) -> Self {
        Self::unexpected(
            ErrorCode::new("embedding", "onnx_download_failed"),
            error.message,
            ErrorClass::NonRetriable,
        )
    }
}

fn parse_provider(value: Option<&str>) -> InfraResult<ProviderKind> {
    let raw = value.unwrap_or("auto").trim();
    let normalized = raw.to_ascii_lowercase();
    match normalized.as_str() {
        "auto" => Ok(ProviderKind::Auto),
        "test" => Ok(ProviderKind::Test),
        "onnx" | "local" => Ok(ProviderKind::Onnx),
        "openai" => Ok(ProviderKind::OpenAi),
        "gemini" => Ok(ProviderKind::Gemini),
        "voyage" | "voyageai" => Ok(ProviderKind::Voyage),
        "ollama" => Ok(ProviderKind::Ollama),
        _ => Err(ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            format!("unsupported embedding provider: {raw}"),
        )
        .with_metadata("provider", raw.to_string())),
    }
}

const fn resolve_local_mode(config: &EmbeddingConfig) -> LocalMode {
    if config.local_only {
        LocalMode::Only
    } else if config.local_first {
        LocalMode::First
    } else {
        LocalMode::Disabled
    }
}

fn missing_provider_error(provider: ProviderKind) -> ErrorEnvelope {
    let message = match provider {
        ProviderKind::OpenAi => "OpenAI API key is required".to_string(),
        ProviderKind::Gemini => "Gemini API key is required".to_string(),
        ProviderKind::Voyage => "Voyage API key is required".to_string(),
        ProviderKind::Ollama => "embedding provider ollama is not configured".to_string(),
        ProviderKind::Onnx => "embedding provider onnx is not configured".to_string(),
        ProviderKind::Auto => "embedding provider auto is not configured".to_string(),
        ProviderKind::Test => "embedding provider test is not configured".to_string(),
    };
    ErrorEnvelope::expected(ErrorCode::invalid_input(), message)
}

fn is_onnx_assets_missing(error: &ErrorEnvelope) -> bool {
    error.code.namespace() == "embedding" && error.code.code() == "onnx_assets_missing"
}

fn fallback_or_missing(
    config: &ValidatedBackendConfig,
    allow_test_fallback: bool,
    provider: ProviderKind,
) -> InfraResult<Arc<dyn EmbeddingPort>> {
    if allow_test_fallback {
        return Ok(wrap_embedding_fixed(
            Some(embed_dimension(config)),
            TestEmbedding::new(embed_dimension(config))?,
        ));
    }
    Err(missing_provider_error(provider))
}

fn embed_dimension(config: &ValidatedBackendConfig) -> u32 {
    config
        .embedding
        .dimension
        .unwrap_or(DEFAULT_TEST_EMBEDDING_DIMENSION)
}

fn wrap_embedding_fixed<P: EmbeddingPort + 'static>(
    dimension: Option<u32>,
    port: P,
) -> Arc<dyn EmbeddingPort> {
    let dimension = dimension.filter(|value| FIXED_EMBEDDING_DIMENSIONS.contains(value));
    match dimension {
        Some(8) => Arc::new(FixedDimensionEmbedding::<P, 8>::new(port)),
        Some(384) => Arc::new(FixedDimensionEmbedding::<P, 384>::new(port)),
        Some(768) => Arc::new(FixedDimensionEmbedding::<P, 768>::new(port)),
        Some(1024) => Arc::new(FixedDimensionEmbedding::<P, 1024>::new(port)),
        Some(1536) => Arc::new(FixedDimensionEmbedding::<P, 1536>::new(port)),
        _ => Arc::new(port),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_ports::DetectDimensionOptions;
    use semantic_code_shared::RequestContext;

    #[tokio::test]
    async fn fixed_embedding_wrapper_reports_expected_dimension() -> InfraResult<()> {
        let ctx = RequestContext::new_request();
        let inner = TestEmbedding::new(768)?;
        let port = wrap_embedding_fixed(Some(768), inner);
        let dimension = port
            .detect_dimension(&ctx, DetectDimensionOptions::default().into())
            .await?;
        assert_eq!(dimension, 768);
        Ok(())
    }

    #[tokio::test]
    async fn fixed_embedding_wrapper_skips_unknown_dimension() -> InfraResult<()> {
        let ctx = RequestContext::new_request();
        let inner = TestEmbedding::new(777)?;
        let port = wrap_embedding_fixed(Some(777), inner);
        let dimension = port
            .detect_dimension(&ctx, DetectDimensionOptions::default().into())
            .await?;
        assert_eq!(dimension, 777);
        Ok(())
    }
}
