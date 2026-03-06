//! Runtime environment view for infra/facade composition.
//!
//! This module exposes only the runtime-relevant subset needed outside this
//! crate, while keeping parser internals (`BackendEnv`, `EnvParseError`) private.

use crate::env::BackendEnv;
use semantic_code_shared::{ErrorEnvelope, SecretString};
use std::collections::BTreeMap;

/// Runtime environment overrides consumed by infra composition.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeEnv {
    /// Override for `embedding.provider`.
    pub embedding_provider: Option<Box<str>>,
    /// Override for `vectorDb.provider`.
    pub vector_db_provider: Option<Box<str>>,

    /// Allow fallback to test embeddings when ONNX assets are missing.
    pub embedding_test_fallback: Option<bool>,

    /// Secret: embedding API key (provider-agnostic).
    pub embedding_api_key: Option<SecretString>,
    /// Provider-specific OpenAI API key.
    pub openai_api_key: Option<SecretString>,
    /// Provider-specific Gemini API key.
    pub gemini_api_key: Option<SecretString>,
    /// Provider-specific Voyage API key.
    pub voyage_api_key: Option<SecretString>,

    /// Provider-specific OpenAI base URL.
    pub openai_base_url: Option<Box<str>>,
    /// Provider-specific OpenAI model override.
    pub openai_model: Option<Box<str>>,
    /// Provider-specific Gemini base URL.
    pub gemini_base_url: Option<Box<str>>,
    /// Provider-specific Gemini model override.
    pub gemini_model: Option<Box<str>>,
    /// Provider-specific Voyage base URL.
    pub voyage_base_url: Option<Box<str>>,
    /// Provider-specific Voyage model override.
    pub voyage_model: Option<Box<str>>,
    /// Provider-specific Ollama model name.
    pub ollama_model: Option<Box<str>>,
    /// Provider-specific Ollama host URL.
    pub ollama_host: Option<Box<str>>,

    /// Secret: vector DB token.
    pub vector_db_token: Option<SecretString>,
    /// Secret: vector DB password.
    pub vector_db_password: Option<SecretString>,
}

impl From<BackendEnv> for RuntimeEnv {
    fn from(value: BackendEnv) -> Self {
        Self {
            embedding_provider: value.embedding_provider,
            vector_db_provider: value.vector_db_provider,
            embedding_test_fallback: value.embedding_test_fallback,
            embedding_api_key: value.embedding_api_key,
            openai_api_key: value.openai_api_key,
            gemini_api_key: value.gemini_api_key,
            voyage_api_key: value.voyage_api_key,
            openai_base_url: value.openai_base_url,
            openai_model: value.openai_model,
            gemini_base_url: value.gemini_base_url,
            gemini_model: value.gemini_model,
            voyage_base_url: value.voyage_base_url,
            voyage_model: value.voyage_model,
            ollama_model: value.ollama_model,
            ollama_host: value.ollama_host,
            vector_db_token: value.vector_db_token,
            vector_db_password: value.vector_db_password,
        }
    }
}

/// Parse runtime environment overrides from an explicit env map.
pub fn load_runtime_env_from_map(
    env: &BTreeMap<String, String>,
) -> Result<RuntimeEnv, ErrorEnvelope> {
    let parsed = BackendEnv::from_map(env).map_err(ErrorEnvelope::from)?;
    Ok(parsed.into())
}

/// Parse runtime environment overrides from process environment variables.
pub fn load_runtime_env_std_env() -> Result<RuntimeEnv, ErrorEnvelope> {
    let parsed = BackendEnv::from_std_env().map_err(ErrorEnvelope::from)?;
    Ok(parsed.into())
}
