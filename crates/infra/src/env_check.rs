//! Environment validation helpers for CLI surfaces.

use semantic_code_config::load_backend_config_from_sources;
use semantic_code_shared::ErrorEnvelope;
use std::collections::BTreeMap;

/// Infra-level error type (shared error envelope).
pub type InfraError = ErrorEnvelope;

/// Infra-level result type.
pub type InfraResult<T> = Result<T, InfraError>;

/// Validate that the provided env overrides can be parsed and merged into a config.
pub fn validate_env_parsing(env: &BTreeMap<String, String>) -> InfraResult<()> {
    let _ = load_backend_config_from_sources(None, None, env)?;
    Ok(())
}
