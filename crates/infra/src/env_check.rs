//! Environment validation helpers for CLI surfaces.

use semantic_code_config::{BackendConfig, BackendEnv, apply_env_overrides};
use semantic_code_shared::ErrorEnvelope;
use std::collections::BTreeMap;

/// Infra-level error type (shared error envelope).
pub type InfraError = ErrorEnvelope;

/// Infra-level result type.
pub type InfraResult<T> = Result<T, InfraError>;

/// Validate that the provided env overrides can be parsed and merged into a config.
pub fn validate_env_parsing(env: &BTreeMap<String, String>) -> InfraResult<()> {
    let parsed = BackendEnv::from_map(env).map_err(ErrorEnvelope::from)?;
    let _ = apply_env_overrides(BackendConfig::default(), &parsed)?;
    Ok(())
}
