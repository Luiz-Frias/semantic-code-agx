//! Config loading helpers for CLI surfaces.

use crate::InfraResult;
use semantic_code_config::{BackendEnv, load_backend_config_from_path, to_pretty_json};
use std::collections::BTreeMap;
use std::path::Path;

/// Load and validate the effective config, returning deterministic pretty JSON.
pub fn load_effective_config_json(
    env: &BTreeMap<String, String>,
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
) -> InfraResult<String> {
    let env = BackendEnv::from_map(env).map_err(semantic_code_shared::ErrorEnvelope::from)?;
    let config = load_backend_config_from_path(config_path, overrides_json, &env)?;
    to_pretty_json(&config)
}
