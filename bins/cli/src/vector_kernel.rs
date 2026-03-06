//! Helpers to resolve and serialize effective vector-kernel metadata.

#[cfg(any(debug_assertions, feature = "dev-tools"))]
use semantic_code_facade::resolve_vector_kernel_kind_from_env;
use semantic_code_facade::{CliVectorKernelKind, InfraError, resolve_vector_kernel_kind_std_env};
#[cfg(any(debug_assertions, feature = "dev-tools"))]
use std::collections::BTreeMap;
use std::path::Path;

/// Effective vector-kernel metadata used by CLI output payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorKernelMetadata {
    effective: CliVectorKernelKind,
}

impl VectorKernelMetadata {
    /// Build metadata from a concrete kernel.
    #[must_use]
    pub const fn new(effective: CliVectorKernelKind) -> Self {
        Self { effective }
    }

    /// Effective kernel label (`hnsw-rs`, `dfrr`).
    #[must_use]
    pub const fn effective_label(self) -> &'static str {
        self.effective.effective_label()
    }

    /// JSON shape used by CLI machine-readable outputs.
    #[must_use]
    pub fn as_json(self) -> serde_json::Value {
        serde_json::json!({
            "effective": self.effective_label(),
        })
    }

    /// Returns `true` when the effective kernel is experimental.
    #[must_use]
    pub const fn is_experimental(self) -> bool {
        self.effective.is_experimental()
    }
}

/// Emit a stderr warning if the resolved kernel is experimental.
pub fn warn_if_experimental(metadata: VectorKernelMetadata) {
    if metadata.is_experimental() {
        eprintln!(
            "warning: using experimental vector kernel '{}'. Results may differ from the default 'hnsw-rs' kernel.",
            metadata.effective_label()
        );
    }
}

/// Resolve effective kernel using process environment.
pub fn resolve_vector_kernel_metadata_std_env(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
) -> Result<VectorKernelMetadata, InfraError> {
    let kernel = resolve_vector_kernel_kind_std_env(config_path, overrides_json)?;
    Ok(VectorKernelMetadata::new(kernel))
}

/// Resolve effective kernel using explicit environment map.
#[cfg(any(debug_assertions, feature = "dev-tools"))]
pub fn resolve_vector_kernel_metadata_from_env(
    env: &BTreeMap<String, String>,
) -> Result<VectorKernelMetadata, InfraError> {
    let kernel = resolve_vector_kernel_kind_from_env(env)?;
    Ok(VectorKernelMetadata::new(kernel))
}
