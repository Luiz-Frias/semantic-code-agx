//! ANE (Apple Neural Engine) embedding adapter.
//!
//! Offloads embedding inference to the dedicated ANE hardware on Apple Silicon,
//! targeting 10-50x throughput vs CPU-only ONNX for the linear projection kernels.
//!
//! # Requirements
//!
//! - macOS 15+ on Apple Silicon (M1/M2/M3/M4)
//! - `model.safetensors` in the model directory (run conversion script first)
//! - `ane` feature flag enabled
//!
//! # Known Limitations
//!
//! - Phase 1: only supports all-MiniLM-L6-v2 (BERT-style)
//! - FFN up bias not applied before GELU (see TODO in ane-inference-rs/model.rs)
//! - Sequential batch processing (no ANE pipelining yet)

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ane_inference::{AneBatchExecutionMode, AneInferenceConfig, AneModel, ModelConfig};
use semantic_code_ports::{
    DetectDimensionRequest, EmbedBatchRequest, EmbedRequest, EmbeddingPort, EmbeddingProviderId,
    EmbeddingProviderInfo, EmbeddingVector,
};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use tracing::Instrument;

/// Configuration for the ANE embedding adapter.
pub struct AneEmbeddingConfig {
    /// Directory containing model weights (`model.safetensors`) and tokenizer.
    pub model_dir: PathBuf,
    /// Override embedding dimension (defaults to model's native dim).
    pub dimension: Option<u32>,
    /// Fixed sequence length for compiled ANE kernels.
    pub sequence_length: u32,
    /// Maximum in-flight ANE evals per compiled kernel.
    pub max_in_flight_evals: NonZeroUsize,
    /// Max worker depth for `embed_batch` host-side overlap.
    pub batch_pipeline_depth: NonZeroUsize,
    /// Optional minimum batch size before conductor mode is used.
    pub conductor_min_batch: Option<NonZeroUsize>,
    /// Max number of CPU-ready chunks handed to one worker wakeup.
    pub conductor_cpu_pop_limit: NonZeroUsize,
    /// Batch execution strategy.
    pub execution_mode: AneExecutionMode,
}

/// Batch execution mode exposed by the adapter boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AneExecutionMode {
    /// Preserve strict compatibility with serial execution.
    SerialCompat,
    /// Overlap CPU stages while ANE kernels execute.
    CpuOverlap,
    /// Experimental multi-in-flight ANE queueing mode.
    AneQueueExperimental,
    /// Conductor/worker pipeline with ANE-CPU overlap.
    ConductorPipeline,
}

impl From<AneExecutionMode> for AneBatchExecutionMode {
    fn from(value: AneExecutionMode) -> Self {
        match value {
            AneExecutionMode::SerialCompat => Self::SerialCompat,
            AneExecutionMode::CpuOverlap => Self::CpuOverlap,
            AneExecutionMode::AneQueueExperimental => Self::AneQueueExperimental,
            AneExecutionMode::ConductorPipeline => Self::ConductorPipeline,
        }
    }
}

/// ANE embedding adapter — runs transformer inference on Apple Neural Engine hardware.
pub struct AneEmbedding {
    provider: EmbeddingProviderInfo,
    model: Arc<AneModel>,
    dimension: u32,
}

impl AneEmbedding {
    /// Create a new ANE embedding adapter, compiling all ANE kernels at startup.
    pub fn new(config: &AneEmbeddingConfig) -> Result<Self> {
        let model_config = resolve_model_config(&config.model_dir);
        let ane_config = AneInferenceConfig {
            model_config: model_config.clone(),
            model_dir: config.model_dir.clone(),
            sequence_length: config.sequence_length,
            max_in_flight_evals: config.max_in_flight_evals,
            batch_pipeline_depth: config.batch_pipeline_depth,
            conductor_min_batch: config.conductor_min_batch,
            conductor_cpu_pop_limit: config.conductor_cpu_pop_limit,
            execution_mode: config.execution_mode.into(),
        };

        let model = AneModel::load(&ane_config).map_err(|e| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("embedding", "ane_load_failed"),
                format!("ANE model load failed: {e}"),
                ErrorClass::NonRetriable,
            )
        })?;

        let dimension = config.dimension.unwrap_or(model_config.embedding_dim);

        Ok(Self {
            provider: EmbeddingProviderInfo {
                id: EmbeddingProviderId::parse("ane").map_err(ErrorEnvelope::from)?,
                name: "ANE (Apple Neural Engine)".into(),
            },
            model: Arc::new(model),
            dimension,
        })
    }
}

fn resolve_model_config(model_dir: &Path) -> ModelConfig {
    let default = ModelConfig::minilm_l6_v2();
    let config_path = model_dir.join("config.json");
    let raw = match std::fs::read_to_string(&config_path) {
        Ok(raw) => raw,
        Err(error) => {
            tracing::debug!(
                path = %config_path.display(),
                %error,
                "ANE config.json missing/unreadable, defaulting to MiniLM preset"
            );
            return default;
        },
    };
    let parsed: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(parsed) => parsed,
        Err(error) => {
            tracing::warn!(
                path = %config_path.display(),
                %error,
                "ANE config.json parse failed, defaulting to MiniLM preset"
            );
            return default;
        },
    };

    if is_qwen3_family_config(&parsed) {
        tracing::info!(
            path = %config_path.display(),
            "Detected Qwen3/pplx model config for ANE embeddings"
        );
        return ModelConfig::pplx_embed_0_6b();
    }

    default
}

fn is_qwen3_family_config(parsed: &serde_json::Value) -> bool {
    let has_qwen_model_type = parsed
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|value| {
            let normalized = value.to_ascii_lowercase();
            normalized.contains("qwen") || normalized.contains("pplx")
        });
    if has_qwen_model_type {
        return true;
    }

    parsed
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|architectures| {
            architectures
                .iter()
                .filter_map(serde_json::Value::as_str)
                .any(|value| {
                    let normalized = value.to_ascii_lowercase();
                    normalized.contains("qwen") || normalized.contains("pplx")
                })
        })
}

impl EmbeddingPort for AneEmbedding {
    fn provider(&self) -> &EmbeddingProviderInfo {
        &self.provider
    }

    fn detect_dimension(
        &self,
        ctx: &RequestContext,
        _request: DetectDimensionRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<u32>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            ctx.ensure_not_cancelled("ane_embedding.detect_dimension")?;
            Ok(self.dimension)
        })
    }

    fn embed(
        &self,
        ctx: &RequestContext,
        request: EmbedRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<EmbeddingVector>> {
        let ctx = ctx.clone();
        let text = request.text;
        let model = Arc::clone(&self.model);
        let span = tracing::info_span!("adapter.embedding.ane.embed", text_len = text.len());

        Box::pin(
            async move {
                ctx.ensure_not_cancelled("ane_embedding.embed")?;
                let vector = tokio::task::spawn_blocking(move || model.embed(&text))
                    .await
                    .map_err(|e| {
                        ErrorEnvelope::unexpected(
                            ErrorCode::internal(),
                            format!("ANE embed task panicked: {e}"),
                            ErrorClass::NonRetriable,
                        )
                    })?
                    .map_err(|e| {
                        ErrorEnvelope::unexpected(
                            ErrorCode::new("embedding", "ane_eval_failed"),
                            format!("ANE embed failed: {e}"),
                            ErrorClass::NonRetriable,
                        )
                    })?;
                Ok(EmbeddingVector::from_vec(vector))
            }
            .instrument(span),
        )
    }

    fn embed_batch(
        &self,
        ctx: &RequestContext,
        request: EmbedBatchRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<EmbeddingVector>>> {
        let ctx = ctx.clone();
        let texts = request.texts;
        let model = Arc::clone(&self.model);
        let span = tracing::info_span!(
            "adapter.embedding.ane.embed_batch",
            batch_size = texts.len()
        );

        Box::pin(
            async move {
                ctx.ensure_not_cancelled("ane_embedding.embed_batch")?;
                let vectors = tokio::task::spawn_blocking(move || {
                    let text_refs: Vec<&str> = texts.iter().map(AsRef::as_ref).collect();
                    model.embed_batch(&text_refs)
                })
                .await
                .map_err(|e| {
                    ErrorEnvelope::unexpected(
                        ErrorCode::internal(),
                        format!("ANE batch task panicked: {e}"),
                        ErrorClass::NonRetriable,
                    )
                })?
                .map_err(|e| {
                    ErrorEnvelope::unexpected(
                        ErrorCode::new("embedding", "ane_eval_failed"),
                        format!("ANE batch embed failed: {e}"),
                        ErrorClass::NonRetriable,
                    )
                })?;
                Ok(vectors.into_iter().map(EmbeddingVector::from_vec).collect())
            }
            .instrument(span),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{is_qwen3_family_config, resolve_model_config};
    use ane_inference::ArchitectureFamily;

    #[test]
    fn detects_qwen3_from_model_type() {
        let parsed = serde_json::json!({
            "model_type": "bidirectional_pplx_qwen3"
        });
        assert!(is_qwen3_family_config(&parsed));
    }

    #[test]
    fn detects_qwen3_from_architectures() {
        let parsed = serde_json::json!({
            "architectures": ["PPLXQwen3Model"]
        });
        assert!(is_qwen3_family_config(&parsed));
    }

    #[test]
    fn resolve_model_config_defaults_to_minilm_without_config_file() {
        let temp = std::env::temp_dir().join(format!(
            "sca-ane-model-detect-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let create_result = std::fs::create_dir_all(&temp);
        assert!(create_result.is_ok());
        let resolved = resolve_model_config(&temp);
        assert_eq!(resolved.architecture, ArchitectureFamily::Bert);
        assert_eq!(resolved.hidden_size, 384);
    }
}
