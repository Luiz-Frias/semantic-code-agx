//! Ollama embedding adapter.

use reqwest::StatusCode;
use semantic_code_config::EmbeddingConfig;
use semantic_code_ports::{
    DetectDimensionRequest, EmbedBatchRequest, EmbedRequest, EmbeddingPort, EmbeddingProviderId,
    EmbeddingProviderInfo, EmbeddingVector,
};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "http://localhost:11434";
const DEFAULT_MODEL: &str = "embeddinggemma";
const DEFAULT_TEST_TEXT: &str = "dimension probe";
const EMBED_PATH: &str = "/api/embed";

/// Ollama embedding adapter configuration.
#[derive(Debug, Clone)]
pub struct OllamaEmbeddingConfig {
    /// Embedding model name (defaults to `embeddinggemma`).
    pub model: Option<Box<str>>,
    /// Base URL override (defaults to `http://localhost:11434`).
    pub base_url: Option<Box<str>>,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Optional output dimension override.
    pub dimension: Option<u32>,
}

impl OllamaEmbeddingConfig {
    /// Build from the shared embedding config.
    #[must_use]
    pub fn from_embedding_config(config: &EmbeddingConfig) -> Self {
        Self {
            model: config.model.clone(),
            base_url: config.base_url.clone(),
            timeout_ms: config.timeout_ms,
            dimension: config.dimension,
        }
    }
}

/// Ollama embedding adapter implementation.
pub struct OllamaEmbedding {
    provider: EmbeddingProviderInfo,
    client: reqwest::Client,
    endpoint: Box<str>,
    model: Box<str>,
    dimension_override: Option<u32>,
}

impl OllamaEmbedding {
    /// Create a new Ollama embedding adapter.
    pub fn new(config: &OllamaEmbeddingConfig) -> Result<Self> {
        let model = normalize_optional_required("model", config.model.as_deref())?
            .unwrap_or_else(|| DEFAULT_MODEL.to_owned().into_boxed_str());
        let base_url = normalize_optional_required("base url", config.base_url.as_deref())?
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned().into_boxed_str());
        let base_url = base_url.trim_end_matches('/').to_owned().into_boxed_str();
        if base_url.is_empty() {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "base url must be non-empty",
            ));
        }
        if config.timeout_ms == 0 {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "timeout must be greater than zero",
            ));
        }

        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .build()
            .map_err(|error| {
                ErrorEnvelope::unexpected(
                    ErrorCode::new("embedding", "ollama_client_init_failed"),
                    format!("failed to build Ollama client: {error}"),
                    ErrorClass::NonRetriable,
                )
            })?;

        let provider = EmbeddingProviderInfo {
            id: EmbeddingProviderId::parse("ollama").map_err(ErrorEnvelope::from)?,
            name: "Ollama".into(),
        };
        let endpoint = format!("{base_url}{EMBED_PATH}").into_boxed_str();

        Ok(Self {
            provider,
            client,
            endpoint,
            model,
            dimension_override: config.dimension,
        })
    }

    async fn embed_many(
        &self,
        ctx: &RequestContext,
        texts: Vec<Box<str>>,
        operation: &'static str,
    ) -> Result<Vec<EmbeddingVector>> {
        ctx.ensure_not_cancelled(operation)?;
        if texts.is_empty() {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "embedding input must be non-empty",
            ));
        }
        let texts = sanitize_texts(texts);
        let expected_count = texts.len();

        let request = OllamaEmbeddingRequest {
            model: self.model.clone(),
            input: if expected_count == 1 {
                let text = texts.into_iter().next().ok_or_else(|| {
                    ErrorEnvelope::unexpected(
                        ErrorCode::internal(),
                        "missing embedding input",
                        ErrorClass::NonRetriable,
                    )
                })?;
                OllamaInput::Single(text)
            } else {
                OllamaInput::Many(texts)
            },
            dimensions: self.dimension_override,
            truncate: None,
        };

        let response = self.send_request(ctx, request, operation).await?;
        map_ollama_embeddings(response, expected_count, self.dimension_override)
    }

    async fn send_request(
        &self,
        ctx: &RequestContext,
        request: OllamaEmbeddingRequest,
        operation: &'static str,
    ) -> Result<OllamaEmbeddingResponse> {
        ctx.ensure_not_cancelled(operation)?;

        let response = tokio::select! {
            () = ctx.cancelled() => return Err(cancelled_error(operation)),
            result = self.client.post(self.endpoint.as_ref()).json(&request).send() => {
                result.map_err(|error| map_reqwest_error(&error))?
            }
        };

        let status = response.status();
        let payload = tokio::select! {
            () = ctx.cancelled() => return Err(cancelled_error(operation)),
            result = response.bytes() => result.map_err(|error| map_reqwest_error(&error))?,
        };

        if !status.is_success() {
            return Err(map_ollama_http_error(status, &payload));
        }

        serde_json::from_slice(&payload).map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("embedding", "ollama_invalid_response"),
                format!("failed to decode Ollama response: {error}"),
                ErrorClass::NonRetriable,
            )
        })
    }
}

impl EmbeddingPort for OllamaEmbedding {
    fn provider(&self) -> &EmbeddingProviderInfo {
        &self.provider
    }

    fn detect_dimension(
        &self,
        ctx: &RequestContext,
        request: DetectDimensionRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<u32>> {
        let ctx = ctx.clone();
        let test_text = request
            .options
            .test_text
            .unwrap_or_else(|| DEFAULT_TEST_TEXT.to_owned().into_boxed_str());
        let dimension_override = self.dimension_override;
        Box::pin(async move {
            ctx.ensure_not_cancelled("ollama_embedding.detect_dimension")?;
            if let Some(dimension) = dimension_override {
                return Ok(dimension);
            }
            let vectors = self
                .embed_many(&ctx, vec![test_text], "ollama_embedding.detect_dimension")
                .await?;
            let vector = vectors.into_iter().next().ok_or_else(|| {
                ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    "missing embedding response",
                    ErrorClass::NonRetriable,
                )
            })?;
            Ok(vector.dimension())
        })
    }

    fn embed(
        &self,
        ctx: &RequestContext,
        request: EmbedRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<EmbeddingVector>> {
        let ctx = ctx.clone();
        let text = request.text;
        Box::pin(async move {
            let mut vectors = self
                .embed_many(&ctx, vec![text], "ollama_embedding.embed")
                .await?;
            vectors.pop().ok_or_else(|| {
                ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    "missing embedding response",
                    ErrorClass::NonRetriable,
                )
            })
        })
    }

    fn embed_batch(
        &self,
        ctx: &RequestContext,
        request: EmbedBatchRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<EmbeddingVector>>> {
        let ctx = ctx.clone();
        let texts = request.texts;
        Box::pin(async move {
            self.embed_many(&ctx, texts, "ollama_embedding.embed_batch")
                .await
        })
    }
}

#[derive(Debug, Serialize)]
struct OllamaEmbeddingRequest {
    model: Box<str>,
    input: OllamaInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncate: Option<bool>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OllamaInput {
    Single(Box<str>),
    Many(Vec<Box<str>>),
}

#[derive(Debug, Deserialize)]
struct OllamaEmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct OllamaErrorResponse {
    error: Option<String>,
}

fn normalize_optional_required(label: &str, value: Option<&str>) -> Result<Option<Box<str>>> {
    let trimmed = match value {
        Some(value) => value.trim(),
        None => return Ok(None),
    };
    if trimmed.is_empty() {
        return Err(ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            format!("{label} must be non-empty"),
        ));
    }
    Ok(Some(trimmed.to_owned().into_boxed_str()))
}

fn sanitize_text(text: Box<str>) -> Box<str> {
    if text.is_empty() { " ".into() } else { text }
}

fn sanitize_texts(texts: Vec<Box<str>>) -> Vec<Box<str>> {
    texts.into_iter().map(sanitize_text).collect()
}

fn cancelled_error(operation: &'static str) -> ErrorEnvelope {
    ErrorEnvelope::cancelled("operation cancelled").with_metadata("operation", operation)
}

fn map_reqwest_error(error: &reqwest::Error) -> ErrorEnvelope {
    if error.is_timeout() {
        return ErrorEnvelope::unexpected(
            ErrorCode::timeout(),
            "Ollama request timed out",
            ErrorClass::Retriable,
        );
    }
    if error.is_connect() {
        return ErrorEnvelope::unexpected(
            ErrorCode::io(),
            format!("Ollama connection failed: {error}"),
            ErrorClass::Retriable,
        );
    }
    ErrorEnvelope::unexpected(
        ErrorCode::new("embedding", "ollama_request_failed"),
        format!("Ollama request failed: {error}"),
        ErrorClass::NonRetriable,
    )
}

fn map_ollama_http_error(status: StatusCode, payload: &[u8]) -> ErrorEnvelope {
    let message = serde_json::from_slice::<OllamaErrorResponse>(payload)
        .ok()
        .and_then(|response| response.error)
        .unwrap_or_else(|| "Ollama request failed".to_string());

    let mut envelope = match status.as_u16() {
        400 | 404 | 422 => ErrorEnvelope::expected(ErrorCode::invalid_input(), message),
        401 | 403 => ErrorEnvelope::expected(ErrorCode::permission_denied(), message),
        408 => ErrorEnvelope::unexpected(ErrorCode::timeout(), message, ErrorClass::Retriable),
        429 => ErrorEnvelope::unexpected(
            ErrorCode::new("core", "rate_limited"),
            message,
            ErrorClass::Retriable,
        ),
        _ if status.is_server_error() => ErrorEnvelope::unexpected(
            ErrorCode::new("core", "dependency_unavailable"),
            message,
            ErrorClass::Retriable,
        ),
        _ => ErrorEnvelope::unexpected(
            ErrorCode::new("embedding", "ollama_http_error"),
            message,
            ErrorClass::NonRetriable,
        ),
    };

    envelope = envelope.with_metadata("status", status.as_u16().to_string());
    envelope
}

fn map_ollama_embeddings(
    response: OllamaEmbeddingResponse,
    expected_count: usize,
    expected_dimension: Option<u32>,
) -> Result<Vec<EmbeddingVector>> {
    if response.embeddings.len() != expected_count {
        return Err(ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            format!(
                "embedding response count mismatch (expected {expected_count}, got {})",
                response.embeddings.len()
            ),
            ErrorClass::NonRetriable,
        ));
    }

    response
        .embeddings
        .into_iter()
        .map(|values| map_embedding_values(values, expected_dimension))
        .collect()
}

fn map_embedding_values(
    values: Vec<f32>,
    expected_dimension: Option<u32>,
) -> Result<EmbeddingVector> {
    let dimension = u32::try_from(values.len()).map_err(|_| {
        ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            "embedding dimension overflow",
            ErrorClass::NonRetriable,
        )
    })?;
    if let Some(expected) = expected_dimension
        && dimension != expected
    {
        return Err(ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            "embedding dimension mismatch",
        )
        .with_metadata("expected", expected.to_string())
        .with_metadata("actual", dimension.to_string()));
    }
    let _ = dimension;
    Ok(EmbeddingVector::new(std::sync::Arc::from(values)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_ports::DetectDimensionOptions;
    use semantic_code_shared::RequestContext;
    use serde_json::json;

    #[test]
    fn ollama_request_serializes_single_input() {
        let request = OllamaEmbeddingRequest {
            model: "embeddinggemma".into(),
            input: OllamaInput::Single("hello".into()),
            dimensions: Some(2),
            truncate: None,
        };
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(
            value,
            json!({
                "model": "embeddinggemma",
                "input": "hello",
                "dimensions": 2
            })
        );
    }

    #[test]
    fn map_ollama_embeddings_rejects_dimension_mismatch() {
        let response = OllamaEmbeddingResponse {
            embeddings: vec![vec![0.1, 0.2]],
        };
        let error = map_ollama_embeddings(response, 1, Some(3)).unwrap_err();
        assert_eq!(error.code, ErrorCode::invalid_input());
    }

    #[test]
    fn map_ollama_http_error_rate_limited_is_retriable() {
        let payload = serde_json::to_vec(&json!({
            "error": "rate limited"
        }))
        .unwrap();
        let envelope = map_ollama_http_error(StatusCode::TOO_MANY_REQUESTS, &payload);
        assert_eq!(envelope.class, ErrorClass::Retriable);
        assert_eq!(envelope.code, ErrorCode::new("core", "rate_limited"));
    }

    #[tokio::test]
    async fn detect_dimension_uses_override() -> Result<()> {
        let config = OllamaEmbeddingConfig {
            model: None,
            base_url: Some("http://localhost".into()),
            timeout_ms: 1_000,
            dimension: Some(16),
        };
        let adapter = OllamaEmbedding::new(&config)?;
        let ctx = RequestContext::new_request();
        let dimension = adapter
            .detect_dimension(&ctx, DetectDimensionOptions::default().into())
            .await?;
        assert_eq!(dimension, 16);
        Ok(())
    }
}
