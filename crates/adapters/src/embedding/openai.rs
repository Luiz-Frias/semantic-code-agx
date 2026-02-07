//! OpenAI embedding adapter.

use reqwest::StatusCode;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use semantic_code_config::EmbeddingConfig;
use semantic_code_ports::{
    DetectDimensionRequest, EmbedBatchRequest, EmbedRequest, EmbeddingPort, EmbeddingProviderId,
    EmbeddingProviderInfo, EmbeddingVector,
};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "text-embedding-3-small";
const DEFAULT_TEST_TEXT: &str = "dimension probe";

/// OpenAI embedding adapter configuration.
#[derive(Debug, Clone)]
pub struct OpenAiEmbeddingConfig {
    /// API key used for authentication.
    pub api_key: Box<str>,
    /// Embedding model name (defaults to `text-embedding-3-small`).
    pub model: Option<Box<str>>,
    /// Base URL override (defaults to `https://api.openai.com/v1`).
    pub base_url: Option<Box<str>>,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Optional output dimension override.
    pub dimension: Option<u32>,
}

impl OpenAiEmbeddingConfig {
    /// Build from the shared embedding config plus an API key.
    #[must_use]
    pub fn from_embedding_config(api_key: Box<str>, config: &EmbeddingConfig) -> Self {
        Self {
            api_key,
            model: config.model.clone(),
            base_url: config.base_url.clone(),
            timeout_ms: config.timeout_ms,
            dimension: config.dimension,
        }
    }
}

/// OpenAI embedding adapter implementation.
pub struct OpenAiEmbedding {
    provider: EmbeddingProviderInfo,
    client: reqwest::Client,
    endpoint: Box<str>,
    model: Box<str>,
    dimension_override: Option<u32>,
}

impl OpenAiEmbedding {
    /// Create a new OpenAI embedding adapter.
    pub fn new(config: &OpenAiEmbeddingConfig) -> Result<Self> {
        let api_key = normalize_required("api key", config.api_key.as_ref())?;
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

        let mut headers = HeaderMap::new();
        let mut auth_header =
            HeaderValue::from_str(&format!("Bearer {api_key}")).map_err(|_| {
                ErrorEnvelope::expected(
                    ErrorCode::invalid_input(),
                    "api key contains invalid header characters",
                )
            })?;
        auth_header.set_sensitive(true);
        headers.insert(AUTHORIZATION, auth_header);

        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .default_headers(headers)
            .build()
            .map_err(|error| {
                ErrorEnvelope::unexpected(
                    ErrorCode::new("embedding", "openai_client_init_failed"),
                    format!("failed to build OpenAI client: {error}"),
                    ErrorClass::NonRetriable,
                )
            })?;

        let provider = EmbeddingProviderInfo {
            id: EmbeddingProviderId::parse("openai").map_err(ErrorEnvelope::from)?,
            name: "OpenAI".into(),
        };
        let endpoint = format!("{base_url}/embeddings").into_boxed_str();

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
        inputs: OpenAiInput,
        expected_count: usize,
        operation: &'static str,
    ) -> Result<Vec<EmbeddingVector>> {
        ctx.ensure_not_cancelled(operation)?;
        if expected_count == 0 {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "embedding input must be non-empty",
            ));
        }

        let request = OpenAiEmbeddingRequest {
            model: self.model.clone(),
            input: sanitize_input(inputs),
            dimensions: self.dimension_override,
        };
        let response = self.send_request(ctx, request, operation).await?;
        map_embeddings(response, expected_count, self.dimension_override)
    }

    async fn send_request(
        &self,
        ctx: &RequestContext,
        request: OpenAiEmbeddingRequest,
        operation: &'static str,
    ) -> Result<OpenAiEmbeddingResponse> {
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
            return Err(map_openai_http_error(status, &payload));
        }

        serde_json::from_slice(&payload).map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("embedding", "openai_invalid_response"),
                format!("failed to decode OpenAI response: {error}"),
                ErrorClass::NonRetriable,
            )
        })
    }
}

impl EmbeddingPort for OpenAiEmbedding {
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
            ctx.ensure_not_cancelled("openai_embedding.detect_dimension")?;
            if let Some(dimension) = dimension_override {
                return Ok(dimension);
            }
            let vectors = self
                .embed_many(
                    &ctx,
                    OpenAiInput::Single(test_text),
                    1,
                    "openai_embedding.detect_dimension",
                )
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
                .embed_many(&ctx, OpenAiInput::Single(text), 1, "openai_embedding.embed")
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
        let expected_count = texts.len();
        Box::pin(async move {
            self.embed_many(
                &ctx,
                OpenAiInput::Many(texts),
                expected_count,
                "openai_embedding.embed_batch",
            )
            .await
        })
    }
}

#[derive(Debug, Serialize)]
struct OpenAiEmbeddingRequest {
    model: Box<str>,
    input: OpenAiInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenAiInput {
    Single(Box<str>),
    Many(Vec<Box<str>>),
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingDatum>,
}

#[derive(Debug, Deserialize)]
struct OpenAiEmbeddingDatum {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct OpenAiErrorResponse {
    error: OpenAiErrorDetail,
}

#[derive(Debug, Deserialize)]
struct OpenAiErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: Option<String>,
    code: Option<String>,
}

fn normalize_required(label: &str, value: &str) -> Result<Box<str>> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            format!("{label} must be set"),
        ));
    }
    Ok(trimmed.to_owned().into_boxed_str())
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

fn sanitize_input(input: OpenAiInput) -> OpenAiInput {
    match input {
        OpenAiInput::Single(text) => OpenAiInput::Single(sanitize_text(text)),
        OpenAiInput::Many(texts) => {
            OpenAiInput::Many(texts.into_iter().map(sanitize_text).collect())
        },
    }
}

fn cancelled_error(operation: &'static str) -> ErrorEnvelope {
    ErrorEnvelope::cancelled("operation cancelled").with_metadata("operation", operation)
}

fn map_reqwest_error(error: &reqwest::Error) -> ErrorEnvelope {
    if error.is_timeout() {
        return ErrorEnvelope::unexpected(
            ErrorCode::timeout(),
            "OpenAI request timed out",
            ErrorClass::Retriable,
        );
    }
    if error.is_connect() {
        return ErrorEnvelope::unexpected(
            ErrorCode::io(),
            format!("OpenAI connection failed: {error}"),
            ErrorClass::Retriable,
        );
    }
    ErrorEnvelope::unexpected(
        ErrorCode::new("embedding", "openai_request_failed"),
        format!("OpenAI request failed: {error}"),
        ErrorClass::NonRetriable,
    )
}

fn map_openai_http_error(status: StatusCode, payload: &[u8]) -> ErrorEnvelope {
    let mut envelope = if let Ok(parsed) = serde_json::from_slice::<OpenAiErrorResponse>(payload) {
        let message = parsed.error.message;
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
                ErrorCode::new("embedding", "openai_http_error"),
                message,
                ErrorClass::NonRetriable,
            ),
        };

        if let Some(error_type) = parsed.error.error_type.as_deref() {
            envelope = envelope.with_metadata("error_type", error_type.to_string());
        }
        if let Some(error_code) = parsed.error.code.as_deref() {
            envelope = envelope.with_metadata("error_code", error_code.to_string());
        }
        envelope
    } else {
        ErrorEnvelope::unexpected(
            ErrorCode::new("embedding", "openai_http_error"),
            "OpenAI request failed with non-JSON error",
            if status.is_server_error() {
                ErrorClass::Retriable
            } else {
                ErrorClass::NonRetriable
            },
        )
    };

    envelope = envelope.with_metadata("status", status.as_u16().to_string());
    envelope
}

fn map_embeddings(
    response: OpenAiEmbeddingResponse,
    expected_count: usize,
    expected_dimension: Option<u32>,
) -> Result<Vec<EmbeddingVector>> {
    if response.data.len() != expected_count {
        return Err(ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            format!(
                "embedding response count mismatch (expected {expected_count}, got {})",
                response.data.len()
            ),
            ErrorClass::NonRetriable,
        ));
    }

    let mut slots: Vec<Option<EmbeddingVector>> = vec![None; expected_count];
    for datum in response.data {
        let dimension = u32::try_from(datum.embedding.len()).map_err(|_| {
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
        let slot = slots.get_mut(datum.index).ok_or_else(|| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "embedding response index out of range",
                ErrorClass::NonRetriable,
            )
        })?;
        if slot.is_some() {
            return Err(ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "embedding response index duplicated",
                ErrorClass::NonRetriable,
            ));
        }
        let _ = dimension;
        *slot = Some(EmbeddingVector::new(std::sync::Arc::from(datum.embedding)));
    }

    slots
        .into_iter()
        .map(|slot| {
            slot.ok_or_else(|| {
                ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    "embedding response missing index",
                    ErrorClass::NonRetriable,
                )
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_ports::DetectDimensionOptions;
    use semantic_code_shared::RequestContext;
    use serde_json::json;

    #[test]
    fn openai_request_serializes_single_input() {
        let request = OpenAiEmbeddingRequest {
            model: "text-embedding-3-small".into(),
            input: OpenAiInput::Single("hello".into()),
            dimensions: None,
        };
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(
            value,
            json!({
                "model": "text-embedding-3-small",
                "input": "hello"
            })
        );
    }

    #[test]
    fn openai_request_serializes_batch_with_dimensions() {
        let request = OpenAiEmbeddingRequest {
            model: "text-embedding-3-small".into(),
            input: OpenAiInput::Many(vec!["a".into(), "b".into()]),
            dimensions: Some(2),
        };
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(
            value,
            json!({
                "model": "text-embedding-3-small",
                "input": ["a", "b"],
                "dimensions": 2
            })
        );
    }

    #[test]
    fn map_embeddings_rejects_dimension_mismatch() {
        let response = OpenAiEmbeddingResponse {
            data: vec![OpenAiEmbeddingDatum {
                embedding: vec![0.1, 0.2],
                index: 0,
            }],
        };
        let error = map_embeddings(response, 1, Some(3)).unwrap_err();
        assert_eq!(error.code, ErrorCode::invalid_input());
    }

    #[test]
    fn map_openai_http_error_rate_limited_is_retriable() {
        let payload = serde_json::to_vec(&json!({
            "error": {
                "message": "rate limited"
            }
        }))
        .unwrap();
        let envelope = map_openai_http_error(StatusCode::TOO_MANY_REQUESTS, &payload);
        assert_eq!(envelope.class, ErrorClass::Retriable);
        assert_eq!(envelope.code, ErrorCode::new("core", "rate_limited"));
    }

    #[tokio::test]
    async fn detect_dimension_uses_override() -> Result<()> {
        let config = OpenAiEmbeddingConfig {
            api_key: "example".into(), // pragma: allowlist secret
            model: None,
            base_url: Some("http://localhost".into()),
            timeout_ms: 1_000,
            dimension: Some(4),
        };
        let adapter = OpenAiEmbedding::new(&config)?;
        let ctx = RequestContext::new_request();
        let dimension = adapter
            .detect_dimension(&ctx, DetectDimensionOptions::default().into())
            .await?;
        assert_eq!(dimension, 4);
        Ok(())
    }
}
