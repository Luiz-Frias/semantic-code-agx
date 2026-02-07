//! Gemini embedding adapter.

use reqwest::StatusCode;
use reqwest::header::{HeaderMap, HeaderValue};
use semantic_code_config::EmbeddingConfig;
use semantic_code_ports::{
    DetectDimensionRequest, EmbedBatchRequest, EmbedRequest, EmbeddingPort, EmbeddingProviderId,
    EmbeddingProviderInfo, EmbeddingVector,
};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_MODEL: &str = "gemini-embedding-001";
const DEFAULT_TEST_TEXT: &str = "dimension probe";
const HEADER_API_KEY: &str = "x-goog-api-key";

/// Gemini embedding adapter configuration.
#[derive(Debug, Clone)]
pub struct GeminiEmbeddingConfig {
    /// API key used for authentication.
    pub api_key: Box<str>,
    /// Embedding model name (defaults to `gemini-embedding-001`).
    pub model: Option<Box<str>>,
    /// Base URL override (defaults to `https://generativelanguage.googleapis.com/v1beta`).
    pub base_url: Option<Box<str>>,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Optional output dimension override.
    pub dimension: Option<u32>,
}

impl GeminiEmbeddingConfig {
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

/// Gemini embedding adapter implementation.
pub struct GeminiEmbedding {
    provider: EmbeddingProviderInfo,
    client: reqwest::Client,
    embed_endpoint: Box<str>,
    batch_endpoint: Box<str>,
    model_resource: Box<str>,
    dimension_override: Option<u32>,
}

impl GeminiEmbedding {
    /// Create a new Gemini embedding adapter.
    pub fn new(config: &GeminiEmbeddingConfig) -> Result<Self> {
        let api_key = normalize_required("api key", config.api_key.as_ref())?;
        let model = normalize_optional_required("model", config.model.as_deref())?
            .unwrap_or_else(|| DEFAULT_MODEL.to_owned().into_boxed_str());
        let model_resource = normalize_model_resource(model.as_ref())?;
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
        let mut api_header = HeaderValue::from_str(api_key.as_ref()).map_err(|_| {
            ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "api key contains invalid header characters",
            )
        })?;
        api_header.set_sensitive(true);
        headers.insert(HEADER_API_KEY, api_header);

        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .default_headers(headers)
            .build()
            .map_err(|error| {
                ErrorEnvelope::unexpected(
                    ErrorCode::new("embedding", "gemini_client_init_failed"),
                    format!("failed to build Gemini client: {error}"),
                    ErrorClass::NonRetriable,
                )
            })?;

        let provider = EmbeddingProviderInfo {
            id: EmbeddingProviderId::parse("gemini").map_err(ErrorEnvelope::from)?,
            name: "Gemini".into(),
        };
        let embed_endpoint = format!("{base_url}/{model_resource}:embedContent").into_boxed_str();
        let batch_endpoint =
            format!("{base_url}/{model_resource}:batchEmbedContents").into_boxed_str();

        Ok(Self {
            provider,
            client,
            embed_endpoint,
            batch_endpoint,
            model_resource,
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
        if texts.len() == 1 {
            let text = texts.into_iter().next().ok_or_else(|| {
                ErrorEnvelope::unexpected(
                    ErrorCode::internal(),
                    "missing embedding input",
                    ErrorClass::NonRetriable,
                )
            })?;
            let response = self.send_single(ctx, text, operation).await?;
            return map_gemini_embeddings(response, 1, self.dimension_override);
        }

        let expected_count = texts.len();
        let response = self.send_batch(ctx, texts, operation).await?;
        map_gemini_embeddings(response, expected_count, self.dimension_override)
    }

    async fn send_single(
        &self,
        ctx: &RequestContext,
        text: Box<str>,
        operation: &'static str,
    ) -> Result<GeminiEmbeddingResponse> {
        let request = GeminiEmbedContentRequest {
            model: self.model_resource.clone(),
            content: GeminiContent::from_text(text),
            output_dimensionality: self.dimension_override,
        };
        self.send_request(ctx, &self.embed_endpoint, request, operation)
            .await
    }

    async fn send_batch(
        &self,
        ctx: &RequestContext,
        texts: Vec<Box<str>>,
        operation: &'static str,
    ) -> Result<GeminiEmbeddingResponse> {
        let requests = texts
            .into_iter()
            .map(|text| GeminiBatchRequestItem {
                model: self.model_resource.clone(),
                content: GeminiContent::from_text(text),
                output_dimensionality: self.dimension_override,
            })
            .collect();
        let request = GeminiBatchEmbedRequest { requests };
        self.send_request(ctx, &self.batch_endpoint, request, operation)
            .await
    }

    async fn send_request<T: Serialize>(
        &self,
        ctx: &RequestContext,
        endpoint: &str,
        request: T,
        operation: &'static str,
    ) -> Result<GeminiEmbeddingResponse> {
        ctx.ensure_not_cancelled(operation)?;

        let response = tokio::select! {
            () = ctx.cancelled() => return Err(cancelled_error(operation)),
            result = self.client.post(endpoint).json(&request).send() => {
                result.map_err(|error| map_reqwest_error(&error))?
            }
        };

        let status = response.status();
        let payload = tokio::select! {
            () = ctx.cancelled() => return Err(cancelled_error(operation)),
            result = response.bytes() => result.map_err(|error| map_reqwest_error(&error))?,
        };

        if !status.is_success() {
            return Err(map_gemini_http_error(status, &payload));
        }

        serde_json::from_slice(&payload).map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("embedding", "gemini_invalid_response"),
                format!("failed to decode Gemini response: {error}"),
                ErrorClass::NonRetriable,
            )
        })
    }
}

impl EmbeddingPort for GeminiEmbedding {
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
            ctx.ensure_not_cancelled("gemini_embedding.detect_dimension")?;
            if let Some(dimension) = dimension_override {
                return Ok(dimension);
            }
            let vectors = self
                .embed_many(&ctx, vec![test_text], "gemini_embedding.detect_dimension")
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
                .embed_many(&ctx, vec![text], "gemini_embedding.embed")
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
            self.embed_many(&ctx, texts, "gemini_embedding.embed_batch")
                .await
        })
    }
}

#[derive(Debug, Serialize)]
struct GeminiEmbedContentRequest {
    model: Box<str>,
    content: GeminiContent,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "output_dimensionality"
    )]
    output_dimensionality: Option<u32>,
}

#[derive(Debug, Serialize)]
struct GeminiBatchEmbedRequest {
    requests: Vec<GeminiBatchRequestItem>,
}

#[derive(Debug, Serialize)]
struct GeminiBatchRequestItem {
    model: Box<str>,
    content: GeminiContent,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "output_dimensionality"
    )]
    output_dimensionality: Option<u32>,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

impl GeminiContent {
    fn from_text(text: Box<str>) -> Self {
        Self {
            parts: vec![GeminiPart { text }],
        }
    }
}

#[derive(Debug, Serialize)]
struct GeminiPart {
    text: Box<str>,
}

#[derive(Debug, Deserialize)]
struct GeminiEmbeddingResponse {
    #[serde(default)]
    embedding: Option<GeminiEmbeddingValues>,
    #[serde(default)]
    embeddings: Option<Vec<GeminiEmbeddingValues>>,
}

#[derive(Debug, Deserialize)]
struct GeminiEmbeddingValues {
    values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct GeminiErrorResponse {
    error: GeminiErrorDetail,
}

#[derive(Debug, Deserialize)]
struct GeminiErrorDetail {
    message: String,
    status: Option<String>,
    code: Option<u16>,
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

fn sanitize_texts(texts: Vec<Box<str>>) -> Vec<Box<str>> {
    texts.into_iter().map(sanitize_text).collect()
}

fn normalize_model_resource(model: &str) -> Result<Box<str>> {
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return Err(ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            "model must be non-empty",
        ));
    }
    if trimmed.starts_with("models/") {
        Ok(trimmed.to_owned().into_boxed_str())
    } else {
        Ok(format!("models/{trimmed}").into_boxed_str())
    }
}

fn cancelled_error(operation: &'static str) -> ErrorEnvelope {
    ErrorEnvelope::cancelled("operation cancelled").with_metadata("operation", operation)
}

fn map_reqwest_error(error: &reqwest::Error) -> ErrorEnvelope {
    if error.is_timeout() {
        return ErrorEnvelope::unexpected(
            ErrorCode::timeout(),
            "Gemini request timed out",
            ErrorClass::Retriable,
        );
    }
    if error.is_connect() {
        return ErrorEnvelope::unexpected(
            ErrorCode::io(),
            format!("Gemini connection failed: {error}"),
            ErrorClass::Retriable,
        );
    }
    ErrorEnvelope::unexpected(
        ErrorCode::new("embedding", "gemini_request_failed"),
        format!("Gemini request failed: {error}"),
        ErrorClass::NonRetriable,
    )
}

fn map_gemini_http_error(status: StatusCode, payload: &[u8]) -> ErrorEnvelope {
    let mut envelope = if let Ok(parsed) = serde_json::from_slice::<GeminiErrorResponse>(payload) {
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
                ErrorCode::new("embedding", "gemini_http_error"),
                message,
                ErrorClass::NonRetriable,
            ),
        };

        if let Some(status) = parsed.error.status.as_deref() {
            envelope = envelope.with_metadata("error_status", status.to_string());
        }
        if let Some(code) = parsed.error.code {
            envelope = envelope.with_metadata("error_code", code.to_string());
        }
        envelope
    } else {
        ErrorEnvelope::unexpected(
            ErrorCode::new("embedding", "gemini_http_error"),
            "Gemini request failed with non-JSON error",
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

fn map_gemini_embeddings(
    response: GeminiEmbeddingResponse,
    expected_count: usize,
    expected_dimension: Option<u32>,
) -> Result<Vec<EmbeddingVector>> {
    let embeddings = if let Some(embeddings) = response.embeddings {
        embeddings
    } else if let Some(embedding) = response.embedding {
        vec![embedding]
    } else {
        return Err(ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            "missing embedding response",
            ErrorClass::NonRetriable,
        ));
    };

    if embeddings.len() != expected_count {
        return Err(ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            format!(
                "embedding response count mismatch (expected {expected_count}, got {})",
                embeddings.len()
            ),
            ErrorClass::NonRetriable,
        ));
    }

    embeddings
        .into_iter()
        .map(|embedding| map_embedding_values(embedding.values, expected_dimension))
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
    fn normalize_model_resource_adds_prefix() {
        let normalized = normalize_model_resource("gemini-embedding-001").unwrap();
        assert_eq!(normalized.as_ref(), "models/gemini-embedding-001");
    }

    #[test]
    fn gemini_batch_request_serializes() {
        let request = GeminiBatchEmbedRequest {
            requests: vec![
                GeminiBatchRequestItem {
                    model: "models/gemini-embedding-001".into(),
                    content: GeminiContent::from_text("first".into()),
                    output_dimensionality: Some(2),
                },
                GeminiBatchRequestItem {
                    model: "models/gemini-embedding-001".into(),
                    content: GeminiContent::from_text("second".into()),
                    output_dimensionality: Some(2),
                },
            ],
        };
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(
            value,
            json!({
                "requests": [
                    {
                        "model": "models/gemini-embedding-001",
                        "content": { "parts": [ { "text": "first" } ] },
                        "output_dimensionality": 2
                    },
                    {
                        "model": "models/gemini-embedding-001",
                        "content": { "parts": [ { "text": "second" } ] },
                        "output_dimensionality": 2
                    }
                ]
            })
        );
    }

    #[test]
    fn map_gemini_embeddings_handles_single_embedding() {
        let response = GeminiEmbeddingResponse {
            embedding: Some(GeminiEmbeddingValues {
                values: vec![0.1, 0.2],
            }),
            embeddings: None,
        };
        let embeddings = map_gemini_embeddings(response, 1, Some(2)).unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].dimension(), 2);
    }

    #[test]
    fn map_gemini_http_error_permission_denied() {
        let payload = serde_json::to_vec(&json!({
            "error": {
                "message": "forbidden",
                "status": "PERMISSION_DENIED",
                "code": 403
            }
        }))
        .unwrap();
        let envelope = map_gemini_http_error(StatusCode::FORBIDDEN, &payload);
        assert_eq!(envelope.code, ErrorCode::permission_denied());
    }

    #[tokio::test]
    async fn detect_dimension_uses_override() -> Result<()> {
        let config = GeminiEmbeddingConfig {
            api_key: "example".into(), // pragma: allowlist secret
            model: None,
            base_url: Some("http://localhost".into()),
            timeout_ms: 1_000,
            dimension: Some(8),
        };
        let adapter = GeminiEmbedding::new(&config)?;
        let ctx = RequestContext::new_request();
        let dimension = adapter
            .detect_dimension(&ctx, DetectDimensionOptions::default().into())
            .await?;
        assert_eq!(dimension, 8);
        Ok(())
    }
}
