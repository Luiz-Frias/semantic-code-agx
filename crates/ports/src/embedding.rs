//! Embedding boundary contract.

use crate::BoxFuture;
use semantic_code_domain::EmbeddingProviderId;
use semantic_code_shared::{RequestContext, Result};
use std::future::Future;
use std::sync::Arc;

/// An embedding vector payload.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingVector {
    /// Dense embedding vector.
    vector: Arc<[f32]>,
    /// Vector dimensionality.
    dimension: u32,
}

impl EmbeddingVector {
    /// Build an embedding vector from a shared slice.
    #[must_use]
    pub fn new(vector: Arc<[f32]>) -> Self {
        let dimension = u32::try_from(vector.len()).unwrap_or(0);
        Self { vector, dimension }
    }

    /// Build an embedding vector from an owned vector.
    #[must_use]
    pub fn from_vec(vector: Vec<f32>) -> Self {
        Self::new(Arc::from(vector))
    }

    /// Borrow the vector as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.vector
    }

    /// Borrow the shared vector buffer.
    #[must_use]
    pub const fn vector(&self) -> &Arc<[f32]> {
        &self.vector
    }

    /// Return the embedding dimension.
    #[must_use]
    pub const fn dimension(&self) -> u32 {
        self.dimension
    }

    /// Consume and return the shared vector buffer.
    #[must_use]
    pub fn into_vector(self) -> Arc<[f32]> {
        self.vector
    }
}

/// Embedding vector with a compile-time dimension.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingVectorFixed<const D: usize> {
    vector: Arc<[f32]>,
}

impl<const D: usize> EmbeddingVectorFixed<D> {
    /// Validate and build a fixed-dimension embedding.
    pub fn new(vector: Arc<[f32]>) -> Result<Self> {
        if vector.len() != D {
            return Err(semantic_code_shared::ErrorEnvelope::expected(
                semantic_code_shared::ErrorCode::invalid_input(),
                format!(
                    "embedding dimension mismatch (expected {D}, got {})",
                    vector.len()
                ),
            ));
        }
        Ok(Self { vector })
    }

    /// Borrow the vector as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.vector
    }

    /// Consume and return the shared vector buffer.
    #[must_use]
    pub fn into_vector(self) -> Arc<[f32]> {
        self.vector
    }
}

impl<const D: usize> TryFrom<EmbeddingVector> for EmbeddingVectorFixed<D> {
    type Error = semantic_code_shared::ErrorEnvelope;

    fn try_from(value: EmbeddingVector) -> Result<Self> {
        Self::new(value.into_vector())
    }
}

impl<const D: usize> From<EmbeddingVectorFixed<D>> for EmbeddingVector {
    fn from(value: EmbeddingVectorFixed<D>) -> Self {
        Self::new(value.into_vector())
    }
}

/// Provider descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingProviderInfo {
    /// Stable provider identifier.
    pub id: EmbeddingProviderId,
    /// Human-readable provider name.
    pub name: Box<str>,
}

/// Options for dimension detection.
#[derive(Debug, Clone, Default)]
pub struct DetectDimensionOptions {
    /// Optional text to probe dimension (provider-specific).
    pub test_text: Option<Box<str>>,
}

/// Owned request to detect embedding dimension.
#[derive(Debug, Clone, Default)]
pub struct DetectDimensionRequest {
    /// Detection options.
    pub options: DetectDimensionOptions,
}

impl From<DetectDimensionOptions> for DetectDimensionRequest {
    fn from(options: DetectDimensionOptions) -> Self {
        Self { options }
    }
}

/// Owned request to embed a single text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedRequest {
    /// Text to embed.
    pub text: Box<str>,
}

impl From<Box<str>> for EmbedRequest {
    fn from(text: Box<str>) -> Self {
        Self { text }
    }
}

impl From<String> for EmbedRequest {
    fn from(text: String) -> Self {
        Self {
            text: text.into_boxed_str(),
        }
    }
}

impl From<&str> for EmbedRequest {
    fn from(text: &str) -> Self {
        Self {
            text: text.to_owned().into_boxed_str(),
        }
    }
}

/// Owned request to embed a batch of texts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedBatchRequest {
    /// Texts to embed.
    pub texts: Vec<Box<str>>,
}

impl From<Vec<Box<str>>> for EmbedBatchRequest {
    fn from(texts: Vec<Box<str>>) -> Self {
        Self { texts }
    }
}

impl From<Vec<String>> for EmbedBatchRequest {
    fn from(texts: Vec<String>) -> Self {
        Self {
            texts: texts.into_iter().map(String::into_boxed_str).collect(),
        }
    }
}

/// Boundary contract for embedding generation.
pub trait EmbeddingPort: Send + Sync {
    /// Provider info for this implementation.
    fn provider(&self) -> &EmbeddingProviderInfo;

    /// Detect the embedding vector dimension.
    fn detect_dimension(
        &self,
        ctx: &RequestContext,
        request: DetectDimensionRequest,
    ) -> BoxFuture<'_, Result<u32>>;

    /// Embed a single text.
    fn embed(
        &self,
        ctx: &RequestContext,
        request: EmbedRequest,
    ) -> BoxFuture<'_, Result<EmbeddingVector>>;

    /// Embed multiple texts in a batch.
    fn embed_batch(
        &self,
        ctx: &RequestContext,
        request: EmbedBatchRequest,
    ) -> BoxFuture<'_, Result<Vec<EmbeddingVector>>>;
}

/// Lending-style embedding port using GAT futures.
pub trait EmbeddingPortLend: Send + Sync {
    /// Future type returned by this port.
    type Future<'a, T>: Future<Output = Result<T>> + Send + 'a
    where
        Self: 'a,
        T: 'a;

    /// Provider info for this implementation.
    fn provider(&self) -> &EmbeddingProviderInfo;

    /// Detect the embedding vector dimension.
    fn detect_dimension(
        &self,
        ctx: &RequestContext,
        request: DetectDimensionRequest,
    ) -> Self::Future<'_, u32>;

    /// Embed a single text.
    fn embed(
        &self,
        ctx: &RequestContext,
        request: EmbedRequest,
    ) -> Self::Future<'_, EmbeddingVector>;

    /// Embed multiple texts in a batch.
    fn embed_batch(
        &self,
        ctx: &RequestContext,
        request: EmbedBatchRequest,
    ) -> Self::Future<'_, Vec<EmbeddingVector>>;
}

impl<T> EmbeddingPortLend for T
where
    T: EmbeddingPort + ?Sized,
{
    type Future<'a, U>
        = BoxFuture<'a, Result<U>>
    where
        T: 'a,
        U: 'a;

    fn provider(&self) -> &EmbeddingProviderInfo {
        EmbeddingPort::provider(self)
    }

    fn detect_dimension(
        &self,
        ctx: &RequestContext,
        request: DetectDimensionRequest,
    ) -> Self::Future<'_, u32> {
        EmbeddingPort::detect_dimension(self, ctx, request)
    }

    fn embed(
        &self,
        ctx: &RequestContext,
        request: EmbedRequest,
    ) -> Self::Future<'_, EmbeddingVector> {
        EmbeddingPort::embed(self, ctx, request)
    }

    fn embed_batch(
        &self,
        ctx: &RequestContext,
        request: EmbedBatchRequest,
    ) -> Self::Future<'_, Vec<EmbeddingVector>> {
        EmbeddingPort::embed_batch(self, ctx, request)
    }
}
