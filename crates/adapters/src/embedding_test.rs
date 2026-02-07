//! Deterministic embedding adapter used for tests.

use semantic_code_ports::{
    DetectDimensionRequest, EmbedBatchRequest, EmbedRequest, EmbeddingPort, EmbeddingProviderId,
    EmbeddingProviderInfo, EmbeddingVector,
};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use sha2::{Digest, Sha256};

/// Deterministic embedding adapter for local testing.
#[derive(Clone)]
pub struct TestEmbedding {
    provider: EmbeddingProviderInfo,
    dimension: u32,
}

impl TestEmbedding {
    /// Build a deterministic test embedder with a fixed dimension.
    pub fn new(dimension: u32) -> Result<Self> {
        if dimension == 0 {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "embedding dimension must be positive",
            ));
        }
        let provider = EmbeddingProviderInfo {
            id: EmbeddingProviderId::parse("test").map_err(ErrorEnvelope::from)?,
            name: "test".into(),
        };
        Ok(Self {
            provider,
            dimension,
        })
    }

    fn dimension_checked(&self) -> Result<u32> {
        if self.dimension == 0 {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "embedding dimension must be positive",
            ));
        }
        Ok(self.dimension)
    }

    fn dimension_usize(&self) -> Result<usize> {
        usize::try_from(self.dimension_checked()?).map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "embedding dimension overflow",
                ErrorClass::NonRetriable,
            )
        })
    }

    fn vector_for(&self, text: &str) -> Result<Vec<f32>> {
        let dimension = self.dimension_usize()?;
        let mut vector = Vec::with_capacity(dimension);
        let mut counter = 0u64;

        while vector.len() < dimension {
            let mut hasher = Sha256::new();
            hasher.update(text.as_bytes());
            hasher.update(b":");
            hasher.update(counter.to_le_bytes());
            let digest = hasher.finalize();
            for byte in digest {
                vector.push(f32::from(byte) / 255.0);
                if vector.len() == dimension {
                    break;
                }
            }
            counter = counter.saturating_add(1);
        }

        Ok(vector)
    }
}

impl EmbeddingPort for TestEmbedding {
    fn provider(&self) -> &EmbeddingProviderInfo {
        &self.provider
    }

    fn detect_dimension(
        &self,
        _ctx: &RequestContext,
        _request: DetectDimensionRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<u32>> {
        let dimension = self.dimension_checked();
        Box::pin(async move { dimension })
    }

    fn embed(
        &self,
        _ctx: &RequestContext,
        request: EmbedRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<EmbeddingVector>> {
        let text = request.text;
        let dimension = match self.dimension_checked() {
            Ok(value) => value,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        let vector = match self.vector_for(text.as_ref()) {
            Ok(value) => value,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        let vector = std::sync::Arc::from(vector);
        let _ = dimension;
        Box::pin(async move { Ok(EmbeddingVector::new(vector)) })
    }

    fn embed_batch(
        &self,
        _ctx: &RequestContext,
        request: EmbedBatchRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<EmbeddingVector>>> {
        let texts = request.texts;
        let dimension = match self.dimension_checked() {
            Ok(value) => value,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        let vectors = texts
            .iter()
            .map(|text| self.vector_for(text.as_ref()))
            .collect::<Result<Vec<_>>>();
        let vectors = match vectors {
            Ok(value) => value,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        Box::pin(async move {
            let _ = dimension;
            Ok(vectors
                .into_iter()
                .map(|vector| EmbeddingVector::new(std::sync::Arc::from(vector)))
                .collect())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::TestEmbedding;
    use semantic_code_ports::EmbeddingPort;
    use semantic_code_shared::{RequestContext, Result};

    #[tokio::test]
    async fn test_embedder_is_deterministic() -> Result<()> {
        let embedder = TestEmbedding::new(8)?;
        let ctx = RequestContext::new_request();
        let a = embedder.embed(&ctx, "hello".into()).await?;
        let b = embedder.embed(&ctx, "hello".into()).await?;
        assert_eq!(a.vector(), b.vector());
        assert_eq!(a.dimension(), 8);
        Ok(())
    }
}
