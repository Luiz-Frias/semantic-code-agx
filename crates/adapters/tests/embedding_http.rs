// HTTP adapter integration tests (feature-gated).
#![allow(missing_docs)]

#[cfg(feature = "openai")]
mod openai {
    use semantic_code_adapters::embedding::openai::{OpenAiEmbedding, OpenAiEmbeddingConfig};
    use semantic_code_ports::embedding::EmbeddingPort;
    use semantic_code_shared::{RequestContext, Result};
    use serde_json::json;
    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn openai_embed_uses_mock_server() -> Result<()> {
        let server = MockServer::start().await;
        let response = ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                { "embedding": [0.1, 0.2], "index": 0 }
            ]
        }));

        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .and(header("authorization", "Bearer example"))
            .and(body_json(json!({
                "model": "text-embedding-3-small",
                "input": "hello",
                "dimensions": 2
            })))
            .respond_with(response)
            .mount(&server)
            .await;

        let config = OpenAiEmbeddingConfig {
            api_key: "example".into(), // pragma: allowlist secret
            model: Some("text-embedding-3-small".into()),
            base_url: Some(server.uri().into()),
            timeout_ms: 5_000,
            dimension: Some(2),
        };
        let adapter = OpenAiEmbedding::new(&config)?;
        let ctx = RequestContext::new_request();
        let embedding = adapter.embed(&ctx, "hello".into()).await?;
        assert_eq!(embedding.dimension(), 2);
        assert_eq!(embedding.as_slice(), &[0.1, 0.2]);
        Ok(())
    }
}

#[cfg(feature = "gemini")]
mod gemini {
    use semantic_code_adapters::embedding::gemini::{GeminiEmbedding, GeminiEmbeddingConfig};
    use semantic_code_ports::embedding::{EmbedBatchRequest, EmbeddingPort};
    use semantic_code_shared::{RequestContext, Result};
    use serde_json::json;
    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn gemini_batch_embed_uses_mock_server() -> Result<()> {
        let server = MockServer::start().await;
        let response = ResponseTemplate::new(200).set_body_json(json!({
            "embeddings": [
                { "values": [0.1, 0.2] },
                { "values": [0.3, 0.4] }
            ]
        }));

        Mock::given(method("POST"))
            .and(path("/models/gemini-embedding-001:batchEmbedContents"))
            .and(header("x-goog-api-key", "example"))
            .and(body_json(json!({
                "requests": [
                    {
                        "model": "models/gemini-embedding-001",
                        "content": { "parts": [ { "text": "a" } ] },
                        "output_dimensionality": 2
                    },
                    {
                        "model": "models/gemini-embedding-001",
                        "content": { "parts": [ { "text": "b" } ] },
                        "output_dimensionality": 2
                    }
                ]
            })))
            .respond_with(response)
            .mount(&server)
            .await;

        let config = GeminiEmbeddingConfig {
            api_key: "example".into(), // pragma: allowlist secret
            model: Some("gemini-embedding-001".into()),
            base_url: Some(server.uri().into()),
            timeout_ms: 5_000,
            dimension: Some(2),
        };
        let adapter = GeminiEmbedding::new(&config)?;
        let ctx = RequestContext::new_request();
        let embeddings = adapter
            .embed_batch(
                &ctx,
                EmbedBatchRequest::from(vec!["a".to_string(), "b".to_string()]),
            )
            .await?;
        assert_eq!(embeddings.len(), 2);
        Ok(())
    }
}

#[cfg(feature = "ollama")]
mod ollama {
    use semantic_code_adapters::embedding::ollama::{OllamaEmbedding, OllamaEmbeddingConfig};
    use semantic_code_ports::embedding::EmbeddingPort;
    use semantic_code_shared::{RequestContext, Result};
    use serde_json::json;
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn ollama_embed_uses_mock_server() -> Result<()> {
        let server = MockServer::start().await;
        let response = ResponseTemplate::new(200).set_body_json(json!({
            "embeddings": [[0.1, 0.2]]
        }));

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .and(body_json(json!({
                "model": "embeddinggemma",
                "input": "hello",
                "dimensions": 2
            })))
            .respond_with(response)
            .mount(&server)
            .await;

        let config = OllamaEmbeddingConfig {
            model: Some("embeddinggemma".into()),
            base_url: Some(server.uri().into()),
            timeout_ms: 5_000,
            dimension: Some(2),
        };
        let adapter = OllamaEmbedding::new(&config)?;
        let ctx = RequestContext::new_request();
        let embedding = adapter.embed(&ctx, "hello".into()).await?;
        assert_eq!(embedding.dimension(), 2);
        Ok(())
    }
}

#[cfg(feature = "voyage")]
mod voyage {
    use semantic_code_adapters::embedding::voyage::{VoyageEmbedding, VoyageEmbeddingConfig};
    use semantic_code_ports::embedding::{EmbedBatchRequest, EmbeddingPort};
    use semantic_code_shared::{RequestContext, Result};
    use serde_json::json;
    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn voyage_batch_embed_uses_mock_server() -> Result<()> {
        let server = MockServer::start().await;
        let response = ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                { "embedding": [0.1, 0.2], "index": 0 },
                { "embedding": [0.3, 0.4], "index": 1 }
            ]
        }));

        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .and(header("authorization", "Bearer example"))
            .and(body_json(json!({
                "model": "voyage-4",
                "input": ["a", "b"],
                "output_dimension": 2
            })))
            .respond_with(response)
            .mount(&server)
            .await;

        let config = VoyageEmbeddingConfig {
            api_key: "example".into(), // pragma: allowlist secret
            model: Some("voyage-4".into()),
            base_url: Some(server.uri().into()),
            timeout_ms: 5_000,
            dimension: Some(2),
        };
        let adapter = VoyageEmbedding::new(&config)?;
        let ctx = RequestContext::new_request();
        let embeddings = adapter
            .embed_batch(
                &ctx,
                EmbedBatchRequest::from(vec!["a".to_string(), "b".to_string()]),
            )
            .await?;
        assert_eq!(embeddings.len(), 2);
        Ok(())
    }
}
