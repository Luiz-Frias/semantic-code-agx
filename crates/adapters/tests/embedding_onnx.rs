// ONNX integration tests (feature-gated).
#![cfg(feature = "onnx")]
#![allow(missing_docs)]

use semantic_code_adapters::embedding::onnx::{OnnxEmbedding, OnnxEmbeddingConfig};
use semantic_code_ports::embedding::{EmbedBatchRequest, EmbeddingPort};
use semantic_code_shared::{RequestContext, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const DEFAULT_ONNX_REPO: &str = "Xenova/all-MiniLM-L6-v2";

#[tokio::test]
async fn onnx_adapter_loads_cached_model() -> Result<()> {
    let Some(model_dir) = locate_cached_model_dir() else {
        eprintln!("skipping ONNX integration test (cached model not found)");
        return Ok(());
    };

    let config = OnnxEmbeddingConfig {
        model_dir,
        model_filename: None,
        tokenizer_filename: None,
        dimension: None,
        session_pool_size: 1,
    };
    let adapter = OnnxEmbedding::new(&config)?;
    let ctx = RequestContext::new_request();
    let embedding = adapter.embed(&ctx, "hello".into()).await?;
    assert!(!embedding.as_slice().is_empty());
    Ok(())
}

#[tokio::test]
async fn onnx_pool_handles_parallel_batches() -> Result<()> {
    let Some(model_dir) = locate_cached_model_dir() else {
        eprintln!("skipping ONNX integration test (cached model not found)");
        return Ok(());
    };

    let config = OnnxEmbeddingConfig {
        model_dir,
        model_filename: None,
        tokenizer_filename: None,
        dimension: None,
        session_pool_size: 2,
    };
    let adapter = Arc::new(OnnxEmbedding::new(&config)?);
    let ctx = RequestContext::new_request();

    let batch_a: Vec<String> = vec!["hello".to_string(), "world".to_string()];
    let batch_b: Vec<String> = vec!["foo".to_string(), "bar".to_string()];

    let task_a = adapter.embed_batch(&ctx, EmbedBatchRequest::from(batch_a));
    let task_b = adapter.embed_batch(&ctx, EmbedBatchRequest::from(batch_b));
    let (a, b) = tokio::join!(task_a, task_b);

    let a = a?;
    let b = b?;
    assert_eq!(a.len(), 2);
    assert_eq!(b.len(), 2);
    Ok(())
}

fn locate_cached_model_dir() -> Option<PathBuf> {
    let repo_slug = DEFAULT_ONNX_REPO.replace('/', "-");
    let cwd = std::env::current_dir().ok()?;
    let base = cwd.join(".context");
    let preferred = base.join("models").join("onnx").join(&repo_slug);
    if has_assets(&preferred) {
        return Some(preferred);
    }
    let legacy = base.join("onnx-cache").join(&repo_slug);
    if has_assets(&legacy) {
        return Some(legacy);
    }
    None
}

fn has_assets(dir: &Path) -> bool {
    model_exists(dir) && dir.join("tokenizer.json").exists()
}

fn model_exists(dir: &Path) -> bool {
    dir.join("onnx").join("model.onnx").exists() || dir.join("model.onnx").exists()
}
