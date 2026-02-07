//! Embedding performance benchmarks (mock + ONNX).

use semantic_code_adapters::embedding::onnx::{OnnxEmbedding, OnnxEmbeddingConfig};
use semantic_code_adapters::embedding_test::TestEmbedding;
use semantic_code_ports::EmbeddingPort;
use semantic_code_shared::{ErrorEnvelope, RequestContext, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_ONNX_REPO: &str = "Xenova/all-MiniLM-L6-v2";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchMode {
    Mock,
    Onnx,
    Both,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--list") {
        println!("embedding_mock: benchmark");
        println!("embedding_onnx: benchmark");
        return Ok(());
    }

    let mode = parse_mode(&args);
    let iterations = parse_u32_arg(&args, "--iterations").unwrap_or(20);
    let batch_size = parse_usize_arg(&args, "--batch-size").unwrap_or(16);
    let dimension = parse_u32_arg(&args, "--dimension").unwrap_or(384);
    let session_pool = parse_usize_arg(&args, "--session-pool").unwrap_or(1);

    let texts = build_texts(batch_size);

    if matches!(mode, BenchMode::Mock | BenchMode::Both) {
        let embedder = Arc::new(TestEmbedding::new(dimension)?);
        run_bench("embedding_mock", embedder, &texts, iterations)?;
    }

    if matches!(mode, BenchMode::Onnx | BenchMode::Both) {
        if let Some(model_dir) = resolve_model_dir(&args) {
            let config = OnnxEmbeddingConfig {
                model_dir,
                model_filename: None,
                tokenizer_filename: None,
                dimension: None,
                session_pool_size: session_pool,
            };
            let embedder = Arc::new(OnnxEmbedding::new(&config)?);
            run_bench("embedding_onnx", embedder, &texts, iterations)?;
        } else {
            eprintln!("skipping ONNX benchmark (cached model not found)");
        }
    }

    Ok(())
}

fn parse_mode(args: &[String]) -> BenchMode {
    if args.iter().any(|arg| arg == "--mock") {
        return BenchMode::Mock;
    }
    if args.iter().any(|arg| arg == "--onnx") {
        return BenchMode::Onnx;
    }
    BenchMode::Both
}

fn parse_u32_arg(args: &[String], key: &str) -> Option<u32> {
    parse_usize_arg(args, key).and_then(|value| u32::try_from(value).ok())
}

fn parse_usize_arg(args: &[String], key: &str) -> Option<usize> {
    args.iter()
        .find_map(|arg| arg.strip_prefix(&format!("{key}=")))
        .and_then(|value| value.parse::<usize>().ok())
}

fn build_texts(batch_size: usize) -> Vec<Box<str>> {
    let mut texts = Vec::with_capacity(batch_size.max(1));
    for idx in 0..batch_size.max(1) {
        let text = format!("embedding sample {idx}");
        texts.push(text.into_boxed_str());
    }
    texts
}

fn run_bench(
    label: &str,
    embedder: Arc<dyn EmbeddingPort>,
    texts: &[Box<str>],
    iterations: u32,
) -> Result<()> {
    let ctx = RequestContext::new_request();
    let runtime = tokio::runtime::Runtime::new().map_err(ErrorEnvelope::from)?;

    let _ = runtime.block_on(embedder.embed_batch(&ctx, texts.to_vec().into()))?;

    let start = Instant::now();
    let mut total_vectors = 0u64;
    for _ in 0..iterations {
        let batch = texts.to_vec();
        let output = runtime.block_on(embedder.embed_batch(&ctx, batch.into()))?;
        total_vectors = total_vectors.saturating_add(output.len() as u64);
    }
    let elapsed_ms = start.elapsed().as_millis();

    println!("{label}: {iterations} iterations, {total_vectors} vectors in {elapsed_ms} ms");
    Ok(())
}

fn resolve_model_dir(args: &[String]) -> Option<PathBuf> {
    if let Some(dir) = parse_path_arg(args, "--model-dir") {
        if has_assets(&dir) {
            return Some(dir);
        }
    }
    locate_cached_model_dir()
}

fn parse_path_arg(args: &[String], key: &str) -> Option<PathBuf> {
    args.iter()
        .find_map(|arg| arg.strip_prefix(&format!("{key}=")))
        .map(PathBuf::from)
}

fn locate_cached_model_dir() -> Option<PathBuf> {
    let repo_slug = DEFAULT_ONNX_REPO.replace('/', "-");
    let context_dir = find_context_dir()?;
    let preferred = context_dir.join("models").join("onnx").join(&repo_slug);
    if has_assets(&preferred) {
        return Some(preferred);
    }
    let legacy = context_dir.join("onnx-cache").join(&repo_slug);
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

fn find_context_dir() -> Option<PathBuf> {
    if let Ok(value) = std::env::var("SCA_CONTEXT_DIR") {
        let candidate = PathBuf::from(value);
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        if let Some(found) = find_context_dir_from(&cwd) {
            return Some(found);
        }
    }
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    find_context_dir_from(manifest_dir)
}

fn find_context_dir_from(start: &Path) -> Option<PathBuf> {
    for ancestor in start.ancestors() {
        let candidate = ancestor.join(".context");
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    None
}
