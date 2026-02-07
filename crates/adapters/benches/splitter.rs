//! Splitter throughput benchmark placeholder.

use semantic_code_adapters::splitter::TreeSitterSplitter;
use semantic_code_ports::{Language, SplitOptions, SplitterPort};
use semantic_code_shared::{ErrorEnvelope, RequestContext, Result};
use std::path::PathBuf;
use std::time::Instant;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testkit/fixtures/splitter/basic/src/lib.rs")
}

fn main() -> Result<()> {
    if std::env::args().any(|arg| arg == "--list") {
        println!("splitter_placeholder: benchmark");
        return Ok(());
    }

    let code = std::fs::read_to_string(fixture_path()).map_err(ErrorEnvelope::from)?;
    let splitter = TreeSitterSplitter::default();
    let ctx = RequestContext::new_request();
    let runtime = tokio::runtime::Runtime::new().map_err(ErrorEnvelope::from)?;

    let start = Instant::now();
    let mut iterations = 0u32;
    for _ in 0..10 {
        iterations += 1;
        let _ = runtime.block_on(splitter.split(
            &ctx,
            code.clone().into_boxed_str(),
            Language::Rust,
            SplitOptions {
                file_path: Some("src/lib.rs".into()),
            },
        ))?;
    }
    let elapsed = start.elapsed().as_millis();

    println!("Splitter benchmark placeholder: {iterations} iterations in {elapsed} ms");
    Ok(())
}
