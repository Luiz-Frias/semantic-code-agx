//! Integration tests for tree-sitter splitter.

use semantic_code_adapters::TreeSitterSplitter;
use semantic_code_ports::{Language, SplitOptions, SplitterPort};
use semantic_code_shared::{ErrorEnvelope, RequestContext, Result};
use std::path::PathBuf;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testkit/fixtures/splitter/basic/src/lib.rs")
}

#[tokio::test]
async fn split_rust_fixture_into_chunks() -> Result<()> {
    let code = std::fs::read_to_string(fixture_path()).map_err(ErrorEnvelope::from)?;
    let splitter = TreeSitterSplitter::new(3, 0);
    let ctx = RequestContext::new_request();

    let chunks = splitter
        .split(
            &ctx,
            code.into_boxed_str(),
            Language::Rust,
            SplitOptions {
                file_path: Some("src/lib.rs".into()),
            },
        )
        .await?;

    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].span.start_line(), 1);
    assert_eq!(chunks[0].span.end_line(), 3);
    assert_eq!(chunks[1].span.start_line(), 4);
    assert_eq!(chunks[1].span.end_line(), 4);
    assert_eq!(chunks[2].span.start_line(), 5);
    assert_eq!(chunks[2].span.end_line(), 7);
    Ok(())
}

#[tokio::test]
async fn split_rust_methods_include_struct_prelude() -> Result<()> {
    let code = r#"struct Calculator;

impl Calculator {
    fn multiply(&self) -> i32 {
        1
    }

    fn subtract(&self) -> i32 {
        2
    }
}
"#;
    let splitter = TreeSitterSplitter::new(2, 0);
    let ctx = RequestContext::new_request();

    let chunks = splitter
        .split(
            &ctx,
            code.to_string().into_boxed_str(),
            Language::Rust,
            SplitOptions {
                file_path: Some("src/lib.rs".into()),
            },
        )
        .await?;

    assert!(!chunks.is_empty());
    assert_eq!(chunks[0].span.start_line(), 1);
    Ok(())
}
