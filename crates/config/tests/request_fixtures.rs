//! Integration tests for request fixture validation.

use semantic_code_config::{
    ClearIndexRequestDto, IndexRequestDto, ReindexByChangeRequestDto, SearchRequestDto,
    validate_clear_index_request, validate_index_request, validate_reindex_by_change_request,
    validate_search_request,
};
use semantic_code_shared::{ErrorCode, Validate};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_dir.to_path_buf())
}

fn read_fixture(relative: &str) -> Result<String, Box<dyn Error>> {
    let path = workspace_root()
        .join("crates")
        .join("testkit")
        .join("fixtures")
        .join(relative);
    Ok(fs::read_to_string(path)?)
}

#[test]
fn validates_request_fixtures() -> Result<(), Box<dyn Error>> {
    let index: IndexRequestDto = serde_json::from_str(&read_fixture("requests/index.valid.json")?)?;
    index
        .validate()
        .map_err(|err| Box::<dyn Error>::from(err))?;
    let index_validated = validate_index_request(&index)?;
    assert!(index_validated.force_reindex);

    let search: SearchRequestDto =
        serde_json::from_str(&read_fixture("requests/search.valid.json")?)?;
    let search_validated = validate_search_request(&search)?;
    assert_eq!(search_validated.top_k, Some(10));
    assert_eq!(search_validated.threshold, Some(0.5));
    assert_eq!(
        search_validated.filter_expr.as_deref(),
        Some("relativePath == 'src/main.rs'")
    );

    let invalid_topk: SearchRequestDto =
        serde_json::from_str(&read_fixture("requests/search.invalid_topk.json")?)?;
    let error = validate_search_request(&invalid_topk).err();
    assert!(
        matches!(error, Some(envelope) if envelope.code == ErrorCode::new("config", "out_of_range"))
    );

    let invalid_filter: SearchRequestDto =
        serde_json::from_str(&read_fixture("requests/search.invalid_filter_expr.json")?)?;
    let error = validate_search_request(&invalid_filter).err();
    assert!(
        matches!(error, Some(envelope) if envelope.code == ErrorCode::new("config", "invalid_filter_expr"))
    );

    let invalid_root: SearchRequestDto =
        serde_json::from_str(&read_fixture("requests/search.invalid_codebase_root.json")?)?;
    let error = validate_search_request(&invalid_root).err();
    assert!(
        matches!(error, Some(envelope) if envelope.code == ErrorCode::new("config", "invalid_codebase_root"))
    );

    let reindex: ReindexByChangeRequestDto =
        serde_json::from_str(&read_fixture("requests/reindex.valid.json")?)?;
    validate_reindex_by_change_request(&reindex)?;

    let clear: ClearIndexRequestDto =
        serde_json::from_str(&read_fixture("requests/clear.valid.json")?)?;
    validate_clear_index_request(&clear)?;

    Ok(())
}
