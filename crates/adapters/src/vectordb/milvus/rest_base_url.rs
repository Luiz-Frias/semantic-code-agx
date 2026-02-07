//! Milvus REST base URL normalization.

use semantic_code_shared::{ErrorCode, ErrorEnvelope, Result};

pub fn to_milvus_rest_base_url(address: &str) -> Result<Box<str>> {
    let trimmed = address.trim();
    if trimmed.is_empty() {
        return Err(ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            "Milvus REST address is required",
        ));
    }

    let mut processed = trimmed.to_owned();
    if !processed.starts_with("http://") && !processed.starts_with("https://") {
        processed = format!("http://{processed}");
    }

    let trimmed_len = processed.trim_end_matches('/').len();
    processed.truncate(trimmed_len);
    if processed.ends_with("/v2/vectordb") {
        return Ok(processed.into_boxed_str());
    }

    Ok(format!("{processed}/v2/vectordb").into_boxed_str())
}
