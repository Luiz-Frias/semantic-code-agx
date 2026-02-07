//! Shared Milvus helpers.

use semantic_code_domain::CollectionName;
use semantic_code_shared::{ErrorCode, ErrorEnvelope, Result};

pub const DEFAULT_VECTOR_FIELD: &str = "vector";
pub const DEFAULT_SPARSE_FIELD: &str = "sparse_vector";
pub const DEFAULT_COLLECTION_DESCRIPTION: &str = "Semantic code search collection";
pub const DEFAULT_HYBRID_COLLECTION_DESCRIPTION: &str = "Semantic code search hybrid collection";

pub const MILVUS_OUTPUT_FIELDS: [&str; 7] = [
    "id",
    "content",
    "relativePath",
    "startLine",
    "endLine",
    "fileExtension",
    "metadata",
];

const MAX_COLLECTION_NAME_LEN: usize = 255;

pub fn ensure_collection_name(collection: &CollectionName) -> Result<()> {
    let name = collection.as_str();
    if name.len() > MAX_COLLECTION_NAME_LEN {
        return Err(ErrorEnvelope::expected(
            ErrorCode::new("vector", "collection_name_too_long"),
            format!("collection name exceeds {MAX_COLLECTION_NAME_LEN} characters",),
        ));
    }
    Ok(())
}

pub fn milvus_in_string(field: &str, values: &[Box<str>]) -> Box<str> {
    let escaped = values
        .iter()
        .map(|value| format!("\"{}\"", escape_milvus_string_literal(value)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{field} in [{escaped}]").into_boxed_str()
}

fn escape_milvus_string_literal(input: &str) -> String {
    input.replace('\\', "\\\\").replace('"', "\\\"")
}
