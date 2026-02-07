//! Milvus metadata serialization helpers.

use semantic_code_domain::{Language, LineSpan, VectorDocumentMetadata};
use semantic_code_shared::{ErrorCode, ErrorEnvelope, Result};
use serde_json::Value;

/// Serializes vector document metadata into a JSON string for storage.
pub fn serialize_metadata(metadata: &VectorDocumentMetadata) -> Result<String> {
    serde_json::to_string(metadata).map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "metadata_serialize_failed"),
            format!("failed to serialize metadata: {error}"),
            semantic_code_shared::ErrorClass::NonRetriable,
        )
    })
}

/// Parses a JSON metadata payload from Milvus into a typed metadata object.
pub fn parse_metadata(raw: Option<&str>) -> Option<VectorDocumentMetadata> {
    let raw = raw?.trim();
    if raw.is_empty() {
        return None;
    }
    let value: Value = serde_json::from_str(raw).ok()?;
    metadata_from_value(&value)
}

/// Builds metadata from Milvus row fields and optional metadata JSON.
pub fn metadata_from_fields(
    relative_path: &str,
    start_line: i64,
    end_line: i64,
    file_extension: Option<&str>,
    metadata_json: Option<&str>,
) -> Result<VectorDocumentMetadata> {
    let start = u32::try_from(start_line).map_err(|_| {
        ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            format!("invalid start line: {start_line}"),
        )
    })?;
    let end = u32::try_from(end_line).map_err(|_| {
        ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            format!("invalid end line: {end_line}"),
        )
    })?;
    let span = LineSpan::new(start, end)
        .map_err(|error| ErrorEnvelope::expected(ErrorCode::invalid_input(), error.to_string()))?;

    let mut metadata = VectorDocumentMetadata {
        relative_path: relative_path.trim().to_owned().into_boxed_str(),
        language: None,
        file_extension: file_extension
            .map(str::trim)
            .filter(|ext| !ext.is_empty())
            .map(|ext| ext.to_owned().into_boxed_str()),
        span,
        node_kind: None,
    };

    if let Some(parsed) = parse_metadata(metadata_json) {
        if parsed.language.is_some() {
            metadata.language = parsed.language;
        }
        if parsed.node_kind.is_some() {
            metadata.node_kind = parsed.node_kind;
        }
        if parsed.file_extension.is_some() {
            metadata.file_extension = parsed.file_extension;
        }
    }

    Ok(metadata)
}

fn metadata_from_value(value: &Value) -> Option<VectorDocumentMetadata> {
    let object = value.as_object()?;
    let relative_path = object.get("relativePath")?.as_str()?.trim();
    if relative_path.is_empty() {
        return None;
    }

    let start_line = u32::try_from(object.get("span")?.get("startLine")?.as_u64()?).ok()?;
    let end_line = u32::try_from(object.get("span")?.get("endLine")?.as_u64()?).ok()?;
    let span = LineSpan::new(start_line, end_line).ok()?;

    let language = object
        .get("language")
        .and_then(Value::as_str)
        .and_then(parse_language);

    let file_extension = object
        .get("fileExtension")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|ext| !ext.is_empty())
        .map(|ext| ext.to_owned().into_boxed_str());

    let node_kind = object
        .get("nodeKind")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_owned().into_boxed_str());

    Some(VectorDocumentMetadata {
        relative_path: relative_path.to_owned().into_boxed_str(),
        language,
        file_extension,
        span,
        node_kind,
    })
}

fn parse_language(raw: &str) -> Option<Language> {
    let normalized = raw.trim();
    if normalized.is_empty() {
        return None;
    }
    match normalized {
        "typescript" => Some(Language::TypeScript),
        "javascript" => Some(Language::JavaScript),
        "python" => Some(Language::Python),
        "java" => Some(Language::Java),
        "cpp" => Some(Language::Cpp),
        "c" => Some(Language::C),
        "csharp" => Some(Language::CSharp),
        "go" => Some(Language::Go),
        "rust" => Some(Language::Rust),
        "php" => Some(Language::Php),
        "ruby" => Some(Language::Ruby),
        "swift" => Some(Language::Swift),
        "kotlin" => Some(Language::Kotlin),
        "scala" => Some(Language::Scala),
        "objective-c" => Some(Language::ObjectiveC),
        "jupyter" => Some(Language::Jupyter),
        "markdown" => Some(Language::Markdown),
        "text" => Some(Language::Text),
        _ => None,
    }
}
