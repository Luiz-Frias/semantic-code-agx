//! Integration coverage for domain primitives and spans.

use semantic_code_domain::{CodebaseId, CollectionName, LineSpan, PrimitiveError};
use semantic_code_shared::ErrorEnvelope;

#[test]
fn primitive_errors_map_into_error_envelopes() -> Result<(), PrimitiveError> {
    let Err(error) = CodebaseId::parse(" ") else {
        return Err(PrimitiveError::InvalidCodebaseId { input_length: 0 });
    };

    let envelope: ErrorEnvelope = error.into();
    assert_eq!(envelope.code.namespace(), "domain");
    assert_eq!(envelope.code.code(), "invalid_codebase_id");
    assert_eq!(
        envelope.metadata.get("input_length"),
        Some(&"1".to_string())
    );

    let Err(span_error) = LineSpan::new(0, 1) else {
        return Err(PrimitiveError::LineSpanNonPositive {
            start_line: 0,
            end_line: 1,
        });
    };

    let envelope: ErrorEnvelope = span_error.into();
    assert_eq!(envelope.code.code(), "invalid_line_span");
    assert_eq!(envelope.metadata.get("start_line"), Some(&"0".to_string()));
    assert_eq!(envelope.metadata.get("end_line"), Some(&"1".to_string()));

    Ok(())
}

#[test]
fn collection_name_validation_surfaces_metadata() -> Result<(), PrimitiveError> {
    let Err(error) = CollectionName::parse("bad-name") else {
        return Err(PrimitiveError::InvalidCollectionName {
            input: "bad-name".to_string(),
        });
    };

    let envelope: ErrorEnvelope = error.into();
    assert_eq!(envelope.code.code(), "invalid_collection_name");
    assert_eq!(
        envelope.metadata.get("input"),
        Some(&"bad-name".to_string())
    );

    Ok(())
}
