//! Milvus error mapping helpers.

use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Identifies the Milvus transport used for error metadata.
pub enum MilvusProviderId {
    /// Milvus gRPC client.
    Grpc,
    /// Milvus REST client.
    Rest,
}

impl MilvusProviderId {
    /// Returns a stable identifier string for error metadata.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Grpc => "milvus_grpc",
            Self::Rest => "milvus_rest",
        }
    }
}

#[derive(Debug, Clone)]
/// Context payload attached to Milvus error envelopes.
pub struct MilvusErrorContext {
    /// Provider identifier for the failing request.
    pub provider: MilvusProviderId,
    /// Operation label for tracing failures.
    pub operation: &'static str,
    /// Collection name, when the request is collection-scoped.
    pub collection_name: Option<String>,
    /// REST endpoint path, when available.
    pub endpoint: Option<String>,
}

#[cfg(feature = "milvus-grpc")]
/// Maps a tonic gRPC status into the shared error envelope format.
pub fn map_grpc_error(status: &tonic::Status, ctx: &MilvusErrorContext) -> ErrorEnvelope {
    let message = status.message().to_string();
    let code = match status.code() {
        tonic::Code::Unauthenticated | tonic::Code::PermissionDenied => vdb_auth_code(),
        tonic::Code::DeadlineExceeded => vdb_timeout_code(),
        tonic::Code::Unavailable => vdb_connection_code(),
        tonic::Code::InvalidArgument => vdb_query_invalid_code(),
        _ => vdb_unknown_code(),
    };

    let class = match code.code() {
        "vdb_timeout" | "vdb_connection" => ErrorClass::Retriable,
        _ => ErrorClass::NonRetriable,
    };

    let envelope = ErrorEnvelope::unexpected(code, message, class)
        .with_metadata("provider", ctx.provider.as_str())
        .with_metadata("operation", ctx.operation)
        .with_metadata("grpc_code", status.code().to_string());

    if let Some(collection) = ctx.collection_name.as_ref() {
        return envelope.with_metadata("collection", collection.to_owned());
    }

    envelope
}

/// Maps Milvus REST error payloads and status codes into shared envelopes.
pub fn map_rest_error(
    message: impl Into<String>,
    http_status: Option<u16>,
    ctx: &MilvusErrorContext,
) -> ErrorEnvelope {
    let message = message.into();
    let code = choose_code_from_message(&message, http_status);
    let class = match code.code() {
        "vdb_timeout" | "vdb_connection" => ErrorClass::Retriable,
        _ => ErrorClass::NonRetriable,
    };

    let mut envelope = ErrorEnvelope::unexpected(code, message, class)
        .with_metadata("provider", ctx.provider.as_str())
        .with_metadata("operation", ctx.operation);

    if let Some(status) = http_status {
        envelope = envelope.with_metadata("http_status", status.to_string());
    }
    if let Some(collection) = ctx.collection_name.as_ref() {
        envelope = envelope.with_metadata("collection", collection.to_owned());
    }
    if let Some(endpoint) = ctx.endpoint.as_ref() {
        envelope = envelope.with_metadata("endpoint", endpoint.to_owned());
    }

    envelope
}

#[cfg(feature = "milvus-rest")]
/// Maps reqwest transport errors into shared error envelopes.
pub fn map_rest_transport_error(error: &reqwest::Error, ctx: &MilvusErrorContext) -> ErrorEnvelope {
    if error.is_timeout() {
        return ErrorEnvelope::unexpected(
            vdb_timeout_code(),
            format!("Milvus REST request timed out: {error}"),
            ErrorClass::Retriable,
        )
        .with_metadata("provider", ctx.provider.as_str())
        .with_metadata("operation", ctx.operation);
    }
    if error.is_connect() {
        return ErrorEnvelope::unexpected(
            vdb_connection_code(),
            format!("Milvus REST connection failed: {error}"),
            ErrorClass::Retriable,
        )
        .with_metadata("provider", ctx.provider.as_str())
        .with_metadata("operation", ctx.operation);
    }

    ErrorEnvelope::unexpected(
        vdb_unknown_code(),
        format!("Milvus REST request failed: {error}"),
        ErrorClass::NonRetriable,
    )
    .with_metadata("provider", ctx.provider.as_str())
    .with_metadata("operation", ctx.operation)
}

#[cfg(feature = "milvus-grpc")]
/// Maps Milvus status error codes into shared error envelopes.
pub fn map_status_error(
    code: crate::vectordb::milvus::proto::common::ErrorCode,
    reason: &str,
    ctx: &MilvusErrorContext,
) -> ErrorEnvelope {
    let error_code = match code {
        crate::vectordb::milvus::proto::common::ErrorCode::PermissionDenied => vdb_auth_code(),
        crate::vectordb::milvus::proto::common::ErrorCode::ConnectFailed => vdb_connection_code(),
        crate::vectordb::milvus::proto::common::ErrorCode::IllegalArgument
        | crate::vectordb::milvus::proto::common::ErrorCode::IllegalTopk
        | crate::vectordb::milvus::proto::common::ErrorCode::IllegalCollectionName => {
            vdb_query_invalid_code()
        },
        crate::vectordb::milvus::proto::common::ErrorCode::IllegalDimension
        | crate::vectordb::milvus::proto::common::ErrorCode::IllegalIndexType
        | crate::vectordb::milvus::proto::common::ErrorCode::IllegalMetricType => {
            vdb_schema_mismatch_code()
        },
        _ => vdb_unknown_code(),
    };

    let class = match error_code.code() {
        "vdb_timeout" | "vdb_connection" => ErrorClass::Retriable,
        _ => ErrorClass::NonRetriable,
    };

    let envelope = ErrorEnvelope::unexpected(error_code, reason.to_owned(), class)
        .with_metadata("provider", ctx.provider.as_str())
        .with_metadata("operation", ctx.operation)
        .with_metadata("status_code", format!("{code:?}"));

    if let Some(collection) = ctx.collection_name.as_ref() {
        return envelope.with_metadata("collection", collection.to_owned());
    }

    envelope
}

fn choose_code_from_message(message: &str, http_status: Option<u16>) -> ErrorCode {
    if message
        .to_ascii_lowercase()
        .contains("exceeded the limit number of collections")
    {
        return vdb_collection_limit_code();
    }

    if let Some(status) = http_status {
        if status == 401 || status == 403 {
            return vdb_auth_code();
        }
        if status == 408 || status == 504 {
            return vdb_timeout_code();
        }
    }

    if message.contains("timeout") {
        return vdb_timeout_code();
    }
    if message.contains("unauthorized") || message.contains("forbidden") || message.contains("auth")
    {
        return vdb_auth_code();
    }
    if message.contains("schema") || message.contains("datatype") || message.contains("field") {
        return vdb_schema_mismatch_code();
    }
    if message.contains("expr") || message.contains("filter") || message.contains("query") {
        return vdb_query_invalid_code();
    }

    vdb_unknown_code()
}

fn vdb_auth_code() -> ErrorCode {
    ErrorCode::new("vector", "vdb_auth")
}

fn vdb_timeout_code() -> ErrorCode {
    ErrorCode::new("vector", "vdb_timeout")
}

fn vdb_connection_code() -> ErrorCode {
    ErrorCode::new("vector", "vdb_connection")
}

fn vdb_schema_mismatch_code() -> ErrorCode {
    ErrorCode::new("vector", "vdb_schema_mismatch")
}

fn vdb_query_invalid_code() -> ErrorCode {
    ErrorCode::new("vector", "vdb_query_invalid")
}

fn vdb_collection_limit_code() -> ErrorCode {
    ErrorCode::new("vector", "vdb_collection_limit")
}

#[cfg(all(test, feature = "milvus-grpc"))]
mod tests {
    use super::*;

    #[test]
    fn grpc_error_maps_auth() {
        let status = tonic::Status::unauthenticated("unauth");
        let ctx = MilvusErrorContext {
            provider: MilvusProviderId::Grpc,
            operation: "milvus_grpc.search",
            collection_name: None,
            endpoint: None,
        };
        let envelope = map_grpc_error(&status, &ctx);
        assert_eq!(envelope.code, ErrorCode::new("vector", "vdb_auth"));
        assert_eq!(
            envelope.metadata.get("provider").map(String::as_str),
            Some("milvus_grpc")
        );
    }
}

fn vdb_unknown_code() -> ErrorCode {
    ErrorCode::new("vector", "vdb_unknown")
}
