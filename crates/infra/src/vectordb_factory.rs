//! Vector DB adapter selection and initialization.

use crate::InfraResult;
use semantic_code_adapters::vectordb::fixed::FixedDimensionVectorDb;
use semantic_code_adapters::vectordb_local::LocalVectorDb;
use semantic_code_config::{SnapshotStorageMode, ValidatedBackendConfig};
use semantic_code_ports::VectorDbPort;
use semantic_code_shared::{ErrorCode, ErrorEnvelope};
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "milvus-grpc")]
use semantic_code_adapters::vectordb::milvus::{MilvusGrpcConfig, MilvusGrpcVectorDb};
#[cfg(any(feature = "milvus-grpc", feature = "milvus-rest"))]
use semantic_code_adapters::vectordb::milvus::{MilvusIndexConfig, MilvusIndexSpec};
#[cfg(feature = "milvus-rest")]
use semantic_code_adapters::vectordb::milvus::{MilvusRestConfig, MilvusRestVectorDb};
#[cfg(any(feature = "milvus-grpc", feature = "milvus-rest"))]
use semantic_code_config::schema::{VectorDbConfig, VectorDbIndexConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderKind {
    Local,
    MilvusGrpc,
    MilvusRest,
}

/// Build a vector DB port using config settings.
pub async fn build_vectordb_port(
    config: &ValidatedBackendConfig,
    codebase_root: &Path,
    snapshot_storage: SnapshotStorageMode,
) -> InfraResult<Arc<dyn VectorDbPort>> {
    let provider = parse_provider(config.vector_db.provider.as_deref())?;
    match provider {
        ProviderKind::Local => {
            let adapter = LocalVectorDb::new(codebase_root.to_path_buf(), snapshot_storage)?;
            Ok(wrap_vectordb_fixed(config.embedding.dimension, adapter))
        },
        ProviderKind::MilvusGrpc => build_milvus_grpc(config).await,
        ProviderKind::MilvusRest => build_milvus_rest(config),
    }
}

fn parse_provider(value: Option<&str>) -> InfraResult<ProviderKind> {
    let raw = value.unwrap_or("local").trim();
    let normalized = raw.to_ascii_lowercase();
    match normalized.as_str() {
        "local" => Ok(ProviderKind::Local),
        "milvus" | "milvus_grpc" | "milvus-grpc" | "grpc" => Ok(ProviderKind::MilvusGrpc),
        "milvus_rest" | "milvus-rest" | "rest" => Ok(ProviderKind::MilvusRest),
        _ => Err(ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            format!("unsupported vector DB provider: {raw}"),
        )
        .with_metadata("provider", raw.to_string())),
    }
}

#[cfg(feature = "milvus-grpc")]
async fn build_milvus_grpc(config: &ValidatedBackendConfig) -> InfraResult<Arc<dyn VectorDbPort>> {
    let address = resolve_address(&config.vector_db)?;
    let address_for_error = address.to_string();
    let index_config = build_milvus_index_config(&config.vector_db.index);
    let adapter = MilvusGrpcVectorDb::new(MilvusGrpcConfig {
        address,
        token: config
            .vector_db
            .token
            .as_deref()
            .map(|value| value.to_owned().into_boxed_str()),
        username: config
            .vector_db
            .username
            .as_deref()
            .map(|value| value.to_owned().into_boxed_str()),
        password: config
            .vector_db
            .password
            .as_deref()
            .map(|value| value.to_owned().into_boxed_str()),
        ssl: config.vector_db.ssl,
        database: config
            .vector_db
            .database
            .as_deref()
            .map(|value| value.to_owned().into_boxed_str()),
        timeout_ms: config.vector_db.timeout_ms,
        index_timeout_ms: config.vector_db.index_timeout_ms,
        index_config,
    })
    .await
    .map_err(|error| enrich_milvus_connection_error(error, &address_for_error))?;
    Ok(wrap_vectordb_fixed(config.embedding.dimension, adapter))
}

#[cfg(not(feature = "milvus-grpc"))]
async fn build_milvus_grpc(_config: &ValidatedBackendConfig) -> InfraResult<Arc<dyn VectorDbPort>> {
    std::future::ready(Err(ErrorEnvelope::expected(
        ErrorCode::invalid_input(),
        "milvus-grpc adapter is not enabled in this build",
    )))
    .await
}

#[cfg(feature = "milvus-rest")]
fn build_milvus_rest(config: &ValidatedBackendConfig) -> InfraResult<Arc<dyn VectorDbPort>> {
    let address = resolve_address(&config.vector_db)?;
    let address_for_error = address.to_string();
    let index_config = build_milvus_index_config(&config.vector_db.index);
    let adapter = MilvusRestVectorDb::new(MilvusRestConfig {
        address,
        token: config
            .vector_db
            .token
            .as_deref()
            .map(|value| value.to_owned().into_boxed_str()),
        username: config
            .vector_db
            .username
            .as_deref()
            .map(|value| value.to_owned().into_boxed_str()),
        password: config
            .vector_db
            .password
            .as_deref()
            .map(|value| value.to_owned().into_boxed_str()),
        database: config
            .vector_db
            .database
            .as_deref()
            .map(|value| value.to_owned().into_boxed_str()),
        timeout_ms: config.vector_db.timeout_ms,
        index_config,
    })
    .map_err(|error| enrich_milvus_connection_error(error, &address_for_error))?;
    Ok(wrap_vectordb_fixed(config.embedding.dimension, adapter))
}

#[cfg(not(feature = "milvus-rest"))]
fn build_milvus_rest(_config: &ValidatedBackendConfig) -> InfraResult<Arc<dyn VectorDbPort>> {
    Err(ErrorEnvelope::expected(
        ErrorCode::invalid_input(),
        "milvus-rest adapter is not enabled in this build",
    ))
}

#[cfg(any(feature = "milvus-grpc", feature = "milvus-rest"))]
fn resolve_address(config: &VectorDbConfig) -> InfraResult<Box<str>> {
    // TODO: refactor repeated Option selection with a mapper helper.
    if let Some(address) = config.address.as_deref() {
        return Ok(address.to_owned().into_boxed_str());
    }
    if let Some(base_url) = config.base_url.as_deref() {
        return Ok(base_url.to_owned().into_boxed_str());
    }
    Err(ErrorEnvelope::expected(
        ErrorCode::invalid_input(),
        "vector DB address is required",
    ))
}

#[cfg(any(feature = "milvus-grpc", feature = "milvus-rest"))]
fn build_milvus_index_config(config: &VectorDbIndexConfig) -> MilvusIndexConfig {
    MilvusIndexConfig {
        dense: MilvusIndexSpec {
            index_type: config.dense.index_type.clone(),
            metric_type: config.dense.metric_type.clone(),
            params: config.dense.params.clone(),
        },
        sparse: MilvusIndexSpec {
            index_type: config.sparse.index_type.clone(),
            metric_type: config.sparse.metric_type.clone(),
            params: config.sparse.params.clone(),
        },
    }
}

const FIXED_VECTOR_DIMENSIONS: &[u32] = &[8, 384, 768, 1024, 1536];

fn wrap_vectordb_fixed<P: VectorDbPort + 'static>(
    dimension: Option<u32>,
    port: P,
) -> Arc<dyn VectorDbPort> {
    let dimension = dimension.filter(|value| FIXED_VECTOR_DIMENSIONS.contains(value));
    match dimension {
        Some(8) => Arc::new(FixedDimensionVectorDb::<P, 8>::new(port)),
        Some(384) => Arc::new(FixedDimensionVectorDb::<P, 384>::new(port)),
        Some(768) => Arc::new(FixedDimensionVectorDb::<P, 768>::new(port)),
        Some(1024) => Arc::new(FixedDimensionVectorDb::<P, 1024>::new(port)),
        Some(1536) => Arc::new(FixedDimensionVectorDb::<P, 1536>::new(port)),
        _ => Arc::new(port),
    }
}

#[cfg(any(feature = "milvus-grpc", feature = "milvus-rest"))]
fn enrich_milvus_connection_error(error: ErrorEnvelope, address: &str) -> ErrorEnvelope {
    if error.code == ErrorCode::new("vector", "vdb_connection") {
        let message = format!(
            "Milvus is not reachable at {address}. Start it with `just milvus-up` or `docker compose up -d`. Original error: {}",
            error.message
        );
        return ErrorEnvelope { message, ..error };
    }
    error
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_adapters::self_check::SelfCheckVectorDb;
    use semantic_code_domain::CollectionName;
    use semantic_code_shared::{ErrorCode, ErrorEnvelope, RequestContext};

    #[tokio::test]
    async fn fixed_vectordb_wrapper_rejects_mismatched_dimension() -> InfraResult<()> {
        let ctx = RequestContext::new_request();
        let inner = SelfCheckVectorDb::new()?;
        let port = wrap_vectordb_fixed(Some(768), inner);
        let name = CollectionName::parse("code_chunks_test").map_err(ErrorEnvelope::from)?;
        let result = port.create_collection(&ctx, name, 512, None).await;
        assert!(matches!(result, Err(error) if error.code == ErrorCode::invalid_input()));
        Ok(())
    }
}
