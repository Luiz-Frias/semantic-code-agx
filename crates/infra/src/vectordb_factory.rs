//! Vector DB adapter selection and initialization.

use crate::InfraResult;
use crate::cli_calibration::read_calibration;
use semantic_code_adapters::{FixedDimensionVectorDb, LocalVectorDb};
#[cfg(feature = "experimental-dfrr-kernel")]
use semantic_code_config::DfrrQueryStrategy;
use semantic_code_config::{
    DfrrBq1Threshold, DfrrSearchConfig, HnswSearchConfig, SnapshotStorageMode,
    ValidatedBackendConfig, VectorKernelKind, VectorSearchStrategy,
};
use semantic_code_ports::VectorDbPort;
use semantic_code_shared::{ErrorCode, ErrorEnvelope};
use semantic_code_vector::{FlatScanKernel, HnswKernel, VectorKernel};
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "milvus-grpc")]
use semantic_code_adapters::{MilvusGrpcConfig, MilvusGrpcVectorDb};
#[cfg(any(feature = "milvus-grpc", feature = "milvus-rest"))]
use semantic_code_adapters::{MilvusIndexConfig, MilvusIndexSpec};
#[cfg(feature = "milvus-rest")]
use semantic_code_adapters::{MilvusRestConfig, MilvusRestVectorDb};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderKind {
    Local,
    MilvusGrpc,
    MilvusRest,
}

/// Build a vector DB port using config settings.
#[tracing::instrument(
    name = "vectordb.factory",
    skip_all,
    fields(
        provider = config.vector_db.provider.as_deref().unwrap_or("local"),
        snapshot_storage = ?snapshot_storage,
        kernel = tracing::field::Empty,
    )
)]
pub async fn build_vectordb_port(
    config: &ValidatedBackendConfig,
    codebase_root: &Path,
    snapshot_storage: SnapshotStorageMode,
) -> InfraResult<Arc<dyn VectorDbPort>> {
    let provider = parse_provider(config.vector_db.provider.as_deref())?;
    match provider {
        ProviderKind::Local => {
            let search_strategy = config.vector_db.effective_search_strategy();
            let kernel_kind = config.vector_db.effective_vector_kernel();
            tracing::Span::current().record("kernel", tracing::field::debug(&kernel_kind));
            tracing::info!(
                kernel = ?kernel_kind,
                selection = if config.vector_db.vector_kernel.is_some() { "explicit" } else { "default" },
                "kernel.selected"
            );
            // Inject calibrated BQ1 threshold when available.
            let dfrr_calibration_override: Option<DfrrSearchConfig>;
            let effective_dfrr_search = if kernel_kind == VectorKernelKind::Dfrr {
                dfrr_calibration_override = inject_calibrated_bq1_threshold(
                    config.vector_db.dfrr_search.as_ref(),
                    config.embedding.dimension,
                    codebase_root,
                );
                dfrr_calibration_override
                    .as_ref()
                    .or(config.vector_db.dfrr_search.as_ref())
            } else {
                config.vector_db.dfrr_search.as_ref()
            };
            let kernel = build_local_kernel(
                kernel_kind,
                config.vector_db.hnsw_search.as_ref(),
                effective_dfrr_search,
            )?;
            if kernel_kind != VectorKernelKind::HnswRs {
                tracing::warn!(
                    kernel = ?kernel_kind,
                    "vectorDb.vectorKernel uses an experimental local kernel"
                );
            }
            if search_strategy != VectorSearchStrategy::F32Hnsw {
                tracing::warn!(
                    strategy = ?search_strategy,
                    "vectorDb.searchStrategy uses an experimental local search path"
                );
            }
            let adapter = LocalVectorDb::new(
                codebase_root.to_path_buf(),
                snapshot_storage,
                config.vector_db.snapshot_format,
                config.vector_db.snapshot_max_bytes,
                kernel,
                config.vector_db.force_reindex_on_kernel_change,
                search_strategy,
            )?;
            Ok(wrap_vectordb_fixed(config.embedding.dimension, adapter))
        },
        ProviderKind::MilvusGrpc => build_milvus_grpc(config).await,
        ProviderKind::MilvusRest => build_milvus_rest(config),
    }
}

/// Build a concrete `VectorKernel` trait object from the config-level kernel kind.
pub fn build_local_kernel(
    kind: VectorKernelKind,
    hnsw_search: Option<&HnswSearchConfig>,
    dfrr_search: Option<&DfrrSearchConfig>,
) -> InfraResult<Arc<dyn VectorKernel + Send + Sync>> {
    match kind {
        VectorKernelKind::HnswRs => {
            let kernel = hnsw_search.map_or_else(HnswKernel::new, |config| {
                HnswKernel::with_ef_search(config.ef_search as usize)
            });
            Ok(Arc::new(kernel))
        },
        VectorKernelKind::Dfrr => build_dfrr_kernel(dfrr_search),
        VectorKernelKind::FlatScan => Ok(Arc::new(FlatScanKernel)),
    }
}

#[cfg(feature = "experimental-dfrr-kernel")]
fn build_dfrr_kernel(
    dfrr_search: Option<&DfrrSearchConfig>,
) -> InfraResult<Arc<dyn VectorKernel + Send + Sync>> {
    use semantic_code_dfrr_hnsw::{
        Bq1Threshold, ClusterCount, ClusteringConfig, DfrrConfig, DfrrKernel, DfrrKernelConfig,
        DfrrLoopConfig, EfSearch, FrontierConfig, MaxIterations, PivotConfig,
        QueryRankStrategyKind,
    };

    let config = if let Some(search) = dfrr_search {
        let query_strategy = match search.query_strategy {
            DfrrQueryStrategy::Static => QueryRankStrategyKind::Static,
            DfrrQueryStrategy::NearestCentroid => QueryRankStrategyKind::NearestCentroid,
            DfrrQueryStrategy::NearestCentroidMultiProbe => {
                QueryRankStrategyKind::NearestCentroidMultiProbe {
                    probe_count: search.query_probe_count,
                    min_cluster_size: search.query_min_cluster_size,
                }
            },
        };

        let max_iterations = MaxIterations::try_from(search.max_iterations).map_err(|message| {
            ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                format!("invalid vectorDb.dfrrSearch.maxIterations: {message}"),
            )
            .with_metadata("field", "vectorDb.dfrrSearch.maxIterations")
            .with_metadata("value", search.max_iterations.to_string())
        })?;
        let ef_search = EfSearch::try_from(search.ef_search).map_err(|message| {
            ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                format!("invalid vectorDb.dfrrSearch.efSearch: {message}"),
            )
            .with_metadata("field", "vectorDb.dfrrSearch.efSearch")
            .with_metadata("value", search.ef_search.to_string())
        })?;
        let bq1_threshold = search
            .bq1_threshold
            .map(DfrrBq1Threshold::into_inner)
            .map(Bq1Threshold::try_from)
            .transpose()
            .map_err(|message| {
                ErrorEnvelope::expected(
                    ErrorCode::invalid_input(),
                    format!("invalid vectorDb.dfrrSearch.bq1Threshold: {message}"),
                )
            })?;
        let cluster_count = ClusterCount::try_from(search.cluster_count).map_err(|message| {
            ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                format!("invalid vectorDb.dfrrSearch.clusterCount: {message}"),
            )
            .with_metadata("field", "vectorDb.dfrrSearch.clusterCount")
            .with_metadata("value", search.cluster_count.to_string())
        })?;

        DfrrKernelConfig {
            loop_config: DfrrLoopConfig {
                frontier: FrontierConfig {
                    bucket_count: search.bucket_count,
                    pull_size: search.pull_size,
                    split_threshold: search.split_threshold,
                    adaptive_bucket_widths: search.adaptive_bucket_widths,
                    refine_after_first_pull: search.refine_after_first_pull,
                },
                pivots: PivotConfig {
                    pivot_count: search.pivot_count,
                    base_level_pivot_multiplier: search.base_level_pivot_multiplier,
                },
                max_iterations,
                ef_search,
                enable_exhaustion_guard: search.enable_exhaustion_guard,
                bq1_threshold,
            },
            graph_config: DfrrConfig {
                ef_search: search.ef_search,
                ..DfrrConfig::default()
            },
            clustering: ClusteringConfig {
                cluster_count,
                query_strategy,
            },
        }
    } else {
        DfrrKernelConfig::default()
    };
    Ok(Arc::new(DfrrKernel::new(config)))
}

#[cfg(not(feature = "experimental-dfrr-kernel"))]
fn build_dfrr_kernel(
    _dfrr_search: Option<&DfrrSearchConfig>,
) -> InfraResult<Arc<dyn VectorKernel + Send + Sync>> {
    Err(ErrorEnvelope::expected(
        ErrorCode::new("vector", "kernel_unsupported"),
        "DFRR kernel requires the experimental-dfrr-kernel feature flag",
    )
    .with_metadata("requestedKernel", "dfrr"))
}

/// Attempt to inject a calibrated BQ1 threshold from `.context/calibration.json`.
///
/// Returns a modified `DfrrSearchConfig` if:
/// 1. A calibration file exists at the codebase root.
/// 2. The calibrated dimension matches the current embedding dimension.
///
/// If either condition fails, returns `None` (fallback to user config or defaults).
fn inject_calibrated_bq1_threshold(
    dfrr_search: Option<&DfrrSearchConfig>,
    embedding_dimension: Option<u32>,
    codebase_root: &Path,
) -> Option<DfrrSearchConfig> {
    let state = match read_calibration(codebase_root) {
        Ok(Some(state)) => state,
        Ok(None) => return None,
        Err(error) => {
            tracing::warn!(
                error = %error,
                "failed to read calibration file — using config defaults"
            );
            return None;
        },
    };

    // Dimension guard: skip if calibration was performed with a different dimension.
    if let Some(dim) = embedding_dimension
        && dim != state.dimension
    {
        tracing::warn!(
            calibrated_dimension = state.dimension,
            config_dimension = dim,
            "calibration dimension mismatch — ignoring calibrated threshold"
        );
        return None;
    }

    let mut config = dfrr_search.copied().unwrap_or_default();
    config.bq1_threshold = Some(DfrrBq1Threshold::new(state.threshold));

    tracing::info!(
        threshold = format_args!("{:.4}", state.threshold),
        recall = format_args!("{:.4}", state.recall_at_threshold),
        skip_rate = format_args!("{:.4}", state.skip_rate),
        "using calibrated BQ1 threshold"
    );

    Some(config)
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
    let address = resolve_address(config)?;
    let address_for_error = address.to_string();
    let index_config = build_milvus_index_config(config);
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
    let address = resolve_address(config)?;
    let address_for_error = address.to_string();
    let index_config = build_milvus_index_config(config);
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
fn resolve_address(config: &ValidatedBackendConfig) -> InfraResult<Box<str>> {
    let vector_db = &config.vector_db;
    // TODO: refactor repeated Option selection with a mapper helper.
    if let Some(address) = vector_db.address.as_deref() {
        return Ok(address.to_owned().into_boxed_str());
    }
    if let Some(base_url) = vector_db.base_url.as_deref() {
        return Ok(base_url.to_owned().into_boxed_str());
    }
    Err(ErrorEnvelope::expected(
        ErrorCode::invalid_input(),
        "vector DB address is required",
    ))
}

#[cfg(any(feature = "milvus-grpc", feature = "milvus-rest"))]
fn build_milvus_index_config(config: &ValidatedBackendConfig) -> MilvusIndexConfig {
    let index = &config.vector_db.index;
    MilvusIndexConfig {
        dense: MilvusIndexSpec {
            index_type: index.dense.index_type.clone(),
            metric_type: index.dense.metric_type.clone(),
            params: index.dense.params.clone(),
        },
        sparse: MilvusIndexSpec {
            index_type: index.sparse.index_type.clone(),
            metric_type: index.sparse.metric_type.clone(),
            params: index.sparse.params.clone(),
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
    use semantic_code_adapters::SelfCheckVectorDb;
    use semantic_code_config::DfrrQueryStrategy;
    use semantic_code_domain::CollectionName;
    use semantic_code_shared::{ErrorCode, ErrorEnvelope, RequestContext};
    use semantic_code_vector::VectorKernelKind as VectorKernelKindEnum;

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

    #[test]
    fn build_local_kernel_hnsw_rs_always_succeeds() {
        let kernel = build_local_kernel(VectorKernelKind::HnswRs, None, None);
        assert!(kernel.is_ok());
        assert_eq!(
            kernel.as_ref().map(|k| k.kind()).ok(),
            Some(VectorKernelKindEnum::HnswRs)
        );
    }

    #[test]
    fn build_local_kernel_flat_scan_always_succeeds() {
        let kernel = build_local_kernel(VectorKernelKind::FlatScan, None, None);
        assert!(kernel.is_ok());
        assert_eq!(
            kernel.as_ref().map(|k| k.kind()).ok(),
            Some(VectorKernelKindEnum::FlatScan)
        );
    }

    #[test]
    fn build_local_kernel_dfrr_depends_on_feature() {
        let dfrr_search = DfrrSearchConfig {
            ef_search: 96,
            pivot_count: 8,
            base_level_pivot_multiplier: 3,
            bucket_count: 10,
            pull_size: 6,
            split_threshold: 48,
            adaptive_bucket_widths: false,
            refine_after_first_pull: false,
            max_iterations: 384,
            enable_exhaustion_guard: false,
            cluster_count: 6,
            query_strategy: DfrrQueryStrategy::NearestCentroidMultiProbe,
            query_probe_count: 4,
            query_min_cluster_size: 12,
            bq1_threshold: Some(DfrrBq1Threshold::new(0.30)),
            auto_calibrate: false,
        };

        let result = build_local_kernel(VectorKernelKind::Dfrr, None, Some(&dfrr_search));
        if cfg!(feature = "experimental-dfrr-kernel") {
            assert!(result.is_ok());
            assert_eq!(
                result.as_ref().map(|k| k.kind()).ok(),
                Some(VectorKernelKindEnum::Dfrr)
            );
        } else {
            assert!(result.is_err());
        }
    }
}
