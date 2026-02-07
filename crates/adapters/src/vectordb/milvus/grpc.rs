//! Milvus gRPC adapter.

use crate::vectordb::milvus::error::{
    MilvusErrorContext, MilvusProviderId, map_grpc_error, map_status_error,
};
use crate::vectordb::milvus::metadata::{metadata_from_fields, serialize_metadata};
use crate::vectordb::milvus::proto::common::{
    ErrorCode as ProtoErrorCode, KeyValuePair, LoadState, MsgBase, MsgType, PlaceholderGroup,
    PlaceholderType, PlaceholderValue,
};
use crate::vectordb::milvus::proto::milvus::milvus_service_client::MilvusServiceClient;
use crate::vectordb::milvus::proto::milvus::{
    CreateCollectionRequest, CreateIndexRequest, DeleteRequest, DescribeCollectionRequest,
    DropCollectionRequest, GetIndexBuildProgressRequest, GetLoadStateRequest, HasCollectionRequest,
    HybridSearchRequest, InsertRequest, LoadCollectionRequest, QueryRequest, SearchRequest,
    ShowCollectionsRequest, ShowType,
};
use crate::vectordb::milvus::proto::schema::{
    DataType, FieldData, FloatArray, LongArray, ScalarField, SearchResultData, StringArray,
    VectorField, field_data,
};
use crate::vectordb::milvus::schema::{
    MilvusSchemaSpec, build_dense_schema_spec, build_grpc_schema, build_hybrid_schema_spec,
};
use crate::vectordb::milvus::shared::{
    DEFAULT_COLLECTION_DESCRIPTION, DEFAULT_HYBRID_COLLECTION_DESCRIPTION, DEFAULT_SPARSE_FIELD,
    DEFAULT_VECTOR_FIELD, MILVUS_OUTPUT_FIELDS, ensure_collection_name, milvus_in_string,
};
use crate::vectordb::milvus::{MilvusIndexConfig, MilvusIndexSpec};
use base64::Engine;
use base64::engine::general_purpose;
use bytes::BytesMut;
use prost::Message;
use semantic_code_domain::{CollectionName, VectorDbProviderId};
use semantic_code_ports::{
    CollectionName as PortsCollectionName, HybridSearchBatchRequest, HybridSearchData,
    HybridSearchOptions, HybridSearchRequest as PortsHybridSearchRequest, HybridSearchResult,
    VectorDbPort, VectorDbProviderInfo, VectorDbRow, VectorDocument, VectorDocumentForInsert,
    VectorSearchRequest, VectorSearchResult,
};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use std::collections::BTreeMap;
use std::future::Future;
use std::str::FromStr;
use std::time::Duration;
use tonic::codegen::InterceptedService;
use tonic::metadata::AsciiMetadataValue;
use tonic::service::Interceptor;
use tonic::transport::{Channel, ClientTlsConfig, Endpoint};

/// Milvus gRPC adapter configuration.
#[derive(Debug, Clone)]
pub struct MilvusGrpcConfig {
    /// Host and port for the Milvus gRPC service.
    pub address: Box<str>,
    /// Optional bearer token for authentication.
    pub token: Option<Box<str>>,
    /// Optional username for basic authentication.
    pub username: Option<Box<str>>,
    /// Optional password for basic authentication.
    pub password: Option<Box<str>>,
    /// Whether to enable TLS when connecting.
    pub ssl: bool,
    /// Optional database name header.
    pub database: Option<Box<str>>,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Index build timeout in milliseconds.
    pub index_timeout_ms: u64,
    /// Index configuration (dense + sparse).
    pub index_config: MilvusIndexConfig,
}

impl MilvusGrpcConfig {
    /// Validates configuration invariants for the gRPC adapter.
    pub fn validate(&self) -> Result<()> {
        if self.address.trim().is_empty() {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "Milvus gRPC address is required",
            ));
        }
        if self.timeout_ms == 0 {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "Milvus timeout must be greater than zero",
            ));
        }
        if self.index_timeout_ms == 0 {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "Milvus index timeout must be greater than zero",
            ));
        }
        Ok(())
    }
}

#[derive(Clone)]
struct AuthInterceptor {
    auth_header: Option<AsciiMetadataValue>,
    db_name: Option<AsciiMetadataValue>,
}

impl Interceptor for AuthInterceptor {
    fn call(
        &mut self,
        mut req: tonic::Request<()>,
    ) -> std::result::Result<tonic::Request<()>, tonic::Status> {
        if let Some(header) = self.auth_header.clone() {
            req.metadata_mut().insert("authorization", header);
        }
        if let Some(db_name) = self.db_name.clone() {
            req.metadata_mut().insert("dbname", db_name);
        }
        Ok(req)
    }
}

/// Milvus gRPC vector DB adapter.
#[derive(Clone)]
pub struct MilvusGrpcVectorDb {
    provider: VectorDbProviderInfo,
    client: MilvusServiceClient<InterceptedService<Channel, AuthInterceptor>>,
    timeout: Duration,
    index_timeout: Duration,
    index_config: MilvusIndexConfig,
    db_name: Option<Box<str>>,
}

impl MilvusGrpcVectorDb {
    /// Creates a Milvus gRPC adapter instance from configuration.
    pub async fn new(config: MilvusGrpcConfig) -> Result<Self> {
        config.validate()?;
        let address = normalize_grpc_address(&config.address, config.ssl);
        let mut endpoint = Endpoint::from_shared(address.clone()).map_err(|error| {
            ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                format!("invalid Milvus address: {error}"),
            )
        })?;
        endpoint = endpoint.timeout(Duration::from_millis(config.timeout_ms));
        if config.ssl {
            let tls = ClientTlsConfig::new();
            endpoint = endpoint.tls_config(tls).map_err(|error| {
                ErrorEnvelope::expected(
                    ErrorCode::invalid_input(),
                    format!("invalid TLS config: {error}"),
                )
            })?;
        }

        let channel = endpoint.connect().await.map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "vdb_connection"),
                format!("failed to connect to Milvus gRPC: {error}"),
                ErrorClass::Retriable,
            )
        })?;

        let auth_header = build_auth_header(&config)?;
        let db_name = build_db_header(&config)?;
        let interceptor = AuthInterceptor {
            auth_header,
            db_name,
        };
        let client = MilvusServiceClient::with_interceptor(channel, interceptor);

        let provider = VectorDbProviderInfo {
            id: VectorDbProviderId::parse("milvus_grpc").map_err(ErrorEnvelope::from)?,
            name: "Milvus gRPC".into(),
        };

        Ok(Self {
            provider,
            client,
            timeout: Duration::from_millis(config.timeout_ms),
            index_timeout: Duration::from_millis(config.index_timeout_ms),
            index_config: config.index_config,
            db_name: config.database,
        })
    }

    fn context(operation: &'static str, collection: Option<&CollectionName>) -> MilvusErrorContext {
        MilvusErrorContext {
            provider: MilvusProviderId::Grpc,
            operation,
            collection_name: collection.map(|c| c.as_str().to_owned()),
            endpoint: None,
        }
    }

    async fn ensure_loaded(&self, ctx: &RequestContext, collection: &CollectionName) -> Result<()> {
        let operation = "milvus_grpc.ensure_loaded";
        ctx.ensure_not_cancelled(operation)?;
        let request = GetLoadStateRequest {
            base: None,
            collection_name: collection.as_str().to_owned(),
            partition_names: Vec::new(),
            db_name: self.db_name.clone().unwrap_or_default().into(),
        };

        let response = self
            .call_with_timeout(
                ctx,
                operation,
                Some(collection),
                self.client.clone().get_load_state(request),
            )
            .await?;
        if let Some(status) = response.status.as_ref() {
            ensure_status_ok(status, &Self::context(operation, Some(collection)))?;
        }

        if response.state == LoadState::Loaded as i32 {
            return Ok(());
        }

        let load_request = LoadCollectionRequest {
            base: Some(MsgBase::new(MsgType::LoadCollection)),
            db_name: self.db_name.clone().unwrap_or_default().into(),
            collection_name: collection.as_str().to_owned(),
            replica_number: 1,
            resource_groups: Vec::new(),
            refresh: false,
            load_fields: Vec::new(),
            skip_load_dynamic_field: false,
            load_params: std::collections::HashMap::new(),
        };

        let response = self
            .call_with_timeout(
                ctx,
                "milvus_grpc.load_collection",
                Some(collection),
                self.client.clone().load_collection(load_request),
            )
            .await?;
        ensure_status_ok(
            &response,
            &Self::context("milvus_grpc.load_collection", Some(collection)),
        )?;
        Ok(())
    }

    async fn wait_for_index(
        &self,
        ctx: &RequestContext,
        collection: &CollectionName,
        field_name: &str,
    ) -> Result<()> {
        let operation = "milvus_grpc.wait_for_index";
        let start = std::time::Instant::now();
        let max_wait = self.index_timeout;
        while start.elapsed() < max_wait {
            ctx.ensure_not_cancelled(operation)?;
            let request = GetIndexBuildProgressRequest {
                base: Some(MsgBase::new(MsgType::GetIndexBuildProgress)),
                db_name: self.db_name.clone().unwrap_or_default().into(),
                collection_name: collection.as_str().to_owned(),
                field_name: field_name.to_owned(),
                index_name: String::new(),
            };

            let response = self
                .call_with_timeout(
                    ctx,
                    operation,
                    Some(collection),
                    self.client.clone().get_index_build_progress(request),
                )
                .await?;
            if let Some(status) = response.status.as_ref() {
                ensure_status_ok(status, &Self::context(operation, Some(collection)))?;
            }

            if response.total_rows == 0 || response.indexed_rows >= response.total_rows {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        Err(ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "vdb_index_timeout"),
            "timed out waiting for Milvus index build",
            ErrorClass::Retriable,
        ))
    }

    async fn call_with_timeout<T>(
        &self,
        ctx: &RequestContext,
        operation: &'static str,
        collection: Option<&CollectionName>,
        fut: impl Future<Output = std::result::Result<tonic::Response<T>, tonic::Status>>,
    ) -> Result<T> {
        ctx.ensure_not_cancelled(operation)?;
        let timeout = self.timeout;
        let result = tokio::select! {
            () = ctx.cancelled() => return Err(cancelled_error(operation)),
            res = tokio::time::timeout(timeout, fut) => res,
        };

        result.map_or_else(
            |_| Err(timeout_error(operation)),
            |response| {
                response.map(tonic::Response::into_inner).map_err(|status| {
                    map_grpc_error(&status, &Self::context(operation, collection))
                })
            },
        )
    }

    fn build_documents(docs: Vec<VectorDocumentForInsert>) -> Result<Vec<FieldData>> {
        let mut ids = Vec::with_capacity(docs.len());
        let mut vectors = Vec::with_capacity(docs.len() * 4);
        let mut contents = Vec::with_capacity(docs.len());
        let mut relative_paths = Vec::with_capacity(docs.len());
        let mut start_lines = Vec::with_capacity(docs.len());
        let mut end_lines = Vec::with_capacity(docs.len());
        let mut extensions = Vec::with_capacity(docs.len());
        let mut metadata = Vec::with_capacity(docs.len());
        let mut dimension = None;

        for doc in docs {
            ids.push(doc.id.as_ref().to_owned());
            if dimension.is_none() {
                dimension = Some(doc.vector.len());
            }
            vectors.extend_from_slice(doc.vector.as_ref());
            contents.push(doc.content.as_ref().to_owned());
            relative_paths.push(doc.metadata.relative_path.as_ref().to_owned());
            start_lines.push(i64::from(doc.metadata.span.start_line()));
            end_lines.push(i64::from(doc.metadata.span.end_line()));
            extensions.push(
                doc.metadata
                    .file_extension
                    .as_deref()
                    .unwrap_or("")
                    .to_owned(),
            );
            metadata.push(serialize_metadata(&doc.metadata)?);
        }

        let dim = i64::try_from(dimension.unwrap_or_default()).unwrap_or(i64::MAX);

        let id_field = FieldData {
            r#type: DataType::VarChar as i32,
            field_name: "id".to_owned(),
            field_id: 0,
            is_dynamic: false,
            valid_data: Vec::new(),
            field: Some(field_data::Field::Scalars(ScalarField {
                data: Some(
                    crate::vectordb::milvus::proto::schema::scalar_field::Data::StringData(
                        StringArray { data: ids },
                    ),
                ),
            })),
        };

        let vector_field = FieldData {
            r#type: DataType::FloatVector as i32,
            field_name: "vector".to_owned(),
            field_id: 1,
            is_dynamic: false,
            valid_data: Vec::new(),
            field: Some(field_data::Field::Vectors(VectorField {
                dim,
                data: Some(
                    crate::vectordb::milvus::proto::schema::vector_field::Data::FloatVector(
                        FloatArray { data: vectors },
                    ),
                ),
            })),
        };

        let content_field = scalar_string_field("content", 2, contents);
        let relative_path_field = scalar_string_field("relativePath", 3, relative_paths);
        let start_line_field = scalar_long_field("startLine", 4, start_lines);
        let end_line_field = scalar_long_field("endLine", 5, end_lines);
        let ext_field = scalar_string_field("fileExtension", 6, extensions);
        let metadata_field = scalar_string_field("metadata", 7, metadata);

        Ok(vec![
            id_field,
            vector_field,
            content_field,
            relative_path_field,
            start_line_field,
            end_line_field,
            ext_field,
            metadata_field,
        ])
    }

    fn parse_results(data: SearchResultData) -> Result<Vec<VectorSearchResult>> {
        let k = usize::try_from(data.topks.first().copied().unwrap_or(0)).unwrap_or_default();
        if k == 0 {
            return Ok(Vec::new());
        }
        let columns = collect_fields(data.fields_data)?;
        let scores = data.scores;
        let mut results = Vec::with_capacity(k);
        for idx in 0..k {
            let doc = build_document_from_columns(&columns, idx)?;
            let score = scores.get(idx).copied().unwrap_or_default();
            results.push(VectorSearchResult {
                document: doc,
                score,
            });
        }
        Ok(results)
    }

    fn parse_hybrid_results(data: SearchResultData) -> Result<Vec<HybridSearchResult>> {
        let k = usize::try_from(data.topks.first().copied().unwrap_or(0)).unwrap_or_default();
        if k == 0 {
            return Ok(Vec::new());
        }
        let columns = collect_fields(data.fields_data)?;
        let scores = data.scores;
        let mut results = Vec::with_capacity(k);
        for idx in 0..k {
            let doc = build_document_from_columns(&columns, idx)?;
            let score = scores.get(idx).copied().unwrap_or_default();
            results.push(HybridSearchResult {
                document: doc,
                score,
            });
        }
        Ok(results)
    }
}

impl VectorDbPort for MilvusGrpcVectorDb {
    fn provider(&self) -> &VectorDbProviderInfo {
        &self.provider
    }

    fn create_collection(
        &self,
        ctx: &RequestContext,
        collection_name: PortsCollectionName,
        dimension: u32,
        description: Option<Box<str>>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let adapter = self.clone();
        Box::pin(async move {
            ensure_collection_name(&collection_name)?;
            let description = description
                .as_deref()
                .unwrap_or(DEFAULT_COLLECTION_DESCRIPTION);
            let spec = build_dense_schema_spec(dimension);
            create_collection(&ctx, &adapter, &collection_name, &spec, description).await?;
            Ok(())
        })
    }

    fn create_hybrid_collection(
        &self,
        ctx: &RequestContext,
        collection_name: PortsCollectionName,
        dimension: u32,
        description: Option<Box<str>>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let adapter = self.clone();
        Box::pin(async move {
            ensure_collection_name(&collection_name)?;
            let description = description
                .as_deref()
                .unwrap_or(DEFAULT_HYBRID_COLLECTION_DESCRIPTION);
            let spec = build_hybrid_schema_spec(dimension);
            create_collection(&ctx, &adapter, &collection_name, &spec, description).await?;
            Ok(())
        })
    }

    fn drop_collection(
        &self,
        ctx: &RequestContext,
        collection_name: PortsCollectionName,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let adapter = self.clone();
        Box::pin(async move {
            ensure_collection_name(&collection_name)?;
            let request = DropCollectionRequest {
                base: Some(MsgBase::new(MsgType::DropCollection)),
                db_name: adapter.db_name.clone().unwrap_or_default().into(),
                collection_name: collection_name.as_str().to_owned(),
            };
            let response = adapter
                .call_with_timeout(
                    &ctx,
                    "milvus_grpc.drop_collection",
                    Some(&collection_name),
                    adapter.client.clone().drop_collection(request),
                )
                .await?;
            ensure_status_ok(
                &response,
                &Self::context("milvus_grpc.drop_collection", Some(&collection_name)),
            )?;
            Ok(())
        })
    }

    fn has_collection(
        &self,
        ctx: &RequestContext,
        collection_name: PortsCollectionName,
    ) -> semantic_code_ports::BoxFuture<'_, Result<bool>> {
        let ctx = ctx.clone();
        let adapter = self.clone();
        Box::pin(async move {
            ensure_collection_name(&collection_name)?;
            let request = HasCollectionRequest {
                base: Some(MsgBase::new(MsgType::HasCollection)),
                db_name: adapter.db_name.clone().unwrap_or_default().into(),
                collection_name: collection_name.as_str().to_owned(),
                time_stamp: 0,
            };
            let response = adapter
                .call_with_timeout(
                    &ctx,
                    "milvus_grpc.has_collection",
                    Some(&collection_name),
                    adapter.client.clone().has_collection(request),
                )
                .await?;
            Ok(response.value)
        })
    }

    fn list_collections(
        &self,
        ctx: &RequestContext,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<PortsCollectionName>>> {
        let ctx = ctx.clone();
        let adapter = self.clone();
        Box::pin(async move {
            let request = ShowCollectionsRequest {
                base: Some(MsgBase::new(MsgType::ShowCollections)),
                db_name: adapter.db_name.clone().unwrap_or_default().into(),
                time_stamp: 0,
                r#type: ShowType::All as i32,
                collection_names: Vec::new(),
            };
            let response = adapter
                .call_with_timeout(
                    &ctx,
                    "milvus_grpc.list_collections",
                    None,
                    adapter.client.clone().show_collections(request),
                )
                .await?;
            if let Some(status) = response.status.as_ref() {
                ensure_status_ok(status, &Self::context("milvus_grpc.list_collections", None))?;
            }
            let names = response.collection_names;
            let mut out = Vec::with_capacity(names.len());
            for name in names {
                if let Ok(parsed) = CollectionName::parse(name.as_str()) {
                    out.push(parsed);
                }
            }
            Ok(out)
        })
    }

    fn insert(
        &self,
        ctx: &RequestContext,
        collection_name: PortsCollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let adapter = self.clone();
        Box::pin(async move {
            ensure_collection_name(&collection_name)?;
            adapter.ensure_loaded(&ctx, &collection_name).await?;
            let fields_data = Self::build_documents(documents)?;
            let row_count = fields_data.first().map_or(0, f_len);
            let num_rows = u32::try_from(row_count).unwrap_or_default();
            let request = InsertRequest {
                base: Some(MsgBase::new(MsgType::Insert)),
                db_name: adapter.db_name.clone().unwrap_or_default().into(),
                collection_name: collection_name.as_str().to_owned(),
                partition_name: String::new(),
                num_rows,
                fields_data,
                hash_keys: Vec::new(),
                schema_timestamp: 0,
            };
            let response = adapter
                .call_with_timeout(
                    &ctx,
                    "milvus_grpc.insert",
                    Some(&collection_name),
                    adapter.client.clone().insert(request),
                )
                .await?;
            if let Some(status) = response.status.as_ref() {
                ensure_status_ok(
                    status,
                    &Self::context("milvus_grpc.insert", Some(&collection_name)),
                )?;
            }
            Ok(())
        })
    }

    fn insert_hybrid(
        &self,
        ctx: &RequestContext,
        collection_name: PortsCollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        self.insert(ctx, collection_name, documents)
    }

    fn search(
        &self,
        ctx: &RequestContext,
        request: VectorSearchRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<VectorSearchResult>>> {
        let ctx = ctx.clone();
        let adapter = self.clone();
        let VectorSearchRequest {
            collection_name,
            query_vector,
            options,
        } = request;
        Box::pin(async move {
            ensure_collection_name(&collection_name)?;
            adapter.ensure_loaded(&ctx, &collection_name).await?;
            let placeholder_group = encode_placeholder_group_float(query_vector.as_ref())?;
            let mut search_params = vec![
                KeyValuePair {
                    key: "topk".to_owned(),
                    value: options.top_k.unwrap_or(10).to_string(),
                },
                KeyValuePair {
                    key: "metric_type".to_owned(),
                    value: adapter.index_config.dense.metric_type.as_ref().to_owned(),
                },
                KeyValuePair {
                    key: "anns_field".to_owned(),
                    value: DEFAULT_VECTOR_FIELD.to_owned(),
                },
            ];
            let params_value = merge_search_params(&search_params)?;
            search_params.push(KeyValuePair {
                key: "params".to_owned(),
                value: params_value,
            });

            let request = SearchRequest {
                base: Some(MsgBase::new(MsgType::Search)),
                db_name: adapter.db_name.clone().unwrap_or_default().into(),
                collection_name: collection_name.as_str().to_owned(),
                partition_names: Vec::new(),
                dsl: options.filter_expr.clone().unwrap_or_default().into(),
                placeholder_group,
                dsl_type: crate::vectordb::milvus::proto::common::DslType::BoolExprV1 as i32,
                output_fields: milvus_output_fields(),
                search_params,
                travel_timestamp: 0,
                guarantee_timestamp: 0,
                nq: 1,
                not_return_all_meta: false,
                consistency_level: crate::vectordb::milvus::proto::common::ConsistencyLevel::Bounded
                    as i32,
                use_default_consistency: false,
                search_by_primary_keys: false,
                expr_template_values: std::collections::HashMap::new(),
                sub_reqs: Vec::new(),
                function_score: None,
            };

            let response = adapter
                .call_with_timeout(
                    &ctx,
                    "milvus_grpc.search",
                    Some(&collection_name),
                    adapter.client.clone().search(request),
                )
                .await?;
            if let Some(status) = response.status.as_ref() {
                ensure_status_ok(
                    status,
                    &Self::context("milvus_grpc.search", Some(&collection_name)),
                )?;
            }
            let data = response.results.ok_or_else(|| {
                ErrorEnvelope::unexpected(
                    ErrorCode::new("vector", "vdb_invalid_response"),
                    "missing search results",
                    ErrorClass::NonRetriable,
                )
            })?;
            Self::parse_results(data)
        })
    }

    fn hybrid_search(
        &self,
        ctx: &RequestContext,
        request: HybridSearchBatchRequest,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<HybridSearchResult>>> {
        let ctx = ctx.clone();
        let adapter = self.clone();
        let HybridSearchBatchRequest {
            collection_name,
            search_requests,
            options,
        } = request;
        Box::pin(async move {
            ensure_collection_name(&collection_name)?;
            adapter.ensure_loaded(&ctx, &collection_name).await?;
            let requests = build_hybrid_search_requests(
                &adapter,
                &collection_name,
                &search_requests,
                &options,
            )?;

            let rank_params = build_rank_params(&options);
            let request = HybridSearchRequest {
                base: None,
                db_name: adapter.db_name.clone().unwrap_or_default().into(),
                collection_name: collection_name.as_str().to_owned(),
                partition_names: Vec::new(),
                requests,
                rank_params,
                travel_timestamp: 0,
                guarantee_timestamp: 0,
                not_return_all_meta: false,
                output_fields: milvus_output_fields(),
                consistency_level: crate::vectordb::milvus::proto::common::ConsistencyLevel::Bounded
                    as i32,
                use_default_consistency: false,
                function_score: None,
            };

            let response = adapter
                .call_with_timeout(
                    &ctx,
                    "milvus_grpc.hybrid_search",
                    Some(&collection_name),
                    adapter.client.clone().hybrid_search(request),
                )
                .await?;
            if let Some(status) = response.status.as_ref() {
                ensure_status_ok(
                    status,
                    &Self::context("milvus_grpc.hybrid_search", Some(&collection_name)),
                )?;
            }

            let data = response.results.ok_or_else(|| {
                ErrorEnvelope::unexpected(
                    ErrorCode::new("vector", "vdb_invalid_response"),
                    "missing hybrid search results",
                    ErrorClass::NonRetriable,
                )
            })?;
            Self::parse_hybrid_results(data)
        })
    }

    fn delete(
        &self,
        ctx: &RequestContext,
        collection_name: PortsCollectionName,
        ids: Vec<Box<str>>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let adapter = self.clone();
        Box::pin(async move {
            ensure_collection_name(&collection_name)?;
            adapter.ensure_loaded(&ctx, &collection_name).await?;
            let expr = milvus_in_string("id", &ids);
            let request = DeleteRequest {
                base: Some(MsgBase::new(MsgType::Delete)),
                db_name: adapter.db_name.clone().unwrap_or_default().into(),
                collection_name: collection_name.as_str().to_owned(),
                partition_name: String::new(),
                expr: expr.as_ref().to_owned(),
                hash_keys: Vec::new(),
                consistency_level: crate::vectordb::milvus::proto::common::ConsistencyLevel::Bounded
                    as i32,
                expr_template_values: std::collections::HashMap::new(),
            };
            let response = adapter
                .call_with_timeout(
                    &ctx,
                    "milvus_grpc.delete",
                    Some(&collection_name),
                    adapter.client.clone().delete(request),
                )
                .await?;
            if let Some(status) = response.status.as_ref() {
                ensure_status_ok(
                    status,
                    &Self::context("milvus_grpc.delete", Some(&collection_name)),
                )?;
            }
            Ok(())
        })
    }

    fn query(
        &self,
        ctx: &RequestContext,
        collection_name: PortsCollectionName,
        filter: Box<str>,
        output_fields: Vec<Box<str>>,
        limit: Option<u32>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<VectorDbRow>>> {
        let ctx = ctx.clone();
        let adapter = self.clone();
        Box::pin(async move {
            ensure_collection_name(&collection_name)?;
            adapter.ensure_loaded(&ctx, &collection_name).await?;
            let mut params = Vec::new();
            if let Some(limit) = limit {
                params.push(KeyValuePair {
                    key: "limit".to_owned(),
                    value: limit.to_string(),
                });
            }
            let request = QueryRequest {
                base: None,
                db_name: adapter.db_name.clone().unwrap_or_default().into(),
                collection_name: collection_name.as_str().to_owned(),
                expr: filter.as_ref().to_owned(),
                output_fields: output_fields
                    .iter()
                    .map(|f| f.as_ref().to_owned())
                    .collect(),
                partition_names: Vec::new(),
                travel_timestamp: 0,
                guarantee_timestamp: 0,
                query_params: params,
                not_return_all_meta: false,
                consistency_level: crate::vectordb::milvus::proto::common::ConsistencyLevel::Bounded
                    as i32,
                use_default_consistency: false,
                expr_template_values: std::collections::HashMap::new(),
            };
            let response = adapter
                .call_with_timeout(
                    &ctx,
                    "milvus_grpc.query",
                    Some(&collection_name),
                    adapter.client.clone().query(request),
                )
                .await?;
            if let Some(status) = response.status.as_ref() {
                ensure_status_ok(
                    status,
                    &Self::context("milvus_grpc.query", Some(&collection_name)),
                )?;
            }
            let fields = response.fields_data;
            let columns = collect_fields(fields)?;
            let row_count = columns.values().next().map_or(0, FieldColumn::len);
            let mut rows = Vec::with_capacity(row_count);
            for idx in 0..row_count {
                let doc = build_row_from_columns(&columns, idx);
                rows.push(doc);
            }
            Ok(rows)
        })
    }
}

async fn create_collection(
    ctx: &RequestContext,
    adapter: &MilvusGrpcVectorDb,
    collection_name: &CollectionName,
    spec: &MilvusSchemaSpec,
    description: &str,
) -> Result<()> {
    let operation = "milvus_grpc.create_collection";
    let schema = build_grpc_schema(spec, collection_name.as_str(), description);
    let mut buf = BytesMut::new();
    schema.encode(&mut buf).map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "vdb_schema_encode"),
            format!("failed to encode Milvus schema: {error}"),
            ErrorClass::NonRetriable,
        )
    })?;

    let request = CreateCollectionRequest {
        base: Some(MsgBase::new(MsgType::CreateCollection)),
        db_name: adapter.db_name.clone().unwrap_or_default().into(),
        collection_name: collection_name.as_str().to_owned(),
        schema: buf.to_vec(),
        shards_num: 0,
        consistency_level: crate::vectordb::milvus::proto::common::ConsistencyLevel::Bounded as i32,
        properties: Vec::new(),
        num_partitions: 0,
    };

    let response = adapter
        .call_with_timeout(
            ctx,
            operation,
            Some(collection_name),
            adapter.client.clone().create_collection(request),
        )
        .await?;
    ensure_status_ok(
        &response,
        &MilvusGrpcVectorDb::context(operation, Some(collection_name)),
    )?;

    // TODO: make index type/metric configurable per deployment.
    create_index(
        ctx,
        adapter,
        collection_name,
        DEFAULT_VECTOR_FIELD,
        "vector_index",
        &adapter.index_config.dense,
    )
    .await?;
    if spec.fields.iter().any(|field| {
        matches!(
            field,
            crate::vectordb::milvus::schema::MilvusFieldSpec::SparseFloatVector { .. }
        )
    }) {
        // TODO: make sparse index type configurable per deployment.
        create_index(
            ctx,
            adapter,
            collection_name,
            DEFAULT_SPARSE_FIELD,
            "sparse_vector_index",
            &adapter.index_config.sparse,
        )
        .await?;
    }

    adapter
        .wait_for_index(ctx, collection_name, DEFAULT_VECTOR_FIELD)
        .await?;
    adapter.ensure_loaded(ctx, collection_name).await?;

    let describe_request = DescribeCollectionRequest {
        base: Some(MsgBase::new(MsgType::DescribeCollection)),
        db_name: adapter.db_name.clone().unwrap_or_default().into(),
        collection_name: collection_name.as_str().to_owned(),
        ..Default::default()
    };

    let _ = adapter
        .call_with_timeout(
            ctx,
            "milvus_grpc.describe_collection",
            Some(collection_name),
            adapter.client.clone().describe_collection(describe_request),
        )
        .await?;

    Ok(())
}

async fn create_index(
    ctx: &RequestContext,
    adapter: &MilvusGrpcVectorDb,
    collection_name: &CollectionName,
    field_name: &str,
    index_name: &str,
    spec: &MilvusIndexSpec,
) -> Result<()> {
    let mut extra_params = vec![
        KeyValuePair {
            key: "index_type".to_owned(),
            value: spec.index_type.as_ref().to_owned(),
        },
        KeyValuePair {
            key: "metric_type".to_owned(),
            value: spec.metric_type.as_ref().to_owned(),
        },
    ];
    for (key, value) in &spec.params {
        extra_params.push(KeyValuePair {
            key: key.as_ref().to_owned(),
            value: value.as_ref().to_owned(),
        });
    }
    let request = CreateIndexRequest {
        base: Some(MsgBase::new(MsgType::CreateIndex)),
        db_name: adapter.db_name.clone().unwrap_or_default().into(),
        collection_name: collection_name.as_str().to_owned(),
        field_name: field_name.to_owned(),
        extra_params,
        index_name: index_name.to_owned(),
    };

    let response = adapter
        .call_with_timeout(
            ctx,
            "milvus_grpc.create_index",
            Some(collection_name),
            adapter.client.clone().create_index(request),
        )
        .await?;
    ensure_status_ok(
        &response,
        &MilvusGrpcVectorDb::context("milvus_grpc.create_index", Some(collection_name)),
    )?;
    Ok(())
}

fn normalize_grpc_address(address: &str, ssl: bool) -> String {
    let trimmed = address.trim();
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        return trimmed.to_owned();
    }
    let scheme = if ssl { "https" } else { "http" };
    format!("{scheme}://{trimmed}")
}

fn ensure_status_ok(
    status: &crate::vectordb::milvus::proto::common::Status,
    ctx: &MilvusErrorContext,
) -> Result<()> {
    if status.code == ProtoErrorCode::Success as i32 {
        return Ok(());
    }
    let code = ProtoErrorCode::try_from(status.code).unwrap_or(ProtoErrorCode::UnexpectedError);
    Err(map_status_error(code, status.reason.as_str(), ctx))
}

fn build_auth_header(config: &MilvusGrpcConfig) -> Result<Option<AsciiMetadataValue>> {
    let token = config
        .token
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let header_value = if let Some(token) = token {
        token.to_owned()
    } else if let (Some(username), Some(password)) =
        (config.username.as_deref(), config.password.as_deref())
    {
        let token = format!("{username}:{password}");
        general_purpose::STANDARD.encode(token)
    } else {
        return Ok(None);
    };
    let value = AsciiMetadataValue::from_str(&header_value).map_err(|_| {
        ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            "Milvus auth header contains invalid characters",
        )
    })?;
    Ok(Some(value))
}

fn build_db_header(config: &MilvusGrpcConfig) -> Result<Option<AsciiMetadataValue>> {
    let db_name = config
        .database
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let Some(db_name) = db_name else {
        return Ok(None);
    };
    let value = AsciiMetadataValue::from_str(db_name).map_err(|_| {
        ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            "Milvus database name contains invalid characters",
        )
    })?;
    Ok(Some(value))
}

fn cancelled_error(operation: &'static str) -> ErrorEnvelope {
    ErrorEnvelope::cancelled("operation cancelled").with_metadata("operation", operation)
}

fn timeout_error(operation: &'static str) -> ErrorEnvelope {
    ErrorEnvelope::unexpected(
        ErrorCode::new("vector", "vdb_timeout"),
        "Milvus request timed out",
        ErrorClass::Retriable,
    )
    .with_metadata("operation", operation)
}

fn f_len(field: &FieldData) -> usize {
    match field.field.as_ref() {
        Some(field_data::Field::Scalars(scalar)) => match scalar.data.as_ref() {
            Some(crate::vectordb::milvus::proto::schema::scalar_field::Data::StringData(data)) => {
                data.data.len()
            },
            Some(crate::vectordb::milvus::proto::schema::scalar_field::Data::LongData(data)) => {
                data.data.len()
            },
            _ => 0,
        },
        Some(field_data::Field::Vectors(vector)) => match vector.data.as_ref() {
            Some(crate::vectordb::milvus::proto::schema::vector_field::Data::FloatVector(data)) => {
                let dim = match usize::try_from(vector.dim) {
                    Ok(value) if value > 0 => value,
                    _ => return 0,
                };
                data.data.len() / dim
            },
            _ => 0,
        },
        _ => 0,
    }
}

fn scalar_string_field(name: &str, field_id: i64, values: Vec<String>) -> FieldData {
    FieldData {
        r#type: DataType::VarChar as i32,
        field_name: name.to_owned(),
        field_id,
        is_dynamic: false,
        valid_data: Vec::new(),
        field: Some(field_data::Field::Scalars(ScalarField {
            data: Some(
                crate::vectordb::milvus::proto::schema::scalar_field::Data::StringData(
                    StringArray { data: values },
                ),
            ),
        })),
    }
}

fn scalar_long_field(name: &str, field_id: i64, values: Vec<i64>) -> FieldData {
    FieldData {
        r#type: DataType::Int64 as i32,
        field_name: name.to_owned(),
        field_id,
        is_dynamic: false,
        valid_data: Vec::new(),
        field: Some(field_data::Field::Scalars(ScalarField {
            data: Some(
                crate::vectordb::milvus::proto::schema::scalar_field::Data::LongData(LongArray {
                    data: values,
                }),
            ),
        })),
    }
}

fn collect_fields(fields: Vec<FieldData>) -> Result<BTreeMap<String, FieldColumn>> {
    let mut out = BTreeMap::new();
    for field in fields {
        let name = field.field_name.clone();
        let column = FieldColumn::try_from(field)?;
        out.insert(name, column);
    }
    Ok(out)
}

#[derive(Debug, Clone)]
struct FieldColumn {
    values: FieldValues,
}

#[derive(Debug, Clone)]
enum FieldValues {
    Strings(Vec<String>),
    Longs(Vec<i64>),
}

impl FieldColumn {
    const fn len(&self) -> usize {
        match &self.values {
            FieldValues::Strings(values) => values.len(),
            FieldValues::Longs(values) => values.len(),
        }
    }

    fn string_at(&self, idx: usize) -> Option<&str> {
        match &self.values {
            FieldValues::Strings(values) => values.get(idx).map(String::as_str),
            FieldValues::Longs(_) => None,
        }
    }

    fn long_at(&self, idx: usize) -> Option<i64> {
        match &self.values {
            FieldValues::Longs(values) => values.get(idx).copied(),
            FieldValues::Strings(_) => None,
        }
    }
}

impl TryFrom<FieldData> for FieldColumn {
    type Error = ErrorEnvelope;

    fn try_from(field: FieldData) -> Result<Self> {
        let values = match field.field {
            Some(field_data::Field::Scalars(scalar)) => match scalar.data {
                Some(crate::vectordb::milvus::proto::schema::scalar_field::Data::StringData(
                    data,
                )) => FieldValues::Strings(data.data),
                Some(crate::vectordb::milvus::proto::schema::scalar_field::Data::LongData(
                    data,
                )) => FieldValues::Longs(data.data),
                _ => {
                    return Err(ErrorEnvelope::unexpected(
                        ErrorCode::new("vector", "vdb_invalid_response"),
                        format!("unexpected field data for {}", field.field_name),
                        ErrorClass::NonRetriable,
                    ));
                },
            },
            _ => {
                return Err(ErrorEnvelope::unexpected(
                    ErrorCode::new("vector", "vdb_invalid_response"),
                    format!("missing scalar data for {}", field.field_name),
                    ErrorClass::NonRetriable,
                ));
            },
        };
        Ok(Self { values })
    }
}

fn build_document_from_columns(
    columns: &BTreeMap<String, FieldColumn>,
    idx: usize,
) -> Result<VectorDocument> {
    let id = columns
        .get("id")
        .and_then(|c| c.string_at(idx))
        .ok_or_else(|| missing_field("id"))?;
    let content = columns
        .get("content")
        .and_then(|c| c.string_at(idx))
        .ok_or_else(|| missing_field("content"))?;
    let relative_path = columns
        .get("relativePath")
        .and_then(|c| c.string_at(idx))
        .ok_or_else(|| missing_field("relativePath"))?;
    let start_line = columns
        .get("startLine")
        .and_then(|c| c.long_at(idx))
        .ok_or_else(|| missing_field("startLine"))?;
    let end_line = columns
        .get("endLine")
        .and_then(|c| c.long_at(idx))
        .ok_or_else(|| missing_field("endLine"))?;
    let file_extension = columns.get("fileExtension").and_then(|c| c.string_at(idx));
    let metadata_raw = columns.get("metadata").and_then(|c| c.string_at(idx));

    let metadata = metadata_from_fields(
        relative_path,
        start_line,
        end_line,
        file_extension,
        metadata_raw,
    )?;

    Ok(VectorDocument {
        id: id.to_owned().into_boxed_str(),
        vector: None,
        content: content.to_owned().into_boxed_str(),
        metadata,
    })
}

fn build_row_from_columns(columns: &BTreeMap<String, FieldColumn>, idx: usize) -> VectorDbRow {
    let mut row = VectorDbRow::new();
    for (name, column) in columns {
        if let Some(value) = column.string_at(idx) {
            row.insert(
                name.clone().into_boxed_str(),
                serde_json::Value::String(value.to_owned()),
            );
        } else if let Some(value) = column.long_at(idx) {
            row.insert(
                name.clone().into_boxed_str(),
                serde_json::Value::from(value),
            );
        }
    }
    row
}

fn missing_field(name: &str) -> ErrorEnvelope {
    ErrorEnvelope::unexpected(
        ErrorCode::new("vector", "vdb_invalid_response"),
        format!("missing field {name} in Milvus response"),
        ErrorClass::NonRetriable,
    )
}

fn encode_placeholder_group_float(vector: &[f32]) -> Result<Vec<u8>> {
    let placeholder = PlaceholderValue {
        tag: "$0".to_owned(),
        r#type: PlaceholderType::FloatVector as i32,
        values: vec![float_vector_to_bytes(vector)],
    };
    encode_placeholder_group(placeholder)
}

fn encode_placeholder_group_text(query: &str) -> Result<Vec<u8>> {
    let placeholder = PlaceholderValue {
        tag: "$0".to_owned(),
        r#type: PlaceholderType::VarChar as i32,
        values: vec![query.as_bytes().to_vec()],
    };
    encode_placeholder_group(placeholder)
}

fn encode_placeholder_group(placeholder: PlaceholderValue) -> Result<Vec<u8>> {
    let group = PlaceholderGroup {
        placeholders: vec![placeholder],
    };
    let mut buf = BytesMut::new();
    group.encode(&mut buf).map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "vdb_invalid_request"),
            format!("failed to encode placeholder group: {error}"),
            ErrorClass::NonRetriable,
        )
    })?;
    Ok(buf.to_vec())
}

fn float_vector_to_bytes(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn merge_search_params(params: &[KeyValuePair]) -> Result<String> {
    let mut map = serde_json::Map::new();
    for param in params {
        if param.key == "params"
            && let Ok(value) = serde_json::from_str::<serde_json::Value>(&param.value)
            && let Some(object) = value.as_object()
        {
            for (k, v) in object {
                map.insert(k.clone(), v.clone());
            }
            continue;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&param.value) {
            map.insert(param.key.clone(), value);
        } else {
            map.insert(
                param.key.clone(),
                serde_json::Value::String(param.value.clone()),
            );
        }
    }
    serde_json::to_string(&map).map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "vdb_invalid_request"),
            format!("failed to serialize search params: {error}"),
            ErrorClass::NonRetriable,
        )
    })
}

fn milvus_output_fields() -> Vec<String> {
    MILVUS_OUTPUT_FIELDS
        .iter()
        .copied()
        .map(str::to_string)
        .collect()
}

fn build_hybrid_search_requests(
    adapter: &MilvusGrpcVectorDb,
    collection_name: &CollectionName,
    search_requests: &[PortsHybridSearchRequest],
    options: &HybridSearchOptions,
) -> Result<Vec<SearchRequest>> {
    let mut requests = Vec::with_capacity(search_requests.len());
    let db_name = adapter.db_name.clone().unwrap_or_default();
    let db_name = db_name.as_ref().to_owned();
    let filter_expr = options.filter_expr.clone().unwrap_or_default();

    for req in search_requests {
        let (placeholder_group, metric_type) = match &req.data {
            HybridSearchData::DenseVector(vector) => (
                encode_placeholder_group_float(vector.as_ref())?,
                adapter.index_config.dense.metric_type.as_ref(),
            ),
            HybridSearchData::SparseQuery(query) => (
                encode_placeholder_group_text(query.as_ref())?,
                adapter.index_config.sparse.metric_type.as_ref(),
            ),
        };

        let mut params: Vec<KeyValuePair> = req
            .params
            .iter()
            .map(|(k, v)| KeyValuePair {
                key: k.as_ref().to_owned(),
                value: v.to_string(),
            })
            .collect();
        params.push(KeyValuePair {
            key: "topk".to_owned(),
            value: req.limit.to_string(),
        });
        params.push(KeyValuePair {
            key: "metric_type".to_owned(),
            value: metric_type.to_owned(),
        });
        params.push(KeyValuePair {
            key: "anns_field".to_owned(),
            value: req.anns_field.as_ref().to_owned(),
        });
        let merged = merge_search_params(&params)?;
        params.push(KeyValuePair {
            key: "params".to_owned(),
            value: merged,
        });

        requests.push(SearchRequest {
            base: None,
            db_name: db_name.clone(),
            collection_name: collection_name.as_str().to_owned(),
            partition_names: Vec::new(),
            dsl: filter_expr.clone().into(),
            placeholder_group,
            dsl_type: crate::vectordb::milvus::proto::common::DslType::BoolExprV1 as i32,
            output_fields: milvus_output_fields(),
            search_params: params,
            travel_timestamp: 0,
            guarantee_timestamp: 0,
            nq: 1,
            not_return_all_meta: false,
            consistency_level: crate::vectordb::milvus::proto::common::ConsistencyLevel::Bounded
                as i32,
            use_default_consistency: false,
            search_by_primary_keys: false,
            expr_template_values: std::collections::HashMap::new(),
            sub_reqs: Vec::new(),
            function_score: None,
        });
    }

    Ok(requests)
}

fn build_rank_params(options: &HybridSearchOptions) -> Vec<KeyValuePair> {
    let mut params = Vec::new();
    if let Some(rerank) = options.rerank.as_ref() {
        params.push(KeyValuePair {
            key: "strategy".to_owned(),
            value: match rerank.strategy {
                semantic_code_ports::RerankStrategyKind::Rrf => "rrf".to_owned(),
                semantic_code_ports::RerankStrategyKind::Weighted => "weighted".to_owned(),
            },
        });
        if let Some(extra) = rerank.params.as_ref() {
            for (key, value) in extra {
                params.push(KeyValuePair {
                    key: key.as_ref().to_owned(),
                    value: value.to_string(),
                });
            }
        }
    }
    params
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collection_name_enforces_length() {
        let long_name = "a".repeat(260);
        let name = match CollectionName::parse(long_name.as_str()) {
            Ok(value) => value,
            Err(_) => {
                assert!(false, "expected CollectionName parse to succeed");
                return;
            },
        };
        let error = ensure_collection_name(&name).err();
        assert!(error.is_some());
    }
}
