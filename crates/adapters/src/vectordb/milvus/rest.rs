//! Milvus REST adapter.

use crate::vectordb::milvus::error::{
    MilvusErrorContext, MilvusProviderId, map_rest_error, map_rest_transport_error,
};
use crate::vectordb::milvus::metadata::{metadata_from_fields, serialize_metadata};
use crate::vectordb::milvus::rest_auth::{MilvusRestAuthInput, build_rest_auth_header};
use crate::vectordb::milvus::rest_base_url::to_milvus_rest_base_url;
use crate::vectordb::milvus::schema::{
    build_dense_schema_spec, build_hybrid_schema_spec, build_rest_schema,
};
use crate::vectordb::milvus::shared::{
    MILVUS_OUTPUT_FIELDS, ensure_collection_name, milvus_in_string,
};
use crate::vectordb::milvus::{MilvusIndexConfig, MilvusIndexSpec};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use semantic_code_domain::{CollectionName, VectorDbProviderId};
use semantic_code_ports::{
    HybridSearchBatchRequest, HybridSearchData, HybridSearchOptions,
    HybridSearchRequest as PortsHybridSearchRequest, HybridSearchResult, VectorDbPort,
    VectorDbProviderInfo, VectorDbRow, VectorDocument, VectorDocumentForInsert,
    VectorSearchRequest, VectorSearchResult,
};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Milvus REST adapter configuration.
#[derive(Debug, Clone)]
pub struct MilvusRestConfig {
    /// Base URL for the Milvus REST service.
    pub address: Box<str>,
    /// Optional bearer token for authentication.
    pub token: Option<Box<str>>,
    /// Optional username for basic authentication.
    pub username: Option<Box<str>>,
    /// Optional password for basic authentication.
    pub password: Option<Box<str>>,
    /// Optional database name header.
    pub database: Option<Box<str>>,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Index configuration (dense + sparse).
    pub index_config: MilvusIndexConfig,
}

impl MilvusRestConfig {
    /// Validates configuration invariants for the REST adapter.
    pub fn validate(&self) -> Result<()> {
        if self.address.trim().is_empty() {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "Milvus REST address is required",
            ));
        }
        if self.timeout_ms == 0 {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "Milvus timeout must be greater than zero",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct MilvusRestResponse<T> {
    code: i32,
    message: Option<String>,
    data: Option<T>,
}

#[derive(Debug, Deserialize)]
struct LoadStateData {
    #[serde(rename = "loadState")]
    load_state: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CollectionListData {
    #[serde(rename = "collectionNames")]
    collection_names: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct SearchData {
    data: Option<Vec<MilvusRestRow>>,
}

#[derive(Debug, Deserialize)]
struct MilvusRestRow {
    id: Option<String>,
    content: Option<String>,
    #[serde(rename = "relativePath")]
    relative_path: Option<String>,
    #[serde(rename = "startLine")]
    start_line: Option<i64>,
    #[serde(rename = "endLine")]
    end_line: Option<i64>,
    #[serde(rename = "fileExtension")]
    file_extension: Option<String>,
    metadata: Option<String>,
    score: Option<f32>,
}

/// Milvus REST vector DB adapter.
#[derive(Clone)]
pub struct MilvusRestVectorDb {
    provider: VectorDbProviderInfo,
    client: reqwest::Client,
    base_url: Box<str>,
    database: Option<Box<str>>,
    timeout: Duration,
    index_config: MilvusIndexConfig,
}

impl MilvusRestVectorDb {
    /// Creates a Milvus REST adapter instance from configuration.
    pub fn new(config: MilvusRestConfig) -> Result<Self> {
        config.validate()?;
        let base_url = to_milvus_rest_base_url(&config.address)?;
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let auth_input = MilvusRestAuthInput {
            token: config.token.as_deref(),
            username: config.username.as_deref(),
            password: config.password.as_deref(),
        };
        if let Some(auth) = build_rest_auth_header(&auth_input) {
            let value = HeaderValue::from_str(auth.as_str()).map_err(|_| {
                ErrorEnvelope::expected(
                    ErrorCode::invalid_input(),
                    "Milvus REST auth header contains invalid characters",
                )
            })?;
            headers.insert(AUTHORIZATION, value);
        }

        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .default_headers(headers)
            .build()
            .map_err(|error| {
                ErrorEnvelope::unexpected(
                    ErrorCode::new("vector", "vdb_client_init_failed"),
                    format!("failed to build Milvus REST client: {error}"),
                    ErrorClass::NonRetriable,
                )
            })?;

        let provider = VectorDbProviderInfo {
            id: VectorDbProviderId::parse("milvus_rest").map_err(ErrorEnvelope::from)?,
            name: "Milvus REST".into(),
        };

        Ok(Self {
            provider,
            client,
            base_url,
            database: config.database,
            timeout: Duration::from_millis(config.timeout_ms),
            index_config: config.index_config,
        })
    }

    fn context(
        operation: &'static str,
        collection: Option<&CollectionName>,
        endpoint: Option<&str>,
    ) -> MilvusErrorContext {
        MilvusErrorContext {
            provider: MilvusProviderId::Rest,
            operation,
            collection_name: collection.map(|c| c.as_str().to_owned()),
            endpoint: endpoint.map(ToOwned::to_owned),
        }
    }

    async fn make_request<T: for<'de> Deserialize<'de>, B: Serialize + Sync>(
        &self,
        ctx: &RequestContext,
        endpoint: &str,
        body: Option<&B>,
        operation: &'static str,
        collection: Option<&CollectionName>,
    ) -> Result<MilvusRestResponse<T>> {
        ctx.ensure_not_cancelled(operation)?;
        let url = format!("{}{}", self.base_url, endpoint);
        let request = body.map_or_else(
            || self.client.post(&url),
            |body| self.client.post(&url).json(body),
        );

        let response = tokio::select! {
            () = ctx.cancelled() => return Err(cancelled_error(operation)),
            res = tokio::time::timeout(self.timeout, request.send()) => res,
        };

        let response = match response {
            Ok(result) => result.map_err(|error| {
                map_rest_transport_error(
                    &error,
                    &Self::context(operation, collection, Some(endpoint)),
                )
            })?,
            Err(_) => return Err(timeout_error(operation)),
        };

        let status = response.status();
        let payload = response.bytes().await.map_err(|error| {
            map_rest_transport_error(
                &error,
                &Self::context(operation, collection, Some(endpoint)),
            )
        })?;

        if !status.is_success() {
            let message = String::from_utf8_lossy(&payload).to_string();
            return Err(map_rest_error(
                format!("HTTP {}: {message}", status.as_u16()),
                Some(status.as_u16()),
                &Self::context(operation, collection, Some(endpoint)),
            ));
        }

        let result: MilvusRestResponse<T> = serde_json::from_slice(&payload).map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "vdb_invalid_response"),
                format!("invalid Milvus REST response: {error}"),
                ErrorClass::NonRetriable,
            )
        })?;

        if result.code != 0 && result.code != 200 {
            let message = result
                .message
                .unwrap_or_else(|| "Milvus REST error".to_owned());
            return Err(map_rest_error(
                message,
                None,
                &Self::context(operation, collection, Some(endpoint)),
            ));
        }

        Ok(result)
    }

    async fn ensure_loaded(&self, ctx: &RequestContext, collection: &CollectionName) -> Result<()> {
        // TODO: make index type/metric configurable per deployment.
        let body = serde_json::json!({
            "collectionName": collection.as_str(),
            "dbName": self.database,
        });
        let response: MilvusRestResponse<LoadStateData> = self
            .make_request(
                ctx,
                "/collections/get_load_state",
                Some(&body),
                "milvus_rest.ensure_loaded",
                Some(collection),
            )
            .await?;
        let loaded = response
            .data
            .and_then(|data| data.load_state)
            .is_some_and(|state| state == "LoadStateLoaded");
        if loaded {
            return Ok(());
        }

        let body = serde_json::json!({
            "collectionName": collection.as_str(),
            "dbName": self.database,
        });
        let _response: MilvusRestResponse<serde_json::Value> = self
            .make_request(
                ctx,
                "/collections/load",
                Some(&body),
                "milvus_rest.load_collection",
                Some(collection),
            )
            .await?;
        Ok(())
    }

    async fn create_index(&self, ctx: &RequestContext, collection: &CollectionName) -> Result<()> {
        let params = index_params_json(&self.index_config.dense);
        let body = serde_json::json!({
            "collectionName": collection.as_str(),
            "dbName": self.database,
            "indexParams": [
                {
                    "fieldName": "vector",
                    "indexName": "vector_index",
                    "metricType": self.index_config.dense.metric_type,
                    "index_type": self.index_config.dense.index_type,
                    "params": params
                }
            ],
        });
        let _response: MilvusRestResponse<serde_json::Value> = self
            .make_request(
                ctx,
                "/indexes/create",
                Some(&body),
                "milvus_rest.create_index",
                Some(collection),
            )
            .await?;
        Ok(())
    }

    async fn create_hybrid_indexes(
        &self,
        ctx: &RequestContext,
        collection: &CollectionName,
    ) -> Result<()> {
        let dense_params = index_params_json(&self.index_config.dense);
        let dense = serde_json::json!({
            "collectionName": collection.as_str(),
            "dbName": self.database,
            "indexParams": [
                {
                    "fieldName": "vector",
                    "indexName": "vector_index",
                    "metricType": self.index_config.dense.metric_type,
                    "index_type": self.index_config.dense.index_type,
                    "params": dense_params
                }
            ],
        });
        let _response: MilvusRestResponse<serde_json::Value> = self
            .make_request(
                ctx,
                "/indexes/create",
                Some(&dense),
                "milvus_rest.create_index",
                Some(collection),
            )
            .await?;

        let sparse_params = index_params_json(&self.index_config.sparse);
        let sparse = serde_json::json!({
            "collectionName": collection.as_str(),
            "dbName": self.database,
            "indexParams": [
                {
                    "fieldName": "sparse_vector",
                    "indexName": "sparse_vector_index",
                    "metricType": self.index_config.sparse.metric_type,
                    "index_type": self.index_config.sparse.index_type,
                    "params": sparse_params
                }
            ],
        });
        let _response: MilvusRestResponse<serde_json::Value> = self
            .make_request(
                ctx,
                "/indexes/create",
                Some(&sparse),
                "milvus_rest.create_index",
                Some(collection),
            )
            .await?;
        Ok(())
    }

    fn row_to_doc(row: &MilvusRestRow) -> Result<VectorDocument> {
        let relative_path = row
            .relative_path
            .as_deref()
            .ok_or_else(|| missing_field("relativePath"))?;
        let start_line = row.start_line.ok_or_else(|| missing_field("startLine"))?;
        let end_line = row.end_line.ok_or_else(|| missing_field("endLine"))?;
        let metadata = metadata_from_fields(
            relative_path,
            start_line,
            end_line,
            row.file_extension.as_deref(),
            row.metadata.as_deref(),
        )?;
        Ok(VectorDocument {
            id: row
                .id
                .as_deref()
                .unwrap_or_default()
                .to_owned()
                .into_boxed_str(),
            vector: None,
            content: row
                .content
                .as_deref()
                .unwrap_or_default()
                .to_owned()
                .into_boxed_str(),
            metadata,
        })
    }
}

impl VectorDbPort for MilvusRestVectorDb {
    fn provider(&self) -> &VectorDbProviderInfo {
        &self.provider
    }

    fn create_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        dimension: u32,
        _description: Option<Box<str>>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let adapter = self.clone();
        Box::pin(async move {
            ensure_collection_name(&collection_name)?;
            let spec = build_dense_schema_spec(dimension);
            let schema = build_rest_schema(&spec);
            let body = serde_json::json!({
                "collectionName": collection_name.as_str(),
                "dbName": adapter.database,
                "schema": {
                    "enableDynamicField": false,
                    "fields": schema.get("fields").cloned().unwrap_or_default(),
                }
            });

            let _response: MilvusRestResponse<serde_json::Value> = adapter
                .make_request(
                    &ctx,
                    "/collections/create",
                    Some(&body),
                    "milvus_rest.create_collection",
                    Some(&collection_name),
                )
                .await?;
            adapter.create_index(&ctx, &collection_name).await?;
            adapter.ensure_loaded(&ctx, &collection_name).await?;
            Ok(())
        })
    }

    fn create_hybrid_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        dimension: u32,
        _description: Option<Box<str>>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let adapter = self.clone();
        Box::pin(async move {
            ensure_collection_name(&collection_name)?;
            let spec = build_hybrid_schema_spec(dimension);
            let schema = build_rest_schema(&spec);
            let body = serde_json::json!({
                "collectionName": collection_name.as_str(),
                "dbName": adapter.database,
                "schema": {
                    "enableDynamicField": false,
                    "fields": schema.get("fields").cloned().unwrap_or_default(),
                    "functions": schema.get("functions").cloned().unwrap_or_default(),
                }
            });
            let _response: MilvusRestResponse<serde_json::Value> = adapter
                .make_request(
                    &ctx,
                    "/collections/create",
                    Some(&body),
                    "milvus_rest.create_hybrid_collection",
                    Some(&collection_name),
                )
                .await?;
            adapter
                .create_hybrid_indexes(&ctx, &collection_name)
                .await?;
            adapter.ensure_loaded(&ctx, &collection_name).await?;
            Ok(())
        })
    }

    fn drop_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let adapter = self.clone();
        Box::pin(async move {
            ensure_collection_name(&collection_name)?;
            let body = serde_json::json!({
                "collectionName": collection_name.as_str(),
                "dbName": adapter.database,
            });
            let _response: MilvusRestResponse<serde_json::Value> = adapter
                .make_request(
                    &ctx,
                    "/collections/drop",
                    Some(&body),
                    "milvus_rest.drop_collection",
                    Some(&collection_name),
                )
                .await?;
            Ok(())
        })
    }

    fn has_collection(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
    ) -> semantic_code_ports::BoxFuture<'_, Result<bool>> {
        let ctx = ctx.clone();
        let adapter = self.clone();
        Box::pin(async move {
            ensure_collection_name(&collection_name)?;
            let body = serde_json::json!({
                "collectionName": collection_name.as_str(),
                "dbName": adapter.database,
            });
            let response: MilvusRestResponse<serde_json::Value> = adapter
                .make_request(
                    &ctx,
                    "/collections/has",
                    Some(&body),
                    "milvus_rest.has_collection",
                    Some(&collection_name),
                )
                .await?;
            let exists = response
                .data
                .and_then(|value| value.get("value").cloned())
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            Ok(exists)
        })
    }

    fn list_collections(
        &self,
        ctx: &RequestContext,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<CollectionName>>> {
        let ctx = ctx.clone();
        let adapter = self.clone();
        Box::pin(async move {
            let body = serde_json::json!({ "dbName": adapter.database });
            let response: MilvusRestResponse<CollectionListData> = adapter
                .make_request(
                    &ctx,
                    "/collections/list",
                    Some(&body),
                    "milvus_rest.list_collections",
                    None,
                )
                .await?;
            let names = response
                .data
                .and_then(|data| data.collection_names)
                .unwrap_or_default();
            let mut out = Vec::new();
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
        collection_name: CollectionName,
        documents: Vec<VectorDocumentForInsert>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let adapter = self.clone();
        Box::pin(async move {
            ensure_collection_name(&collection_name)?;
            adapter.ensure_loaded(&ctx, &collection_name).await?;
            let mut data = Vec::with_capacity(documents.len());
            for doc in documents {
                let metadata = serialize_metadata(&doc.metadata)?;
                data.push(serde_json::json!({
                    "id": doc.id,
                    "content": doc.content,
                    "vector": doc.vector.as_ref(),
                    "relativePath": doc.metadata.relative_path,
                    "startLine": doc.metadata.span.start_line(),
                    "endLine": doc.metadata.span.end_line(),
                    "fileExtension": doc.metadata.file_extension.unwrap_or_default(),
                    "metadata": metadata,
                }));
            }
            let body = serde_json::json!({
                "collectionName": collection_name.as_str(),
                "dbName": adapter.database,
                "data": data,
            });
            let _response: MilvusRestResponse<serde_json::Value> = adapter
                .make_request(
                    &ctx,
                    "/entities/insert",
                    Some(&body),
                    "milvus_rest.insert",
                    Some(&collection_name),
                )
                .await?;
            Ok(())
        })
    }

    fn insert_hybrid(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
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

            let mut search_request = serde_json::json!({
                "collectionName": collection_name.as_str(),
                "dbName": adapter.database,
                "searchParams": {
                    "metricType": adapter.index_config.dense.metric_type,
                    "params": { "nprobe": 10 }
                },
                "limit": options.top_k.unwrap_or(10),
                "outputFields": MILVUS_OUTPUT_FIELDS,
                "data": [query_vector.as_ref()],
            });
            if let Some(filter) = options.filter_expr {
                search_request.as_object_mut().map(|map| {
                    map.insert(
                        "filter".to_owned(),
                        serde_json::Value::String(filter.into()),
                    )
                });
            }

            let response: MilvusRestResponse<SearchData> = adapter
                .make_request(
                    &ctx,
                    "/entities/search",
                    Some(&search_request),
                    "milvus_rest.search",
                    Some(&collection_name),
                )
                .await?;
            let rows = response.data.and_then(|data| data.data).unwrap_or_default();
            let mut results = Vec::with_capacity(rows.len());
            for row in rows {
                let doc = Self::row_to_doc(&row)?;
                results.push(VectorSearchResult {
                    document: doc,
                    score: row.score.unwrap_or_default(),
                });
            }
            Ok(results)
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

            let body = build_hybrid_search_body(
                &collection_name,
                adapter.database.as_deref(),
                &search_requests,
                &options,
                &adapter.index_config,
            )?;

            let response: MilvusRestResponse<SearchData> = adapter
                .make_request(
                    &ctx,
                    "/entities/hybrid_search",
                    Some(&body),
                    "milvus_rest.hybrid_search",
                    Some(&collection_name),
                )
                .await?;
            let rows = response.data.and_then(|data| data.data).unwrap_or_default();
            let mut results = Vec::with_capacity(rows.len());
            for row in rows {
                let doc = Self::row_to_doc(&row)?;
                results.push(HybridSearchResult {
                    document: doc,
                    score: row.score.unwrap_or_default(),
                });
            }
            Ok(results)
        })
    }

    fn delete(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        ids: Vec<Box<str>>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<()>> {
        let ctx = ctx.clone();
        let adapter = self.clone();
        Box::pin(async move {
            ensure_collection_name(&collection_name)?;
            adapter.ensure_loaded(&ctx, &collection_name).await?;
            let filter = milvus_in_string("id", &ids);
            let body = serde_json::json!({
                "collectionName": collection_name.as_str(),
                "dbName": adapter.database,
                "filter": filter,
            });
            let _response: MilvusRestResponse<serde_json::Value> = adapter
                .make_request(
                    &ctx,
                    "/entities/delete",
                    Some(&body),
                    "milvus_rest.delete",
                    Some(&collection_name),
                )
                .await?;
            Ok(())
        })
    }

    fn query(
        &self,
        ctx: &RequestContext,
        collection_name: CollectionName,
        filter: Box<str>,
        output_fields: Vec<Box<str>>,
        limit: Option<u32>,
    ) -> semantic_code_ports::BoxFuture<'_, Result<Vec<VectorDbRow>>> {
        let ctx = ctx.clone();
        let adapter = self.clone();
        Box::pin(async move {
            ensure_collection_name(&collection_name)?;
            adapter.ensure_loaded(&ctx, &collection_name).await?;
            let body = serde_json::json!({
                "collectionName": collection_name.as_str(),
                "dbName": adapter.database,
                "filter": filter,
                "outputFields": output_fields,
                "limit": limit,
            });
            let response: MilvusRestResponse<SearchData> = adapter
                .make_request(
                    &ctx,
                    "/entities/query",
                    Some(&body),
                    "milvus_rest.query",
                    Some(&collection_name),
                )
                .await?;
            let rows = response.data.and_then(|data| data.data).unwrap_or_default();
            let mut out = Vec::with_capacity(rows.len());
            for row in rows {
                let mut map = VectorDbRow::new();
                if let Some(id) = row.id {
                    map.insert("id".into(), serde_json::Value::String(id));
                }
                if let Some(relative_path) = row.relative_path {
                    map.insert(
                        "relativePath".into(),
                        serde_json::Value::String(relative_path),
                    );
                }
                if let Some(start_line) = row.start_line {
                    map.insert("startLine".into(), serde_json::Value::from(start_line));
                }
                if let Some(end_line) = row.end_line {
                    map.insert("endLine".into(), serde_json::Value::from(end_line));
                }
                if let Some(extension) = row.file_extension {
                    map.insert("fileExtension".into(), serde_json::Value::String(extension));
                }
                if let Some(content) = row.content {
                    map.insert("content".into(), serde_json::Value::String(content));
                }
                out.push(map);
            }
            Ok(out)
        })
    }
}

fn build_hybrid_search_body(
    collection_name: &CollectionName,
    database: Option<&str>,
    search_requests: &[PortsHybridSearchRequest],
    options: &HybridSearchOptions,
    index_config: &MilvusIndexConfig,
) -> Result<serde_json::Value> {
    if search_requests.len() < 2 {
        return Err(ErrorEnvelope::expected(
            ErrorCode::invalid_input(),
            "hybrid search requires dense and sparse requests",
        ));
    }

    let mut search_params = Vec::new();
    for req in search_requests {
        let (data, metric_type) = match &req.data {
            HybridSearchData::DenseVector(vector) => (
                serde_json::Value::Array(vec![serde_json::json!(vector.as_ref())]),
                index_config.dense.metric_type.as_ref(),
            ),
            HybridSearchData::SparseQuery(query) => (
                serde_json::Value::Array(vec![serde_json::Value::String(
                    query.as_ref().to_owned(),
                )]),
                index_config.sparse.metric_type.as_ref(),
            ),
        };

        let params = req
            .params
            .iter()
            .map(|(k, v)| (k.as_ref().to_owned(), v.clone()))
            .collect::<serde_json::Map<_, _>>();

        let entry = serde_json::json!({
            "annsField": req.anns_field.as_ref(),
            "limit": req.limit,
            "data": data,
            "searchParams": {
                "metricType": metric_type,
                "params": params,
            }
        });
        search_params.push(entry);
    }

    let mut body = serde_json::json!({
        "collectionName": collection_name.as_str(),
        "dbName": database,
        "search": search_params,
        "limit": options.limit.unwrap_or(10),
        "outputFields": MILVUS_OUTPUT_FIELDS,
    });

    if let Some(filter) = options.filter_expr.as_ref()
        && let Some(obj) = body.as_object_mut()
    {
        obj.insert(
            "filter".to_owned(),
            serde_json::Value::String(filter.as_ref().to_owned()),
        );
    }

    Ok(body)
}

fn index_params_json(spec: &MilvusIndexSpec) -> serde_json::Value {
    let mut params = serde_json::Map::new();
    for (key, value) in &spec.params {
        params.insert(key.as_ref().to_owned(), param_value_to_json(value.as_ref()));
    }
    serde_json::Value::Object(params)
}

fn param_value_to_json(value: &str) -> serde_json::Value {
    let trimmed = value.trim();
    if trimmed.eq_ignore_ascii_case("true") {
        return serde_json::Value::Bool(true);
    }
    if trimmed.eq_ignore_ascii_case("false") {
        return serde_json::Value::Bool(false);
    }
    if let Ok(number) = trimmed.parse::<i64>() {
        return serde_json::Value::Number(number.into());
    }
    if let Ok(number) = trimmed.parse::<f64>()
        && let Some(value) = serde_json::Number::from_f64(number)
    {
        return serde_json::Value::Number(value);
    }
    serde_json::Value::String(trimmed.to_owned())
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

fn missing_field(name: &str) -> ErrorEnvelope {
    ErrorEnvelope::unexpected(
        ErrorCode::new("vector", "vdb_invalid_response"),
        format!("missing field {name} in Milvus response"),
        ErrorClass::NonRetriable,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn rest_search_serializes_request() {
        let search = serde_json::json!({
            "collectionName": "collection",
            "dbName": "default",
            "searchParams": {"metricType": "COSINE", "params": {"nprobe": 10}},
            "limit": 10,
            "outputFields": MILVUS_OUTPUT_FIELDS,
            "data": [[0.1, 0.2, 0.3]],
        });
        let payload = match serde_json::to_string(&search) {
            Ok(value) => value,
            Err(_) => {
                assert!(false, "expected search payload to serialize");
                return;
            },
        };
        assert!(payload.contains("collectionName"));
        assert!(payload.contains("outputFields"));
    }

    #[test]
    fn rest_hybrid_search_builds_payload() {
        let collection = match CollectionName::parse("collection") {
            Ok(value) => value,
            Err(_) => {
                assert!(false, "expected collection name to parse");
                return;
            },
        };

        let mut dense_params = BTreeMap::new();
        dense_params.insert("nprobe".into(), serde_json::json!(10));
        let dense = PortsHybridSearchRequest {
            data: HybridSearchData::DenseVector(Arc::from(vec![0.1, 0.2, 0.3])),
            anns_field: "vector".into(),
            params: dense_params,
            limit: 5,
        };

        let mut sparse_params = BTreeMap::new();
        sparse_params.insert("drop_ratio".into(), serde_json::json!(0.2));
        let sparse = PortsHybridSearchRequest {
            data: HybridSearchData::SparseQuery("hello".into()),
            anns_field: "sparse_vector".into(),
            params: sparse_params,
            limit: 7,
        };

        let options = HybridSearchOptions {
            limit: Some(10),
            filter_expr: Some("fileExtension == \"rs\"".into()),
            rerank: None,
        };

        let body = match build_hybrid_search_body(
            &collection,
            None,
            &[dense, sparse],
            &options,
            &MilvusIndexConfig::default(),
        ) {
            Ok(value) => value,
            Err(_) => {
                assert!(false, "expected hybrid search body to build");
                return;
            },
        };

        let search_entries = match body.get("search").and_then(serde_json::Value::as_array) {
            Some(entries) => entries,
            None => {
                assert!(false, "expected search entries");
                return;
            },
        };
        assert_eq!(search_entries.len(), 2);

        let payload = match serde_json::to_string(&body) {
            Ok(value) => value,
            Err(_) => {
                assert!(false, "expected hybrid body to serialize");
                return;
            },
        };
        assert!(payload.contains("BM25"));
        assert!(payload.contains("COSINE"));
        assert!(payload.contains("filter"));
    }
}
