//! Milvus schema spec builders shared by gRPC and REST.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "dataType")]
/// Schema field definition used by Milvus REST and gRPC builders.
pub enum MilvusFieldSpec {
    /// Variable-length string field.
    #[serde(rename = "VarChar")]
    VarChar {
        /// Field name.
        name: Box<str>,
        /// Maximum length allowed for this field.
        max_length: i32,
        /// Whether this field is the collection primary key.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        is_primary_key: bool,
        /// Whether analyzer support is enabled for this field.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        enable_analyzer: bool,
    },
    /// Dense float vector field.
    #[serde(rename = "FloatVector")]
    FloatVector {
        /// Field name.
        name: Box<str>,
        /// Vector dimension.
        dim: i64,
    },
    /// Sparse float vector field.
    #[serde(rename = "SparseFloatVector")]
    SparseFloatVector {
        /// Field name.
        name: Box<str>,
    },
    /// 64-bit integer field.
    #[serde(rename = "Int64")]
    Int64 {
        /// Field name.
        name: Box<str>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Milvus server-side function definition (e.g., BM25).
pub struct MilvusFunctionSpec {
    /// Function type identifier.
    #[serde(rename = "type")]
    pub kind: Box<str>,
    /// Function name.
    pub name: Box<str>,
    /// Human-readable function description.
    pub description: Box<str>,
    /// Input field names.
    pub input_field_names: Vec<Box<str>>,
    /// Output field names.
    pub output_field_names: Vec<Box<str>>,
    /// Function parameter payload.
    #[serde(default)]
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Collection schema definition for Milvus collections.
pub struct MilvusSchemaSpec {
    /// Field definitions for the collection.
    pub fields: Vec<MilvusFieldSpec>,
    /// Optional function definitions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub functions: Vec<MilvusFunctionSpec>,
}

/// Builds the dense-vector collection schema used by Milvus.
pub fn build_dense_schema_spec(dimension: u32) -> MilvusSchemaSpec {
    MilvusSchemaSpec {
        fields: vec![
            MilvusFieldSpec::VarChar {
                name: "id".into(),
                max_length: 512,
                is_primary_key: true,
                enable_analyzer: false,
            },
            MilvusFieldSpec::FloatVector {
                name: "vector".into(),
                dim: i64::from(dimension),
            },
            MilvusFieldSpec::VarChar {
                name: "content".into(),
                max_length: 65_535,
                is_primary_key: false,
                enable_analyzer: false,
            },
            MilvusFieldSpec::VarChar {
                name: "relativePath".into(),
                max_length: 1024,
                is_primary_key: false,
                enable_analyzer: false,
            },
            MilvusFieldSpec::Int64 {
                name: "startLine".into(),
            },
            MilvusFieldSpec::Int64 {
                name: "endLine".into(),
            },
            MilvusFieldSpec::VarChar {
                name: "fileExtension".into(),
                max_length: 32,
                is_primary_key: false,
                enable_analyzer: false,
            },
            MilvusFieldSpec::VarChar {
                name: "metadata".into(),
                max_length: 65_535,
                is_primary_key: false,
                enable_analyzer: false,
            },
        ],
        functions: Vec::new(),
    }
}

/// Builds the hybrid (dense + sparse) collection schema used by Milvus.
pub fn build_hybrid_schema_spec(dimension: u32) -> MilvusSchemaSpec {
    MilvusSchemaSpec {
        fields: vec![
            MilvusFieldSpec::VarChar {
                name: "id".into(),
                max_length: 512,
                is_primary_key: true,
                enable_analyzer: false,
            },
            MilvusFieldSpec::VarChar {
                name: "content".into(),
                max_length: 65_535,
                is_primary_key: false,
                enable_analyzer: true,
            },
            MilvusFieldSpec::FloatVector {
                name: "vector".into(),
                dim: i64::from(dimension),
            },
            MilvusFieldSpec::SparseFloatVector {
                name: "sparse_vector".into(),
            },
            MilvusFieldSpec::VarChar {
                name: "relativePath".into(),
                max_length: 1024,
                is_primary_key: false,
                enable_analyzer: false,
            },
            MilvusFieldSpec::Int64 {
                name: "startLine".into(),
            },
            MilvusFieldSpec::Int64 {
                name: "endLine".into(),
            },
            MilvusFieldSpec::VarChar {
                name: "fileExtension".into(),
                max_length: 32,
                is_primary_key: false,
                enable_analyzer: false,
            },
            MilvusFieldSpec::VarChar {
                name: "metadata".into(),
                max_length: 65_535,
                is_primary_key: false,
                enable_analyzer: false,
            },
        ],
        functions: vec![MilvusFunctionSpec {
            kind: "BM25".into(),
            name: "content_bm25_emb".into(),
            description: "content bm25 function".into(),
            input_field_names: vec!["content".into()],
            output_field_names: vec!["sparse_vector".into()],
            params: serde_json::json!({}),
        }],
    }
}

#[cfg(feature = "milvus-grpc")]
/// Converts a schema spec into a gRPC `CollectionSchema` payload.
pub fn build_grpc_schema(
    spec: &MilvusSchemaSpec,
    collection_name: &str,
    description: &str,
) -> crate::vectordb::milvus::proto::schema::CollectionSchema {
    use crate::vectordb::milvus::proto::schema::CollectionSchema;

    let fields = build_grpc_fields(spec);
    let functions = build_grpc_functions(spec);

    CollectionSchema {
        name: collection_name.to_owned(),
        description: description.to_owned(),
        auto_id: false,
        fields,
        enable_dynamic_field: false,
        properties: Vec::new(),
        functions,
        db_name: String::new(),
        struct_array_fields: Vec::new(),
    }
}

#[cfg(feature = "milvus-grpc")]
fn grpc_field_id(idx: usize) -> i64 {
    i64::try_from(idx).unwrap_or(i64::MAX)
}

#[cfg(feature = "milvus-grpc")]
fn build_grpc_fields(
    spec: &MilvusSchemaSpec,
) -> Vec<crate::vectordb::milvus::proto::schema::FieldSchema> {
    spec.fields
        .iter()
        .enumerate()
        .map(|(idx, field)| build_grpc_field_schema(grpc_field_id(idx), field))
        .collect()
}

#[cfg(feature = "milvus-grpc")]
fn build_grpc_field_schema(
    field_id: i64,
    field: &MilvusFieldSpec,
) -> crate::vectordb::milvus::proto::schema::FieldSchema {
    match field {
        MilvusFieldSpec::VarChar {
            name,
            max_length,
            is_primary_key,
            enable_analyzer,
        } => build_var_char_schema(
            field_id,
            name.as_ref(),
            *max_length,
            *is_primary_key,
            *enable_analyzer,
        ),
        MilvusFieldSpec::FloatVector { name, dim } => {
            build_vector_schema(field_id, name.as_ref(), *dim)
        },
        MilvusFieldSpec::SparseFloatVector { name } => {
            build_sparse_vector_schema(field_id, name.as_ref())
        },
        MilvusFieldSpec::Int64 { name } => build_int64_schema(field_id, name.as_ref()),
    }
}

#[cfg(feature = "milvus-grpc")]
fn build_var_char_schema(
    field_id: i64,
    name: &str,
    max_length: i32,
    is_primary_key: bool,
    enable_analyzer: bool,
) -> crate::vectordb::milvus::proto::schema::FieldSchema {
    use crate::vectordb::milvus::proto::common::KeyValuePair;
    use crate::vectordb::milvus::proto::schema::DataType;

    let mut params = vec![KeyValuePair {
        key: "max_length".to_owned(),
        value: max_length.to_string(),
    }];
    if enable_analyzer {
        params.push(KeyValuePair {
            key: "enable_analyzer".to_owned(),
            value: "true".to_owned(),
        });
    }

    build_base_field_schema(field_id, name, DataType::VarChar, params, is_primary_key)
}

#[cfg(feature = "milvus-grpc")]
fn build_vector_schema(
    field_id: i64,
    name: &str,
    dim: i64,
) -> crate::vectordb::milvus::proto::schema::FieldSchema {
    use crate::vectordb::milvus::proto::common::KeyValuePair;
    use crate::vectordb::milvus::proto::schema::DataType;

    let params = vec![KeyValuePair {
        key: "dim".to_owned(),
        value: dim.to_string(),
    }];

    build_base_field_schema(field_id, name, DataType::FloatVector, params, false)
}

#[cfg(feature = "milvus-grpc")]
fn build_sparse_vector_schema(
    field_id: i64,
    name: &str,
) -> crate::vectordb::milvus::proto::schema::FieldSchema {
    use crate::vectordb::milvus::proto::schema::DataType;

    build_base_field_schema(
        field_id,
        name,
        DataType::SparseFloatVector,
        Vec::new(),
        false,
    )
}

#[cfg(feature = "milvus-grpc")]
fn build_int64_schema(
    field_id: i64,
    name: &str,
) -> crate::vectordb::milvus::proto::schema::FieldSchema {
    use crate::vectordb::milvus::proto::schema::DataType;

    build_base_field_schema(field_id, name, DataType::Int64, Vec::new(), false)
}

#[cfg(feature = "milvus-grpc")]
fn build_base_field_schema(
    field_id: i64,
    name: &str,
    data_type: crate::vectordb::milvus::proto::schema::DataType,
    type_params: Vec<crate::vectordb::milvus::proto::common::KeyValuePair>,
    is_primary_key: bool,
) -> crate::vectordb::milvus::proto::schema::FieldSchema {
    crate::vectordb::milvus::proto::schema::FieldSchema {
        field_id,
        name: name.to_owned(),
        is_primary_key,
        description: name.to_owned(),
        data_type: data_type as i32,
        type_params,
        index_params: Vec::new(),
        auto_id: false,
        state: 0,
        element_type: 0,
        default_value: None,
        is_dynamic: false,
        is_partition_key: false,
        is_clustering_key: false,
        nullable: false,
        is_function_output: false,
    }
}

#[cfg(feature = "milvus-grpc")]
fn build_grpc_functions(
    spec: &MilvusSchemaSpec,
) -> Vec<crate::vectordb::milvus::proto::schema::FunctionSchema> {
    use crate::vectordb::milvus::proto::schema::{FunctionSchema, FunctionType};

    spec.functions
        .iter()
        .enumerate()
        .map(|(idx, func)| FunctionSchema {
            name: func.name.as_ref().to_owned(),
            id: grpc_field_id(idx),
            description: func.description.as_ref().to_owned(),
            r#type: FunctionType::Bm25 as i32,
            input_field_names: func
                .input_field_names
                .iter()
                .map(AsRef::as_ref)
                .map(ToOwned::to_owned)
                .collect(),
            input_field_ids: Vec::new(),
            output_field_names: func
                .output_field_names
                .iter()
                .map(AsRef::as_ref)
                .map(ToOwned::to_owned)
                .collect(),
            output_field_ids: Vec::new(),
            params: Vec::new(),
        })
        .collect()
}

#[cfg(feature = "milvus-rest")]
/// Converts a schema spec into a Milvus REST schema payload.
pub fn build_rest_schema(spec: &MilvusSchemaSpec) -> serde_json::Value {
    let fields = spec
        .fields
        .iter()
        .map(|field| match field {
            MilvusFieldSpec::VarChar {
                name,
                max_length,
                is_primary_key,
                enable_analyzer,
            } => serde_json::json!({
                "fieldName": name.as_ref(),
                "dataType": "VarChar",
                "isPrimary": is_primary_key,
                "elementTypeParams": {
                    "max_length": max_length,
                    "enable_analyzer": enable_analyzer,
                }
            }),
            MilvusFieldSpec::FloatVector { name, dim } => serde_json::json!({
                "fieldName": name.as_ref(),
                "dataType": "FloatVector",
                "elementTypeParams": { "dim": dim }
            }),
            MilvusFieldSpec::SparseFloatVector { name } => serde_json::json!({
                "fieldName": name.as_ref(),
                "dataType": "SparseFloatVector"
            }),
            MilvusFieldSpec::Int64 { name } => serde_json::json!({
                "fieldName": name.as_ref(),
                "dataType": "Int64"
            }),
        })
        .collect::<Vec<_>>();

    let functions = spec
        .functions
        .iter()
        .map(|func| {
            serde_json::json!({
                "name": func.name.as_ref(),
                "description": func.description.as_ref(),
                "type": func.kind.as_ref(),
                "inputFieldNames": func
                    .input_field_names
                    .iter()
                    .map(AsRef::as_ref)
                    .collect::<Vec<_>>(),
                "outputFieldNames": func
                    .output_field_names
                    .iter()
                    .map(AsRef::as_ref)
                    .collect::<Vec<_>>(),
                "params": func.params,
            })
        })
        .collect::<Vec<_>>();

    serde_json::json!({
        "fields": fields,
        "functions": functions,
    })
}
