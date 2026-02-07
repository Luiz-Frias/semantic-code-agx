//! Milvus vector database adapters (gRPC and REST).

mod error;
mod index;
mod metadata;
mod schema;
mod shared;

#[cfg(feature = "milvus-rest")]
mod rest;
#[cfg(feature = "milvus-rest")]
mod rest_auth;
#[cfg(feature = "milvus-rest")]
mod rest_base_url;

#[cfg(feature = "milvus-grpc")]
mod grpc;
#[cfg(feature = "milvus-grpc")]
mod proto;

pub use error::{MilvusErrorContext, MilvusProviderId, map_rest_error};
pub use index::{MilvusIndexConfig, MilvusIndexSpec};
pub use metadata::{parse_metadata, serialize_metadata};
pub use schema::{MilvusSchemaSpec, build_dense_schema_spec, build_hybrid_schema_spec};

#[cfg(feature = "milvus-grpc")]
pub use error::map_grpc_error;
#[cfg(feature = "milvus-grpc")]
pub use grpc::{MilvusGrpcConfig, MilvusGrpcVectorDb};

#[cfg(feature = "milvus-rest")]
pub use rest::{MilvusRestConfig, MilvusRestVectorDb};
