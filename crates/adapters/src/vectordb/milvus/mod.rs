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

#[cfg(feature = "milvus-grpc")]
pub use grpc::{MilvusGrpcConfig, MilvusGrpcVectorDb};
pub use index::{MilvusIndexConfig, MilvusIndexSpec};

#[cfg(feature = "milvus-rest")]
pub use rest::{MilvusRestConfig, MilvusRestVectorDb};
