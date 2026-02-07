//! Vector database adapters.

#[cfg(any(feature = "milvus-grpc", feature = "milvus-rest"))]
pub mod milvus;

pub mod fixed;
