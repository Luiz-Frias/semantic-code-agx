//! Embedding cache with optional disk persistence.

mod config;
mod disk;
mod embedding;
mod memory;

pub use config::{DiskCacheProvider, EmbeddingCacheConfig};
pub use embedding::CachingEmbedding;

use disk::DiskCache;
use memory::MemoryCache;
use semantic_code_ports::embedding::EmbeddingVector;
use semantic_code_shared::Result;
use sha2::{Digest, Sha256};

/// Cache source for telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheSource {
    /// In-memory cache hit.
    Memory,
    /// Disk cache hit.
    Disk,
}

/// Cache lookup result.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheLookup {
    /// Cached embedding vector.
    pub value: EmbeddingVector,
    /// Where the value was found.
    pub source: CacheSource,
}

/// Embedding cache with optional disk persistence.
#[derive(Debug)]
pub struct EmbeddingCache {
    memory: Option<MemoryCache>,
    disk: Option<DiskCache>,
}

impl EmbeddingCache {
    /// Create a new cache from config.
    pub fn new(config: &EmbeddingCacheConfig) -> Result<Self> {
        let memory = if config.enabled {
            Some(MemoryCache::new(config.max_entries, config.max_bytes)?)
        } else {
            None
        };
        let disk = if config.disk_enabled {
            Some(DiskCache::new(
                config.disk_provider,
                config.disk_path.clone(),
                config.disk_connection.as_deref(),
                config.disk_table.as_deref(),
                config.disk_max_bytes,
            )?)
        } else {
            None
        };
        Ok(Self { memory, disk })
    }

    /// Compute a stable cache key for an embedding payload.
    #[must_use]
    pub fn make_key(namespace: &str, text: &str) -> Box<str> {
        let mut hasher = Sha256::new();
        hasher.update(namespace.as_bytes());
        hasher.update([0u8]);
        hasher.update(text.as_bytes());
        let hash = hasher.finalize();
        format!("{hash:x}").into_boxed_str()
    }

    /// Read from cache.
    pub async fn get(&self, key: &str) -> Result<Option<CacheLookup>> {
        if let Some(memory) = &self.memory
            && let Some(value) = memory.get(key).await
        {
            return Ok(Some(CacheLookup {
                value,
                source: CacheSource::Memory,
            }));
        }

        if let Some(disk) = &self.disk
            && let Some(value) = disk.get(key).await?
        {
            if let Some(memory) = &self.memory {
                memory.insert(key, value.clone()).await;
            }
            return Ok(Some(CacheLookup {
                value,
                source: CacheSource::Disk,
            }));
        }

        Ok(None)
    }

    /// Insert into cache.
    pub async fn insert(&self, key: &str, value: EmbeddingVector) -> Result<()> {
        if let Some(memory) = &self.memory {
            memory.insert(key, value.clone()).await;
        }
        if let Some(disk) = &self.disk {
            disk.insert(key, &value).await?;
        }
        Ok(())
    }
}
