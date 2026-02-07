use semantic_code_ports::embedding::EmbeddingVector;
use semantic_code_shared::{ErrorCode, ErrorEnvelope, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct MemoryCache {
    max_entries: usize,
    max_bytes: u64,
    state: tokio::sync::Mutex<CacheState>,
}

#[derive(Debug)]
struct CacheState {
    entries: HashMap<Box<str>, CacheEntry>,
    order: VecDeque<Box<str>>,
    total_bytes: u64,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    vector: Arc<[f32]>,
    size_bytes: u64,
}

impl MemoryCache {
    pub(crate) fn new(max_entries: usize, max_bytes: u64) -> Result<Self> {
        if max_entries == 0 {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "cache max_entries must be greater than zero",
            ));
        }
        if max_bytes == 0 {
            return Err(ErrorEnvelope::expected(
                ErrorCode::invalid_input(),
                "cache max_bytes must be greater than zero",
            ));
        }
        Ok(Self {
            max_entries,
            max_bytes,
            state: tokio::sync::Mutex::new(CacheState {
                entries: HashMap::new(),
                order: VecDeque::new(),
                total_bytes: 0,
            }),
        })
    }

    const fn estimate_size(vector: &[f32]) -> u64 {
        (vector.len() as u64).saturating_mul(4)
    }

    pub(crate) async fn get(&self, key: &str) -> Option<EmbeddingVector> {
        let entry = {
            let mut state = self.state.lock().await;
            let entry = state.entries.get(key)?.clone();
            Self::touch(&mut state, key);
            drop(state);
            entry
        };
        Some(EmbeddingVector::new(entry.vector))
    }

    pub(crate) async fn insert(&self, key: &str, value: EmbeddingVector) {
        let mut state = self.state.lock().await;
        let key_box: Box<str> = key.to_owned().into_boxed_str();
        let vector = value.into_vector();
        let size = Self::estimate_size(vector.as_ref());

        if let Some(existing) = state.entries.insert(
            key_box,
            CacheEntry {
                vector,
                size_bytes: size,
            },
        ) {
            state.total_bytes = state.total_bytes.saturating_sub(existing.size_bytes);
        }

        state.total_bytes = state.total_bytes.saturating_add(size);
        Self::touch(&mut state, key);
        Self::evict(&mut state, self.max_entries, self.max_bytes);
        drop(state);
    }

    fn touch(state: &mut CacheState, key: &str) {
        if let Some(pos) = state.order.iter().position(|k| k.as_ref() == key) {
            state.order.remove(pos);
        }
        state.order.push_back(key.to_owned().into_boxed_str());
    }

    fn evict(state: &mut CacheState, max_entries: usize, max_bytes: u64) {
        while state.entries.len() > max_entries || state.total_bytes > max_bytes {
            let Some(oldest) = state.order.pop_front() else {
                break;
            };
            if let Some(entry) = state.entries.remove(&oldest) {
                state.total_bytes = state.total_bytes.saturating_sub(entry.size_bytes);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{CacheSource, DiskCacheProvider, EmbeddingCache, EmbeddingCacheConfig};
    use semantic_code_ports::embedding::EmbeddingVector;
    use semantic_code_shared::Result;

    #[tokio::test]
    async fn cache_roundtrip_memory() -> Result<()> {
        let cache_config = EmbeddingCacheConfig {
            enabled: true,
            max_entries: 2,
            max_bytes: 1024,
            disk_enabled: false,
            disk_provider: DiskCacheProvider::Sqlite,
            disk_path: None,
            disk_connection: None,
            disk_table: None,
            disk_max_bytes: None,
        };
        let cache = EmbeddingCache::new(&cache_config)?;

        let key = EmbeddingCache::make_key("test", "hello");
        let value = EmbeddingVector::from_vec(vec![1.0, 2.0, 3.0]);
        cache.insert(&key, value.clone()).await?;

        let lookup = cache.get(&key).await?.expect("cache hit");
        assert_eq!(lookup.source, CacheSource::Memory);
        assert_eq!(lookup.value, value);
        Ok(())
    }

    #[tokio::test]
    async fn cache_eviction_by_entries() -> Result<()> {
        let cache_config = EmbeddingCacheConfig {
            enabled: true,
            max_entries: 1,
            max_bytes: 1024,
            disk_enabled: false,
            disk_provider: DiskCacheProvider::Sqlite,
            disk_path: None,
            disk_connection: None,
            disk_table: None,
            disk_max_bytes: None,
        };
        let cache = EmbeddingCache::new(&cache_config)?;

        let first = EmbeddingCache::make_key("test", "a");
        let second = EmbeddingCache::make_key("test", "b");

        cache
            .insert(&first, EmbeddingVector::from_vec(vec![1.0]))
            .await?;
        cache
            .insert(&second, EmbeddingVector::from_vec(vec![2.0]))
            .await?;

        assert!(cache.get(&first).await?.is_none());
        assert!(cache.get(&second).await?.is_some());
        Ok(())
    }

    #[tokio::test]
    async fn cache_lookup_contains_expected_source() -> Result<()> {
        let cache_config = EmbeddingCacheConfig {
            enabled: true,
            max_entries: 2,
            max_bytes: 1024,
            disk_enabled: false,
            disk_provider: DiskCacheProvider::Sqlite,
            disk_path: None,
            disk_connection: None,
            disk_table: None,
            disk_max_bytes: None,
        };
        let cache = EmbeddingCache::new(&cache_config)?;
        let key = EmbeddingCache::make_key("test", "source");
        let value = EmbeddingVector::from_vec(vec![1.0, 2.0]);
        cache.insert(&key, value).await?;

        let lookup = cache.get(&key).await?.expect("cache hit");
        assert_eq!(lookup.source, CacheSource::Memory);
        Ok(())
    }
}
