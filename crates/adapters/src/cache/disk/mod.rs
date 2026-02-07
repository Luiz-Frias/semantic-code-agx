use super::config::DiskCacheProvider;
use semantic_code_ports::embedding::EmbeddingVector;
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, Result};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "cache-mssql")]
mod mssql;
#[cfg(feature = "cache-mysql")]
mod mysql;
#[cfg(feature = "cache-postgres")]
mod postgres;
mod sqlite;

pub(super) const SCHEMA_VERSION: i64 = 2;

#[derive(Debug)]
pub(super) struct DiskCache {
    backend: DiskCacheBackend,
}

#[derive(Debug)]
enum DiskCacheBackend {
    Sqlite(sqlite::SqliteCache),
    #[cfg(feature = "cache-postgres")]
    Postgres(postgres::PostgresCache),
    #[cfg(feature = "cache-mysql")]
    Mysql(mysql::MySqlCache),
    #[cfg(feature = "cache-mssql")]
    Mssql(mssql::MsSqlCache),
}

impl DiskCache {
    pub(crate) fn new(
        provider: DiskCacheProvider,
        path: Option<PathBuf>,
        connection: Option<&str>,
        table: Option<&str>,
        max_bytes: Option<u64>,
    ) -> Result<Self> {
        let _ = (&connection, &table);
        let backend = match provider {
            DiskCacheProvider::Sqlite => {
                let path =
                    path.ok_or_else(|| disk_error("disk path is required for sqlite cache"))?;
                DiskCacheBackend::Sqlite(sqlite::SqliteCache::new(path, max_bytes))
            },
            DiskCacheProvider::Postgres => {
                #[cfg(feature = "cache-postgres")]
                {
                    let connection = connection.ok_or_else(|| {
                        disk_error("disk connection string is required for postgres cache")
                    })?;
                    let table = table.unwrap_or("embedding_cache");
                    DiskCacheBackend::Postgres(postgres::PostgresCache::new(
                        connection, table, max_bytes,
                    )?)
                }
                #[cfg(not(feature = "cache-postgres"))]
                {
                    return Err(disk_error("postgres cache provider not enabled"));
                }
            },
            DiskCacheProvider::Mysql => {
                #[cfg(feature = "cache-mysql")]
                {
                    let connection = connection.ok_or_else(|| {
                        disk_error("disk connection string is required for mysql cache")
                    })?;
                    let table = table.unwrap_or("embedding_cache");
                    DiskCacheBackend::Mysql(mysql::MySqlCache::new(connection, table, max_bytes)?)
                }
                #[cfg(not(feature = "cache-mysql"))]
                {
                    return Err(disk_error("mysql cache provider not enabled"));
                }
            },
            DiskCacheProvider::Mssql => {
                #[cfg(feature = "cache-mssql")]
                {
                    let connection = connection.ok_or_else(|| {
                        disk_error("disk connection string is required for mssql cache")
                    })?;
                    let table = table.unwrap_or("embedding_cache");
                    DiskCacheBackend::Mssql(mssql::MsSqlCache::new(connection, table, max_bytes)?)
                }
                #[cfg(not(feature = "cache-mssql"))]
                {
                    return Err(disk_error("mssql cache provider not enabled"));
                }
            },
        };

        Ok(Self { backend })
    }

    pub(crate) async fn get(&self, key: &str) -> Result<Option<EmbeddingVector>> {
        match &self.backend {
            DiskCacheBackend::Sqlite(cache) => cache.get(key).await,
            #[cfg(feature = "cache-postgres")]
            DiskCacheBackend::Postgres(cache) => cache.get(key).await,
            #[cfg(feature = "cache-mysql")]
            DiskCacheBackend::Mysql(cache) => cache.get(key).await,
            #[cfg(feature = "cache-mssql")]
            DiskCacheBackend::Mssql(cache) => cache.get(key).await,
        }
    }

    pub(crate) async fn insert(&self, key: &str, value: &EmbeddingVector) -> Result<()> {
        match &self.backend {
            DiskCacheBackend::Sqlite(cache) => cache.insert(key, value).await,
            #[cfg(feature = "cache-postgres")]
            DiskCacheBackend::Postgres(cache) => cache.insert(key, value).await,
            #[cfg(feature = "cache-mysql")]
            DiskCacheBackend::Mysql(cache) => cache.insert(key, value).await,
            #[cfg(feature = "cache-mssql")]
            DiskCacheBackend::Mssql(cache) => cache.insert(key, value).await,
        }
    }
}

pub(super) fn now_epoch_ms() -> Result<i64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
        .map_err(|error| disk_error(&format!("disk cache clock error: {error}")))
}

pub(super) fn legacy_suffix(found_version: &str) -> String {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    format!("{found_version}_{stamp}")
}

pub(super) fn disk_error(message: &str) -> ErrorEnvelope {
    ErrorEnvelope::unexpected(
        ErrorCode::new("embedding", "cache_disk"),
        message.to_string(),
        ErrorClass::NonRetriable,
    )
}
