use std::path::PathBuf;

/// Cache configuration for embeddings.
#[derive(Debug, Clone)]
pub struct EmbeddingCacheConfig {
    /// Enable in-memory cache.
    pub enabled: bool,
    /// Maximum number of entries in memory.
    pub max_entries: usize,
    /// Maximum memory bytes to keep.
    pub max_bytes: u64,
    /// Enable disk cache.
    pub disk_enabled: bool,
    /// Disk cache provider.
    pub disk_provider: DiskCacheProvider,
    /// Optional disk path for the `SQLite` cache.
    pub disk_path: Option<PathBuf>,
    /// Optional disk connection string for database-backed caches.
    pub disk_connection: Option<Box<str>>,
    /// Optional disk cache table name override.
    pub disk_table: Option<Box<str>>,
    /// Maximum disk cache size in bytes.
    pub disk_max_bytes: Option<u64>,
}

/// Disk cache provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiskCacheProvider {
    /// `SQLite` file-backed cache.
    Sqlite,
    /// Postgres-backed cache.
    Postgres,
    /// MySQL-backed cache.
    Mysql,
    /// Microsoft SQL Server-backed cache.
    Mssql,
}
