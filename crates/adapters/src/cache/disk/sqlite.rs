use super::{SCHEMA_VERSION, disk_error, legacy_suffix, now_epoch_ms};
use rusqlite::{Connection, OptionalExtension};
use semantic_code_ports::embedding::EmbeddingVector;
use semantic_code_shared::{ErrorEnvelope, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::task::spawn_blocking;

#[derive(Debug)]
pub(super) struct SqliteCache {
    path: PathBuf,
    max_bytes: Option<u64>,
}

impl SqliteCache {
    pub(crate) const fn new(path: PathBuf, max_bytes: Option<u64>) -> Self {
        Self { path, max_bytes }
    }

    pub(crate) async fn get(&self, key: &str) -> Result<Option<EmbeddingVector>> {
        let path = self.path.clone();
        let key = key.to_owned();
        let value = spawn_blocking(move || {
            let conn = open_connection(&path)?;
            let row: Option<(String, u32)> = conn
                .query_row(
                    "SELECT vector_json, dimension FROM embeddings WHERE cache_key = ?1",
                    [&key],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()
                .map_err(|error| disk_error(&format!("disk cache query failed: {error}")))?;

            let Some((vector_json, dimension)) = row else {
                return Ok::<Option<EmbeddingVector>, ErrorEnvelope>(None);
            };

            let now = now_epoch_ms()?;
            conn.execute(
                "UPDATE embeddings SET last_accessed_ms = ?1 WHERE cache_key = ?2",
                (now, &key),
            )
            .map_err(|error| disk_error(&format!("disk cache update failed: {error}")))?;

            let vector: Vec<f32> = serde_json::from_str(&vector_json)
                .map_err(|error| disk_error(&format!("disk cache decode failed: {error}")))?;

            let _ = dimension;
            Ok(Some(EmbeddingVector::new(Arc::from(vector))))
        })
        .await
        .map_err(|error| disk_error(&format!("disk cache task failed: {error}")))??;
        Ok(value)
    }

    pub(crate) async fn insert(&self, key: &str, value: &EmbeddingVector) -> Result<()> {
        let path = self.path.clone();
        let key = key.to_owned();
        let vector_json = serde_json::to_string(value.as_slice())
            .map_err(|error| disk_error(&format!("disk cache encode failed: {error}")))?;
        let dimension = value.dimension();
        let size_bytes = vector_json.len() as u64;
        let max_bytes = self.max_bytes;

        spawn_blocking(move || {
            let conn = open_connection(&path)?;
            let now = now_epoch_ms()?;
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (cache_key, vector_json, dimension, size_bytes, created_at_ms, last_accessed_ms) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                (
                    &key,
                    &vector_json,
                    dimension,
                    size_bytes,
                    now,
                    now,
                ),
            )
            .map_err(|error| disk_error(&format!("disk cache insert failed: {error}")))?;

            if let Some(limit) = max_bytes {
                evict_disk_cache(&conn, limit)?;
            }

            Ok::<(), ErrorEnvelope>(())
        })
        .await
        .map_err(|error| disk_error(&format!("disk cache task failed: {error}")))??;

        Ok(())
    }
}

fn open_connection(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| disk_error(&format!("disk cache mkdir failed: {error}")))?;
    }

    let conn = Connection::open(path)
        .map_err(|error| disk_error(&format!("disk cache open failed: {error}")))?;

    conn.execute_batch("PRAGMA journal_mode = WAL;")
        .map_err(|error| disk_error(&format!("disk cache pragma failed: {error}")))?;

    init_sqlite_schema(&conn)?;

    let version: i64 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|error| disk_error(&format!("disk cache version failed: {error}")))?;

    if version == 0 {
        conn.execute("PRAGMA user_version = ?1", [SCHEMA_VERSION.to_string()])
            .map_err(|error| disk_error(&format!("disk cache version set failed: {error}")))?;
        return Ok(conn);
    }

    if version != SCHEMA_VERSION {
        drop(conn);
        rotate_sqlite_legacy(path, version)?;
        return open_connection_fresh(path);
    }

    Ok(conn)
}

fn evict_disk_cache(conn: &Connection, max_bytes: u64) -> Result<()> {
    let mut total: u64 = conn
        .query_row(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM embeddings",
            [],
            |row| row.get(0),
        )
        .map_err(|error| disk_error(&format!("disk cache size failed: {error}")))?;

    while total > max_bytes {
        let candidate: Option<(String, u64)> = conn
            .query_row(
                "SELECT cache_key, size_bytes FROM embeddings ORDER BY last_accessed_ms ASC LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|error| disk_error(&format!("disk cache eviction scan failed: {error}")))?;

        let Some((key, size_bytes)) = candidate else {
            break;
        };

        conn.execute("DELETE FROM embeddings WHERE cache_key = ?1", [&key])
            .map_err(|error| disk_error(&format!("disk cache eviction delete failed: {error}")))?;
        total = total.saturating_sub(size_bytes);
    }

    Ok(())
}

fn init_sqlite_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS meta (
            meta_key TEXT PRIMARY KEY,
            meta_value TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS embeddings (
            cache_key TEXT PRIMARY KEY,
            vector_json TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            size_bytes INTEGER NOT NULL,
            created_at_ms INTEGER NOT NULL,
            last_accessed_ms INTEGER NOT NULL
         );",
    )
    .map_err(|error| disk_error(&format!("disk cache schema failed: {error}")))?;
    Ok(())
}

fn open_connection_fresh(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path)
        .map_err(|error| disk_error(&format!("disk cache open failed: {error}")))?;
    conn.execute_batch("PRAGMA journal_mode = WAL;")
        .map_err(|error| disk_error(&format!("disk cache pragma failed: {error}")))?;
    init_sqlite_schema(&conn)?;
    conn.execute("PRAGMA user_version = ?1", [SCHEMA_VERSION.to_string()])
        .map_err(|error| disk_error(&format!("disk cache version set failed: {error}")))?;
    Ok(conn)
}

fn rotate_sqlite_legacy(path: &Path, version: i64) -> Result<()> {
    let suffix = legacy_suffix(&version.to_string());
    let file_name = path
        .file_name()
        .ok_or_else(|| disk_error("disk cache path missing filename"))?
        .to_string_lossy();
    let legacy_name = format!("{file_name}.legacy.{suffix}");
    let legacy_path = path.with_file_name(legacy_name);
    if path.exists() {
        std::fs::rename(path, &legacy_path)
            .map_err(|error| disk_error(&format!("disk cache rotate failed: {error}")))?;
    }
    let wal_path = path.with_file_name(format!("{file_name}-wal"));
    if wal_path.exists() {
        let legacy_wal = wal_path.with_file_name(format!("{file_name}-wal.legacy.{suffix}"));
        std::fs::rename(&wal_path, &legacy_wal)
            .map_err(|error| disk_error(&format!("disk cache wal rotate failed: {error}")))?;
    }
    let shm_path = path.with_file_name(format!("{file_name}-shm"));
    if shm_path.exists() {
        let legacy_shm = shm_path.with_file_name(format!("{file_name}-shm.legacy.{suffix}"));
        std::fs::rename(&shm_path, &legacy_shm)
            .map_err(|error| disk_error(&format!("disk cache shm rotate failed: {error}")))?;
    }
    Ok(())
}
