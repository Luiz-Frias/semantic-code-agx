use super::{SCHEMA_VERSION, disk_error, legacy_suffix, now_epoch_ms};
use semantic_code_ports::embedding::EmbeddingVector;
use semantic_code_shared::Result;
use sqlx::{Postgres, Row};
use std::sync::Arc;
use tokio::sync::OnceCell;

#[derive(Debug)]
pub(super) struct PostgresCache {
    pool: sqlx::Pool<Postgres>,
    table: String,
    meta_table: String,
    max_bytes: Option<u64>,
    init: OnceCell<()>,
}

impl PostgresCache {
    pub(crate) fn new(connection: &str, table: &str, max_bytes: Option<u64>) -> Result<Self> {
        let pool = sqlx::Pool::<Postgres>::connect_lazy(connection)
            .map_err(|error| disk_error(&format!("postgres cache connect failed: {error}")))?;
        Ok(Self {
            pool,
            table: table.to_string(),
            meta_table: format!("{table}_meta"),
            max_bytes,
            init: OnceCell::new(),
        })
    }

    async fn ensure_init(&self) -> Result<()> {
        self.init
            .get_or_try_init(|| async { self.init_schema().await })
            .await?;
        Ok(())
    }

    async fn init_schema(&self) -> Result<()> {
        let expected = SCHEMA_VERSION.to_string();
        loop {
            let create_meta = format!(
                "CREATE TABLE IF NOT EXISTS {} (meta_key TEXT PRIMARY KEY, meta_value TEXT NOT NULL)",
                self.meta_table
            );
            sqlx::query(&create_meta)
                .execute(&self.pool)
                .await
                .map_err(|error| {
                    disk_error(&format!("postgres cache meta create failed: {error}"))
                })?;

            let create_table = format!(
                "CREATE TABLE IF NOT EXISTS {} (
                    cache_key TEXT PRIMARY KEY,
                    vector_json TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    size_bytes BIGINT NOT NULL,
                    created_at_ms BIGINT NOT NULL,
                    last_accessed_ms BIGINT NOT NULL
                )",
                self.table
            );
            sqlx::query(&create_table)
                .execute(&self.pool)
                .await
                .map_err(|error| disk_error(&format!("postgres cache schema failed: {error}")))?;

            let version_query = format!(
                "SELECT meta_value FROM {} WHERE meta_key = $1",
                self.meta_table
            );
            let version: Option<String> = sqlx::query_scalar(&version_query)
                .bind("schema_version")
                .fetch_optional(&self.pool)
                .await
                .map_err(|error| {
                    disk_error(&format!("postgres cache version read failed: {error}"))
                })?;

            match version {
                None => {
                    let insert = format!(
                        "INSERT INTO {} (meta_key, meta_value) VALUES ($1, $2)",
                        self.meta_table
                    );
                    sqlx::query(&insert)
                        .bind("schema_version")
                        .bind(&expected)
                        .execute(&self.pool)
                        .await
                        .map_err(|error| {
                            disk_error(&format!("postgres cache version set failed: {error}"))
                        })?;
                    return Ok(());
                },
                Some(found) if found != expected => {
                    self.rotate_legacy(&found).await?;
                },
                _ => return Ok(()),
            }
        }
    }

    async fn rotate_legacy(&self, found: &str) -> Result<()> {
        let suffix = legacy_suffix(found);
        let legacy_table = format!("{}_legacy_{}", self.table, suffix);
        let legacy_meta = format!("{}_legacy_{}", self.meta_table, suffix);
        let rename_table = format!("ALTER TABLE {} RENAME TO {}", self.table, legacy_table);
        sqlx::query(&rename_table)
            .execute(&self.pool)
            .await
            .map_err(|error| disk_error(&format!("postgres cache rename failed: {error}")))?;
        let rename_meta = format!("ALTER TABLE {} RENAME TO {}", self.meta_table, legacy_meta);
        sqlx::query(&rename_meta)
            .execute(&self.pool)
            .await
            .map_err(|error| disk_error(&format!("postgres cache meta rename failed: {error}")))?;
        Ok(())
    }

    pub(crate) async fn get(&self, key: &str) -> Result<Option<EmbeddingVector>> {
        self.ensure_init().await?;
        let query = format!(
            "SELECT vector_json, dimension FROM {} WHERE cache_key = $1",
            self.table
        );
        let row = sqlx::query(&query)
            .bind(key)
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| disk_error(&format!("postgres cache query failed: {error}")))?;
        let Some(row) = row else {
            return Ok(None);
        };
        let vector_json: String = row
            .try_get(0)
            .map_err(|error| disk_error(&format!("postgres cache decode failed: {error}")))?;
        let dimension: i32 = row
            .try_get(1)
            .map_err(|error| disk_error(&format!("postgres cache decode failed: {error}")))?;
        let update = format!(
            "UPDATE {} SET last_accessed_ms = $1 WHERE cache_key = $2",
            self.table
        );
        sqlx::query(&update)
            .bind(now_epoch_ms()?)
            .bind(key)
            .execute(&self.pool)
            .await
            .map_err(|error| disk_error(&format!("postgres cache update failed: {error}")))?;
        let vector: Vec<f32> = serde_json::from_str(&vector_json)
            .map_err(|error| disk_error(&format!("postgres cache decode failed: {error}")))?;
        let _ = dimension;
        Ok(Some(EmbeddingVector::new(Arc::from(vector))))
    }

    pub(crate) async fn insert(&self, key: &str, value: &EmbeddingVector) -> Result<()> {
        self.ensure_init().await?;
        let vector_json = serde_json::to_string(value.as_slice())
            .map_err(|error| disk_error(&format!("postgres cache encode failed: {error}")))?;
        let dimension = i32::try_from(value.dimension())
            .map_err(|_| disk_error("postgres cache dimension overflow"))?;
        let size_bytes = i64::try_from(vector_json.len())
            .map_err(|_| disk_error("postgres cache size overflow"))?;
        let now = now_epoch_ms()?;

        let update = format!(
            "UPDATE {} SET vector_json = $1, dimension = $2, size_bytes = $3, last_accessed_ms = $4 WHERE cache_key = $5",
            self.table
        );
        let updated = sqlx::query(&update)
            .bind(&vector_json)
            .bind(dimension)
            .bind(size_bytes)
            .bind(now)
            .bind(key)
            .execute(&self.pool)
            .await
            .map_err(|error| disk_error(&format!("postgres cache update failed: {error}")))?;

        if updated.rows_affected() == 0 {
            let insert = format!(
                "INSERT INTO {} (cache_key, vector_json, dimension, size_bytes, created_at_ms, last_accessed_ms) VALUES ($1, $2, $3, $4, $5, $6)",
                self.table
            );
            sqlx::query(&insert)
                .bind(key)
                .bind(&vector_json)
                .bind(dimension)
                .bind(size_bytes)
                .bind(now)
                .bind(now)
                .execute(&self.pool)
                .await
                .map_err(|error| disk_error(&format!("postgres cache insert failed: {error}")))?;
        }

        if let Some(limit) = self.max_bytes {
            self.evict(limit).await?;
        }
        Ok(())
    }

    async fn evict(&self, max_bytes: u64) -> Result<()> {
        let size_query = format!("SELECT COALESCE(SUM(size_bytes), 0) FROM {}", self.table);
        let mut total: i64 = sqlx::query_scalar(&size_query)
            .fetch_one(&self.pool)
            .await
            .map_err(|error| disk_error(&format!("postgres cache size failed: {error}")))?;
        if total < 0 {
            total = 0;
        }

        while u64::try_from(total).map_err(|_| disk_error("postgres cache size underflow"))?
            > max_bytes
        {
            let candidate_query = format!(
                "SELECT cache_key, size_bytes FROM {} ORDER BY last_accessed_ms ASC LIMIT 1",
                self.table
            );
            let candidate = sqlx::query(&candidate_query)
                .fetch_optional(&self.pool)
                .await
                .map_err(|error| {
                    disk_error(&format!("postgres cache eviction scan failed: {error}"))
                })?;
            let Some(row) = candidate else {
                break;
            };
            let cache_key: String = row
                .try_get(0)
                .map_err(|error| disk_error(&format!("postgres cache decode failed: {error}")))?;
            let size_bytes: i64 = row
                .try_get(1)
                .map_err(|error| disk_error(&format!("postgres cache decode failed: {error}")))?;
            let delete = format!("DELETE FROM {} WHERE cache_key = $1", self.table);
            sqlx::query(&delete)
                .bind(&cache_key)
                .execute(&self.pool)
                .await
                .map_err(|error| {
                    disk_error(&format!("postgres cache eviction delete failed: {error}"))
                })?;
            total = total.saturating_sub(size_bytes).max(0);
        }
        Ok(())
    }
}
