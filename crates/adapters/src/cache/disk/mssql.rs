use super::{SCHEMA_VERSION, disk_error, legacy_suffix, now_epoch_ms};
use futures_util::TryStreamExt;
use semantic_code_ports::embedding::EmbeddingVector;
use semantic_code_shared::Result;
use std::sync::Arc;
use tiberius::{Client, Config, Row as MsRow};
use tokio::net::TcpStream;
use tokio::sync::OnceCell;
use tokio_util::compat::TokioAsyncWriteCompatExt;

#[derive(Debug)]
pub(super) struct MsSqlCache {
    config: Config,
    table: String,
    meta_table: String,
    max_bytes: Option<u64>,
    init: OnceCell<()>,
}

impl MsSqlCache {
    pub(crate) fn new(connection: &str, table: &str, max_bytes: Option<u64>) -> Result<Self> {
        let config = Config::from_ado_string(connection)
            .map_err(|error| disk_error(&format!("mssql cache config failed: {error}")))?;
        Ok(Self {
            config,
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
            let mut client = self.connect().await?;
            let create_meta = format!(
                "IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = '{}') BEGIN CREATE TABLE {} (meta_key NVARCHAR(255) PRIMARY KEY, meta_value NVARCHAR(255) NOT NULL) END",
                self.meta_table, self.meta_table
            );
            exec_mssql_simple(&mut client, &create_meta).await?;

            let create_table = format!(
                "IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = '{}') BEGIN CREATE TABLE {} (
                    cache_key NVARCHAR(255) PRIMARY KEY,
                    vector_json NVARCHAR(MAX) NOT NULL,
                    dimension INT NOT NULL,
                    size_bytes BIGINT NOT NULL,
                    created_at_ms BIGINT NOT NULL,
                    last_accessed_ms BIGINT NOT NULL
                ) END",
                self.table, self.table
            );
            exec_mssql_simple(&mut client, &create_table).await?;

            let version_query = format!(
                "SELECT meta_value FROM {} WHERE meta_key = @P1",
                self.meta_table
            );
            let version =
                query_mssql_rows(&mut client, &version_query, &[&"schema_version"]).await?;
            match version.first() {
                None => {
                    let insert = format!(
                        "INSERT INTO {} (meta_key, meta_value) VALUES (@P1, @P2)",
                        self.meta_table
                    );
                    exec_mssql_query(&mut client, &insert, &[&"schema_version", &expected]).await?;
                    return Ok(());
                },
                Some(row) => {
                    let found = row
                        .get::<&str, _>(0)
                        .map(str::to_string)
                        .ok_or_else(|| disk_error("mssql cache version decode failed"))?;
                    if found == expected {
                        return Ok(());
                    }
                    self.rotate_legacy(&mut client, &found).await?;
                },
            }
        }
    }

    async fn rotate_legacy(
        &self,
        client: &mut Client<tokio_util::compat::Compat<TcpStream>>,
        found: &str,
    ) -> Result<()> {
        let suffix = legacy_suffix(found);
        let legacy_table = format!("{}_legacy_{}", self.table, suffix);
        let legacy_meta = format!("{}_legacy_{}", self.meta_table, suffix);
        let rename_table = format!("EXEC sp_rename '{}', '{}'", self.table, legacy_table);
        exec_mssql_simple(client, &rename_table).await?;
        let rename_meta = format!("EXEC sp_rename '{}', '{}'", self.meta_table, legacy_meta);
        exec_mssql_simple(client, &rename_meta).await?;
        Ok(())
    }

    async fn connect(&self) -> Result<Client<tokio_util::compat::Compat<TcpStream>>> {
        let addr = self.config.get_addr();
        let tcp = TcpStream::connect(addr)
            .await
            .map_err(|error| disk_error(&format!("mssql cache connect failed: {error}")))?;
        tcp.set_nodelay(true)
            .map_err(|error| disk_error(&format!("mssql cache nodelay failed: {error}")))?;
        Client::connect(self.config.clone(), tcp.compat_write())
            .await
            .map_err(|error| disk_error(&format!("mssql cache handshake failed: {error}")))
    }

    pub(crate) async fn get(&self, key: &str) -> Result<Option<EmbeddingVector>> {
        self.ensure_init().await?;
        let mut client = self.connect().await?;
        let query = format!(
            "SELECT vector_json, dimension FROM {} WHERE cache_key = @P1",
            self.table
        );
        let rows = query_mssql_rows(&mut client, &query, &[&key]).await?;
        let Some(row) = rows.first() else {
            return Ok(None);
        };
        let vector_json = row
            .get::<&str, _>(0)
            .map(str::to_string)
            .ok_or_else(|| disk_error("mssql cache decode failed"))?;
        let dimension = row
            .get::<i32, _>(1)
            .ok_or_else(|| disk_error("mssql cache decode failed"))?;
        let update = format!(
            "UPDATE {} SET last_accessed_ms = @P1 WHERE cache_key = @P2",
            self.table
        );
        exec_mssql_query(&mut client, &update, &[&now_epoch_ms()?, &key]).await?;
        let vector: Vec<f32> = serde_json::from_str(&vector_json)
            .map_err(|error| disk_error(&format!("mssql cache decode failed: {error}")))?;
        let _ = dimension;
        Ok(Some(EmbeddingVector::new(Arc::from(vector))))
    }

    pub(crate) async fn insert(&self, key: &str, value: &EmbeddingVector) -> Result<()> {
        self.ensure_init().await?;
        let mut client = self.connect().await?;
        let vector_json = serde_json::to_string(value.as_slice())
            .map_err(|error| disk_error(&format!("mssql cache encode failed: {error}")))?;
        let dimension = i32::try_from(value.dimension())
            .map_err(|_| disk_error("mssql cache dimension overflow"))?;
        let size_bytes = i64::try_from(vector_json.len())
            .map_err(|_| disk_error("mssql cache size overflow"))?;
        let now = now_epoch_ms()?;

        let update = format!(
            "UPDATE {} SET vector_json = @P1, dimension = @P2, size_bytes = @P3, last_accessed_ms = @P4 WHERE cache_key = @P5",
            self.table
        );
        let updated = exec_mssql_query(
            &mut client,
            &update,
            &[&vector_json, &dimension, &size_bytes, &now, &key],
        )
        .await?;

        if updated == 0 {
            let insert = format!(
                "INSERT INTO {} (cache_key, vector_json, dimension, size_bytes, created_at_ms, last_accessed_ms) VALUES (@P1, @P2, @P3, @P4, @P5, @P6)",
                self.table
            );
            exec_mssql_query(
                &mut client,
                &insert,
                &[&key, &vector_json, &dimension, &size_bytes, &now, &now],
            )
            .await?;
        }

        if let Some(limit) = self.max_bytes {
            self.evict(&mut client, limit).await?;
        }
        Ok(())
    }

    async fn evict(
        &self,
        client: &mut Client<tokio_util::compat::Compat<TcpStream>>,
        max_bytes: u64,
    ) -> Result<()> {
        let size_query = format!("SELECT COALESCE(SUM(size_bytes), 0) FROM {}", self.table);
        let rows = query_mssql_rows(client, &size_query, &[]).await?;
        let mut total: i64 = rows
            .first()
            .and_then(|row| row.get::<i64, _>(0))
            .unwrap_or(0);
        if total < 0 {
            total = 0;
        }

        while u64::try_from(total).map_err(|_| disk_error("mssql cache size underflow"))?
            > max_bytes
        {
            let candidate_query = format!(
                "SELECT TOP (1) cache_key, size_bytes FROM {} ORDER BY last_accessed_ms ASC",
                self.table
            );
            let rows = query_mssql_rows(client, &candidate_query, &[]).await?;
            let Some(row) = rows.first() else {
                break;
            };
            let cache_key = row
                .get::<&str, _>(0)
                .map(str::to_string)
                .ok_or_else(|| disk_error("mssql cache decode failed"))?;
            let size_bytes = row
                .get::<i64, _>(1)
                .ok_or_else(|| disk_error("mssql cache decode failed"))?;
            let delete = format!("DELETE FROM {} WHERE cache_key = @P1", self.table);
            exec_mssql_query(client, &delete, &[&cache_key]).await?;
            total = total.saturating_sub(size_bytes).max(0);
        }
        Ok(())
    }
}

async fn exec_mssql_simple(
    client: &mut Client<tokio_util::compat::Compat<TcpStream>>,
    query: &str,
) -> Result<()> {
    let mut stream = client
        .simple_query(query)
        .await
        .map_err(|error| disk_error(&format!("mssql cache query failed: {error}")))?;
    while stream
        .try_next()
        .await
        .map_err(|error| disk_error(&format!("mssql cache query failed: {error}")))?
        .is_some()
    {}
    Ok(())
}

async fn exec_mssql_query(
    client: &mut Client<tokio_util::compat::Compat<TcpStream>>,
    query: &str,
    params: &[&dyn tiberius::ToSql],
) -> Result<u64> {
    let result = client
        .execute(query, params)
        .await
        .map_err(|error| disk_error(&format!("mssql cache query failed: {error}")))?;
    Ok(result.total())
}

async fn query_mssql_rows(
    client: &mut Client<tokio_util::compat::Compat<TcpStream>>,
    query: &str,
    params: &[&dyn tiberius::ToSql],
) -> Result<Vec<MsRow>> {
    let mut stream = client
        .query(query, params)
        .await
        .map_err(|error| disk_error(&format!("mssql cache query failed: {error}")))?;
    let mut rows = Vec::new();
    while let Some(item) = stream
        .try_next()
        .await
        .map_err(|error| disk_error(&format!("mssql cache query failed: {error}")))?
    {
        if let tiberius::QueryItem::Row(row) = item {
            rows.push(row);
        }
    }
    Ok(rows)
}
