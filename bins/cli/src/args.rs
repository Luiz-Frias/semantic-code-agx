//! CLI argument enums, override structs, and JSON override builders.
//!
//! This module defines the public CLI contract — every subcommand variant,
//! their flag shapes, and the conversion logic that turns CLI flags into
//! the JSON override strings consumed by the config layer.

use crate::error::CliError;
use clap::Subcommand;
use semantic_code_facade::RequestKind;
use std::path::PathBuf;

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Validate the build and environment wiring.
    #[cfg(any(debug_assertions, feature = "dev-tools"))]
    SelfCheck,
    /// Show build and version details.
    Info,
    /// Print the machine-readable agent protocol spec (YAML).
    #[command(after_help = "Output is always YAML regardless of --output flag.")]
    AgentDoc {
        /// Scope output to a specific command (e.g., `search`, `index`).
        #[arg(value_name = "COMMAND")]
        command: Option<String>,
    },
    /// Config-related commands.
    Config {
        #[command(subcommand)]
        command: ConfigCommands,
    },
    /// Background job commands.
    Jobs {
        #[command(subcommand)]
        command: JobsCommands,
    },
    /// Initialize config and manifest for a codebase.
    #[command(after_help = "Agents: run `sca agent-doc init` for this command's protocol spec.")]
    Init {
        /// Optional config file path (JSON/TOML). Defaults to `.context/config.toml`.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Codebase root (defaults to current directory).
        #[arg(long)]
        codebase_root: Option<PathBuf>,
        /// Snapshot storage mode: disabled, project, or custom:<path>.
        #[arg(long)]
        storage_mode: Option<String>,
        /// Overwrite existing config/manifest when present.
        #[arg(long)]
        force: bool,
    },
    /// Estimate index storage requirements and local free-space headroom.
    EstimateStorage {
        /// Optional config file path (JSON/TOML). Defaults to `.context/config.toml` when present.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Codebase root (defaults to current directory).
        #[arg(long)]
        codebase_root: Option<PathBuf>,
        /// Embedding provider (e.g. `onnx`, `openai`, `gemini`).
        #[arg(long)]
        embedding_provider: Option<String>,
        /// Embedding model override.
        #[arg(long)]
        embedding_model: Option<String>,
        /// Embedding base URL override.
        #[arg(long)]
        embedding_base_url: Option<String>,
        /// Embedding dimension override.
        #[arg(long)]
        embedding_dimension: Option<u32>,
        /// Prefer local embeddings over remote providers.
        #[arg(long)]
        embedding_local_first: Option<bool>,
        /// Force local embeddings only.
        #[arg(long)]
        embedding_local_only: Option<bool>,
        /// Embedding routing mode (`localFirst`, `remoteFirst`, `split`).
        #[arg(long)]
        embedding_routing_mode: Option<String>,
        /// Max remote batches for split routing.
        #[arg(long)]
        embedding_split_remote_batches: Option<u32>,
        /// Vector DB provider (e.g. `local`, `milvus_grpc`, `milvus_rest`).
        #[arg(long)]
        vector_db_provider: Option<String>,
        /// Local vector kernel (`hnsw-rs`, `dfrr`).
        #[arg(long)]
        vector_kernel: Option<String>,
        /// Vector DB address/host.
        #[arg(long)]
        vector_db_address: Option<String>,
        /// Vector DB base URL.
        #[arg(long)]
        vector_db_base_url: Option<String>,
        /// Vector DB database name.
        #[arg(long)]
        vector_db_database: Option<String>,
        /// Vector DB SSL enablement.
        #[arg(long)]
        vector_db_ssl: Option<bool>,
        /// Vector DB auth token.
        #[arg(long)]
        vector_db_token: Option<String>,
        /// Vector DB auth username.
        #[arg(long)]
        vector_db_username: Option<String>,
        /// Vector DB auth password.
        #[arg(long)]
        vector_db_password: Option<String>,
        /// Use a tighter storage factor (1.25x estimate) for emergency runs.
        #[arg(long, hide = true)]
        danger_close_storage: bool,
    },
    /// Index the local codebase.
    #[command(after_help = "Agents: run `sca agent-doc index` for this command's protocol spec.")]
    Index {
        /// Optional config file path (JSON/TOML). Defaults to `.context/config.toml` when present.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Codebase root (defaults to current directory).
        #[arg(long)]
        codebase_root: Option<PathBuf>,
        /// Initialize local manifest if missing.
        #[arg(long)]
        init: bool,
        /// Run indexing in the background (returns a job id).
        #[arg(long)]
        background: bool,
        /// Embedding provider (e.g. `onnx`, `openai`, `gemini`).
        #[arg(long)]
        embedding_provider: Option<String>,
        /// Embedding model override.
        #[arg(long)]
        embedding_model: Option<String>,
        /// Embedding base URL override.
        #[arg(long)]
        embedding_base_url: Option<String>,
        /// Embedding dimension override.
        #[arg(long)]
        embedding_dimension: Option<u32>,
        /// Prefer local embeddings over remote providers.
        #[arg(long)]
        embedding_local_first: Option<bool>,
        /// Force local embeddings only.
        #[arg(long)]
        embedding_local_only: Option<bool>,
        /// Embedding routing mode (`localFirst`, `remoteFirst`, `split`).
        #[arg(long)]
        embedding_routing_mode: Option<String>,
        /// Max remote batches for split routing.
        #[arg(long)]
        embedding_split_remote_batches: Option<u32>,
        /// Vector DB provider (e.g. `local`, `milvus_grpc`, `milvus_rest`).
        #[arg(long)]
        vector_db_provider: Option<String>,
        /// Local vector kernel (`hnsw-rs`, `dfrr`).
        #[arg(long)]
        vector_kernel: Option<String>,
        /// Vector DB address/host.
        #[arg(long)]
        vector_db_address: Option<String>,
        /// Vector DB base URL.
        #[arg(long)]
        vector_db_base_url: Option<String>,
        /// Vector DB database name.
        #[arg(long)]
        vector_db_database: Option<String>,
        /// Vector DB SSL enablement.
        #[arg(long)]
        vector_db_ssl: Option<bool>,
        /// Vector DB auth token.
        #[arg(long)]
        vector_db_token: Option<String>,
        /// Vector DB auth username.
        #[arg(long)]
        vector_db_username: Option<String>,
        /// Vector DB auth password.
        #[arg(long)]
        vector_db_password: Option<String>,
        /// Use a tighter storage factor (1.25x estimate) for emergency runs.
        #[arg(long, hide = true)]
        danger_close_storage: bool,
        /// Raw JSON config overrides (takes precedence over individual CLI flags).
        #[arg(long)]
        overrides_json: Option<String>,
    },
    /// Search the local codebase.
    #[command(after_help = "Agents: run `sca agent-doc search` for this command's protocol spec.")]
    Search {
        /// Search query text.
        #[arg(long)]
        query: Option<String>,
        /// Read the query text from stdin.
        #[arg(long, conflicts_with_all = ["query", "stdin_batch"])]
        stdin: bool,
        /// Batch mode: read NDJSON queries from stdin, write NDJSON results to stdout.
        ///
        /// Loads the index once and processes multiple queries without re-loading.
        /// Each input line: `{"query":"...","topK":10,"threshold":0.0}`
        /// Each output line: `{"status":"ok","results":[...],"searchStats":{...}}`
        #[arg(long, conflicts_with_all = ["query", "stdin"])]
        stdin_batch: bool,
        /// Internal: skip embedding initialization and require pre-computed query vectors.
        #[arg(long, hide = true, requires = "stdin_batch")]
        query_vectors_only: bool,
        /// Optional top-k override.
        #[arg(long, alias = "max-results")]
        top_k: Option<u32>,
        /// Optional score threshold.
        #[arg(long)]
        threshold: Option<f32>,
        /// Optional filter expression.
        #[arg(long)]
        filter_expr: Option<String>,
        /// Include content payloads in results.
        #[arg(long)]
        include_content: bool,
        /// Optional config file path (JSON/TOML). Defaults to `.context/config.toml` when present.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Codebase root (defaults to current directory).
        #[arg(long)]
        codebase_root: Option<PathBuf>,
        /// Vector DB provider (e.g. `local`, `milvus_grpc`, `milvus_rest`).
        #[arg(long)]
        vector_db_provider: Option<String>,
        /// Local vector kernel (`hnsw-rs`, `dfrr`).
        #[arg(long)]
        vector_kernel: Option<String>,
        /// Vector DB address/host.
        #[arg(long)]
        vector_db_address: Option<String>,
        /// Vector DB base URL.
        #[arg(long)]
        vector_db_base_url: Option<String>,
        /// Vector DB database name.
        #[arg(long)]
        vector_db_database: Option<String>,
        /// Vector DB SSL enablement.
        #[arg(long)]
        vector_db_ssl: Option<bool>,
        /// Vector DB auth token.
        #[arg(long)]
        vector_db_token: Option<String>,
        /// Vector DB auth username.
        #[arg(long)]
        vector_db_username: Option<String>,
        /// Vector DB auth password.
        #[arg(long)]
        vector_db_password: Option<String>,
        /// Raw JSON config overrides (takes precedence over individual CLI flags).
        #[arg(long)]
        overrides_json: Option<String>,
    },
    /// Clear the local index and snapshot.
    #[command(after_help = "Agents: run `sca agent-doc clear` for this command's protocol spec.")]
    Clear {
        /// Optional config file path (JSON/TOML). Defaults to `.context/config.toml` when present.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Codebase root (defaults to current directory).
        #[arg(long)]
        codebase_root: Option<PathBuf>,
        /// Vector DB provider (e.g. `local`, `milvus_grpc`, `milvus_rest`).
        #[arg(long)]
        vector_db_provider: Option<String>,
        /// Vector DB address/host.
        #[arg(long)]
        vector_db_address: Option<String>,
        /// Vector DB base URL.
        #[arg(long)]
        vector_db_base_url: Option<String>,
        /// Vector DB database name.
        #[arg(long)]
        vector_db_database: Option<String>,
        /// Vector DB SSL enablement.
        #[arg(long)]
        vector_db_ssl: Option<bool>,
        /// Vector DB auth token.
        #[arg(long)]
        vector_db_token: Option<String>,
        /// Vector DB auth username.
        #[arg(long)]
        vector_db_username: Option<String>,
        /// Vector DB auth password.
        #[arg(long)]
        vector_db_password: Option<String>,
    },
    /// Report local index status.
    #[command(after_help = "Agents: run `sca agent-doc status` for this command's protocol spec.")]
    Status {
        /// Optional config file path (JSON/TOML). Defaults to `.context/config.toml` when present.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Codebase root (defaults to current directory).
        #[arg(long)]
        codebase_root: Option<PathBuf>,
        /// Vector DB provider (e.g. `local`, `milvus_grpc`, `milvus_rest`).
        #[arg(long)]
        vector_db_provider: Option<String>,
        /// Vector DB address/host.
        #[arg(long)]
        vector_db_address: Option<String>,
        /// Vector DB base URL.
        #[arg(long)]
        vector_db_base_url: Option<String>,
        /// Vector DB database name.
        #[arg(long)]
        vector_db_database: Option<String>,
        /// Vector DB SSL enablement.
        #[arg(long)]
        vector_db_ssl: Option<bool>,
        /// Vector DB auth token.
        #[arg(long)]
        vector_db_token: Option<String>,
        /// Vector DB auth username.
        #[arg(long)]
        vector_db_username: Option<String>,
        /// Vector DB auth password.
        #[arg(long)]
        vector_db_password: Option<String>,
    },
    /// Reindex based on snapshot changes.
    #[command(after_help = "Agents: run `sca agent-doc reindex` for this command's protocol spec.")]
    Reindex {
        /// Optional config file path (JSON/TOML). Defaults to `.context/config.toml` when present.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Codebase root (defaults to current directory).
        #[arg(long)]
        codebase_root: Option<PathBuf>,
        /// Run reindexing in the background (returns a job id).
        #[arg(long)]
        background: bool,
        /// Embedding provider (e.g. `onnx`, `openai`, `gemini`).
        #[arg(long)]
        embedding_provider: Option<String>,
        /// Embedding model override.
        #[arg(long)]
        embedding_model: Option<String>,
        /// Embedding base URL override.
        #[arg(long)]
        embedding_base_url: Option<String>,
        /// Embedding dimension override.
        #[arg(long)]
        embedding_dimension: Option<u32>,
        /// Prefer local embeddings over remote providers.
        #[arg(long)]
        embedding_local_first: Option<bool>,
        /// Force local embeddings only.
        #[arg(long)]
        embedding_local_only: Option<bool>,
        /// Embedding routing mode (`localFirst`, `remoteFirst`, `split`).
        #[arg(long)]
        embedding_routing_mode: Option<String>,
        /// Max remote batches for split routing.
        #[arg(long)]
        embedding_split_remote_batches: Option<u32>,
        /// Vector DB provider (e.g. `local`, `milvus_grpc`, `milvus_rest`).
        #[arg(long)]
        vector_db_provider: Option<String>,
        /// Local vector kernel (`hnsw-rs`, `dfrr`).
        #[arg(long)]
        vector_kernel: Option<String>,
        /// Vector DB address/host.
        #[arg(long)]
        vector_db_address: Option<String>,
        /// Vector DB base URL.
        #[arg(long)]
        vector_db_base_url: Option<String>,
        /// Vector DB database name.
        #[arg(long)]
        vector_db_database: Option<String>,
        /// Vector DB SSL enablement.
        #[arg(long)]
        vector_db_ssl: Option<bool>,
        /// Vector DB auth token.
        #[arg(long)]
        vector_db_token: Option<String>,
        /// Vector DB auth username.
        #[arg(long)]
        vector_db_username: Option<String>,
        /// Vector DB auth password.
        #[arg(long)]
        vector_db_password: Option<String>,
    },
    /// Calibrate BQ1 threshold for the local DFRR kernel.
    Calibrate {
        /// Optional config file path (JSON/TOML). Defaults to `.context/config.toml` when present.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Codebase root (defaults to current directory).
        #[arg(long)]
        codebase_root: Option<PathBuf>,
        /// Target recall for calibration (default: 0.99).
        #[arg(long)]
        target_recall: Option<f32>,
        /// Binary search convergence precision (default: 0.005).
        #[arg(long)]
        precision: Option<f32>,
        /// Number of probe queries for calibration (default: 50).
        #[arg(long)]
        num_queries: Option<u32>,
        /// Recall is measured at this K (default: 10).
        #[arg(long)]
        top_k: Option<u32>,
        /// Vector DB provider (e.g. `local`, `milvus_grpc`, `milvus_rest`).
        #[arg(long)]
        vector_db_provider: Option<String>,
        /// Local vector kernel (`hnsw-rs`, `dfrr`).
        #[arg(long)]
        vector_kernel: Option<String>,
        /// Vector DB address/host.
        #[arg(long)]
        vector_db_address: Option<String>,
        /// Vector DB base URL.
        #[arg(long)]
        vector_db_base_url: Option<String>,
        /// Vector DB database name.
        #[arg(long)]
        vector_db_database: Option<String>,
        /// Vector DB SSL enablement.
        #[arg(long)]
        vector_db_ssl: Option<bool>,
        /// Vector DB auth token.
        #[arg(long)]
        vector_db_token: Option<String>,
        /// Vector DB auth username.
        #[arg(long)]
        vector_db_username: Option<String>,
        /// Vector DB auth password.
        #[arg(long)]
        vector_db_password: Option<String>,
    },
    /// Validate a request payload against the request validators.
    #[command(hide = true)]
    ValidateRequest {
        /// Request kind.
        #[arg(long, value_enum)]
        kind: ValidateRequestKind,
        /// Request payload encoded as JSON.
        #[arg(long)]
        input_json: String,
    },
}

#[derive(Debug, Subcommand)]
pub enum ConfigCommands {
    /// Validate config loading, merging, and normalization.
    Check {
        /// Optional config file path (JSON/TOML).
        #[arg(long)]
        path: Option<PathBuf>,
        /// Optional JSON overrides (partial config).
        #[arg(long)]
        overrides_json: Option<String>,
    },
    /// Show the effective config after applying overrides.
    Show {
        /// Optional config file path (JSON/TOML).
        #[arg(long)]
        path: Option<PathBuf>,
        /// Optional JSON overrides (partial config).
        #[arg(long)]
        overrides_json: Option<String>,
    },
    /// Validate config loading and overrides.
    Validate {
        /// Optional config file path (JSON/TOML).
        #[arg(long)]
        path: Option<PathBuf>,
        /// Optional JSON overrides (partial config).
        #[arg(long)]
        overrides_json: Option<String>,
    },
}

#[derive(Debug, Subcommand)]
pub enum JobsCommands {
    /// Show job status.
    Status {
        /// Job identifier.
        #[arg(value_name = "JOB_ID")]
        job_id: String,
        /// Codebase root (defaults to current directory).
        #[arg(long)]
        codebase_root: Option<PathBuf>,
    },
    /// Cancel a running job.
    Cancel {
        /// Job identifier.
        #[arg(value_name = "JOB_ID")]
        job_id: String,
        /// Codebase root (defaults to current directory).
        #[arg(long)]
        codebase_root: Option<PathBuf>,
    },
    /// Internal worker command (runs a job by id).
    #[command(hide = true)]
    Run {
        /// Job identifier.
        #[arg(long)]
        job_id: String,
        /// Codebase root (defaults to current directory).
        #[arg(long)]
        codebase_root: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum ValidateRequestKind {
    Index,
    Search,
    ReindexByChange,
    ClearIndex,
}

impl ValidateRequestKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Index => "index",
            Self::Search => "search",
            Self::ReindexByChange => "reindexByChange",
            Self::ClearIndex => "clearIndex",
        }
    }
}

impl std::fmt::Display for ValidateRequestKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl From<ValidateRequestKind> for RequestKind {
    fn from(value: ValidateRequestKind) -> Self {
        match value {
            ValidateRequestKind::Index => Self::Index,
            ValidateRequestKind::Search => Self::Search,
            ValidateRequestKind::ReindexByChange => Self::ReindexByChange,
            ValidateRequestKind::ClearIndex => Self::ClearIndex,
        }
    }
}

// ---------------------------------------------------------------------------
// Override structs + JSON builders
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct VectorDbCliOverridesArgs<'a> {
    pub provider: Option<&'a str>,
    pub vector_kernel: Option<&'a str>,
    pub address: Option<&'a str>,
    pub base_url: Option<&'a str>,
    pub database: Option<&'a str>,
    pub ssl: Option<bool>,
    pub token: Option<&'a str>,
    pub username: Option<&'a str>,
    pub password: Option<&'a str>,
}

#[derive(Debug, Clone, Copy)]
pub struct EmbeddingCliOverridesArgs<'a> {
    pub provider: Option<&'a str>,
    pub model: Option<&'a str>,
    pub base_url: Option<&'a str>,
    pub dimension: Option<u32>,
    pub local_first: Option<bool>,
    pub local_only: Option<bool>,
    pub routing_mode: Option<&'a str>,
    pub split_max_remote_batches: Option<u32>,
}

impl VectorDbCliOverridesArgs<'_> {
    pub fn to_map(self) -> serde_json::Map<String, serde_json::Value> {
        let mut map = serde_json::Map::new();
        if let Some(provider) = self.provider {
            map.insert(
                "provider".to_owned(),
                serde_json::Value::String(provider.to_owned()),
            );
        }
        if let Some(vector_kernel) = self.vector_kernel {
            map.insert(
                "vectorKernel".to_owned(),
                serde_json::Value::String(vector_kernel.to_owned()),
            );
        }
        if let Some(address) = self.address {
            map.insert(
                "address".to_owned(),
                serde_json::Value::String(address.to_owned()),
            );
        }
        if let Some(base_url) = self.base_url {
            map.insert(
                "baseUrl".to_owned(),
                serde_json::Value::String(base_url.to_owned()),
            );
        }
        if let Some(database) = self.database {
            map.insert(
                "database".to_owned(),
                serde_json::Value::String(database.to_owned()),
            );
        }
        if let Some(ssl) = self.ssl {
            map.insert("ssl".to_owned(), serde_json::Value::Bool(ssl));
        }
        if let Some(token) = self.token {
            map.insert(
                "token".to_owned(),
                serde_json::Value::String(token.to_owned()),
            );
        }
        if let Some(username) = self.username {
            map.insert(
                "username".to_owned(),
                serde_json::Value::String(username.to_owned()),
            );
        }
        if let Some(password) = self.password {
            map.insert(
                "password".to_owned(),
                serde_json::Value::String(password.to_owned()),
            );
        }
        map
    }
}

impl EmbeddingCliOverridesArgs<'_> {
    pub const fn empty() -> Self {
        Self {
            provider: None,
            model: None,
            base_url: None,
            dimension: None,
            local_first: None,
            local_only: None,
            routing_mode: None,
            split_max_remote_batches: None,
        }
    }

    pub fn to_map(self) -> serde_json::Map<String, serde_json::Value> {
        let mut map = serde_json::Map::new();
        if let Some(provider) = self.provider {
            map.insert(
                "provider".to_owned(),
                serde_json::Value::String(provider.to_owned()),
            );
        }
        if let Some(model) = self.model {
            map.insert(
                "model".to_owned(),
                serde_json::Value::String(model.to_owned()),
            );
        }
        if let Some(base_url) = self.base_url {
            map.insert(
                "baseUrl".to_owned(),
                serde_json::Value::String(base_url.to_owned()),
            );
        }
        if let Some(dimension) = self.dimension {
            map.insert(
                "dimension".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(dimension)),
            );
        }
        if let Some(local_first) = self.local_first {
            map.insert(
                "localFirst".to_owned(),
                serde_json::Value::Bool(local_first),
            );
        }
        if let Some(local_only) = self.local_only {
            map.insert("localOnly".to_owned(), serde_json::Value::Bool(local_only));
        }
        let mut routing = serde_json::Map::new();
        if let Some(mode) = self.routing_mode {
            routing.insert(
                "mode".to_owned(),
                serde_json::Value::String(mode.to_owned()),
            );
        }
        let mut split = serde_json::Map::new();
        if let Some(max_remote_batches) = self.split_max_remote_batches {
            split.insert(
                "maxRemoteBatches".to_owned(),
                serde_json::Value::Number(serde_json::Number::from(max_remote_batches)),
            );
            if !routing.contains_key("mode") {
                routing.insert(
                    "mode".to_owned(),
                    serde_json::Value::String("split".to_owned()),
                );
            }
        }
        if !split.is_empty() {
            routing.insert("split".to_owned(), serde_json::Value::Object(split));
        }
        if !routing.is_empty() {
            map.insert("routing".to_owned(), serde_json::Value::Object(routing));
        }
        map
    }
}

pub fn build_overrides_json(
    vector_args: VectorDbCliOverridesArgs<'_>,
    embedding_args: EmbeddingCliOverridesArgs<'_>,
) -> Result<Option<String>, CliError> {
    let vector_map = vector_args.to_map();
    let embedding_map = embedding_args.to_map();
    if vector_map.is_empty() && embedding_map.is_empty() {
        return Ok(None);
    }

    let mut root = serde_json::Map::new();
    if !vector_map.is_empty() {
        root.insert("vectorDb".to_owned(), serde_json::Value::Object(vector_map));
    }
    if !embedding_map.is_empty() {
        root.insert(
            "embedding".to_owned(),
            serde_json::Value::Object(embedding_map),
        );
    }
    let payload = serde_json::Value::Object(root);
    Ok(Some(serde_json::to_string(&payload)?))
}

pub fn build_vector_overrides_json(
    vector_args: VectorDbCliOverridesArgs<'_>,
) -> Result<Option<String>, CliError> {
    build_overrides_json(vector_args, EmbeddingCliOverridesArgs::empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_kernel_override_serializes_into_vector_db_payload()
    -> Result<(), Box<dyn std::error::Error>> {
        let overrides = build_vector_overrides_json(VectorDbCliOverridesArgs {
            provider: None,
            vector_kernel: Some("hnsw-rs"),
            address: None,
            base_url: None,
            database: None,
            ssl: None,
            token: None,
            username: None,
            password: None,
        })?
        .ok_or_else(|| std::io::Error::other("missing overrides payload"))?;

        let payload: serde_json::Value = serde_json::from_str(&overrides)?;
        assert_eq!(
            payload
                .get("vectorDb")
                .and_then(|value| value.get("vectorKernel"))
                .and_then(|value| value.as_str()),
            Some("hnsw-rs")
        );
        Ok(())
    }
}
