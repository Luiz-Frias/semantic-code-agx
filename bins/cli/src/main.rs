//! CLI binary entrypoint.

mod commands;
mod error;
mod format;

use clap::{Parser, Subcommand};
use commands::{
    SearchCommandInput, run_clear, run_index, run_info, run_init, run_jobs_cancel, run_jobs_run,
    run_jobs_status, run_reindex, run_search, run_status,
};
use error::{CliError, ExitCode};
use format::{OutputArgs, OutputMode};
#[cfg(any(debug_assertions, feature = "dev-tools"))]
use semantic_code_core::build_info;
use semantic_code_facade::{
    ApiV1ErrorDto, ApiV1ErrorKind, InfraError, RequestKind, infra_error_to_api_v1, is_secret_key,
    load_effective_config_json, validate_request_json,
};
#[cfg(any(debug_assertions, feature = "dev-tools"))]
use semantic_code_facade::{
    facade_crate_version, run_clear_smoke, run_index_smoke, run_search_smoke, validate_env_parsing,
};
use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(
    name = "sca",
    version,
    about = "Semantic code search CLI",
    long_about = None
)]
struct Cli {
    #[command(flatten)]
    output: OutputArgs,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Validate the build and environment wiring.
    #[cfg(any(debug_assertions, feature = "dev-tools"))]
    SelfCheck,
    /// Show build and version details.
    Info,
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
    /// Index the local codebase.
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
    /// Search the local codebase.
    Search {
        /// Search query text.
        #[arg(long)]
        query: Option<String>,
        /// Read the query text from stdin.
        #[arg(long, conflicts_with = "query")]
        stdin: bool,
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
    /// Clear the local index and snapshot.
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
enum ConfigCommands {
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
enum JobsCommands {
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
enum ValidateRequestKind {
    Index,
    Search,
    ReindexByChange,
    ClearIndex,
}

impl ValidateRequestKind {
    const fn as_str(self) -> &'static str {
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

pub(crate) struct CliOutput {
    stdout: String,
    stderr: String,
    exit_code: ExitCode,
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    let mode = OutputMode::from_args(&cli.output);

    match run(&cli.command, mode) {
        Ok(output) => match write_output(&output) {
            Ok(()) => std::process::ExitCode::from(output.exit_code.as_u8()),
            Err(error) => exit_with_error(&error),
        },
        Err(error) => exit_with_error(&error),
    }
}

fn exit_with_error(error: &CliError) -> std::process::ExitCode {
    let _ = writeln!(io::stderr(), "error: {error}");
    std::process::ExitCode::from(error.exit_code().as_u8())
}

fn run(command: &Commands, mode: OutputMode) -> Result<CliOutput, CliError> {
    match command {
        #[cfg(any(debug_assertions, feature = "dev-tools"))]
        Commands::SelfCheck => self_check(mode),
        Commands::Info => run_info(mode),
        Commands::Config { command } => match command {
            ConfigCommands::Check {
                path,
                overrides_json,
            } => config_check(mode, path.as_deref(), overrides_json.as_deref()),
            ConfigCommands::Show {
                path,
                overrides_json,
            } => config_show(mode, path.as_deref(), overrides_json.as_deref()),
            ConfigCommands::Validate {
                path,
                overrides_json,
            } => config_validate(mode, path.as_deref(), overrides_json.as_deref()),
        },
        Commands::Init {
            config,
            codebase_root,
            storage_mode,
            force,
        } => run_init_command(
            mode,
            config.as_deref(),
            codebase_root.as_ref(),
            storage_mode.as_deref(),
            *force,
        ),
        Commands::Jobs { command } => match command {
            JobsCommands::Status {
                job_id,
                codebase_root,
            } => run_jobs_status(
                mode,
                &resolve_codebase_root(codebase_root.as_ref())?,
                job_id,
            ),
            JobsCommands::Cancel {
                job_id,
                codebase_root,
            } => run_jobs_cancel(
                mode,
                &resolve_codebase_root(codebase_root.as_ref())?,
                job_id,
            ),
            JobsCommands::Run {
                job_id,
                codebase_root,
            } => run_jobs_run(
                mode,
                &resolve_codebase_root(codebase_root.as_ref())?,
                job_id,
            ),
        },
        Commands::Index { .. }
        | Commands::Search { .. }
        | Commands::Clear { .. }
        | Commands::Status { .. }
        | Commands::Reindex { .. } => run_vector_db_command(command, mode),
        Commands::ValidateRequest { kind, input_json } => {
            validate_request(*kind, input_json.as_str(), mode)
        },
    }
}

fn run_vector_db_command(command: &Commands, mode: OutputMode) -> Result<CliOutput, CliError> {
    match command {
        Commands::Index { .. } => run_index_from_command(command, mode),
        Commands::Search { .. } => run_search_from_command(command, mode),
        Commands::Clear { .. } => run_clear_from_command(command, mode),
        Commands::Status { .. } => run_status_from_command(command, mode),
        Commands::Reindex { .. } => run_reindex_from_command(command, mode),
        _ => Err(CliError::InvalidInput("unsupported CLI command".to_owned())),
    }
}

#[derive(Debug, Clone, Copy)]
struct VectorDbCliOverridesArgs<'a> {
    provider: Option<&'a str>,
    address: Option<&'a str>,
    base_url: Option<&'a str>,
    database: Option<&'a str>,
    ssl: Option<bool>,
    token: Option<&'a str>,
    username: Option<&'a str>,
    password: Option<&'a str>,
}

#[derive(Debug, Clone, Copy)]
struct EmbeddingCliOverridesArgs<'a> {
    provider: Option<&'a str>,
    model: Option<&'a str>,
    base_url: Option<&'a str>,
    dimension: Option<u32>,
    local_first: Option<bool>,
    local_only: Option<bool>,
    routing_mode: Option<&'a str>,
    split_max_remote_batches: Option<u32>,
}

impl VectorDbCliOverridesArgs<'_> {
    fn to_map(self) -> serde_json::Map<String, serde_json::Value> {
        let mut map = serde_json::Map::new();
        if let Some(provider) = self.provider {
            map.insert(
                "provider".to_owned(),
                serde_json::Value::String(provider.to_owned()),
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
    const fn empty() -> Self {
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

    fn to_map(self) -> serde_json::Map<String, serde_json::Value> {
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

fn build_overrides_json(
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

fn build_vector_overrides_json(
    vector_args: VectorDbCliOverridesArgs<'_>,
) -> Result<Option<String>, CliError> {
    build_overrides_json(vector_args, EmbeddingCliOverridesArgs::empty())
}

fn run_index_command(
    mode: OutputMode,
    config: Option<&Path>,
    codebase_root: Option<&PathBuf>,
    init: bool,
    background: bool,
    vector_overrides: VectorDbCliOverridesArgs<'_>,
    embedding_overrides: EmbeddingCliOverridesArgs<'_>,
) -> Result<CliOutput, CliError> {
    let root = resolve_codebase_root(codebase_root)?;
    let overrides = build_overrides_json(vector_overrides, embedding_overrides)?;
    run_index(mode, config, overrides.as_deref(), &root, init, background)
}

fn run_clear_command(
    mode: OutputMode,
    config: Option<&Path>,
    codebase_root: Option<&PathBuf>,
    overrides: VectorDbCliOverridesArgs<'_>,
) -> Result<CliOutput, CliError> {
    let root = resolve_codebase_root(codebase_root)?;
    let overrides = build_vector_overrides_json(overrides)?;
    run_clear(mode, config, overrides.as_deref(), &root)
}

fn run_status_command(
    mode: OutputMode,
    config: Option<&Path>,
    codebase_root: Option<&PathBuf>,
    overrides: VectorDbCliOverridesArgs<'_>,
) -> Result<CliOutput, CliError> {
    let root = resolve_codebase_root(codebase_root)?;
    let overrides = build_vector_overrides_json(overrides)?;
    run_status(mode, config, overrides.as_deref(), &root)
}

fn run_index_from_command(command: &Commands, mode: OutputMode) -> Result<CliOutput, CliError> {
    let Commands::Index {
        config,
        codebase_root,
        init,
        background,
        embedding_provider,
        embedding_model,
        embedding_base_url,
        embedding_dimension,
        embedding_local_first,
        embedding_local_only,
        embedding_routing_mode,
        embedding_split_remote_batches,
        vector_db_provider,
        vector_db_address,
        vector_db_base_url,
        vector_db_database,
        vector_db_ssl,
        vector_db_token,
        vector_db_username,
        vector_db_password,
    } = command
    else {
        return Err(CliError::InvalidInput("unsupported CLI command".to_owned()));
    };

    run_index_command(
        mode,
        config.as_deref(),
        codebase_root.as_ref(),
        *init,
        *background,
        VectorDbCliOverridesArgs {
            provider: vector_db_provider.as_deref(),
            address: vector_db_address.as_deref(),
            base_url: vector_db_base_url.as_deref(),
            database: vector_db_database.as_deref(),
            ssl: *vector_db_ssl,
            token: vector_db_token.as_deref(),
            username: vector_db_username.as_deref(),
            password: vector_db_password.as_deref(),
        },
        EmbeddingCliOverridesArgs {
            provider: embedding_provider.as_deref(),
            model: embedding_model.as_deref(),
            base_url: embedding_base_url.as_deref(),
            dimension: *embedding_dimension,
            local_first: *embedding_local_first,
            local_only: *embedding_local_only,
            routing_mode: embedding_routing_mode.as_deref(),
            split_max_remote_batches: *embedding_split_remote_batches,
        },
    )
}

fn run_search_from_command(command: &Commands, mode: OutputMode) -> Result<CliOutput, CliError> {
    let Commands::Search {
        query,
        stdin,
        top_k,
        threshold,
        filter_expr,
        include_content,
        config,
        codebase_root,
        vector_db_provider,
        vector_db_address,
        vector_db_base_url,
        vector_db_database,
        vector_db_ssl,
        vector_db_token,
        vector_db_username,
        vector_db_password,
    } = command
    else {
        return Err(CliError::InvalidInput("unsupported CLI command".to_owned()));
    };

    let query = resolve_query(*stdin, query.as_deref())?;
    let root = resolve_codebase_root(codebase_root.as_ref())?;
    let vector_overrides = VectorDbCliOverridesArgs {
        provider: vector_db_provider.as_deref(),
        address: vector_db_address.as_deref(),
        base_url: vector_db_base_url.as_deref(),
        database: vector_db_database.as_deref(),
        ssl: *vector_db_ssl,
        token: vector_db_token.as_deref(),
        username: vector_db_username.as_deref(),
        password: vector_db_password.as_deref(),
    };
    let overrides = build_vector_overrides_json(vector_overrides)?;
    let input = SearchCommandInput {
        config_path: config.as_deref(),
        overrides_json: overrides.as_deref(),
        codebase_root: &root,
        query: &query,
        top_k: *top_k,
        threshold: *threshold,
        filter_expr: filter_expr.as_deref(),
        include_content: *include_content,
    };
    run_search(mode, &input)
}

fn run_clear_from_command(command: &Commands, mode: OutputMode) -> Result<CliOutput, CliError> {
    let Commands::Clear {
        config,
        codebase_root,
        vector_db_provider,
        vector_db_address,
        vector_db_base_url,
        vector_db_database,
        vector_db_ssl,
        vector_db_token,
        vector_db_username,
        vector_db_password,
    } = command
    else {
        return Err(CliError::InvalidInput("unsupported CLI command".to_owned()));
    };

    run_clear_command(
        mode,
        config.as_deref(),
        codebase_root.as_ref(),
        VectorDbCliOverridesArgs {
            provider: vector_db_provider.as_deref(),
            address: vector_db_address.as_deref(),
            base_url: vector_db_base_url.as_deref(),
            database: vector_db_database.as_deref(),
            ssl: *vector_db_ssl,
            token: vector_db_token.as_deref(),
            username: vector_db_username.as_deref(),
            password: vector_db_password.as_deref(),
        },
    )
}

fn run_status_from_command(command: &Commands, mode: OutputMode) -> Result<CliOutput, CliError> {
    let Commands::Status {
        config,
        codebase_root,
        vector_db_provider,
        vector_db_address,
        vector_db_base_url,
        vector_db_database,
        vector_db_ssl,
        vector_db_token,
        vector_db_username,
        vector_db_password,
    } = command
    else {
        return Err(CliError::InvalidInput("unsupported CLI command".to_owned()));
    };

    run_status_command(
        mode,
        config.as_deref(),
        codebase_root.as_ref(),
        VectorDbCliOverridesArgs {
            provider: vector_db_provider.as_deref(),
            address: vector_db_address.as_deref(),
            base_url: vector_db_base_url.as_deref(),
            database: vector_db_database.as_deref(),
            ssl: *vector_db_ssl,
            token: vector_db_token.as_deref(),
            username: vector_db_username.as_deref(),
            password: vector_db_password.as_deref(),
        },
    )
}

fn run_reindex_from_command(command: &Commands, mode: OutputMode) -> Result<CliOutput, CliError> {
    let Commands::Reindex {
        config,
        codebase_root,
        background,
        embedding_provider,
        embedding_model,
        embedding_base_url,
        embedding_dimension,
        embedding_local_first,
        embedding_local_only,
        embedding_routing_mode,
        embedding_split_remote_batches,
        vector_db_provider,
        vector_db_address,
        vector_db_base_url,
        vector_db_database,
        vector_db_ssl,
        vector_db_token,
        vector_db_username,
        vector_db_password,
    } = command
    else {
        return Err(CliError::InvalidInput("unsupported CLI command".to_owned()));
    };

    let root = resolve_codebase_root(codebase_root.as_ref())?;
    let overrides = build_overrides_json(
        VectorDbCliOverridesArgs {
            provider: vector_db_provider.as_deref(),
            address: vector_db_address.as_deref(),
            base_url: vector_db_base_url.as_deref(),
            database: vector_db_database.as_deref(),
            ssl: *vector_db_ssl,
            token: vector_db_token.as_deref(),
            username: vector_db_username.as_deref(),
            password: vector_db_password.as_deref(),
        },
        EmbeddingCliOverridesArgs {
            provider: embedding_provider.as_deref(),
            model: embedding_model.as_deref(),
            base_url: embedding_base_url.as_deref(),
            dimension: *embedding_dimension,
            local_first: *embedding_local_first,
            local_only: *embedding_local_only,
            routing_mode: embedding_routing_mode.as_deref(),
            split_max_remote_batches: *embedding_split_remote_batches,
        },
    )?;
    run_reindex(
        mode,
        config.as_deref(),
        overrides.as_deref(),
        &root,
        *background,
    )
}

fn run_init_command(
    mode: OutputMode,
    config: Option<&Path>,
    codebase_root: Option<&PathBuf>,
    storage_mode: Option<&str>,
    force: bool,
) -> Result<CliOutput, CliError> {
    let root = resolve_codebase_root(codebase_root)?;
    let storage_mode = parse_storage_mode(storage_mode)?;
    run_init(mode, config, &root, storage_mode, force)
}

#[cfg(any(debug_assertions, feature = "dev-tools"))]
fn self_check(mode: OutputMode) -> Result<CliOutput, CliError> {
    let env = collect_scoped_env("SCA_");
    self_check_with_env(mode, &env)
}

#[cfg(any(debug_assertions, feature = "dev-tools"))]
fn self_check_with_env(
    mode: OutputMode,
    env: &BTreeMap<String, String>,
) -> Result<CliOutput, CliError> {
    if let Err(error) = validate_env_parsing(env) {
        return Ok(format_error_output(mode, &error, ExitCode::InvalidInput));
    }

    if let Err(error) = run_index_smoke() {
        return Ok(format_error_output(mode, &error, ExitCode::Internal));
    }
    if let Err(error) = run_search_smoke() {
        return Ok(format_error_output(mode, &error, ExitCode::Internal));
    }
    if let Err(error) = run_clear_smoke() {
        return Ok(format_error_output(mode, &error, ExitCode::Internal));
    }

    let build = build_info();
    let facade_version = facade_crate_version();
    let mut stderr = String::new();

    log_info(&mut stderr, "self-check completed", mode.no_progress);

    let stdout = if mode.is_json() {
        format_self_check_json(&build, facade_version, true, true, true)?
    } else {
        format_self_check_text(&build, facade_version, true, true, true)
    };

    Ok(CliOutput {
        stdout,
        stderr,
        exit_code: ExitCode::Ok,
    })
}

fn validate_request(
    kind: ValidateRequestKind,
    input_json: &str,
    mode: OutputMode,
) -> Result<CliOutput, CliError> {
    if let Err(error) = validate_request_json(kind.into(), input_json).map(|_| ()) {
        return Ok(format_error_output(mode, &error, ExitCode::InvalidInput));
    }

    let mut stderr = String::new();
    log_info(
        &mut stderr,
        "request validation completed",
        mode.no_progress,
    );

    let stdout = if mode.is_ndjson() {
        format_ndjson_summary("ok", kind.as_str(), None)
    } else if mode.is_json() {
        let payload = serde_json::json!({
            "status": "ok",
            "kind": kind.as_str(),
        });
        let mut output = serde_json::to_string_pretty(&payload)?;
        output.push('\n');
        output
    } else {
        format!("status: ok\nkind: {}\n", kind.as_str())
    };

    Ok(CliOutput {
        stdout,
        stderr,
        exit_code: ExitCode::Ok,
    })
}

fn config_check(
    mode: OutputMode,
    path: Option<&Path>,
    overrides_json: Option<&str>,
) -> Result<CliOutput, CliError> {
    let env = collect_scoped_env("SCA_");
    config_check_with_env(mode, &env, path, overrides_json)
}

fn config_check_with_env(
    mode: OutputMode,
    env: &BTreeMap<String, String>,
    path: Option<&Path>,
    overrides_json: Option<&str>,
) -> Result<CliOutput, CliError> {
    let config_json = match load_effective_config_json(env, path, overrides_json) {
        Ok(config) => config,
        Err(error) => return Ok(format_error_output(mode, &error, ExitCode::InvalidInput)),
    };

    let mut stderr = String::new();
    log_info(&mut stderr, "config check completed", mode.no_progress);

    let stdout = if mode.is_ndjson() {
        format_ndjson_summary("ok", "config", None)
    } else if mode.is_json() {
        let config_value: serde_json::Value = serde_json::from_str(config_json.trim())?;
        let payload = serde_json::json!({
            "status": "ok",
            "configPath": path.map(|value| value.to_string_lossy().to_string()),
            "effectiveConfig": config_value,
        });
        let mut output = serde_json::to_string_pretty(&payload)?;
        output.push('\n');
        output
    } else {
        path.map_or_else(
            || "status: ok\nconfig: ok\n".to_string(),
            |path| format!("status: ok\nconfig: ok\npath: {}\n", path.to_string_lossy()),
        )
    };

    Ok(CliOutput {
        stdout,
        stderr,
        exit_code: ExitCode::Ok,
    })
}

fn config_show(
    mode: OutputMode,
    path: Option<&Path>,
    overrides_json: Option<&str>,
) -> Result<CliOutput, CliError> {
    let env = collect_scoped_env("SCA_");
    let config_json = match load_effective_config_json(&env, path, overrides_json) {
        Ok(config) => config,
        Err(error) => return Ok(format_error_output(mode, &error, ExitCode::InvalidInput)),
    };

    let mut stderr = String::new();
    log_info(&mut stderr, "config show completed", mode.no_progress);

    let stdout = if mode.is_ndjson() {
        format_ndjson_summary("ok", "config", None)
    } else if mode.is_json() {
        let config_value: serde_json::Value = serde_json::from_str(config_json.trim())?;
        let payload = serde_json::json!({
            "status": "ok",
            "configPath": path.map(|value| value.to_string_lossy().to_string()),
            "effectiveConfig": config_value,
        });
        let mut output = serde_json::to_string_pretty(&payload)?;
        output.push('\n');
        output
    } else {
        let mut out = String::new();
        out.push_str("status: ok\nconfig:\n");
        out.push_str(&config_json);
        out
    };

    Ok(CliOutput {
        stdout,
        stderr,
        exit_code: ExitCode::Ok,
    })
}

fn config_validate(
    mode: OutputMode,
    path: Option<&Path>,
    overrides_json: Option<&str>,
) -> Result<CliOutput, CliError> {
    let env = collect_scoped_env("SCA_");
    if let Err(error) = load_effective_config_json(&env, path, overrides_json) {
        return Ok(format_error_output(mode, &error, ExitCode::InvalidInput));
    }

    let mut stderr = String::new();
    log_info(&mut stderr, "config validate completed", mode.no_progress);

    let stdout = if mode.is_ndjson() {
        format_ndjson_summary("ok", "config", None)
    } else if mode.is_json() {
        let payload = serde_json::json!({
            "status": "ok",
            "configPath": path.map(|value| value.to_string_lossy().to_string()),
        });
        let mut output = serde_json::to_string_pretty(&payload)?;
        output.push('\n');
        output
    } else {
        path.map_or_else(
            || "status: ok\nconfig: ok\n".to_string(),
            |path| format!("status: ok\nconfig: ok\npath: {}\n", path.to_string_lossy()),
        )
    };

    Ok(CliOutput {
        stdout,
        stderr,
        exit_code: ExitCode::Ok,
    })
}

#[cfg(any(debug_assertions, feature = "dev-tools"))]
fn format_self_check_text(
    build: &semantic_code_core::BuildInfo,
    facade_version: &str,
    index_ok: bool,
    search_ok: bool,
    clear_ok: bool,
) -> String {
    format!(
        "status: ok\nenv: ok\nindex: {}\nsearch: {}\nclear: {}\nname: {}\nversion: {}\nfacade: {}\nrustc: {}\ntarget: {}\nprofile: {}\ngit: {}{}\n",
        if index_ok { "ok" } else { "error" },
        if search_ok { "ok" } else { "error" },
        if clear_ok { "ok" } else { "error" },
        build.name,
        build.version,
        facade_version,
        build.rustc_version,
        build.target,
        build.profile,
        build.git_hash.unwrap_or("none"),
        if build.git_dirty { " (dirty)" } else { "" }
    )
}

#[cfg(any(debug_assertions, feature = "dev-tools"))]
fn format_self_check_json(
    build: &semantic_code_core::BuildInfo,
    facade_version: &str,
    index_ok: bool,
    search_ok: bool,
    clear_ok: bool,
) -> Result<String, CliError> {
    let payload = serde_json::json!({
        "status": "ok",
        "env": { "status": "ok" },
        "index": { "status": if index_ok { "ok" } else { "error" } },
        "search": { "status": if search_ok { "ok" } else { "error" } },
        "clear": { "status": if clear_ok { "ok" } else { "error" } },
        "build": {
            "name": build.name,
            "version": build.version,
            "facadeVersion": facade_version,
            "rustcVersion": build.rustc_version,
            "target": build.target,
            "profile": build.profile,
            "gitHash": build.git_hash,
            "gitDirty": build.git_dirty,
        }
    });

    let mut output = serde_json::to_string_pretty(&payload)?;
    output.push('\n');
    Ok(output)
}

pub(crate) fn format_error_output(
    mode: OutputMode,
    error: &InfraError,
    exit_code: ExitCode,
) -> CliOutput {
    let api_error = sanitize_api_error(infra_error_to_api_v1(error));

    let mut stderr = String::new();
    log_info(&mut stderr, "command failed", mode.no_progress);

    let stdout = if mode.is_ndjson() {
        format_ndjson_error(&api_error)
    } else if mode.is_json() {
        let payload = serde_json::json!({
            "status": "error",
            "error": api_error,
        });

        // This is a CLI boundary, so JSON serialization errors are internal.
        let mut output = serde_json::to_string_pretty(&payload).unwrap_or_else(|_| {
            "{\"status\":\"error\",\"error\":{\"code\":\"ERR_CORE_INTERNAL\",\"message\":\"internal error\",\"kind\":\"INVARIANT\"}}".to_string()
        });
        output.push('\n');
        output
    } else {
        format_api_error_text(&api_error)
    };

    CliOutput {
        stdout,
        stderr,
        exit_code,
    }
}

pub(crate) fn infra_exit_code(error: &InfraError) -> ExitCode {
    match infra_error_to_api_v1(error).kind {
        ApiV1ErrorKind::Expected => ExitCode::InvalidInput,
        ApiV1ErrorKind::Invariant => ExitCode::Internal,
    }
}

fn resolve_codebase_root(path: Option<&PathBuf>) -> Result<PathBuf, CliError> {
    match path {
        Some(value) => Ok(value.clone()),
        None => Ok(std::env::current_dir()?),
    }
}

fn resolve_query(from_stdin: bool, query: Option<&str>) -> Result<String, CliError> {
    if from_stdin {
        return read_stdin_query();
    }
    query
        .map(str::to_owned)
        .ok_or_else(|| CliError::InvalidInput("missing --query or --stdin".to_string()))
}

fn read_stdin_query() -> Result<String, CliError> {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    let trimmed = buf.trim();
    if trimmed.is_empty() {
        return Err(CliError::InvalidInput("stdin query is empty".to_string()));
    }
    Ok(trimmed.to_string())
}

fn parse_storage_mode(
    value: Option<&str>,
) -> Result<Option<semantic_code_config::SnapshotStorageMode>, CliError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let normalized = value.trim();
    if normalized.is_empty() {
        return Err(CliError::InvalidInput(
            "storage mode cannot be empty".to_string(),
        ));
    }
    let lower = normalized.to_ascii_lowercase();
    match lower.as_str() {
        "disabled" => Ok(Some(semantic_code_config::SnapshotStorageMode::Disabled)),
        "project" => Ok(Some(semantic_code_config::SnapshotStorageMode::Project)),
        "custom" => Err(CliError::InvalidInput(
            "custom storage mode requires a path (custom:/path)".to_string(),
        )),
        _ => {
            if lower.starts_with("custom:") || lower.starts_with("custom=") {
                let path = normalized[7..].trim();
                if path.is_empty() {
                    return Err(CliError::InvalidInput(
                        "custom storage mode requires a path (custom:/path)".to_string(),
                    ));
                }
                return Ok(Some(semantic_code_config::SnapshotStorageMode::Custom(
                    PathBuf::from(path),
                )));
            }
            Err(CliError::InvalidInput(format!(
                "unsupported storage mode: {normalized}"
            )))
        },
    }
}

fn sanitize_api_error(mut error: ApiV1ErrorDto) -> ApiV1ErrorDto {
    if let Some(meta) = error.meta.as_mut() {
        for (key, value) in meta.iter_mut() {
            if is_secret_key(key) {
                *value = "<redacted>".to_string();
            }
        }
    }
    error
}

fn format_api_error_text(error: &ApiV1ErrorDto) -> String {
    let mut out = String::new();
    out.push_str("status: error\n");
    out.push_str("code: ");
    out.push_str(&error.code);
    out.push('\n');
    out.push_str("message: ");
    out.push_str(&error.message);
    out.push('\n');
    out.push_str("kind: ");
    out.push_str(match error.kind {
        ApiV1ErrorKind::Expected => "EXPECTED",
        ApiV1ErrorKind::Invariant => "INVARIANT",
    });
    out.push('\n');

    if let Some(meta) = error.meta.as_ref()
        && !meta.is_empty()
    {
        out.push_str("meta:\n");
        for (key, value) in meta {
            out.push_str("  ");
            out.push_str(key);
            out.push_str(": ");
            out.push_str(value);
            out.push('\n');
        }
    }

    out
}

fn log_info(stderr: &mut String, message: &str, no_progress: bool) {
    if no_progress {
        return;
    }
    stderr.push_str("info: ");
    stderr.push_str(message);
    stderr.push('\n');
}

fn format_ndjson_summary(status: &str, kind: &str, extra: Option<serde_json::Value>) -> String {
    let mut payload = serde_json::Map::new();
    payload.insert(
        "type".to_string(),
        serde_json::Value::String("summary".to_string()),
    );
    payload.insert(
        "status".to_string(),
        serde_json::Value::String(status.to_string()),
    );
    payload.insert(
        "kind".to_string(),
        serde_json::Value::String(kind.to_string()),
    );
    if let Some(serde_json::Value::Object(map)) = extra {
        for (key, value) in map {
            payload.insert(key, value);
        }
    }
    let mut out = serde_json::to_string(&serde_json::Value::Object(payload)).unwrap_or_else(|_| {
        "{\"type\":\"summary\",\"status\":\"error\",\"kind\":\"internal\"}".to_string()
    });
    out.push('\n');
    out
}

fn format_ndjson_error(error: &ApiV1ErrorDto) -> String {
    let payload = serde_json::json!({
        "type": "error",
        "status": "error",
        "error": error,
    });
    let mut out = serde_json::to_string(&payload).unwrap_or_else(|_| {
        "{\"type\":\"error\",\"status\":\"error\",\"error\":{\"code\":\"ERR_CORE_INTERNAL\",\"message\":\"internal error\",\"kind\":\"INVARIANT\"}}".to_string()
    });
    out.push('\n');
    out
}

fn write_output(output: &CliOutput) -> Result<(), CliError> {
    let mut stdout = io::stdout();
    stdout.write_all(output.stdout.as_bytes())?;

    if !output.stderr.is_empty() {
        let mut stderr = io::stderr();
        stderr.write_all(output.stderr.as_bytes())?;
        stderr.flush()?;
    }

    Ok(())
}

fn collect_scoped_env(prefix: &str) -> BTreeMap<String, String> {
    std::env::vars()
        .filter(|(key, _)| key.starts_with(prefix))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::OutputFormat;
    use clap::CommandFactory;

    fn workspace_root() -> PathBuf {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(Path::to_path_buf)
            .unwrap_or_else(|| manifest_dir.to_path_buf())
    }

    fn fixture_path(relative: &str) -> PathBuf {
        workspace_root()
            .join("crates")
            .join("testkit")
            .join("fixtures")
            .join(relative)
    }

    #[test]
    fn version_flag_is_supported() {
        let result = Cli::command().try_get_matches_from(["cli", "--version"]);
        let is_version = matches!(
            result,
            Err(error) if error.kind() == clap::error::ErrorKind::DisplayVersion
        );

        assert!(is_version, "expected clap to render version");
    }

    #[test]
    #[cfg(any(debug_assertions, feature = "dev-tools"))]
    fn self_check_json_output_shape() -> Result<(), Box<dyn std::error::Error>> {
        let mode = OutputMode::from_args(&OutputArgs {
            output: Some(OutputFormat::Json),
            json: false,
            agent: false,
            no_progress: true,
            interactive: false,
        });
        let output = self_check_with_env(mode, &BTreeMap::new())?;
        let value: serde_json::Value = serde_json::from_str(output.stdout.trim())?;

        let status = value
            .get("status")
            .and_then(|value| value.as_str())
            .ok_or_else(|| io::Error::other("missing status field"))?;
        assert_eq!(status, "ok");

        let env_status = value
            .get("env")
            .and_then(|value| value.get("status"))
            .and_then(|value| value.as_str())
            .ok_or_else(|| io::Error::other("missing env status"))?;
        assert_eq!(env_status, "ok");

        let build = value
            .get("build")
            .ok_or_else(|| io::Error::other("missing build field"))?;
        let build_obj = build
            .as_object()
            .ok_or_else(|| io::Error::other("build object missing"))?;
        let keys = [
            "name",
            "version",
            "facadeVersion",
            "rustcVersion",
            "target",
            "profile",
            "gitHash",
            "gitDirty",
        ];

        for key in keys {
            assert!(build_obj.contains_key(key), "missing {key}");
        }

        Ok(())
    }

    #[test]
    fn exit_codes_for_errors() -> Result<(), Box<dyn std::error::Error>> {
        let io_error = CliError::Io(io::Error::other("io"));
        let serialization_error = match serde_json::from_str::<serde_json::Value>("not-json") {
            Ok(_) => return Err("expected serialization error".into()),
            Err(error) => CliError::Serialization(error),
        };

        assert_eq!(io_error.exit_code(), ExitCode::Io);
        assert_eq!(serialization_error.exit_code(), ExitCode::Internal);
        Ok(())
    }

    #[test]
    fn error_formatting_redacts_sensitive_meta_keys() {
        let error = ApiV1ErrorDto {
            code: "ERR_CONFIG_INVALID_ENV_URL".to_string(),
            message: "bad env".to_string(),
            kind: ApiV1ErrorKind::Expected,
            meta: Some(BTreeMap::from([
                ("apiKey".to_string(), "secret-value".to_string()),
                ("field".to_string(), "timeoutMs".to_string()),
            ])),
        };

        let sanitized = sanitize_api_error(error);
        let meta = sanitized.meta.expect("meta should be present");
        assert_eq!(meta.get("apiKey").map(String::as_str), Some("<redacted>"));
        assert_eq!(meta.get("field").map(String::as_str), Some("timeoutMs"));
    }

    #[test]
    fn config_check_failure_exit_code_is_invalid_input() -> Result<(), Box<dyn std::error::Error>> {
        let env = BTreeMap::new();
        let missing = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("missing-config.json");

        let mode = OutputMode::from_args(&OutputArgs {
            output: None,
            json: false,
            agent: false,
            no_progress: true,
            interactive: false,
        });
        let output = config_check_with_env(mode, &env, Some(missing.as_path()), None)?;
        assert_eq!(output.exit_code, ExitCode::InvalidInput);
        assert!(output.stdout.contains("status: error"));
        Ok(())
    }

    #[test]
    fn validate_request_errors_are_invalid_input() -> Result<(), Box<dyn std::error::Error>> {
        let mode = OutputMode::from_args(&OutputArgs {
            output: None,
            json: false,
            agent: false,
            no_progress: true,
            interactive: false,
        });
        let output = validate_request(ValidateRequestKind::Search, "{bad", mode)?;
        assert_eq!(output.exit_code, ExitCode::InvalidInput);
        assert!(output.stdout.contains("status: error"));
        Ok(())
    }

    #[test]
    fn reindex_rejects_invalid_codebase_root() -> Result<(), Box<dyn std::error::Error>> {
        let mode = OutputMode::from_args(&OutputArgs {
            output: None,
            json: false,
            agent: false,
            no_progress: true,
            interactive: false,
        });
        let invalid_root = Path::new("   ");
        let output = run_reindex(mode, None, None, invalid_root, false)?;
        assert_eq!(output.exit_code, ExitCode::InvalidInput);
        assert!(output.stdout.contains("status: error"));
        Ok(())
    }

    #[test]
    fn config_overrides_are_applied() -> Result<(), Box<dyn std::error::Error>> {
        let env = BTreeMap::new();
        let path = fixture_path("config/backend-config.valid.json");
        let overrides = r#"{"core":{"timeoutMs":12345}}"#;
        let mode = OutputMode::from_args(&OutputArgs {
            output: Some(OutputFormat::Json),
            json: false,
            agent: false,
            no_progress: true,
            interactive: false,
        });
        let output = config_check_with_env(mode, &env, Some(path.as_path()), Some(overrides))?;
        let value: serde_json::Value = serde_json::from_str(output.stdout.trim())?;
        let timeout_ms = value
            .get("effectiveConfig")
            .and_then(|value| value.get("core"))
            .and_then(|value| value.get("timeoutMs"))
            .and_then(|value| value.as_u64())
            .ok_or_else(|| io::Error::other("missing core.timeoutMs"))?;
        assert_eq!(timeout_ms, 12345);
        Ok(())
    }

    #[test]
    fn cli_parses_index_flags() -> Result<(), Box<dyn std::error::Error>> {
        let cli = Cli::try_parse_from([
            "cli",
            "index",
            "--init",
            "--config",
            "/tmp/dspy/dspy/config.json",
            "--codebase-root",
            "/tmp/dspy/dspy",
        ])?;
        assert!(!cli.output.json);
        match cli.command {
            Commands::Index {
                config,
                codebase_root,
                init,
                ..
            } => {
                assert!(init);
                assert_eq!(config, Some(PathBuf::from("/tmp/dspy/dspy/config.json")));
                assert_eq!(codebase_root, Some(PathBuf::from("/tmp/dspy/dspy")));
            },
            _ => return Err("expected index command".into()),
        }
        Ok(())
    }

    #[test]
    fn cli_parses_search_flags() -> Result<(), Box<dyn std::error::Error>> {
        let cli = Cli::try_parse_from([
            "cli",
            "--json",
            "search",
            "--query",
            "needle",
            "--top-k",
            "7",
            "--threshold",
            "0.42",
            "--filter-expr",
            "relativePath == 'README.md'",
            "--include-content",
            "--config",
            "/tmp/dspy/dspy/config.json",
            "--codebase-root",
            "/tmp/dspy/dspy",
        ])?;
        assert!(cli.output.json);
        match cli.command {
            Commands::Search {
                query,
                top_k,
                threshold,
                filter_expr,
                include_content,
                config,
                codebase_root,
                ..
            } => {
                assert_eq!(query.as_deref(), Some("needle"));
                assert_eq!(top_k, Some(7));
                match threshold {
                    Some(value) => {
                        assert!((value - 0.42).abs() < 1e-6);
                    },
                    None => return Err("missing threshold".into()),
                }
                assert_eq!(filter_expr.as_deref(), Some("relativePath == 'README.md'"));
                assert!(include_content);
                assert_eq!(config, Some(PathBuf::from("/tmp/dspy/dspy/config.json")));
                assert_eq!(codebase_root, Some(PathBuf::from("/tmp/dspy/dspy")));
            },
            _ => return Err("expected search command".into()),
        }
        Ok(())
    }

    #[test]
    fn agent_mode_forces_ndjson_and_quiet() {
        let mode = OutputMode::from_args(&OutputArgs {
            output: None,
            json: false,
            agent: true,
            no_progress: false,
            interactive: true,
        });
        assert!(mode.is_ndjson());
        assert!(mode.no_progress);
    }

    #[test]
    fn parse_storage_mode_custom_requires_path() {
        let result = parse_storage_mode(Some("custom"));
        assert!(result.is_err());

        let result = parse_storage_mode(Some("custom:/tmp/snapshots"));
        assert!(result.is_ok());
    }

    #[test]
    fn resolve_query_requires_input() {
        let result = resolve_query(false, None);
        assert!(result.is_err());

        let result = resolve_query(false, Some("needle"));
        assert_eq!(result.ok().as_deref(), Some("needle"));
    }

    #[test]
    fn log_info_respects_no_progress() {
        let mut stderr = String::new();
        log_info(&mut stderr, "message", true);
        assert!(stderr.is_empty());
    }
}
