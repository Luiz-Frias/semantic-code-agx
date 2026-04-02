//! CLI library entrypoint for binary wrappers.

mod agent_doc;
mod args;
mod commands;
mod config_cmd;
mod dev;
mod error;
mod format;
mod output;
mod redact_layer;
mod resolve;
mod tracing_init;
mod vector_kernel;

use args::{
    Commands, ConfigCommands, EmbeddingCliOverridesArgs, JobsCommands, VectorDbCliOverridesArgs,
    build_overrides_json, build_vector_overrides_json,
};
use clap::Parser;
use commands::{
    CalibrateCommandInput, SearchCommandInput, run_calibrate, run_clear, run_estimate_storage,
    run_index, run_info, run_init, run_jobs_cancel, run_jobs_run, run_jobs_status, run_reindex,
    run_search, run_status,
};
use config_cmd::{config_check, config_show, config_validate};
use dev::validate_request;
use error::CliError;
use format::{OutputArgs, OutputMode};
use output::{CliOutput, write_output};
use resolve::{parse_storage_mode, resolve_codebase_root, resolve_query};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

// Re-export for command handlers that import from `crate::`.
pub(crate) use output::{format_error_output, infra_exit_code};

#[derive(Debug, Parser)]
#[command(
    name = "sca",
    version,
    about = "Semantic code search CLI",
    long_about = None,
    after_help = "Agents: run `sca agent-doc` for the machine-readable protocol spec."
)]
struct Cli {
    #[command(flatten)]
    output: OutputArgs,

    #[command(subcommand)]
    command: Commands,
}

/// Executes the CLI command dispatcher and returns process exit code.
pub fn main_entry() -> std::process::ExitCode {
    let cli = Cli::parse();
    tracing_init::init_tracing(&cli.output);
    let mode = OutputMode::from_args(&cli.output);

    tracing::debug!(
        format = ?mode.format,
        no_progress = mode.no_progress,
        "resolved CLI output mode"
    );

    match run(&cli.command, mode) {
        Ok(output) => {
            tracing::debug!(exit_code = output.exit_code.as_u8(), "command completed");
            match write_output(&output) {
                Ok(()) => std::process::ExitCode::from(output.exit_code.as_u8()),
                Err(error) => exit_with_error(&error),
            }
        },
        Err(error) => {
            tracing::info!(exit_code = error.exit_code().as_u8(), "command failed");
            exit_with_error(&error)
        },
    }
}

fn exit_with_error(error: &CliError) -> std::process::ExitCode {
    let _ = writeln!(io::stderr(), "error: {error}");
    std::process::ExitCode::from(error.exit_code().as_u8())
}

const fn command_name(command: &Commands) -> &'static str {
    match command {
        #[cfg(any(debug_assertions, feature = "dev-tools"))]
        Commands::SelfCheck => "self-check",
        Commands::Info => "info",
        Commands::AgentDoc { .. } => "agent-doc",
        Commands::Config { .. } => "config",
        Commands::Jobs { .. } => "jobs",
        Commands::Init { .. } => "init",
        Commands::EstimateStorage { .. } => "estimate-storage",
        Commands::Index { .. } => "index",
        Commands::Search { .. } => "search",
        Commands::Clear { .. } => "clear",
        Commands::Status { .. } => "status",
        Commands::Reindex { .. } => "reindex",
        Commands::Calibrate { .. } => "calibrate",
        Commands::ValidateRequest { .. } => "validate-request",
    }
}

#[tracing::instrument(
    name = "cli.dispatch",
    skip_all,
    fields(command = %command_name(command))
)]
fn run(command: &Commands, mode: OutputMode) -> Result<CliOutput, CliError> {
    tracing::debug!(command = %command_name(command), "dispatching command");
    tracing::debug!(
        output_format = ?mode.format,
        no_progress = mode.no_progress,
        "dispatch context"
    );

    match command {
        #[cfg(any(debug_assertions, feature = "dev-tools"))]
        Commands::SelfCheck => dev::self_check(mode),
        Commands::Info => run_info(mode),
        Commands::AgentDoc { command } => agent_doc::run_agent_doc(command.as_deref()),
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
        Commands::EstimateStorage { .. } => run_estimate_storage_from_command(command, mode),
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
        Commands::Calibrate { .. } => run_calibrate_from_command(command, mode),
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
    tracing::debug!(command = %command_name(command), "routing vector-db command");
    match command {
        Commands::Index { .. } => run_index_from_command(command, mode),
        Commands::Search { .. } => run_search_from_command(command, mode),
        Commands::Clear { .. } => run_clear_from_command(command, mode),
        Commands::Status { .. } => run_status_from_command(command, mode),
        Commands::Reindex { .. } => run_reindex_from_command(command, mode),
        _ => Err(CliError::InvalidInput("unsupported CLI command".to_owned())),
    }
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
        vector_kernel,
        vector_db_address,
        vector_db_base_url,
        vector_db_database,
        vector_db_ssl,
        vector_db_token,
        vector_db_username,
        vector_db_password,
        danger_close_storage,
        overrides_json,
    } = command
    else {
        return Err(CliError::InvalidInput("unsupported CLI command".to_owned()));
    };

    let root = resolve_codebase_root(codebase_root.as_ref())?;
    // --overrides-json takes precedence over individual CLI flags.
    let overrides = if let Some(raw) = overrides_json {
        Some(raw.clone())
    } else {
        build_overrides_json(
            VectorDbCliOverridesArgs {
                provider: vector_db_provider.as_deref(),
                vector_kernel: vector_kernel.as_deref(),
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
        )?
    };

    run_index(
        mode,
        config.as_deref(),
        overrides.as_deref(),
        &root,
        *init,
        *background,
        *danger_close_storage,
    )
}

fn run_estimate_storage_from_command(
    command: &Commands,
    mode: OutputMode,
) -> Result<CliOutput, CliError> {
    let Commands::EstimateStorage {
        config,
        codebase_root,
        embedding_provider,
        embedding_model,
        embedding_base_url,
        embedding_dimension,
        embedding_local_first,
        embedding_local_only,
        embedding_routing_mode,
        embedding_split_remote_batches,
        vector_db_provider,
        vector_kernel,
        vector_db_address,
        vector_db_base_url,
        vector_db_database,
        vector_db_ssl,
        vector_db_token,
        vector_db_username,
        vector_db_password,
        danger_close_storage,
    } = command
    else {
        return Err(CliError::InvalidInput("unsupported CLI command".to_owned()));
    };

    let root = resolve_codebase_root(codebase_root.as_ref())?;
    let overrides = build_overrides_json(
        VectorDbCliOverridesArgs {
            provider: vector_db_provider.as_deref(),
            vector_kernel: vector_kernel.as_deref(),
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
    run_estimate_storage(
        mode,
        config.as_deref(),
        overrides.as_deref(),
        &root,
        *danger_close_storage,
    )
}

fn run_search_from_command(command: &Commands, mode: OutputMode) -> Result<CliOutput, CliError> {
    let Commands::Search {
        query,
        stdin,
        stdin_batch,
        query_vectors_only,
        top_k,
        threshold,
        filter_expr,
        include_content,
        config,
        codebase_root,
        vector_db_provider,
        vector_kernel,
        vector_db_address,
        vector_db_base_url,
        vector_db_database,
        vector_db_ssl,
        vector_db_token,
        vector_db_username,
        vector_db_password,
        overrides_json,
    } = command
    else {
        return Err(CliError::InvalidInput("unsupported CLI command".to_owned()));
    };

    let root = resolve_codebase_root(codebase_root.as_ref())?;
    // --overrides-json takes precedence over individual CLI flags.
    let overrides = if let Some(raw) = overrides_json {
        Some(raw.clone())
    } else {
        let vector_overrides = VectorDbCliOverridesArgs {
            provider: vector_db_provider.as_deref(),
            vector_kernel: vector_kernel.as_deref(),
            address: vector_db_address.as_deref(),
            base_url: vector_db_base_url.as_deref(),
            database: vector_db_database.as_deref(),
            ssl: *vector_db_ssl,
            token: vector_db_token.as_deref(),
            username: vector_db_username.as_deref(),
            password: vector_db_password.as_deref(),
        };
        build_vector_overrides_json(vector_overrides)?
    };

    if *stdin_batch {
        return commands::run_search_stdin_batch(
            config.as_deref(),
            overrides.as_deref(),
            &root,
            *top_k,
            *threshold,
            *include_content,
            *query_vectors_only,
        );
    }

    let query = resolve_query(*stdin, query.as_deref())?;
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
            vector_kernel: None,
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
            vector_kernel: None,
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
        vector_kernel,
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
            vector_kernel: vector_kernel.as_deref(),
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

fn run_calibrate_from_command(command: &Commands, mode: OutputMode) -> Result<CliOutput, CliError> {
    let Commands::Calibrate {
        config,
        codebase_root,
        target_recall,
        precision,
        num_queries,
        top_k,
        vector_db_provider,
        vector_kernel,
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
    let overrides = build_vector_overrides_json(VectorDbCliOverridesArgs {
        provider: vector_db_provider.as_deref(),
        vector_kernel: vector_kernel.as_deref(),
        address: vector_db_address.as_deref(),
        base_url: vector_db_base_url.as_deref(),
        database: vector_db_database.as_deref(),
        ssl: *vector_db_ssl,
        token: vector_db_token.as_deref(),
        username: vector_db_username.as_deref(),
        password: vector_db_password.as_deref(),
    })?;

    let input = CalibrateCommandInput {
        config_path: config.as_deref(),
        overrides_json: overrides.as_deref(),
        codebase_root: &root,
        target_recall: *target_recall,
        precision: *precision,
        num_queries: *num_queries,
        top_k: *top_k,
    };
    run_calibrate(mode, &input)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ExitCode;
    use clap::CommandFactory;

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
    fn cli_parses_estimate_storage_flags() -> Result<(), Box<dyn std::error::Error>> {
        let cli = Cli::try_parse_from([
            "cli",
            "estimate-storage",
            "--config",
            "/tmp/dspy/dspy/config.json",
            "--codebase-root",
            "/tmp/dspy/dspy",
            "--danger-close-storage",
        ])?;

        match cli.command {
            Commands::EstimateStorage {
                config,
                codebase_root,
                danger_close_storage,
                ..
            } => {
                assert_eq!(config, Some(PathBuf::from("/tmp/dspy/dspy/config.json")));
                assert_eq!(codebase_root, Some(PathBuf::from("/tmp/dspy/dspy")));
                assert!(danger_close_storage);
            },
            _ => return Err("expected estimate-storage command".into()),
        }
        Ok(())
    }

    #[test]
    fn cli_parses_reindex_vector_kernel_flag() -> Result<(), Box<dyn std::error::Error>> {
        let cli = Cli::try_parse_from([
            "cli",
            "reindex",
            "--vector-kernel",
            "hnsw-rs",
            "--codebase-root",
            "/tmp/dspy/dspy",
        ])?;

        match cli.command {
            Commands::Reindex {
                vector_kernel,
                codebase_root,
                ..
            } => {
                assert_eq!(vector_kernel.as_deref(), Some("hnsw-rs"));
                assert_eq!(codebase_root, Some(PathBuf::from("/tmp/dspy/dspy")));
            },
            _ => return Err("expected reindex command".into()),
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
            log_level: format::LogLevel::Info,
        });
        assert!(mode.is_ndjson());
        assert!(mode.no_progress);
    }

    #[test]
    fn cli_parses_agent_doc_bare() -> Result<(), Box<dyn std::error::Error>> {
        let cli = Cli::try_parse_from(["cli", "agent-doc"])?;
        match cli.command {
            Commands::AgentDoc { command } => {
                assert!(command.is_none(), "bare agent-doc should have no scope");
            },
            _ => return Err("expected agent-doc command".into()),
        }
        Ok(())
    }

    #[test]
    fn cli_parses_agent_doc_scoped() -> Result<(), Box<dyn std::error::Error>> {
        let cli = Cli::try_parse_from(["cli", "agent-doc", "search"])?;
        match cli.command {
            Commands::AgentDoc { command } => {
                assert_eq!(command.as_deref(), Some("search"));
            },
            _ => return Err("expected agent-doc command".into()),
        }
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
            log_level: format::LogLevel::Info,
        });
        let invalid_root = Path::new("   ");
        let output = run_reindex(mode, None, None, invalid_root, false)?;
        assert_eq!(output.exit_code, ExitCode::InvalidInput);
        assert!(output.stdout.contains("status: error"));
        Ok(())
    }
}
