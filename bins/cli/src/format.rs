//! Output format helpers for CLI commands.

use clap::{Args, ValueEnum};

/// Output format choices for CLI responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    /// Human-friendly text output.
    Text,
    /// Machine-friendly JSON output.
    Json,
    /// Line-delimited JSON (NDJSON) output.
    Ndjson,
}

/// Log level choices for tracing output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum LogLevel {
    /// Error-level events only.
    Error,
    /// Warning and error events.
    Warn,
    /// Info, warning, and error events.
    #[default]
    Info,
    /// Debug and above.
    Debug,
    /// Trace and above.
    Trace,
}

impl LogLevel {
    /// Returns the env-filter directive string for this level.
    #[must_use]
    pub const fn as_filter_directive(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
        }
    }

    /// Returns a scoped env-filter directive that limits tracing to workspace crates.
    ///
    /// Sets a global `warn` baseline so noisy third-party crates (hyper, tonic,
    /// reqwest, h2, tower) stay quiet, then enables all `semantic_code_*` workspace
    /// crates at the requested level. When `RUST_LOG` is set, this is bypassed.
    #[must_use]
    pub fn as_scoped_directive(self) -> String {
        let level = self.as_filter_directive();
        if matches!(self, Self::Warn) {
            return level.to_owned();
        }
        // All workspace crates share the `semantic_code_` prefix.
        // Cover the CLI binary + every library crate that emits tracing.
        let targets = [
            "semantic_code_cli",
            "semantic_code_facade",
            "semantic_code_infra",
            "semantic_code_app",
            "semantic_code_adapters",
            "semantic_code_config",
            "semantic_code_vector",
            "semantic_code_dfrr_hnsw",
        ];
        let mut directive = String::from("warn");
        for target in targets {
            directive.push(',');
            directive.push_str(target);
            directive.push('=');
            directive.push_str(level);
        }
        directive
    }
}

/// Output-related CLI flags.
#[derive(Debug, Args)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "CLI flags are intentionally boolean to keep UX predictable."
)]
pub struct OutputArgs {
    /// Output format for command responses.
    #[arg(long, global = true, value_enum)]
    pub output: Option<OutputFormat>,
    /// Emit machine-friendly defaults (NDJSON output, no prompts, no progress).
    #[arg(long, global = true)]
    pub agent: bool,
    /// Suppress progress/logging output.
    #[arg(long, global = true)]
    pub no_progress: bool,
    /// Enable interactive prompts (human-only).
    #[arg(long, global = true)]
    pub interactive: bool,
    /// Emit machine-readable JSON output (legacy alias).
    #[arg(long, global = true, hide = true)]
    pub json: bool,
    /// Log level for tracing output when `RUST_LOG` is not set.
    #[arg(long, global = true, value_enum, default_value_t = LogLevel::Info)]
    pub log_level: LogLevel,
}

/// Output mode derived from CLI flags.
#[derive(Debug, Clone, Copy)]
pub struct OutputMode {
    pub format: OutputFormat,
    pub no_progress: bool,
}

impl OutputMode {
    /// Build output mode from CLI flags.
    #[must_use]
    pub const fn from_args(args: &OutputArgs) -> Self {
        let format = match (args.output, args.json, args.agent) {
            (Some(value), _, _) => value,
            (None, true, _) => OutputFormat::Json,
            (None, false, true) => OutputFormat::Ndjson,
            (None, false, false) => OutputFormat::Text,
        };

        let no_progress = if args.agent {
            true
        } else if args.interactive {
            false
        } else {
            args.no_progress
        };

        Self {
            format,
            no_progress,
        }
    }

    /// Returns true when JSON output is requested.
    #[must_use]
    pub const fn is_json(self) -> bool {
        matches!(self.format, OutputFormat::Json)
    }

    /// Returns true when NDJSON output is requested.
    #[must_use]
    pub const fn is_ndjson(self) -> bool {
        matches!(self.format, OutputFormat::Ndjson)
    }
}
