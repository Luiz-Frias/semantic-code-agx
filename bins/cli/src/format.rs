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
