//! Snapshot subset/tile command handler.

use crate::error::{CliError, ExitCode};
use crate::format::OutputMode;
use crate::output::CliOutput;
use semantic_code_vector::{
    SNAPSHOT_V2_META_FILE_NAME, read_metadata, subset_snapshot_v2, tile_snapshot_v2,
};
use std::fmt::Write;
use std::path::Path;

/// Inputs for the snapshot-subset command.
pub struct SnapshotSubsetCommandInput<'a> {
    pub source: &'a Path,
    pub dest: &'a Path,
    pub target_count: u64,
    pub seed: u64,
    pub noise_sigma: f32,
}

/// Run the snapshot-subset command.
///
/// If `target_count <= source_count`, creates a random subset.
/// If `target_count > source_count`, creates a tiled (replicated) snapshot.
pub fn run_snapshot_subset(
    mode: OutputMode,
    input: &SnapshotSubsetCommandInput<'_>,
) -> Result<CliOutput, CliError> {
    // Read source meta to determine subset vs tile.
    let meta_path = input.source.join(SNAPSHOT_V2_META_FILE_NAME);
    let meta = read_metadata(&meta_path).map_err(|error| {
        CliError::InvalidInput(format!("failed to read source snapshot: {error}"))
    })?;

    let source_count = meta.count;
    let result = if input.target_count <= source_count {
        subset_snapshot_v2(input.source, input.dest, input.target_count, input.seed)
    } else {
        tile_snapshot_v2(
            input.source,
            input.dest,
            input.target_count,
            input.seed,
            input.noise_sigma,
        )
    };

    match result {
        Ok(_meta) => format_success_output(
            mode,
            input.source,
            input.dest,
            source_count,
            input.target_count,
            input.seed,
        ),
        Err(error) => Ok(CliOutput {
            stdout: format_error_text(mode, &error)?,
            stderr: String::new(),
            exit_code: ExitCode::Internal,
        }),
    }
}

fn format_success_output(
    mode: OutputMode,
    source: &Path,
    dest: &Path,
    source_count: u64,
    target_count: u64,
    seed: u64,
) -> Result<CliOutput, CliError> {
    let operation = if target_count <= source_count {
        "subset"
    } else {
        "tile"
    };

    let stdout = if mode.is_ndjson() {
        format_ndjson(source, dest, source_count, target_count, seed, operation)?
    } else if mode.is_json() {
        format_json(source, dest, source_count, target_count, seed, operation)?
    } else {
        format_text(source, dest, source_count, target_count, seed, operation)?
    };

    Ok(CliOutput {
        stdout,
        stderr: String::new(),
        exit_code: ExitCode::Ok,
    })
}

fn format_json(
    source: &Path,
    dest: &Path,
    source_count: u64,
    target_count: u64,
    seed: u64,
    operation: &str,
) -> Result<String, CliError> {
    let payload = serde_json::json!({
        "status": "ok",
        "operation": operation,
        "source": source.display().to_string(),
        "dest": dest.display().to_string(),
        "sourceCount": source_count,
        "targetCount": target_count,
        "seed": seed,
    });
    let mut out = serde_json::to_string_pretty(&payload)?;
    out.push('\n');
    Ok(out)
}

fn format_ndjson(
    source: &Path,
    dest: &Path,
    source_count: u64,
    target_count: u64,
    seed: u64,
    operation: &str,
) -> Result<String, CliError> {
    let payload = serde_json::json!({
        "type": "summary",
        "status": "ok",
        "operation": operation,
        "source": source.display().to_string(),
        "dest": dest.display().to_string(),
        "sourceCount": source_count,
        "targetCount": target_count,
        "seed": seed,
    });
    let mut out = serde_json::to_string(&payload)?;
    out.push('\n');
    Ok(out)
}

fn format_text(
    source: &Path,
    dest: &Path,
    source_count: u64,
    target_count: u64,
    seed: u64,
    operation: &str,
) -> Result<String, CliError> {
    let mut out = String::new();
    out.push_str("status: ok\n");

    macro_rules! write_field {
        ($out:expr, $label:expr, $fmt:expr, $val:expr) => {
            if let Err(error) = write!(&mut $out, concat!($label, ": ", $fmt, "\n"), $val) {
                return Err(CliError::Io(std::io::Error::other(error.to_string())));
            }
        };
    }

    write_field!(out, "operation", "{}", operation);
    write_field!(out, "source", "{}", source.display());
    write_field!(out, "dest", "{}", dest.display());
    write_field!(out, "source_count", "{}", source_count);
    write_field!(out, "target_count", "{}", target_count);
    write_field!(out, "seed", "{}", seed);

    Ok(out)
}

fn format_error_text(
    mode: OutputMode,
    error: &semantic_code_vector::SnapshotError,
) -> Result<String, CliError> {
    let message = error.to_string();
    if mode.is_ndjson() {
        let payload = serde_json::json!({
            "type": "error",
            "status": "error",
            "message": message,
        });
        let mut out = serde_json::to_string(&payload)?;
        out.push('\n');
        Ok(out)
    } else if mode.is_json() {
        let payload = serde_json::json!({
            "status": "error",
            "message": message,
        });
        let mut out = serde_json::to_string_pretty(&payload)?;
        out.push('\n');
        Ok(out)
    } else {
        Ok(format!("status: error\nmessage: {message}\n"))
    }
}
