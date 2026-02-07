//! Job command handlers.

use crate::error::{CliError, ExitCode};
use crate::format::OutputMode;
use crate::{CliOutput, format_error_output, infra_exit_code};
use semantic_code_facade::{JobResult, JobStatus, cancel_job, read_job_status, run_job};
use std::fmt::Write as _;
use std::path::Path;
use std::process::{Command, Stdio};

/// Run the jobs status command.
pub fn run_jobs_status(
    mode: OutputMode,
    codebase_root: &Path,
    job_id: &str,
) -> Result<CliOutput, CliError> {
    match read_job_status(codebase_root, job_id) {
        Ok(status) => format_job_status(mode, &status),
        Err(error) => Ok(format_error_output(mode, &error, infra_exit_code(&error))),
    }
}

/// Run the jobs cancel command.
pub fn run_jobs_cancel(
    mode: OutputMode,
    codebase_root: &Path,
    job_id: &str,
) -> Result<CliOutput, CliError> {
    match cancel_job(codebase_root, job_id) {
        Ok(status) => format_job_status(mode, &status),
        Err(error) => Ok(format_error_output(mode, &error, infra_exit_code(&error))),
    }
}

/// Run the internal jobs worker command.
pub fn run_jobs_run(
    mode: OutputMode,
    codebase_root: &Path,
    job_id: &str,
) -> Result<CliOutput, CliError> {
    match run_job(codebase_root, job_id) {
        Ok(status) => format_job_status(mode, &status),
        Err(error) => Ok(format_error_output(mode, &error, infra_exit_code(&error))),
    }
}

/// Spawn a detached job runner.
pub fn spawn_job_runner(job_id: &str, codebase_root: &Path) -> Result<(), CliError> {
    let exe = std::env::current_exe()?;
    let mut command = Command::new(exe);
    command
        .arg("jobs")
        .arg("run")
        .arg("--job-id")
        .arg(job_id)
        .arg("--codebase-root")
        .arg(codebase_root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let _child = command.spawn()?;
    Ok(())
}

pub fn format_job_status(mode: OutputMode, status: &JobStatus) -> Result<CliOutput, CliError> {
    let stdout = if mode.is_ndjson() {
        let payload = serde_json::json!({
            "type": "job_status",
            "status": "ok",
            "job": status,
        });
        let mut out = serde_json::to_string(&payload)?;
        out.push('\n');
        out
    } else if mode.is_json() {
        let payload = serde_json::json!({
            "status": "ok",
            "job": status,
        });
        let mut out = serde_json::to_string_pretty(&payload)?;
        out.push('\n');
        out
    } else {
        format_job_text(status)
    };

    Ok(CliOutput {
        stdout,
        stderr: String::new(),
        exit_code: ExitCode::Ok,
    })
}

fn format_job_text(status: &JobStatus) -> String {
    let mut out = String::new();
    out.push_str("status: ok\n");
    out.push_str("jobId: ");
    out.push_str(&status.id);
    out.push('\n');
    out.push_str("state: ");
    out.push_str(&format!("{:?}", status.state).to_ascii_lowercase());
    out.push('\n');
    out.push_str("kind: ");
    out.push_str(&format!("{:?}", status.kind).to_ascii_lowercase());
    out.push('\n');
    if let Some(progress) = status.progress.as_ref() {
        out.push_str("phase: ");
        out.push_str(progress.phase.as_ref());
        out.push('\n');
        out.push_str("progress: ");
        let _ = write!(
            out,
            "{}/{} ({}%)",
            progress.current, progress.total, progress.percentage
        );
        out.push('\n');
    }
    if status.cancel_requested {
        out.push_str("cancelRequested: true\n");
    }
    if let Some(result) = status.result.as_ref() {
        out.push_str("result:\n");
        match result {
            JobResult::Index {
                indexed_files,
                total_chunks,
                index_status,
                stage_stats,
            } => {
                out.push_str("  indexedFiles: ");
                out.push_str(&indexed_files.to_string());
                out.push('\n');
                out.push_str("  totalChunks: ");
                out.push_str(&total_chunks.to_string());
                out.push('\n');
                out.push_str("  indexStatus: ");
                out.push_str(index_status.as_ref());
                out.push('\n');
                out.push_str("  stageStats:\n");
                out.push_str("    scan: files=");
                out.push_str(&stage_stats.scan.files.to_string());
                out.push_str(" durationMs=");
                out.push_str(&stage_stats.scan.duration_ms.to_string());
                out.push('\n');
                out.push_str("    split: files=");
                out.push_str(&stage_stats.split.files.to_string());
                out.push_str(" chunks=");
                out.push_str(&stage_stats.split.chunks.to_string());
                out.push_str(" durationMs=");
                out.push_str(&stage_stats.split.duration_ms.to_string());
                out.push('\n');
                out.push_str("    embed: batches=");
                out.push_str(&stage_stats.embed.batches.to_string());
                out.push_str(" chunks=");
                out.push_str(&stage_stats.embed.chunks.to_string());
                out.push_str(" durationMs=");
                out.push_str(&stage_stats.embed.duration_ms.to_string());
                out.push('\n');
                out.push_str("    insert: batches=");
                out.push_str(&stage_stats.insert.batches.to_string());
                out.push_str(" chunks=");
                out.push_str(&stage_stats.insert.chunks.to_string());
                out.push_str(" durationMs=");
                out.push_str(&stage_stats.insert.duration_ms.to_string());
                out.push('\n');
            },
            JobResult::Reindex {
                added,
                removed,
                modified,
            } => {
                out.push_str("  added: ");
                out.push_str(&added.to_string());
                out.push('\n');
                out.push_str("  removed: ");
                out.push_str(&removed.to_string());
                out.push('\n');
                out.push_str("  modified: ");
                out.push_str(&modified.to_string());
                out.push('\n');
            },
        }
    }
    if let Some(error) = status.error.as_ref() {
        out.push_str("error: ");
        out.push_str(error.message.as_ref());
        out.push('\n');
    }
    out
}
