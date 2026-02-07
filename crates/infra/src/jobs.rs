//! Background job helpers for CLI workflows.

use crate::cli_local::{run_index_local_with_progress, run_reindex_local_with_progress};
use crate::{InfraError, InfraResult};
use semantic_code_app::{
    IndexCodebaseOutput, IndexCodebaseStatus, IndexProgress, IndexStageStats, ReindexByChangeOutput,
};
use semantic_code_config::{
    IndexRequestDto, ReindexByChangeRequestDto, validate_index_request,
    validate_reindex_by_change_request,
};
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

const JOBS_DIR_NAME: &str = "jobs";
const JOB_REQUEST_FILE: &str = "request.json";
const JOB_STATUS_FILE: &str = "status.json";
const JOB_CANCEL_FILE: &str = "cancel";

/// Background job kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum JobKind {
    /// Index a codebase.
    Index,
    /// Reindex by change.
    Reindex,
}

/// Background job state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum JobState {
    /// Job is queued for execution.
    Queued,
    /// Job is running.
    Running,
    /// Job completed successfully.
    Completed,
    /// Job failed.
    Failed,
    /// Job was cancelled.
    Cancelled,
}

/// Background job request payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobRequest {
    /// Job identifier.
    pub id: Box<str>,
    /// Job kind.
    pub kind: JobKind,
    /// Codebase root for this job.
    pub codebase_root: PathBuf,
    /// Optional config path override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_path: Option<PathBuf>,
    /// Optional config overrides JSON.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overrides_json: Option<String>,
    /// Whether to auto-init manifests for index jobs.
    #[serde(default)]
    pub init_if_missing: bool,
    /// Creation timestamp (ms since epoch).
    pub created_at_ms: u64,
}

impl JobRequest {
    /// Create a new job request.
    pub fn new(
        kind: JobKind,
        codebase_root: &Path,
        config_path: Option<&Path>,
        overrides_json: Option<String>,
        init_if_missing: bool,
    ) -> InfraResult<Self> {
        let created_at_ms = now_epoch_ms()?;
        let id = Uuid::new_v4().to_string().into_boxed_str();
        let root =
            std::path::absolute(codebase_root).unwrap_or_else(|_| codebase_root.to_path_buf());
        let config_path = config_path
            .map(|path| std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf()));
        Ok(Self {
            id,
            kind,
            codebase_root: root,
            config_path,
            overrides_json,
            init_if_missing,
            created_at_ms,
        })
    }
}

/// Job progress snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobProgress {
    /// Stage label for quick filtering.
    pub stage: Box<str>,
    /// Human-friendly phase label.
    pub phase: Box<str>,
    /// Current count in this phase.
    pub current: u64,
    /// Total count in this phase.
    pub total: u64,
    /// Completion percentage (0-100).
    pub percentage: u8,
}

impl From<IndexProgress> for JobProgress {
    fn from(progress: IndexProgress) -> Self {
        let stage = stage_from_phase(progress.phase.as_ref());
        Self {
            stage,
            phase: progress.phase,
            current: progress.current,
            total: progress.total,
            percentage: progress.percentage,
        }
    }
}

/// Job result payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum JobResult {
    /// Index job result.
    Index {
        /// Indexed files count.
        indexed_files: usize,
        /// Total chunks indexed.
        total_chunks: usize,
        /// Index completion status.
        index_status: Box<str>,
        /// Stage-level ingestion stats.
        stage_stats: JobStageStats,
    },
    /// Reindex job result.
    Reindex {
        /// Added files count.
        added: usize,
        /// Removed files count.
        removed: usize,
        /// Modified files count.
        modified: usize,
    },
}

/// Job-friendly stage stats for indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobStageStats {
    /// Scan stage stats.
    pub scan: JobScanStats,
    /// Split stage stats.
    pub split: JobSplitStats,
    /// Embedding stage stats.
    pub embed: JobEmbedStats,
    /// Insert stage stats.
    pub insert: JobInsertStats,
}

/// Job-friendly scan stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobScanStats {
    /// Files discovered for indexing.
    pub files: u64,
    /// Elapsed time in milliseconds.
    pub duration_ms: u64,
}

/// Job-friendly split stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobSplitStats {
    /// Files processed by the splitter.
    pub files: u64,
    /// Chunks produced by the splitter.
    pub chunks: u64,
    /// Elapsed time in milliseconds.
    pub duration_ms: u64,
}

/// Job-friendly embed stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobEmbedStats {
    /// Embedding batches executed.
    pub batches: u64,
    /// Chunks embedded.
    pub chunks: u64,
    /// Elapsed time in milliseconds.
    pub duration_ms: u64,
}

/// Job-friendly insert stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobInsertStats {
    /// Insert batches executed.
    pub batches: u64,
    /// Chunks inserted.
    pub chunks: u64,
    /// Elapsed time in milliseconds.
    pub duration_ms: u64,
}

impl From<IndexCodebaseOutput> for JobResult {
    fn from(output: IndexCodebaseOutput) -> Self {
        Self::Index {
            indexed_files: output.indexed_files,
            total_chunks: output.total_chunks,
            index_status: index_status_label(output.status),
            stage_stats: JobStageStats::from(&output.stage_stats),
        }
    }
}

impl From<&IndexStageStats> for JobStageStats {
    fn from(stats: &IndexStageStats) -> Self {
        Self {
            scan: JobScanStats {
                files: stats.scan.files,
                duration_ms: stats.scan.duration_ms,
            },
            split: JobSplitStats {
                files: stats.split.files,
                chunks: stats.split.chunks,
                duration_ms: stats.split.duration_ms,
            },
            embed: JobEmbedStats {
                batches: stats.embed.batches,
                chunks: stats.embed.chunks,
                duration_ms: stats.embed.duration_ms,
            },
            insert: JobInsertStats {
                batches: stats.insert.batches,
                chunks: stats.insert.chunks,
                duration_ms: stats.insert.duration_ms,
            },
        }
    }
}

impl From<ReindexByChangeOutput> for JobResult {
    fn from(output: ReindexByChangeOutput) -> Self {
        Self::Reindex {
            added: output.added,
            removed: output.removed,
            modified: output.modified,
        }
    }
}

/// Job error payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobError {
    /// Stable error code.
    pub code: Box<str>,
    /// Human-readable message.
    pub message: Box<str>,
    /// Error class.
    pub class: Box<str>,
}

impl From<&ErrorEnvelope> for JobError {
    fn from(error: &ErrorEnvelope) -> Self {
        Self {
            code: error.code.to_string().into_boxed_str(),
            message: error.message.clone().into_boxed_str(),
            class: format!("{:?}", error.class).to_lowercase().into_boxed_str(),
        }
    }
}

/// Current job status.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobStatus {
    /// Job identifier.
    pub id: Box<str>,
    /// Job kind.
    pub kind: JobKind,
    /// Job state.
    pub state: JobState,
    /// Job creation time (ms since epoch).
    pub created_at_ms: u64,
    /// Job last update time (ms since epoch).
    pub updated_at_ms: u64,
    /// Latest progress snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<JobProgress>,
    /// Job result payload (when completed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<JobResult>,
    /// Error payload (when failed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JobError>,
    /// Whether cancel was requested.
    #[serde(default)]
    pub cancel_requested: bool,
    /// Non-fatal warnings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<JobError>,
}

/// Create a new background job request and persist initial status.
pub fn create_job(request: &JobRequest) -> InfraResult<JobStatus> {
    let job_dir = job_dir(&request.codebase_root, &request.id);
    std::fs::create_dir_all(&job_dir)?;
    write_job_request(&job_dir, request)?;
    let status = JobStatus {
        id: request.id.clone(),
        kind: request.kind,
        state: JobState::Queued,
        created_at_ms: request.created_at_ms,
        updated_at_ms: request.created_at_ms,
        progress: None,
        result: None,
        error: None,
        cancel_requested: false,
        warnings: Vec::new(),
    };
    write_job_status(&job_dir, &status)?;
    Ok(status)
}

/// Read job status from disk.
pub fn read_job_status(root: &Path, job_id: &str) -> InfraResult<JobStatus> {
    let status_path = job_status_path(root, job_id);
    let payload = std::fs::read_to_string(&status_path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            missing_job_error(job_id)
        } else {
            InfraError::from(error)
        }
    })?;
    serde_json::from_str(&payload).map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::invalid_input(),
            format!("job status parse failed: {error}"),
            ErrorClass::NonRetriable,
        )
    })
}

/// Cancel a running job (best-effort).
pub fn cancel_job(root: &Path, job_id: &str) -> InfraResult<JobStatus> {
    let job_dir = job_dir(root, job_id);
    let mut status = read_job_status(root, job_id)?;
    status.cancel_requested = true;
    status.updated_at_ms = now_epoch_ms()?;
    write_job_status(&job_dir, &status)?;
    std::fs::write(job_dir.join(JOB_CANCEL_FILE), b"")?;
    Ok(status)
}

/// Run a job by ID (used by the background worker command).
pub fn run_job(root: &Path, job_id: &str) -> InfraResult<JobStatus> {
    let job_dir = job_dir(root, job_id);
    let request = read_job_request(&job_dir, job_id)?;
    let mut status = read_job_status(root, job_id)?;
    status.state = JobState::Running;
    status.updated_at_ms = now_epoch_ms()?;
    write_job_status(&job_dir, &status)?;

    let progress_writer = JobProgressWriter::new(
        Arc::new(Mutex::new(status)),
        job_dir.clone(),
        Duration::from_millis(job_progress_interval_ms(&request)?),
    );
    let progress_cb = Some(progress_writer.progress_callback());
    let cancel_path = job_dir.join(JOB_CANCEL_FILE);

    let result = match request.kind {
        JobKind::Index => run_index_job(&request, progress_cb, Some(cancel_path)),
        JobKind::Reindex => run_reindex_job(&request, progress_cb, Some(cancel_path)),
    };

    let mut status = progress_writer.status_snapshot();
    if let Some(warning) = progress_writer.take_warning() {
        status.warnings.push(warning);
    }

    let (result, error) = match result {
        Ok(result) => (Some(result), None),
        Err(error) => (None, Some(error)),
    };

    if let Some(result) = result {
        status.state = JobState::Completed;
        status.result = Some(result);
    }

    if let Some(error) = error.as_ref() {
        if error.is_cancelled() {
            status.state = JobState::Cancelled;
        } else {
            status.state = JobState::Failed;
            status.error = Some(JobError::from(error));
        }
    }

    status.updated_at_ms = now_epoch_ms()?;
    write_job_status(&job_dir, &status)?;
    if let Some(error) = error {
        return Err(error);
    }
    Ok(status)
}

fn run_index_job(
    request: &JobRequest,
    on_progress: Option<Arc<dyn Fn(IndexProgress) + Send + Sync>>,
    cancel_path: Option<PathBuf>,
) -> InfraResult<JobResult> {
    let dto = IndexRequestDto {
        codebase_root: request.codebase_root.to_string_lossy().to_string(),
        collection_name: None,
        force_reindex: None,
    };
    let validated = validate_index_request(&dto)?;
    let output = run_index_local_with_progress(
        request.config_path.as_deref(),
        request.overrides_json.as_deref(),
        &validated,
        request.init_if_missing,
        on_progress,
        cancel_path,
    )?;
    Ok(output.into())
}

fn run_reindex_job(
    request: &JobRequest,
    on_progress: Option<Arc<dyn Fn(IndexProgress) + Send + Sync>>,
    cancel_path: Option<PathBuf>,
) -> InfraResult<JobResult> {
    let dto = ReindexByChangeRequestDto {
        codebase_root: request.codebase_root.to_string_lossy().to_string(),
    };
    let validated = validate_reindex_by_change_request(&dto)?;
    let output = run_reindex_local_with_progress(
        request.config_path.as_deref(),
        request.overrides_json.as_deref(),
        &validated,
        on_progress,
        cancel_path,
    )?;
    Ok(output.into())
}

fn read_job_request(job_dir: &Path, job_id: &str) -> InfraResult<JobRequest> {
    let path = job_dir.join(JOB_REQUEST_FILE);
    let payload = std::fs::read_to_string(&path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            missing_job_error(job_id)
        } else {
            InfraError::from(error)
        }
    })?;
    serde_json::from_str(&payload).map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::invalid_input(),
            format!("job request parse failed: {error}"),
            ErrorClass::NonRetriable,
        )
    })
}

fn write_job_request(job_dir: &Path, request: &JobRequest) -> InfraResult<()> {
    let mut payload = serde_json::to_string_pretty(request).map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            format!("job request serialize failed: {error}"),
            ErrorClass::NonRetriable,
        )
    })?;
    payload.push('\n');
    std::fs::write(job_dir.join(JOB_REQUEST_FILE), payload)?;
    Ok(())
}

fn write_job_status(job_dir: &Path, status: &JobStatus) -> InfraResult<()> {
    let mut payload = serde_json::to_string_pretty(status).map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            format!("job status serialize failed: {error}"),
            ErrorClass::NonRetriable,
        )
    })?;
    payload.push('\n');
    std::fs::write(job_dir.join(JOB_STATUS_FILE), payload)?;
    Ok(())
}

fn job_dir(root: &Path, job_id: &str) -> PathBuf {
    root.join(".context").join(JOBS_DIR_NAME).join(job_id)
}

fn job_status_path(root: &Path, job_id: &str) -> PathBuf {
    job_dir(root, job_id).join(JOB_STATUS_FILE)
}

fn missing_job_error(job_id: &str) -> ErrorEnvelope {
    ErrorEnvelope::expected(ErrorCode::not_found(), "job not found")
        .with_metadata("jobId", job_id.to_string())
}

fn now_epoch_ms() -> InfraResult<u64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                format!("clock error: {error}"),
                ErrorClass::NonRetriable,
            )
        })?;
    let millis = u64::try_from(duration.as_millis()).map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            format!("clock error: {error}"),
            ErrorClass::NonRetriable,
        )
    })?;
    Ok(millis)
}

fn index_status_label(status: IndexCodebaseStatus) -> Box<str> {
    match status {
        IndexCodebaseStatus::Completed => "completed".into(),
        IndexCodebaseStatus::LimitReached => "limitReached".into(),
    }
}

fn stage_from_phase(phase: &str) -> Box<str> {
    let phase = phase.to_ascii_lowercase();
    if phase.contains("scan") {
        "scan".into()
    } else if phase.contains("process") || phase.contains("chunk") {
        "chunk".into()
    } else if phase.contains("collection") {
        "prepare".into()
    } else {
        "index".into()
    }
}

fn job_progress_interval_ms(request: &JobRequest) -> InfraResult<u64> {
    let env = semantic_code_config::BackendEnv::from_std_env().map_err(ErrorEnvelope::from)?;
    let config = semantic_code_config::load_backend_config_from_path(
        request.config_path.as_deref(),
        request.overrides_json.as_deref(),
        &env,
    )?;
    Ok(config.embedding.jobs.progress_interval_ms)
}

struct JobProgressWriter {
    status: Arc<Mutex<JobStatus>>,
    job_dir: PathBuf,
    interval: Duration,
    last_write: Arc<Mutex<Instant>>,
    warning: Arc<Mutex<Option<JobError>>>,
}

impl JobProgressWriter {
    fn new(status: Arc<Mutex<JobStatus>>, job_dir: PathBuf, interval: Duration) -> Self {
        Self {
            status,
            job_dir,
            interval,
            last_write: Arc::new(Mutex::new(Instant::now())),
            warning: Arc::new(Mutex::new(None)),
        }
    }

    fn progress_callback(&self) -> Arc<dyn Fn(IndexProgress) + Send + Sync> {
        let status = Arc::clone(&self.status);
        let job_dir = self.job_dir.clone();
        let interval = self.interval;
        let last_write = Arc::clone(&self.last_write);
        let warning = Arc::clone(&self.warning);

        Arc::new(move |progress| {
            if let Ok(mut last) = last_write.lock() {
                let should_write = last.elapsed() >= interval || progress.percentage == 100;
                if !should_write {
                    return;
                }
                *last = Instant::now();
            }

            let mut guard = match status.lock() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    let mut warning_guard = warning
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if warning_guard.is_none() {
                        *warning_guard = Some(JobError {
                            code: ErrorCode::internal().to_string().into_boxed_str(),
                            message: "job status lock poisoned".into(),
                            class: "nonretriable".into(),
                        });
                    }
                    drop(warning_guard);
                    drop(poisoned.into_inner());
                    return;
                },
            };
            guard.progress = Some(JobProgress::from(progress));
            if let Ok(now) = now_epoch_ms() {
                guard.updated_at_ms = now;
            }
            if let Err(error) = write_job_status(&job_dir, &guard) {
                let mut warning_guard = warning
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if warning_guard.is_none() {
                    *warning_guard = Some(JobError::from(&error));
                }
                drop(warning_guard);
            }
            drop(guard);
        })
    }

    fn take_warning(&self) -> Option<JobError> {
        let mut guard = self
            .warning
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.take()
    }

    fn status_snapshot(&self) -> JobStatus {
        match self.status.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_request_serializes_roundtrip() -> InfraResult<()> {
        let dir = std::env::temp_dir().join("sca-job-test");
        let request = JobRequest::new(JobKind::Index, &dir, None, None, false)?;
        let job_dir = job_dir(&request.codebase_root, &request.id);
        std::fs::create_dir_all(&job_dir)?;
        write_job_request(&job_dir, &request)?;
        let loaded = read_job_request(&job_dir, &request.id)?;
        assert_eq!(loaded.kind, JobKind::Index);
        Ok(())
    }
}
