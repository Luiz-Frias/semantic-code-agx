//! File splitting and chunk preparation.

use super::scanner::file_extension_of;
use super::types::{FileResult, FileTaskContext, IndexStageStatsCollector};
use semantic_code_domain::Language;
use semantic_code_ports::{CodeChunk, FileSystemEntryKind, SplitOptions};
use semantic_code_shared::{RequestContext, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

pub struct SplitStatsGuard {
    stats: Arc<IndexStageStatsCollector>,
    started: Instant,
    files: u64,
    chunks: u64,
}

impl SplitStatsGuard {
    pub(crate) fn new(stats: Arc<IndexStageStatsCollector>) -> Self {
        Self {
            stats,
            started: Instant::now(),
            files: 1,
            chunks: 0,
        }
    }

    fn set_chunks(&mut self, count: usize) {
        self.chunks = u64::try_from(count).unwrap_or(u64::MAX);
    }
}

impl Drop for SplitStatsGuard {
    fn drop(&mut self) {
        self.stats
            .record_split(self.files, self.chunks, self.started.elapsed());
    }
}

pub struct SpawnFileTaskContext {
    request_ctx: RequestContext,
    deps: super::types::IndexCodebaseDeps,
    codebase_root: std::path::PathBuf,
    safe_file: semantic_code_ports::SafeRelativePath,
    relative_path: Box<str>,
    max_file_size_bytes: Option<u64>,
    stats: Arc<IndexStageStatsCollector>,
}

impl SpawnFileTaskContext {
    pub(crate) fn new(task_ctx: &FileTaskContext<'_>, relative_path: Box<str>) -> Result<Self> {
        let safe_file = task_ctx
            .deps
            .path_policy
            .to_safe_relative_path(relative_path.as_ref())?;
        Ok(Self {
            request_ctx: task_ctx.request_ctx.clone(),
            deps: task_ctx.deps.clone(),
            codebase_root: task_ctx.codebase_root.clone(),
            safe_file,
            relative_path,
            max_file_size_bytes: task_ctx.max_file_size_bytes,
            stats: Arc::clone(&task_ctx.stats),
        })
    }

    pub(crate) async fn run(self) -> Result<FileResult> {
        let Self {
            request_ctx,
            deps,
            codebase_root,
            safe_file,
            relative_path,
            max_file_size_bytes,
            stats,
        } = self;

        let mut split_timer = SplitStatsGuard::new(stats);
        request_ctx.ensure_not_cancelled("index_codebase.file_task")?;

        if !file_passes_size_check(
            &request_ctx,
            &deps,
            &codebase_root,
            &safe_file,
            max_file_size_bytes,
        )
        .await?
        {
            return Ok(FileResult::Skipped);
        }

        let Some(code) =
            read_file_text_or_skip(&request_ctx, &deps, &codebase_root, &safe_file).await?
        else {
            return Ok(FileResult::Skipped);
        };

        let ext = file_extension_of(relative_path.as_ref());
        let language = language_from_extension(ext.as_deref());

        let Some(chunks) =
            split_file_or_skip(&request_ctx, &deps, code, language, relative_path.as_ref()).await?
        else {
            return Ok(FileResult::Skipped);
        };
        split_timer.set_chunks(chunks.len());

        Ok(FileResult::Ok {
            relative_path,
            language,
            chunks,
        })
    }
}

pub fn submit_file_task<'a>(
    task_ctx: &FileTaskContext<'a>,
    inflight: &mut HashMap<usize, super::types::BoxFuture<'a, Result<FileResult>>>,
    file_index: usize,
) -> Result<()> {
    if inflight.contains_key(&file_index) {
        return Ok(());
    }
    let Some(relative_path) = task_ctx.files.get(file_index) else {
        inflight.insert(file_index, Box::pin(async { Ok(FileResult::Skipped) }));
        return Ok(());
    };

    let relative_path = relative_path.clone();
    let spawn_ctx = SpawnFileTaskContext::new(task_ctx, relative_path)?;

    let fut = task_ctx
        .file_pool
        .submit(move || async move { spawn_ctx.run().await });

    inflight.insert(file_index, Box::pin(fut));
    Ok(())
}

async fn file_passes_size_check(
    ctx: &RequestContext,
    deps: &super::types::IndexCodebaseDeps,
    codebase_root: &Path,
    safe_file: &semantic_code_ports::SafeRelativePath,
    max_file_size_bytes: Option<u64>,
) -> Result<bool> {
    let Some(max_file_size_bytes) = max_file_size_bytes else {
        return Ok(true);
    };

    let stat = deps
        .filesystem
        .stat(ctx, codebase_root.to_path_buf(), safe_file.clone())
        .await;
    match stat {
        Ok(stat) => {
            if stat.kind != FileSystemEntryKind::File {
                return Ok(false);
            }
            if stat.size_bytes > max_file_size_bytes {
                if let Some(logger) = deps.logger.as_ref() {
                    logger.warn(
                        "index.file.skipped_max_size",
                        "Skipping file over maxFileSizeBytes",
                        None,
                    );
                }
                return Ok(false);
            }
        },
        Err(error) => {
            if error.is_cancelled() {
                return Err(error);
            }
            if let Some(logger) = deps.logger.as_ref() {
                logger.warn(
                    "index.file.skipped_stat_error",
                    "Skipping file due to stat error",
                    None,
                );
            }
            return Ok(false);
        },
    }

    Ok(true)
}

async fn read_file_text_or_skip(
    ctx: &RequestContext,
    deps: &super::types::IndexCodebaseDeps,
    codebase_root: &Path,
    safe_file: &semantic_code_ports::SafeRelativePath,
) -> Result<Option<Box<str>>> {
    match deps
        .filesystem
        .read_file_text(ctx, codebase_root.to_path_buf(), safe_file.clone())
        .await
    {
        Ok(code) => Ok(Some(code)),
        Err(error) => {
            if error.is_cancelled() {
                return Err(error);
            }
            if let Some(logger) = deps.logger.as_ref() {
                logger.warn(
                    "index.file.skipped_read_error",
                    "Skipping file due to read error",
                    None,
                );
            }
            Ok(None)
        },
    }
}

async fn split_file_or_skip(
    ctx: &RequestContext,
    deps: &super::types::IndexCodebaseDeps,
    code: Box<str>,
    language: Language,
    file_path: &str,
) -> Result<Option<Vec<CodeChunk>>> {
    match deps
        .splitter
        .split(
            ctx,
            code,
            language,
            SplitOptions {
                file_path: Some(file_path.to_owned().into_boxed_str()),
            },
        )
        .await
    {
        Ok(chunks) => Ok(Some(chunks)),
        Err(error) => {
            if error.is_cancelled() {
                return Err(error);
            }
            if let Some(logger) = deps.logger.as_ref() {
                logger.warn(
                    "index.file.skipped_split_error",
                    "Skipping file due to split error",
                    None,
                );
            }
            Ok(None)
        },
    }
}

fn language_from_extension(ext: Option<&str>) -> Language {
    ext.map_or(Language::Text, Language::from_extension)
}
