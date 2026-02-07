//! Change detection pipeline used by `reindex_by_change`.

use crate::index_codebase::IndexProgress;
use crate::reindex_by_change::{ReindexByChangeDeps, ReindexByChangeInput};
use semantic_code_domain::{CollectionName, IndexMode};
use semantic_code_ports::{FileChangeSet, FileSyncInitOptions, FileSyncOptions, VectorDbRow};
use semantic_code_shared::{RequestContext, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

pub async fn detect_changes(
    ctx: &RequestContext,
    deps: &ReindexByChangeDeps,
    input: &ReindexByChangeInput,
) -> Result<FileChangeSet> {
    deps.file_sync
        .initialize(
            ctx,
            FileSyncInitOptions {
                codebase_root: input.codebase_root.clone(),
                ignore_patterns: input.ignore_patterns.clone(),
            },
        )
        .await?;

    let changes = deps
        .file_sync
        .check_for_changes(ctx, FileSyncOptions::default())
        .await?;
    Ok(normalize_change_set(changes))
}

pub const fn total_changes(changes: &FileChangeSet) -> usize {
    changes.added.len() + changes.removed.len() + changes.modified.len()
}

pub async fn delete_removed_files(
    ctx: &RequestContext,
    deps: &ReindexByChangeDeps,
    input: &ReindexByChangeInput,
    changes: &FileChangeSet,
    total: usize,
    processed: &mut usize,
) -> Result<()> {
    for relative_path in &changes.removed {
        ctx.ensure_not_cancelled("reindex_by_change.removed_loop")?;
        let delete_tags = tags_delete_reason(input.index_mode, "removed");
        let delete_timer = deps.telemetry.as_ref().map(|telemetry| {
            telemetry.start_timer("backend.reindex.deleteFileChunks", Some(&delete_tags))
        });
        delete_file_chunks_by_relative_path(
            ctx,
            deps,
            input.collection_name.clone(),
            relative_path.as_ref(),
        )
        .await?;
        if let Some(timer) = delete_timer.as_ref() {
            timer.stop();
        }
        *processed += 1;
        emit_progress(
            input.on_progress.as_ref(),
            &format!("Removed {relative_path}"),
            *processed as u64,
            total as u64,
            None,
        );
    }
    Ok(())
}

pub async fn delete_modified_files(
    ctx: &RequestContext,
    deps: &ReindexByChangeDeps,
    input: &ReindexByChangeInput,
    changes: &FileChangeSet,
    total: usize,
    processed: &mut usize,
) -> Result<()> {
    for relative_path in &changes.modified {
        ctx.ensure_not_cancelled("reindex_by_change.modified_loop")?;
        let delete_tags = tags_delete_reason(input.index_mode, "modified");
        let delete_timer = deps.telemetry.as_ref().map(|telemetry| {
            telemetry.start_timer("backend.reindex.deleteFileChunks", Some(&delete_tags))
        });
        delete_file_chunks_by_relative_path(
            ctx,
            deps,
            input.collection_name.clone(),
            relative_path.as_ref(),
        )
        .await?;
        if let Some(timer) = delete_timer.as_ref() {
            timer.stop();
        }
        *processed += 1;
        emit_progress(
            input.on_progress.as_ref(),
            &format!("Deleted old chunks for {relative_path}"),
            *processed as u64,
            total as u64,
            None,
        );
    }
    Ok(())
}

pub async fn delete_file_chunks_by_relative_path(
    ctx: &RequestContext,
    deps: &ReindexByChangeDeps,
    collection_name: CollectionName,
    relative_path: &str,
) -> Result<()> {
    ctx.ensure_not_cancelled("reindex_by_change.delete_file_chunks")?;

    let filter = milvus_eq_string("relativePath", relative_path);
    let rows = deps
        .vectordb
        .query(
            ctx,
            collection_name.clone(),
            filter,
            vec!["id".into()],
            None,
        )
        .await?;

    let ids = extract_ids_from_rows(rows);
    if ids.is_empty() {
        return Ok(());
    }

    ctx.ensure_not_cancelled("reindex_by_change.delete_file_chunks.delete")?;
    deps.vectordb.delete(ctx, collection_name, ids).await?;

    Ok(())
}

fn extract_ids_from_rows(rows: Vec<VectorDbRow>) -> Vec<Box<str>> {
    rows.into_iter()
        .filter_map(|row| match row.get("id") {
            Some(Value::String(value)) if !value.trim().is_empty() => {
                Some(value.to_owned().into_boxed_str())
            },
            _ => None,
        })
        .collect()
}

fn milvus_eq_string(field: &str, value: &str) -> Box<str> {
    let escaped = escape_milvus_string_literal(value);
    format!("{field} == \"{escaped}\"").into_boxed_str()
}

fn escape_milvus_string_literal(input: &str) -> String {
    input.replace('\\', "\\\\").replace('"', "\\\"")
}

pub fn normalize_change_set(mut changes: FileChangeSet) -> FileChangeSet {
    changes.added = sort_dedup(changes.added);
    changes.removed = sort_dedup(changes.removed);
    changes.modified = sort_dedup(changes.modified);
    changes
}

fn sort_dedup(mut values: Vec<Box<str>>) -> Vec<Box<str>> {
    values.sort();
    values.dedup();
    values
}

pub fn emit_progress(
    callback: Option<&Arc<dyn Fn(IndexProgress) + Send + Sync>>,
    phase: &str,
    current: u64,
    total: u64,
    percentage_override: Option<u8>,
) {
    let Some(callback) = callback else {
        return;
    };
    let percentage = percentage_override.unwrap_or_else(|| progress_percentage(current, total));
    callback(IndexProgress {
        phase: phase.to_owned().into_boxed_str(),
        current,
        total,
        percentage,
    });
}

pub fn progress_percentage(current: u64, total: u64) -> u8 {
    if total == 0 {
        return 100;
    }
    let percentage = (current.saturating_mul(100) / total).min(100);
    u8::try_from(percentage).unwrap_or(100)
}

fn tags_delete_reason(index_mode: IndexMode, reason: &str) -> BTreeMap<Box<str>, Box<str>> {
    let mut tags = BTreeMap::new();
    tags.insert(
        "indexMode".to_owned().into_boxed_str(),
        index_mode.as_str().to_owned().into_boxed_str(),
    );
    tags.insert("reason".to_owned().into_boxed_str(), reason.into());
    tags
}
