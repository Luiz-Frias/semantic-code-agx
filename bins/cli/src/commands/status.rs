//! Status command handler.

use crate::error::{CliError, ExitCode};
use crate::format::OutputMode;
use crate::{CliOutput, format_error_output, infra_exit_code};
use semantic_code_facade::{CliConfigSummary, CliStatus, SnapshotStatus, read_status_local};
use std::path::Path;

/// Run the status command.
pub fn run_status(
    mode: OutputMode,
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    codebase_root: &Path,
) -> Result<CliOutput, CliError> {
    match read_status_local(config_path, overrides_json, codebase_root) {
        Ok(status) => format_status_output(mode, &status),
        Err(error) => Ok(format_error_output(mode, &error, infra_exit_code(&error))),
    }
}

fn format_status_output(mode: OutputMode, status: &CliStatus) -> Result<CliOutput, CliError> {
    let stdout = if mode.is_ndjson() {
        format_status_ndjson(status)?
    } else if mode.is_json() {
        format_status_json(status)?
    } else {
        format_status_text(status)
    };

    Ok(CliOutput {
        stdout,
        stderr: String::new(),
        exit_code: ExitCode::Ok,
    })
}

fn format_status_json(status: &CliStatus) -> Result<String, CliError> {
    let payload = serde_json::json!({
        "status": "ok",
        "manifest": status.manifest,
        "vectorSnapshot": snapshot_json(&status.vector_snapshot),
        "syncSnapshot": snapshot_json(&status.sync_snapshot),
        "config": config_json(&status.config),
    });
    let mut out = serde_json::to_string_pretty(&payload)?;
    out.push('\n');
    Ok(out)
}

fn format_status_ndjson(status: &CliStatus) -> Result<String, CliError> {
    let payload = serde_json::json!({
        "type": "summary",
        "status": "ok",
        "manifest": status.manifest,
        "vectorSnapshot": snapshot_json(&status.vector_snapshot),
        "syncSnapshot": snapshot_json(&status.sync_snapshot),
        "config": config_json(&status.config),
    });
    let mut out = serde_json::to_string(&payload)?;
    out.push('\n');
    Ok(out)
}

fn format_status_text(status: &CliStatus) -> String {
    let mut out = String::new();
    out.push_str("status: ok\n");
    out.push_str("collection: ");
    out.push_str(status.manifest.collection_name.as_str());
    out.push('\n');
    out.push_str("indexMode: ");
    out.push_str(status.manifest.index_mode.as_str());
    out.push('\n');
    out.push_str("manifestUpdatedAtMs: ");
    out.push_str(&status.manifest.updated_at_ms.to_string());
    out.push('\n');
    write_snapshot_text(&mut out, "vector", &status.vector_snapshot);
    write_snapshot_text(&mut out, "sync", &status.sync_snapshot);
    write_config_text(&mut out, &status.config);
    out
}

fn snapshot_json(snapshot: &SnapshotStatus) -> serde_json::Value {
    serde_json::json!({
        "path": snapshot.path.as_ref().map(|path| path.to_string_lossy().to_string()),
        "exists": snapshot.exists,
        "updatedAtMs": snapshot.updated_at_ms,
        "recordCount": snapshot.record_count,
    })
}

fn config_json(config: &CliConfigSummary) -> serde_json::Value {
    serde_json::json!({
        "indexMode": config.index_mode,
        "snapshotStorage": config.snapshot_storage,
        "embeddingDimension": config.embedding_dimension,
        "embeddingCache": {
            "enabled": config.embedding_cache_enabled,
            "diskEnabled": config.embedding_cache_disk_enabled,
            "maxEntries": config.embedding_cache_max_entries,
            "maxBytes": config.embedding_cache_max_bytes,
            "diskPath": config.embedding_cache_disk_path,
            "diskProvider": config.embedding_cache_disk_provider,
            "diskConnection": config.embedding_cache_disk_connection,
            "diskTable": config.embedding_cache_disk_table,
            "diskMaxBytes": config.embedding_cache_disk_max_bytes,
        },
        "retry": {
            "maxAttempts": config.retry_max_attempts,
            "baseDelayMs": config.retry_base_delay_ms,
            "maxDelayMs": config.retry_max_delay_ms,
            "jitterRatioPct": config.retry_jitter_ratio_pct,
        },
        "limits": {
            "maxInFlightFiles": config.max_in_flight_files,
            "maxInFlightEmbeddingBatches": config.max_in_flight_embedding_batches,
            "maxInFlightInserts": config.max_in_flight_inserts,
            "maxBufferedChunks": config.max_buffered_chunks,
            "maxBufferedEmbeddings": config.max_buffered_embeddings,
        }
    })
}

fn write_snapshot_text(out: &mut String, label: &str, snapshot: &SnapshotStatus) {
    let base = [
        ("SnapshotPath", option_path(snapshot.path.as_ref())),
        (
            "SnapshotExists",
            if snapshot.exists { "true" } else { "false" }.to_string(),
        ),
        ("SnapshotUpdatedAtMs", option_u64(snapshot.updated_at_ms)),
    ];
    for (key, value) in base {
        push_snapshot_kv(out, label, key, &value);
    }
    if let Some(count) = snapshot.record_count {
        push_snapshot_kv(out, label, "SnapshotRecords", &count.to_string());
    }
}

fn write_config_text(out: &mut String, config: &CliConfigSummary) {
    let items = [
        ("configIndexMode", config.index_mode.as_str().to_string()),
        ("configSnapshotStorage", format_snapshot_storage(config)),
        (
            "configEmbeddingDimension",
            option_u32(config.embedding_dimension),
        ),
        (
            "configEmbeddingCacheEnabled",
            bool_str(config.embedding_cache_enabled).to_string(),
        ),
        (
            "configEmbeddingCacheDiskEnabled",
            bool_str(config.embedding_cache_disk_enabled).to_string(),
        ),
        (
            "configEmbeddingCacheMaxEntries",
            config.embedding_cache_max_entries.to_string(),
        ),
        (
            "configEmbeddingCacheMaxBytes",
            config.embedding_cache_max_bytes.to_string(),
        ),
        (
            "configEmbeddingCacheDiskPath",
            option_str(config.embedding_cache_disk_path.as_deref()),
        ),
        (
            "configEmbeddingCacheDiskProvider",
            option_str(config.embedding_cache_disk_provider.as_deref()),
        ),
        (
            "configEmbeddingCacheDiskConnection",
            option_str(config.embedding_cache_disk_connection.as_deref()),
        ),
        (
            "configEmbeddingCacheDiskTable",
            option_str(config.embedding_cache_disk_table.as_deref()),
        ),
        (
            "configEmbeddingCacheDiskMaxBytes",
            config.embedding_cache_disk_max_bytes.to_string(),
        ),
        (
            "configRetryMaxAttempts",
            config.retry_max_attempts.to_string(),
        ),
        (
            "configRetryBaseDelayMs",
            config.retry_base_delay_ms.to_string(),
        ),
        (
            "configRetryMaxDelayMs",
            config.retry_max_delay_ms.to_string(),
        ),
        (
            "configRetryJitterRatioPct",
            config.retry_jitter_ratio_pct.to_string(),
        ),
        (
            "configMaxInFlightFiles",
            option_u32(config.max_in_flight_files),
        ),
        (
            "configMaxInFlightEmbeddingBatches",
            option_u32(config.max_in_flight_embedding_batches),
        ),
        (
            "configMaxInFlightInserts",
            option_u32(config.max_in_flight_inserts),
        ),
        (
            "configMaxBufferedChunks",
            option_u32(config.max_buffered_chunks),
        ),
        (
            "configMaxBufferedEmbeddings",
            option_u32(config.max_buffered_embeddings),
        ),
    ];
    for (key, value) in items {
        push_kv(out, key, &value);
    }
}

fn format_snapshot_storage(config: &CliConfigSummary) -> String {
    let rendered = serde_json::to_string(&config.snapshot_storage)
        .unwrap_or_else(|_| "\"unknown\"".to_string());
    let stripped = rendered
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .map(str::to_string);
    stripped.map_or_else(|| rendered, std::convert::identity)
}

fn option_path(path: Option<&std::path::PathBuf>) -> String {
    path.map_or_else(
        || "<none>".to_string(),
        |value| value.to_string_lossy().to_string(),
    )
}

fn option_u32(value: Option<u32>) -> String {
    value.map_or_else(|| "<none>".to_string(), |value| value.to_string())
}

fn option_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "<none>".to_string(), |value| value.to_string())
}

fn option_str(value: Option<&str>) -> String {
    value.map_or_else(|| "<none>".to_string(), ToString::to_string)
}

const fn bool_str(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

#[inline]
fn push_kv(out: &mut String, key: &str, value: &str) {
    out.push_str(key);
    out.push_str(": ");
    out.push_str(value);
    out.push('\n');
}

#[inline]
fn push_snapshot_kv(out: &mut String, label: &str, key: &str, value: &str) {
    out.push_str(label);
    out.push_str(key);
    out.push_str(": ");
    out.push_str(value);
    out.push('\n');
}
