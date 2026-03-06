//! Storage estimate command handler.

use crate::error::ExitCode;
use crate::format::OutputMode;
use crate::output::{
    CliOutput, format_error_output, format_ndjson_summary, infra_exit_code, log_info,
};
use semantic_code_facade::{CliStorageEstimate, StorageThresholdStatus, estimate_storage_local};
use std::path::Path;

/// Run the storage estimate command.
pub fn run_estimate_storage(
    mode: OutputMode,
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    codebase_root: &Path,
    danger_close_storage: bool,
) -> Result<CliOutput, crate::error::CliError> {
    let estimate = match estimate_storage_local(
        config_path,
        overrides_json,
        codebase_root,
        danger_close_storage,
    ) {
        Ok(value) => value,
        Err(error) => return Ok(format_error_output(mode, &error, infra_exit_code(&error))),
    };

    let mut stderr = String::new();
    log_info(&mut stderr, "storage estimate completed", mode.no_progress);

    let stdout = if mode.is_ndjson() {
        format_estimate_ndjson(&estimate)?
    } else if mode.is_json() {
        format_estimate_json(&estimate)?
    } else {
        format_estimate_text(&estimate)
    };

    Ok(CliOutput {
        stdout,
        stderr,
        exit_code: ExitCode::Ok,
    })
}

fn format_estimate_json(estimate: &CliStorageEstimate) -> Result<String, crate::error::CliError> {
    let threshold_status = threshold_status_label(estimate.threshold_status);
    let payload = serde_json::json!({
        "status": "ok",
        "kind": "estimateStorage",
        "codebaseRoot": estimate.codebase_root,
        "vectorProvider": estimate.vector_provider,
        "localStorageEnforced": estimate.local_storage_enforced,
        "localStorageRoot": estimate.local_storage_root,
        "indexMode": estimate.index_mode,
        "filesScanned": estimate.files_scanned,
        "filesIndexable": estimate.files_indexable,
        "bytesIndexable": estimate.bytes_indexable,
        "charsIndexable": estimate.chars_indexable,
        "estimatedChunks": estimate.estimated_chunks,
        "dimensionLow": estimate.dimension_low,
        "dimensionHigh": estimate.dimension_high,
        "estimatedBytesLow": estimate.estimated_bytes_low,
        "estimatedBytesHigh": estimate.estimated_bytes_high,
        "requiredFreeBytes": estimate.required_free_bytes,
        "safetyFactorNum": estimate.safety_factor_num,
        "safetyFactorDen": estimate.safety_factor_den,
        "safetyFactor": safety_factor_string(estimate),
        "availableBytes": estimate.available_bytes,
        "thresholdStatus": threshold_status,
    });
    let mut out = serde_json::to_string_pretty(&payload)?;
    out.push('\n');
    Ok(out)
}

fn format_estimate_ndjson(estimate: &CliStorageEstimate) -> Result<String, crate::error::CliError> {
    let extra = serde_json::json!({
        "codebaseRoot": estimate.codebase_root,
        "vectorProvider": estimate.vector_provider,
        "localStorageEnforced": estimate.local_storage_enforced,
        "localStorageRoot": estimate.local_storage_root,
        "indexMode": estimate.index_mode,
        "filesScanned": estimate.files_scanned,
        "filesIndexable": estimate.files_indexable,
        "bytesIndexable": estimate.bytes_indexable,
        "charsIndexable": estimate.chars_indexable,
        "estimatedChunks": estimate.estimated_chunks,
        "dimensionLow": estimate.dimension_low,
        "dimensionHigh": estimate.dimension_high,
        "estimatedBytesLow": estimate.estimated_bytes_low,
        "estimatedBytesHigh": estimate.estimated_bytes_high,
        "requiredFreeBytes": estimate.required_free_bytes,
        "safetyFactorNum": estimate.safety_factor_num,
        "safetyFactorDen": estimate.safety_factor_den,
        "safetyFactor": safety_factor_string(estimate),
        "availableBytes": estimate.available_bytes,
        "thresholdStatus": threshold_status_label(estimate.threshold_status),
    });
    if !extra.is_object() {
        return Err(crate::error::CliError::InvalidInput(
            "internal error: invalid NDJSON payload".to_string(),
        ));
    }
    Ok(format_ndjson_summary("ok", "estimateStorage", Some(extra)))
}

fn format_estimate_text(estimate: &CliStorageEstimate) -> String {
    let mut out = String::new();
    out.push_str("status: ok\n");
    out.push_str("kind: estimateStorage\n");
    out.push_str("codebaseRoot: ");
    out.push_str(&estimate.codebase_root.to_string_lossy());
    out.push('\n');
    out.push_str("vectorProvider: ");
    out.push_str(estimate.vector_provider.as_ref());
    out.push('\n');
    out.push_str("localStorageEnforced: ");
    out.push_str(if estimate.local_storage_enforced {
        "true"
    } else {
        "false"
    });
    out.push('\n');
    out.push_str("localStorageRoot: ");
    out.push_str(&option_path(estimate.local_storage_root.as_deref()));
    out.push('\n');
    out.push_str("indexMode: ");
    out.push_str(estimate.index_mode.as_str());
    out.push('\n');
    out.push_str("filesScanned: ");
    out.push_str(&estimate.files_scanned.to_string());
    out.push('\n');
    out.push_str("filesIndexable: ");
    out.push_str(&estimate.files_indexable.to_string());
    out.push('\n');
    out.push_str("bytesIndexable: ");
    out.push_str(&estimate.bytes_indexable.to_string());
    out.push('\n');
    out.push_str("charsIndexable: ");
    out.push_str(&estimate.chars_indexable.to_string());
    out.push('\n');
    out.push_str("estimatedChunks: ");
    out.push_str(&estimate.estimated_chunks.to_string());
    out.push('\n');
    out.push_str("dimensionLow: ");
    out.push_str(&estimate.dimension_low.to_string());
    out.push('\n');
    out.push_str("dimensionHigh: ");
    out.push_str(&estimate.dimension_high.to_string());
    out.push('\n');
    out.push_str("estimatedBytesLow: ");
    out.push_str(&estimate.estimated_bytes_low.to_string());
    out.push('\n');
    out.push_str("estimatedBytesHigh: ");
    out.push_str(&estimate.estimated_bytes_high.to_string());
    out.push('\n');
    out.push_str("requiredFreeBytes: ");
    out.push_str(&estimate.required_free_bytes.to_string());
    out.push('\n');
    out.push_str("safetyFactor: ");
    out.push_str(&safety_factor_string(estimate));
    out.push('\n');
    out.push_str("availableBytes: ");
    out.push_str(&option_u64(estimate.available_bytes));
    out.push('\n');
    out.push_str("thresholdStatus: ");
    out.push_str(threshold_status_label(estimate.threshold_status));
    out.push('\n');
    out
}

fn safety_factor_string(estimate: &CliStorageEstimate) -> String {
    format!(
        "{}.{:02}",
        estimate.safety_factor_num / estimate.safety_factor_den,
        (estimate.safety_factor_num % estimate.safety_factor_den) * 100
            / estimate.safety_factor_den
    )
}

const fn threshold_status_label(status: StorageThresholdStatus) -> &'static str {
    match status {
        StorageThresholdStatus::Pass => "pass",
        StorageThresholdStatus::Fail => "fail",
        StorageThresholdStatus::Unknown => "unknown",
    }
}

fn option_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "none".to_string(), |inner| inner.to_string())
}

fn option_path(value: Option<&Path>) -> String {
    value.map_or_else(
        || "none".to_string(),
        |inner| inner.to_string_lossy().to_string(),
    )
}
