//! BQ1 calibration command handler.

use crate::error::{CliError, ExitCode};
use crate::format::OutputMode;
use crate::vector_kernel::{resolve_vector_kernel_metadata_std_env, warn_if_experimental};
use crate::{CliOutput, format_error_output, infra_exit_code};
use semantic_code_facade::{
    CalibrationParams, CalibrationPrecision, CalibrationQueryCount, CalibrationState,
    CalibrationTopK, InfraError, TargetRecall, run_calibrate_local,
    validate_index_request_for_root,
};
use semantic_code_shared::{ErrorCode, ErrorEnvelope};
use std::fmt::Write;
use std::path::Path;

/// Inputs for calibrate command execution.
pub struct CalibrateCommandInput<'a> {
    pub config_path: Option<&'a Path>,
    pub overrides_json: Option<&'a str>,
    pub codebase_root: &'a Path,
    pub target_recall: Option<f32>,
    pub precision: Option<f32>,
    pub num_queries: Option<u32>,
    pub top_k: Option<u32>,
}

/// Run the calibrate command.
pub fn run_calibrate(
    mode: OutputMode,
    input: &CalibrateCommandInput<'_>,
) -> Result<CliOutput, CliError> {
    // Validate the codebase root is usable.
    if let Err(error) = validate_index_request_for_root(input.codebase_root) {
        return Ok(format_error_output(mode, &error, infra_exit_code(&error)));
    }

    if let Err(error) = validate_calibration_overrides(input) {
        let error: InfraError = error.into();
        return Ok(format_error_output(mode, &error, ExitCode::InvalidInput));
    }

    let vector_kernel =
        match resolve_vector_kernel_metadata_std_env(input.config_path, input.overrides_json) {
            Ok(metadata) => metadata,
            Err(error) => return Ok(format_error_output(mode, &error, infra_exit_code(&error))),
        };
    warn_if_experimental(vector_kernel);

    // Build calibration params with optional CLI overrides.
    let mut params = CalibrationParams::default();
    if let Some(target_recall) = input.target_recall {
        match TargetRecall::try_from(target_recall) {
            Ok(value) => params.target_recall = value,
            Err(error) => {
                let infra_error = InfraError::from(calibration_param_error(
                    "invalid_target_recall",
                    error.field(),
                    error.value().to_owned(),
                    error.message(),
                ));
                return Ok(format_error_output(
                    mode,
                    &infra_error,
                    ExitCode::InvalidInput,
                ));
            },
        }
    }
    if let Some(precision) = input.precision {
        match CalibrationPrecision::try_from(precision) {
            Ok(value) => params.precision = value,
            Err(error) => {
                let infra_error = InfraError::from(calibration_param_error(
                    "invalid_precision",
                    error.field(),
                    error.value().to_owned(),
                    error.message(),
                ));
                return Ok(format_error_output(
                    mode,
                    &infra_error,
                    ExitCode::InvalidInput,
                ));
            },
        }
    }
    if let Some(num_queries) = input.num_queries {
        match CalibrationQueryCount::try_from(num_queries) {
            Ok(value) => params.num_queries = value,
            Err(error) => {
                let infra_error = InfraError::from(calibration_param_error(
                    "invalid_num_queries",
                    error.field(),
                    error.value().to_owned(),
                    error.message(),
                ));
                return Ok(format_error_output(
                    mode,
                    &infra_error,
                    ExitCode::InvalidInput,
                ));
            },
        }
    }
    if let Some(top_k) = input.top_k {
        match CalibrationTopK::try_from(top_k) {
            Ok(value) => params.top_k = value,
            Err(error) => {
                let infra_error = InfraError::from(calibration_param_error(
                    "invalid_top_k",
                    error.field(),
                    error.value().to_owned(),
                    error.message(),
                ));
                return Ok(format_error_output(
                    mode,
                    &infra_error,
                    ExitCode::InvalidInput,
                ));
            },
        }
    }

    match run_calibrate_local(
        input.config_path,
        input.overrides_json,
        input.codebase_root,
        &params,
    ) {
        Ok(state) => format_calibrate_output(mode, &state, vector_kernel),
        Err(error) => Ok(format_error_output(mode, &error, infra_exit_code(&error))),
    }
}

fn format_calibrate_output(
    mode: OutputMode,
    state: &CalibrationState,
    vector_kernel: crate::vector_kernel::VectorKernelMetadata,
) -> Result<CliOutput, CliError> {
    let stdout = if mode.is_ndjson() {
        format_calibrate_ndjson(state, vector_kernel)?
    } else if mode.is_json() {
        format_calibrate_json(state, vector_kernel)?
    } else {
        format_calibrate_text(state)?
    };

    Ok(CliOutput {
        stdout,
        stderr: String::new(),
        exit_code: ExitCode::Ok,
    })
}

fn format_calibrate_json(
    state: &CalibrationState,
    vector_kernel: crate::vector_kernel::VectorKernelMetadata,
) -> Result<String, CliError> {
    let mut payload = serde_json::Map::new();
    payload.insert("status".into(), serde_json::Value::String("ok".into()));
    payload.insert("calibration".into(), serde_json::to_value(state)?);
    payload.insert("vectorKernel".into(), vector_kernel.as_json());
    let mut out = serde_json::to_string_pretty(&serde_json::Value::Object(payload))?;
    out.push('\n');
    Ok(out)
}

fn format_calibrate_ndjson(
    state: &CalibrationState,
    vector_kernel: crate::vector_kernel::VectorKernelMetadata,
) -> Result<String, CliError> {
    let mut summary = serde_json::Map::new();
    summary.insert("type".into(), serde_json::Value::String("summary".into()));
    summary.insert("status".into(), serde_json::Value::String("ok".into()));
    summary.insert("calibration".into(), serde_json::to_value(state)?);
    summary.insert("vectorKernel".into(), vector_kernel.as_json());
    let mut out = serde_json::to_string(&serde_json::Value::Object(summary))?;
    out.push('\n');
    Ok(out)
}

fn format_calibrate_text(state: &CalibrationState) -> Result<String, CliError> {
    let mut out = String::new();
    out.push_str("status: ok\n");
    out.push_str("calibration:\n");

    macro_rules! write_field {
        ($out:expr, $label:expr, $fmt:expr, $val:expr) => {
            if let Err(error) = write!(&mut $out, concat!("  ", $label, ": ", $fmt, "\n"), $val) {
                return Err(CliError::Io(std::io::Error::other(error.to_string())));
            }
        };
    }

    write_field!(out, "threshold", "{:.4}", state.threshold);
    write_field!(out, "recall", "{:.4}", state.recall_at_threshold);
    write_field!(out, "skip_rate", "{:.4}", state.skip_rate);
    write_field!(out, "steps", "{}", state.binary_search_steps);
    write_field!(out, "dimension", "{}", state.dimension);
    write_field!(out, "corpus_size", "{}", state.corpus_size);
    write_field!(out, "num_queries", "{}", state.num_queries);

    Ok(out)
}

fn validate_calibration_overrides(input: &CalibrateCommandInput<'_>) -> Result<(), ErrorEnvelope> {
    if let Some(target_recall) = input.target_recall
        && (!target_recall.is_finite() || !(0.0..=1.0).contains(&target_recall))
    {
        return Err(calibration_param_error(
            "invalid_target_recall",
            "target_recall",
            target_recall.to_string(),
            "target_recall must be a finite value in [0.0, 1.0]",
        ));
    }

    if let Some(precision) = input.precision
        && (!precision.is_finite() || precision <= 0.0 || precision > 1.0)
    {
        return Err(calibration_param_error(
            "invalid_precision",
            "precision",
            precision.to_string(),
            "precision must be a finite value in (0.0, 1.0]",
        ));
    }

    if let Some(num_queries) = input.num_queries
        && num_queries == 0
    {
        return Err(calibration_param_error(
            "invalid_num_queries",
            "num_queries",
            num_queries.to_string(),
            "num_queries must be greater than 0",
        ));
    }

    if let Some(top_k) = input.top_k
        && top_k == 0
    {
        return Err(calibration_param_error(
            "invalid_top_k",
            "top_k",
            top_k.to_string(),
            "top_k must be greater than 0",
        ));
    }

    Ok(())
}

fn calibration_param_error(
    code: &'static str,
    field: &'static str,
    value: String,
    message: &'static str,
) -> ErrorEnvelope {
    ErrorEnvelope::expected(ErrorCode::new("calibration", code), message)
        .with_metadata("field", field)
        .with_metadata("value", value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input(
        target_recall: Option<f32>,
        precision: Option<f32>,
        num_queries: Option<u32>,
        top_k: Option<u32>,
    ) -> CalibrateCommandInput<'static> {
        CalibrateCommandInput {
            config_path: None,
            overrides_json: None,
            codebase_root: Path::new("."),
            target_recall,
            precision,
            num_queries,
            top_k,
        }
    }

    #[test]
    fn validate_calibration_overrides_rejects_invalid_target_recall() {
        let input = make_input(Some(1.5), None, None, None);
        let error = validate_calibration_overrides(&input).expect_err("expected invalid recall");

        assert_eq!(
            error.code,
            ErrorCode::new("calibration", "invalid_target_recall")
        );
        assert!(error.message.contains("[0.0, 1.0]"));
    }

    #[test]
    fn validate_calibration_overrides_rejects_invalid_precision() {
        let input = make_input(None, Some(0.0), None, None);
        let error = validate_calibration_overrides(&input).expect_err("expected invalid precision");

        assert_eq!(
            error.code,
            ErrorCode::new("calibration", "invalid_precision")
        );
        assert!(error.message.contains("(0.0, 1.0]"));
    }

    #[test]
    fn validate_calibration_overrides_rejects_zero_num_queries() {
        let input = make_input(None, None, Some(0), None);
        let error =
            validate_calibration_overrides(&input).expect_err("expected invalid num_queries");

        assert_eq!(
            error.code,
            ErrorCode::new("calibration", "invalid_num_queries")
        );
        assert_eq!(
            error.metadata.get("field").map(String::as_str),
            Some("num_queries")
        );
    }

    #[test]
    fn validate_calibration_overrides_rejects_zero_top_k() {
        let input = make_input(None, None, None, Some(0));
        let error = validate_calibration_overrides(&input).expect_err("expected invalid top_k");

        assert_eq!(error.code, ErrorCode::new("calibration", "invalid_top_k"));
        assert_eq!(error.metadata.get("value").map(String::as_str), Some("0"));
    }

    #[test]
    fn validate_calibration_overrides_accepts_valid_values() {
        let input = make_input(Some(0.99), Some(0.005), Some(50), Some(10));
        assert!(validate_calibration_overrides(&input).is_ok());
    }
}
