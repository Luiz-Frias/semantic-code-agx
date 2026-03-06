//! BQ1 calibration persistence helpers for local CLI commands.
//!
//! Follows the same sidecar pattern as `cli_manifest.rs`, persisting
//! calibration state to `.context/calibration.json` alongside the manifest.

use crate::{InfraError, InfraResult};
use semantic_code_domain::CalibrationState;
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope};
use std::fs::OpenOptions;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const CONTEXT_DIR_NAME: &str = ".context";
const CALIBRATION_FILE_NAME: &str = "calibration.json";

/// Read the calibration state from `.context/calibration.json`, if present.
pub fn read_calibration(root: &Path) -> InfraResult<Option<CalibrationState>> {
    let path = calibration_path(root);
    match std::fs::read_to_string(&path) {
        Ok(contents) => {
            let state = serde_json::from_str(&contents).map_err(|error| {
                ErrorEnvelope::unexpected(
                    ErrorCode::invalid_input(),
                    format!("calibration parse failed: {error}"),
                    ErrorClass::NonRetriable,
                )
            })?;
            Ok(Some(state))
        },
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(InfraError::from(error)),
    }
}

/// Write the calibration state to `.context/calibration.json`.
pub fn write_calibration(root: &Path, state: &CalibrationState) -> InfraResult<()> {
    let context_dir = context_dir(root);
    std::fs::create_dir_all(&context_dir)?;
    let mut payload = serde_json::to_string_pretty(state).map_err(|error| {
        ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            format!("calibration serialize failed: {error}"),
            ErrorClass::NonRetriable,
        )
    })?;
    payload.push('\n');
    write_atomic(calibration_path(root), payload.as_bytes())?;
    Ok(())
}

/// Delete the calibration file, if it exists.
pub fn delete_calibration(root: &Path) -> InfraResult<()> {
    let path = calibration_path(root);
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(InfraError::from(error)),
    }
}

/// Calibration file path under `.context/`.
pub fn calibration_path(root: &Path) -> PathBuf {
    context_dir(root).join(CALIBRATION_FILE_NAME)
}

fn context_dir(root: &Path) -> PathBuf {
    root.join(CONTEXT_DIR_NAME)
}

fn write_atomic(path: PathBuf, payload: &[u8]) -> InfraResult<()> {
    let parent = path.parent().ok_or_else(|| {
        ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            "calibration path has no parent directory",
            ErrorClass::NonRetriable,
        )
    })?;
    let file_name = path
        .file_name()
        .ok_or_else(|| {
            ErrorEnvelope::unexpected(
                ErrorCode::internal(),
                "calibration path has no file name",
                ErrorClass::NonRetriable,
            )
        })?
        .to_string_lossy()
        .to_string();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    let mut temp_path = None;
    for attempt in 0..32u32 {
        let candidate = parent.join(format!(
            ".{file_name}.tmp-{}-{nonce}-{attempt}",
            std::process::id()
        ));
        match OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&candidate)
        {
            Ok(mut file) => {
                file.write_all(payload)?;
                file.sync_all()?;
                temp_path = Some(candidate);
                break;
            },
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {},
            Err(error) => return Err(InfraError::from(error)),
        }
    }

    let temp_path = temp_path.ok_or_else(|| {
        ErrorEnvelope::unexpected(
            ErrorCode::internal(),
            "failed to create temporary calibration file",
            ErrorClass::NonRetriable,
        )
    })?;
    std::fs::rename(temp_path, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_domain::EmaState;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nanos}"))
    }

    fn sample_state() -> CalibrationState {
        CalibrationState {
            threshold: 0.55,
            recall_at_threshold: 0.99,
            skip_rate: 0.42,
            calibrated_at_ms: 1_700_000_000_000,
            dimension: 1024,
            corpus_size: 50_000,
            num_queries: 50,
            binary_search_steps: 8,
            ema: None,
        }
    }

    #[test]
    fn calibration_roundtrip() -> InfraResult<()> {
        let temp = temp_dir("cli-calibration");
        std::fs::create_dir_all(&temp)?;

        let state = sample_state();
        write_calibration(&temp, &state)?;

        let loaded = read_calibration(&temp)?.expect("calibration should exist");
        assert_eq!(loaded, state);
        assert!(calibration_path(&temp).is_file());
        Ok(())
    }

    #[test]
    fn read_returns_none_when_missing() -> InfraResult<()> {
        let temp = temp_dir("cli-calibration-missing");
        let result = read_calibration(&temp)?;
        assert!(result.is_none());
        Ok(())
    }

    #[test]
    fn delete_removes_file() -> InfraResult<()> {
        let temp = temp_dir("cli-calibration-delete");
        std::fs::create_dir_all(&temp)?;

        let state = sample_state();
        write_calibration(&temp, &state)?;
        assert!(calibration_path(&temp).is_file());

        delete_calibration(&temp)?;
        assert!(!calibration_path(&temp).exists());
        Ok(())
    }

    #[test]
    fn delete_is_idempotent() -> InfraResult<()> {
        let temp = temp_dir("cli-calibration-delete-noop");
        // Deleting nonexistent file should succeed
        delete_calibration(&temp)?;
        Ok(())
    }

    #[test]
    fn calibration_with_ema_roundtrip() -> InfraResult<()> {
        let temp = temp_dir("cli-calibration-ema");
        std::fs::create_dir_all(&temp)?;

        let state = CalibrationState {
            ema: Some(EmaState {
                alpha: 0.1,
                current_skip_rate: 0.40,
                samples_seen: 100,
            }),
            ..sample_state()
        };
        write_calibration(&temp, &state)?;

        let loaded = read_calibration(&temp)?.expect("calibration should exist");
        assert_eq!(loaded, state);
        assert!(loaded.ema.is_some());
        Ok(())
    }

    #[test]
    fn calibration_json_uses_camel_case() -> InfraResult<()> {
        let temp = temp_dir("cli-calibration-camel");
        std::fs::create_dir_all(&temp)?;

        write_calibration(&temp, &sample_state())?;
        let json = std::fs::read_to_string(calibration_path(&temp))?;
        assert!(json.contains("calibratedAtMs"));
        assert!(json.contains("recallAtThreshold"));
        assert!(json.contains("skipRate"));
        assert!(json.contains("binarySearchSteps"));
        assert!(json.contains("corpusSize"));
        // Verify no ema field when None
        assert!(!json.contains("ema"));
        Ok(())
    }
}
