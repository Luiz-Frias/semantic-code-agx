//! CLI input resolution utilities.
//!
//! Pure helpers that resolve codebase roots, query text, storage modes, and
//! environment variables from raw CLI arguments. No side effects beyond
//! reading `stdin` or `env::vars`.

use crate::error::CliError;
use semantic_code_facade::SnapshotStorageMode;
use std::collections::BTreeMap;
use std::io::{self, Read};
use std::path::PathBuf;

pub fn resolve_codebase_root(path: Option<&PathBuf>) -> Result<PathBuf, CliError> {
    match path {
        Some(value) => Ok(value.clone()),
        None => Ok(std::env::current_dir()?),
    }
}

pub fn resolve_query(from_stdin: bool, query: Option<&str>) -> Result<String, CliError> {
    if from_stdin {
        return read_stdin_query();
    }
    query
        .map(str::to_owned)
        .ok_or_else(|| CliError::InvalidInput("missing --query or --stdin".to_string()))
}

fn read_stdin_query() -> Result<String, CliError> {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    let trimmed = buf.trim();
    if trimmed.is_empty() {
        return Err(CliError::InvalidInput("stdin query is empty".to_string()));
    }
    Ok(trimmed.to_string())
}

pub fn parse_storage_mode(value: Option<&str>) -> Result<Option<SnapshotStorageMode>, CliError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let normalized = value.trim();
    if normalized.is_empty() {
        return Err(CliError::InvalidInput(
            "storage mode cannot be empty".to_string(),
        ));
    }
    let lower = normalized.to_ascii_lowercase();
    match lower.as_str() {
        "disabled" => Ok(Some(SnapshotStorageMode::Disabled)),
        "project" => Ok(Some(SnapshotStorageMode::Project)),
        "custom" => Err(CliError::InvalidInput(
            "custom storage mode requires a path (custom:/path)".to_string(),
        )),
        _ => {
            if lower.starts_with("custom:") || lower.starts_with("custom=") {
                let path = normalized[7..].trim();
                if path.is_empty() {
                    return Err(CliError::InvalidInput(
                        "custom storage mode requires a path (custom:/path)".to_string(),
                    ));
                }
                return Ok(Some(SnapshotStorageMode::Custom(PathBuf::from(path))));
            }
            Err(CliError::InvalidInput(format!(
                "unsupported storage mode: {normalized}"
            )))
        },
    }
}

pub fn collect_scoped_env(prefix: &str) -> BTreeMap<String, String> {
    std::env::vars()
        .filter(|(key, _)| key.starts_with(prefix))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_storage_mode_custom_requires_path() {
        let result = parse_storage_mode(Some("custom"));
        assert!(result.is_err());

        let result = parse_storage_mode(Some("custom:/tmp/snapshots"));
        assert!(result.is_ok());
    }

    #[test]
    fn resolve_query_requires_input() {
        let result = resolve_query(false, None);
        assert!(result.is_err());

        let result = resolve_query(false, Some("needle"));
        assert_eq!(result.ok().as_deref(), Some("needle"));
    }
}
