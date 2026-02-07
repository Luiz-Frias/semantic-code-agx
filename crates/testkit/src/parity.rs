//! Parity harness for fixtures derived from the reference snapshot.

use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::{fmt, fs};

/// Errors raised by the parity harness.
#[derive(Debug)]
pub enum ParityError {
    /// Fixture file does not exist.
    MissingFixture {
        /// Path that could not be found.
        path: PathBuf,
    },
    /// Fixture file could not be read.
    FixtureRead {
        /// Path that failed to read.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// Fixture file could not be parsed.
    FixtureParse {
        /// Path that failed to parse.
        path: PathBuf,
        /// Underlying JSON error.
        source: serde_json::Error,
    },
}

impl fmt::Display for ParityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingFixture { path } => {
                write!(formatter, "missing fixture: {}", path.display())
            },
            Self::FixtureRead { path, source } => {
                write!(
                    formatter,
                    "failed to read fixture {}: {}",
                    path.display(),
                    source
                )
            },
            Self::FixtureParse { path, source } => {
                write!(
                    formatter,
                    "failed to parse fixture {}: {}",
                    path.display(),
                    source
                )
            },
        }
    }
}

impl std::error::Error for ParityError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::FixtureRead { source, .. } => Some(source),
            Self::FixtureParse { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// JSON fixtures for API v1 DTOs.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiV1JsonFixtures {
    /// Error DTO example.
    pub error_dto: Value,
    /// Ok result wrapper example.
    pub ok_result: Value,
    /// Error result wrapper example.
    pub error_result: Value,
    /// Index request fixture.
    pub index_request: Value,
    /// Index response fixture.
    pub index_response: Value,
    /// Search request fixture.
    pub search_request: Value,
    /// Search result fixture.
    pub search_result: Value,
    /// Search response fixture.
    pub search_response: Value,
    /// Reindex-by-change request fixture.
    pub reindex_by_change_request: Value,
    /// Reindex-by-change response fixture.
    pub reindex_by_change_response: Value,
    /// Clear-index request fixture.
    pub clear_index_request: Value,
    /// Clear-index response fixture.
    pub clear_index_response: Value,
}

/// ID parity fixtures for domain derivations.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiV1IdParityFixtures {
    /// Collection naming fixture.
    pub collection: ApiV1CollectionParityFixture,
    /// Codebase id fixture.
    pub codebase: ApiV1CodebaseParityFixture,
    /// Chunk id fixture.
    pub chunk: ApiV1ChunkParityFixture,
}

/// Collection naming parity fixture entry.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiV1CollectionParityFixture {
    /// Codebase root path input.
    pub codebase_root: String,
    /// Index mode identifier (dense/hybrid).
    pub index_mode: String,
    /// Expected collection name.
    pub expected: String,
}

/// Codebase id parity fixture entry.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiV1CodebaseParityFixture {
    /// Codebase root path input.
    pub codebase_root: String,
    /// Expected codebase id.
    pub expected: String,
}

/// Chunk id parity fixture entry.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiV1ChunkParityFixture {
    /// Relative path input.
    pub relative_path: String,
    /// Starting line input.
    pub start_line: u32,
    /// Ending line input.
    pub end_line: u32,
    /// Content input.
    pub content: String,
    /// Expected chunk id.
    pub expected: String,
}

/// Load API v1 JSON fixtures from the testkit fixture directory.
pub fn api_v1_json_fixtures() -> Result<ApiV1JsonFixtures, ParityError> {
    load_fixture("api-v1/json-fixtures.json")
}

/// Load API v1 ID parity fixtures from the testkit fixture directory.
pub fn api_v1_id_parity_fixtures() -> Result<ApiV1IdParityFixtures, ParityError> {
    load_fixture("api-v1/id-parity.json")
}

fn load_fixture<T: DeserializeOwned>(relative_path: &str) -> Result<T, ParityError> {
    let root = fixture_root();
    let path =
        if root.ends_with(Path::new("api/v1/fixtures")) && relative_path.starts_with("api-v1/") {
            root.join(&relative_path["api-v1/".len()..])
        } else {
            root.join(relative_path)
        };
    let contents = match fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(ParityError::MissingFixture { path });
        },
        Err(error) => {
            return Err(ParityError::FixtureRead {
                path,
                source: error,
            });
        },
    };

    serde_json::from_str(&contents).map_err(|error| ParityError::FixtureParse {
        path,
        source: error,
    })
}

fn fixture_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_dir.to_path_buf());
    let reference_fixtures = workspace_root
        .join("references")
        .join("semantic-code-for-agents")
        .join("packages")
        .join("backend")
        .join("src")
        .join("api")
        .join("v1")
        .join("fixtures");

    if reference_fixtures.join("json-fixtures.json").exists() {
        reference_fixtures
    } else {
        manifest_dir.join("fixtures")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_domain::{
        ChunkIdInput, CollectionNamingInput, IndexMode, LineSpan, derive_chunk_id,
        derive_codebase_id, derive_collection_name,
    };

    #[test]
    fn missing_fixture_errors_are_reported() {
        let result: Result<ApiV1JsonFixtures, ParityError> = load_fixture("api-v1/missing.json");
        assert!(matches!(result, Err(ParityError::MissingFixture { .. })));
    }

    #[test]
    fn id_parity_fixtures_match_domain_derivations() -> Result<(), Box<dyn std::error::Error>> {
        let fixtures = api_v1_id_parity_fixtures()?;

        let index_mode = match fixtures.collection.index_mode.as_str() {
            "dense" => IndexMode::Dense,
            "hybrid" => IndexMode::Hybrid,
            other => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("unsupported index mode: {other}"),
                )
                .into());
            },
        };
        let input = CollectionNamingInput::new(fixtures.collection.codebase_root, index_mode);
        let collection = derive_collection_name(&input)?;
        assert_eq!(collection.as_str(), fixtures.collection.expected);

        let codebase = derive_codebase_id(fixtures.codebase.codebase_root)?;
        assert_eq!(codebase.as_str(), fixtures.codebase.expected);

        let span = LineSpan::new(fixtures.chunk.start_line, fixtures.chunk.end_line)?;
        let chunk_input =
            ChunkIdInput::new(fixtures.chunk.relative_path, span, fixtures.chunk.content);
        let chunk_id = derive_chunk_id(&chunk_input)?;
        assert_eq!(chunk_id.as_str(), fixtures.chunk.expected);

        Ok(())
    }
}
