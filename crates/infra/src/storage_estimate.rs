//! Storage estimation and preflight checks for local indexing.

use crate::InfraResult;
use crate::cli_manifest::config_path as context_config_path;
use semantic_code_adapters::IgnoreMatcher;
use semantic_code_config::ValidatedBackendConfig;
use semantic_code_domain::IndexMode;
use semantic_code_ports::{IgnoreMatchInput, IgnorePort};
use semantic_code_shared::{ErrorCode, ErrorEnvelope};
use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};

const CONTEXT_IGNORE_FILE: &str = ".contextignore";
const CHUNK_UTILIZATION_NUM: u64 = 88;
const CHUNK_UTILIZATION_DEN: u64 = 100;
const METADATA_BYTES_PER_CHUNK: u64 = 1_000;
const DENSE_FACTOR_NUM: u64 = 3;
const DENSE_FACTOR_DEN: u64 = 2;
const HYBRID_FACTOR_NUM: u64 = 8;
const HYBRID_FACTOR_DEN: u64 = 5;
const SAFE_FACTOR_NUM: u64 = 2;
const SAFE_FACTOR_DEN: u64 = 1;
const DANGER_FACTOR_NUM: u64 = 5;
const DANGER_FACTOR_DEN: u64 = 4;
const MIN_WORKING_SET_BYTES: u64 = 128 * 1024 * 1024;

/// Storage threshold status when free-space data is available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageThresholdStatus {
    /// Available free space is at or above required bytes.
    Pass,
    /// Available free space is below required bytes.
    Fail,
    /// Free-space information could not be resolved.
    Unknown,
}

/// Storage estimate summary used by CLI output and index preflight.
#[derive(Debug, Clone)]
pub struct CliStorageEstimate {
    /// Codebase root used for scanning.
    pub codebase_root: PathBuf,
    /// Effective vector provider identifier.
    pub vector_provider: Box<str>,
    /// Whether local snapshot storage is the active write target.
    pub local_storage_enforced: bool,
    /// Local storage root path when applicable.
    pub local_storage_root: Option<PathBuf>,
    /// Effective index mode.
    pub index_mode: IndexMode,
    /// Number of regular files scanned after ignore checks.
    pub files_scanned: u64,
    /// Number of files considered for indexing (after extension + size + UTF-8 checks).
    pub files_indexable: u64,
    /// Aggregate UTF-8 byte count across indexable files.
    pub bytes_indexable: u64,
    /// Aggregate character count across indexable files.
    pub chars_indexable: u64,
    /// Estimated chunk count.
    pub estimated_chunks: u64,
    /// Lower-bound embedding dimension used for estimate ranges.
    pub dimension_low: u32,
    /// Upper-bound embedding dimension used for estimate ranges.
    pub dimension_high: u32,
    /// Lower-bound estimated index bytes.
    pub estimated_bytes_low: u64,
    /// Upper-bound estimated index bytes.
    pub estimated_bytes_high: u64,
    /// Required free-space bytes using selected safety factor.
    pub required_free_bytes: u64,
    /// Safety factor numerator.
    pub safety_factor_num: u64,
    /// Safety factor denominator.
    pub safety_factor_den: u64,
    /// Available free-space bytes at the resolved storage root, when known.
    pub available_bytes: Option<u64>,
    /// Pass/fail/unknown threshold status.
    pub threshold_status: StorageThresholdStatus,
}

#[derive(Debug, Clone, Copy, Default)]
struct FileScanStats {
    files_scanned: u64,
    files_indexable: u64,
    bytes_indexable: u64,
    chars_indexable: u64,
}

/// Estimate local index storage requirements for a codebase.
pub fn estimate_storage_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    codebase_root: &Path,
    danger_close_storage: bool,
) -> InfraResult<CliStorageEstimate> {
    let normalized_root = normalize_root(codebase_root);
    let resolved_config_path = resolve_config_path(config_path, &normalized_root);
    let config = load_backend_config(resolved_config_path.as_deref(), overrides_json)?;
    let scan = scan_indexable_files(&normalized_root, &config)?;

    let max_chunk_chars = u64::from(config.limits().core_max_chunk_chars.get());
    let effective_chunk_chars = mul_div_floor(
        max_chunk_chars,
        CHUNK_UTILIZATION_NUM,
        CHUNK_UTILIZATION_DEN,
    )
    .max(1);
    let estimated_chunks = ceil_div(scan.chars_indexable, effective_chunk_chars);

    let (dimension_low, dimension_high) = estimate_dimension_bounds(&config);
    let estimated_bytes_low =
        estimate_index_bytes(estimated_chunks, dimension_low, config.vector_db.index_mode);
    let estimated_bytes_high = estimate_index_bytes(
        estimated_chunks,
        dimension_high,
        config.vector_db.index_mode,
    );

    let base_required = estimated_bytes_high.max(MIN_WORKING_SET_BYTES);
    let (safety_factor_num, safety_factor_den) = if danger_close_storage {
        (DANGER_FACTOR_NUM, DANGER_FACTOR_DEN)
    } else {
        (SAFE_FACTOR_NUM, SAFE_FACTOR_DEN)
    };
    let required_free_bytes = mul_div_ceil(base_required, safety_factor_num, safety_factor_den);

    let vector_provider = normalize_vector_provider(config.vector_db.provider.as_deref());
    let local_storage_enforced = vector_provider == "local";
    let local_storage_root = if local_storage_enforced {
        config
            .vector_db
            .snapshot_storage
            .resolve_root(&normalized_root)
    } else {
        None
    };
    let available_bytes = local_storage_root
        .as_deref()
        .and_then(probe_available_space_bytes);
    let threshold_status = evaluate_threshold(available_bytes, required_free_bytes);

    Ok(CliStorageEstimate {
        codebase_root: normalized_root,
        vector_provider: vector_provider.into_boxed_str(),
        local_storage_enforced,
        local_storage_root,
        index_mode: config.vector_db.index_mode,
        files_scanned: scan.files_scanned,
        files_indexable: scan.files_indexable,
        bytes_indexable: scan.bytes_indexable,
        chars_indexable: scan.chars_indexable,
        estimated_chunks,
        dimension_low,
        dimension_high,
        estimated_bytes_low,
        estimated_bytes_high,
        required_free_bytes,
        safety_factor_num,
        safety_factor_den,
        available_bytes,
        threshold_status,
    })
}

/// Enforce storage headroom for local indexing.
pub fn ensure_storage_headroom_local(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
    codebase_root: &Path,
    danger_close_storage: bool,
) -> InfraResult<CliStorageEstimate> {
    let estimate = estimate_storage_local(
        config_path,
        overrides_json,
        codebase_root,
        danger_close_storage,
    )?;
    if !estimate.local_storage_enforced {
        return Ok(estimate);
    }
    if estimate.threshold_status != StorageThresholdStatus::Fail {
        return Ok(estimate);
    }

    let available = estimate.available_bytes.unwrap_or(0);
    let safety_factor = format!(
        "{}.{:02}",
        estimate.safety_factor_num / estimate.safety_factor_den,
        (estimate.safety_factor_num % estimate.safety_factor_den) * 100
            / estimate.safety_factor_den
    );
    Err(ErrorEnvelope::expected(
        ErrorCode::new("storage", "insufficient_free_space"),
        format!(
            "insufficient free space for indexing (required: {}, available: {}, safety factor: {}x). Run `sca estimate-storage` for details.",
            estimate.required_free_bytes, available, safety_factor
        ),
    )
    .with_metadata("requiredBytes", estimate.required_free_bytes.to_string())
    .with_metadata("availableBytes", available.to_string())
    .with_metadata(
        "estimatedBytesHigh",
        estimate.estimated_bytes_high.to_string(),
    )
    .with_metadata("filesIndexable", estimate.files_indexable.to_string())
    .with_metadata("estimatedChunks", estimate.estimated_chunks.to_string()))
}

fn resolve_config_path(config_path: Option<&Path>, codebase_root: &Path) -> Option<PathBuf> {
    config_path.map_or_else(
        || {
            let default_path = context_config_path(codebase_root);
            if default_path.exists() {
                Some(default_path)
            } else {
                None
            }
        },
        |path| Some(path.to_path_buf()),
    )
}

fn load_backend_config(
    config_path: Option<&Path>,
    overrides_json: Option<&str>,
) -> InfraResult<ValidatedBackendConfig> {
    semantic_code_config::load_backend_config_std_env(config_path, overrides_json)
}

fn estimate_dimension_bounds(config: &ValidatedBackendConfig) -> (u32, u32) {
    if let Some(dimension) = config.embedding.dimension {
        return (dimension, dimension);
    }

    let provider = normalize_embedding_provider(config.embedding.provider.as_deref());
    match provider.as_str() {
        "test" => (8, 8),
        "onnx" | "local" => (384, 384),
        "openai" => (1536, 1536),
        "voyage" | "voyageai" => (1024, 1536),
        "gemini" | "ollama" => (768, 1536),
        _ => (384, 1536),
    }
}

fn estimate_index_bytes(estimated_chunks: u64, dimension: u32, index_mode: IndexMode) -> u64 {
    let vector_bytes = u64::from(dimension).saturating_mul(4);
    let raw_chunk_bytes = vector_bytes.saturating_add(METADATA_BYTES_PER_CHUNK);
    let dense_raw = estimated_chunks.saturating_mul(raw_chunk_bytes);
    let dense_est = mul_div_ceil(dense_raw, DENSE_FACTOR_NUM, DENSE_FACTOR_DEN);
    match index_mode {
        IndexMode::Dense => dense_est,
        IndexMode::Hybrid => mul_div_ceil(dense_est, HYBRID_FACTOR_NUM, HYBRID_FACTOR_DEN),
    }
}

fn scan_indexable_files(
    codebase_root: &Path,
    config: &ValidatedBackendConfig,
) -> InfraResult<FileScanStats> {
    let mut stats = FileScanStats::default();
    let mut candidate_files: u64 = 0;
    let max_files = u64::from(config.sync.max_files);
    let max_file_size_bytes = config.sync.max_file_size_bytes;
    let supported_extensions = normalize_extensions(&config.sync.allowed_extensions);
    let filter_by_extension = !supported_extensions.is_empty();
    let ignore_patterns = load_ignore_patterns(codebase_root, &config.sync.ignore_patterns)?;
    let matcher = IgnoreMatcher::new();

    let mut dirs = VecDeque::from([String::from(".")]);
    while let Some(dir) = dirs.pop_front() {
        let dir_path = if dir == "." {
            codebase_root.to_path_buf()
        } else {
            codebase_root.join(&dir)
        };

        let Ok(read_dir) = std::fs::read_dir(&dir_path) else {
            continue;
        };

        let mut entries = read_dir.filter_map(Result::ok).collect::<Vec<_>>();
        entries.sort_by_key(std::fs::DirEntry::file_name);

        for entry in entries {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let name = entry.file_name().to_string_lossy().to_string();
            let relative_path = join_relative(&dir, &name);
            if is_ignored(matcher, &ignore_patterns, &relative_path) {
                continue;
            }

            if file_type.is_dir() {
                dirs.push_back(relative_path);
                continue;
            }
            if !file_type.is_file() {
                continue;
            }

            stats.files_scanned = stats.files_scanned.saturating_add(1);

            if filter_by_extension {
                let Some(ext) = file_extension_of(&relative_path) else {
                    continue;
                };
                if !supported_extensions.contains(ext.as_str()) {
                    continue;
                }
            }

            candidate_files = candidate_files.saturating_add(1);

            let Ok(metadata) = entry.metadata() else {
                if candidate_files >= max_files {
                    return Ok(stats);
                }
                continue;
            };

            if metadata.len() > max_file_size_bytes {
                if candidate_files >= max_files {
                    return Ok(stats);
                }
                continue;
            }

            let full_path = codebase_root.join(&relative_path);
            let Ok(content) = std::fs::read_to_string(full_path) else {
                if candidate_files >= max_files {
                    return Ok(stats);
                }
                continue;
            };

            stats.files_indexable = stats.files_indexable.saturating_add(1);
            stats.bytes_indexable = stats
                .bytes_indexable
                .saturating_add(u64_from_usize(content.len()));
            stats.chars_indexable = stats
                .chars_indexable
                .saturating_add(u64_from_usize(content.chars().count()));

            if candidate_files >= max_files {
                return Ok(stats);
            }
        }
    }

    Ok(stats)
}

fn load_ignore_patterns(
    codebase_root: &Path,
    base_patterns: &[Box<str>],
) -> InfraResult<Vec<Box<str>>> {
    let mut patterns = base_patterns.to_vec();
    patterns.push(CONTEXT_IGNORE_FILE.into());
    let context_ignore_path = codebase_root.join(CONTEXT_IGNORE_FILE);
    if context_ignore_path.is_file() {
        let contents = std::fs::read_to_string(context_ignore_path).map_err(ErrorEnvelope::from)?;
        patterns.extend(parse_context_ignore(&contents));
    }
    patterns.sort();
    patterns.dedup();
    Ok(patterns)
}

fn parse_context_ignore(contents: &str) -> Vec<Box<str>> {
    contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| line.to_owned().into_boxed_str())
        .collect()
}

fn normalize_extensions(values: &[Box<str>]) -> HashSet<String> {
    let mut out = HashSet::new();
    for raw in values {
        let trimmed = raw.trim().trim_start_matches('.');
        if trimmed.is_empty() {
            continue;
        }
        out.insert(trimmed.to_ascii_lowercase());
    }
    out
}

fn file_extension_of(path: &str) -> Option<String> {
    let file = path.rsplit('/').next().unwrap_or(path);
    let (_, ext) = file.rsplit_once('.')?;
    if ext.is_empty() {
        return None;
    }
    Some(ext.to_ascii_lowercase())
}

fn join_relative(parent: &str, child: &str) -> String {
    if parent == "." || parent.trim().is_empty() {
        child.to_string()
    } else {
        format!("{parent}/{child}")
    }
}

fn is_ignored(matcher: IgnoreMatcher, ignore_patterns: &[Box<str>], relative_path: &str) -> bool {
    matcher.is_ignored(&IgnoreMatchInput {
        ignore_patterns: ignore_patterns.to_vec(),
        relative_path: relative_path.to_owned().into_boxed_str(),
    })
}

fn normalize_vector_provider(value: Option<&str>) -> String {
    value.unwrap_or("local").trim().to_ascii_lowercase()
}

fn normalize_embedding_provider(value: Option<&str>) -> String {
    value.unwrap_or("auto").trim().to_ascii_lowercase()
}

fn probe_available_space_bytes(path: &Path) -> Option<u64> {
    let probe_path = existing_probe_path(path)?;
    fs2::available_space(probe_path).ok()
}

fn existing_probe_path(path: &Path) -> Option<&Path> {
    let mut cursor = Some(path);
    while let Some(candidate) = cursor {
        if candidate.exists() {
            return Some(candidate);
        }
        cursor = candidate.parent();
    }
    None
}

const fn evaluate_threshold(
    available_bytes: Option<u64>,
    required_free_bytes: u64,
) -> StorageThresholdStatus {
    match available_bytes {
        Some(value) if value >= required_free_bytes => StorageThresholdStatus::Pass,
        Some(_) => StorageThresholdStatus::Fail,
        None => StorageThresholdStatus::Unknown,
    }
}

fn normalize_root(path: &Path) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}

const fn ceil_div(value: u64, divisor: u64) -> u64 {
    if divisor == 0 {
        return 0;
    }
    if value == 0 {
        return 0;
    }
    value.saturating_add(divisor.saturating_sub(1)) / divisor
}

fn mul_div_floor(value: u64, numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        return 0;
    }
    let product = u128::from(value).saturating_mul(u128::from(numerator));
    let quotient = product / u128::from(denominator);
    u64::try_from(quotient).unwrap_or(u64::MAX)
}

fn mul_div_ceil(value: u64, numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        return 0;
    }
    if value == 0 || numerator == 0 {
        return 0;
    }
    let product = u128::from(value).saturating_mul(u128::from(numerator));
    let adjusted = product.saturating_add(u128::from(denominator.saturating_sub(1)));
    let quotient = adjusted / u128::from(denominator);
    u64::try_from(quotient).unwrap_or(u64::MAX)
}

fn u64_from_usize(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_div_handles_zero() {
        assert_eq!(ceil_div(0, 10), 0);
        assert_eq!(ceil_div(11, 10), 2);
    }

    #[test]
    fn threshold_evaluation_works() {
        assert_eq!(
            evaluate_threshold(Some(10), 9),
            StorageThresholdStatus::Pass
        );
        assert_eq!(evaluate_threshold(Some(8), 9), StorageThresholdStatus::Fail);
        assert_eq!(evaluate_threshold(None, 9), StorageThresholdStatus::Unknown);
    }

    #[test]
    fn embedding_dimension_defaults_are_stable() -> InfraResult<()> {
        let validated = semantic_code_config::BackendConfig::default()
            .validate_and_normalize()
            .map_err(ErrorEnvelope::from)?;

        let (low, high) = estimate_dimension_bounds(&validated);
        assert_eq!(low, 384);
        assert_eq!(high, 1536);
        Ok(())
    }
}
