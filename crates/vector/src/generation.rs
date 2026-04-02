//! Exact base-generation bundle helpers.
//!
//! This module defines a kernel-neutral, rebuild-grade collection bundle for
//! exact `f32` rows in canonical origin order. It is intentionally separate
//! from the existing v2 quantized snapshot path: the published generation is
//! the durable source of truth, while kernel-private ready-state is layered on
//! top under namespaced directories.

use crate::{
    ExactVectorRow, ExactVectorRowSource, ExactVectorRowView, ExactVectorRows, OriginId, Result,
    VectorKernelKind, fingerprint_exact_rows,
};
use crc32fast::Hasher;
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope};
use serde::{Deserialize, Serialize};
use std::mem::size_of;
use std::path::{Path, PathBuf};

type ExactGenerationPayload = (Vec<u8>, Vec<Box<str>>, Vec<u64>);

/// Active-generation pointer filename.
pub const GENERATION_ACTIVE_FILE_NAME: &str = "ACTIVE";
/// `SQLite` control-plane catalog filename.
pub const GENERATION_CATALOG_DB_FILE_NAME: &str = "catalog.sqlite";
/// Root generations directory name.
pub const GENERATIONS_DIR_NAME: &str = "generations";
/// Base bundle directory name within one generation.
pub const GENERATION_BASE_DIR_NAME: &str = "base";
/// Kernel-ready states directory name within one generation.
pub const GENERATION_KERNELS_DIR_NAME: &str = "kernels";
/// Derived non-canonical payload directory name within one generation.
pub const GENERATION_DERIVED_DIR_NAME: &str = "derived";

/// Stable generation identifier used for one published build generation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GenerationId(Box<str>);

impl GenerationId {
    /// Build a validated generation id safe for filesystem path joining.
    pub fn new(value: impl AsRef<str>) -> Result<Self> {
        let value = value.as_ref();
        if value.is_empty() {
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("vector", "generation_id_invalid"),
                "generation id must not be empty",
            ));
        }
        if value.contains('/') || value.contains('\\') {
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("vector", "generation_id_invalid"),
                "generation id must not contain path separators",
            )
            .with_metadata("generationId", value.to_string()));
        }
        Ok(Self(value.to_owned().into_boxed_str()))
    }

    /// Borrow the generation id as a string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.0.as_ref()
    }
}

/// Generation-root layout helpers for one collection artifact root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionGenerationPaths {
    root: PathBuf,
}

impl CollectionGenerationPaths {
    /// Build path helpers rooted at one collection artifact directory.
    #[must_use]
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    /// Root directory for this collection artifact set.
    #[must_use]
    pub fn root(&self) -> &Path {
        self.root.as_path()
    }

    /// Path to the active-generation pointer file.
    #[must_use]
    pub fn active_file(&self) -> PathBuf {
        self.root.join(GENERATION_ACTIVE_FILE_NAME)
    }

    /// Read the currently active published generation, if one is present.
    pub fn read_active_generation_id(&self) -> Result<Option<GenerationId>> {
        match std::fs::read_to_string(self.active_file()) {
            Ok(contents) => {
                let trimmed = contents.trim();
                if trimmed.is_empty() {
                    return Ok(None);
                }
                Ok(Some(GenerationId::new(trimmed)?))
            },
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(source) => Err(ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "active_generation_read_failed"),
                "failed to read ACTIVE generation pointer",
                ErrorClass::NonRetriable,
            )
            .with_metadata("path", self.active_file().display().to_string())
            .with_metadata("source", source.to_string())),
        }
    }

    /// Path to the `SQLite` control-plane catalog.
    #[must_use]
    pub fn catalog_db(&self) -> PathBuf {
        self.root.join(GENERATION_CATALOG_DB_FILE_NAME)
    }

    /// Directory containing all immutable generations.
    #[must_use]
    pub fn generations_dir(&self) -> PathBuf {
        self.root.join(GENERATIONS_DIR_NAME)
    }

    /// Build the path helper for one immutable generation.
    #[must_use]
    pub fn generation(&self, generation_id: &GenerationId) -> PublishedGenerationPaths {
        let generation_dir = self.generations_dir().join(generation_id.as_str());
        PublishedGenerationPaths {
            generation_id: generation_id.clone(),
            generation_dir: generation_dir.clone(),
            base_dir: generation_dir.join(GENERATION_BASE_DIR_NAME),
            kernels_dir: generation_dir.join(GENERATION_KERNELS_DIR_NAME),
            derived_dir: generation_dir.join(GENERATION_DERIVED_DIR_NAME),
        }
    }
}

/// Resolved directory layout for one immutable published generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishedGenerationPaths {
    generation_id: GenerationId,
    generation_dir: PathBuf,
    base_dir: PathBuf,
    kernels_dir: PathBuf,
    derived_dir: PathBuf,
}

impl PublishedGenerationPaths {
    /// Generation identifier for this layout.
    #[must_use]
    pub const fn generation_id(&self) -> &GenerationId {
        &self.generation_id
    }

    /// Root directory for this immutable generation.
    #[must_use]
    pub fn generation_dir(&self) -> &Path {
        self.generation_dir.as_path()
    }

    /// Kernel-neutral base bundle directory.
    #[must_use]
    pub fn base_dir(&self) -> &Path {
        self.base_dir.as_path()
    }

    /// Exact generation metadata file.
    #[must_use]
    pub fn base_meta_file(&self) -> PathBuf {
        self.base_dir.join(EXACT_GENERATION_META_FILE_NAME)
    }

    /// Exact generation vector payload file.
    #[must_use]
    pub fn base_vectors_file(&self) -> PathBuf {
        self.base_dir.join(EXACT_GENERATION_VECTORS_FILE_NAME)
    }

    /// Exact generation ids file.
    #[must_use]
    pub fn base_ids_file(&self) -> PathBuf {
        self.base_dir.join(EXACT_GENERATION_IDS_FILE_NAME)
    }

    /// Exact generation origins file.
    #[must_use]
    pub fn base_origins_file(&self) -> PathBuf {
        self.base_dir.join(EXACT_GENERATION_ORIGINS_FILE_NAME)
    }

    /// Root of all kernel-private ready-state directories.
    #[must_use]
    pub fn kernels_dir(&self) -> &Path {
        self.kernels_dir.as_path()
    }

    /// Root of derived non-canonical payloads.
    #[must_use]
    pub fn derived_dir(&self) -> &Path {
        self.derived_dir.as_path()
    }

    /// Path to one kernel-private ready-state directory.
    #[must_use]
    pub fn kernel_dir(&self, kernel: VectorKernelKind) -> PathBuf {
        self.kernels_dir.join(kernel_dir_name(kernel))
    }
}

/// Exact base-generation metadata filename.
pub const EXACT_GENERATION_META_FILE_NAME: &str = "snapshot.meta";
/// Exact `f32` vector payload filename.
pub const EXACT_GENERATION_VECTORS_FILE_NAME: &str = "vectors.f32.bin";
/// Exact row ids filename.
pub const EXACT_GENERATION_IDS_FILE_NAME: &str = "ids.json";
/// Exact row origins filename.
pub const EXACT_GENERATION_ORIGINS_FILE_NAME: &str = "origins.json";
/// Current exact base-generation schema version.
pub const EXACT_GENERATION_VERSION_V3: u8 = 3;

/// Normalization mode used by the exact vector payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExactVectorNormalization {
    /// Vectors are exact unit-length cosine-ready rows.
    UnitCosine,
}

/// Metadata for an exact base-generation bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExactGenerationMeta {
    /// Schema version for the exact generation bundle.
    pub version: u8,
    /// Shared vector dimension.
    pub dimension: u32,
    /// Row count.
    pub count: u64,
    /// Normalization mode for persisted vectors.
    pub normalization: ExactVectorNormalization,
    /// Deterministic fingerprint over exact rows in canonical origin order.
    pub rows_fingerprint: u64,
    /// CRC32 checksum for `vectors.f32.bin`.
    pub vectors_crc32: u32,
}

impl ExactGenerationMeta {
    /// Build validated exact-generation metadata.
    pub fn new(
        dimension: u32,
        count: u64,
        normalization: ExactVectorNormalization,
        rows_fingerprint: u64,
        vectors_crc32: u32,
    ) -> Result<Self> {
        if dimension == 0 {
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("vector", "invalid_dimension"),
                "exact generation dimension must be greater than zero",
            ));
        }

        Ok(Self {
            version: EXACT_GENERATION_VERSION_V3,
            dimension,
            count,
            normalization,
            rows_fingerprint,
            vectors_crc32,
        })
    }
}

/// Write an exact base-generation bundle for canonical rows.
pub fn write_exact_generation(
    generation_dir: impl AsRef<Path>,
    rows: &ExactVectorRows,
) -> Result<ExactGenerationMeta> {
    let generation_dir = generation_dir.as_ref();
    std::fs::create_dir_all(generation_dir).map_err(|source| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "exact_generation_dir_create_failed"),
            "failed to create exact generation directory",
            ErrorClass::NonRetriable,
        )
        .with_metadata("path", generation_dir.display().to_string())
        .with_metadata("source", source.to_string())
    })?;

    let (vector_bytes, ids, origins) = collect_exact_generation_payload(rows)?;
    let vectors_crc32 = compute_crc32(vector_bytes.as_slice());
    let meta = build_exact_generation_meta(rows, vectors_crc32)?;

    write_exact_generation_files(generation_dir, &meta, &ids, &origins, &vector_bytes)?;

    Ok(meta)
}

/// Read an exact base-generation bundle back into owned canonical rows.
pub fn read_exact_generation(generation_dir: impl AsRef<Path>) -> Result<ExactVectorRows> {
    let generation_dir = generation_dir.as_ref();
    let meta = read_exact_generation_meta(generation_dir)?;
    let ids = read_exact_generation_ids(generation_dir)?;
    let origins = read_exact_generation_origins(generation_dir)?;
    let vector_bytes = read_exact_generation_vectors(generation_dir)?;

    verify_exact_generation_consistency(&meta, &ids, &origins, vector_bytes.as_slice())?;
    let exact_rows = decode_exact_generation_rows(&meta, ids, origins, vector_bytes.as_slice())?;

    let fingerprint = fingerprint_exact_rows(
        exact_rows.dimension(),
        exact_rows.row_count(),
        exact_rows.rows(),
    );
    if fingerprint != meta.rows_fingerprint {
        return Err(ErrorEnvelope::expected(
            ErrorCode::new("vector", "exact_generation_rows_fingerprint_mismatch"),
            "exact generation rows fingerprint mismatch",
        )
        .with_metadata("expected", meta.rows_fingerprint.to_string())
        .with_metadata("found", fingerprint.to_string()));
    }

    Ok(exact_rows)
}

fn compute_crc32(bytes: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

const fn kernel_dir_name(kernel: VectorKernelKind) -> &'static str {
    match kernel {
        VectorKernelKind::HnswRs => "hnsw-rs",
        VectorKernelKind::Dfrr => "dfrr",
        VectorKernelKind::FlatScan => "flat-scan",
    }
}

fn collect_exact_generation_payload(rows: &ExactVectorRows) -> Result<ExactGenerationPayload> {
    let mut vector_bytes = Vec::with_capacity(
        rows.row_count().saturating_mul(
            usize::try_from(rows.dimension())
                .unwrap_or(0)
                .saturating_mul(size_of::<f32>()),
        ),
    );
    let mut ids = Vec::with_capacity(rows.row_count());
    let mut origins = Vec::with_capacity(rows.row_count());

    for row in rows.rows() {
        ids.push(Box::<str>::from(row.id()));
        origins.push(u64::try_from(row.origin().as_usize()).map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "exact_generation_origin_overflow"),
                "exact generation origin conversion overflow",
                ErrorClass::NonRetriable,
            )
            .with_metadata("origin", row.origin().as_usize().to_string())
        })?);
        for value in row.vector() {
            vector_bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    Ok((vector_bytes, ids, origins))
}

fn build_exact_generation_meta(
    rows: &ExactVectorRows,
    vectors_crc32: u32,
) -> Result<ExactGenerationMeta> {
    ExactGenerationMeta::new(
        rows.dimension(),
        u64::try_from(rows.row_count()).map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "exact_generation_count_overflow"),
                "exact generation row-count conversion overflow",
                ErrorClass::NonRetriable,
            )
            .with_metadata("count", rows.row_count().to_string())
        })?,
        ExactVectorNormalization::UnitCosine,
        rows.fingerprint(),
        vectors_crc32,
    )
}

fn write_exact_generation_files(
    generation_dir: &Path,
    meta: &ExactGenerationMeta,
    ids: &[Box<str>],
    origins: &[u64],
    vector_bytes: &[u8],
) -> Result<()> {
    let meta_path = generation_dir.join(EXACT_GENERATION_META_FILE_NAME);
    let ids_path = generation_dir.join(EXACT_GENERATION_IDS_FILE_NAME);
    let origins_path = generation_dir.join(EXACT_GENERATION_ORIGINS_FILE_NAME);
    let vectors_path = generation_dir.join(EXACT_GENERATION_VECTORS_FILE_NAME);

    let meta_bytes = serde_json::to_vec_pretty(meta).map_err(|source| {
        ErrorEnvelope::expected(
            ErrorCode::new("vector", "exact_generation_meta_serialize_failed"),
            "failed to serialize exact generation metadata",
        )
        .with_metadata("path", meta_path.display().to_string())
        .with_metadata("source", source.to_string())
    })?;
    std::fs::write(&meta_path, meta_bytes).map_err(|source| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "exact_generation_meta_write_failed"),
            "failed to write exact generation metadata",
            ErrorClass::NonRetriable,
        )
        .with_metadata("path", meta_path.display().to_string())
        .with_metadata("source", source.to_string())
    })?;

    let ids_bytes = serde_json::to_vec(ids).map_err(|source| {
        ErrorEnvelope::expected(
            ErrorCode::new("vector", "exact_generation_ids_serialize_failed"),
            "failed to serialize exact generation ids",
        )
        .with_metadata("path", ids_path.display().to_string())
        .with_metadata("source", source.to_string())
    })?;
    std::fs::write(&ids_path, ids_bytes).map_err(|source| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "exact_generation_ids_write_failed"),
            "failed to write exact generation ids",
            ErrorClass::NonRetriable,
        )
        .with_metadata("path", ids_path.display().to_string())
        .with_metadata("source", source.to_string())
    })?;

    let origins_bytes = serde_json::to_vec(origins).map_err(|source| {
        ErrorEnvelope::expected(
            ErrorCode::new("vector", "exact_generation_origins_serialize_failed"),
            "failed to serialize exact generation origins",
        )
        .with_metadata("path", origins_path.display().to_string())
        .with_metadata("source", source.to_string())
    })?;
    std::fs::write(&origins_path, origins_bytes).map_err(|source| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "exact_generation_origins_write_failed"),
            "failed to write exact generation origins",
            ErrorClass::NonRetriable,
        )
        .with_metadata("path", origins_path.display().to_string())
        .with_metadata("source", source.to_string())
    })?;

    std::fs::write(&vectors_path, vector_bytes).map_err(|source| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "exact_generation_vectors_write_failed"),
            "failed to write exact generation vectors",
            ErrorClass::NonRetriable,
        )
        .with_metadata("path", vectors_path.display().to_string())
        .with_metadata("source", source.to_string())
    })?;

    Ok(())
}

fn read_exact_generation_meta(generation_dir: &Path) -> Result<ExactGenerationMeta> {
    let meta_path = generation_dir.join(EXACT_GENERATION_META_FILE_NAME);
    let meta = serde_json::from_slice::<ExactGenerationMeta>(
        std::fs::read(&meta_path)
            .map_err(|source| {
                ErrorEnvelope::unexpected(
                    ErrorCode::new("vector", "exact_generation_meta_read_failed"),
                    "failed to read exact generation metadata",
                    ErrorClass::NonRetriable,
                )
                .with_metadata("path", meta_path.display().to_string())
                .with_metadata("source", source.to_string())
            })?
            .as_slice(),
    )
    .map_err(|source| {
        ErrorEnvelope::expected(
            ErrorCode::new("vector", "exact_generation_meta_parse_failed"),
            "failed to parse exact generation metadata",
        )
        .with_metadata("path", meta_path.display().to_string())
        .with_metadata("source", source.to_string())
    })?;

    if meta.version != EXACT_GENERATION_VERSION_V3 {
        return Err(ErrorEnvelope::expected(
            ErrorCode::new("vector", "exact_generation_version_mismatch"),
            "unsupported exact generation version",
        )
        .with_metadata("expected", EXACT_GENERATION_VERSION_V3.to_string())
        .with_metadata("found", meta.version.to_string()));
    }

    Ok(meta)
}

fn read_exact_generation_ids(generation_dir: &Path) -> Result<Vec<Box<str>>> {
    let ids_path = generation_dir.join(EXACT_GENERATION_IDS_FILE_NAME);
    serde_json::from_slice::<Vec<Box<str>>>(
        std::fs::read(&ids_path)
            .map_err(|source| {
                ErrorEnvelope::unexpected(
                    ErrorCode::new("vector", "exact_generation_ids_read_failed"),
                    "failed to read exact generation ids",
                    ErrorClass::NonRetriable,
                )
                .with_metadata("path", ids_path.display().to_string())
                .with_metadata("source", source.to_string())
            })?
            .as_slice(),
    )
    .map_err(|source| {
        ErrorEnvelope::expected(
            ErrorCode::new("vector", "exact_generation_ids_parse_failed"),
            "failed to parse exact generation ids",
        )
        .with_metadata("path", ids_path.display().to_string())
        .with_metadata("source", source.to_string())
    })
}

fn read_exact_generation_origins(generation_dir: &Path) -> Result<Vec<u64>> {
    let origins_path = generation_dir.join(EXACT_GENERATION_ORIGINS_FILE_NAME);
    serde_json::from_slice::<Vec<u64>>(
        std::fs::read(&origins_path)
            .map_err(|source| {
                ErrorEnvelope::unexpected(
                    ErrorCode::new("vector", "exact_generation_origins_read_failed"),
                    "failed to read exact generation origins",
                    ErrorClass::NonRetriable,
                )
                .with_metadata("path", origins_path.display().to_string())
                .with_metadata("source", source.to_string())
            })?
            .as_slice(),
    )
    .map_err(|source| {
        ErrorEnvelope::expected(
            ErrorCode::new("vector", "exact_generation_origins_parse_failed"),
            "failed to parse exact generation origins",
        )
        .with_metadata("path", origins_path.display().to_string())
        .with_metadata("source", source.to_string())
    })
}

fn read_exact_generation_vectors(generation_dir: &Path) -> Result<Vec<u8>> {
    let vectors_path = generation_dir.join(EXACT_GENERATION_VECTORS_FILE_NAME);
    std::fs::read(&vectors_path).map_err(|source| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "exact_generation_vectors_read_failed"),
            "failed to read exact generation vectors",
            ErrorClass::NonRetriable,
        )
        .with_metadata("path", vectors_path.display().to_string())
        .with_metadata("source", source.to_string())
    })
}

fn verify_exact_generation_consistency(
    meta: &ExactGenerationMeta,
    ids: &[Box<str>],
    origins: &[u64],
    vector_bytes: &[u8],
) -> Result<()> {
    if ids.len() != origins.len() {
        return Err(ErrorEnvelope::expected(
            ErrorCode::new("vector", "exact_generation_row_count_mismatch"),
            "exact generation ids and origins count mismatch",
        )
        .with_metadata("ids", ids.len().to_string())
        .with_metadata("origins", origins.len().to_string()));
    }

    let actual_crc = compute_crc32(vector_bytes);
    if actual_crc != meta.vectors_crc32 {
        return Err(ErrorEnvelope::expected(
            ErrorCode::new("vector", "exact_generation_vectors_crc_mismatch"),
            "exact generation vectors checksum mismatch",
        )
        .with_metadata("expected", meta.vectors_crc32.to_string())
        .with_metadata("found", actual_crc.to_string()));
    }

    let dimension = usize::try_from(meta.dimension).map_err(|_| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "exact_generation_dimension_overflow"),
            "exact generation dimension conversion overflow",
            ErrorClass::NonRetriable,
        )
        .with_metadata("dimension", meta.dimension.to_string())
    })?;
    let expected_bytes = ids
        .len()
        .checked_mul(dimension)
        .and_then(|value| value.checked_mul(size_of::<f32>()))
        .ok_or_else(|| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "exact_generation_vector_bytes_overflow"),
                "exact generation vector byte length overflow",
                ErrorClass::NonRetriable,
            )
            .with_metadata("count", ids.len().to_string())
            .with_metadata("dimension", dimension.to_string())
        })?;
    if vector_bytes.len() != expected_bytes {
        return Err(ErrorEnvelope::expected(
            ErrorCode::new("vector", "exact_generation_vector_bytes_mismatch"),
            "exact generation vector byte length mismatch",
        )
        .with_metadata("expected", expected_bytes.to_string())
        .with_metadata("found", vector_bytes.len().to_string()));
    }

    Ok(())
}

fn decode_exact_generation_rows(
    meta: &ExactGenerationMeta,
    ids: Vec<Box<str>>,
    origins: Vec<u64>,
    vector_bytes: &[u8],
) -> Result<ExactVectorRows> {
    let dimension = usize::try_from(meta.dimension).map_err(|_| {
        ErrorEnvelope::unexpected(
            ErrorCode::new("vector", "exact_generation_dimension_overflow"),
            "exact generation dimension conversion overflow",
            ErrorClass::NonRetriable,
        )
        .with_metadata("dimension", meta.dimension.to_string())
    })?;

    let mut rows = Vec::with_capacity(ids.len());
    let mut chunk_iter = vector_bytes.chunks_exact(size_of::<f32>());
    for (row_index, (id, origin)) in ids.into_iter().zip(origins).enumerate() {
        let mut vector = Vec::with_capacity(dimension);
        for _ in 0..dimension {
            let Some(chunk) = chunk_iter.next() else {
                return Err(ErrorEnvelope::expected(
                    ErrorCode::new("vector", "exact_generation_vector_bytes_mismatch"),
                    "exact generation vector bytes ended early",
                )
                .with_metadata("rowIndex", row_index.to_string()));
            };
            let bytes = <[u8; 4]>::try_from(chunk).map_err(|_| {
                ErrorEnvelope::unexpected(
                    ErrorCode::new("vector", "exact_generation_vector_decode_failed"),
                    "failed to decode exact generation vector bytes",
                    ErrorClass::NonRetriable,
                )
                .with_metadata("rowIndex", row_index.to_string())
            })?;
            vector.push(f32::from_le_bytes(bytes));
        }
        let origin = usize::try_from(origin).map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "exact_generation_origin_overflow"),
                "exact generation origin conversion overflow",
                ErrorClass::NonRetriable,
            )
            .with_metadata("origin", origin.to_string())
        })?;
        rows.push(ExactVectorRow::new(
            id,
            OriginId::from_usize(origin),
            vector,
        ));
    }

    ExactVectorRows::new(meta.dimension, rows)
}

#[cfg(test)]
mod tests {
    use super::{
        CollectionGenerationPaths, EXACT_GENERATION_META_FILE_NAME, ExactVectorNormalization,
        GENERATION_ACTIVE_FILE_NAME, GENERATION_BASE_DIR_NAME, GENERATION_CATALOG_DB_FILE_NAME,
        GENERATION_DERIVED_DIR_NAME, GENERATION_KERNELS_DIR_NAME, GENERATIONS_DIR_NAME,
        GenerationId, read_exact_generation, write_exact_generation,
    };
    use crate::{
        ExactVectorRow, ExactVectorRowSource, ExactVectorRowView, ExactVectorRows, OriginId,
        VectorKernelKind,
    };
    use semantic_code_shared::ErrorEnvelope;
    use std::io;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn create(prefix: &str) -> io::Result<Self> {
            let path = unique_temp_path(prefix);
            std::fs::create_dir_all(&path)?;
            Ok(Self { path })
        }

        fn path(&self) -> &Path {
            self.path.as_path()
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn unique_temp_path(prefix: &str) -> PathBuf {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);

        let pid = std::process::id();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let seq = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("{prefix}-{pid}-{nanos}-{seq}"))
    }

    #[test]
    fn exact_generation_roundtrip_preserves_rows_and_metadata() -> crate::Result<()> {
        let temp = TempDir::create("exact-generation-roundtrip").map_err(ErrorEnvelope::from)?;
        let rows = ExactVectorRows::new(
            2,
            vec![
                ExactVectorRow::new("a", OriginId::from_usize(0), vec![1.0, 0.0]),
                ExactVectorRow::new("b", OriginId::from_usize(1), vec![0.0, 1.0]),
            ],
        )?;

        let meta = write_exact_generation(temp.path(), &rows)?;
        assert_eq!(meta.version, super::EXACT_GENERATION_VERSION_V3);
        assert_eq!(meta.normalization, ExactVectorNormalization::UnitCosine);
        assert!(temp.path().join(EXACT_GENERATION_META_FILE_NAME).is_file());

        let restored = read_exact_generation(temp.path())?;
        let restored_rows = restored
            .rows()
            .map(|row| {
                (
                    row.origin().as_usize(),
                    row.id().to_string(),
                    row.vector().to_vec(),
                )
            })
            .collect::<Vec<(usize, String, Vec<f32>)>>();
        assert_eq!(
            restored_rows,
            vec![
                (0, "a".to_string(), vec![1.0, 0.0]),
                (1, "b".to_string(), vec![0.0, 1.0]),
            ]
        );
        assert_eq!(restored.fingerprint(), rows.fingerprint());

        Ok(())
    }

    #[test]
    fn generation_id_rejects_path_separators() {
        let error = GenerationId::new("bad/path")
            .err()
            .expect("path separators should be rejected");
        assert!(
            error.to_string().contains("path separators"),
            "unexpected generation-id error: {error}"
        );
    }

    #[test]
    fn collection_generation_paths_resolve_generation_scaffold() -> crate::Result<()> {
        let root = PathBuf::from("/tmp/collection-root");
        let layout = CollectionGenerationPaths::new(root.as_path());
        let generation_id = GenerationId::new("gen-001")?;
        let generation = layout.generation(&generation_id);

        assert_eq!(layout.active_file(), root.join(GENERATION_ACTIVE_FILE_NAME));
        assert_eq!(
            layout.catalog_db(),
            root.join(GENERATION_CATALOG_DB_FILE_NAME)
        );
        assert_eq!(layout.generations_dir(), root.join(GENERATIONS_DIR_NAME));
        assert_eq!(
            generation.base_dir(),
            root.join(GENERATIONS_DIR_NAME)
                .join("gen-001")
                .join(GENERATION_BASE_DIR_NAME)
        );
        assert_eq!(
            generation.kernels_dir(),
            root.join(GENERATIONS_DIR_NAME)
                .join("gen-001")
                .join(GENERATION_KERNELS_DIR_NAME)
        );
        assert_eq!(
            generation.derived_dir(),
            root.join(GENERATIONS_DIR_NAME)
                .join("gen-001")
                .join(GENERATION_DERIVED_DIR_NAME)
        );
        assert_eq!(
            generation.kernel_dir(VectorKernelKind::HnswRs),
            root.join(GENERATIONS_DIR_NAME)
                .join("gen-001")
                .join(GENERATION_KERNELS_DIR_NAME)
                .join("hnsw-rs")
        );
        assert_eq!(
            generation.kernel_dir(VectorKernelKind::Dfrr),
            root.join(GENERATIONS_DIR_NAME)
                .join("gen-001")
                .join(GENERATION_KERNELS_DIR_NAME)
                .join("dfrr")
        );
        assert_eq!(
            generation.kernel_dir(VectorKernelKind::FlatScan),
            root.join(GENERATIONS_DIR_NAME)
                .join("gen-001")
                .join(GENERATION_KERNELS_DIR_NAME)
                .join("flat-scan")
        );
        assert_eq!(
            generation.base_meta_file(),
            root.join(GENERATIONS_DIR_NAME)
                .join("gen-001")
                .join(GENERATION_BASE_DIR_NAME)
                .join(EXACT_GENERATION_META_FILE_NAME)
        );

        Ok(())
    }

    #[test]
    fn active_generation_reader_handles_present_and_missing_files() -> crate::Result<()> {
        let temp =
            TempDir::create("exact-generation-active-reader").map_err(ErrorEnvelope::from)?;
        let layout = CollectionGenerationPaths::new(temp.path());

        assert!(layout.read_active_generation_id()?.is_none());

        std::fs::write(layout.active_file(), "gen-abc").map_err(ErrorEnvelope::from)?;
        let active = layout.read_active_generation_id()?;
        assert_eq!(active.as_ref().map(GenerationId::as_str), Some("gen-abc"));

        Ok(())
    }
}
