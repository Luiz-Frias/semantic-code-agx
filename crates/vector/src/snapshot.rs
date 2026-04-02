//! Snapshot v2 metadata and binary bundle helpers.
//!
//! This module defines a typed metadata envelope plus companion binary payload
//! read/write helpers backed by safe mmap wrappers.

use crate::mmap::{MmapBytes, MmapError};
use crate::quantization::{
    QuantizationError, QuantizationParams, QuantizedSlice, Quantizer, fit_min_max,
};
use crate::{HnswParams, VectorKernelKind, VectorSnapshot};
use crc32fast::Hasher;
use serde::{Deserialize, Serialize};
use std::io;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::instrument;

/// Result type for snapshot metadata and bundle operations.
pub type SnapshotResult<T> = Result<T, SnapshotError>;

/// Legacy JSON snapshot version identifier.
pub const SNAPSHOT_VERSION_V1: u8 = 1;
/// Binary companion snapshot version identifier.
pub const SNAPSHOT_VERSION_V2: u8 = 2;

/// Legacy v1 JSON snapshot filename used by upgrade helpers.
pub const SNAPSHOT_V1_FILE_NAME: &str = "snapshot.v1.json";

/// v2 metadata filename within a snapshot directory.
pub const SNAPSHOT_V2_META_FILE_NAME: &str = "snapshot.meta";
/// v2 quantized vector payload filename.
pub const SNAPSHOT_V2_VECTORS_FILE_NAME: &str = "vectors.u8.bin";
/// Basename for the `hnsw_rs` graph dump files within a v2 snapshot directory.
///
/// `file_dump` produces `$basename.hnsw.graph` (topology) and `$basename.hnsw.data`
/// (vector data), both living inside the snapshot directory.
pub const SNAPSHOT_V2_HNSW_GRAPH_BASENAME: &str = "hnsw_graph";

/// Magic header prefix for v2 metadata files.
pub const SNAPSHOT_V2_MAGIC_HEADER: &[u8; 13] = b"SCA-SNAPSHOT\n";
/// Offset where metadata JSON bytes start after the magic header.
pub const SNAPSHOT_V2_META_JSON_OFFSET: usize = SNAPSHOT_V2_MAGIC_HEADER.len();
/// Checksum algorithm used for vector payload integrity.
pub const SNAPSHOT_V2_CHECKSUM_ALGORITHM: &str = "crc32";

/// Snapshot schema version discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorSnapshotVersion {
    /// Legacy JSON snapshot (`VectorSnapshot` in `lib.rs`).
    V1,
    /// v2 metadata + binary companion files.
    V2,
}

impl VectorSnapshotVersion {
    /// Numeric wire version.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::V1 => SNAPSHOT_VERSION_V1,
            Self::V2 => SNAPSHOT_VERSION_V2,
        }
    }
}

impl TryFrom<u8> for VectorSnapshotVersion {
    type Error = SnapshotError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            SNAPSHOT_VERSION_V1 => Ok(Self::V1),
            SNAPSHOT_VERSION_V2 => Ok(Self::V2),
            found => Err(SnapshotError::UnsupportedVersion { found }),
        }
    }
}

impl Serialize for VectorSnapshotVersion {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u8(self.as_u8())
    }
}

impl<'de> Deserialize<'de> for VectorSnapshotVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = u8::deserialize(deserializer)?;
        Self::try_from(raw).map_err(serde::de::Error::custom)
    }
}

/// v2 snapshot metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorSnapshotMeta {
    /// Snapshot schema version.
    pub version: VectorSnapshotVersion,
    /// Vector dimensionality.
    pub dimension: u32,
    /// Number of vectors in the companion payload.
    pub count: u64,
    /// HNSW parameters used by the index.
    pub params: HnswParams,
    /// Kernel family used when this snapshot was produced.
    #[serde(default)]
    pub kernel: VectorKernelKind,
    /// SQ8 quantization parameters used for the `vectors.u8.bin` payload.
    pub quantization: QuantizationParams,
    /// CRC32 checksum for `vectors.u8.bin` bytes.
    pub vectors_crc32: u32,
}

impl VectorSnapshotMeta {
    /// Build a validated v2 metadata payload.
    pub fn new(
        dimension: u32,
        count: u64,
        params: HnswParams,
        quantization: QuantizationParams,
        vectors_crc32: u32,
    ) -> SnapshotResult<Self> {
        Self::new_with_kernel(
            dimension,
            count,
            params,
            VectorKernelKind::HnswRs,
            quantization,
            vectors_crc32,
        )
    }

    /// Build a validated v2 metadata payload with an explicit kernel family.
    pub fn new_with_kernel(
        dimension: u32,
        count: u64,
        params: HnswParams,
        kernel: VectorKernelKind,
        quantization: QuantizationParams,
        vectors_crc32: u32,
    ) -> SnapshotResult<Self> {
        let meta = Self {
            version: VectorSnapshotVersion::V2,
            dimension,
            count,
            params,
            kernel,
            quantization,
            vectors_crc32,
        };
        validate_meta(&meta)?;
        Ok(meta)
    }
}

/// Options for reading v2 snapshots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReadSnapshotV2Options {
    /// When true and `snapshot.meta` is missing, attempt a v1 -> v2 upgrade
    /// using `snapshot.v1.json` inside the same directory.
    pub(crate) auto_upgrade_v1: bool,
}

/// Loaded v2 snapshot bundle.
#[derive(Debug)]
pub struct VectorSnapshotV2 {
    /// Metadata loaded from `snapshot.meta`.
    pub(crate) meta: VectorSnapshotMeta,
    vectors: MmapBytes,
}

impl VectorSnapshotV2 {
    /// Borrow mapped quantized vector bytes.
    #[must_use]
    pub(crate) fn vectors(&self) -> &[u8] {
        self.vectors.as_slice()
    }

    /// Borrow mapped vectors as checked dimensional chunks.
    pub(crate) fn quantized_vectors(&self) -> SnapshotResult<QuantizedSlice<'_>> {
        let dimension = usize::try_from(self.meta.dimension).map_err(|_| {
            SnapshotError::DimensionConversionOverflow {
                dimension: self.meta.dimension,
            }
        })?;
        QuantizedSlice::new(self.vectors(), dimension)
            .map_err(|source| SnapshotError::Quantization { source })
    }
}

/// Typed snapshot metadata and bundle errors.
#[derive(Debug, Error)]
pub enum SnapshotError {
    /// Snapshot version is unknown.
    #[error("unsupported snapshot version: {found}")]
    UnsupportedVersion {
        /// Unsupported version value.
        found: u8,
    },
    /// Snapshot magic header was missing.
    #[error("invalid snapshot metadata magic header")]
    InvalidMagicHeader,
    /// Snapshot dimension must be greater than zero.
    #[error("snapshot dimension must be greater than zero")]
    InvalidDimension,
    /// Quantization dimension does not match metadata dimension.
    #[error(
        "snapshot quantization dimension mismatch: metadata={metadata_dimension}, quantization={quantization_dimension}"
    )]
    QuantizationDimensionMismatch {
        /// Metadata dimension.
        metadata_dimension: u32,
        /// Quantization dimension.
        quantization_dimension: usize,
    },
    /// Snapshot dimension failed conversion to `usize`.
    #[error("snapshot dimension conversion overflow: {dimension}")]
    DimensionConversionOverflow {
        /// Dimension value that overflowed.
        dimension: u32,
    },
    /// Quantization dimension failed conversion to `u32`.
    #[error("quantization dimension conversion overflow: {dimension}")]
    QuantizationDimensionOverflow {
        /// Quantization dimension value that overflowed.
        dimension: usize,
    },
    /// `usize` vector byte length failed conversion to `u64`.
    #[error("vector byte length conversion overflow: {len}")]
    VectorLengthConversionOverflow {
        /// Vector byte length that overflowed.
        len: usize,
    },
    /// Legacy v1 record count failed conversion.
    #[error("legacy snapshot record count conversion overflow: {count}")]
    RecordCountConversionOverflow {
        /// Record count that overflowed.
        count: usize,
    },
    /// Metadata payload cannot be serialized.
    #[error("failed to serialize snapshot metadata")]
    SerializeMetadata {
        /// Underlying serialization error.
        #[source]
        source: serde_json::Error,
    },
    /// Metadata payload cannot be deserialized.
    #[error("failed to deserialize snapshot metadata")]
    DeserializeMetadata {
        /// Underlying deserialization error.
        #[source]
        source: serde_json::Error,
    },
    /// File read failed.
    #[error("failed to read snapshot metadata: {path}")]
    ReadMetadata {
        /// Path that failed to read.
        path: PathBuf,
        /// Underlying IO error.
        #[source]
        source: io::Error,
    },
    /// File write failed.
    #[error("failed to write snapshot metadata: {path}")]
    WriteMetadata {
        /// Path that failed to write.
        path: PathBuf,
        /// Underlying IO error.
        #[source]
        source: io::Error,
    },
    /// Parent directory creation failed.
    #[error("failed to create snapshot metadata directory: {path}")]
    CreateMetadataDir {
        /// Parent directory path.
        path: PathBuf,
        /// Underlying IO error.
        #[source]
        source: io::Error,
    },
    /// Snapshot bundle directory creation failed.
    #[error("failed to create snapshot directory: {path}")]
    CreateSnapshotDir {
        /// Snapshot directory path.
        path: PathBuf,
        /// Underlying IO error.
        #[source]
        source: io::Error,
    },
    /// Vector payload write failed.
    #[error("failed to write snapshot vectors payload: {path}")]
    WriteVectors {
        /// Vector payload path.
        path: PathBuf,
        /// Underlying IO error.
        #[source]
        source: io::Error,
    },
    /// Header version and metadata version diverged.
    #[error(
        "snapshot metadata version mismatch: header={header_version}, metadata={metadata_version}"
    )]
    MetadataVersionMismatch {
        /// Version decoded from header.
        header_version: u8,
        /// Version decoded from metadata JSON.
        metadata_version: u8,
    },
    /// Expected vector length did not match payload length.
    #[error(
        "snapshot vector byte length mismatch for {path}: expected {expected} bytes, found {found} bytes"
    )]
    VectorByteLengthMismatch {
        /// Vector payload path.
        path: PathBuf,
        /// Expected bytes from metadata.
        expected: u64,
        /// Actual bytes found.
        found: u64,
    },
    /// `count * dimension` overflowed the output type.
    #[error("expected vector payload length overflow for count={count}, dimension={dimension}")]
    ExpectedVectorLengthOverflow {
        /// Number of vectors.
        count: u64,
        /// Vector dimension.
        dimension: u32,
    },
    /// Mapping vector payload failed.
    #[error("failed to map snapshot vectors payload: {path}")]
    MapVectors {
        /// Vector payload path.
        path: PathBuf,
        /// Underlying mmap error.
        #[source]
        source: MmapError,
    },
    /// Vector payload checksum mismatch indicates corruption.
    #[error(
        "snapshot vectors checksum mismatch for {path}: expected {expected:08x}, found {found:08x} ({algorithm})"
    )]
    VectorsChecksumMismatch {
        /// Vector payload path.
        path: PathBuf,
        /// Expected checksum from metadata.
        expected: u32,
        /// Checksum computed from file bytes.
        found: u32,
        /// Checksum algorithm.
        algorithm: &'static str,
    },
    /// Legacy v1 snapshot read failed.
    #[error("failed to read legacy v1 snapshot: {path}")]
    ReadLegacySnapshot {
        /// Legacy snapshot path.
        path: PathBuf,
        /// Underlying IO error.
        #[source]
        source: io::Error,
    },
    /// Legacy v1 snapshot parse failed.
    #[error("failed to parse legacy v1 snapshot: {path}")]
    DeserializeLegacySnapshot {
        /// Legacy snapshot path.
        path: PathBuf,
        /// Underlying parse error.
        #[source]
        source: serde_json::Error,
    },
    /// Legacy snapshot version does not match expected v1.
    #[error("legacy snapshot version mismatch: expected {expected}, found {found}")]
    LegacyVersionMismatch {
        /// Expected version value.
        expected: u32,
        /// Found version value.
        found: u32,
    },
    /// Legacy snapshot contains a record with the wrong dimension.
    #[error(
        "legacy snapshot record dimension mismatch at index {record_index}: expected {expected}, found {found}"
    )]
    LegacyRecordDimensionMismatch {
        /// Record index.
        record_index: usize,
        /// Expected dimension.
        expected: usize,
        /// Found dimension.
        found: usize,
    },
    /// Quantization operation failed.
    #[error("snapshot quantization error")]
    Quantization {
        /// Underlying quantization error.
        #[source]
        source: QuantizationError,
    },

    /// Subset target count exceeds the source snapshot count.
    #[error("subset target count {target} exceeds source count {source_count}")]
    SubsetTargetExceedsSource {
        /// Requested target count.
        target: u64,
        /// Available source count.
        source_count: u64,
    },

    /// Tile target count does not exceed the source snapshot count.
    #[error("tile target count {target} does not exceed source count {source_count}")]
    TileTargetBelowSource {
        /// Requested target count.
        target: u64,
        /// Available source count.
        source_count: u64,
    },

    /// Reading snapshot IDs file failed.
    #[error("failed to read snapshot ids from {path}")]
    ReadSnapshotIds {
        /// Path that failed to read.
        path: PathBuf,
        /// Underlying IO error.
        #[source]
        source: io::Error,
    },

    /// Parsing snapshot IDs file failed.
    #[error("failed to parse snapshot ids from {path}")]
    ParseSnapshotIds {
        /// Path that failed to parse.
        path: PathBuf,
        /// Underlying parse error.
        #[source]
        source: serde_json::Error,
    },

    /// Writing snapshot IDs file failed.
    #[error("failed to write snapshot ids to {path}")]
    WriteSnapshotIds {
        /// Path that failed to write.
        path: PathBuf,
        /// Underlying IO error.
        #[source]
        source: io::Error,
    },

    /// Serializing snapshot IDs failed.
    #[error("failed to serialize snapshot ids")]
    SerializeSnapshotIds {
        /// Underlying serialization error.
        #[source]
        source: serde_json::Error,
    },

    /// Snapshot IDs count does not match metadata count.
    #[error("snapshot ids count mismatch: expected {expected}, found {found}")]
    IdsCountMismatch {
        /// Expected count from metadata.
        expected: u64,
        /// Found count from IDs file.
        found: u64,
    },
}

#[cfg(test)]
#[derive(Debug, Deserialize)]
struct VersionProbe {
    version: u8,
}

/// Serialize metadata into the v2 metadata byte layout.
pub fn encode_metadata(meta: &VectorSnapshotMeta) -> SnapshotResult<Vec<u8>> {
    validate_meta(meta)?;
    if meta.version != VectorSnapshotVersion::V2 {
        return Err(SnapshotError::MetadataVersionMismatch {
            header_version: SNAPSHOT_VERSION_V2,
            metadata_version: meta.version.as_u8(),
        });
    }

    let json_bytes =
        serde_json::to_vec(meta).map_err(|source| SnapshotError::SerializeMetadata { source })?;
    let mut output = Vec::with_capacity(SNAPSHOT_V2_META_JSON_OFFSET + json_bytes.len());
    output.extend_from_slice(SNAPSHOT_V2_MAGIC_HEADER);
    output.extend_from_slice(&json_bytes);
    Ok(output)
}

/// Deserialize metadata from the v2 metadata byte layout.
pub fn decode_metadata(bytes: &[u8]) -> SnapshotResult<VectorSnapshotMeta> {
    let json_bytes = bytes
        .strip_prefix(SNAPSHOT_V2_MAGIC_HEADER)
        .ok_or(SnapshotError::InvalidMagicHeader)?;
    let meta: VectorSnapshotMeta = serde_json::from_slice(json_bytes)
        .map_err(|source| SnapshotError::DeserializeMetadata { source })?;
    if meta.version != VectorSnapshotVersion::V2 {
        return Err(SnapshotError::MetadataVersionMismatch {
            header_version: SNAPSHOT_VERSION_V2,
            metadata_version: meta.version.as_u8(),
        });
    }

    validate_meta(&meta)?;
    Ok(meta)
}

/// Write metadata bytes to disk at `path`.
pub fn write_metadata(path: impl AsRef<Path>, meta: &VectorSnapshotMeta) -> SnapshotResult<()> {
    let path = path.as_ref().to_path_buf();
    if let Some(parent) = path.parent().filter(|value| !value.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|source| SnapshotError::CreateMetadataDir {
            path: parent.to_path_buf(),
            source,
        })?;
    }

    let bytes = encode_metadata(meta)?;
    std::fs::write(&path, bytes).map_err(|source| SnapshotError::WriteMetadata { path, source })
}

/// Read metadata bytes from disk at `path`.
pub fn read_metadata(path: impl AsRef<Path>) -> SnapshotResult<VectorSnapshotMeta> {
    let path = path.as_ref().to_path_buf();
    let bytes = std::fs::read(&path).map_err(|source| SnapshotError::ReadMetadata {
        path: path.clone(),
        source,
    })?;
    decode_metadata(&bytes)
}

/// Detect snapshot version from metadata bytes.
///
/// Supports:
/// - v2 metadata files with `SNAPSHOT_V2_MAGIC_HEADER`
/// - legacy v1 JSON snapshots (without magic header)
#[cfg(test)]
fn detect_version(bytes: &[u8]) -> SnapshotResult<VectorSnapshotVersion> {
    if let Some(json_bytes) = bytes.strip_prefix(SNAPSHOT_V2_MAGIC_HEADER) {
        return parse_version_from_json(json_bytes);
    }

    parse_version_from_json(bytes)
}

/// Compute the expected byte length for the quantized vector payload.
pub fn expected_vector_byte_len(meta: &VectorSnapshotMeta) -> SnapshotResult<u64> {
    u64::from(meta.dimension).checked_mul(meta.count).ok_or(
        SnapshotError::ExpectedVectorLengthOverflow {
            count: meta.count,
            dimension: meta.dimension,
        },
    )
}

/// Compute CRC32 checksum for vector payload bytes.
#[must_use]
pub fn compute_vectors_crc32(bytes: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

/// Write a full v2 snapshot bundle for quantized (`u8`) vectors.
///
/// Writes:
/// - `snapshot.meta`
/// - `vectors.u8.bin`
///
/// The returned metadata contains persisted checksum and quantization params.
#[instrument(
    name = "vector.snapshot.write_v2",
    skip_all,
    fields(count = count, vector_bytes = vectors.len())
)]
pub fn write_snapshot_v2(
    snapshot_dir: impl AsRef<Path>,
    params: HnswParams,
    quantization: QuantizationParams,
    count: u64,
    vectors: &[u8],
) -> SnapshotResult<VectorSnapshotMeta> {
    write_snapshot_v2_with_kernel(
        snapshot_dir,
        params,
        VectorKernelKind::HnswRs,
        quantization,
        count,
        vectors,
    )
}

/// Write a full v2 snapshot bundle for quantized vectors with an explicit kernel.
pub fn write_snapshot_v2_with_kernel(
    snapshot_dir: impl AsRef<Path>,
    params: HnswParams,
    kernel: VectorKernelKind,
    quantization: QuantizationParams,
    count: u64,
    vectors: &[u8],
) -> SnapshotResult<VectorSnapshotMeta> {
    let snapshot_dir = snapshot_dir.as_ref().to_path_buf();
    std::fs::create_dir_all(&snapshot_dir).map_err(|source| SnapshotError::CreateSnapshotDir {
        path: snapshot_dir.clone(),
        source,
    })?;

    let dimension = u32::try_from(quantization.dimension()).map_err(|_| {
        SnapshotError::QuantizationDimensionOverflow {
            dimension: quantization.dimension(),
        }
    })?;

    let vectors_path = snapshot_dir.join(SNAPSHOT_V2_VECTORS_FILE_NAME);
    let vectors_crc32 = compute_vectors_crc32(vectors);
    let meta = VectorSnapshotMeta::new_with_kernel(
        dimension,
        count,
        params,
        kernel,
        quantization,
        vectors_crc32,
    )?;

    validate_vectors_len(&vectors_path, vectors, &meta)?;

    std::fs::write(&vectors_path, vectors).map_err(|source| SnapshotError::WriteVectors {
        path: vectors_path,
        source,
    })?;

    let meta_path = snapshot_dir.join(SNAPSHOT_V2_META_FILE_NAME);
    write_metadata(meta_path, &meta)?;
    Ok(meta)
}

/// Read a full v2 snapshot bundle (metadata + mapped vector bytes).
#[cfg(test)]
#[instrument(name = "vector.snapshot.read_v2", skip_all)]
pub fn read_snapshot_v2(snapshot_dir: impl AsRef<Path>) -> SnapshotResult<VectorSnapshotV2> {
    read_snapshot_v2_with_options(snapshot_dir, ReadSnapshotV2Options::default())
}

/// Read v2 snapshot bundle with optional auto-upgrade from legacy v1.
#[instrument(
    name = "vector.snapshot.read_v2_with_options",
    skip_all,
    fields(load_ms = tracing::field::Empty)
)]
pub fn read_snapshot_v2_with_options(
    snapshot_dir: impl AsRef<Path>,
    options: ReadSnapshotV2Options,
) -> SnapshotResult<VectorSnapshotV2> {
    let start = std::time::Instant::now();
    let snapshot_dir = snapshot_dir.as_ref().to_path_buf();
    let meta_path = snapshot_dir.join(SNAPSHOT_V2_META_FILE_NAME);
    let meta = match read_metadata(&meta_path) {
        Ok(meta) => meta,
        Err(error)
            if options.auto_upgrade_v1
                && matches!(
                    &error,
                    SnapshotError::ReadMetadata { path, source }
                        if path == &meta_path && source.kind() == io::ErrorKind::NotFound
                ) =>
        {
            upgrade_v1_to_v2(snapshot_dir.as_path())?
        },
        Err(error) => return Err(error),
    };

    let vectors_path = snapshot_dir.join(SNAPSHOT_V2_VECTORS_FILE_NAME);
    let expected_len = expected_vector_byte_len(&meta)?;
    let mapped = MmapBytes::open_readonly(&vectors_path, expected_len)
        .map_err(|source| map_mmap_error(source, vectors_path.clone()))?;

    let found = compute_vectors_crc32(mapped.as_slice());
    if found != meta.vectors_crc32 {
        return Err(SnapshotError::VectorsChecksumMismatch {
            path: vectors_path,
            expected: meta.vectors_crc32,
            found,
            algorithm: SNAPSHOT_V2_CHECKSUM_ALGORITHM,
        });
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "snapshot load time in ms will never overflow u64"
    )]
    let load_ms = start.elapsed().as_millis() as u64;
    tracing::Span::current().record("load_ms", load_ms);

    Ok(VectorSnapshotV2 {
        meta,
        vectors: mapped,
    })
}

/// Upgrade a `snapshot.v1.json` file in `snapshot_dir` into v2 metadata +
/// `vectors.u8.bin`.
pub fn upgrade_v1_to_v2(snapshot_dir: impl AsRef<Path>) -> SnapshotResult<VectorSnapshotMeta> {
    let snapshot_dir = snapshot_dir.as_ref();
    let v1_path = snapshot_dir.join(SNAPSHOT_V1_FILE_NAME);
    let payload = std::fs::read(&v1_path).map_err(|source| SnapshotError::ReadLegacySnapshot {
        path: v1_path.clone(),
        source,
    })?;
    let snapshot: VectorSnapshot = serde_json::from_slice(&payload).map_err(|source| {
        SnapshotError::DeserializeLegacySnapshot {
            path: v1_path,
            source,
        }
    })?;

    if snapshot.version != u32::from(SNAPSHOT_VERSION_V1) {
        return Err(SnapshotError::LegacyVersionMismatch {
            expected: u32::from(SNAPSHOT_VERSION_V1),
            found: snapshot.version,
        });
    }

    let expected_dimension = usize::try_from(snapshot.dimension).map_err(|_| {
        SnapshotError::DimensionConversionOverflow {
            dimension: snapshot.dimension,
        }
    })?;

    for (record_index, record) in snapshot.records.iter().enumerate() {
        if record.vector.len() != expected_dimension {
            return Err(SnapshotError::LegacyRecordDimensionMismatch {
                record_index,
                expected: expected_dimension,
                found: record.vector.len(),
            });
        }
    }

    let quantization = if snapshot.records.is_empty() {
        let scales = vec![1.0; expected_dimension];
        let zeros = vec![0.0; expected_dimension];
        QuantizationParams::new(scales, zeros)
            .map_err(|source| SnapshotError::Quantization { source })?
    } else {
        let dataset = snapshot
            .records
            .iter()
            .map(|record| record.vector.as_slice())
            .collect::<Vec<_>>();
        fit_min_max(dataset.as_slice()).map_err(|source| SnapshotError::Quantization { source })?
    };

    let quantizer = Quantizer::new(quantization.clone())
        .map_err(|source| SnapshotError::Quantization { source })?;

    let dataset = snapshot
        .records
        .iter()
        .map(|record| record.vector.as_slice())
        .collect::<Vec<_>>();
    let vectors = quantizer
        .quantize_batch(dataset.as_slice())
        .map_err(|source| SnapshotError::Quantization { source })?;

    let count = u64::try_from(snapshot.records.len()).map_err(|_| {
        SnapshotError::RecordCountConversionOverflow {
            count: snapshot.records.len(),
        }
    })?;

    write_snapshot_v2(
        snapshot_dir,
        snapshot.params,
        quantization,
        count,
        vectors.as_slice(),
    )
}

#[cfg(test)]
fn parse_version_from_json(json_bytes: &[u8]) -> SnapshotResult<VectorSnapshotVersion> {
    let version = serde_json::from_slice::<VersionProbe>(json_bytes)
        .map_err(|source| SnapshotError::DeserializeMetadata { source })?
        .version;
    VectorSnapshotVersion::try_from(version)
}

fn validate_meta(meta: &VectorSnapshotMeta) -> SnapshotResult<()> {
    if meta.dimension == 0 {
        return Err(SnapshotError::InvalidDimension);
    }

    let dimension = usize::try_from(meta.dimension).map_err(|_| {
        SnapshotError::DimensionConversionOverflow {
            dimension: meta.dimension,
        }
    })?;
    if meta.quantization.dimension() != dimension {
        return Err(SnapshotError::QuantizationDimensionMismatch {
            metadata_dimension: meta.dimension,
            quantization_dimension: meta.quantization.dimension(),
        });
    }

    Ok(())
}

fn validate_vectors_len(
    vectors_path: &Path,
    vectors: &[u8],
    meta: &VectorSnapshotMeta,
) -> SnapshotResult<()> {
    let expected = expected_vector_byte_len(meta)?;
    let found = u64::try_from(vectors.len())
        .map_err(|_| SnapshotError::VectorLengthConversionOverflow { len: vectors.len() })?;

    if found != expected {
        return Err(SnapshotError::VectorByteLengthMismatch {
            path: vectors_path.to_path_buf(),
            expected,
            found,
        });
    }
    Ok(())
}

fn map_mmap_error(source: MmapError, vectors_path: PathBuf) -> SnapshotError {
    match source {
        MmapError::SizeMismatch {
            expected, found, ..
        } => SnapshotError::VectorByteLengthMismatch {
            path: vectors_path,
            expected,
            found,
        },
        other => SnapshotError::MapVectors {
            path: vectors_path,
            source: other,
        },
    }
}

// ---------------------------------------------------------------------------
// Snapshot IDs helpers
// ---------------------------------------------------------------------------

const SNAPSHOT_V2_IDS_FILE_NAME: &str = "ids.json";

/// Read vector IDs from a snapshot directory's `ids.json` file.
pub fn read_snapshot_ids(snapshot_dir: &Path) -> SnapshotResult<Vec<Box<str>>> {
    let path = snapshot_dir.join(SNAPSHOT_V2_IDS_FILE_NAME);
    let bytes = std::fs::read(&path).map_err(|source| SnapshotError::ReadSnapshotIds {
        path: path.clone(),
        source,
    })?;
    serde_json::from_slice(&bytes)
        .map_err(|source| SnapshotError::ParseSnapshotIds { path, source })
}

/// Write vector IDs to a snapshot directory's `ids.json` file.
pub fn write_snapshot_ids(snapshot_dir: &Path, ids: &[Box<str>]) -> SnapshotResult<()> {
    std::fs::create_dir_all(snapshot_dir).map_err(|source| SnapshotError::CreateSnapshotDir {
        path: snapshot_dir.to_path_buf(),
        source,
    })?;
    let path = snapshot_dir.join(SNAPSHOT_V2_IDS_FILE_NAME);
    let bytes =
        serde_json::to_vec(ids).map_err(|source| SnapshotError::SerializeSnapshotIds { source })?;
    std::fs::write(&path, bytes).map_err(|source| SnapshotError::WriteSnapshotIds { path, source })
}

#[cfg(test)]
mod tests {
    use super::{
        ReadSnapshotV2Options, SNAPSHOT_V1_FILE_NAME, SNAPSHOT_V2_META_FILE_NAME,
        SNAPSHOT_V2_VECTORS_FILE_NAME, SNAPSHOT_VERSION_V2, SnapshotError, SnapshotResult,
        VectorSnapshotMeta, VectorSnapshotVersion, compute_vectors_crc32, decode_metadata,
        detect_version, encode_metadata, expected_vector_byte_len, read_snapshot_v2,
        read_snapshot_v2_with_options, upgrade_v1_to_v2, write_metadata, write_snapshot_v2,
    };
    use crate::quantization::QuantizationParams;
    use crate::{HnswParams, VectorRecord, VectorSnapshot};
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

    fn default_quantization(dimension: usize) -> SnapshotResult<QuantizationParams> {
        QuantizationParams::new(vec![1.0; dimension], vec![0.0; dimension])
            .map_err(|source| SnapshotError::Quantization { source })
    }

    #[test]
    fn version_detection_supports_v2_metadata_header() -> SnapshotResult<()> {
        let quantization = default_quantization(3)?;
        let meta = VectorSnapshotMeta::new(3, 2, HnswParams::default(), quantization, 0)?;
        let bytes = encode_metadata(&meta)?;
        assert_eq!(detect_version(&bytes)?, VectorSnapshotVersion::V2);
        Ok(())
    }

    #[test]
    fn version_detection_supports_legacy_json_snapshot() -> SnapshotResult<()> {
        let bytes = br#"{"version":1,"dimension":2,"params":{},"records":[]}"#;
        assert_eq!(detect_version(bytes)?, VectorSnapshotVersion::V1);
        Ok(())
    }

    #[test]
    fn metadata_roundtrip_is_stable() -> SnapshotResult<()> {
        let quantization = default_quantization(4)?;
        let meta = VectorSnapshotMeta::new(4, 3, HnswParams::default(), quantization, 123)?;
        let bytes = encode_metadata(&meta)?;
        let restored = decode_metadata(&bytes)?;

        assert_eq!(restored, meta);
        assert_eq!(expected_vector_byte_len(&restored)?, 12);
        Ok(())
    }

    #[test]
    fn decode_rejects_missing_magic_header() {
        let bytes = br#"{"version":2}"#;
        let error = decode_metadata(bytes).err();
        assert!(matches!(error, Some(SnapshotError::InvalidMagicHeader)));
    }

    #[test]
    fn decode_rejects_non_v2_version() {
        let bytes = br#"SCA-SNAPSHOT
{"version":1,"dimension":2,"count":1,"params":{"maxNbConnection":16,"maxLayer":16,"efConstruction":200,"efSearch":50,"maxElements":1},"quantization":{"scales":[1.0,1.0],"zeros":[0.0,0.0]},"vectorsCrc32":0}"#;
        let error = decode_metadata(bytes).err();
        assert!(matches!(
            error,
            Some(SnapshotError::MetadataVersionMismatch {
                header_version: SNAPSHOT_VERSION_V2,
                metadata_version: 1
            })
        ));
    }

    #[test]
    fn expected_vector_length_detects_overflow() -> SnapshotResult<()> {
        let quantization = default_quantization(2)?;
        let meta = VectorSnapshotMeta::new(2, u64::MAX, HnswParams::default(), quantization, 0)?;
        let error = expected_vector_byte_len(&meta).err();
        assert!(matches!(
            error,
            Some(SnapshotError::ExpectedVectorLengthOverflow { .. })
        ));
        Ok(())
    }

    #[test]
    fn write_read_snapshot_v2_roundtrip() -> SnapshotResult<()> {
        let temp = TempDir::create("vector-snapshot-v2-roundtrip").map_err(|source| {
            SnapshotError::CreateSnapshotDir {
                path: PathBuf::from("<test-temp-dir>"),
                source,
            }
        })?;
        let quantization = default_quantization(3)?;
        let vectors = vec![1u8, 2, 3, 4, 5, 6];

        let written = write_snapshot_v2(
            temp.path(),
            HnswParams::default(),
            quantization.clone(),
            2,
            vectors.as_slice(),
        )?;
        let loaded = read_snapshot_v2(temp.path())?;

        assert_eq!(loaded.meta, written);
        assert_eq!(loaded.vectors(), vectors.as_slice());
        let view = loaded.quantized_vectors()?;
        assert_eq!(view.len(), 2);
        assert_eq!(view.dimension(), 3);
        assert_eq!(view.vector(0), Some(&[1, 2, 3][..]));
        Ok(())
    }

    #[test]
    fn mmap_load_length_check_rejects_mismatch() -> SnapshotResult<()> {
        let temp = TempDir::create("vector-snapshot-v2-length-check").map_err(|source| {
            SnapshotError::CreateSnapshotDir {
                path: PathBuf::from("<test-temp-dir>"),
                source,
            }
        })?;
        let quantization = default_quantization(3)?;
        let wrong_vectors = vec![1u8, 2, 3, 4, 5];
        let vectors_path = temp.path().join(SNAPSHOT_V2_VECTORS_FILE_NAME);
        let meta = VectorSnapshotMeta::new(
            3,
            2,
            HnswParams::default(),
            quantization,
            compute_vectors_crc32(wrong_vectors.as_slice()),
        )?;

        write_metadata(temp.path().join(SNAPSHOT_V2_META_FILE_NAME), &meta)?;
        std::fs::write(&vectors_path, wrong_vectors.as_slice()).map_err(|source| {
            SnapshotError::WriteVectors {
                path: vectors_path.clone(),
                source,
            }
        })?;

        let error = read_snapshot_v2(temp.path()).err();
        assert!(matches!(
            error,
            Some(SnapshotError::VectorByteLengthMismatch {
                expected: 6,
                found: 5,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn checksum_mismatch_returns_typed_error() -> SnapshotResult<()> {
        let temp = TempDir::create("vector-snapshot-v2-checksum").map_err(|source| {
            SnapshotError::CreateSnapshotDir {
                path: PathBuf::from("<test-temp-dir>"),
                source,
            }
        })?;
        let quantization = default_quantization(2)?;
        let vectors = vec![10u8, 20, 30, 40];
        let _written = write_snapshot_v2(
            temp.path(),
            HnswParams::default(),
            quantization,
            2,
            vectors.as_slice(),
        )?;

        let vectors_path = temp.path().join(SNAPSHOT_V2_VECTORS_FILE_NAME);
        let mut corrupted =
            std::fs::read(&vectors_path).map_err(|source| SnapshotError::WriteVectors {
                path: vectors_path.clone(),
                source,
            })?;
        corrupted[0] = 0;
        std::fs::write(&vectors_path, corrupted).map_err(|source| SnapshotError::WriteVectors {
            path: vectors_path.clone(),
            source,
        })?;

        let error = read_snapshot_v2(temp.path()).err();
        assert!(matches!(
            error,
            Some(SnapshotError::VectorsChecksumMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn auto_upgrade_reads_v1_and_writes_v2_bundle() -> SnapshotResult<()> {
        let temp = TempDir::create("vector-snapshot-v1-upgrade").map_err(|source| {
            SnapshotError::CreateSnapshotDir {
                path: PathBuf::from("<test-temp-dir>"),
                source,
            }
        })?;

        let legacy = VectorSnapshot {
            version: 1,
            dimension: 2,
            params: HnswParams::default(),
            records: vec![
                VectorRecord {
                    id: "a".into(),
                    vector: vec![0.0, 1.0],
                },
                VectorRecord {
                    id: "b".into(),
                    vector: vec![2.0, 3.0],
                },
            ],
        };

        let legacy_path = temp.path().join(SNAPSHOT_V1_FILE_NAME);
        let payload = serde_json::to_vec(&legacy)
            .map_err(|source| SnapshotError::DeserializeMetadata { source })?;
        std::fs::write(&legacy_path, payload).map_err(|source| {
            SnapshotError::ReadLegacySnapshot {
                path: legacy_path,
                source,
            }
        })?;

        let loaded = read_snapshot_v2_with_options(
            temp.path(),
            ReadSnapshotV2Options {
                auto_upgrade_v1: true,
            },
        )?;

        assert_eq!(loaded.meta.dimension, 2);
        assert_eq!(loaded.meta.count, 2);
        assert_eq!(loaded.vectors().len(), 4);
        assert!(temp.path().join(SNAPSHOT_V2_META_FILE_NAME).is_file());
        assert!(temp.path().join(SNAPSHOT_V2_VECTORS_FILE_NAME).is_file());
        Ok(())
    }

    #[test]
    fn upgrade_v1_to_v2_returns_expected_crc() -> SnapshotResult<()> {
        let temp = TempDir::create("vector-snapshot-v1-upgrade-crc").map_err(|source| {
            SnapshotError::CreateSnapshotDir {
                path: PathBuf::from("<test-temp-dir>"),
                source,
            }
        })?;
        let legacy = VectorSnapshot {
            version: 1,
            dimension: 2,
            params: HnswParams::default(),
            records: vec![VectorRecord {
                id: "z".into(),
                vector: vec![5.0, -3.0],
            }],
        };
        let legacy_path = temp.path().join(SNAPSHOT_V1_FILE_NAME);
        let payload = serde_json::to_vec(&legacy)
            .map_err(|source| SnapshotError::DeserializeMetadata { source })?;
        std::fs::write(&legacy_path, payload).map_err(|source| {
            SnapshotError::ReadLegacySnapshot {
                path: legacy_path,
                source,
            }
        })?;

        let meta = upgrade_v1_to_v2(temp.path())?;
        let vectors =
            std::fs::read(temp.path().join(SNAPSHOT_V2_VECTORS_FILE_NAME)).map_err(|source| {
                SnapshotError::ReadLegacySnapshot {
                    path: temp.path().join(SNAPSHOT_V2_VECTORS_FILE_NAME),
                    source,
                }
            })?;
        let expected_crc = compute_vectors_crc32(vectors.as_slice());
        assert_eq!(meta.vectors_crc32, expected_crc);
        Ok(())
    }
}
