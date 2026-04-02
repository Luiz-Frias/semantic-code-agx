//! Exact row models and lending readers for rebuild-grade collection state.
//!
//! These types provide a kernel-neutral view over the active vector payload in
//! canonical origin order. The owned container (`ExactVectorRows`) is intended
//! to back staged and published collection generations, while the lending trait
//! (`ExactVectorRowSource`) lets hot internal code read rows without forcing
//! clones at every boundary.

use crate::{ErrorClass, ErrorCode, ErrorEnvelope, OriginId, Result};

const EXACT_VECTOR_ROWS_HASH_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const EXACT_VECTOR_ROWS_HASH_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Borrowed exact row view in canonical origin order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExactVectorRowRef<'a> {
    origin: OriginId,
    id: &'a str,
    vector: &'a [f32],
}

impl<'a> ExactVectorRowRef<'a> {
    /// Build a borrowed row view from canonical row components.
    #[must_use]
    pub const fn new(origin: OriginId, id: &'a str, vector: &'a [f32]) -> Self {
        Self { origin, id, vector }
    }
}

/// Shared behavior for exact row views.
pub trait ExactVectorRowView {
    /// Durable origin identifier in canonical origin order.
    fn origin(&self) -> OriginId;

    /// Stable external identifier.
    fn id(&self) -> &str;

    /// Exact rebuild-grade vector payload.
    fn vector(&self) -> &[f32];
}

impl ExactVectorRowView for ExactVectorRowRef<'_> {
    fn origin(&self) -> OriginId {
        self.origin
    }

    fn id(&self) -> &str {
        self.id
    }

    fn vector(&self) -> &[f32] {
        self.vector
    }
}

/// Owned exact row stored in canonical origin order.
#[derive(Debug, Clone, PartialEq)]
pub struct ExactVectorRow {
    origin: OriginId,
    id: Box<str>,
    vector: Vec<f32>,
}

impl ExactVectorRow {
    /// Build an owned exact row.
    #[must_use]
    pub fn new(id: impl Into<Box<str>>, origin: OriginId, vector: Vec<f32>) -> Self {
        Self {
            origin,
            id: id.into(),
            vector,
        }
    }

    /// Borrow this row as a lending view.
    #[must_use]
    pub fn as_ref(&self) -> ExactVectorRowRef<'_> {
        ExactVectorRowRef::new(self.origin, self.id.as_ref(), self.vector.as_slice())
    }
}

impl ExactVectorRowView for ExactVectorRow {
    fn origin(&self) -> OriginId {
        self.origin
    }

    fn id(&self) -> &str {
        self.id.as_ref()
    }

    fn vector(&self) -> &[f32] {
        self.vector.as_slice()
    }
}

/// Owned exact rows ready for staging or publication.
#[derive(Debug, Clone, PartialEq)]
pub struct ExactVectorRows {
    dimension: u32,
    rows: Vec<ExactVectorRow>,
}

impl ExactVectorRows {
    /// Build an owned exact-row set and validate canonical invariants.
    pub fn new(dimension: u32, rows: Vec<ExactVectorRow>) -> Result<Self> {
        if dimension == 0 {
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("vector", "invalid_dimension"),
                "exact row dimension must be greater than zero",
            ));
        }

        let expected_dimension = usize::try_from(dimension).map_err(|_| {
            ErrorEnvelope::unexpected(
                ErrorCode::new("vector", "exact_rows_dimension_overflow"),
                "exact row dimension conversion overflow",
                ErrorClass::NonRetriable,
            )
            .with_metadata("dimension", dimension.to_string())
        })?;

        let mut previous_origin: Option<OriginId> = None;
        for (row_index, row) in rows.iter().enumerate() {
            let found_dimension = row.vector().len();
            if found_dimension != expected_dimension {
                return Err(ErrorEnvelope::expected(
                    ErrorCode::new("vector", "exact_rows_dimension_mismatch"),
                    "exact row vector dimension mismatch",
                )
                .with_metadata("rowIndex", row_index.to_string())
                .with_metadata("id", row.id().to_string())
                .with_metadata("expected", expected_dimension.to_string())
                .with_metadata("found", found_dimension.to_string()));
            }

            if let Some(previous_origin) = previous_origin
                && previous_origin.as_usize() >= row.origin().as_usize()
            {
                return Err(ErrorEnvelope::expected(
                    ErrorCode::new("vector", "exact_rows_origin_order_invalid"),
                    "exact rows must be in strictly increasing origin order",
                )
                .with_metadata("rowIndex", row_index.to_string())
                .with_metadata("previousOrigin", previous_origin.as_usize().to_string())
                .with_metadata("currentOrigin", row.origin().as_usize().to_string()));
            }

            previous_origin = Some(row.origin());
        }

        Ok(Self { dimension, rows })
    }

    /// Vector dimension shared by all rows.
    #[must_use]
    pub const fn dimension(&self) -> u32 {
        self.dimension
    }

    /// Number of exact rows.
    #[must_use]
    pub const fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Deterministic fingerprint of this exact row set.
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        fingerprint_exact_rows(self.dimension, self.rows.len(), self.rows())
    }
}

/// Lending iterator over [`ExactVectorRows`].
pub struct ExactVectorRowsIter<'a> {
    inner: std::slice::Iter<'a, ExactVectorRow>,
}

impl<'a> Iterator for ExactVectorRowsIter<'a> {
    type Item = ExactVectorRowRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(ExactVectorRow::as_ref)
    }
}

/// Lending row-source contract for exact rebuild-grade collection rows.
pub trait ExactVectorRowSource {
    /// Borrowed row type produced by this source.
    type Row<'a>: ExactVectorRowView
    where
        Self: 'a;

    /// Iterator type over canonical rows.
    type Iter<'a>: Iterator<Item = Self::Row<'a>> + 'a
    where
        Self: 'a;

    /// Shared row dimension.
    fn dimension(&self) -> u32;

    /// Row count in canonical origin order.
    fn row_count(&self) -> usize;

    /// Borrow canonical rows in strictly increasing origin order.
    fn rows(&self) -> Self::Iter<'_>;

    /// Deterministic fingerprint over canonical rows.
    fn fingerprint(&self) -> u64
    where
        for<'a> Self::Row<'a>: ExactVectorRowView,
    {
        fingerprint_exact_rows(self.dimension(), self.row_count(), self.rows())
    }
}

impl ExactVectorRowSource for ExactVectorRows {
    type Row<'a>
        = ExactVectorRowRef<'a>
    where
        Self: 'a;

    type Iter<'a>
        = ExactVectorRowsIter<'a>
    where
        Self: 'a;

    fn dimension(&self) -> u32 {
        self.dimension
    }

    fn row_count(&self) -> usize {
        self.rows.len()
    }

    fn rows(&self) -> Self::Iter<'_> {
        ExactVectorRowsIter {
            inner: self.rows.iter(),
        }
    }
}

/// Compute the stable fingerprint for exact canonical rows.
#[must_use]
pub fn fingerprint_exact_rows<R>(
    dimension: u32,
    row_count: usize,
    rows: impl IntoIterator<Item = R>,
) -> u64
where
    R: ExactVectorRowView,
{
    let mut hash = EXACT_VECTOR_ROWS_HASH_OFFSET_BASIS;
    update_hash(&mut hash, &dimension.to_le_bytes());
    update_hash(&mut hash, &row_count.to_le_bytes());
    for row in rows {
        update_hash(&mut hash, &row.origin().as_usize().to_le_bytes());
        update_hash(&mut hash, &row.id().len().to_le_bytes());
        update_hash(&mut hash, row.id().as_bytes());
        update_hash(&mut hash, &row.vector().len().to_le_bytes());
        for value in row.vector() {
            update_hash(&mut hash, &value.to_le_bytes());
        }
    }
    hash
}

fn update_hash(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(EXACT_VECTOR_ROWS_HASH_PRIME);
    }
}
