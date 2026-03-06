//! Scalar SQ8 quantization helpers.
//!
//! # Assumptions and limits
//! - Uses independent per-dimension min/max fitting (linear scalar quantization).
//! - Uses 8-bit unsigned storage (`u8`) with values clamped into `[0, 255]`.
//! - Out-of-range runtime values are clamped to the nearest representable bin.
//! - No outlier handling, no residual coding, and no learned calibration.
//! - Accuracy depends on fit data quality; poor calibration data will increase error.
//! - Designed as a simple, deterministic foundation for later SIMD/parallel paths.

use serde::{Deserialize, Serialize};
use thiserror::Error;

const U8_MAX_F32: f32 = 255.0;

/// Result type for quantization operations.
pub type QuantizationResult<T> = Result<T, QuantizationError>;

/// Typed quantization errors.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum QuantizationError {
    /// Dataset is empty.
    #[error("dataset must contain at least one vector")]
    EmptyDataset,
    /// Dimension is zero.
    #[error("dimension must be greater than zero")]
    ZeroDimension,
    /// Single vector dimension mismatch.
    #[error("vector dimension mismatch: expected {expected}, found {found}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Found dimension.
        found: usize,
    },
    /// Dataset vector dimension mismatch.
    #[error("dataset vector #{vector_index} has dimension {found}, expected {expected}")]
    DatasetDimensionMismatch {
        /// Vector position in dataset.
        vector_index: usize,
        /// Expected dimension.
        expected: usize,
        /// Found dimension.
        found: usize,
    },
    /// Scale/zero length mismatch.
    #[error("parameter length mismatch: scales={scales_len}, zeros={zeros_len}")]
    ParameterLengthMismatch {
        /// Number of scale entries.
        scales_len: usize,
        /// Number of zero entries.
        zeros_len: usize,
    },
    /// Invalid scale value.
    #[error("scale at dim {dimension} must be finite and > 0, found {value}")]
    InvalidScale {
        /// Dimension index.
        dimension: usize,
        /// Invalid value.
        value: f32,
    },
    /// Invalid zero-point value.
    #[error("zero at dim {dimension} must be finite, found {value}")]
    InvalidZero {
        /// Dimension index.
        dimension: usize,
        /// Invalid value.
        value: f32,
    },
    /// Non-finite value in single-vector input.
    #[error("input value at dim {dimension} must be finite, found {value}")]
    NonFiniteInput {
        /// Dimension index.
        dimension: usize,
        /// Invalid value.
        value: f32,
    },
    /// Non-finite value in dataset input.
    #[error(
        "dataset value at vector #{vector_index}, dim {dimension} must be finite, found {value}"
    )]
    NonFiniteDatasetInput {
        /// Vector position in dataset.
        vector_index: usize,
        /// Dimension index.
        dimension: usize,
        /// Invalid value.
        value: f32,
    },
    /// Flattened byte slice length mismatch.
    #[error("quantized byte length {len} is not divisible by dimension {dimension}")]
    QuantizedLengthMismatch {
        /// Byte length.
        len: usize,
        /// Expected vector dimension.
        dimension: usize,
    },
    /// Capacity overflow when preallocating buffers.
    #[error(
        "allocation overflow while preallocating for {vectors} vectors of dimension {dimension}"
    )]
    AllocationOverflow {
        /// Vector dimension.
        dimension: usize,
        /// Vector count.
        vectors: usize,
    },
}

/// Per-dimension SQ8 quantization parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuantizationParams {
    scales: Vec<f32>,
    zeros: Vec<f32>,
}

impl QuantizationParams {
    /// Build validated parameters.
    pub fn new(scales: Vec<f32>, zeros: Vec<f32>) -> QuantizationResult<Self> {
        if scales.len() != zeros.len() {
            return Err(QuantizationError::ParameterLengthMismatch {
                scales_len: scales.len(),
                zeros_len: zeros.len(),
            });
        }
        if scales.is_empty() {
            return Err(QuantizationError::ZeroDimension);
        }

        for (dimension, scale) in scales.iter().copied().enumerate() {
            if !scale.is_finite() || scale <= 0.0 {
                return Err(QuantizationError::InvalidScale {
                    dimension,
                    value: scale,
                });
            }
        }

        for (dimension, zero) in zeros.iter().copied().enumerate() {
            if !zero.is_finite() {
                return Err(QuantizationError::InvalidZero {
                    dimension,
                    value: zero,
                });
            }
        }

        Ok(Self { scales, zeros })
    }

    /// Number of dimensions covered by params.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.scales.len()
    }

    /// Per-dimension scales.
    #[must_use]
    pub const fn scales(&self) -> &[f32] {
        self.scales.as_slice()
    }

    /// Per-dimension zero points (stored as `f32` offsets).
    #[must_use]
    pub const fn zeros(&self) -> &[f32] {
        self.zeros.as_slice()
    }
}

/// Fit per-dimension min/max bounds from a dataset.
pub fn fit_min_max<T>(dataset: &[T]) -> QuantizationResult<QuantizationParams>
where
    T: AsRef<[f32]>,
{
    let first = dataset
        .first()
        .ok_or(QuantizationError::EmptyDataset)?
        .as_ref();
    if first.is_empty() {
        return Err(QuantizationError::ZeroDimension);
    }

    let expected = first.len();
    let mut mins = Vec::with_capacity(expected);
    let mut maxs = Vec::with_capacity(expected);

    for (dimension, value) in first.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(QuantizationError::NonFiniteDatasetInput {
                vector_index: 0,
                dimension,
                value,
            });
        }
        mins.push(value);
        maxs.push(value);
    }

    for (vector_index, vector) in dataset.iter().enumerate().skip(1) {
        let values = vector.as_ref();
        if values.len() != expected {
            return Err(QuantizationError::DatasetDimensionMismatch {
                vector_index,
                expected,
                found: values.len(),
            });
        }

        for (((min, max), value), dimension) in mins
            .iter_mut()
            .zip(maxs.iter_mut())
            .zip(values.iter().copied())
            .zip(0usize..)
        {
            if !value.is_finite() {
                return Err(QuantizationError::NonFiniteDatasetInput {
                    vector_index,
                    dimension,
                    value,
                });
            }
            if value < *min {
                *min = value;
            }
            if value > *max {
                *max = value;
            }
        }
    }

    let mut scales = Vec::with_capacity(expected);
    let mut zeros = Vec::with_capacity(expected);
    for (min, max) in mins.into_iter().zip(maxs) {
        let range = max - min;
        let scale = if range <= f32::EPSILON {
            1.0
        } else {
            range / U8_MAX_F32
        };
        scales.push(scale);
        zeros.push(min);
    }

    QuantizationParams::new(scales, zeros)
}

/// Quantize one floating-point vector into SQ8 bytes with clamping.
pub fn quantize_f32_to_u8(
    values: &[f32],
    params: &QuantizationParams,
) -> QuantizationResult<Vec<u8>> {
    ensure_single_dimension(values.len(), params.dimension())?;

    let mut output = Vec::with_capacity(values.len());
    for (dimension, (value, (scale, zero))) in values
        .iter()
        .copied()
        .zip(
            params
                .scales()
                .iter()
                .copied()
                .zip(params.zeros().iter().copied()),
        )
        .enumerate()
    {
        if !value.is_finite() {
            return Err(QuantizationError::NonFiniteInput { dimension, value });
        }
        output.push(quantize_scalar(value, scale, zero));
    }
    Ok(output)
}

/// Dequantize one SQ8 vector back to floating-point values.
pub fn dequantize_u8_to_f32(
    quantized: &[u8],
    params: &QuantizationParams,
) -> QuantizationResult<Vec<f32>> {
    ensure_single_dimension(quantized.len(), params.dimension())?;

    let mut output = Vec::with_capacity(quantized.len());
    for (value, (scale, zero)) in quantized.iter().copied().zip(
        params
            .scales()
            .iter()
            .copied()
            .zip(params.zeros().iter().copied()),
    ) {
        output.push(decode_u8_to_f32(value).mul_add(scale, zero));
    }
    Ok(output)
}

/// Safely decode one quantized byte into an `f32` component.
#[must_use]
pub const fn decode_u8_to_f32(value: u8) -> f32 {
    value as f32
}

/// Batch helper for quantization/dequantization.
#[derive(Debug, Clone, PartialEq)]
pub struct Quantizer {
    params: QuantizationParams,
}

impl Quantizer {
    /// Build a quantizer from validated parameters.
    pub fn new(params: QuantizationParams) -> QuantizationResult<Self> {
        QuantizationParams::new(params.scales.clone(), params.zeros.clone())?;
        Ok(Self { params })
    }

    /// Fit params from dataset and build quantizer.
    pub fn from_dataset<T>(dataset: &[T]) -> QuantizationResult<Self>
    where
        T: AsRef<[f32]>,
    {
        let params = fit_min_max(dataset)?;
        Ok(Self { params })
    }

    /// Access fitted parameters.
    #[must_use]
    pub const fn params(&self) -> &QuantizationParams {
        &self.params
    }

    /// Quantize a single vector.
    pub fn quantize(&self, values: &[f32]) -> QuantizationResult<Vec<u8>> {
        quantize_f32_to_u8(values, &self.params)
    }

    /// Dequantize a single SQ8 vector.
    pub fn dequantize(&self, quantized: &[u8]) -> QuantizationResult<Vec<f32>> {
        dequantize_u8_to_f32(quantized, &self.params)
    }

    /// Quantize a dataset and return flattened bytes.
    pub fn quantize_batch<T>(&self, dataset: &[T]) -> QuantizationResult<Vec<u8>>
    where
        T: AsRef<[f32]>,
    {
        let dimension = self.params.dimension();
        let vectors = dataset.len();
        let total = vectors
            .checked_mul(dimension)
            .ok_or(QuantizationError::AllocationOverflow { dimension, vectors })?;
        let mut output = Vec::with_capacity(total);

        for (vector_index, vector) in dataset.iter().enumerate() {
            let values = vector.as_ref();
            if values.len() != dimension {
                return Err(QuantizationError::DatasetDimensionMismatch {
                    vector_index,
                    expected: dimension,
                    found: values.len(),
                });
            }

            for (dim, (value, (scale, zero))) in values
                .iter()
                .copied()
                .zip(
                    self.params
                        .scales()
                        .iter()
                        .copied()
                        .zip(self.params.zeros().iter().copied()),
                )
                .enumerate()
            {
                if !value.is_finite() {
                    return Err(QuantizationError::NonFiniteDatasetInput {
                        vector_index,
                        dimension: dim,
                        value,
                    });
                }
                output.push(quantize_scalar(value, scale, zero));
            }
        }

        Ok(output)
    }
}

/// Read-only view over flattened quantized bytes.
#[derive(Debug, Clone, Copy)]
pub struct QuantizedSlice<'a> {
    bytes: &'a [u8],
    dimension: usize,
}

impl<'a> QuantizedSlice<'a> {
    /// Build a validated quantized view.
    pub const fn new(bytes: &'a [u8], dimension: usize) -> QuantizationResult<Self> {
        if dimension == 0 {
            return Err(QuantizationError::ZeroDimension);
        }
        if !bytes.len().is_multiple_of(dimension) {
            return Err(QuantizationError::QuantizedLengthMismatch {
                len: bytes.len(),
                dimension,
            });
        }
        Ok(Self { bytes, dimension })
    }

    /// Number of vectors in this view.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.bytes.len() / self.dimension
    }

    /// Vector dimension.
    #[must_use]
    #[cfg(test)]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Borrow one quantized vector by index.
    #[must_use]
    #[cfg(test)]
    pub fn vector(&self, index: usize) -> Option<&'a [u8]> {
        let start = index.checked_mul(self.dimension)?;
        let end = start.checked_add(self.dimension)?;
        self.bytes.get(start..end)
    }

    /// Iterate over vector chunks.
    pub fn iter(&self) -> impl Iterator<Item = &'a [u8]> {
        self.bytes.chunks_exact(self.dimension)
    }
}

const fn ensure_single_dimension(found: usize, expected: usize) -> QuantizationResult<()> {
    if found != expected {
        return Err(QuantizationError::DimensionMismatch { expected, found });
    }
    Ok(())
}

fn quantize_scalar(value: f32, scale: f32, zero: f32) -> u8 {
    let normalized = ((value - zero) / scale).clamp(0.0, U8_MAX_F32);
    round_to_u8(normalized)
}

fn round_to_u8(value: f32) -> u8 {
    for candidate in 0u8..u8::MAX {
        let threshold = f32::from(candidate) + 0.5;
        if value < threshold {
            return candidate;
        }
    }
    u8::MAX
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseError;

    #[test]
    fn invalid_params_rejected() {
        let empty = QuantizationParams::new(Vec::new(), Vec::new());
        assert!(matches!(empty, Err(QuantizationError::ZeroDimension)));

        let mismatch = QuantizationParams::new(vec![1.0], Vec::new());
        assert!(matches!(
            mismatch,
            Err(QuantizationError::ParameterLengthMismatch { .. })
        ));

        let invalid_scale = QuantizationParams::new(vec![0.0], vec![0.0]);
        assert!(matches!(
            invalid_scale,
            Err(QuantizationError::InvalidScale { .. })
        ));
    }

    #[test]
    fn quantization_is_deterministic_for_fixed_data() -> QuantizationResult<()> {
        let dataset = vec![
            vec![-1.0, 0.0, 1.0],
            vec![0.5, 0.25, -0.25],
            vec![10.0, -3.0, 7.25],
        ];
        let quantizer = Quantizer::from_dataset(&dataset)?;

        let first = quantizer.quantize_batch(&dataset)?;
        let second = quantizer.quantize_batch(&dataset)?;
        assert_eq!(first, second);
        Ok(())
    }

    #[test]
    fn quantized_slice_validates_alignment() {
        let bytes = vec![1u8, 2, 3];
        let result = QuantizedSlice::new(bytes.as_slice(), 2);
        assert!(matches!(
            result,
            Err(QuantizationError::QuantizedLengthMismatch { .. })
        ));
    }

    proptest! {
        #[test]
        fn roundtrip_is_within_sq8_bounds(
            rows in proptest::collection::vec(
                (
                    -500.0f32..500.0f32,
                    -500.0f32..500.0f32,
                    -500.0f32..500.0f32,
                    -500.0f32..500.0f32,
                ),
                1usize..32usize
            )
        ) {
            let dataset = rows
                .into_iter()
                .map(|(a, b, c, d)| vec![a, b, c, d])
                .collect::<Vec<Vec<f32>>>();

            let quantizer = Quantizer::from_dataset(&dataset)
                .map_err(|_| TestCaseError::fail("fit_min_max failed"))?;
            let quantized = quantizer
                .quantize_batch(&dataset)
                .map_err(|_| TestCaseError::fail("quantize_batch failed"))?;
            let view = QuantizedSlice::new(quantized.as_slice(), quantizer.params().dimension())
                .map_err(|_| TestCaseError::fail("QuantizedSlice::new failed"))?;

            let params = quantizer.params();
            let lower = params.zeros().to_vec();
            let upper = params
                .zeros()
                .iter()
                .copied()
                .zip(params.scales().iter().copied())
                .map(|(zero, scale)| U8_MAX_F32.mul_add(scale, zero))
                .collect::<Vec<f32>>();

            for vector in view.iter() {
                let dequantized = quantizer
                    .dequantize(vector)
                    .map_err(|_| TestCaseError::fail("dequantize failed"))?;

                for ((value, min), max) in dequantized
                    .into_iter()
                    .zip(lower.iter().copied())
                    .zip(upper.iter().copied())
                {
                    prop_assert!(value >= min - 1e-4);
                    prop_assert!(value <= max + 1e-4);
                }
            }
        }
    }
}
