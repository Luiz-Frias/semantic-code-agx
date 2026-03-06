//! Integration tests for SQ8 quantization primitives.

use semantic_code_vector::{QuantizationError, Quantizer};

#[test]
fn quantize_dequantize_dataset_roundtrip() -> Result<(), QuantizationError> {
    let dataset = vec![
        vec![0.0, -1.0, 4.0, 10.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![-4.0, 5.0, 8.0, -2.0],
        vec![7.5, -3.25, 0.75, 9.0],
    ];

    let quantizer = Quantizer::from_dataset(&dataset)?;
    let params = quantizer.params().clone();

    let quantized = quantizer.quantize_batch(&dataset)?;
    let dimension = params.dimension();
    assert_eq!(quantized.len(), dataset.len() * dimension);
    let vectors = quantized.chunks_exact(dimension);
    assert!(vectors.remainder().is_empty());
    assert_eq!(vectors.len(), dataset.len());

    for (original, encoded) in dataset.iter().zip(vectors) {
        let restored = quantizer.dequantize(encoded)?;
        assert_eq!(restored.len(), original.len());

        for (((source, recovered), zero), scale) in original
            .iter()
            .copied()
            .zip(restored.iter().copied())
            .zip(params.zeros().iter().copied())
            .zip(params.scales().iter().copied())
        {
            let upper = f32::from(u8::MAX).mul_add(scale, zero);
            assert!(recovered >= zero - 1e-4);
            assert!(recovered <= upper + 1e-4);

            // Values inside calibration bounds quantize with <= 0.5 step error.
            let step = scale * 0.5 + 1e-4;
            assert!((source - recovered).abs() <= step);
        }
    }

    Ok(())
}
