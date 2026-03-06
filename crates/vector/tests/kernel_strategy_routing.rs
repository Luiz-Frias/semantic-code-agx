//! Integration test: all `VectorSearchBackend` variants route through the kernel abstraction.

use semantic_code_vector::{
    FlatScanKernel, HnswKernel, HnswParams, VectorIndex, VectorKernel, VectorRecord,
    VectorSearchBackend,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[allow(
    clippy::cast_precision_loss,
    reason = "test seed values are small enough for f32"
)]
fn make_vector(seed: u32, dimension: usize) -> Vec<f32> {
    let values = (0..dimension)
        .map(|offset| {
            let base = seed as f32;
            let axis = offset as f32;
            let sine = (base.mul_add(0.17, axis * 0.11)).sin() * 0.75;
            let cosine = (base.mul_add(0.07, axis * 0.19)).cos() * 0.25;
            sine + cosine + 0.05
        })
        .collect::<Vec<_>>();
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= 0.0 {
        return values;
    }
    values.into_iter().map(|value| value / norm).collect()
}

fn build_index(dimension: usize, count: u32) -> Result<VectorIndex, Box<dyn std::error::Error>> {
    let dim = u32::try_from(dimension)?;
    let mut index = VectorIndex::new(dim, HnswParams::default())?;
    let records = (0..count)
        .map(|seed| VectorRecord {
            id: format!("doc_{seed:04}").into_boxed_str(),
            vector: make_vector(seed, dimension),
        })
        .collect::<Vec<_>>();
    index.insert(records)?;
    Ok(index)
}

#[test]
fn hnsw_kernel_f32_backend_returns_results() -> TestResult {
    let index = build_index(24, 64)?;
    let query = make_vector(9_999, 24);
    let results = index.search_with_kernel(
        query.as_slice(),
        5,
        &HnswKernel::new(),
        VectorSearchBackend::F32Hnsw,
    )?;
    assert_eq!(results.matches.len(), 5);
    Ok(())
}

#[test]
fn hnsw_kernel_u8_rerank_backend_returns_results() -> TestResult {
    let index = build_index(24, 64)?;
    let query = make_vector(9_999, 24);
    let results = index.search_with_kernel(
        query.as_slice(),
        5,
        &HnswKernel::new(),
        VectorSearchBackend::ExperimentalU8ThenF32Rerank,
    )?;
    assert_eq!(results.matches.len(), 5);
    Ok(())
}

#[test]
fn search_with_backend_delegates_to_hnsw_kernel() -> TestResult {
    let index = build_index(24, 64)?;
    let query = make_vector(9_999, 24);

    let via_backend =
        index.search_with_backend(query.as_slice(), 5, VectorSearchBackend::F32Hnsw)?;
    let via_kernel = index.search_with_kernel(
        query.as_slice(),
        5,
        &HnswKernel::new(),
        VectorSearchBackend::F32Hnsw,
    )?;
    assert_eq!(via_backend.matches, via_kernel.matches);
    Ok(())
}

#[test]
fn all_backend_variants_produce_nonempty_results() -> TestResult {
    let index = build_index(24, 64)?;
    let query = make_vector(9_999, 24);

    let backends = [
        VectorSearchBackend::F32Hnsw,
        VectorSearchBackend::ExperimentalU8Quantized,
        VectorSearchBackend::ExperimentalU8ThenF32Rerank,
    ];
    for backend in backends {
        let results = index.search_with_kernel(query.as_slice(), 5, &HnswKernel::new(), backend)?;
        assert!(
            !results.matches.is_empty(),
            "expected non-empty results for backend {backend:?}"
        );
    }
    Ok(())
}

#[test]
fn trait_object_dispatch_matches_concrete_dispatch() -> TestResult {
    let index = build_index(24, 64)?;
    let query = make_vector(9_999, 24);

    let kernel: &dyn VectorKernel = &HnswKernel::new();
    let via_trait_object =
        index.search_with_kernel(query.as_slice(), 5, kernel, VectorSearchBackend::F32Hnsw)?;
    let via_concrete = index.search_with_kernel(
        query.as_slice(),
        5,
        &HnswKernel::new(),
        VectorSearchBackend::F32Hnsw,
    )?;
    assert_eq!(via_trait_object.matches, via_concrete.matches);
    Ok(())
}

#[test]
fn flat_scan_returns_exact_top_k() -> TestResult {
    let index = build_index(24, 64)?;
    let query = make_vector(9_999, 24);
    let results = index.search_with_kernel(
        query.as_slice(),
        5,
        &FlatScanKernel,
        VectorSearchBackend::F32Hnsw,
    )?;
    assert_eq!(results.matches.len(), 5);
    // Verify results are sorted by score descending
    #[allow(
        clippy::indexing_slicing,
        reason = ".windows(2) guarantees slices of exactly length 2"
    )]
    for window in results.matches.windows(2) {
        assert!(
            window[0].score >= window[1].score,
            "flat scan results should be sorted by score descending"
        );
    }
    Ok(())
}

#[test]
fn flat_scan_superset_of_hnsw_on_small_index() -> TestResult {
    let index = build_index(24, 64)?;
    let query = make_vector(9_999, 24);
    let hnsw_results = index.search_with_kernel(
        query.as_slice(),
        10,
        &HnswKernel::new(),
        VectorSearchBackend::F32Hnsw,
    )?;
    // Use a larger k for flat scan to give headroom for HNSW's approximation.
    // HNSW is approximate — it may return a result ranked 11th-nearest while
    // missing a true top-10 neighbor, so flat scan needs a wider window.
    let flat_results = index.search_with_kernel(
        query.as_slice(),
        20,
        &FlatScanKernel,
        VectorSearchBackend::F32Hnsw,
    )?;

    let flat_ids: std::collections::HashSet<_> =
        flat_results.matches.iter().map(|m| &m.id).collect();
    for hnsw_match in &hnsw_results.matches {
        assert!(
            flat_ids.contains(&hnsw_match.id),
            "HNSW result {:?} should be in flat scan top-20 (N=64)",
            hnsw_match.id
        );
    }
    Ok(())
}

#[test]
#[allow(
    clippy::unnecessary_wraps,
    reason = "consistent TestResult return type across all tests in this module"
)]
fn flat_scan_kernel_kind_is_flat_scan() -> TestResult {
    let kernel = FlatScanKernel;
    assert_eq!(
        kernel.kind(),
        semantic_code_vector::VectorKernelKind::FlatScan
    );
    Ok(())
}
