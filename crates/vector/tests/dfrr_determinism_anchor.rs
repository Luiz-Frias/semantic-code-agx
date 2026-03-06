//! Determinism anchor tests for vector search kernels.
//!
//! Verifies that:
//! 1. Repeated kernel calls on identical data produce identical results.
//! 2. HNSW results are a subset of flat-scan exact results.
//! 3. `FlatScan` is fully deterministic (the ground-truth baseline).
//!
//! These tests anchor the production pipeline's faithfulness: if a kernel
//! produces non-deterministic results, recall comparisons are meaningless.

use std::collections::HashSet;

use semantic_code_vector::{
    FlatScanKernel, HnswKernel, HnswParams, VectorIndex, VectorMatch, VectorRecord,
    VectorSearchBackend,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Number of determinism iterations — enough to catch stochastic failures.
const DETERMINISM_ITERATIONS: usize = 10;

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

fn match_ids(matches: &[VectorMatch]) -> Vec<&str> {
    matches.iter().map(|m| m.id.as_ref()).collect()
}

fn match_id_set(matches: &[VectorMatch]) -> HashSet<&str> {
    matches.iter().map(|m| m.id.as_ref()).collect()
}

// ── Flat scan determinism ────────────────────────────────────────────────────

#[test]
fn flat_scan_is_fully_deterministic() -> TestResult {
    let index = build_index(24, 128)?;
    let query = make_vector(9_999, 24);

    let reference = index.search_with_kernel(
        query.as_slice(),
        10,
        &FlatScanKernel,
        VectorSearchBackend::F32Hnsw,
    )?;

    for iteration in 0..DETERMINISM_ITERATIONS {
        let result = index.search_with_kernel(
            query.as_slice(),
            10,
            &FlatScanKernel,
            VectorSearchBackend::F32Hnsw,
        )?;
        assert_eq!(
            match_ids(&result.matches),
            match_ids(&reference.matches),
            "flat scan iteration {iteration} produced different result IDs"
        );
        // Also verify scores are bit-identical.
        for (a, b) in result.matches.iter().zip(reference.matches.iter()) {
            assert_eq!(
                a.score.to_bits(),
                b.score.to_bits(),
                "flat scan iteration {iteration}: score mismatch for id={}",
                a.id
            );
        }
    }
    Ok(())
}

// ── HNSW determinism ─────────────────────────────────────────────────────────

#[test]
fn hnsw_is_deterministic_on_small_index() -> TestResult {
    let index = build_index(24, 128)?;
    let query = make_vector(9_999, 24);

    let reference = index.search_with_kernel(
        query.as_slice(),
        10,
        &HnswKernel::new(),
        VectorSearchBackend::F32Hnsw,
    )?;

    for iteration in 0..DETERMINISM_ITERATIONS {
        let result = index.search_with_kernel(
            query.as_slice(),
            10,
            &HnswKernel::new(),
            VectorSearchBackend::F32Hnsw,
        )?;
        assert_eq!(
            match_ids(&result.matches),
            match_ids(&reference.matches),
            "HNSW iteration {iteration} produced different result IDs"
        );
    }
    Ok(())
}

// ── HNSW ⊆ FlatScan (subset property) ────────────────────────────────────────

#[test]
fn hnsw_results_are_subset_of_flat_scan() -> TestResult {
    let index = build_index(24, 128)?;
    let query = make_vector(9_999, 24);

    let flat_results = index.search_with_kernel(
        query.as_slice(),
        20,
        &FlatScanKernel,
        VectorSearchBackend::F32Hnsw,
    )?;
    let hnsw_results = index.search_with_kernel(
        query.as_slice(),
        10,
        &HnswKernel::new(),
        VectorSearchBackend::F32Hnsw,
    )?;

    let flat_ids = match_id_set(&flat_results.matches);

    for hnsw_match in &hnsw_results.matches {
        assert!(
            flat_ids.contains(hnsw_match.id.as_ref()),
            "HNSW result {:?} (score={:.6}) is not in flat scan top-20",
            hnsw_match.id,
            hnsw_match.score,
        );
    }
    Ok(())
}

#[test]
fn hnsw_subset_holds_across_multiple_queries() -> TestResult {
    let index = build_index(24, 256)?;

    // Test with 5 different query vectors.
    for query_seed in [42, 100, 777, 1234, 9999_u32] {
        let query = make_vector(query_seed, 24);

        let flat_results = index.search_with_kernel(
            query.as_slice(),
            30,
            &FlatScanKernel,
            VectorSearchBackend::F32Hnsw,
        )?;
        let hnsw_results = index.search_with_kernel(
            query.as_slice(),
            10,
            &HnswKernel::new(),
            VectorSearchBackend::F32Hnsw,
        )?;

        let flat_ids = match_id_set(&flat_results.matches);

        for hnsw_match in &hnsw_results.matches {
            assert!(
                flat_ids.contains(hnsw_match.id.as_ref()),
                "query_seed={query_seed}: HNSW result {:?} not in flat scan top-30",
                hnsw_match.id,
            );
        }
    }
    Ok(())
}

// ── Flat scan scores are correctly sorted ─────────────────────────────────────

#[test]
fn flat_scan_scores_are_monotonically_decreasing() -> TestResult {
    let index = build_index(24, 128)?;
    let query = make_vector(9_999, 24);

    let results = index.search_with_kernel(
        query.as_slice(),
        20,
        &FlatScanKernel,
        VectorSearchBackend::F32Hnsw,
    )?;

    #[allow(
        clippy::indexing_slicing,
        reason = ".windows(2) guarantees slices of exactly length 2"
    )]
    for window in results.matches.windows(2) {
        assert!(
            window[0].score >= window[1].score,
            "flat scan scores not monotonically decreasing: {} ({:.6}) followed by {} ({:.6})",
            window[0].id,
            window[0].score,
            window[1].id,
            window[1].score,
        );
    }
    Ok(())
}

// ── Kernel stats metadata ────────────────────────────────────────────────────

#[test]
fn flat_scan_reports_correct_kernel_kind() -> TestResult {
    let index = build_index(24, 32)?;
    let query = make_vector(9_999, 24);

    let output = index.search_with_kernel(
        query.as_slice(),
        5,
        &FlatScanKernel,
        VectorSearchBackend::F32Hnsw,
    )?;

    assert_eq!(
        output.stats.kernel,
        semantic_code_vector::VectorKernelKind::FlatScan,
        "flat scan should report FlatScan kernel kind in stats"
    );
    assert!(
        output.stats.kernel_search_duration_ns.is_some(),
        "flat scan should report kernel search duration"
    );
    Ok(())
}

#[test]
fn hnsw_reports_correct_kernel_kind() -> TestResult {
    let index = build_index(24, 32)?;
    let query = make_vector(9_999, 24);

    let output = index.search_with_kernel(
        query.as_slice(),
        5,
        &HnswKernel::new(),
        VectorSearchBackend::F32Hnsw,
    )?;

    assert_eq!(
        output.stats.kernel,
        semantic_code_vector::VectorKernelKind::HnswRs,
        "HNSW should report HnswRs kernel kind in stats"
    );
    Ok(())
}

// ── Edge cases ───────────────────────────────────────────────────────────────

#[test]
fn flat_scan_empty_index_returns_empty() -> TestResult {
    let dim = 24_u32;
    let index = VectorIndex::new(dim, HnswParams::default())?;
    let query = make_vector(1, 24);

    let results = index.search_with_kernel(
        query.as_slice(),
        10,
        &FlatScanKernel,
        VectorSearchBackend::F32Hnsw,
    )?;

    assert!(results.matches.is_empty());
    Ok(())
}

#[test]
fn flat_scan_limit_zero_returns_empty() -> TestResult {
    let index = build_index(24, 64)?;
    let query = make_vector(9_999, 24);

    let results = index.search_with_kernel(
        query.as_slice(),
        0,
        &FlatScanKernel,
        VectorSearchBackend::F32Hnsw,
    )?;

    assert!(results.matches.is_empty());
    Ok(())
}

#[test]
fn flat_scan_limit_exceeding_index_size_returns_all() -> TestResult {
    let count = 16_u32;
    let index = build_index(24, count)?;
    let query = make_vector(9_999, 24);

    let results = index.search_with_kernel(
        query.as_slice(),
        100, // much larger than 16 records
        &FlatScanKernel,
        VectorSearchBackend::F32Hnsw,
    )?;

    #[allow(clippy::cast_possible_truncation, reason = "test value fits in usize")]
    let expected = count as usize;
    assert_eq!(results.matches.len(), expected);
    Ok(())
}
