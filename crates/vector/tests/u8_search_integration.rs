//! Integration coverage for experimental u8 search.

use semantic_code_vector::{HnswParams, VectorIndex, VectorRecord, VectorSearchBackend};
use std::collections::BTreeSet;

type TestResult = Result<(), Box<dyn std::error::Error>>;

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
    normalize(values)
}

fn normalize(values: Vec<f32>) -> Vec<f32> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= 0.0 {
        return values;
    }
    values.into_iter().map(|value| value / norm).collect()
}

fn cosine_similarity(lhs: &[f32], rhs: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_lhs = 0.0f32;
    let mut norm_rhs = 0.0f32;
    for (left, right) in lhs.iter().copied().zip(rhs.iter().copied()) {
        dot += left * right;
        norm_lhs += left * left;
        norm_rhs += right * right;
    }
    if norm_lhs <= 0.0 || norm_rhs <= 0.0 {
        return 0.0;
    }
    dot / (norm_lhs * norm_rhs).sqrt()
}

fn exact_top_k(records: &[VectorRecord], query: &[f32], top_k: usize) -> BTreeSet<Box<str>> {
    let mut scored = records
        .iter()
        .map(|record| {
            (
                record.id.clone(),
                cosine_similarity(record.vector.as_slice(), query),
            )
        })
        .collect::<Vec<_>>();
    scored.sort_by(|(left_id, left_score), (right_id, right_score)| {
        let score = right_score.total_cmp(left_score);
        if score != std::cmp::Ordering::Equal {
            return score;
        }
        left_id.cmp(right_id)
    });
    scored
        .into_iter()
        .take(top_k)
        .map(|(id, _)| id)
        .collect::<BTreeSet<_>>()
}

#[test]
fn u8_then_f32_rerank_matches_exact_top_k_overlap() -> TestResult {
    let dimension = 48usize;
    let mut index = VectorIndex::new(u32::try_from(dimension)?, HnswParams::default())?;
    let records = (0u32..600u32)
        .map(|seed| VectorRecord {
            id: format!("doc_{seed:04}").into_boxed_str(),
            vector: make_vector(seed, dimension),
        })
        .collect::<Vec<_>>();
    index.insert(records.clone())?;

    let top_k = 8usize;
    let mut overlapped = 0usize;
    let mut total = 0usize;

    for query_seed in 0u32..150u32 {
        let query = make_vector(query_seed + 1_000, dimension);
        let exact_ids = exact_top_k(records.as_slice(), query.as_slice(), top_k);
        let rerank_ids = index
            .search_with_backend(
                query.as_slice(),
                top_k,
                VectorSearchBackend::ExperimentalU8ThenF32Rerank,
            )?
            .matches
            .into_iter()
            .map(|item| item.id)
            .collect::<BTreeSet<_>>();

        overlapped += exact_ids.intersection(&rerank_ids).count();
        total += top_k;
    }

    let ratio = overlapped as f64 / total as f64;
    assert!(ratio >= 0.98, "expected overlap >= 0.98, found {ratio:.4}");
    Ok(())
}

#[test]
fn backend_enum_routes_correctly() -> TestResult {
    let dimension = 24usize;
    let mut index = VectorIndex::new(u32::try_from(dimension)?, HnswParams::default())?;
    let records = (0u32..32u32)
        .map(|seed| VectorRecord {
            id: format!("doc_{seed:04}").into_boxed_str(),
            vector: make_vector(seed, dimension),
        })
        .collect::<Vec<_>>();
    index.insert(records)?;
    let query = make_vector(7_777, dimension);

    let f32_results =
        index.search_with_backend(query.as_slice(), 5, VectorSearchBackend::F32Hnsw)?;
    let rerank_results = index.search_with_backend(
        query.as_slice(),
        5,
        VectorSearchBackend::ExperimentalU8ThenF32Rerank,
    )?;

    assert_eq!(f32_results.matches.len(), 5);
    assert_eq!(rerank_results.matches.len(), 5);
    Ok(())
}
