//! Vector search benchmark placeholder for f32 vs experimental u8 search.

use semantic_code_shared::Result;
use semantic_code_vector::{HnswParams, VectorIndex, VectorRecord, VectorSearchBackend};
use std::time::Instant;

#[allow(
    clippy::cast_precision_loss,
    reason = "benchmark synthetic vector generation intentionally uses coarse f32 casts"
)]
fn make_vector(seed: u32, dimension: usize) -> Vec<f32> {
    (0..dimension)
        .map(|offset| {
            let base = seed as f32;
            let axis = offset as f32;
            let sine = (base.mul_add(0.17, axis * 0.11)).sin() * 0.75;
            let cosine = (base.mul_add(0.07, axis * 0.19)).cos() * 0.25;
            sine + cosine + 0.05
        })
        .collect()
}

fn main() -> Result<()> {
    if std::env::args().any(|arg| arg == "--list") {
        println!("vector_search_f32: benchmark");
        println!("vector_search_u8: benchmark");
        return Ok(());
    }

    let dimension = 48usize;
    let mut index = VectorIndex::new(
        u32::try_from(dimension).unwrap_or(48),
        HnswParams::default(),
    )?;
    let records = (0u32..600u32)
        .map(|seed| VectorRecord {
            id: format!("doc_{seed:04}").into_boxed_str(),
            vector: make_vector(seed, dimension),
        })
        .collect::<Vec<_>>();
    index.insert(records)?;

    let queries = (0u32..150u32)
        .map(|seed| make_vector(seed + 1_000, dimension))
        .collect::<Vec<_>>();
    let top_k = 10usize;

    let start_f32 = Instant::now();
    let mut f32_total = 0usize;
    for query in &queries {
        f32_total += index.search(query.as_slice(), top_k)?.matches.len();
    }
    let elapsed_f32 = start_f32.elapsed().as_millis();

    let start_u8 = Instant::now();
    let mut u8_total = 0usize;
    for query in &queries {
        u8_total += index
            .search_with_backend(
                query.as_slice(),
                top_k,
                VectorSearchBackend::ExperimentalU8ThenF32Rerank,
            )?
            .matches
            .len();
    }
    let elapsed_u8 = start_u8.elapsed().as_millis();

    println!(
        "vector_search_f32: {} queries in {} ms",
        queries.len(),
        elapsed_f32
    );
    println!(
        "vector_search_u8: {} queries in {} ms",
        queries.len(),
        elapsed_u8
    );
    println!("totals: f32={f32_total} u8={u8_total}");
    Ok(())
}
