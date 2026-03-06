//! BQ1 calibration adapter for local DFRR kernel.
//!
//! Implements `CalibrationPort` by running a binary search over BQ1 thresholds
//! against a loaded `VectorIndex`. Probe vectors are sampled from the existing
//! index using a seeded PRNG for deterministic results.

use semantic_code_domain::{CalibrationParams, CalibrationState};
use semantic_code_ports::CalibrationPort;
use semantic_code_shared::{ErrorClass, ErrorCode, ErrorEnvelope, RequestContext, Result};
use semantic_code_vector::{
    HnswParams, VectorIndex, VectorKernel, VectorRecord, VectorSearchBackend, VectorSearchOutput,
};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::instrument;

/// Maximum binary search iterations to prevent infinite loops when
/// floating-point precision prevents convergence.
const MAX_BINARY_SEARCH_STEPS: u32 = 64;

/// Local calibration adapter backed by a concrete `VectorKernel` + `VectorIndex`.
///
/// The binary search calibration algorithm:
/// 1. Sample N probe vectors from the existing index (seeded PRNG).
/// 2. Run baseline searches **without** BQ1 (`config_json = None`) as ground truth.
/// 3. Binary search on `bq1_threshold` in `[0.0, 1.0]`:
///    - At each midpoint, run all probe queries with that threshold.
///    - Compute average `recall@K` vs baseline.
///    - Converge when `(hi - lo) < precision` or max steps reached.
/// 4. Return `CalibrationState` with the optimal threshold.
pub struct LocalCalibrationAdapter {
    kernel: Arc<dyn VectorKernel + Send + Sync>,
    index: Arc<VectorIndex>,
    search_backend: VectorSearchBackend,
}

impl LocalCalibrationAdapter {
    /// Build the adapter from a pre-loaded kernel and index.
    ///
    /// The adapter takes shared ownership of both — the kernel is typically
    /// shared with `LocalVectorDb`, while the index may be an independent
    /// load from the snapshot (calibration runs as a dedicated command).
    #[must_use]
    pub fn new(
        kernel: Arc<dyn VectorKernel + Send + Sync>,
        index: Arc<VectorIndex>,
        search_backend: VectorSearchBackend,
    ) -> Self {
        Self {
            kernel,
            index,
            search_backend,
        }
    }

    /// Load a calibration adapter by reading the vector snapshot from disk.
    ///
    /// Parses only the fields needed for calibration (record id and vector),
    /// ignoring document content and metadata. The snapshot is the
    /// `CollectionSnapshot` format written by [`LocalVectorDb`]. If the JSON
    /// snapshot is unavailable or version-mismatched, this loader will fall
    /// back to a sibling `*.v2/` bundle when present.
    pub fn load_from_snapshot_path(
        kernel: Arc<dyn VectorKernel + Send + Sync>,
        snapshot_path: &Path,
        search_backend: VectorSearchBackend,
    ) -> Result<Self> {
        let payload = match std::fs::read(snapshot_path) {
            Ok(payload) => payload,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if let Some(index) = try_load_v2_companion_index(snapshot_path)? {
                    return Ok(Self::new(kernel, Arc::new(index), search_backend));
                }
                return Err(ErrorEnvelope::expected(
                    ErrorCode::new("calibration", "snapshot_not_found"),
                    "vector snapshot not found; run `sca index --init` first",
                )
                .with_metadata("path", snapshot_path.display().to_string()));
            },
            Err(error) => return Err(ErrorEnvelope::from(error)),
        };

        let snapshot: MinimalSnapshot = match serde_json::from_slice(&payload) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                if let Some(index) = try_load_v2_companion_index(snapshot_path)? {
                    tracing::warn!(
                        path = %snapshot_path.display(),
                        "snapshot JSON parse failed; using v2 companion for calibration"
                    );
                    return Ok(Self::new(kernel, Arc::new(index), search_backend));
                }
                return Err(ErrorEnvelope::unexpected(
                    ErrorCode::new("calibration", "snapshot_parse_failed"),
                    format!("failed to parse snapshot for calibration: {error}"),
                    ErrorClass::NonRetriable,
                ));
            },
        };

        if snapshot.version != 1 {
            if let Some(index) = try_load_v2_companion_index(snapshot_path)? {
                tracing::warn!(
                    version = snapshot.version,
                    path = %snapshot_path.display(),
                    "unsupported JSON snapshot version; using v2 companion for calibration"
                );
                return Ok(Self::new(kernel, Arc::new(index), search_backend));
            }
            return Err(ErrorEnvelope::expected(
                ErrorCode::new("calibration", "snapshot_version_mismatch"),
                format!("unsupported snapshot version: {}", snapshot.version),
            )
            .with_metadata("version", snapshot.version.to_string()));
        }

        let params = HnswParams::default();
        let mut index = VectorIndex::new(snapshot.dimension, params)?;
        let records: Vec<VectorRecord> = snapshot
            .records
            .into_iter()
            .map(|r| VectorRecord {
                id: r.id,
                vector: r.vector,
            })
            .collect();
        index.insert(records)?;

        tracing::info!(
            dimension = snapshot.dimension,
            records = index.active_count(),
            path = %snapshot_path.display(),
            "loaded vector index for calibration"
        );

        Ok(Self::new(kernel, Arc::new(index), search_backend))
    }
}

fn try_load_v2_companion_index(snapshot_path: &Path) -> Result<Option<VectorIndex>> {
    let Some(v2_dir) = v2_companion_dir(snapshot_path) else {
        return Ok(None);
    };
    if !v2_dir.is_dir() {
        return Ok(None);
    }
    let index = VectorIndex::from_snapshot_v2(&v2_dir)?;
    tracing::info!(
        dimension = index.dimension(),
        records = index.active_count(),
        path = %v2_dir.display(),
        "loaded vector index from v2 companion for calibration"
    );
    Ok(Some(index))
}

fn v2_companion_dir(snapshot_path: &Path) -> Option<PathBuf> {
    let stem = snapshot_path.file_stem()?;
    let parent = snapshot_path.parent()?;
    Some(parent.join(format!("{}.v2", stem.to_string_lossy())))
}

/// Minimal snapshot structure for calibration purposes.
///
/// Only deserializes the fields needed to build a `VectorIndex` —
/// serde skips unknown fields (like `content`, `metadata`, `indexMode`)
/// by default.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MinimalSnapshot {
    version: u32,
    dimension: u32,
    records: Vec<MinimalRecord>,
}

#[derive(Deserialize)]
struct MinimalRecord {
    id: Box<str>,
    vector: Vec<f32>,
}

impl CalibrationPort for LocalCalibrationAdapter {
    fn calibrate_bq1(
        &self,
        ctx: &RequestContext,
        params: &CalibrationParams,
    ) -> semantic_code_ports::BoxFuture<'_, Result<CalibrationState>> {
        let ctx = ctx.clone();
        let kernel = Arc::clone(&self.kernel);
        let index = Arc::clone(&self.index);
        let backend = self.search_backend;
        let params = params.clone();

        Box::pin(async move {
            ctx.ensure_not_cancelled("calibration.calibrate_bq1")?;

            let corpus_size = index.active_count();
            if corpus_size == 0 {
                return Err(ErrorEnvelope::expected(
                    ErrorCode::new("calibration", "empty_index"),
                    "cannot calibrate BQ1 on an empty index",
                ));
            }

            let dimension = index.dimension();
            let requested_top_k = params.top_k.value();
            let target_recall = params.target_recall.value();

            // Run the CPU-intensive binary search on a blocking thread.
            let state = tokio::task::spawn_blocking(move || {
                run_binary_search(&kernel, &index, backend, &params, dimension, corpus_size)
            })
            .await
            .map_err(|error| {
                ErrorEnvelope::unexpected(
                    ErrorCode::new("calibration", "join_error"),
                    format!("calibration task panicked: {error}"),
                    ErrorClass::NonRetriable,
                )
            })??;

            // Validate the calibration produced a usable result.
            if state.recall_at_threshold < target_recall {
                tracing::warn!(
                    threshold = state.threshold,
                    recall = state.recall_at_threshold,
                    target = target_recall,
                    "calibration converged below target recall — \
                     BQ1 may not be effective for this index"
                );
            }

            tracing::info!(
                threshold = format_args!("{:.4}", state.threshold),
                recall = format_args!("{:.4}", state.recall_at_threshold),
                skip_rate = format_args!("{:.4}", state.skip_rate),
                steps = state.binary_search_steps,
                top_k = requested_top_k,
                "BQ1 calibration complete"
            );

            Ok(state)
        })
    }
}

// ─── Binary search core (sync, runs on blocking thread) ─────────────────────

/// Run the binary search calibration loop (sync).
#[instrument(
    name = "adapter.calibration.binary_search",
    skip_all,
    fields(
        target_recall = %params.target_recall,
        precision = %params.precision,
        num_queries = params.num_queries.value(),
        top_k = params.top_k.value(),
        corpus_size,
        dimension,
    )
)]
fn run_binary_search(
    kernel: &Arc<dyn VectorKernel + Send + Sync>,
    index: &VectorIndex,
    backend: VectorSearchBackend,
    params: &CalibrationParams,
    dimension: u32,
    corpus_size: usize,
) -> Result<CalibrationState> {
    let top_k = usize::try_from(params.top_k.value()).map_err(|_| {
        ErrorEnvelope::expected(
            ErrorCode::new("calibration", "invalid_top_k"),
            "top_k value is too large to represent as usize",
        )
    })?;
    let limit = top_k.saturating_mul(5).max(top_k);

    // 1. Sample probe vectors from the index.
    let num_queries = usize::try_from(params.num_queries.value()).map_err(|_| {
        ErrorEnvelope::expected(
            ErrorCode::new("calibration", "invalid_num_queries"),
            "num_queries value is too large to represent as usize",
        )
    })?;
    let probes = sample_probe_vectors(index, num_queries, params.seed);
    if probes.is_empty() {
        return Err(ErrorEnvelope::expected(
            ErrorCode::new("calibration", "no_probes"),
            "no probe vectors available for calibration",
        ));
    }

    tracing::debug!(
        num_probes = probes.len(),
        limit,
        "sampled probe vectors for calibration"
    );

    // 2. Baseline: search WITHOUT BQ1 (config_json = None → default behavior).
    let baselines: Vec<VectorSearchOutput> = probes
        .iter()
        .map(|query| kernel.search_with_config_override(index, query, limit, backend, None))
        .collect::<Result<Vec<_>>>()?;

    tracing::debug!("baseline searches complete");

    // 3. Binary search on `bq1_threshold`.
    let mut lo: f32 = 0.0;
    let mut hi: f32 = 1.0;
    let mut steps: u32 = 0;
    let mut found_feasible = false;
    let mut best_threshold: f32 = 1.0; // conservative: no skipping
    let mut best_recall: f32 = 0.0;
    let mut best_skip_rate: f32 = 0.0;

    // Use a step counter instead of float comparison in the loop condition
    // to satisfy clippy::while_float.
    loop {
        if steps >= MAX_BINARY_SEARCH_STEPS || (hi - lo) < params.precision.value() {
            break;
        }
        let mid = lo.mul_add(0.5, hi * 0.5); // (lo + hi) / 2 via fused mul-add
        steps += 1;

        let config_json = build_bq1_config_json(mid);
        let (recall, skip_rate) = evaluate_threshold(
            kernel,
            index,
            backend,
            &probes,
            &baselines,
            &config_json,
            top_k,
        )?;

        tracing::debug!(
            step = steps,
            threshold = format_args!("{mid:.4}"),
            recall = format_args!("{recall:.4}"),
            skip_rate = format_args!("{skip_rate:.4}"),
            "binary search step"
        );

        if recall >= params.target_recall.value() {
            // Recall meets target — can try more aggressive (lower) threshold.
            found_feasible = true;
            best_threshold = mid;
            best_recall = recall;
            best_skip_rate = skip_rate;
            hi = mid;
        } else {
            // Too aggressive — relax (raise) threshold.
            lo = mid;
        }
    }

    if !found_feasible {
        (best_recall, best_skip_rate) =
            measure_threshold_metrics(kernel, index, backend, &probes, &baselines, top_k, 1.0)?;
    }

    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));

    #[expect(
        clippy::cast_possible_truncation,
        reason = "corpus_size and probes.len() are bounded by usize ≤ u64/u32 in practice"
    )]
    let state = CalibrationState {
        threshold: best_threshold,
        recall_at_threshold: best_recall,
        skip_rate: best_skip_rate,
        calibrated_at_ms: now_ms,
        dimension,
        corpus_size: corpus_size as u64,
        num_queries: probes.len() as u32,
        binary_search_steps: steps,
        ema: None,
    };

    Ok(state)
}

/// Evaluate a single BQ1 threshold against all probes and return
/// `(avg_recall, skip_rate)`.
fn evaluate_threshold(
    kernel: &Arc<dyn VectorKernel + Send + Sync>,
    index: &VectorIndex,
    backend: VectorSearchBackend,
    probes: &[Vec<f32>],
    baselines: &[VectorSearchOutput],
    config_json: &str,
    top_k: usize,
) -> Result<(f32, f32)> {
    let mut total_recall = 0.0_f64;
    let mut total_hamming_skipped = 0_u64;
    let mut total_candidates = 0_u64;
    let limit = top_k.saturating_mul(5).max(top_k);

    for (probe, baseline) in probes.iter().zip(baselines.iter()) {
        let candidate =
            kernel.search_with_config_override(index, probe, limit, backend, Some(config_json))?;

        // Recall@K: intersection of top-K result IDs from baseline and candidate.
        let baseline_ids: HashSet<&str> = baseline
            .matches
            .iter()
            .take(top_k)
            .map(|m| m.id.as_ref())
            .collect();
        let candidate_ids: HashSet<&str> = candidate
            .matches
            .iter()
            .take(top_k)
            .map(|m| m.id.as_ref())
            .collect();

        let intersection = baseline_ids.intersection(&candidate_ids).count();
        let denominator = baseline_ids.len().max(1);

        #[expect(
            clippy::cast_precision_loss,
            reason = "intersection and denominator are small (top_k sized) — no precision loss"
        )]
        {
            total_recall += intersection as f64 / denominator as f64;
        }

        // Accumulate skip stats from kernel extra metrics.
        let hamming_skipped_f64 = candidate
            .stats
            .extra
            .get("hammingSkipped")
            .copied()
            .unwrap_or(0.0);
        // The metric is always non-negative and integral.
        let hamming_skipped = if hamming_skipped_f64 >= 0.0 {
            // Saturate at u64::MAX for extreme values.
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "hamming_skipped is a non-negative integer count from kernel metrics"
            )]
            {
                hamming_skipped_f64 as u64
            }
        } else {
            0
        };
        let expansions = candidate.stats.expansions.unwrap_or(0);
        total_hamming_skipped += hamming_skipped;
        total_candidates += expansions + hamming_skipped;
    }

    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "probes.len() fits exactly in f64; recall is in [0.0, 1.0] so f64→f32 truncation is harmless"
    )]
    let avg_recall = (total_recall / probes.len() as f64) as f32;

    #[expect(
        clippy::cast_precision_loss,
        reason = "skip rate is a ratio — f32 precision is sufficient for reporting"
    )]
    let skip_rate = if total_candidates > 0 {
        total_hamming_skipped as f32 / total_candidates as f32
    } else {
        0.0
    };

    Ok((avg_recall, skip_rate))
}

fn measure_threshold_metrics(
    kernel: &Arc<dyn VectorKernel + Send + Sync>,
    index: &VectorIndex,
    backend: VectorSearchBackend,
    probes: &[Vec<f32>],
    baselines: &[VectorSearchOutput],
    top_k: usize,
    threshold: f32,
) -> Result<(f32, f32)> {
    let config_json = build_bq1_config_json(threshold);
    evaluate_threshold(
        kernel,
        index,
        backend,
        probes,
        baselines,
        &config_json,
        top_k,
    )
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Build the JSON config override that sets `bq1_threshold` on a `DfrrLoopConfig`.
///
/// We use `serde_json::json!` to build just the threshold field. The kernel's
/// `search_with_config_override` deserializes this into a full `DfrrLoopConfig`
/// with defaults for all other fields.
fn build_bq1_config_json(threshold: f32) -> String {
    // DfrrLoopConfig has serde defaults on all fields, so we only need to set
    // the one field we're varying.
    serde_json::json!({ "bq1_threshold": threshold }).to_string()
}

/// Deterministically sample probe vectors from the index using a simple
/// seeded LCG (Linear Congruential Generator).
///
/// We avoid pulling in `rand` just for this — the sampling only needs to be
/// deterministic and roughly uniform, not cryptographically secure.
fn sample_probe_vectors(index: &VectorIndex, num_queries: usize, prng_seed: u64) -> Vec<Vec<f32>> {
    let records = index.active_records();
    if records.is_empty() {
        return Vec::new();
    }

    let count = num_queries.min(records.len());
    let mut lcg_state = prng_seed;

    // When sampling a large fraction of the corpus, rejection sampling
    // spends significant time drawing duplicates. A seeded Fisher-Yates
    // shuffle keeps runtime linear and deterministic.
    if count.saturating_mul(2) >= records.len() {
        let mut indices: Vec<usize> = (0..records.len()).collect();
        for i in (1..indices.len()).rev() {
            lcg_state = next_lcg_state(lcg_state);
            let j = (lcg_state >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }
        indices.truncate(count);
        return indices
            .into_iter()
            .filter_map(|idx| records.get(idx).map(|r| r.vector.clone()))
            .collect();
    }

    let mut indices: Vec<usize> = Vec::with_capacity(count);
    let mut visited = HashSet::with_capacity(count);

    while indices.len() < count {
        lcg_state = next_lcg_state(lcg_state);
        let idx = (lcg_state >> 33) as usize % records.len();
        if visited.insert(idx) {
            indices.push(idx);
        }
    }

    indices
        .into_iter()
        .filter_map(|idx| records.get(idx).map(|r| r.vector.clone()))
        .collect()
}

#[inline]
const fn next_lcg_state(state: u64) -> u64 {
    // LCG: next = (a * state + c) mod m (Knuth's constants).
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_domain::{
        CalibrationPrecision, CalibrationQueryCount, CalibrationTopK, TargetRecall,
    };
    use semantic_code_vector::{
        HnswKernel, HnswParams, KernelSearchStats, VectorKernelKind, VectorMatch, VectorRecord,
        VectorSearchOutput,
    };
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};
    use std::{env, fs};

    /// Build a test index with `n` unit vectors uniformly distributed on a circle.
    ///
    /// Config overrides are ignored by HNSW (no BQ1), which means calibration
    /// recall will always be 1.0 and `skip_rate` 0.0.
    fn make_test_index(dim: u32, n: usize) -> VectorIndex {
        #[allow(clippy::unwrap_used, reason = "test helper — panics are acceptable")]
        let mut index = VectorIndex::new(dim, HnswParams::default()).unwrap();

        #[allow(
            clippy::cast_precision_loss,
            reason = "test helper — small integers are exact in f32"
        )]
        let records: Vec<VectorRecord> = (0..n)
            .map(|i| {
                let angle = (i as f32) * std::f32::consts::TAU / (n as f32);
                let mut vec = vec![0.0_f32; dim as usize];
                vec[0] = angle.cos();
                vec[1] = angle.sin();
                VectorRecord {
                    id: format!("v{i}").into(),
                    vector: vec,
                }
            })
            .collect();

        #[allow(clippy::unwrap_used, reason = "test helper")]
        index.insert(records).unwrap();

        index
    }

    fn temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        env::temp_dir().join(format!("{prefix}-{nanos}"))
    }

    fn query_count(value: u32) -> CalibrationQueryCount {
        CalibrationQueryCount::try_from(value).unwrap_or(CalibrationQueryCount::DEFAULT)
    }

    fn top_k(value: u32) -> CalibrationTopK {
        CalibrationTopK::try_from(value).unwrap_or(CalibrationTopK::DEFAULT)
    }

    fn target_recall(value: f32) -> TargetRecall {
        TargetRecall::try_from(value).unwrap_or(TargetRecall::DEFAULT)
    }

    fn precision(value: f32) -> CalibrationPrecision {
        CalibrationPrecision::try_from(value).unwrap_or(CalibrationPrecision::DEFAULT)
    }

    #[test]
    fn sample_probe_vectors_deterministic() {
        let index = make_test_index(4, 100);
        let a = sample_probe_vectors(&index, 10, 42);
        let b = sample_probe_vectors(&index, 10, 42);
        assert_eq!(a, b, "same seed must produce same probes");
    }

    #[test]
    fn sample_probe_vectors_respects_count() {
        let index = make_test_index(4, 5);
        let probes = sample_probe_vectors(&index, 100, 42);
        assert_eq!(probes.len(), 5, "cannot sample more than corpus size");
    }

    #[test]
    fn sample_probe_vectors_full_sample_is_deterministic_and_unique() {
        let index = make_test_index(4, 50);
        let probes_a = sample_probe_vectors(&index, 50, 42);
        let probes_b = sample_probe_vectors(&index, 50, 42);
        assert_eq!(probes_a, probes_b, "full sampling must be deterministic");
        let unique: HashSet<(u32, u32)> = probes_a
            .iter()
            .map(|probe| (probe[0].to_bits(), probe[1].to_bits()))
            .collect();
        assert_eq!(unique.len(), 50, "full sampling must not duplicate probes");
    }

    #[test]
    fn sample_probe_vectors_empty_index() {
        #[allow(clippy::unwrap_used, reason = "test")]
        let index = VectorIndex::new(4, HnswParams::default()).unwrap();
        let probes = sample_probe_vectors(&index, 10, 42);
        assert!(probes.is_empty());
    }

    #[test]
    fn build_bq1_config_json_contains_threshold() {
        let json = build_bq1_config_json(0.45);
        #[allow(clippy::unwrap_used, reason = "test")]
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        #[allow(clippy::unwrap_used, reason = "test")]
        let threshold = parsed["bq1_threshold"].as_f64().unwrap();
        assert!((threshold - 0.45).abs() < 1e-6);
    }

    #[test]
    fn load_from_snapshot_path_falls_back_to_v2_companion() {
        let root = temp_dir("calibration-v2-fallback");
        #[allow(clippy::unwrap_used, reason = "test")]
        fs::create_dir_all(&root).unwrap();
        let snapshot_path = root.join("code_chunks.json");
        let v2_dir = root.join("code_chunks.v2");
        let index = make_test_index(4, 12);
        #[allow(clippy::unwrap_used, reason = "test")]
        index.snapshot_v2(&v2_dir).unwrap();

        let kernel: Arc<dyn VectorKernel + Send + Sync> = Arc::new(HnswKernel::new());
        #[allow(clippy::unwrap_used, reason = "test")]
        let adapter = LocalCalibrationAdapter::load_from_snapshot_path(
            kernel,
            &snapshot_path,
            VectorSearchBackend::F32Hnsw,
        )
        .unwrap();

        assert_eq!(adapter.index.dimension(), 4);
        assert_eq!(adapter.index.active_count(), 12);
        #[allow(clippy::unwrap_used, reason = "test cleanup")]
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn calibration_on_hnsw_kernel_returns_full_recall() {
        use semantic_code_shared::RequestContext;
        use semantic_code_vector::HnswKernel;

        let index = make_test_index(4, 50);
        let kernel: Arc<dyn VectorKernel + Send + Sync> = Arc::new(HnswKernel::new());
        let adapter =
            LocalCalibrationAdapter::new(kernel, Arc::new(index), VectorSearchBackend::F32Hnsw);

        let ctx = RequestContext::new_request();
        let params = CalibrationParams {
            num_queries: query_count(5),
            top_k: top_k(3),
            ..CalibrationParams::default()
        };

        #[allow(clippy::unwrap_used, reason = "test")]
        let state = adapter.calibrate_bq1(&ctx, &params).await.unwrap();

        // HNSW kernel ignores config overrides → always returns identical
        // results → recall = 1.0, skip_rate = 0.0, threshold converges to 0.0.
        assert!(
            (state.recall_at_threshold - 1.0).abs() < f32::EPSILON,
            "expected perfect recall on HNSW kernel, got {}",
            state.recall_at_threshold
        );
        assert_eq!(state.dimension, 4);
        assert_eq!(state.corpus_size, 50);
        assert!(state.binary_search_steps > 0);
    }

    #[tokio::test]
    async fn calibration_rejects_empty_index() {
        use semantic_code_shared::RequestContext;

        #[allow(clippy::unwrap_used, reason = "test")]
        let index = VectorIndex::new(4, HnswParams::default()).unwrap();
        let kernel: Arc<dyn VectorKernel + Send + Sync> = Arc::new(HnswKernel::new());
        let adapter =
            LocalCalibrationAdapter::new(kernel, Arc::new(index), VectorSearchBackend::F32Hnsw);

        let ctx = RequestContext::new_request();
        let params = CalibrationParams::default();

        #[allow(clippy::unwrap_used, reason = "test")]
        let err = adapter.calibrate_bq1(&ctx, &params).await.unwrap_err();
        assert!(
            err.message.contains("empty index"),
            "expected empty index error, got: {}",
            err.message
        );
    }

    #[derive(Debug)]
    struct NeverFeasibleKernel;

    impl VectorKernel for NeverFeasibleKernel {
        fn kind(&self) -> VectorKernelKind {
            VectorKernelKind::HnswRs
        }

        fn search(
            &self,
            index: &VectorIndex,
            query: &[f32],
            limit: usize,
            backend: VectorSearchBackend,
        ) -> Result<VectorSearchOutput> {
            self.search_with_config_override(index, query, limit, backend, None)
        }

        fn search_with_config_override(
            &self,
            index: &VectorIndex,
            _query: &[f32],
            limit: usize,
            _backend: VectorSearchBackend,
            config_json: Option<&str>,
        ) -> Result<VectorSearchOutput> {
            if config_json.is_none() {
                let matches = index
                    .active_records()
                    .iter()
                    .take(limit)
                    .map(|record| VectorMatch {
                        id: record.id.clone(),
                        score: 1.0,
                    })
                    .collect();
                return Ok(VectorSearchOutput {
                    matches,
                    stats: KernelSearchStats {
                        expansions: Some(limit as u64),
                        kernel: VectorKernelKind::HnswRs,
                        extra: BTreeMap::new(),
                        kernel_search_duration_ns: None,
                    },
                });
            }

            let mut extra = BTreeMap::new();
            extra.insert("hammingSkipped".into(), 10.0);
            // Candidate IDs intentionally never overlap baseline IDs.
            let matches = (0..limit)
                .map(|i| VectorMatch {
                    id: format!("off-{i}").into_boxed_str(),
                    score: 0.5,
                })
                .collect();
            Ok(VectorSearchOutput {
                matches,
                stats: KernelSearchStats {
                    expansions: Some(0),
                    kernel: VectorKernelKind::HnswRs,
                    extra,
                    kernel_search_duration_ns: None,
                },
            })
        }
    }

    #[test]
    fn run_binary_search_measures_threshold_one_when_no_solution_exists() {
        let index = make_test_index(4, 20);
        let kernel: Arc<dyn VectorKernel + Send + Sync> = Arc::new(NeverFeasibleKernel);
        let params = CalibrationParams {
            target_recall: target_recall(0.99),
            precision: precision(0.005),
            num_queries: query_count(8),
            top_k: top_k(5),
            seed: 7,
        };

        #[allow(clippy::unwrap_used, reason = "test")]
        let state = run_binary_search(
            &kernel,
            &index,
            VectorSearchBackend::F32Hnsw,
            &params,
            index.dimension(),
            index.active_count(),
        )
        .unwrap();

        assert!(
            (state.threshold - 1.0).abs() < f32::EPSILON,
            "expected conservative threshold when no candidate meets target"
        );
        assert!(
            state.recall_at_threshold < params.target_recall.value(),
            "expected measured recall below target, got {}",
            state.recall_at_threshold
        );
        assert!(
            state.recall_at_threshold <= f32::EPSILON,
            "expected near-zero measured recall, got {}",
            state.recall_at_threshold
        );
    }
}
