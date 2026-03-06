//! BQ1 threshold calibration use case.
//!
//! Orchestrates: call `CalibrationPort::calibrate_bq1()` → log summary.
//! The binary search algorithm itself runs inside the adapter (which has
//! direct kernel access). This use case handles the app-layer concerns:
//! validation, telemetry, and logging.

use semantic_code_domain::{CalibrationParams, CalibrationState, SearchStats};
use semantic_code_ports::{CalibrationPort, LoggerPort, TelemetryPort};
use semantic_code_shared::{RequestContext, Result};
use std::sync::Arc;
use std::time::Instant;

/// Input payload for BQ1 calibration.
#[derive(Debug, Clone)]
pub struct CalibrateBq1Input {
    /// Calibration algorithm parameters.
    pub params: CalibrationParams,
}

/// Output payload from BQ1 calibration.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrateBq1Output {
    /// Calibrated state (threshold, recall, skip rate, etc.).
    pub state: CalibrationState,
}

/// Dependencies required by the BQ1 calibration use case.
#[derive(Clone)]
pub struct CalibrateBq1Deps {
    /// Calibration adapter (runs binary search against the kernel).
    pub calibration: Arc<dyn CalibrationPort>,
    /// Optional logger.
    pub logger: Option<Arc<dyn LoggerPort>>,
    /// Optional telemetry sink.
    pub telemetry: Option<Arc<dyn TelemetryPort>>,
}

/// Run BQ1 threshold calibration against the current index.
#[tracing::instrument(
    name = "app.calibrate_bq1",
    skip_all,
    fields(
        target_recall = %input.params.target_recall,
        precision = %input.params.precision,
        num_queries = input.params.num_queries.value(),
        top_k = input.params.top_k.value(),
    )
)]
pub async fn calibrate_bq1(
    ctx: &RequestContext,
    deps: &CalibrateBq1Deps,
    input: &CalibrateBq1Input,
) -> Result<CalibrateBq1Output> {
    ctx.ensure_not_cancelled("calibrate_bq1")?;

    if let Some(logger) = deps.logger.as_ref() {
        logger.info(
            "calibrate_bq1.start",
            &format!(
                "Starting BQ1 calibration: target_recall={:.3}, precision={:.4}, queries={}",
                input.params.target_recall.value(),
                input.params.precision.value(),
                input.params.num_queries.value()
            ),
            None,
        );
    }

    let started = Instant::now();

    let state = deps.calibration.calibrate_bq1(ctx, &input.params).await?;

    let elapsed = started.elapsed();

    if let Some(telemetry) = deps.telemetry.as_ref() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "value is clamped to u64::MAX before casting"
        )]
        let duration_ms = elapsed.as_millis().min(u128::from(u64::MAX)) as u64;
        telemetry.record_timer_ms("calibrate_bq1.duration_ms", duration_ms, None);
    }

    if let Some(logger) = deps.logger.as_ref() {
        logger.info(
            "calibrate_bq1.complete",
            &format!(
                "BQ1 calibration complete: threshold={:.4}, recall={:.4}, skip_rate={:.4}, steps={}, elapsed={:.1}s",
                state.threshold,
                state.recall_at_threshold,
                state.skip_rate,
                state.binary_search_steps,
                elapsed.as_secs_f64()
            ),
            None,
        );
    }

    Ok(CalibrateBq1Output { state })
}

// ── EMA online observation ──────────────────────────────────────────

/// Default drift tolerance: 15 percentage-point absolute difference in skip rate.
const DEFAULT_DRIFT_TOLERANCE: f32 = 0.15;

/// Default persistence interval: persist EMA state every 50 observed queries.
const DEFAULT_PERSIST_INTERVAL: u64 = 50;

/// Key in [`SearchStats::extra`] where the DFRR kernel reports Hamming-skipped count.
const HAMMING_SKIPPED_KEY: &str = "hammingSkipped";

/// Outcome of observing BQ1 metrics from a single search.
#[derive(Debug, Clone, PartialEq)]
pub struct Bq1Observation {
    /// Observed skip rate for this search (`hamming_skipped / total_candidates`).
    pub skip_rate: f32,
    /// Whether skip rate has drifted from the calibration baseline.
    pub drift_detected: bool,
    /// Whether the updated calibration state should be persisted to disk.
    pub should_persist: bool,
}

/// Observe BQ1 search metrics and update the calibration EMA tracker.
///
/// Extracts `hammingSkipped` and `expansions` from [`SearchStats`] and computes
/// the observed skip rate. Returns `None` when the stats don't contain DFRR
/// metrics (e.g. when using the HNSW kernel or hybrid search).
#[expect(
    clippy::cast_possible_truncation,
    reason = "skip_rate is in [0.0, 1.0] so f64→f32 truncation is harmless"
)]
pub fn observe_bq1_search(
    search_stats: &SearchStats,
    calibration: &mut CalibrationState,
) -> Option<Bq1Observation> {
    let hamming_skipped = search_stats.extra.get(HAMMING_SKIPPED_KEY).copied()?;
    let expansions = search_stats.expansions.unwrap_or(0);

    #[expect(
        clippy::cast_precision_loss,
        reason = "expansions is a small search counter — fits exactly in f64"
    )]
    let total = hamming_skipped + expansions as f64;
    if total <= 0.0 {
        return None;
    }

    let skip_rate = (hamming_skipped / total) as f32;
    calibration.observe_skip_rate(skip_rate);

    Some(Bq1Observation {
        skip_rate,
        drift_detected: calibration.is_drifted(DEFAULT_DRIFT_TOLERANCE),
        should_persist: calibration.should_persist_ema(DEFAULT_PERSIST_INTERVAL),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_domain::{CalibrationParams, EmaState};
    use std::collections::BTreeMap;

    #[test]
    fn calibrate_bq1_input_default_params() {
        let input = CalibrateBq1Input {
            params: CalibrationParams::default(),
        };
        assert!((input.params.target_recall.value() - 0.99).abs() < f32::EPSILON);
        assert_eq!(input.params.num_queries.value(), 50);
    }

    fn sample_calibration() -> CalibrationState {
        CalibrationState {
            threshold: 0.55,
            recall_at_threshold: 0.99,
            skip_rate: 0.42,
            calibrated_at_ms: 1_700_000_000_000,
            dimension: 1024,
            corpus_size: 50_000,
            num_queries: 50,
            binary_search_steps: 8,
            ema: None,
        }
    }

    fn stats_with_dfrr(hamming_skipped: f64, expansions: u64) -> SearchStats {
        let mut extra = BTreeMap::new();
        extra.insert("hammingSkipped".into(), hamming_skipped);
        SearchStats {
            expansions: Some(expansions),
            kernel: "dfrr".into(),
            extra,
            kernel_search_duration_ns: None,
            index_size: None,
        }
    }

    #[test]
    fn observe_returns_none_for_hnsw_stats() {
        let stats = SearchStats {
            expansions: Some(100),
            kernel: "hnsw-rs".into(),
            extra: BTreeMap::new(),
            kernel_search_duration_ns: None,
            index_size: None,
        };
        let mut state = sample_calibration();
        assert!(observe_bq1_search(&stats, &mut state).is_none());
    }

    #[test]
    fn observe_returns_none_for_zero_candidates() {
        let stats = stats_with_dfrr(0.0, 0);
        let mut state = sample_calibration();
        assert!(observe_bq1_search(&stats, &mut state).is_none());
    }

    #[test]
    fn observe_computes_skip_rate() {
        // 30 skipped out of 100 total (30 + 70) = 0.30 skip rate
        let stats = stats_with_dfrr(30.0, 70);
        let mut state = sample_calibration();
        let obs = observe_bq1_search(&stats, &mut state).expect("should observe");
        assert!((obs.skip_rate - 0.30).abs() < f32::EPSILON);
        assert!(!obs.drift_detected);
        assert!(!obs.should_persist);
    }

    #[test]
    fn observe_initializes_ema() {
        let stats = stats_with_dfrr(40.0, 60);
        let mut state = sample_calibration();
        assert!(state.ema.is_none());

        observe_bq1_search(&stats, &mut state);

        let ema = state.ema.as_ref().expect("EMA should be initialized");
        assert!((ema.current_skip_rate - 0.40).abs() < f32::EPSILON);
        assert_eq!(ema.samples_seen, 1);
    }

    #[test]
    fn observe_detects_drift_after_warmup() {
        let mut state = CalibrationState {
            ema: Some(EmaState {
                alpha: 0.1,
                current_skip_rate: 0.20,
                samples_seen: 49, // will become 50 after observe
            }),
            ..sample_calibration()
        };
        // skip_rate = 10 / (10 + 90) = 0.10 → EMA moves toward 0.10
        // calibrated skip_rate is 0.42, EMA ~0.19 → drift > 0.15
        let stats = stats_with_dfrr(10.0, 90);
        let obs = observe_bq1_search(&stats, &mut state).expect("should observe");
        assert!(obs.drift_detected);
        // samples_seen = 50, which is a persist interval
        assert!(obs.should_persist);
    }

    #[test]
    fn observe_no_drift_when_close_to_baseline() {
        let mut state = CalibrationState {
            ema: Some(EmaState {
                alpha: 0.1,
                current_skip_rate: 0.40,
                samples_seen: 49,
            }),
            ..sample_calibration()
        };
        // 40 / 100 = 0.40 skip rate, very close to calibrated 0.42
        let stats = stats_with_dfrr(40.0, 60);
        let obs = observe_bq1_search(&stats, &mut state).expect("should observe");
        assert!(!obs.drift_detected);
    }
}
