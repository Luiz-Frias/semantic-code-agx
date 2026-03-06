//! BQ1 threshold calibration domain types.
//!
//! These types represent the state and parameters for adaptive
//! BQ1 (Binary Quantization level-1) Hamming prefilter calibration.
//! The calibration system binary-searches for the optimal threshold
//! that maximizes candidate skip rate while maintaining target recall.

use serde::{Deserialize, Serialize};
use std::error::Error as StdError;
use std::fmt;

/// Result of a BQ1 calibration run, persisted as `.context/calibration.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationState {
    /// Calibrated BQ1 threshold ratio in `[0.0, 1.0]`.
    pub threshold: f32,
    /// Measured recall at the calibrated threshold (e.g. 0.99).
    pub recall_at_threshold: f32,
    /// Fraction of candidates skipped by BQ1 at the calibrated threshold.
    pub skip_rate: f32,
    /// Epoch milliseconds when calibration was performed.
    pub calibrated_at_ms: u64,
    /// Embedding dimension at calibration time (e.g. 384, 1024).
    pub dimension: u32,
    /// Number of vectors in the index at calibration time.
    pub corpus_size: u64,
    /// Number of probe queries used during calibration.
    pub num_queries: u32,
    /// Number of binary search iterations to converge.
    pub binary_search_steps: u32,
    /// Online EMA tracker state, if active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ema: Option<EmaState>,
}

/// Validation error for calibration parameter newtypes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationParamError {
    field: &'static str,
    value: String,
    message: &'static str,
}

impl CalibrationParamError {
    const fn new(field: &'static str, value: String, message: &'static str) -> Self {
        Self {
            field,
            value,
            message,
        }
    }

    /// Parameter field name that failed validation.
    #[must_use]
    pub const fn field(&self) -> &'static str {
        self.field
    }

    /// Invalid value rendered as text.
    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Human-readable validation failure reason.
    #[must_use]
    pub const fn message(&self) -> &'static str {
        self.message
    }
}

impl fmt::Display for CalibrationParamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={} ({})", self.field, self.value, self.message)
    }
}

impl StdError for CalibrationParamError {}

/// Minimum acceptable recall target in `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TargetRecall(f32);

impl TargetRecall {
    /// Default recall target used for calibration.
    pub const DEFAULT: Self = Self(0.99);

    /// Borrow inner scalar value.
    #[must_use]
    pub const fn value(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for TargetRecall {
    type Error = CalibrationParamError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(CalibrationParamError::new(
                "target_recall",
                value.to_string(),
                "must be a finite value in [0.0, 1.0]",
            ));
        }
        Ok(Self(value))
    }
}

impl fmt::Display for TargetRecall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<TargetRecall> for f32 {
    fn from(value: TargetRecall) -> Self {
        value.value()
    }
}

/// Binary-search convergence precision in `(0.0, 1.0]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CalibrationPrecision(f32);

impl CalibrationPrecision {
    /// Default convergence precision.
    pub const DEFAULT: Self = Self(0.005);

    /// Borrow inner scalar value.
    #[must_use]
    pub const fn value(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for CalibrationPrecision {
    type Error = CalibrationParamError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        if !value.is_finite() || value <= 0.0 || value > 1.0 {
            return Err(CalibrationParamError::new(
                "precision",
                value.to_string(),
                "must be a finite value in (0.0, 1.0]",
            ));
        }
        Ok(Self(value))
    }
}

impl fmt::Display for CalibrationPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<CalibrationPrecision> for f32 {
    fn from(value: CalibrationPrecision) -> Self {
        value.value()
    }
}

/// Number of probe queries used during calibration (must be `>= 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CalibrationQueryCount(u32);

impl CalibrationQueryCount {
    /// Default number of probe queries.
    pub const DEFAULT: Self = Self(50);

    /// Borrow inner scalar value.
    #[must_use]
    pub const fn value(self) -> u32 {
        self.0
    }
}

impl TryFrom<u32> for CalibrationQueryCount {
    type Error = CalibrationParamError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value == 0 {
            return Err(CalibrationParamError::new(
                "num_queries",
                value.to_string(),
                "must be greater than 0",
            ));
        }
        Ok(Self(value))
    }
}

impl fmt::Display for CalibrationQueryCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<CalibrationQueryCount> for u32 {
    fn from(value: CalibrationQueryCount) -> Self {
        value.value()
    }
}

/// Top-K value for recall measurement (must be `>= 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CalibrationTopK(u32);

impl CalibrationTopK {
    /// Default recall cut-off.
    pub const DEFAULT: Self = Self(10);

    /// Borrow inner scalar value.
    #[must_use]
    pub const fn value(self) -> u32 {
        self.0
    }
}

impl TryFrom<u32> for CalibrationTopK {
    type Error = CalibrationParamError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value == 0 {
            return Err(CalibrationParamError::new(
                "top_k",
                value.to_string(),
                "must be greater than 0",
            ));
        }
        Ok(Self(value))
    }
}

impl fmt::Display for CalibrationTopK {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<CalibrationTopK> for u32 {
    fn from(value: CalibrationTopK) -> Self {
        value.value()
    }
}

/// Parameters controlling the BQ1 calibration algorithm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationParams {
    /// Minimum acceptable recall (e.g. 0.99).
    pub target_recall: TargetRecall,
    /// Binary search convergence epsilon (e.g. 0.005).
    pub precision: CalibrationPrecision,
    /// Number of probe queries for calibration.
    pub num_queries: CalibrationQueryCount,
    /// Recall is measured at this K value.
    pub top_k: CalibrationTopK,
    /// Deterministic seed for probe query sampling.
    pub seed: u64,
}

impl Default for CalibrationParams {
    fn default() -> Self {
        Self {
            target_recall: TargetRecall::DEFAULT,
            precision: CalibrationPrecision::DEFAULT,
            num_queries: CalibrationQueryCount::DEFAULT,
            top_k: CalibrationTopK::DEFAULT,
            seed: 0xD4F6_59DE_0A97_2B4F,
        }
    }
}

/// Exponentially-weighted moving average tracker for online BQ1 refinement.
///
/// Tracks the observed skip rate across search queries to detect drift
/// from the calibrated baseline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmaState {
    /// Smoothing factor in `(0.0, 1.0]` — higher values weight recent observations more.
    pub alpha: f32,
    /// Current exponentially-weighted moving average of the skip rate.
    pub current_skip_rate: f32,
    /// Total number of search queries observed since last calibration.
    pub samples_seen: u64,
}

impl Default for EmaState {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            current_skip_rate: 0.0,
            samples_seen: 0,
        }
    }
}

impl EmaState {
    /// Update the EMA with a new observed skip rate.
    pub fn observe(&mut self, skip_rate: f32) {
        if self.samples_seen == 0 {
            self.current_skip_rate = skip_rate;
        } else {
            self.current_skip_rate = self
                .alpha
                .mul_add(skip_rate, (1.0 - self.alpha) * self.current_skip_rate);
        }
        self.samples_seen = self.samples_seen.saturating_add(1);
    }
}

/// Minimum number of EMA observations before drift detection activates.
const MIN_DRIFT_SAMPLES: u64 = 10;

impl CalibrationState {
    /// Observe a search's BQ1 skip rate and update the EMA tracker.
    ///
    /// Lazily initializes the EMA if not already present.
    pub fn observe_skip_rate(&mut self, observed_skip_rate: f32) {
        let ema = self.ema.get_or_insert_with(EmaState::default);
        ema.observe(observed_skip_rate);
    }

    /// Whether the EMA skip rate has drifted from the calibrated baseline.
    ///
    /// Requires at least [`MIN_DRIFT_SAMPLES`] observations before activating
    /// to avoid false positives during warm-up.
    #[must_use]
    pub fn is_drifted(&self, drift_tolerance: f32) -> bool {
        self.ema.as_ref().is_some_and(|ema| {
            ema.samples_seen >= MIN_DRIFT_SAMPLES
                && (self.skip_rate - ema.current_skip_rate).abs() > drift_tolerance
        })
    }

    /// Whether the EMA state should be persisted based on a query count interval.
    ///
    /// Returns `true` every `interval` observations to batch disk writes.
    #[must_use]
    pub fn should_persist_ema(&self, interval: u64) -> bool {
        self.ema
            .as_ref()
            .is_some_and(|ema| ema.samples_seen > 0 && ema.samples_seen % interval == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_state_serde_round_trip() {
        let state = CalibrationState {
            threshold: 0.55,
            recall_at_threshold: 0.99,
            skip_rate: 0.42,
            calibrated_at_ms: 1_700_000_000_000,
            dimension: 1024,
            corpus_size: 50_000,
            num_queries: 50,
            binary_search_steps: 8,
            ema: None,
        };

        let json = serde_json::to_string_pretty(&state).expect("serialization should succeed");
        let deserialized: CalibrationState =
            serde_json::from_str(&json).expect("deserialization should succeed");
        assert_eq!(state, deserialized);
    }

    #[test]
    fn calibration_state_camel_case_keys() {
        let state = CalibrationState {
            threshold: 0.5,
            recall_at_threshold: 1.0,
            skip_rate: 0.3,
            calibrated_at_ms: 0,
            dimension: 384,
            corpus_size: 1000,
            num_queries: 20,
            binary_search_steps: 7,
            ema: None,
        };

        let value = serde_json::to_value(&state).expect("serialization should succeed");
        assert!(value.get("calibratedAtMs").is_some());
        assert!(value.get("recallAtThreshold").is_some());
        assert!(value.get("skipRate").is_some());
        assert!(value.get("binarySearchSteps").is_some());
    }

    #[test]
    fn calibration_params_defaults() {
        let params = CalibrationParams::default();
        assert!((params.target_recall.value() - 0.99).abs() < f32::EPSILON);
        assert!((params.precision.value() - 0.005).abs() < f32::EPSILON);
        assert_eq!(params.num_queries.value(), 50);
        assert_eq!(params.top_k.value(), 10);
    }

    #[test]
    fn newtypes_reject_invalid_values() {
        assert!(TargetRecall::try_from(1.2).is_err());
        assert!(CalibrationPrecision::try_from(0.0).is_err());
        assert!(CalibrationQueryCount::try_from(0).is_err());
        assert!(CalibrationTopK::try_from(0).is_err());
    }

    #[test]
    fn ema_first_observation_sets_value() {
        let mut ema = EmaState::default();
        ema.observe(0.42);
        assert!((ema.current_skip_rate - 0.42).abs() < f32::EPSILON);
        assert_eq!(ema.samples_seen, 1);
    }

    #[test]
    fn ema_subsequent_observations_smooth() {
        let mut ema = EmaState {
            alpha: 0.5,
            current_skip_rate: 0.0,
            samples_seen: 0,
        };

        ema.observe(1.0); // First: sets directly to 1.0
        assert!((ema.current_skip_rate - 1.0).abs() < f32::EPSILON);

        ema.observe(0.0); // Second: 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        assert!((ema.current_skip_rate - 0.5).abs() < f32::EPSILON);

        ema.observe(0.0); // Third: 0.5 * 0.0 + 0.5 * 0.5 = 0.25
        assert!((ema.current_skip_rate - 0.25).abs() < f32::EPSILON);
        assert_eq!(ema.samples_seen, 3);
    }

    #[test]
    fn calibration_state_with_ema() {
        let state = CalibrationState {
            threshold: 0.55,
            recall_at_threshold: 0.99,
            skip_rate: 0.42,
            calibrated_at_ms: 1_700_000_000_000,
            dimension: 1024,
            corpus_size: 50_000,
            num_queries: 50,
            binary_search_steps: 8,
            ema: Some(EmaState {
                alpha: 0.1,
                current_skip_rate: 0.40,
                samples_seen: 100,
            }),
        };

        let json = serde_json::to_string_pretty(&state).expect("serialization should succeed");
        let deserialized: CalibrationState =
            serde_json::from_str(&json).expect("deserialization should succeed");
        assert_eq!(state, deserialized);
        assert!(deserialized.ema.is_some());
    }

    #[test]
    fn calibration_state_without_ema_omits_field() {
        let state = CalibrationState {
            threshold: 0.5,
            recall_at_threshold: 1.0,
            skip_rate: 0.3,
            calibrated_at_ms: 0,
            dimension: 384,
            corpus_size: 1000,
            num_queries: 20,
            binary_search_steps: 7,
            ema: None,
        };

        let json = serde_json::to_string(&state).expect("serialization should succeed");
        assert!(!json.contains("ema"));
    }

    #[test]
    fn observe_skip_rate_initializes_ema() {
        let mut state = CalibrationState {
            threshold: 0.55,
            recall_at_threshold: 0.99,
            skip_rate: 0.42,
            calibrated_at_ms: 0,
            dimension: 1024,
            corpus_size: 1000,
            num_queries: 50,
            binary_search_steps: 8,
            ema: None,
        };

        state.observe_skip_rate(0.40);
        let ema = state.ema.as_ref().expect("EMA should be initialized");
        assert!((ema.current_skip_rate - 0.40).abs() < f32::EPSILON);
        assert_eq!(ema.samples_seen, 1);
    }

    #[test]
    fn observe_skip_rate_updates_existing_ema() {
        let mut state = CalibrationState {
            threshold: 0.55,
            recall_at_threshold: 0.99,
            skip_rate: 0.42,
            calibrated_at_ms: 0,
            dimension: 1024,
            corpus_size: 1000,
            num_queries: 50,
            binary_search_steps: 8,
            ema: Some(EmaState {
                alpha: 0.5,
                current_skip_rate: 0.40,
                samples_seen: 5,
            }),
        };

        state.observe_skip_rate(0.50);
        let ema = state.ema.as_ref().expect("EMA should exist");
        // 0.5 * 0.50 + 0.5 * 0.40 = 0.45
        assert!((ema.current_skip_rate - 0.45).abs() < f32::EPSILON);
        assert_eq!(ema.samples_seen, 6);
    }

    #[test]
    fn drift_not_detected_with_few_samples() {
        let state = CalibrationState {
            threshold: 0.55,
            recall_at_threshold: 0.99,
            skip_rate: 0.42,
            calibrated_at_ms: 0,
            dimension: 1024,
            corpus_size: 1000,
            num_queries: 50,
            binary_search_steps: 8,
            ema: Some(EmaState {
                alpha: 0.1,
                current_skip_rate: 0.10, // very different, but too few samples
                samples_seen: 5,
            }),
        };
        assert!(!state.is_drifted(0.15));
    }

    #[test]
    fn drift_detected_when_skip_rate_diverges() {
        let state = CalibrationState {
            threshold: 0.55,
            recall_at_threshold: 0.99,
            skip_rate: 0.42,
            calibrated_at_ms: 0,
            dimension: 1024,
            corpus_size: 1000,
            num_queries: 50,
            binary_search_steps: 8,
            ema: Some(EmaState {
                alpha: 0.1,
                current_skip_rate: 0.20, // 0.22 difference > 0.15 tolerance
                samples_seen: 50,
            }),
        };
        assert!(state.is_drifted(0.15));
    }

    #[test]
    fn no_drift_when_skip_rate_is_close() {
        let state = CalibrationState {
            threshold: 0.55,
            recall_at_threshold: 0.99,
            skip_rate: 0.42,
            calibrated_at_ms: 0,
            dimension: 1024,
            corpus_size: 1000,
            num_queries: 50,
            binary_search_steps: 8,
            ema: Some(EmaState {
                alpha: 0.1,
                current_skip_rate: 0.38, // 0.04 difference < 0.15 tolerance
                samples_seen: 50,
            }),
        };
        assert!(!state.is_drifted(0.15));
    }

    #[test]
    fn no_drift_without_ema() {
        let state = CalibrationState {
            threshold: 0.55,
            recall_at_threshold: 0.99,
            skip_rate: 0.42,
            calibrated_at_ms: 0,
            dimension: 1024,
            corpus_size: 1000,
            num_queries: 50,
            binary_search_steps: 8,
            ema: None,
        };
        assert!(!state.is_drifted(0.15));
    }

    #[test]
    fn should_persist_ema_at_interval() {
        let state = CalibrationState {
            threshold: 0.55,
            recall_at_threshold: 0.99,
            skip_rate: 0.42,
            calibrated_at_ms: 0,
            dimension: 1024,
            corpus_size: 1000,
            num_queries: 50,
            binary_search_steps: 8,
            ema: Some(EmaState {
                alpha: 0.1,
                current_skip_rate: 0.40,
                samples_seen: 50,
            }),
        };
        assert!(state.should_persist_ema(50));
        assert!(!state.should_persist_ema(30));
    }

    #[test]
    fn should_not_persist_without_ema() {
        let state = CalibrationState {
            threshold: 0.55,
            recall_at_threshold: 0.99,
            skip_rate: 0.42,
            calibrated_at_ms: 0,
            dimension: 1024,
            corpus_size: 1000,
            num_queries: 50,
            binary_search_steps: 8,
            ema: None,
        };
        assert!(!state.should_persist_ema(50));
    }
}
