//! BQ1 calibration boundary contract.

use crate::BoxFuture;
use semantic_code_domain::{CalibrationParams, CalibrationState};
use semantic_code_shared::{RequestContext, Result};

/// Port for BQ1 Hamming prefilter threshold calibration.
///
/// The adapter runs a binary search over BQ1 thresholds against the
/// current index, measuring recall vs a baseline (no BQ1) to find the
/// optimal threshold that maximizes candidate skip rate while maintaining
/// the target recall.
pub trait CalibrationPort: Send + Sync {
    /// Run binary search calibration against the current index.
    ///
    /// Returns the optimal `CalibrationState` containing the calibrated
    /// threshold, measured recall, and skip rate.
    fn calibrate_bq1(
        &self,
        ctx: &RequestContext,
        params: &CalibrationParams,
    ) -> BoxFuture<'_, Result<CalibrationState>>;
}
