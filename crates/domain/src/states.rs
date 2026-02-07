//! Domain state machine types.

use semantic_code_shared::ErrorEnvelope;
use serde::{Deserialize, Serialize};

/// High-level indexing state for orchestration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum IndexingState {
    /// Idle (no index in progress).
    Idle,
    /// Indexing in progress.
    Indexing,
    /// Indexing completed successfully.
    Ready,
    /// Indexing failed with a human-readable reason.
    Error {
        /// Human-readable reason for the failure.
        reason: Box<str>,
    },
}

/// Detailed indexing status for UI consumers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum IndexStatus {
    /// No index exists for the codebase.
    NotIndexed,
    /// Indexing in progress.
    Indexing,
    /// Indexing completed successfully.
    Indexed,
    /// Indexing stopped due to resource limits.
    LimitReached,
    /// Indexing failed with an error envelope.
    Failed {
        /// Error envelope describing the failure.
        error: ErrorEnvelope,
    },
}

/// Progress events emitted during indexing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProgressEvent {
    /// Progress update for a named phase.
    Progress {
        /// Phase identifier (e.g. "scan").
        phase: Box<str>,
        /// Current item count.
        current: u64,
        /// Total item count.
        total: u64,
        /// Completion percentage (0-100).
        percentage: u8,
    },
    /// Index status update.
    Status {
        /// Current status snapshot.
        status: IndexStatus,
    },
}

impl ProgressEvent {
    /// Build a progress event with computed percentage.
    #[must_use]
    pub fn progress(phase: impl AsRef<str>, current: u64, total: u64) -> Self {
        Self::Progress {
            phase: phase.as_ref().to_owned().into_boxed_str(),
            current,
            total,
            percentage: progress_percentage(current, total),
        }
    }

    /// Build a status event.
    #[must_use]
    pub const fn status(status: IndexStatus) -> Self {
        Self::Status { status }
    }
}

fn progress_percentage(current: u64, total: u64) -> u8 {
    if total == 0 {
        return 0;
    }
    let capped = if current > total { total } else { current };
    let percent = (capped.saturating_mul(100)) / total;
    u8::try_from(percent).unwrap_or(u8::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_shared::{ErrorCode, ErrorEnvelope};
    use std::error::Error;

    #[test]
    fn progress_event_computes_percentage() {
        let event = ProgressEvent::progress("scan", 1, 2);
        let percentage = match event {
            ProgressEvent::Progress { percentage, .. } => Some(percentage),
            ProgressEvent::Status { .. } => None,
        };
        assert_eq!(percentage, Some(50));
    }

    #[test]
    fn status_and_progress_serialization_shape() -> Result<(), Box<dyn Error>> {
        let status = IndexStatus::Failed {
            error: ErrorEnvelope::expected(ErrorCode::invalid_input(), "bad input"),
        };
        let event = ProgressEvent::status(status);

        let value = serde_json::to_value(&event)?;
        let expected = serde_json::json!({
            "type": "status",
            "status": {
                "status": "failed",
                "error": {
                    "kind": "Expected",
                    "class": "NonRetriable",
                    "code": { "namespace": "core", "code": "invalid_input" },
                    "message": "bad input"
                }
            }
        });
        assert_eq!(value, expected);
        Ok(())
    }
}
