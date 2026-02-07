//! Ignore matcher boundary contract.

/// Input to ignore matching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IgnoreMatchInput {
    /// Normalized ignore patterns (Phase 03 rules).
    pub ignore_patterns: Vec<Box<str>>,
    /// Candidate path relative to codebase root (normalized separators).
    pub relative_path: Box<str>,
}

/// Boundary contract for ignore matching.
pub trait IgnorePort: Send + Sync {
    /// Returns true when the path should be ignored.
    fn is_ignored(&self, input: &IgnoreMatchInput) -> bool;
}
