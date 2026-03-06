//! Search diagnostics metadata.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Optional vector-search diagnostics surfaced to API consumers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchStats {
    /// Kernel-specific expansion count, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expansions: Option<u64>,
    /// Effective kernel used for search.
    pub kernel: Box<str>,
    /// Kernel-specific extended metrics (e.g. DFRR pulls, splits, bucket utilization).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<Box<str>, f64>,
    /// Wall-clock nanoseconds spent in the kernel search algorithm.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_search_duration_ns: Option<u64>,
    /// Number of active vectors in the index at search time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_size: Option<u64>,
}
