//! # semantic-code-app
//!
//! Application use cases for indexing and search.
//! This crate depends on `ports`, `domain`, and `shared`.

mod calibrate_bq1;
mod clear_index;
mod index_codebase;
mod reindex_by_change;
mod semantic_search;

/// Generated FSM definitions for indexing pipeline.
mod generated {
    include!(concat!(env!("OUT_DIR"), "/index_pipeline_fsm.rs"));
}

/// Placeholder module for application use cases.
mod placeholder {
    /// Placeholder function to verify the crate compiles.
    #[must_use]
    pub const fn app_crate_version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}

pub use calibrate_bq1::{
    Bq1Observation, CalibrateBq1Deps, CalibrateBq1Input, CalibrateBq1Output, calibrate_bq1,
    observe_bq1_search,
};
pub use clear_index::{ClearIndexDeps, ClearIndexInput, clear_index};
pub use generated::{INDEX_PIPELINE_STATES, INDEX_PIPELINE_TRANSITIONS, IndexPipelineState};
pub use index_codebase::{
    EmbedFunctionStats, EmbedStageStats, FunctionTimingStats, IndexCodebaseDeps,
    IndexCodebaseInput, IndexCodebaseOutput, IndexCodebaseStatus, IndexProgress, IndexStageStats,
    InsertFunctionStats, InsertStageStats, PrepareFunctionStats, PrepareStageStats,
    ScanFunctionStats, ScanStageStats, SplitFunctionStats, SplitStageStats, index_codebase,
};
pub use placeholder::app_crate_version;
pub use reindex_by_change::{
    ReindexByChangeDeps, ReindexByChangeInput, ReindexByChangeOutput, reindex_by_change,
};
pub use semantic_search::{
    SemanticSearchDeps, SemanticSearchInput, SemanticSearchOutput, semantic_search,
};

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_code_domain::domain_crate_version;
    use semantic_code_ports::ports_crate_version;
    use semantic_code_shared::shared_crate_version;

    #[test]
    fn app_crate_compiles() {
        let version = app_crate_version();
        assert!(!version.is_empty());
    }

    #[test]
    fn app_can_use_ports_domain_shared() {
        let ports_version = ports_crate_version();
        let domain_version = domain_crate_version();
        let shared_version = shared_crate_version();

        assert!(!ports_version.is_empty());
        assert!(!domain_version.is_empty());
        assert!(!shared_version.is_empty());
    }
}
