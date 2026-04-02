//! Local CLI command handlers.

pub mod calibrate;
pub mod clear;
pub mod estimate_storage;
pub mod index;
pub mod info;
pub mod init;
pub mod jobs;
pub mod reindex;
pub mod search;
pub mod status;

pub use calibrate::{CalibrateCommandInput, run_calibrate};
pub use clear::run_clear;
pub use estimate_storage::run_estimate_storage;
pub use index::run_index;
pub use info::run_info;
pub use init::run_init;
pub use jobs::{run_jobs_cancel, run_jobs_run, run_jobs_status};
pub use reindex::run_reindex;
pub use search::{SearchCommandInput, run_search, run_search_stdin_batch};
pub use status::run_status;
