//! Tracing subscriber initialization for the CLI.

use crate::format::OutputArgs;
use crate::redact_layer::RedactingStderrMakeWriter;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Initialize JSON tracing for the CLI process.
///
/// Behavior:
/// - If `RUST_LOG` is set, it takes precedence.
/// - Otherwise, uses the CLI `--log-level` value.
/// - Initialization failures are ignored so startup remains non-fatal.
pub fn init_tracing(output: &OutputArgs) {
    let filter = build_filter(output);
    let format_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_writer(RedactingStderrMakeWriter)
        .with_current_span(true)
        .with_span_list(true);

    let subscriber = tracing_subscriber::registry()
        .with(filter)
        .with(format_layer);

    let _ = subscriber.try_init();
}

fn build_filter(output: &OutputArgs) -> EnvFilter {
    std::env::var_os(EnvFilter::DEFAULT_ENV).map_or_else(
        || EnvFilter::new(output.log_level.as_scoped_directive()),
        |_| {
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(output.log_level.as_scoped_directive()))
        },
    )
}
