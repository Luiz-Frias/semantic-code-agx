//! Print the effective backend config (defaults + env overrides) as JSON.

use semantic_code_config::{BackendConfig, BackendEnv, apply_env_overrides};
use std::io;
use std::io::Write;

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::ExitCode::from(1)
        },
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let env = BackendEnv::from_std_env()?;
    let config = apply_env_overrides(BackendConfig::default(), &env)?;

    let mut output = serde_json::to_string_pretty(config.as_ref())?;
    output.push('\n');

    let mut stdout = io::stdout();
    stdout.write_all(output.as_bytes())?;
    stdout.flush()?;

    Ok(())
}
