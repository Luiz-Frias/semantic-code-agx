//! Print the effective backend config (defaults + env overrides) as JSON.

use semantic_code_config::load_backend_config_std_env;
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
    let config = load_backend_config_std_env(None, None)?;

    let mut output = serde_json::to_string_pretty(config.as_ref())?;
    output.push('\n');

    let mut stdout = io::stdout();
    stdout.write_all(output.as_bytes())?;
    stdout.flush()?;

    Ok(())
}
