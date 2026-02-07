//! Phase 07 observability E2E tests.

use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn run_cli_in_dir_with_env(
    dir: &Path,
    args: &[&str],
    envs: &[(&str, &str)],
) -> io::Result<std::process::Output> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_sca"));
    command.current_dir(dir).args(args);
    scrub_scoped_env(&mut command);
    for (key, value) in envs {
        command.env(key, value);
    }
    command.output()
}

fn scrub_scoped_env(command: &mut Command) {
    for (key, _) in std::env::vars() {
        if key.starts_with("SCA_") {
            command.env_remove(key);
        }
    }
    for key in embedding_env_aliases() {
        command.env_remove(key);
    }
    for key in provider_env_keys() {
        command.env_remove(key);
    }
}

fn embedding_env_aliases() -> &'static [&'static str] {
    &[
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "EMBEDDING_TIMEOUT_MS",
        "EMBEDDING_BATCH_SIZE",
        "EMBEDDING_DIMENSION",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_LOCAL_FIRST",
        "EMBEDDING_LOCAL_ONLY",
        "EMBEDDING_ROUTING_MODE",
        "EMBEDDING_SPLIT_MAX_REMOTE_BATCHES",
        "EMBEDDING_JOBS_PROGRESS_INTERVAL_MS",
        "EMBEDDING_JOBS_CANCEL_POLL_INTERVAL_MS",
        "EMBEDDING_TEST_FALLBACK",
        "EMBEDDING_ONNX_MODEL_DIR",
        "EMBEDDING_ONNX_MODEL_FILENAME",
        "EMBEDDING_ONNX_TOKENIZER_FILENAME",
        "EMBEDDING_ONNX_REPO",
        "EMBEDDING_ONNX_DOWNLOAD",
        "EMBEDDING_ONNX_SESSION_POOL_SIZE",
        "EMBEDDING_CACHE_ENABLED",
        "EMBEDDING_CACHE_MAX_ENTRIES",
        "EMBEDDING_CACHE_MAX_BYTES",
        "EMBEDDING_CACHE_DISK_ENABLED",
        "EMBEDDING_CACHE_DISK_PATH",
        "EMBEDDING_CACHE_DISK_PROVIDER",
        "EMBEDDING_CACHE_DISK_CONNECTION",
        "EMBEDDING_CACHE_DISK_TABLE",
        "EMBEDDING_CACHE_DISK_MAX_BYTES",
        "EMBEDDING_API_KEY",
    ]
}

fn provider_env_keys() -> &'static [&'static str] {
    &[
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "GEMINI_API_KEY",
        "GEMINI_BASE_URL",
        "GEMINI_MODEL",
        "VOYAGE_API_KEY",
        "VOYAGE_API_URL",
        "VOYAGEAI_API_KEY",
        "VOYAGEAI_BASE_URL",
        "VOYAGEAI_MODEL",
        "OLLAMA_HOST",
        "OLLAMA_MODEL",
    ]
}

fn workspace_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_dir.to_path_buf())
}

fn fixture_path(relative: &str) -> PathBuf {
    workspace_root()
        .join("crates")
        .join("testkit")
        .join("fixtures")
        .join(relative)
}

fn copy_fixture_repo(relative: &str) -> io::Result<PathBuf> {
    let source = fixture_path(relative);
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dest = std::env::temp_dir().join(format!("sca-e2e-observability-{unique}"));
    copy_dir_recursive(&source, &dest)?;
    Ok(dest)
}

fn copy_dir_recursive(source: &Path, dest: &Path) -> io::Result<()> {
    std::fs::create_dir_all(dest)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let source_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_recursive(&source_path, &dest_path)?;
        } else if file_type.is_file() {
            std::fs::copy(&source_path, &dest_path)?;
        }
    }
    Ok(())
}

#[test]
fn phase7_observability_logs_and_metrics() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;

    let output = run_cli_in_dir_with_env(
        &temp,
        &["index", "--init"],
        &[("SCA_EMBEDDING_PROVIDER", "test")],
    )?;
    assert!(output.status.success(), "index --init failed");

    let output = run_cli_in_dir_with_env(
        &temp,
        &["search", "--query", "local-index"],
        &[
            ("SCA_EMBEDDING_PROVIDER", "test"),
            ("SCA_LOG_FORMAT", "json"),
            ("SCA_TELEMETRY_FORMAT", "json"),
            ("SCA_TRACE_SAMPLE_RATE", "1.0"),
        ],
    )?;
    assert!(output.status.success(), "search failed");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut saw_event = false;
    let mut saw_metric = false;
    for line in stderr.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if value.get("event").and_then(|event| event.as_str()).is_some() {
            saw_event = true;
        }
        if value
            .get("type")
            .and_then(|ty| ty.as_str())
            .is_some_and(|ty| ty == "metric")
        {
            saw_metric = true;
        }
    }

    assert!(saw_event, "expected structured log event in stderr");
    assert!(saw_metric, "expected telemetry metric in stderr");
    Ok(())
}
