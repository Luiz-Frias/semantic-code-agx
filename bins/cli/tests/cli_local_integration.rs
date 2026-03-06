//! Integration tests for local CLI commands.

use std::io;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_ONNX_REPO_SLUG: &str = "Xenova-all-MiniLM-L6-v2";

fn sanitize_provider_env(command: &mut Command) {
    for key in [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "GEMINI_API_KEY",
        "GEMINI_BASE_URL",
        "GEMINI_MODEL",
        "VOYAGE_API_KEY",
        "VOYAGE_BASE_URL",
        "VOYAGE_MODEL",
        "OLLAMA_MODEL",
        "OLLAMA_HOST",
        "SCA_EMBEDDING_ONNX_MODEL_DIR",
        "SCA_EMBEDDING_ONNX_MODEL_FILENAME",
        "SCA_EMBEDDING_ONNX_TOKENIZER_FILENAME",
        "SCA_EMBEDDING_ONNX_REPO",
        "SCA_EMBEDDING_ONNX_DOWNLOAD",
        "SCA_EMBEDDING_ONNX_SESSION_POOL_SIZE",
        "EMBEDDING_ONNX_MODEL_DIR",
        "EMBEDDING_ONNX_MODEL_FILENAME",
        "EMBEDDING_ONNX_TOKENIZER_FILENAME",
        "EMBEDDING_ONNX_REPO",
        "EMBEDDING_ONNX_DOWNLOAD",
        "EMBEDDING_ONNX_SESSION_POOL_SIZE",
    ] {
        command.env_remove(key);
    }
}

fn run_cli_in_dir(dir: &Path, args: &[&str]) -> io::Result<std::process::Output> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_sca"));
    command
        .current_dir(dir)
        .env("SCA_EMBEDDING_PROVIDER", "onnx");
    sanitize_provider_env(&mut command);

    if requires_onnx_model(dir, args) {
        let onnx_model_dir = onnx_model_dir()?;
        command
            .env(
                "SCA_EMBEDDING_ONNX_MODEL_DIR",
                onnx_model_dir.to_string_lossy().to_string(),
            )
            .env("SCA_EMBEDDING_ONNX_DOWNLOAD", "false");
    }

    command.args(args).output()
}

fn requires_onnx_model(dir: &Path, args: &[&str]) -> bool {
    if args
        .iter()
        .any(|arg| matches!(*arg, "index" | "reindex" | "clear"))
    {
        return true;
    }
    if !args.iter().any(|arg| *arg == "search") {
        return false;
    }
    dir.join(".context").join("manifest.json").is_file()
}

fn run_cli_in_dir_or_skip(dir: &Path, args: &[&str]) -> io::Result<Option<std::process::Output>> {
    match run_cli_in_dir(dir, args) {
        Ok(output) => Ok(Some(output)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            eprintln!("skipping local CLI integration test: {error}");
            Ok(None)
        },
        Err(error) => Err(error),
    }
}

fn run_cli_in_dir_with_env(
    dir: &Path,
    args: &[&str],
    extra_env: &[(&str, &str)],
) -> io::Result<std::process::Output> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_sca"));
    command
        .current_dir(dir)
        .env("SCA_EMBEDDING_PROVIDER", "onnx");
    sanitize_provider_env(&mut command);
    if requires_onnx_model(dir, args) {
        let onnx_model_dir = onnx_model_dir()?;
        command
            .env(
                "SCA_EMBEDDING_ONNX_MODEL_DIR",
                onnx_model_dir.to_string_lossy().to_string(),
            )
            .env("SCA_EMBEDDING_ONNX_DOWNLOAD", "false");
    }
    for (key, value) in extra_env {
        command.env(key, value);
    }
    command.args(args).output()
}

fn run_cli_in_dir_with_custom_env(
    dir: &Path,
    args: &[&str],
    env: &[(&str, &str)],
) -> io::Result<std::process::Output> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_sca"));
    command.current_dir(dir);
    sanitize_provider_env(&mut command);
    for (key, value) in env {
        command.env(key, value);
    }
    command.args(args).output()
}

fn onnx_model_dir() -> io::Result<std::path::PathBuf> {
    let dir = workspace_root()
        .join(".context")
        .join("models")
        .join("onnx")
        .join(DEFAULT_ONNX_REPO_SLUG);
    let nested = dir.join("onnx").join("model.onnx");
    let root = dir.join("model.onnx");
    let tokenizer = dir.join("tokenizer.json");

    if (nested.exists() || root.exists()) && tokenizer.exists() {
        return Ok(dir);
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!(
            "ONNX assets missing under {}. Expected {} or {} and {}. Set SCA_EMBEDDING_ONNX_MODEL_DIR or download Xenova/all-MiniLM-L6-v2 with the hf CLI.",
            dir.display(),
            nested.display(),
            root.display(),
            tokenizer.display()
        ),
    ))
}

fn workspace_root() -> std::path::PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_dir.to_path_buf())
}

fn fixture_path(relative: &str) -> std::path::PathBuf {
    workspace_root()
        .join("crates")
        .join("testkit")
        .join("fixtures")
        .join(relative)
}

fn golden_fixture_path(relative: &str) -> std::path::PathBuf {
    workspace_root()
        .join("crates")
        .join("testkit")
        .join("fixtures")
        .join("cli-output")
        .join(relative)
}

fn copy_fixture_repo(relative: &str) -> io::Result<std::path::PathBuf> {
    let source = fixture_path(relative);
    let dest = create_unique_fixture_dir("sca-cli-fixture")?;
    copy_dir_recursive(&source, &dest)?;
    Ok(dest)
}

fn create_unique_fixture_dir(prefix: &str) -> io::Result<std::path::PathBuf> {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    let pid = std::process::id();
    for _ in 0..64 {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let seq = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let candidate = std::env::temp_dir().join(format!("{prefix}-{pid}-{nanos}-{seq}"));
        match std::fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "failed to allocate unique fixture temp directory",
    ))
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

const SEARCH_SCORE_EPSILON: f64 = 1e-6;

fn json_f64_field(value: &serde_json::Value, field: &str) -> io::Result<f64> {
    value.get(field).and_then(|v| v.as_f64()).ok_or_else(|| {
        io::Error::other(format!(
            "missing or non-f64 field `{field}` in value: {value}"
        ))
    })
}

fn assert_json_search_result_close(
    actual: &serde_json::Value,
    expected: &serde_json::Value,
    score_field: &str,
) -> io::Result<()> {
    assert_eq!(actual.get("content"), expected.get("content"));
    assert_eq!(actual.get("language"), expected.get("language"));
    assert_eq!(actual.get("key"), expected.get("key"));

    let actual_score = json_f64_field(actual, score_field)?;
    let expected_score = json_f64_field(expected, score_field)?;
    assert!(
        (actual_score - expected_score).abs() <= SEARCH_SCORE_EPSILON,
        "score mismatch for field `{score_field}`: actual={actual_score}, expected={expected_score}"
    );
    Ok(())
}

fn assert_ndjson_search_result_close(
    actual: &serde_json::Value,
    expected: &serde_json::Value,
) -> io::Result<()> {
    assert_eq!(
        actual.get("type"),
        Some(&serde_json::Value::String("result".into()))
    );
    assert_eq!(actual.get("type"), expected.get("type"));
    assert_eq!(actual.get("content"), expected.get("content"));
    assert_eq!(actual.get("relativePath"), expected.get("relativePath"));
    assert_eq!(actual.get("startLine"), expected.get("startLine"));
    assert_eq!(actual.get("endLine"), expected.get("endLine"));

    let actual_score = json_f64_field(actual, "score")?;
    let expected_score = json_f64_field(expected, "score")?;
    assert!(
        (actual_score - expected_score).abs() <= SEARCH_SCORE_EPSILON,
        "ndjson score mismatch: actual={actual_score}, expected={expected_score}"
    );
    Ok(())
}

#[test]
fn cli_index_init_creates_manifest_and_gitignore() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    std::fs::write(temp.join(".gitignore"), "target/\n")?;

    let Some(output) = run_cli_in_dir_or_skip(&temp, &["index", "--init"])? else {
        return Ok(());
    };
    assert!(output.status.success(), "index --init failed");
    assert!(temp.join(".context/manifest.json").is_file());
    assert!(temp.join(".context/config.toml").is_file());
    let gitignore = std::fs::read_to_string(temp.join(".gitignore")).unwrap_or_default();
    assert!(gitignore.lines().any(|line| line.trim() == ".context/"));
    Ok(())
}

#[test]
fn cli_init_creates_manifest_and_config() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let output = run_cli_in_dir(&temp, &["init"])?;
    assert!(output.status.success(), "init failed");
    assert!(temp.join(".context/manifest.json").is_file());
    assert!(temp.join(".context/config.toml").is_file());
    Ok(())
}

#[test]
fn cli_index_auto_loads_context_config() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let context_dir = temp.join(".context");
    std::fs::create_dir_all(&context_dir)?;
    std::fs::write(
        context_dir.join("config.toml"),
        "version = 1\n\n[vectorDb]\nindexMode = \"hybrid\"\n",
    )?;

    let Some(output) = run_cli_in_dir_or_skip(&temp, &["index", "--init"])? else {
        return Ok(());
    };
    assert!(output.status.success(), "index --init failed");

    let manifest_path = temp.join(".context/manifest.json");
    let manifest = std::fs::read_to_string(manifest_path)?;
    let manifest_json: serde_json::Value =
        serde_json::from_str(&manifest).map_err(io::Error::other)?;
    assert_eq!(
        manifest_json
            .get("indexMode")
            .and_then(|value| value.as_str()),
        Some("hybrid")
    );
    Ok(())
}

#[test]
fn cli_search_errors_without_manifest() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let output = run_cli_in_dir(&temp, &["search", "--query", "needle"])?;
    assert_eq!(output.status.code(), Some(2));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("manifest"),
        "expected manifest error in stdout"
    );
    Ok(())
}

#[test]
fn cli_status_errors_without_manifest() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let output = run_cli_in_dir(&temp, &["status"])?;
    assert_eq!(output.status.code(), Some(2));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("manifest"),
        "expected manifest error in stdout"
    );
    Ok(())
}

#[test]
fn cli_search_json_output_schema() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let Some(output) = run_cli_in_dir_or_skip(&temp, &["index", "--init"])? else {
        return Ok(());
    };
    assert!(output.status.success(), "index --init failed");
    let output = run_cli_in_dir(&temp, &["--json", "search", "--query", "local-index"])?;
    let value: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(value.get("status").and_then(|v| v.as_str()), Some("ok"));
    assert!(value.get("results").and_then(|v| v.as_array()).is_some());
    assert_eq!(
        value
            .get("vectorKernel")
            .and_then(|value| value.get("effective"))
            .and_then(|value| value.as_str()),
        Some("hnsw-rs")
    );
    Ok(())
}

#[test]
fn cli_search_ndjson_output_schema() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let Some(output) = run_cli_in_dir_or_skip(&temp, &["index", "--init"])? else {
        return Ok(());
    };
    assert!(output.status.success(), "index --init failed");
    let output = run_cli_in_dir(
        &temp,
        &["--output", "ndjson", "search", "--query", "local-index"],
    )?;
    assert!(output.status.success(), "search failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = stdout.lines();
    let last = lines
        .next_back()
        .ok_or_else(|| io::Error::other("missing summary"))?;
    let summary: serde_json::Value = serde_json::from_str(last).map_err(io::Error::other)?;
    assert_eq!(
        summary.get("type").and_then(|v| v.as_str()),
        Some("summary")
    );
    assert_eq!(summary.get("status").and_then(|v| v.as_str()), Some("ok"));
    assert_eq!(
        summary
            .get("vectorKernel")
            .and_then(|value| value.get("effective"))
            .and_then(|value| value.as_str()),
        Some("hnsw-rs")
    );
    Ok(())
}

#[test]
fn cli_search_reads_query_from_stdin() -> io::Result<()> {
    use std::io::Write;
    use std::process::Stdio;

    let temp = copy_fixture_repo("local-index/basic")?;
    let Some(output) = run_cli_in_dir_or_skip(&temp, &["index", "--init"])? else {
        return Ok(());
    };
    assert!(output.status.success(), "index --init failed");

    let mut command = Command::new(env!("CARGO_BIN_EXE_sca"));
    sanitize_provider_env(&mut command);
    let mut child = command
        .current_dir(&temp)
        .env("SCA_EMBEDDING_PROVIDER", "onnx")
        .env(
            "SCA_EMBEDDING_ONNX_MODEL_DIR",
            onnx_model_dir()?.to_string_lossy().to_string(),
        )
        .env("SCA_EMBEDDING_ONNX_DOWNLOAD", "false")
        .args(["search", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;

    if let Some(stdin) = child.stdin.as_mut() {
        stdin.write_all(b"local-index\n")?;
    }

    let output = child.wait_with_output()?;
    assert!(output.status.success(), "search --stdin failed");
    Ok(())
}

#[test]
fn cli_search_output_matches_golden_fixtures() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let Some(output) = run_cli_in_dir_or_skip(&temp, &["index", "--init"])? else {
        return Ok(());
    };
    assert!(output.status.success(), "index --init failed");

    let expected_json = std::fs::read_to_string(golden_fixture_path("search.basic.json"))?;
    let expected_json: serde_json::Value =
        serde_json::from_str(&expected_json).map_err(io::Error::other)?;
    let output = run_cli_in_dir(
        &temp,
        &["--output", "json", "search", "--query", "local-index"],
    )?;
    assert!(output.status.success(), "search --output json failed");
    let actual_json: serde_json::Value =
        serde_json::from_slice(&output.stdout).map_err(io::Error::other)?;
    assert_eq!(actual_json.get("status"), expected_json.get("status"));
    let actual_results = actual_json
        .get("results")
        .and_then(|value| value.as_array())
        .ok_or_else(|| io::Error::other("missing `results` array in actual json output"))?;
    let expected_results = expected_json
        .get("results")
        .and_then(|value| value.as_array())
        .ok_or_else(|| io::Error::other("missing `results` array in expected json fixture"))?;
    assert_eq!(actual_results.len(), expected_results.len());
    for (actual, expected) in actual_results.iter().zip(expected_results.iter()) {
        assert_json_search_result_close(actual, expected, "score")?;
    }
    assert_eq!(
        actual_json
            .get("vectorKernel")
            .and_then(|value| value.get("effective"))
            .and_then(|value| value.as_str()),
        Some("hnsw-rs")
    );

    let expected_ndjson = std::fs::read_to_string(golden_fixture_path("search.basic.ndjson"))?;
    let expected_lines: Vec<serde_json::Value> = expected_ndjson
        .lines()
        .map(|line| serde_json::from_str(line).map_err(io::Error::other))
        .collect::<Result<_, _>>()?;
    let output = run_cli_in_dir(
        &temp,
        &["--output", "ndjson", "search", "--query", "local-index"],
    )?;
    assert!(output.status.success(), "search --output ndjson failed");
    let actual_lines: Vec<serde_json::Value> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| serde_json::from_str(line).map_err(io::Error::other))
        .collect::<Result<_, _>>()?;
    assert_eq!(actual_lines.len(), expected_lines.len());
    for (actual, expected) in actual_lines.iter().zip(expected_lines.iter()) {
        let expected_type = expected.get("type").and_then(|value| value.as_str());
        if expected_type == Some("summary") {
            assert_eq!(actual.get("type"), expected.get("type"));
            assert_eq!(actual.get("status"), expected.get("status"));
            assert_eq!(actual.get("count"), expected.get("count"));
            assert_eq!(
                actual
                    .get("vectorKernel")
                    .and_then(|value| value.get("effective"))
                    .and_then(|value| value.as_str()),
                Some("hnsw-rs")
            );
        } else if expected_type == Some("result") {
            assert_ndjson_search_result_close(actual, expected)?;
        } else {
            assert_eq!(actual, expected);
        }
    }
    Ok(())
}

#[test]
fn cli_index_json_output_includes_vector_kernel() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let Some(output) = run_cli_in_dir_or_skip(&temp, &["--output", "json", "index", "--init"])?
    else {
        return Ok(());
    };
    assert!(output.status.success(), "index --output json --init failed");
    let value: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(
        value.get("status").and_then(|value| value.as_str()),
        Some("ok")
    );
    assert_eq!(
        value
            .get("vectorKernel")
            .and_then(|value| value.get("effective"))
            .and_then(|value| value.as_str()),
        Some("hnsw-rs")
    );
    Ok(())
}

#[test]
fn cli_agent_scripted_flow_emits_ndjson() -> io::Result<()> {
    use std::io::Write;
    use std::process::Stdio;

    let temp = copy_fixture_repo("local-index/basic")?;
    let Some(output) = run_cli_in_dir_or_skip(&temp, &["--agent", "index", "--init"])? else {
        return Ok(());
    };
    assert!(output.status.success(), "agent index failed");

    let summary: serde_json::Value =
        serde_json::from_slice(&output.stdout).map_err(io::Error::other)?;
    assert_eq!(
        summary.get("type").and_then(|value| value.as_str()),
        Some("summary")
    );
    assert_eq!(
        summary.get("status").and_then(|value| value.as_str()),
        Some("ok")
    );

    let mut command = Command::new(env!("CARGO_BIN_EXE_sca"));
    sanitize_provider_env(&mut command);
    let mut child = command
        .current_dir(&temp)
        .env("SCA_EMBEDDING_PROVIDER", "onnx")
        .env(
            "SCA_EMBEDDING_ONNX_MODEL_DIR",
            onnx_model_dir()?.to_string_lossy().to_string(),
        )
        .env("SCA_EMBEDDING_ONNX_DOWNLOAD", "false")
        .args(["--agent", "search", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;

    if let Some(stdin) = child.stdin.as_mut() {
        stdin.write_all(b"local-index\n")?;
    }

    let output = child.wait_with_output()?;
    assert!(output.status.success(), "agent search failed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut saw_result = false;
    let mut summary_line: Option<serde_json::Value> = None;
    for line in stdout.lines() {
        let value: serde_json::Value = serde_json::from_str(line).map_err(io::Error::other)?;
        match value.get("type").and_then(|value| value.as_str()) {
            Some("result") => saw_result = true,
            Some("summary") => summary_line = Some(value),
            _ => {},
        }
    }

    let summary = summary_line.ok_or_else(|| io::Error::other("missing summary"))?;
    assert_eq!(
        summary.get("status").and_then(|value| value.as_str()),
        Some("ok")
    );
    assert!(saw_result, "expected at least one result line");
    Ok(())
}

#[test]
fn cli_clear_removes_vector_snapshot() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let Some(output) = run_cli_in_dir_or_skip(&temp, &["index", "--init"])? else {
        return Ok(());
    };
    assert!(output.status.success(), "index --init failed");

    let manifest_path = temp.join(".context/manifest.json");
    let manifest = std::fs::read_to_string(manifest_path)?;
    let manifest_json: serde_json::Value =
        serde_json::from_str(&manifest).map_err(io::Error::other)?;
    let collection = manifest_json
        .get("collectionName")
        .and_then(|value| value.as_str())
        .ok_or_else(|| io::Error::other("manifest missing collectionName"))?;
    let snapshot_path = temp
        .join(".context")
        .join("vector")
        .join("collections")
        .join(format!("{collection}.json"));
    assert!(
        snapshot_path.is_file(),
        "expected vector snapshot after index"
    );

    let output = run_cli_in_dir(&temp, &["clear"])?;
    assert!(output.status.success(), "clear failed");
    assert!(
        !snapshot_path.exists(),
        "expected vector snapshot removed after clear"
    );
    Ok(())
}

#[test]
fn cli_reindex_runs_after_index() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let Some(output) = run_cli_in_dir_or_skip(&temp, &["index", "--init"])? else {
        return Ok(());
    };
    assert!(output.status.success(), "index --init failed");

    let output = run_cli_in_dir(&temp, &["reindex"])?;
    assert!(output.status.success(), "reindex failed");
    Ok(())
}

#[test]
fn cli_reindex_json_output_includes_vector_kernel() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let Some(output) = run_cli_in_dir_or_skip(&temp, &["index", "--init"])? else {
        return Ok(());
    };
    assert!(output.status.success(), "index --init failed");

    let output = run_cli_in_dir(&temp, &["--output", "json", "reindex"])?;
    assert!(output.status.success(), "reindex --output json failed");
    let value: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(
        value.get("status").and_then(|value| value.as_str()),
        Some("ok")
    );
    assert_eq!(
        value
            .get("vectorKernel")
            .and_then(|value| value.get("effective"))
            .and_then(|value| value.as_str()),
        Some("hnsw-rs")
    );
    Ok(())
}

#[test]
fn cli_invalid_vector_kernel_override_rejected_for_index_search_reindex() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let cases: [Vec<&str>; 3] = [
        vec![
            "search",
            "--query",
            "needle",
            "--vector-kernel",
            "not-a-kernel",
        ],
        vec!["index", "--init", "--vector-kernel", "not-a-kernel"],
        vec!["reindex", "--vector-kernel", "not-a-kernel"],
    ];

    for args in &cases {
        let output =
            run_cli_in_dir_with_custom_env(&temp, args, &[("SCA_EMBEDDING_PROVIDER", "test")])?;
        assert_eq!(
            output.status.code(),
            Some(2),
            "expected invalid input for args: {args:?}"
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("status: error"),
            "expected error payload for args: {args:?}"
        );
        assert!(
            stdout.contains("not-a-kernel"),
            "expected invalid kernel token in error payload for args: {args:?}"
        );
    }

    Ok(())
}

#[test]
fn cli_status_reports_after_index() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let Some(output) = run_cli_in_dir_or_skip(&temp, &["index", "--init"])? else {
        return Ok(());
    };
    assert!(output.status.success(), "index --init failed");

    let output = run_cli_in_dir(&temp, &["status"])?;
    assert!(output.status.success(), "status failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("status: ok"));
    Ok(())
}

#[test]
fn cli_emits_structured_logs_when_enabled() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let Some(output) = run_cli_in_dir_or_skip(&temp, &["index", "--init"])? else {
        return Ok(());
    };
    assert!(output.status.success(), "index --init failed");

    let output = run_cli_in_dir_with_env(
        &temp,
        &["search", "--query", "local-index"],
        &[("SCA_LOG_FORMAT", "json"), ("SCA_TELEMETRY_FORMAT", "json")],
    )?;
    assert!(output.status.success(), "search failed");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut saw_event = false;
    let mut saw_correlation = false;
    for line in stderr.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if value
            .get("event")
            .and_then(|event| event.as_str())
            .is_some()
        {
            saw_event = true;
        }
        if let Some(fields) = value.get("fields").and_then(|fields| fields.as_object()) {
            if fields.contains_key("correlationId") {
                saw_correlation = true;
            }
        }
    }

    assert!(saw_event, "expected structured log event in stderr");
    assert!(
        saw_correlation,
        "expected correlationId to be present in structured logs"
    );
    Ok(())
}

#[test]
fn cli_estimate_storage_runs_without_manifest() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let output = run_cli_in_dir(&temp, &["estimate-storage"])?;
    assert!(output.status.success(), "estimate-storage failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("status: ok"));
    assert!(stdout.contains("kind: estimateStorage"));
    Ok(())
}

#[test]
fn cli_estimate_storage_json_schema() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;
    let output = run_cli_in_dir(&temp, &["--output", "json", "estimate-storage"])?;
    assert!(
        output.status.success(),
        "estimate-storage --output json failed"
    );
    let payload: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(
        payload.get("status").and_then(|value| value.as_str()),
        Some("ok")
    );
    assert_eq!(
        payload.get("kind").and_then(|value| value.as_str()),
        Some("estimateStorage")
    );
    assert!(
        payload
            .get("requiredFreeBytes")
            .and_then(|v| v.as_u64())
            .is_some()
    );
    assert!(
        payload
            .get("estimatedBytesHigh")
            .and_then(|v| v.as_u64())
            .is_some()
    );
    Ok(())
}

#[test]
fn cli_estimate_storage_danger_flag_changes_safety_factor() -> io::Result<()> {
    let temp = copy_fixture_repo("local-index/basic")?;

    let normal = run_cli_in_dir(&temp, &["--output", "json", "estimate-storage"])?;
    assert!(normal.status.success(), "normal estimate-storage failed");
    let normal_json: serde_json::Value = serde_json::from_slice(&normal.stdout)?;
    assert_eq!(
        normal_json
            .get("safetyFactor")
            .and_then(|value| value.as_str()),
        Some("2.00")
    );

    let danger = run_cli_in_dir(
        &temp,
        &[
            "--output",
            "json",
            "estimate-storage",
            "--danger-close-storage",
        ],
    )?;
    assert!(danger.status.success(), "danger estimate-storage failed");
    let danger_json: serde_json::Value = serde_json::from_slice(&danger.stdout)?;
    assert_eq!(
        danger_json
            .get("safetyFactor")
            .and_then(|value| value.as_str()),
        Some("1.25")
    );
    Ok(())
}
