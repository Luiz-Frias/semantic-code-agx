//! CLI integration tests for external adapters.

use std::io;
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const DEFAULT_FIXTURE_REPO: &str = "tmp/dspy/dspy";
const ENV_FIXTURE_REPO: &str = "SCA_E2E_FIXTURE_REPO";
const DEFAULT_ONNX_REPO_SLUG: &str = "Xenova-all-MiniLM-L6-v2";
const DEFAULT_MILVUS_ADDRESS: &str = "127.0.0.1:19530";
const MILVUS_READY_RETRIES: u32 = 120;
const MILVUS_PROXY_RETRY_ATTEMPTS: u32 = 15;
const MILVUS_PROXY_RETRY_DELAY: Duration = Duration::from_secs(2);

fn milvus_enabled() -> bool {
    cfg!(feature = "milvus-grpc") || cfg!(feature = "milvus-rest")
}

/// Quick probe: try a single TCP connect with a short timeout.
/// Returns `true` if Milvus gRPC port is reachable right now.
fn milvus_reachable(address: &str) -> bool {
    let normalized = normalize_milvus_address(address);
    let Ok(mut addrs) = normalized.to_socket_addrs() else {
        return false;
    };
    let Some(addr) = addrs.next() else {
        return false;
    };
    TcpStream::connect_timeout(&addr, Duration::from_secs(2)).is_ok()
}

fn run_cli_in_dir_with_env(
    dir: &Path,
    args: &[&str],
    envs: &[(String, String)],
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
        "VOYAGE_BASE_URL",
        "VOYAGE_MODEL",
        "OLLAMA_HOST",
        "OLLAMA_MODEL",
    ]
}

fn assert_command_success(output: &std::process::Output, label: &str) {
    if output.status.success() {
        return;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    panic!("{label} failed\nstdout:\n{stdout}\nstderr:\n{stderr}");
}

fn is_milvus_proxy_not_ready(output: &std::process::Output) -> bool {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    stdout.contains("Milvus Proxy is not ready yet")
        || stderr.contains("Milvus Proxy is not ready yet")
}

fn run_cli_with_milvus_proxy_retry(
    dir: &Path,
    args: &[&str],
    envs: &[(String, String)],
) -> io::Result<std::process::Output> {
    for attempt in 1..=MILVUS_PROXY_RETRY_ATTEMPTS {
        let output = run_cli_in_dir_with_env(dir, args, envs)?;
        if output.status.success() {
            return Ok(output);
        }
        if !is_milvus_proxy_not_ready(&output) || attempt == MILVUS_PROXY_RETRY_ATTEMPTS {
            return Ok(output);
        }
        std::thread::sleep(MILVUS_PROXY_RETRY_DELAY);
    }
    run_cli_in_dir_with_env(dir, args, envs)
}

fn fixture_repo_root() -> io::Result<PathBuf> {
    if let Ok(value) = std::env::var(ENV_FIXTURE_REPO) {
        return Ok(resolve_repo_path(&value));
    }
    if let Some(value) = read_env_value(ENV_FIXTURE_REPO)? {
        return Ok(resolve_repo_path(&value));
    }
    Ok(resolve_repo_path(DEFAULT_FIXTURE_REPO))
}

fn resolve_repo_path(value: &str) -> PathBuf {
    let trimmed = value.trim();
    let candidate = PathBuf::from(trimmed);
    if candidate.is_absolute() {
        candidate
    } else {
        workspace_root().join(candidate)
    }
}

fn copy_fixture_repo() -> io::Result<PathBuf> {
    let source = fixture_repo_root()?;
    if !source.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "fixture repo missing at {}; set {} or update .env",
                source.display(),
                ENV_FIXTURE_REPO
            ),
        ));
    }
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dest = std::env::temp_dir().join(format!("sca-cli-external-{unique}"));
    copy_dir_recursive(&source, &dest)?;
    Ok(dest)
}

fn copy_fixture_repo_or_skip() -> io::Result<Option<PathBuf>> {
    match copy_fixture_repo() {
        Ok(path) => Ok(Some(path)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            eprintln!("skipping integration test: {error}");
            Ok(None)
        },
        Err(error) => Err(error),
    }
}

fn copy_dir_recursive(source: &Path, dest: &Path) -> io::Result<()> {
    std::fs::create_dir_all(dest)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        if entry.file_name() == std::ffi::OsStr::new(".context") {
            continue;
        }
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

fn workspace_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_dir.to_path_buf())
}

fn env_path() -> PathBuf {
    workspace_root().join(".env")
}

fn read_env_value(key: &str) -> io::Result<Option<String>> {
    let path = env_path();
    let contents = match std::fs::read_to_string(&path) {
        Ok(value) => value,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((left, right)) = line.split_once('=') else {
            continue;
        };
        if left.trim() == key {
            return Ok(Some(unquote_env_value(right)));
        }
    }
    Ok(None)
}

fn unquote_env_value(value: &str) -> String {
    let trimmed = value.trim();
    if let Some(stripped) = trimmed
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
    {
        return stripped.to_string();
    }
    if let Some(stripped) = trimmed
        .strip_prefix('\'')
        .and_then(|value| value.strip_suffix('\''))
    {
        return stripped.to_string();
    }
    trimmed.to_string()
}

fn onnx_model_dir() -> io::Result<PathBuf> {
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

    Err(io::Error::other(format!(
        "ONNX assets missing under {}. Expected {} or {} and {}. Set SCA_EMBEDDING_ONNX_MODEL_DIR or download Xenova/all-MiniLM-L6-v2 with the hf CLI.",
        dir.display(),
        nested.display(),
        root.display(),
        tokenizer.display()
    )))
}

fn normalize_milvus_address(address: &str) -> String {
    let trimmed = address.trim();
    if let Some(value) = trimmed.strip_prefix("http://") {
        return value.to_string();
    }
    if let Some(value) = trimmed.strip_prefix("https://") {
        return value.to_string();
    }
    trimmed.to_string()
}

fn require_milvus_ready(address: &str) -> io::Result<()> {
    let normalized = normalize_milvus_address(address);
    let mut addrs = normalized.to_socket_addrs()?;
    let Some(addr) = addrs.next() else {
        return Err(io::Error::other(format!(
            "invalid Milvus address: {address}"
        )));
    };

    // Phase 1: Wait for TCP on gRPC port (19530).
    let retries = MILVUS_READY_RETRIES;
    for attempt in 1..=retries {
        match TcpStream::connect_timeout(&addr, Duration::from_secs(2)) {
            Ok(_) => break,
            Err(error) => {
                if attempt == retries {
                    return Err(io::Error::other(format!(
                        "Milvus not reachable at {address}: {error}. Start it with `just milvus-up` or `docker compose up -d`."
                    )));
                }
                std::thread::sleep(Duration::from_secs(1));
            },
        }
    }

    // Phase 2: Wait for Proxy gRPC readiness via HTTP health endpoint (port 9091).
    // TCP on 19530 opens before the gRPC Proxy is initialized; /healthz returns
    // 200 only once the Proxy can actually serve requests.
    let health_addr: std::net::SocketAddr = "127.0.0.1:9091"
        .parse()
        .map_err(|err| io::Error::other(format!("bad health address: {err}")))?;
    for attempt in 1..=retries {
        if check_milvus_health(&health_addr).is_ok() {
            return Ok(());
        }
        if attempt == retries {
            return Err(io::Error::other(
                "Milvus gRPC Proxy not ready (healthz on :9091 did not return 200). \
                 Start it with `just milvus-up` or `docker compose up -d`."
                    .to_string(),
            ));
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    Ok(())
}

/// Sends a minimal HTTP/1.1 GET /healthz to the given address and returns Ok(())
/// if the response status line contains "200".
fn check_milvus_health(addr: &std::net::SocketAddr) -> io::Result<()> {
    use std::io::{Read, Write};
    let mut stream = TcpStream::connect_timeout(addr, Duration::from_secs(2))?;
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    stream
        .write_all(b"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1:9091\r\nConnection: close\r\n\r\n")?;
    let mut buf = [0u8; 256];
    let n = stream.read(&mut buf)?;
    let response = String::from_utf8_lossy(&buf[..n]);
    if response.contains("200") {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "healthz returned non-200: {response}"
        )))
    }
}

fn config_fixture_path(relative: &str) -> PathBuf {
    workspace_root()
        .join("crates")
        .join("testkit")
        .join("fixtures")
        .join(relative)
}

#[test]
fn cli_index_search_with_onnx_and_local_vdb() -> io::Result<()> {
    let Some(temp) = copy_fixture_repo_or_skip()? else {
        return Ok(());
    };
    let model_dir = onnx_model_dir()?;

    let envs = vec![
        ("SCA_EMBEDDING_PROVIDER".to_string(), "onnx".to_string()),
        (
            "SCA_EMBEDDING_ONNX_MODEL_DIR".to_string(),
            model_dir.to_string_lossy().to_string(),
        ),
        (
            "SCA_EMBEDDING_ONNX_DOWNLOAD".to_string(),
            "false".to_string(),
        ),
        ("SCA_VECTOR_DB_PROVIDER".to_string(), "local".to_string()),
    ];

    let output = run_cli_in_dir_with_env(&temp, &["index", "--init"], &envs)?;
    assert_command_success(&output, "index");

    let output = run_cli_in_dir_with_env(&temp, &["search", "--query", "local-index"], &envs)?;
    assert_command_success(&output, "search");

    Ok(())
}

#[test]
fn cli_index_search_with_milvus_and_auto_embedding() -> io::Result<()> {
    if !milvus_enabled() || !milvus_reachable(DEFAULT_MILVUS_ADDRESS) {
        eprintln!("skipping: milvus feature not enabled or not reachable");
        return Ok(());
    }
    let Some(temp) = copy_fixture_repo_or_skip()? else {
        return Ok(());
    };
    require_milvus_ready(DEFAULT_MILVUS_ADDRESS)?;

    let model_dir = onnx_model_dir()?;
    let envs = vec![
        (
            "SCA_EMBEDDING_ONNX_MODEL_DIR".to_string(),
            model_dir.to_string_lossy().to_string(),
        ),
        (
            "SCA_EMBEDDING_ONNX_DOWNLOAD".to_string(),
            "false".to_string(),
        ),
        ("SCA_EMBEDDING_LOCAL_FIRST".to_string(), "true".to_string()),
    ];

    let config = config_fixture_path("config/backend-config.milvus-external.toml");
    let config_arg = config.to_string_lossy().to_string();

    let output = run_cli_with_milvus_proxy_retry(
        &temp,
        &["index", "--init", "--config", &config_arg],
        &envs,
    )?;
    assert_command_success(&output, "index");

    let output = run_cli_with_milvus_proxy_retry(
        &temp,
        &["search", "--query", "local-index", "--config", &config_arg],
        &envs,
    )?;
    assert_command_success(&output, "search");

    Ok(())
}

#[test]
fn cli_index_search_with_openai_and_milvus() -> io::Result<()> {
    if !milvus_enabled() || !milvus_reachable(DEFAULT_MILVUS_ADDRESS) {
        eprintln!("skipping: milvus feature not enabled or not reachable");
        return Ok(());
    }
    let Some(temp) = copy_fixture_repo_or_skip()? else {
        return Ok(());
    };
    require_milvus_ready(DEFAULT_MILVUS_ADDRESS)?;

    let api_key = match std::env::var("OPENAI_API_KEY") {
        Ok(value) if !value.trim().is_empty() => value,
        _ => {
            eprintln!("skipping OpenAI external test; OPENAI_API_KEY not set");
            return Ok(());
        },
    };

    let config = config_fixture_path("config/backend-config.milvus-external.toml");
    let config_arg = config.to_string_lossy().to_string();

    let envs = vec![
        ("SCA_EMBEDDING_PROVIDER".to_string(), "openai".to_string()),
        ("OPENAI_API_KEY".to_string(), api_key),
    ];

    let output = run_cli_with_milvus_proxy_retry(
        &temp,
        &["index", "--init", "--config", &config_arg],
        &envs,
    )?;
    assert_command_success(&output, "index");

    let output = run_cli_with_milvus_proxy_retry(
        &temp,
        &["search", "--query", "local-index", "--config", &config_arg],
        &envs,
    )?;
    assert_command_success(&output, "search");

    Ok(())
}
