//! Machine-readable agent protocol spec (YAML output).
//!
//! Generates a structured YAML document describing the full CLI contract
//! for AI agent consumers. Output is deterministic (`BTreeMap` key ordering)
//! and parseable by Python `yaml.safe_load`.

use crate::error::{CliError, ExitCode};
use crate::output::CliOutput;
use semantic_code_facade::build_info;
use serde::Serialize;
use std::collections::BTreeMap;

// ── Protocol version ─────────────────────────────────────────────────────────

/// Bump when the agent-doc schema changes in a breaking way.
const PROTOCOL_VERSION: &str = "agx/v1";

// ── Known command scope names ────────────────────────────────────────────────

const KNOWN_COMMANDS: &[&str] = &[
    "calibrate",
    "clear",
    "config",
    "estimate-storage",
    "index",
    "info",
    "init",
    "jobs",
    "reindex",
    "search",
    "status",
];

// ── Top-level protocol structs ───────────────────────────────────────────────

#[derive(Serialize)]
struct AgentProtocol {
    protocol: &'static str,
    binary: &'static str,
    version: String,
    output_modes: OutputModes,
    commands: BTreeMap<&'static str, CommandContract>,
    ndjson_shapes: BTreeMap<&'static str, NdjsonShape>,
    exit_codes: BTreeMap<u8, &'static str>,
    recovery: BTreeMap<&'static str, &'static str>,
    filter_syntax: FilterSyntax,
    workflows: BTreeMap<&'static str, Vec<&'static str>>,
    known_gaps: Vec<&'static str>,
}

/// Scoped view: protocol header + single command + shared context.
#[derive(Serialize)]
struct ScopedAgentProtocol {
    protocol: &'static str,
    binary: &'static str,
    version: String,
    commands: BTreeMap<&'static str, CommandContract>,
    ndjson_shapes: BTreeMap<&'static str, NdjsonShape>,
    exit_codes: BTreeMap<u8, &'static str>,
    recovery: BTreeMap<&'static str, &'static str>,
}

// ── Component structs ────────────────────────────────────────────────────────

#[derive(Serialize)]
struct OutputModes {
    default: &'static str,
    flags: Vec<&'static str>,
    precedence: &'static str,
}

#[derive(Serialize, Clone)]
struct CommandContract {
    purpose: &'static str,
    required_flags: Vec<&'static str>,
    common_flags: Vec<&'static str>,
    success_signal: SuccessSignal,
    error_codes: Vec<&'static str>,
    idempotent: bool,
    background: bool,
}

#[derive(Serialize, Clone)]
struct SuccessSignal {
    ndjson_type: &'static str,
    description: &'static str,
}

#[derive(Serialize, Clone)]
struct NdjsonShape {
    type_field: &'static str,
    fields: BTreeMap<&'static str, &'static str>,
}

#[derive(Serialize)]
struct FilterSyntax {
    syntax: &'static str,
    fields: Vec<&'static str>,
    operators: Vec<&'static str>,
    examples: Vec<&'static str>,
}

// ── Public handler ───────────────────────────────────────────────────────────

/// Generate the agent protocol doc, optionally scoped to a single command.
pub fn run_agent_doc(command_scope: Option<&str>) -> Result<CliOutput, CliError> {
    if let Some(scope) = command_scope
        && !KNOWN_COMMANDS.contains(&scope)
    {
        return Err(CliError::InvalidInput(format!(
            "unknown command '{scope}'. Valid commands: {}",
            KNOWN_COMMANDS.join(", ")
        )));
    }

    let yaml = match command_scope {
        Some(scope) => build_scoped_yaml(scope)?,
        None => build_full_yaml()?,
    };

    Ok(CliOutput {
        stdout: yaml,
        stderr: String::new(),
        exit_code: ExitCode::Ok,
    })
}

// ── YAML builders ────────────────────────────────────────────────────────────

fn build_full_yaml() -> Result<String, CliError> {
    let build = build_info();
    let protocol = AgentProtocol {
        protocol: PROTOCOL_VERSION,
        binary: "sca",
        version: build.version.to_owned(),
        output_modes: build_output_modes(),
        commands: build_all_commands(),
        ndjson_shapes: build_ndjson_shapes(),
        exit_codes: build_exit_codes(),
        recovery: build_recovery_table(),
        filter_syntax: build_filter_syntax(),
        workflows: build_workflows(),
        known_gaps: vec![
            "filterExpr is validated at the request layer but not yet forwarded to vector queries",
            "includeContent is validated but not yet forwarded through the search pipeline",
        ],
    };
    serde_yaml_ng::to_string(&protocol)
        .map_err(|err| CliError::InvalidInput(format!("agent-doc serialization: {err}")))
}

fn build_scoped_yaml(scope: &str) -> Result<String, CliError> {
    let build = build_info();
    let all_commands = build_all_commands();
    let mut scoped_commands = BTreeMap::new();
    if let Some(contract) = all_commands.get(scope) {
        scoped_commands.insert(scope_to_key(scope), contract.clone());
    }

    let protocol = ScopedAgentProtocol {
        protocol: PROTOCOL_VERSION,
        binary: "sca",
        version: build.version.to_owned(),
        commands: scoped_commands,
        ndjson_shapes: build_ndjson_shapes(),
        exit_codes: build_exit_codes(),
        recovery: build_recovery_table(),
    };
    serde_yaml_ng::to_string(&protocol)
        .map_err(|err| CliError::InvalidInput(format!("agent-doc serialization: {err}")))
}

/// Map scope strings to their static key equivalents for `BTreeMap` insertion.
fn scope_to_key(scope: &str) -> &'static str {
    KNOWN_COMMANDS
        .iter()
        .find(|&&k| k == scope)
        .copied()
        .unwrap_or_else(|| scope_to_key_fallback())
}

/// Fallback for `scope_to_key` — returns a safe static reference.
///
/// This path is unreachable because we validate the scope against
/// `KNOWN_COMMANDS` before calling `build_scoped_yaml`, but the type
/// system requires a `&'static str` for the `BTreeMap` key.
const fn scope_to_key_fallback() -> &'static str {
    "unknown"
}

// ── Output modes ─────────────────────────────────────────────────────────────

fn build_output_modes() -> OutputModes {
    OutputModes {
        default: "text",
        flags: vec![
            "--output text|json|ndjson",
            "--agent  (alias: --output ndjson --no-progress)",
            "--json   (legacy alias, hidden)",
        ],
        precedence: "--output > --json > --agent > default text",
    }
}

// ── Command contracts ────────────────────────────────────────────────────────

fn build_all_commands() -> BTreeMap<&'static str, CommandContract> {
    let mut commands = BTreeMap::new();
    insert_setup_commands(&mut commands);
    insert_operational_commands(&mut commands);
    insert_pipeline_commands(&mut commands);
    commands
}

/// Setup commands: info, config, init, estimate-storage.
fn insert_setup_commands(commands: &mut BTreeMap<&'static str, CommandContract>) {
    commands.insert(
        "info",
        CommandContract {
            purpose: "Show build and version details",
            required_flags: vec![],
            common_flags: vec![],
            success_signal: SuccessSignal {
                ndjson_type: "summary",
                description: "kind:'info', status:'ok'",
            },
            error_codes: vec![],
            idempotent: true,
            background: false,
        },
    );
    commands.insert(
        "config",
        CommandContract {
            purpose: "Validate runtime config and environment (subcommands: check, show, validate)",
            required_flags: vec![],
            common_flags: vec!["--path <config-file>", "--overrides-json <json>"],
            success_signal: SuccessSignal {
                ndjson_type: "summary",
                description: "kind:'config', status:'ok'",
            },
            error_codes: vec!["ERR_CONFIG_*"],
            idempotent: true,
            background: false,
        },
    );
    commands.insert(
        "init",
        CommandContract {
            purpose: "Initialize config and manifest for a codebase",
            required_flags: vec![],
            common_flags: vec![
                "--config <path>",
                "--codebase-root <path>",
                "--storage-mode <disabled|project|custom:path>",
                "--force",
            ],
            success_signal: SuccessSignal {
                ndjson_type: "summary",
                description: "kind:'init', status:'ok'",
            },
            error_codes: vec!["ERR_CONFIG_*", "ERR_STORAGE_*"],
            idempotent: false,
            background: false,
        },
    );
    commands.insert(
        "estimate-storage",
        CommandContract {
            purpose: "Estimate index storage requirements and local free-space headroom",
            required_flags: vec![],
            common_flags: vec![
                "--codebase-root <path>",
                "--config <path>",
                "--danger-close-storage",
            ],
            success_signal: SuccessSignal {
                ndjson_type: "summary",
                description: "kind:'estimate_storage', status:'ok'",
            },
            error_codes: vec!["ERR_CONFIG_*", "ERR_STORAGE_*"],
            idempotent: true,
            background: false,
        },
    );
}

/// Operational commands: jobs, calibrate.
fn insert_operational_commands(commands: &mut BTreeMap<&'static str, CommandContract>) {
    commands.insert(
        "jobs",
        CommandContract {
            purpose: "Background job management (subcommands: status <job_id>, cancel <job_id>)",
            required_flags: vec!["<JOB_ID> (positional)"],
            common_flags: vec!["--codebase-root <path>"],
            success_signal: SuccessSignal {
                ndjson_type: "job_status",
                description: "job.state: queued|running|completed|failed|cancelled",
            },
            error_codes: vec!["ERR_CORE_NOT_FOUND"],
            idempotent: true,
            background: false,
        },
    );
    commands.insert(
        "calibrate",
        CommandContract {
            purpose: "Calibrate BQ1 threshold for the local DFRR kernel",
            required_flags: vec![],
            common_flags: vec![
                "--target-recall <f32>",
                "--precision <f32>",
                "--num-queries <u32>",
                "--top-k <u32>",
                "--codebase-root <path>",
            ],
            success_signal: SuccessSignal {
                ndjson_type: "summary",
                description: "kind:'calibrate', status:'ok'",
            },
            error_codes: vec!["ERR_CORE_INVALID_INPUT", "ERR_VECTOR_*"],
            idempotent: true,
            background: false,
        },
    );
}

/// Core pipeline commands: index, search, reindex, clear, status.
fn insert_pipeline_commands(commands: &mut BTreeMap<&'static str, CommandContract>) {
    commands.insert(
        "index",
        CommandContract {
            purpose: "Index the local codebase (scan, split, embed, insert into vector DB)",
            required_flags: vec![],
            common_flags: vec![
                "--init",
                "--codebase-root <path>",
                "--config <path>",
                "--background",
                "--embedding-provider <onnx|openai|gemini|voyage|ollama>",
                "--vector-kernel <hnsw-rs|dfrr>",
                "--overrides-json <json>",
            ],
            success_signal: SuccessSignal {
                ndjson_type: "summary",
                description: "kind:'index', status:'ok', with indexStatus field",
            },
            error_codes: vec![
                "ERR_CONFIG_*",
                "ERR_STORAGE_INSUFFICIENT_FREE_SPACE",
                "ERR_CORE_INVALID_INPUT",
                "ERR_VECTOR_*",
            ],
            idempotent: false,
            background: true,
        },
    );
    commands.insert(
        "search",
        CommandContract {
            purpose: "Semantic similarity search against the indexed codebase",
            required_flags: vec!["--query <text> | --stdin | --stdin-batch"],
            common_flags: vec![
                "--top-k <u32> (default: 5)",
                "--threshold <f32> (default: 0.0)",
                "--filter-expr <expr>",
                "--include-content",
                "--codebase-root <path>",
                "--overrides-json <json>",
            ],
            success_signal: SuccessSignal {
                ndjson_type: "result",
                description: "streamed 'result' lines + final 'summary' with status:'ok'",
            },
            error_codes: vec![
                "ERR_CORE_INVALID_INPUT",
                "ERR_CONFIG_INVALID_FILTER_EXPR",
                "ERR_VECTOR_*",
                "ERR_CORE_NOT_FOUND",
            ],
            idempotent: true,
            background: false,
        },
    );
    commands.insert("reindex", CommandContract {
        purpose: "Reindex based on Merkle-detected snapshot changes (selective re-embed + upsert)",
        required_flags: vec![],
        common_flags: vec![
            "--codebase-root <path>", "--config <path>", "--background",
            "--embedding-provider <onnx|openai|gemini|voyage|ollama>", "--overrides-json <json>",
        ],
        success_signal: SuccessSignal { ndjson_type: "summary", description: "kind:'reindex', status:'ok'" },
        error_codes: vec!["ERR_CORE_INVALID_INPUT"],
        idempotent: true,
        background: true,
    });
    commands.insert(
        "clear",
        CommandContract {
            purpose: "Clear the local index and snapshot",
            required_flags: vec![],
            common_flags: vec!["--codebase-root <path>", "--config <path>"],
            success_signal: SuccessSignal {
                ndjson_type: "summary",
                description: "kind:'clear', status:'ok'",
            },
            error_codes: vec!["ERR_VECTOR_*"],
            idempotent: true,
            background: false,
        },
    );
    commands.insert(
        "status",
        CommandContract {
            purpose: "Report local index status (collection info, record count, snapshot state)",
            required_flags: vec![],
            common_flags: vec!["--codebase-root <path>", "--config <path>"],
            success_signal: SuccessSignal {
                ndjson_type: "summary",
                description: "kind:'status', status:'ok'",
            },
            error_codes: vec!["ERR_CORE_NOT_FOUND"],
            idempotent: true,
            background: false,
        },
    );
}

// ── NDJSON shapes ────────────────────────────────────────────────────────────

fn build_ndjson_shapes() -> BTreeMap<&'static str, NdjsonShape> {
    let mut shapes = BTreeMap::new();

    let mut summary_fields = BTreeMap::new();
    summary_fields.insert("status", "ok | error");
    summary_fields.insert(
        "kind",
        "string (command-specific: index, search, reindex, clear, etc.)",
    );
    shapes.insert(
        "summary",
        NdjsonShape {
            type_field: "summary",
            fields: summary_fields,
        },
    );

    let mut result_fields = BTreeMap::new();
    result_fields.insert("relativePath", "string");
    result_fields.insert("startLine", "u32");
    result_fields.insert("endLine", "u32");
    result_fields.insert("score", "f64");
    result_fields.insert("content", "string | null");
    shapes.insert(
        "result",
        NdjsonShape {
            type_field: "result",
            fields: result_fields,
        },
    );

    let mut job_fields = BTreeMap::new();
    job_fields.insert("job.id", "uuid");
    job_fields.insert("job.kind", "index | reindex");
    job_fields.insert(
        "job.state",
        "queued | running | completed | failed | cancelled",
    );
    shapes.insert(
        "job_status",
        NdjsonShape {
            type_field: "job_status",
            fields: job_fields,
        },
    );

    let mut error_fields = BTreeMap::new();
    error_fields.insert("error.code", "ERR_* (see recovery table)");
    error_fields.insert("error.message", "string");
    error_fields.insert("error.kind", "EXPECTED | INVARIANT");
    error_fields.insert("error.meta", "object | null");
    shapes.insert(
        "error",
        NdjsonShape {
            type_field: "error",
            fields: error_fields,
        },
    );

    shapes
}

// ── Exit codes ───────────────────────────────────────────────────────────────

fn build_exit_codes() -> BTreeMap<u8, &'static str> {
    let mut codes = BTreeMap::new();
    codes.insert(0, "success");
    codes.insert(1, "internal / invariant / serialization");
    codes.insert(2, "invalid input (includes API EXPECTED kind)");
    codes.insert(3, "IO error");
    codes
}

// ── Recovery table ───────────────────────────────────────────────────────────

fn build_recovery_table() -> BTreeMap<&'static str, &'static str> {
    let mut recovery = BTreeMap::new();
    recovery.insert("ERR_CONFIG_*", "Fix request/env/config. Do not retry.");
    recovery.insert(
        "ERR_CORE_INVALID_INPUT",
        "Run `sca index --init` if manifest is missing.",
    );
    recovery.insert(
        "ERR_CORE_NOT_FOUND",
        "Resource not found. Verify job ID or ensure index exists.",
    );
    recovery.insert(
        "ERR_CORE_TIMEOUT / ERR_CORE_IO",
        "Retry with bounded backoff and jitter.",
    );
    recovery.insert("ERR_DOMAIN_*", "Fix request/env/config. Do not retry.");
    recovery.insert(
        "ERR_STORAGE_INSUFFICIENT_FREE_SPACE",
        "Free disk space or reduce scope. Do not retry.",
    );
    recovery.insert(
        "ERR_VECTOR_INVALID_FILTER_EXPR",
        "Fix filter expression syntax (see filter_syntax).",
    );
    recovery.insert(
        "ERR_VECTOR_SNAPSHOT_*",
        "Rebuild index: `sca clear` then `sca index --init`.",
    );
    recovery.insert(
        "ERR_VECTOR_VDB_CONNECTION / ERR_VECTOR_VDB_TIMEOUT",
        "Retry with bounded backoff. Verify SCA_VECTOR_DB_* env.",
    );
    recovery.insert("INVARIANT kind", "Internal error. Escalate — do not retry.");
    recovery
}

// ── Filter syntax ────────────────────────────────────────────────────────────

fn build_filter_syntax() -> FilterSyntax {
    FilterSyntax {
        syntax: "<field> <op> <quoted_value>",
        fields: vec!["relativePath", "language", "fileExtension"],
        operators: vec!["==", "!="],
        examples: vec![
            "relativePath == 'src/main.rs'",
            "language != 'rust'",
            "fileExtension == 'rs'",
        ],
    }
}

// ── Workflows ────────────────────────────────────────────────────────────────

fn build_workflows() -> BTreeMap<&'static str, Vec<&'static str>> {
    let mut workflows = BTreeMap::new();
    workflows.insert(
        "fresh_repo",
        vec![
            "sca config check",
            "sca index --init --codebase-root <repo>",
            "sca search --codebase-root <repo> --query '<q>' --top-k 5",
        ],
    );
    workflows.insert(
        "incremental",
        vec![
            "sca reindex --codebase-root <repo>",
            "sca search --codebase-root <repo> --query '<q>' --top-k 5",
        ],
    );
    workflows.insert(
        "background_index",
        vec![
            "sca index --init --background --codebase-root <repo>",
            "sca jobs status <job_id>",
            "sca search --codebase-root <repo> --query '<q>' --top-k 5",
        ],
    );
    workflows
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_output_is_valid_yaml() -> Result<(), Box<dyn std::error::Error>> {
        let output = run_agent_doc(None)?;
        assert_eq!(output.exit_code, ExitCode::Ok);

        // Parse the YAML to verify it's structurally valid.
        let parsed: serde_yaml_ng::Value = serde_yaml_ng::from_str(&output.stdout)?;
        let mapping = parsed.as_mapping().ok_or("top-level should be a mapping")?;

        // Verify top-level keys exist.
        let has_key =
            |key: &str| mapping.contains_key(&serde_yaml_ng::Value::String(key.to_owned()));
        assert!(has_key("protocol"), "missing 'protocol' key");
        assert!(has_key("commands"), "missing 'commands' key");
        assert!(has_key("ndjson_shapes"), "missing 'ndjson_shapes' key");
        assert!(has_key("exit_codes"), "missing 'exit_codes' key");
        assert!(has_key("recovery"), "missing 'recovery' key");
        Ok(())
    }

    #[test]
    fn scoped_to_search_contains_search() -> Result<(), Box<dyn std::error::Error>> {
        let output = run_agent_doc(Some("search"))?;
        assert!(
            output.stdout.contains("search"),
            "scoped output should contain 'search'"
        );
        Ok(())
    }

    #[test]
    fn scoped_to_search_omits_index() -> Result<(), Box<dyn std::error::Error>> {
        let output = run_agent_doc(Some("search"))?;
        let parsed: serde_yaml_ng::Value = serde_yaml_ng::from_str(&output.stdout)?;

        let commands = parsed
            .as_mapping()
            .and_then(|m| m.get(&serde_yaml_ng::Value::String("commands".to_owned())))
            .and_then(serde_yaml_ng::Value::as_mapping);

        if let Some(cmds) = commands {
            assert!(
                !cmds.contains_key(&serde_yaml_ng::Value::String("index".to_owned())),
                "scoped output should not contain 'index' command"
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_unknown_command() {
        let result = run_agent_doc(Some("foobar"));
        assert!(result.is_err(), "should reject unknown command");
        if let Err(error) = result {
            assert_eq!(error.exit_code(), ExitCode::InvalidInput);
            assert!(error.to_string().contains("foobar"));
        }
    }
}
