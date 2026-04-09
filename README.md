# semantic-code-agx

[![CI](https://github.com/Luiz-Frias/semantic-code-agx/workflows/CI/badge.svg)](https://github.com/Luiz-Frias/semantic-code-agx/actions)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.96%2B-orange.svg)](https://www.rust-lang.org/)

> **Semantic code search engine** powered by embeddings and vector databases. Find code by meaning, not just keywords.

Primary command: `sca` (alias: `semantic-code`).

## Why This?

Traditional code search (grep, ripgrep) finds exact matches. Semantic search understands meaning:

- **Ask questions**: "How is authentication handled?" → finds auth code
- **Explore unfamiliar codebases**: "database connection setup" → finds DB init
- **Find patterns**: "error handling and recovery" → finds Result/error flows

## Features

- **Embedding providers**: ONNX (local), OpenAI, Gemini, Voyage, Ollama, Apple Neural Engine (feature-gated)
- **Local vector kernels**: HNSW by default, experimental DFRR, and exact `flat-scan` ground truth
- **Generation-based collections**: Staged publish lifecycle with SQLite catalog, exact f32 row bundles, and kernel-ready persistence — collections are durable, resumable, and garbage-collected
- **Snapshot v2 + quantization**: mmap-backed local bundles, SQ8 quantization, and storage preflight
- **AST-aware splitting**: Tree-sitter parsing with line-based fallback
- **Change-aware reindex**: Snapshot-driven change detection with WAL-backed local durability
- **Agent-native CLI**: Deterministic NDJSON output, machine-readable protocol spec via `sca agent-doc`, `estimate-storage`, `calibrate`, `search --stdin-batch`, and structured tracing

## Quick Start

### 1. Install

Choose one install method:

- `brew install luiz-frias/tap/semantic-code`
- `curl -fsSL https://github.com/Luiz-Frias/semantic-code-agx/releases/latest/download/install.sh | sh`
- `winget install --id Luiz-Frias.SemanticCode -e` (Windows)
- `scoop bucket add semantic-code https://github.com/Luiz-Frias/semantic-code-agx && scoop install semantic-code` (Windows)
- `mise use -g github:Luiz-Frias/semantic-code-agx@latest`
- `cargo install semantic-code-cli --locked`

This installs the `sca` and `semantic-code` commands.

For the full install matrix, see `docs/release.md`.

### 2. Initialize

From the root of your codebase:

```bash
sca init
```

This creates `.context/manifest.json` and a default `.context/config.toml`.

### 3. Index

Optional preflight:

```bash
sca estimate-storage
```

Then index:

```bash
sca index --init
```

### 4. Search

```bash
sca search --query "error handling and recovery"
```

### Agent-friendly CLI

Use `--agent` for NDJSON output, no prompts, and quiet stderr:

```bash
sca --agent search --query "error handling"
```

Get the machine-readable protocol spec (all commands, NDJSON shapes, exit codes, recovery table):

```bash
sca agent-doc          # full spec
sca agent-doc search   # scoped to one command
```

See [`TOOL.agents.md`](TOOL.agents.md) for the compact agent integration guide, or [`TOOL.md`](TOOL.md) for the full wire contract.

## Documentation

Full documentation is available in [`docs/`](docs/README.md):

| Audience | Document |
|----------|----------|
| New users | [Getting Started](docs/getting-started.md) |
| Users | [Configuration](docs/guides/configuration.md) · [Indexing](docs/guides/indexing.md) · [Searching](docs/guides/searching.md) |
| Agents | [`TOOL.agents.md`](TOOL.agents.md) · [`TOOL.md`](TOOL.md) · `sca agent-doc` |
| Contributors | [Architecture](docs/architecture/README.md) · [ADRs](docs/adrs/README.md) |
| Reference | [CLI](docs/reference/cli.md) · [Config Schema](docs/reference/config-schema.md) · [Error Codes](docs/reference/error-codes.md) |
| Operations | [Troubleshooting](docs/troubleshooting.md) · [FAQ](docs/faq.md) · [Release](docs/release.md) |

## Development

### Prerequisites

- Rust 1.96+
- [mise](https://mise.jdx.dev/) (optional)
- [just](https://github.com/casey/just)

### Setup

```bash
git clone https://github.com/Luiz-Frias/semantic-code-agx.git
cd semantic-code-agx
mise install
just setup
```

### Common Commands

| Command | Description |
|---------|-------------|
| `just pc` | Pre-commit checks (staged files) |
| `just pc-full` | Full pre-commit gate |
| `just test-unit` | Fast unit tests |
| `just test-all` | Full test suite |
| `cargo run --bin sca -- --help` | Run the CLI |

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE](LICENSE))

at your option.
