# semantic-code-agx

[![CI](https://github.com/Luiz-Frias/semantic-code-agx/workflows/CI/badge.svg)](https://github.com/Luiz-Frias/semantic-code-agx/actions)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.95%2B-orange.svg)](https://www.rust-lang.org/)

> **Semantic code search engine** powered by embeddings and vector databases. Find code by meaning, not just keywords.

Primary command: `sca` (alias: `semantic-code`).

## Why This?

Traditional code search (grep, ripgrep) finds exact matches. Semantic search understands meaning:

- **Ask questions**: "How is authentication handled?" → finds auth code
- **Explore unfamiliar codebases**: "database connection setup" → finds DB init
- **Find patterns**: "error handling and recovery" → finds Result/error flows

## Features

- **Embedding providers**: ONNX (local), OpenAI, Gemini, Voyage, Ollama
- **Vector databases**: Local HNSW index, Milvus (gRPC/REST)
- **AST-aware splitting**: Tree-sitter parsing with line-based fallback
- **Change-aware reindex**: Snapshot-driven change detection
- **CLI-first**: Deterministic output for automation and AI agents

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

## Documentation

Full documentation is available in `docs/`:

- **[Getting Started](docs/GETTING_STARTED.md)**
- **[Release & Install](docs/release.md)**
- **[Architecture](docs/architecture/README.md)**
- **[Configuration](docs/guides/configuration.md)**
- **[CLI Reference](docs/reference/cli.md)**
- **[Troubleshooting](docs/TROUBLESHOOTING.md)**
- **[FAQ](docs/FAQ.md)**

## Development

### Prerequisites

- Rust 1.95+
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
