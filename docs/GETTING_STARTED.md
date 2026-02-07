# Getting Started

Get up and running with semantic-code-agx in 5 minutes.

Primary command: `sca` (alias: `semantic-code`).

## Prerequisites

- Rust 1.95+
- Optional: embedding API key (OpenAI, Gemini, Voyage) or an Ollama runtime

## Installation

### GitHub Releases (recommended)

Download the prebuilt artifact for your OS/arch and install `sca` into your PATH.
If you prefer a one-liner, use the release install script:

```bash
curl -fsSL https://github.com/Luiz-Frias/semantic-code-agx/releases/latest/download/install.sh | sh
```

Note: the install script targets macOS/Linux. Windows users should use Winget or Scoop.

### Homebrew

```bash
brew install luiz-frias/tap/semantic-code
```

### Winget (Windows)

```powershell
winget install --id Luiz-Frias.SemanticCode -e
```

### Scoop (Windows)

```powershell
scoop bucket add semantic-code https://github.com/Luiz-Frias/semantic-code-agx
scoop install semantic-code
```

### mise (GitHub backend)

```bash
mise use -g github:Luiz-Frias/semantic-code-agx@latest
```

### Cargo (from source)

```bash
cargo install semantic-code-cli --locked
```

## Quick Start: first index + search

### 1. Initialize

From your codebase root:

```bash
sca init
```

This creates `.context/manifest.json` and a default `.context/config.toml`.

### 2. Index

```bash
sca index --init
```

### 3. Search

```bash
sca search --query "error handling and recovery"
```

### 4. Agent-friendly output

```bash
sca --agent search --query "auth flow"
```

## Optional: `.contextignore`

Add `.contextignore` to skip files and folders (same style as `.gitignore`):

```
node_modules/
.git/
target/
*.log
```

## Configuration essentials

The generated `.context/config.toml` is a full, validated config. Most users only
need to tweak a few fields:

```toml
version = 1

[embedding]
provider = "onnx" # or openai, gemini, voyage, ollama

[vectorDb]
provider = "local" # or milvus-grpc/milvus-rest
```

Cloud providers require env vars:

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export VOYAGEAI_API_KEY="..."
```

## Common tasks

```bash
sca status
sca reindex
sca clear
```

## Learn more

- **[Release & Install](./release.md)**
- **[Configuration Guide](./guides/configuration.md)**
- **[Embedding Providers](./guides/embedding-providers.md)**
- **[Troubleshooting](./TROUBLESHOOTING.md)**
