# =============================================================================
# JUSTFILE - semantic-code-agx
# =============================================================================
# CI-focused task runner. Local dev workflows are in justfile.local.
# Run `just` or `just --list` to see available commands.
# Docs: https://github.com/casey/just
# =============================================================================

# Use bash for shell commands
set shell := ["bash", "-cu"]

# Import local development workflows
import 'justfile.local'

# Default recipe - show help
default:
    @just --list --unsorted

# =============================================================================
# QUALITY GATES (CI + pre-commit)
# =============================================================================

# Run all quality checks (used by pre-commit)
check: fmt-check clippy test-unit
    @echo "✓ All quality checks passed"

# Run all checks including slow ones (used by pre-push)
check-all: check test-all deny audit
    @echo "✓ All pre-push checks passed"

# Format check (no modifications)
fmt-check:
    @echo "→ Checking formatting..."
    cargo fmt --all -- --check

# Run clippy with strict lints (production + tests)
clippy:
    @echo "→ Running clippy (production - no allows permitted)..."
    cargo clippy --workspace --lib --bins --exclude semantic-code-testkit -- \
        -D warnings \
        -D clippy::unwrap_used \
        -D clippy::expect_used \
        -D clippy::panic \
        -D clippy::allow_attributes \
        -W clippy::await_holding_lock \
        -W clippy::dbg_macro \
        -W clippy::todo \
        -W clippy::unimplemented \
        -D unused_must_use
    @echo "→ Running clippy (tests - allows require reason)..."
    cargo clippy --workspace --tests --benches --exclude semantic-code-testkit -- \
        -D warnings \
        -D clippy::allow_attributes_without_reason \
        -W clippy::await_holding_lock \
        -W clippy::dbg_macro

# =============================================================================
# TESTING (using cargo-nextest for speed + lower memory)
# =============================================================================

# Run unit tests only (fast, for pre-commit)
test-unit:
    @echo "→ Running unit tests (nextest)..."
    cargo nextest run --workspace --lib

# Run tests across all targets
test:
    @echo "→ Running tests (nextest)..."
    cargo nextest run --workspace --all-targets

# Run all tests including integration and doc tests
test-all:
    @echo "→ Running all tests (nextest + doctest)..."
    cargo nextest run --workspace --all-targets
    cargo test --workspace --doc

# =============================================================================
# SECURITY & DEPENDENCIES
# =============================================================================

# Run cargo deny checks (licenses, advisories, bans)
deny:
    @echo "→ Running cargo deny..."
    cargo deny check

# Run cargo audit for CVEs
audit:
    @echo "→ Running security audit..."
    cargo audit

# =============================================================================
# BUILD
# =============================================================================

# Build debug
build:
    @echo "→ Building debug..."
    cargo build --workspace

# Build release
build-release:
    @echo "→ Building release..."
    cargo build --workspace --release

# =============================================================================
# DOCUMENTATION
# =============================================================================

# Build documentation without opening
doc-build:
    cargo doc --workspace --no-deps

# =============================================================================
# CI/CD
# =============================================================================

# Full CI pipeline (what GitHub Actions runs)
ci: check-all doc-build
    @echo "✓ CI pipeline passed"

# CI-optimized gate: pre-commit + pre-push on changed files only
pc-ci: cargo-sweep
    @echo "→ Running CI quality gates (changed files only)..."
    prek run
    prek run --stage pre-push
    @echo "✓ All CI gates passed"

# =============================================================================
# SETUP
# =============================================================================

# Install OS-level dependencies (pkg-config) for optional crates.
_install_os_deps:
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v pkg-config >/dev/null 2>&1; then
        echo "✓ pkg-config already installed"
        exit 0
    fi
    OS="$(uname -s)"
    case "$OS" in
        Darwin*)
            if ! command -v brew >/dev/null 2>&1; then
                echo "✗ Homebrew not found. Install brew to get pkg-config."
                exit 1
            fi
            brew install pkg-config
            ;;
        Linux*)
            if command -v apt-get >/dev/null 2>&1; then
                if command -v sudo >/dev/null 2>&1 && [[ "$(id -u)" -ne 0 ]]; then
                    sudo apt-get update
                    sudo apt-get install -y pkg-config
                else
                    apt-get update
                    apt-get install -y pkg-config
                fi
            else
                echo "✗ apt-get not found. Install pkg-config via your distro package manager."
                exit 1
            fi
            ;;
        *)
            echo "✗ Unsupported OS: $OS"
            exit 1
            ;;
    esac

# Install development tools
setup: _install_os_deps
    @echo "→ Installing mise-managed tools..."
    mise install
    @# Start sccache daemon if installed (not in CI)
    @# Use mise which to find sccache even if shims aren't in shell's hash table yet
    @if mise which sccache >/dev/null 2>&1; then \
        echo "→ Starting sccache daemon..."; \
        mise exec -- sccache --start-server || true; \
    else \
        echo "→ sccache not installed (CI mode), skipping..."; \
    fi
    @echo "→ Installing cargo dev tools..."
    @# In CI, clear RUSTC_WRAPPER for all Rust/cargo commands (mise may re-set it)
    @if [ -n "$CI" ]; then \
        unset RUSTC_WRAPPER && \
        cargo install cargo-watch cargo-audit cargo-deny cargo-machete cargo-tarpaulin cargo-nextest taplo-cli && \
        echo "→ Setting up prek hooks..." && \
        prek install && \
        prek install-hooks; \
    else \
        cargo install cargo-watch cargo-audit cargo-deny cargo-machete cargo-tarpaulin cargo-nextest taplo-cli && \
        echo "→ Setting up prek hooks..." && \
        prek install && \
        prek install-hooks; \
    fi
    @echo "✓ Development environment ready"
    @# Only show local dev tips if not in CI
    @if [ -z "$CI" ]; then \
        echo ""; \
        echo "TIP: Run 'bacon' to start background checking"; \
        echo "TIP: sccache will cache builds across branches"; \
    fi

# Verify development environment
verify:
    @echo "→ Verifying development environment..."
    @command -v cargo >/dev/null 2>&1 || { echo "✗ cargo not found"; exit 1; }
    @command -v rustfmt >/dev/null 2>&1 || { echo "✗ rustfmt not found"; exit 1; }
    @command -v clippy-driver >/dev/null 2>&1 || { echo "✗ clippy not found"; exit 1; }
    @command -v cargo-nextest >/dev/null 2>&1 || { echo "✗ cargo-nextest not found"; exit 1; }
    @command -v prek >/dev/null 2>&1 || { echo "✗ prek not found"; exit 1; }
    @command -v sccache >/dev/null 2>&1 || { echo "✗ sccache not found"; exit 1; }
    @command -v bacon >/dev/null 2>&1 || { echo "✗ bacon not found"; exit 1; }
    @sccache --show-stats >/dev/null 2>&1 && echo "✓ sccache running" || echo "⚠ sccache not running (start with: sccache --start-server)"
    @echo "✓ All required tools found"
