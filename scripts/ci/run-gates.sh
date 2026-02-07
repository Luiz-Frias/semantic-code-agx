#!/usr/bin/env bash
# =============================================================================
# CI Quality Gates
# =============================================================================
# Runs all quality gates in the CI container.
# Model decompression is handled by the container entrypoint.
#
# This script focuses purely on gate execution - separation of concerns:
#   - Dockerfile: tool installation (reproducible)
#   - Entrypoint: model decompression (one-time)
#   - This script: gate execution (the actual tests)
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Mark workspace as safe directory for Git (fixes ownership warning in containers)
git config --global --add safe.directory "$ROOT_DIR"

# Force CI-specific mise config (avoids local dev tools like sccache)
export MISE_CONFIG_FILE="${ROOT_DIR}/.github/ci/mise.ci.toml"

# Explicitly disable sccache wrapper (mise may override docker-compose env)
unset RUSTC_WRAPPER

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
if ! command -v mise >/dev/null 2>&1; then
  echo "✗ mise is required but not found"
  exit 1
fi

mise trust --all --yes

# Ensure local dev tools are available (no-op in CI image, needed for local)
just setup

# Verify ONNX model is ready (entrypoint should have decompressed it)
scripts/ci/ensure_onnx_assets.sh

# ---------------------------------------------------------------------------
# Gate 1: Formatting
# ---------------------------------------------------------------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Gate 1: Formatting"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cargo fmt --all

# ---------------------------------------------------------------------------
# Gate 2: Pre-commit + Pre-push hooks (prek) - changed files only
# ---------------------------------------------------------------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Gate 2: Prek hooks (pre-commit + pre-push)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
just pc-ci

# ---------------------------------------------------------------------------
# Gate 3: Verify no uncommitted changes
# ---------------------------------------------------------------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Gate 3: Clean working directory"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
git diff --exit-code

# ---------------------------------------------------------------------------
# Gate 4: Config determinism check
# ---------------------------------------------------------------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Gate 4: Config determinism"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
export SCA_SYNC_ALLOWED_EXTENSIONS=" TSX , .rs,ts,*.RS,tsx"
export SCA_SYNC_IGNORE_PATTERNS=' target/,node_modules/,dist\\,node_modules/'
scripts/print-effective-config.sh >/tmp/effective-config-1.json
scripts/print-effective-config.sh >/tmp/effective-config-2.json
diff -u /tmp/effective-config-1.json /tmp/effective-config-2.json

# ---------------------------------------------------------------------------
# Gate 5: Strict clippy (main/staging branches only)
# ---------------------------------------------------------------------------
if [[ "${GITHUB_REF_NAME:-}" == "main" || "${GITHUB_REF_NAME:-}" == "staging" ]]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Gate 5: Strict clippy (${GITHUB_REF_NAME})"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  cargo clippy --workspace --lib --bins --exclude semantic-code-testkit --all-features -- \
    -D warnings \
    -D clippy::unwrap_used \
    -D clippy::expect_used \
    -D clippy::panic \
    -D clippy::allow_attributes \
    -W clippy::await_holding_lock \
    -W clippy::dbg_macro \
    -D clippy::todo \
    -D clippy::unimplemented \
    -D unused_must_use
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ All gates passed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
