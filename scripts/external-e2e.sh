#!/usr/bin/env bash
set -euo pipefail

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found" >&2
  exit 1
fi

if [[ "${SCA_E2E_MOCK_ONLY:-}" == "1" ]]; then
  printf "→ Running Phase 06 external E2E (mock-only)...\n"
  cargo test -p semantic-code-cli --test phase6_external --no-default-features
  exit 0
fi

printf "→ Running Phase 06 external E2E...\n"
cargo test -p semantic-code-cli --test phase6_external
