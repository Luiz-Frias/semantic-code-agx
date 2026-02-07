#!/usr/bin/env bash
set -euo pipefail

printf "→ Running Phase 03 config E2E...\n"
cargo test -p semantic-code-cli --test phase3_config
