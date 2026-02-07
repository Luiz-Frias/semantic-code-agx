#!/usr/bin/env bash
set -euo pipefail

printf "→ Formatting code...\n"
cargo fmt --all

printf "→ Running clippy...\n"
cargo clippy --workspace --lib --bins --exclude semantic-code-testkit -- -D warnings

printf "→ Running tests...\n"
cargo test --workspace --all-targets
