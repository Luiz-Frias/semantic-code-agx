#!/usr/bin/env bash
set -euo pipefail

printf "→ Printing effective config (defaults + env overrides)...\n" >&2
cargo run --quiet -p semantic-code-config --bin print_effective_config
