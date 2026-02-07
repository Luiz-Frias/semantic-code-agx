#!/usr/bin/env bash
set -euo pipefail

printf "→ Validating config fixtures...\n"
cargo test -p semantic-code-config --test config_fixtures
