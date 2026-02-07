#!/usr/bin/env bash
set -euo pipefail

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found" >&2
  exit 1
fi

self_check_output=$(cargo run --bin sca -- self-check)
version_output=$(cargo run --bin sca -- --version)

if [[ -z "$self_check_output" ]]; then
  echo "self-check produced no output" >&2
  exit 1
fi

if ! grep -q "status: ok" <<<"$self_check_output"; then
  echo "self-check output missing status" >&2
  exit 1
fi

if [[ -z "$version_output" ]]; then
  echo "--version produced no output" >&2
  exit 1
fi

printf "CLI smoke checks passed.\n"
