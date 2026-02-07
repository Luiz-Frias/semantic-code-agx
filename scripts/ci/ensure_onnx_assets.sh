#!/usr/bin/env bash
# =============================================================================
# ONNX Asset Verification
# =============================================================================
# In CI: Verifies the entrypoint decompressed the model correctly.
# In local dev: Downloads if missing (fallback for non-docker workflows).
#
# The Docker entrypoint handles decompression from the baked-in compressed
# model. This script only downloads as a fallback for local development
# without Docker.
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_SLUG="Xenova-all-MiniLM-L6-v2"
MODEL_DIR_DEFAULT="$ROOT_DIR/.context/models/onnx/$MODEL_SLUG"
MODEL_DIR="${SCA_EMBEDDING_ONNX_MODEL_DIR:-$MODEL_DIR_DEFAULT}"
MODEL_DIR="${MODEL_DIR%/}"

TOKENIZER="$MODEL_DIR/tokenizer.json"
NESTED_MODEL="$MODEL_DIR/onnx/model.onnx"
ROOT_MODEL="$MODEL_DIR/model.onnx"

# ---------------------------------------------------------------------------
# Check if assets are already present
# ---------------------------------------------------------------------------
if [[ -f "$TOKENIZER" ]] && [[ -f "$NESTED_MODEL" || -f "$ROOT_MODEL" ]]; then
  echo "✓ ONNX assets present at $MODEL_DIR"
  exit 0
fi

# ---------------------------------------------------------------------------
# CI mode: assets should have been decompressed by entrypoint
# ---------------------------------------------------------------------------
if [[ "${CI:-}" == "true" ]]; then
  # Check for compressed model (entrypoint may not have run yet)
  COMPRESSED="/opt/onnx-models/$MODEL_SLUG/onnx/model.onnx.zst"
  if [[ -f "$COMPRESSED" ]]; then
    echo "→ Decompressing ONNX model from Docker image..."
    mkdir -p "$MODEL_DIR/onnx"
    zstd -d "$COMPRESSED" -o "$NESTED_MODEL"
    cp "/opt/onnx-models/$MODEL_SLUG/tokenizer.json" "$TOKENIZER"
    echo "✓ ONNX model decompressed to $MODEL_DIR"
    exit 0
  fi

  echo "⚠ CI mode but no compressed model found at $COMPRESSED"
  exit 1
fi
