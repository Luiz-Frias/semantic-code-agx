# syntax=docker/dockerfile:1.7
# =============================================================================
# CI Image - Multi-stage build with ONNX model layer
# =============================================================================
# Architecture:
#   base ──> tools ──> ci (final, with ONNX models from external image)
#
# Model layer: COPY --from pre-built ONNX image (onnx-models.yml workflow)
# To swap models: edit the COPY --from image reference in the ci stage
# =============================================================================

# Ubuntu 24.04 LTS (noble) provides glibc 2.39 with C23 support (required by ort-sys)
ARG UBUNTU_VERSION=24.04

# =============================================================================
# Stage: base - Common dependencies
# =============================================================================
FROM ubuntu:${UBUNTU_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    pkg-config \
    clang \
    lld \
    mold \
    cmake \
    libssl-dev \
    python3 \
    python3-venv \
    xz-utils \
    unzip \
    zstd \
    docker.io \
  && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage: tools - Rust toolchain + cargo tools
# =============================================================================
FROM base AS tools

ENV PATH="/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Install mise
RUN curl -fsSL https://mise.jdx.dev/install.sh -o /tmp/mise.sh \
  && bash /tmp/mise.sh -y \
  && rm /tmp/mise.sh

WORKDIR /work

# Copy only files needed for tool installation (better cache)
# Use CI-specific mise config (excludes sccache/bacon which don't support arm64)
COPY .github/ci/mise.ci.toml ./mise.toml
COPY justfile .pre-commit-config.yaml Cargo.toml Cargo.lock rustfmt.toml clippy.toml deny.toml ./

# Install tools with cache mount for faster rebuilds
RUN --mount=type=cache,target=/root/.cache/mise \
    --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    mise trust --all --yes \
    && mise install \
    && mise exec -- cargo install \
      cargo-watch \
      cargo-audit \
      cargo-deny \
      cargo-machete \
      cargo-tarpaulin \
      taplo-cli \
    && mise exec -- cargo install --locked cargo-nextest

# =============================================================================
# Stage: ci - Final CI image (tools + ONNX models)
# =============================================================================
FROM base AS ci

ENV PATH="/root/.local/share/mise/shims:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV SCA_EMBEDDING_ONNX_MODEL_DIR="/workspace/.context/models/onnx/Xenova-all-MiniLM-L6-v2"

# Copy mise + cargo from tools stage
COPY --from=tools /root/.local/share/mise /root/.local/share/mise
COPY --from=tools /root/.local/bin/mise /root/.local/bin/mise
COPY --from=tools /root/.cargo /root/.cargo

# Setup mise config dir + ensure PATH survives login shell
# (bash -l sources /etc/profile which clobbers PATH; .profile is sourced after)
# hadolint ignore=SC2016
RUN mkdir -p /root/.config/mise \
  && echo 'export PATH="/root/.local/share/mise/shims:/root/.local/bin:/root/.cargo/bin:$PATH"' >> /root/.profile

# Copy ONNX models (compressed) from pre-built image
# To swap models: edit this image reference and rebuild (must be lowercase!)
COPY --from=ghcr.io/luiz-frias/semantic-code-agents-rs-onnx:xenova-all-minilm-l6-v2 /models /opt/onnx-models

WORKDIR /workspace

# Entrypoint script to decompress model on first run (if needed)
COPY <<'EOF' /usr/local/bin/ci-entrypoint.sh
#!/bin/bash
set -euo pipefail

MODEL_DIR="${SCA_EMBEDDING_ONNX_MODEL_DIR:-/workspace/.context/models/onnx/Xenova-all-MiniLM-L6-v2}"
COMPRESSED_MODEL="/opt/onnx-models/Xenova-all-MiniLM-L6-v2/onnx/model.onnx.zst"
TARGET_MODEL="${MODEL_DIR}/onnx/model.onnx"

# Decompress on first run if target doesn't exist
if [[ -f "$COMPRESSED_MODEL" ]] && [[ ! -f "$TARGET_MODEL" ]]; then
  echo "→ Decompressing ONNX model..."
  mkdir -p "$(dirname "$TARGET_MODEL")"
  zstd -d "$COMPRESSED_MODEL" -o "$TARGET_MODEL"

  # Copy tokenizer too
  cp /opt/onnx-models/Xenova-all-MiniLM-L6-v2/tokenizer.json "${MODEL_DIR}/"
  echo "✓ ONNX model ready at $MODEL_DIR"
fi

exec "$@"
EOF

RUN chmod +x /usr/local/bin/ci-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/ci-entrypoint.sh"]
CMD ["bash"]

# OCI labels for GHCR integration
# IMPORTANT: org.opencontainers.image.source links the package to the repo,
# granting GITHUB_TOKEN automatic write access in Actions workflows.
LABEL org.opencontainers.image.source="https://github.com/Luiz-Frias/semantic-code-agx"
LABEL org.opencontainers.image.title="semantic-code-agx CI"
LABEL org.opencontainers.image.description="CI toolchain image with Rust, pre-commit hooks, and ONNX models"
LABEL org.opencontainers.image.licenses="MIT"
