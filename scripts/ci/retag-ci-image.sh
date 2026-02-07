#!/usr/bin/env bash
# =============================================================================
# Retag CI Image - Manual Tag Propagation
# =============================================================================
# Usage: ./scripts/ci/retag-ci-image.sh <git-sha>
#
# Pulls a SHA-tagged CI image and retags it with all rolling tags.
# Use this to propagate a critical fix across all branches immediately.
#
# Prerequisites:
#   - Docker installed and running
#   - Logged into GHCR: docker login ghcr.io -u <username>
#   - GITHUB_TOKEN with packages:write permission
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_LC="$(echo "${GITHUB_REPOSITORY:-luiz-frias/semantic-code-agx}" | tr '[:upper:]' '[:lower:]')"
IMAGE_BASE="ghcr.io/${REPO_LC}-ci"

# All rolling tags that should point to the same image
ROLLING_TAGS=(
  "nightly"     # main branch
  "latest"      # main branch
  "nightly-dev" # dev branch
  "ag-dev"      # agent branches (promoted)
  "staging"     # staging branch
)

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
if [ $# -ne 1 ]; then
  echo "Usage: $0 <git-sha>"
  echo ""
  echo "Example:"
  echo "  $0 abc1234"
  echo ""
  echo "This will:"
  echo "  1. Pull ghcr.io/${REPO_LC}-ci:abc1234"
  echo "  2. Tag it with: ${ROLLING_TAGS[*]}"
  echo "  3. Push all tags to GHCR"
  exit 1
fi

SHA="$1"
SOURCE_TAG="${IMAGE_BASE}:${SHA}"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "CI Image Retagging Tool"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Source image: ${SOURCE_TAG}"
echo "Target tags:  ${ROLLING_TAGS[*]}"
echo ""

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
  echo "✗ Docker is not running or not accessible"
  exit 1
fi

# Check if logged into GHCR
if ! docker pull "${IMAGE_BASE}:${SHA}" >/dev/null 2>&1; then
  echo "✗ Failed to pull ${SOURCE_TAG}"
  echo ""
  echo "Make sure you're logged into GHCR:"
  echo "  docker login ghcr.io -u <username> -p <GITHUB_TOKEN>"
  echo ""
  echo "Your token needs 'write:packages' permission."
  exit 1
fi

# ---------------------------------------------------------------------------
# Pull source image
# ---------------------------------------------------------------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Pulling source image"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker pull "${SOURCE_TAG}"
echo "✓ Pulled ${SOURCE_TAG}"
echo ""

# ---------------------------------------------------------------------------
# Tag with all rolling tags
# ---------------------------------------------------------------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Creating local tags"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for tag in "${ROLLING_TAGS[@]}"; do
  docker tag "${SOURCE_TAG}" "${IMAGE_BASE}:${tag}"
  echo "✓ Tagged ${IMAGE_BASE}:${tag}"
done
echo ""

# ---------------------------------------------------------------------------
# Push all tags
# ---------------------------------------------------------------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Pushing tags to GHCR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for tag in "${ROLLING_TAGS[@]}"; do
  echo "Pushing ${IMAGE_BASE}:${tag}..."
  docker push "${IMAGE_BASE}:${tag}"
  echo "✓ Pushed ${IMAGE_BASE}:${tag}"
done
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ Success! All rolling tags now point to ${SHA}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Updated tags:"
for tag in "${ROLLING_TAGS[@]}"; do
  echo "  • ${IMAGE_BASE}:${tag}"
done
echo ""
echo "All branches can now pull the Ubuntu 24.04-based image!"
