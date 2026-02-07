#!/usr/bin/env bash
# =============================================================================
# Monitor CI Build and Retag - Full Automation
# =============================================================================
# This script:
# 1. Monitors ci-image.yml workflow until build completes
# 2. Runs retag-ci-image.sh to propagate the new image
# 3. Cancels/reruns the Prek Quality Gates job to pick up new image
#
# Prerequisites:
#   - gh CLI authenticated: gh auth status
#   - Docker logged into GHCR
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="Luiz-Frias/semantic-code-agx"

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
log() {
  echo "$(date '+%H:%M:%S') | $*" >&2
}

wait_for_workflow() {
  local workflow_name="$1"
  local branch="$2"
  local max_wait="${3:-300}" # 5 minutes default

  log "Waiting for ${workflow_name} workflow on ${branch}..."

  local elapsed=0
  local run_id
  while [ "$elapsed" -lt "$max_wait" ]; do
    # Get most recent run for this workflow and branch
    run_id=$(gh run list \
      --repo "$REPO" \
      --workflow "$workflow_name" \
      --branch "$branch" \
      --limit 1 \
      --json databaseId \
      --jq '.[0].databaseId' 2>/dev/null)

    if [ -n "$run_id" ] && [ "$run_id" != "null" ]; then
      log "Found run ID: ${run_id}"
      echo "$run_id"
      return 0
    fi

    sleep 5
    elapsed=$((elapsed + 5))
  done

  log "ERROR: Workflow not found after ${max_wait}s"
  return 1
}

get_run_status() {
  local run_id="$1"
  gh run view "$run_id" --repo "$REPO" --json status,conclusion --jq '.status + ":" + (.conclusion // "pending")'
}

get_job_id() {
  local run_id="$1"
  local job_name="$2"

  gh run view "$run_id" --repo "$REPO" --json jobs --jq \
    ".jobs[] | select(.name == \"${job_name}\") | .databaseId"
}

# ---------------------------------------------------------------------------
# Step 1: Wait for CI Image build job to complete
# ---------------------------------------------------------------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Monitor CI Image build (nested in ci.yml)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Get current branch and SHA
BRANCH=$(git rev-parse --abbrev-ref HEAD)
SHA=$(git rev-parse --short HEAD)

log "Branch: ${BRANCH}"
log "SHA: ${SHA}"

# Find the ci.yml workflow run for this SHA
log "Looking for ci.yml workflow run for SHA ${SHA}..."
CI_RUN_ID=$(wait_for_workflow "ci.yml" "$BRANCH" 60)

if [ -z "$CI_RUN_ID" ]; then
  log "ERROR: Could not find ci.yml workflow run"
  exit 1
fi

log "Found CI run: ${CI_RUN_ID}"
log "View at: https://github.com/${REPO}/actions/runs/${CI_RUN_ID}"

# Monitor the "Build CI Image / Build CI Image" job
log "Watching for 'Build CI Image / Build CI Image' job..."

while true; do
  # Get job status
  JOB_STATUS=$(gh run view "$CI_RUN_ID" --repo "$REPO" --json jobs --jq \
    '.jobs[] | select(.name == "Build CI Image / Build CI Image") | .status + ":" + (.conclusion // "pending")')

  if [ -z "$JOB_STATUS" ]; then
    log "Build CI Image job not started yet, waiting..."
    sleep 10
    continue
  fi

  log "Build CI Image status: ${JOB_STATUS}"

  case "$JOB_STATUS" in
    completed:success)
      log "✓ CI Image build completed successfully!"
      break
      ;;
    completed:failure | completed:cancelled | completed:timed_out)
      log "✗ CI Image build failed with status: ${JOB_STATUS}"
      exit 1
      ;;
    *)
      sleep 15
      ;;
  esac
done

echo ""

# ---------------------------------------------------------------------------
# Step 2: Retag the image
# ---------------------------------------------------------------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Retag CI image with rolling tags"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log "Checking Docker authentication..."
if ! docker info >/dev/null 2>&1; then
  log "ERROR: Docker is not running"
  exit 1
fi

# Try to use gh token for docker login if not already authenticated
REPO_LC="$(echo "${REPO}" | tr '[:upper:]' '[:lower:]')"
TEST_IMAGE="ghcr.io/${REPO_LC}-ci:${SHA}"

if ! docker pull "$TEST_IMAGE" >/dev/null 2>&1; then
  log "Not authenticated to GHCR, logging in with gh token..."
  gh auth token | docker login ghcr.io -u "$(gh api user --jq .login)" --password-stdin
fi

log "Running retag script for SHA: ${SHA}"
"${SCRIPT_DIR}/retag-ci-image.sh" "$SHA"

echo ""

# ---------------------------------------------------------------------------
# Step 3: Cancel and rerun CI with new image
# ---------------------------------------------------------------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Rerun CI workflow with retagged image"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log "Using CI run: ${CI_RUN_ID}"

# Check current status
CURRENT_STATUS=$(get_run_status "$CI_RUN_ID")
log "Current CI status: ${CURRENT_STATUS}"

case "$CURRENT_STATUS" in
  completed:*)
    log "CI already completed, triggering a new run to use retagged image..."
    gh workflow run ci.yml --repo "$REPO" --ref "$BRANCH"
    log "✓ Triggered new ci.yml workflow"
    ;;
  in_progress:* | queued:*)
    log "CI still running, cancelling to force fresh start with retagged image..."
    gh run cancel "$CI_RUN_ID" --repo "$REPO"
    log "Cancelled run ${CI_RUN_ID}"

    log "Waiting 5 seconds before rerunning..."
    sleep 5

    log "Rerunning workflow..."
    gh run rerun "$CI_RUN_ID" --repo "$REPO"
    log "✓ Restarted ci.yml run ${CI_RUN_ID} with retagged image"
    ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ Automation complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Image SHA ${SHA} is now tagged with all rolling tags"
log "CI workflow will now use Ubuntu 24.04-based image"
log ""
log "Monitor the new run at:"
log "  https://github.com/${REPO}/actions/workflows/ci.yml"
