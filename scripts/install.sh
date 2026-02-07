#!/usr/bin/env bash
set -euo pipefail

REPO="Luiz-Frias/semantic-code-agx"
BIN_PRIMARY="sca"
BIN_ALIAS="semantic-code"

VERSION="latest"
INSTALL_DIR=""
NO_ALIAS="false"
FORCE="false"
QUIET="false"

usage() {
  cat <<'EOF'
Usage: install.sh [options]

Options:
  -v, --version <version>   Install a specific version (default: latest)
  -b, --bin-dir <path>      Install directory (default: XDG_BIN_HOME or ~/.local/bin)
  --no-alias                Do not create the semantic-code alias
  --force                   Overwrite existing binaries
  -q, --quiet               Reduce output
  -h, --help                Show this help text
EOF
}

log() {
  if [[ "${QUIET}" != "true" ]]; then
    printf '%s\n' "$*"
  fi
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -v | --version)
      VERSION="${2:-}"
      [[ -n "${VERSION}" ]] || die "missing value for --version"
      shift 2
      ;;
    -b | --bin-dir)
      INSTALL_DIR="${2:-}"
      [[ -n "${INSTALL_DIR}" ]] || die "missing value for --bin-dir"
      shift 2
      ;;
    --no-alias)
      NO_ALIAS="true"
      shift
      ;;
    --force)
      FORCE="true"
      shift
      ;;
    -q | --quiet)
      QUIET="true"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

require_cmd curl
require_cmd tar

if [[ -z "${INSTALL_DIR}" ]]; then
  INSTALL_DIR="${XDG_BIN_HOME:-${HOME}/.local/bin}"
fi

OS="$(uname -s)"
ARCH="$(uname -m)"

case "${OS}" in
  Darwin) TARGET_OS="apple-darwin" ;;
  Linux) TARGET_OS="unknown-linux-gnu" ;;
  *)
    die "unsupported OS: ${OS} (supported: macOS, Linux)"
    ;;
esac

case "${ARCH}" in
  x86_64 | amd64) TARGET_ARCH="x86_64" ;;
  arm64 | aarch64) TARGET_ARCH="aarch64" ;;
  *)
    die "unsupported architecture: ${ARCH} (supported: x86_64, arm64/aarch64)"
    ;;
esac

TARGET="${TARGET_ARCH}-${TARGET_OS}"

if [[ "${VERSION}" == "latest" ]]; then
  log "Resolving latest release tag..."
  # Prefer the redirect target over the GitHub API to avoid rate limits.
  FINAL_URL="$(curl -fsSL -o /dev/null -w '%{url_effective}' "https://github.com/${REPO}/releases/latest")"
  TAG="${FINAL_URL##*/}"
  [[ "${TAG}" == v* ]] || die "failed to resolve latest release tag"
else
  TAG="${VERSION}"
  if [[ "${TAG}" != v* ]]; then
    TAG="v${TAG}"
  fi
fi

VERSION_NO_V="${TAG#v}"
ASSET="sca-${VERSION_NO_V}-${TARGET}.tar.gz"
CHECKSUM="sca-${VERSION_NO_V}-${TARGET}.sha256"

if [[ "${VERSION}" == "latest" ]]; then
  BASE_URL="https://github.com/${REPO}/releases/latest/download"
else
  BASE_URL="https://github.com/${REPO}/releases/download/${TAG}"
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

log "Downloading ${ASSET}..."
curl -fsSL "${BASE_URL}/${ASSET}" -o "${TMP_DIR}/${ASSET}"
curl -fsSL "${BASE_URL}/${CHECKSUM}" -o "${TMP_DIR}/${CHECKSUM}"

if command -v sha256sum >/dev/null 2>&1; then
  (cd "${TMP_DIR}" && sha256sum -c "${CHECKSUM}")
elif command -v shasum >/dev/null 2>&1; then
  (cd "${TMP_DIR}" && shasum -a 256 -c "${CHECKSUM}")
else
  die "sha256sum or shasum is required to verify downloads"
fi

tar -xzf "${TMP_DIR}/${ASSET}" -C "${TMP_DIR}"

mkdir -p "${INSTALL_DIR}"

PRIMARY_PATH="${INSTALL_DIR}/${BIN_PRIMARY}"
ALIAS_PATH="${INSTALL_DIR}/${BIN_ALIAS}"

if [[ -e "${PRIMARY_PATH}" && "${FORCE}" != "true" ]]; then
  die "${PRIMARY_PATH} already exists (use --force to overwrite)"
fi

install -m 0755 "${TMP_DIR}/${BIN_PRIMARY}" "${PRIMARY_PATH}"

if [[ "${NO_ALIAS}" != "true" ]]; then
  if [[ -e "${ALIAS_PATH}" && "${FORCE}" != "true" ]]; then
    die "${ALIAS_PATH} already exists (use --force to overwrite)"
  fi
  ln -sf "${PRIMARY_PATH}" "${ALIAS_PATH}"
fi

if ! command -v "${BIN_PRIMARY}" >/dev/null 2>&1; then
  log "Installed to ${INSTALL_DIR} (add it to your PATH)"
else
  log "Installed ${BIN_PRIMARY} (${VERSION_NO_V})"
fi
