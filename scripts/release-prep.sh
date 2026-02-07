#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  just release-prep [<target-branch>] [--version <semver>]

Examples:
  just release-prep
  just release-prep dev
  just release-prep main --version 0.2.0

What it does:
  - Analyzes commits/files changed vs the target branch (default: dev)
  - Suggests a semver bump when --version is omitted
  - Updates:
      - Cargo.toml (workspace package version)
      - CHANGELOG.md (adds a new release section and updates compare links)

Notes:
  - This command does not commit. Review diffs before tagging.
  - Run `just pc-full` before publishing a release tag.
EOF
}

TARGET_BRANCH="dev"
VERSION_OVERRIDE=""

if [[ $# -gt 0 && "${1}" != -* ]]; then
  TARGET_BRANCH="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION_OVERRIDE="${2:-}"
      [[ -n "${VERSION_OVERRIDE}" ]] || {
        echo "error: --version requires a value" >&2
        exit 2
      }
      shift 2
      ;;
    --version=*)
      VERSION_OVERRIDE="${1#--version=}"
      [[ -n "${VERSION_OVERRIDE}" ]] || {
        echo "error: --version requires a value" >&2
        exit 2
      }
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! git rev-parse --verify "${TARGET_BRANCH}" >/dev/null 2>&1; then
  echo "error: target branch not found: ${TARGET_BRANCH}" >&2
  exit 2
fi

# Best-effort derive repo slug for compare links (owner/repo).
# Override with SCA_REPO=owner/repo if you're running from a fork.
REMOTE_URL="$(git remote get-url origin 2>/dev/null || true)"
REPO_SLUG=""
case "${REMOTE_URL}" in
  https://github.com/*)
    REPO_SLUG="${REMOTE_URL#https://github.com/}"
    REPO_SLUG="${REPO_SLUG%.git}"
    ;;
  git@github.com:*)
    REPO_SLUG="${REMOTE_URL#git@github.com:}"
    REPO_SLUG="${REPO_SLUG%.git}"
    ;;
esac

REPO="${SCA_REPO:-${REPO_SLUG}}"
if [[ -z "${REPO}" ]]; then
  echo "error: failed to determine GitHub repo slug (owner/repo). Set SCA_REPO=owner/repo and retry." >&2
  exit 2
fi

CURRENT_BRANCH="$(git branch --show-current 2>/dev/null || true)"
DATE_UTC="$(date -u +%Y-%m-%d)"

CURRENT_VERSION="$(
  python3 - <<'PY'
import re
from pathlib import Path

text = Path("Cargo.toml").read_text(encoding="utf-8")
m = re.search(r'\n\[workspace\.package\]\n(?:.*\n)*?version\s*=\s*"([^"]+)"', text)
if not m:
    raise SystemExit("error: failed to read current version from Cargo.toml")
print(m.group(1))
PY
)"

OLD_TAG_EXISTS="false"
if git rev-parse --verify "refs/tags/v${CURRENT_VERSION}" >/dev/null 2>&1; then
  OLD_TAG_EXISTS="true"
fi

COMMITS="$(git log "${TARGET_BRANCH}..HEAD" --oneline --no-merges || true)"
DIFF_STAT="$(git diff "${TARGET_BRANCH}...HEAD" --stat || true)"

VERSION="$(
  CURRENT_VERSION="${CURRENT_VERSION}" \
    VERSION_OVERRIDE="${VERSION_OVERRIDE}" \
    COMMITS="${COMMITS}" \
    python3 - <<'PY'
import os
import re
import sys

current = os.environ["CURRENT_VERSION"]
override = os.environ.get("VERSION_OVERRIDE", "").strip() or None

semver_re = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")

if override is not None:
    if not semver_re.match(override):
        print(
            f"error: invalid --version {override!r} (expected semver like 0.2.0 or 0.2.0-rc.1)",
            file=sys.stderr,
        )
        sys.exit(2)
    print(override)
    sys.exit(0)

commits = os.environ.get("COMMITS", "").splitlines()

# Decide bump based on conventional-commit-ish messages.
bump = "patch"
for line in commits:
    subject = line.split(" ", 1)[1] if " " in line else line
    if "BREAKING CHANGE" in subject or "BREAKING:" in subject:
        bump = "major"
        break
    if re.match(r"^[a-z]+(?:\([^)]*\))?!:", subject):
        bump = "major"
        break

if bump != "major":
    for line in commits:
        subject = line.split(" ", 1)[1] if " " in line else line
        if subject.startswith("feat"):
            bump = "minor"
            break

m = re.match(r"^(\d+)\.(\d+)\.(\d+)", current)
if not m:
    print(f"error: current version is not semver: {current}", file=sys.stderr)
    sys.exit(2)
maj, min_, pat = map(int, m.groups())

if bump == "major":
    maj += 1
    min_ = 0
    pat = 0
elif bump == "minor":
    min_ += 1
    pat = 0
else:
    pat += 1

print(f"{maj}.{min_}.{pat}")
PY
)"

echo "→ Release prep"
echo "  current branch : ${CURRENT_BRANCH:-<detached>}"
echo "  target branch  : ${TARGET_BRANCH}"
echo "  current version: ${CURRENT_VERSION}"
echo "  next version   : ${VERSION}"
echo ""

VERSION="${VERSION}" python3 - <<'PY'
import os
import re
from pathlib import Path

path = Path("Cargo.toml")
text = path.read_text(encoding="utf-8")

# Replace only the workspace.package version.
pattern = r'(\n\[workspace\.package\]\n(?:.*\n)*?version\s*=\s*")([^"]+)(")'
match = re.search(pattern, text)
if not match:
    raise SystemExit("error: failed to locate [workspace.package] version in Cargo.toml")

before = match.group(2)
after = os.environ["VERSION"]
if before == after:
    print("✓ Cargo.toml version already up to date")
else:
    text = re.sub(pattern, lambda m: m.group(1) + after + m.group(3), text, count=1)
    print(f"✓ Updated Cargo.toml: {before} → {after}")

# Keep internal workspace dependency versions in sync with the workspace package version.
# This avoids semver-range drift (e.g., bumping to 0.2.0 while deps still require ^0.1.0).
dep_pattern = re.compile(
    r'^(semantic-code-[\w-]+\s*=\s*\{[^\n}]*\bpath\s*=\s*"[^"]+"[^\n}]*\bversion\s*=\s*")([^"]+)(")',
    flags=re.M,
)
text, n = dep_pattern.subn(lambda m: m.group(1) + after + m.group(3), text)
if n:
    print(f"✓ Updated {n} internal workspace dependency version(s) to {after}")

path.write_text(text, encoding="utf-8")
PY

OLD_VERSION="${CURRENT_VERSION}" \
  NEW_VERSION="${VERSION}" \
  RELEASE_DATE="${DATE_UTC}" \
  COMMITS="${COMMITS}" \
  REPO="${REPO}" \
  OLD_TAG_EXISTS="${OLD_TAG_EXISTS}" \
  python3 - <<'PY'
import os
import re
from pathlib import Path

repo = os.environ["REPO"]
old_version = os.environ["OLD_VERSION"]
new_version = os.environ["NEW_VERSION"]
release_date = os.environ["RELEASE_DATE"]
commit_lines = os.environ.get("COMMITS", "").splitlines()
old_tag_exists = os.environ.get("OLD_TAG_EXISTS", "false").lower() == "true"


def classify(subject: str) -> str:
    if (
        "BREAKING CHANGE" in subject
        or "BREAKING:" in subject
        or re.match(r"^[a-z]+(?:\([^)]*\))?!:", subject)
    ):
        return "breaking"
    if subject.startswith("feat"):
        return "added"
    if subject.startswith("fix"):
        return "fixed"
    if subject.startswith("docs"):
        return "docs"
    if subject.startswith(("refactor", "perf")):
        return "changed"
    if subject.startswith(("ci", "build", "chore")):
        return "infra"
    return "changed"


subjects = []
for line in commit_lines:
    parts = line.split(" ", 1)
    subjects.append(parts[1] if len(parts) == 2 else line)

buckets = {
    "breaking": [],
    "added": [],
    "changed": [],
    "fixed": [],
    "docs": [],
    "infra": [],
}

for subject in subjects:
    buckets[classify(subject)].append(subject)

lines = []
changelog = Path("CHANGELOG.md")
text = changelog.read_text(encoding="utf-8")

if re.search(rf"^## \[{re.escape(new_version)}\]\b", text, flags=re.M):
    raise SystemExit(f"error: CHANGELOG.md already contains a section for {new_version}")

m = re.search(r"^## \[Unreleased\]\s*$", text, flags=re.M)
if not m:
    raise SystemExit('error: CHANGELOG.md must contain a "## [Unreleased]" section')

# Determine the full [Unreleased] section span so we can move its body into the release.
unreleased_line_end = m.end()
if unreleased_line_end < len(text) and text[unreleased_line_end : unreleased_line_end + 1] == "\n":
    unreleased_line_end += 1

next_h = re.search(r"^## ", text[unreleased_line_end:], flags=re.M)
unreleased_end = unreleased_line_end + next_h.start() if next_h else len(text)

unreleased_body = text[unreleased_line_end:unreleased_end].strip("\n")

# Reset Unreleased to a consistent empty skeleton.
unreleased_skeleton = "\n\n### Added\n\n### Changed\n\n### Fixed\n"

def has_user_notes(body: str) -> bool:
    # Treat headings-only blocks as empty; any non-heading content counts.
    for line in body.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        return True
    return False


if has_user_notes(unreleased_body):
    # Keep-a-changelog flow: move curated Unreleased notes into the new version section.
    release_block = f"## [{new_version}] - {release_date}\n\n{unreleased_body.rstrip()}\n"
else:
    # If Unreleased is empty, synthesize notes from commit subjects.
    lines.append(f"## [{new_version}] - {release_date}")
    lines.append("")

    if buckets["breaking"]:
        lines.append("### Breaking Changes")
        for s in buckets["breaking"]:
            lines.append(f"- {s}")
        lines.append("")

    if buckets["added"]:
        lines.append("### Added")
        for s in buckets["added"]:
            lines.append(f"- {s}")
        lines.append("")

    if buckets["changed"]:
        lines.append("### Changed")
        for s in buckets["changed"]:
            lines.append(f"- {s}")
        lines.append("")

    if buckets["fixed"]:
        lines.append("### Fixed")
        for s in buckets["fixed"]:
            lines.append(f"- {s}")
        lines.append("")

    if buckets["docs"]:
        lines.append("### Documentation")
        for s in buckets["docs"]:
            lines.append(f"- {s}")
        lines.append("")

    if buckets["infra"]:
        lines.append("### Infrastructure")
        for s in buckets["infra"]:
            lines.append(f"- {s}")
        lines.append("")

    if all(not v for v in buckets.values()):
        lines.append("- No notable changes recorded.")
        lines.append("")

    release_block = "\n".join(lines).rstrip() + "\n"

updated = text[:unreleased_line_end] + unreleased_skeleton + "\n" + release_block + "\n" + text[unreleased_end:]

# Update compare links if present.
updated = re.sub(
    r"^\[Unreleased\]:\s+.*$",
    f"[Unreleased]: https://github.com/{repo}/compare/v{new_version}...HEAD",
    updated,
    flags=re.M,
)

if (old_version == new_version) or (not old_tag_exists):
    new_link = f"[{new_version}]: https://github.com/{repo}/releases/tag/v{new_version}"
else:
    new_link = f"[{new_version}]: https://github.com/{repo}/compare/v{old_version}...v{new_version}"
if re.search(rf"^\[{re.escape(new_version)}\]:\s+.*$", updated, flags=re.M):
    updated = re.sub(rf"^\[{re.escape(new_version)}\]:\s+.*$", new_link, updated, flags=re.M)
else:
    unreleased_link = re.search(r"^\[Unreleased\]:\s+.*$", updated, flags=re.M)
    if unreleased_link:
        idx = unreleased_link.start()
        updated = updated[:idx] + new_link + "\n" + updated[idx:]
    else:
        updated = updated.rstrip() + "\n" + new_link + "\n"

changelog.write_text(updated, encoding="utf-8")
print(f"✓ Updated CHANGELOG.md with {new_version} section")
PY

echo ""
echo "→ Diff summary vs ${TARGET_BRANCH}:"
echo "${DIFF_STAT}"
echo ""
echo "→ Commits:"
echo "${COMMITS:-<none>}"
echo ""
echo "Next steps:"
echo "  1) Review: git diff"
echo "  2) Gate:   just pc-full"
echo "  3) Tag on main HEAD:"
echo "     git checkout main && git pull origin main"
echo "     git tag v${VERSION}"
echo "     git push origin main"
echo "     git push origin v${VERSION}"
