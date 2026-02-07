#!/usr/bin/env python3
"""
Package a built CLI binary into a release archive + checksum.

This script is designed to run both locally and in CI.
It intentionally avoids third-party dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def package_tar_gz(archive_path: Path, stage_dir: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as tar:
        for entry in sorted(stage_dir.iterdir()):
            tar.add(entry, arcname=entry.name)


def package_zip(archive_path: Path, stage_dir: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for entry in sorted(stage_dir.iterdir()):
            zf.write(entry, arcname=entry.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="Version without leading 'v' (e.g. 0.1.0)")
    parser.add_argument("--target", required=True, help="Rust target triple (e.g. x86_64-unknown-linux-gnu)")
    parser.add_argument("--bin-path", required=True, help="Path to the built sca binary (or sca.exe)")
    parser.add_argument("--format", required=True, choices=["tar.gz", "zip"], help="Archive format")
    parser.add_argument("--out-dir", required=True, help="Output directory for archives/checksums")
    args = parser.parse_args()

    version = args.version.strip()
    if not version or version.startswith("v"):
        print("error: --version must be set and must not start with 'v'", file=sys.stderr)
        return 2

    target = args.target.strip()
    if not target:
        print("error: --target must be set", file=sys.stderr)
        return 2

    bin_path = Path(args.bin_path).resolve()
    if not bin_path.exists():
        print(f"error: binary not found: {bin_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_dir = out_dir / f"stage-{target}"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    ext = bin_path.suffix  # ".exe" on Windows, "" elsewhere
    staged_sca = stage_dir / f"sca{ext}"
    staged_alias = stage_dir / f"semantic-code{ext}"

    shutil.copy2(bin_path, staged_sca)
    # Include the alias binary in the archive for manual installs (the install script will still
    # create a symlink on Unix).
    shutil.copy2(bin_path, staged_alias)

    archive_ext = "tar.gz" if args.format == "tar.gz" else "zip"
    archive_name = f"sca-{version}-{target}.{archive_ext}"
    archive_path = out_dir / archive_name
    if archive_path.exists():
        archive_path.unlink()

    if args.format == "tar.gz":
        package_tar_gz(archive_path, stage_dir)
    else:
        package_zip(archive_path, stage_dir)

    digest = sha256_file(archive_path)
    checksum_name = f"sca-{version}-{target}.sha256"
    checksum_path = out_dir / checksum_name
    checksum_path.write_text(f"{digest}  {archive_name}\n", encoding="utf-8")

    # Keep CI output stable and machine-parsable.
    print(str(archive_path))
    print(str(checksum_path))

    # Best-effort cleanup to avoid leaking staged binaries when run locally.
    shutil.rmtree(stage_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
