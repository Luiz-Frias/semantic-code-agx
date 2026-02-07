#!/usr/bin/env python3
"""
Verify a packaged release archive contains a runnable `sca` binary.

Used by CI to ensure the artifact users download is valid.
"""

from __future__ import annotations

import argparse
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path


def extract(archive: Path, out_dir: Path) -> None:
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(out_dir)
        return
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
        return
    raise SystemExit(f"unsupported archive type: {archive}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, help="Path to sca-<version>-<target> archive")
    args = parser.parse_args()

    archive = Path(args.archive).resolve()
    if not archive.exists():
        raise SystemExit(f"missing archive: {archive}")

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        extract(archive, out)

        exe = out / "sca"
        if (out / "sca.exe").exists():
            exe = out / "sca.exe"

        if not exe.exists():
            raise SystemExit(f"missing extracted binary: {exe}")

        result = subprocess.run([str(exe), "--version"], check=True, capture_output=True, text=True)
        if not result.stdout.strip():
            raise SystemExit("empty --version output")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
