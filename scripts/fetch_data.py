#!/usr/bin/env python3
"""Fetch the third-party chord corpus into data/raw/.

The corpus is not vendored into this repository: it is derived from iReal Pro
community playlists and carries no license of its own.  Fetching it at setup
time keeps that data out of our history and makes its provenance explicit.

Usage:  python scripts/fetch_data.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SOURCE_REPO = "https://github.com/mikeoliphant/JazzStandards.git"
FILENAME = "JazzStandards.json"
DEST_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> int:
    dest = DEST_DIR / FILENAME
    if dest.exists():
        print(f"already present: {dest} ({dest.stat().st_size // 1024} KB)")
        return 0

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "JazzStandards"
        print(f"cloning {SOURCE_REPO} ...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", SOURCE_REPO, str(clone)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            print("\nCould not fetch the corpus. Download JazzStandards.json from")
            print(f"{SOURCE_REPO} and place it at {dest}", file=sys.stderr)
            return 1
        shutil.copy(clone / FILENAME, dest)

    print(f"wrote {dest} ({dest.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
