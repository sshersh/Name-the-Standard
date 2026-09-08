#!/usr/bin/env python3
"""Assemble the browser demo into a single self-contained page.

The page has to run as one file with no network access beyond fonts, so the
corpus is inlined rather than fetched. Neither the inlined corpus nor the built
page is committed - both embed the third-party chord data that
scripts/fetch_data.py pulls down, and that data carries no license of its own.

Usage:  python scripts/build_web_app.py            # -> web/name-the-standard.html
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "web" / "app.html"
CORPUS = ROOT / "web" / "corpus.js"
OUTPUT = ROOT / "web" / "name-the-standard.html"
PLACEHOLDER = '<script src="CORPUS_PLACEHOLDER"></script>'


def main() -> int:
    if not CORPUS.exists():
        print(f"building {CORPUS.name} ...")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_web_data.py")],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
            return 1
        CORPUS.write_text(result.stdout)

    template = TEMPLATE.read_text()
    corpus = CORPUS.read_text()
    if PLACEHOLDER not in template:
        print(f"{TEMPLATE.name} has no corpus placeholder", file=sys.stderr)
        return 1
    # A stray </script> inside the data would close the tag early and break the page.
    if "</script>" in corpus:
        print("corpus data contains </script>", file=sys.stderr)
        return 1

    OUTPUT.write_text(template.replace(PLACEHOLDER, "<script>\n" + corpus + "\n</script>"))
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
