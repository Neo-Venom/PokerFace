#!/usr/bin/env python3
"""Project RAKSHA CLI entrypoint — Team Poker Face."""
from __future__ import annotations

import os
import sys

# Locale independence: under a bare C/POSIX locale with UTF-8 mode disabled, Python decodes file names as ASCII,
# so Devanagari file names (e.g. "पर्ची_रमेश.txt") could not be checked for PII or written to report.json. Re-launch
# once in UTF-8 mode before anything else runs (the secret is still in the environment; nothing has been read yet).
if sys.getfilesystemencoding().lower().replace("-", "") != "utf8" and not os.environ.get("RAKSHA_UTF8_REEXEC"):
    os.environ["RAKSHA_UTF8_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable, "-X", "utf8", os.path.abspath(__file__), *sys.argv[1:]])

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

from src.pipeline import run_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Project RAKSHA privacy firewall")
    parser.add_argument("--input", required=True, help="Read-only input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    return run_pipeline(Path(args.input), Path(args.output))


if __name__ == "__main__":
    sys.exit(main())
