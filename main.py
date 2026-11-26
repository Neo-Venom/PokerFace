#!/usr/bin/env python3
"""Project RAKSHA CLI entrypoint — Team Poker Face."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Project RAKSHA privacy firewall")
    parser.add_argument("--input", required=True, help="Read-only input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    return run_pipeline(Path(args.input), Path(args.output))


if __name__ == "__main__":
    sys.exit(main())
