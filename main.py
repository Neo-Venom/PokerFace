#!/usr/bin/env python3
"""RakshaGuard CLI entrypoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from raksha.pipeline import run_pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="RakshaGuard privacy firewall")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    return run_pipeline(Path(args.input), Path(args.output))


if __name__ == "__main__":
    sys.exit(main())
