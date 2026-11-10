#!/usr/bin/env python3
"""Convenience launcher: `python scripts/run_ape.py [args]` → `indicqe-ape ape [args]`."""
from __future__ import annotations

import sys

from indicqe_ape.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["ape", *sys.argv[1:]]))
