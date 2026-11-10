#!/usr/bin/env python3
"""Convenience launcher: `python scripts/run_qe.py [args]` → `indicqe-ape qe [args]`."""
from __future__ import annotations

import sys

from indicqe_ape.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["qe", *sys.argv[1:]]))
