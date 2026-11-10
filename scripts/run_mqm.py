#!/usr/bin/env python3
"""Convenience launcher: `python scripts/run_mqm.py [args]` → `indicqe-ape mqm [args]`."""
from __future__ import annotations

import sys

from indicqe_ape.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["mqm", *sys.argv[1:]]))
