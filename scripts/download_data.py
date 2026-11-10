#!/usr/bin/env python3
"""Download the unified IndicQE-APE dataset ONCE to local disk, then run everything offline.

Cluster pattern: run this on a login node (internet + HF token). Compute nodes then read
the local snapshot with no network — `indicqe_ape.data.DataLoader` picks it up via
`INDICQEAPE_DATA`. Public configs only (the gated en-ml / RL tree is never pulled).

Usage (login node, once):
  export INDICQEAPE_DATA=$SCRATCH/indicqeape/snapshot
  huggingface-cli login          # only needed for gated models; the dataset is public
  python3 scripts/download_data.py
"""
from __future__ import annotations

import os
import sys

REPO = os.environ.get("INDICQEAPE_REPO", "surrey-nlp/IndicQE-APE")
DEST = os.environ.get("INDICQEAPE_DATA")


def main() -> int:
    if not DEST:
        print("Set INDICQEAPE_DATA to the local snapshot dir, e.g. $SCRATCH/indicqeape/snapshot",
              file=sys.stderr)
        return 2
    from huggingface_hub import snapshot_download
    print(f">> downloading {REPO} -> {DEST} (public configs only)", file=sys.stderr)
    path = snapshot_download(repo_id=REPO, repo_type="dataset", local_dir=DEST,
                             allow_patterns=["data/**", "README.md"])
    import glob
    import pandas as pd
    files = glob.glob(os.path.join(path, "data", "**", "*.parquet"), recursive=True)
    total = sum(len(pd.read_parquet(f, columns=["uid"])) for f in files)
    print(f">> {len(files)} parquet files, {total} rows under {path}/data")
    print(f"Snapshot ready. Set INDICQEAPE_DATA={path} and HF_HUB_OFFLINE=1 for jobs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
