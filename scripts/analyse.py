#!/usr/bin/env python3
"""Aggregate JSONL result files into a single CSV summary table.

Walks --results, reads the `_meta` and `_summary` records from each JSONL
file, and writes one row per (task, provider, model, pair, subset, shots).

Usage:
    python scripts/analyse.py --results results/ --out results/summary.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_jsonl_meta_and_summary(path: Path) -> dict | None:
    meta = summary = None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in rec:
                meta = rec["_meta"]
            elif "_summary" in rec:
                summary = rec["_summary"]
    if meta is None or summary is None:
        return None
    return {**meta, **summary, "_path": str(path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/summary.csv"))
    args = ap.parse_args()

    rows = []
    for jsonl in sorted(args.results.rglob("*.jsonl")):
        rec = _read_jsonl_meta_and_summary(jsonl)
        if rec is not None:
            rows.append(rec)
    if not rows:
        print(f"No result files found under {args.results}")
        return 1

    df = pd.DataFrame(rows)
    cols = [
        "task", "provider", "model", "pair", "subset", "shots", "n",
        "pearson", "spearman", "kendall",
        "bleu", "chrf", "ter", "comet_kiwi",
        "segment_pearson", "segment_spearman",
        "span_precision_macro", "span_recall_macro", "span_f1_macro",
        "_path",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values(["task", "pair", "provider", "model", "shots"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows → {args.out}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
