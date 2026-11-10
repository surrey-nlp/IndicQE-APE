#!/usr/bin/env python3
"""Aggregate per-experiment JSONL outputs into a canonical, append-only results table.

Designed for distributed evaluation: laptop runs and University-GPU runs each
write JSONL files into `results/`, push to git, and any clone can rebuild the
shared table by running this script. Deduplication is keyed by an
``experiment_id`` deterministically derived from the `_meta` block of each
JSONL — so re-running the same config never doubles a row.

Outputs:
- ``results/summary.tsv`` — wide table, one row per experiment, hand-edit safe
- ``results/summary.jsonl`` — same content as JSONL (machine-readable)
- ``results/summary.md`` — pretty Markdown table for quick scanning

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --results /path/to/results --out /path/to/out
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


METRIC_KEYS = (
    "pearson", "spearman", "kendall",
    "bleu", "chrf", "ter", "comet_kiwi",
    "segment_pearson", "segment_spearman",
    "span_precision_macro", "span_recall_macro", "span_f1_macro",
    "n", "n_macro", "n_attempted", "n_parsed", "n_parse_failures",
)

META_KEYS = ("task", "subset", "pair", "shots", "provider", "model", "limit", "seed", "fewshot_signature")


def experiment_id(meta: dict) -> str:
    """Stable id: hash of (task, subset, pair, shots, provider, model, seed, fewshot_signature)."""
    payload = {k: meta.get(k) for k in ("task", "subset", "pair", "shots", "provider", "model", "seed", "fewshot_signature")}
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def read_jsonl(path: Path) -> dict | None:
    meta: dict | None = None
    summary: dict | None = None
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
    if not meta or not summary:
        return None
    eid = experiment_id(meta)
    return {"experiment_id": eid, **{k: meta.get(k) for k in META_KEYS}, **{k: summary.get(k) for k in METRIC_KEYS}, "_path": str(path)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate JSONL results into TSV/MD/JSONL summary tables.")
    ap.add_argument("--results", type=Path, default=Path("results"), help="root directory to walk (default: results/)")
    ap.add_argument("--out", type=Path, default=Path("results"), help="output directory for summary.{tsv,md,jsonl}")
    ap.add_argument("--strict", action="store_true", help="fail if any JSONL lacks _meta or _summary")
    args = ap.parse_args()

    rows: list[dict] = []
    skipped: list[str] = []
    for jsonl in sorted(args.results.rglob("*.jsonl")):
        if jsonl.name.startswith("summary"):
            continue
        rec = read_jsonl(jsonl)
        if rec is None:
            skipped.append(str(jsonl))
            continue
        rows.append(rec)
    if not rows:
        print(f"No usable result files under {args.results}", file=sys.stderr)
        if skipped:
            print(f"Skipped (no _meta/_summary): {len(skipped)} files.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["experiment_id"], keep="last")
    df = df.sort_values(["task", "pair", "subset", "provider", "model", "shots"]).reset_index(drop=True)

    args.out.mkdir(parents=True, exist_ok=True)
    tsv_path = args.out / "summary.tsv"
    jsonl_path = args.out / "summary.jsonl"
    md_path = args.out / "summary.md"

    df.to_csv(tsv_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in df.to_dict(orient="records"):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    md_path.write_text(_to_markdown(df), encoding="utf-8")

    print(f"Aggregated {len(df)} experiments → {tsv_path}")
    if skipped:
        print(f"Note: skipped {len(skipped)} JSONL files lacking _meta or _summary.", file=sys.stderr)
        if args.strict:
            return 2
    print(df.to_string(index=False, max_colwidth=40))
    return 0


def _to_markdown(df: pd.DataFrame) -> str:
    """Render the most useful columns as a tidy markdown table per task."""
    if df.empty:
        return "_No results yet._\n"
    lines: list[str] = ["# IndicQE-APE — Results summary", ""]
    for task, sub in df.groupby("task", sort=False):
        lines.append(f"## task: {task}")
        # Pick task-relevant metric columns
        if task == "qe":
            cols = ["pair", "subset", "provider", "model", "shots", "n", "pearson", "spearman", "kendall", "n_parse_failures"]
        elif task == "ape":
            cols = ["pair", "subset", "provider", "model", "shots", "n", "bleu", "chrf", "ter", "comet_kiwi"]
        elif task == "mqm":
            cols = ["pair", "subset", "provider", "model", "shots", "n", "n_macro", "segment_pearson", "span_f1_macro"]
        else:
            cols = list(sub.columns)
        cols = [c for c in cols if c in sub.columns]
        sub2 = sub[cols].copy()
        # Round floats for readability
        for c in sub2.columns:
            if pd.api.types.is_float_dtype(sub2[c]):
                sub2[c] = sub2[c].round(3)
        lines.append(sub2.to_markdown(index=False))
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
