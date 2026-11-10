#!/usr/bin/env python3
"""Re-derive every QE `pred` from the stored `raw` model text using the current (hardened)
parse_qe_score, in place. No model re-run / no API spend — `raw` is the source of truth.
Drops stale `_summary` lines so analyse_challenge recomputes correlations from corrected preds.

Usage:  python scripts/reparse_responses.py [results_dir]   (default: results)
"""
import json, glob, os, sys
from indicqe_ape.prompts import parse_qe_score

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
files = sorted(glob.glob(os.path.join(ROOT, "qe_open", "*.jsonl"))) + \
        sorted(glob.glob(os.path.join(ROOT, "qe_closed", "*.jsonl")))

total_changed = 0
for f in files:
    out, changed = [], 0
    for ln in open(f):
        ln = ln.rstrip("\n")
        try:
            r = json.loads(ln)
        except Exception:
            out.append(ln); continue
        if "_summary" in r:
            continue  # recomputed downstream
        raw = r.get("raw")
        if isinstance(raw, dict):
            old = r.get("pred") or {}
            new = {t: parse_qe_score(v) for t, v in raw.items()}
            for t, nv in new.items():
                ov = old.get(t)
                if (ov is None) != (nv is None) or (ov is not None and nv is not None and abs(ov - nv) > 1e-9):
                    changed += 1
            r["pred"] = new
        out.append(json.dumps(r, ensure_ascii=False))
    with open(f, "w") as fh:
        fh.write("\n".join(out) + "\n")
    if changed:
        print(f"  {os.path.basename(f)}: {changed} preds corrected")
    total_changed += changed
print(f"TOTAL preds corrected: {total_changed} (raw responses untouched; summaries dropped for recompute)")
