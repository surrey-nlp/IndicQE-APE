#!/usr/bin/env python3
"""parse_rate_audit.py — output-format compliance for the prompted QE runs.

A prompted QE score only enters a correlation if the model emitted something parseable. If
compliance differs across models or shot counts, the correlations are computed on different
populations and are not comparable. WMT reports this; we did not. This script measures it.

Reads results/qe_open/*.jsonl and results/qe_closed/*.jsonl, named <model>.s<shots>.<pair>.jsonl.
Reports per model and shot count: rows, parsed, parse rate, and the per-pair minimum.
"""
from __future__ import annotations
import json, re, sys
from collections import defaultdict
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIRS = [ROOT / "results" / "qe_open", ROOT / "results" / "qe_closed"]
OUT = ROOT / "results" / "analysis" / "parse_rates.md"
NAME = re.compile(r"^(?P<model>.+)\.s(?P<shots>\d+)\.(?P<pair>[a-z]+)\.jsonl$")


def _num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        # a bare number, optionally with surrounding whitespace. Text in the target language
        # (the dominant few-shot failure mode) yields no digits and is counted as unparsed.
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        return float(m.group()) if m else None
    return None


def scores_of(rec: dict) -> dict:
    """Return {prompt_name: numeric score or None} for each prompt present in the record.

    GEMBA-DA and GEMBA-SQM are separate prompts and fail independently, so they are counted
    separately: one record can carry a valid DA score and an unparseable SQM score.
    """
    out = {}
    for k in ("pred_score", "score", "prediction", "pred", "raw", "response"):
        if k not in rec:
            continue
        v = rec[k]
        if isinstance(v, str) and v.strip().startswith("{"):
            try:
                v = json.loads(v.replace("'", '"'))
            except Exception:
                pass
        if isinstance(v, dict):
            for kk, vv in v.items():
                out.setdefault(str(kk).lower().replace("_", "-"), _num(vv))
        elif v is not None and "score" not in out:
            out["score"] = _num(v)
    return out


def score_of(rec: dict):
    s = scores_of(rec)
    for k in ("gemba-da", "da", "score"):
        if k in s:
            return s[k]
    return next(iter(s.values()), None)


def main():
    rows = []
    for d in DIRS:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.jsonl")):
            m = NAME.match(f.name)
            if not m:
                continue
            n = 0
            per = defaultdict(int)
            for line in f.open():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if "_meta" in rec and len(rec) == 1:
                    continue
                n += 1
                for k, v in scores_of(rec).items():
                    if v is not None:
                        per[k] += 1
            if n:
                base = {"model": m["model"], "shots": int(m["shots"]), "pair": m["pair"],
                        "n": n, "dir": d.name}
                for prompt in (set(per) or {"score"}):
                    rows.append({**base, "prompt": prompt, "parsed": per.get(prompt, 0),
                                 "rate": per.get(prompt, 0) / n})
    if not rows:
        print("no prediction files found"); return 1
    df = pd.DataFrame(rows)

    print(f"{'model/prompt':<34}{'shots':>6}{'files':>7}{'rows':>8}{'parsed':>8}{'rate':>8}{'min pair':>10}")
    L = ["# Prompted-QE output-format compliance\n\n",
         "Parse rate is the share of segments for which the model emitted a parseable score. "
         "Correlations are computed only over parsed segments, so a low rate means the reported "
         "number rests on a smaller and self-selected population.\n\n",
         "| model | prompt | shots | files | rows | parsed | parse rate | worst pair |\n",
         "|---|---|--:|--:|--:|--:|--:|---|\n"]
    for (mod, sh, prompt), g in df.groupby(["model", "shots", "prompt"]):
        rate = g.parsed.sum() / g.n.sum()
        worst = g.loc[g.rate.idxmin()]
        print(f"{mod+'/'+prompt:<34}{sh:>6}{len(g):>7}{g.n.sum():>8}{g.parsed.sum():>8}{rate:>8.3f}"
              f"{worst['pair']+' '+format(worst['rate'],'.2f'):>10}")
        L.append(f"| {mod} | {prompt} | {sh} | {len(g)} | {g.n.sum():,} | {g.parsed.sum():,} | "
                 f"**{rate:.3f}** | {worst['pair']} {worst['rate']:.2f} |\n")

    # 0-shot vs 4-shot delta per model
    L.append("\n## 0-shot vs 4-shot compliance\n\n| model | 0-shot | 4-shot | change |\n|---|--:|--:|--:|\n")
    print("\n0-shot vs 4-shot:")
    piv = df[df.prompt.isin(["gemba-da","score"])].groupby(["model", "shots"]).apply(
        lambda g: g.parsed.sum() / g.n.sum(), include_groups=False).unstack()
    for mod, r in piv.iterrows():
        z, f4 = r.get(0), r.get(4)
        if pd.notna(z) and pd.notna(f4):
            print(f"  {mod:<26} {z:.3f} -> {f4:.3f}  ({f4-z:+.3f})")
            L.append(f"| {mod} | {z:.3f} | {f4:.3f} | {f4-z:+.3f} |\n")
        elif pd.notna(z):
            L.append(f"| {mod} | {z:.3f} | not run | |\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(L))
    df.to_csv(ROOT / "results" / "analysis" / "parse_rates.csv", index=False)
    print(f"\nwritten {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
