#!/usr/bin/env python3
"""fewshot_paired.py — does few-shot prompting lower QE correlation, or only lower compliance?

The 0-shot and 4-shot correlations in the QE table are computed over the segments each condition
managed to parse. Those populations differ (0-shot parses 100% for most models, 4-shot as little as
74%), so a drop in correlation confounds two things: the model scoring worse, and the model scoring
a different, self-selected subset.

This script separates them. For each model and pair it computes:
  rho_0_all   0-shot correlation on all segments it parsed
  rho_4_all   4-shot correlation on all segments it parsed        (what the paper currently reports)
  rho_0_int   0-shot correlation restricted to segments BOTH conditions parsed
  rho_4_int   4-shot correlation on the same intersection          (the paired comparison)

If rho_4_int is still below rho_0_int, few-shot genuinely degrades scoring. If the gap closes, the
apparent degradation was a population effect.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DIRS = [ROOT / "results" / "qe_open", ROOT / "results" / "qe_closed"]
OUT = ROOT / "results" / "analysis" / "fewshot_paired.md"
NAME = re.compile(r"^(?P<model>.+)\.s(?P<shots>\d+)\.(?P<pair>[a-z]+)\.jsonl$")
sys.path.insert(0, str(ROOT / "analysis"))
from parse_rate_audit import scores_of  # noqa: E402


def load(f: Path, prompt="gemba-da") -> dict:
    """{segment_key: score} for the given prompt, skipping unparsed rows."""
    out, i = {}, 0
    for line in f.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "_meta" in r and len(r) == 1:
            continue
        key = r.get("id") or r.get("uid") or r.get("item_id") or i
        i += 1
        s = scores_of(r)
        v = s.get(prompt, s.get("score"))
        g = r.get("gold_da_mean", r.get("gold", r.get("da_mean")))
        try:
            g = float(g) if g is not None and str(g).strip() != "" else None
        except (TypeError, ValueError):
            g = None
        if v is not None and g is not None:
            out[key] = (float(v), g)
    return out


def main(prompt="gemba-da"):
    files = {}
    for d in DIRS:
        if not d.exists():
            continue
        for f in d.glob("*.jsonl"):
            m = NAME.match(f.name)
            if m:
                files[(m["model"], int(m["shots"]), m["pair"])] = f

    models = sorted({k[0] for k in files})
    rows = []
    for mod in models:
        pairs = sorted({k[2] for k in files if k[0] == mod and k[1] == 4})
        for p in pairs:
            f0, f4 = files.get((mod, 0, p)), files.get((mod, 4, p))
            if not (f0 and f4):
                continue
            d0, d4 = load(f0, prompt), load(f4, prompt)
            if not d0 or not d4:
                continue
            inter = sorted(set(d0) & set(d4))
            if len(inter) < 30:
                continue
            def rho(d, keys=None):
                ks = keys if keys is not None else sorted(d)
                v = np.array([d[k][0] for k in ks]); g = np.array([d[k][1] for k in ks])
                if len(set(v)) < 2 or len(set(g)) < 2:
                    return np.nan
                return spearmanr(v, g)[0]
            rows.append({"model": mod, "pair": p,
                         "n0": len(d0), "n4": len(d4), "n_int": len(inter),
                         "rho0_all": rho(d0), "rho4_all": rho(d4),
                         "rho0_int": rho(d0, inter), "rho4_int": rho(d4, inter)})
    if not rows:
        print("no paired data found (records may lack a gold field)"); return 1
    df = pd.DataFrame(rows)
    L = [f"# Few-shot: scoring effect vs compliance effect ({prompt})\n\n",
         "`_all` uses every segment a condition parsed. `_int` restricts both conditions to the "
         "segments they both parsed, which is the paired comparison.\n\n",
         "| model | pairs | n int | rho0 all | rho4 all | rho0 int | rho4 int | drop (all) | drop (paired) |\n",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|\n"]
    print(f"{'model':<26}{'n_int':>8}{'r0_all':>8}{'r4_all':>8}{'r0_int':>8}{'r4_int':>8}"
          f"{'d_all':>8}{'d_pair':>8}")
    for mod, g in df.groupby("model"):
        m0a, m4a = g.rho0_all.mean(), g.rho4_all.mean()
        m0i, m4i = g.rho0_int.mean(), g.rho4_int.mean()
        print(f"{mod:<26}{int(g.n_int.sum()):>8}{m0a:>8.3f}{m4a:>8.3f}{m0i:>8.3f}{m4i:>8.3f}"
              f"{m4a-m0a:>8.3f}{m4i-m0i:>8.3f}")
        L.append(f"| {mod} | {len(g)} | {int(g.n_int.sum()):,} | {m0a:.3f} | {m4a:.3f} | "
                 f"{m0i:.3f} | {m4i:.3f} | {m4a-m0a:+.3f} | **{m4i-m0i:+.3f}** |\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(L))
    df.to_csv(ROOT / "results" / "analysis" / "fewshot_paired.csv", index=False)
    print(f"\nwritten {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "gemba-da"))
