#!/usr/bin/env python3
r"""verify_fullda.py — recompute tab:fullda and its population from the scoring files.

tab:fullda is hand-written: twenty-seven correlations and a population, none of them produced by a
generator and none of them checked. The population, $81{,}315$, appears three times in the paper
(Appendix E prose, tab:fullda's caption, tab:direction's second block) and appears in no result
file, so an audit that greps the evidence cannot confirm it. It turns out to be right, and this
script is what makes that reproducible: 81,315 is the number of DA-bearing rows that the
full-population scoring pass actually covers, and all three metrics cover exactly the same rows.

It is not the 111,974 DA-bearing rows of Table 1, and the paper used to say "the whole release",
which invited the reader to expect 111,974. Five pairs are scored short. The paper now says so and
this script fails if the shortfall changes.

    python3 analysis/verify_fullda.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
# The paper source lives in its own repository and is optional here; see
# analysis/paper_source.py for the search order and the env vars that override it.
sys.path.insert(0, str(ROOT / "analysis"))
from paper_source import find_tex as _find_tex  # noqa: E402
_PAPER_TEX = _find_tex() or (ROOT / "paper" / "latex" / "acl_latex.tex")

TEX = _PAPER_TEX
PAIRS = ["en-hi", "en-mr", "en-ta", "en-ml", "en-gu", "en-te", "et-en", "ne-en", "si-en"]
#: metric file stem -> the column heading it fills in tab:fullda
METRICS = {"cometkiwi-xl-reffree": r"\mCKXL", "xcomet-xl-reffree": r"\mXC",
           "metricx24-xl-qe": "MetricX"}


def scores(stem: str, pair: str) -> dict[str, float]:
    f = ROOT / "results" / "qe_metrics_full" / f"{stem}.{pair.replace('-', '')}.jsonl"
    out: dict[str, float] = {}
    if not f.exists():
        return out
    for line in f.open(encoding="utf-8"):
        r = json.loads(line)
        if "_meta" in r or "_summary" in r or r.get("id") is None:
            continue
        v = r.get("pred_score", r.get("pred"))
        if v is not None:
            out[r["id"]] = float(v)
    return out


def main() -> int:
    full = pd.concat([pd.read_parquet(f) for f in
                      sorted(glob.glob(str(ROOT / "dataset/hf_snapshot/data/full/*.parquet")))],
                     ignore_index=True)
    gold = full[full.has_da].set_index("uid").da_mean.to_dict()
    da_n = full[full.has_da].language_pair.value_counts().to_dict()

    flat = " ".join(TEX.read_text().split())
    fails: list[str] = []
    populations: set[int] = set()

    for stem, macro in METRICS.items():
        row = []
        n_metric = 0
        for pair in PAIRS:
            p = scores(stem, pair)
            ids = [u for u in p if u in gold]
            n_metric += len(ids)
            if len(ids) < 30:
                fails.append(f"{stem} {pair}: only {len(ids)} scored DA rows")
                continue
            rho = spearmanr([gold[u] for u in ids], [p[u] for u in ids]).statistic
            printed = f"{rho:.3f}"
            want = printed.replace("-", "$-") + ("$" if rho < 0 else "")
            if want not in flat:
                fails.append(f"{macro} {pair}: recomputed {printed}, not in the tex")
            row.append(f"{pair} {rho:+.3f}")
        populations.add(n_metric)
        print(f"  {macro:8s} n={n_metric}  " + "  ".join(row))

    if len(populations) != 1:
        fails.append(f"the three metrics cover different populations: {sorted(populations)}")
    else:
        n = populations.pop()
        print(f"\n  population covered by all three: {n}")
        if f"${n:,}$".replace(",", "{,}") not in flat:
            fails.append(f"the population {n} is not stated in the tex")

    # the shortfall against Table 1, which is what the reader will try to reconcile
    print("\n  coverage against the DA-bearing rows of Table 1")
    short = []
    for pair in PAIRS:
        got = len([u for u in scores("cometkiwi-xl-reffree", pair) if u in gold])
        have = int(da_n.get(pair, 0))
        mark = "" if got == have else "   short"
        print(f"    {pair}  {got:6d} of {have:6d}{mark}")
        if got != have:
            short.append((pair, got, have))
    for pair, got, have in short:
        s = f"{pair} ${got:,}$ of ${have:,}$".replace(",", "{,}")
        if s not in flat:
            fails.append(f"the shortfall on {pair} ({got} of {have}) is not stated in the tex")

    for f_ in fails:
        print(f"FAIL: {f_}")
    if fails:
        return 1
    print("\ntab:fullda, its population and its shortfall all reproduce")
    return 0


if __name__ == "__main__":
    sys.exit(main())
