#!/usr/bin/env python3
r"""axis_head_contrast.py — the difficulty-axis contrast repeated on the trained COMET head.

The panel of S3 is nine systems nobody in this paper trained. This script asks the same question of
the head of S5, whose per-instance predictions are saved, so the axis result is tested once on
off-the-shelf systems and once on a model fitted to this benchmark's own training split.

Each head covers six en-to-Indic pairs in one run and the three X-to-en pairs in another; the two
are read together because the contrast is computed inside a pair and never across pairs.

    python3 analysis/axis_head_contrast.py
    python3 analysis/axis_head_contrast.py --check
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# The paper source lives in its own repository and is optional here; see
# analysis/paper_source.py for the search order and the env vars that override it.
sys.path.insert(0, str(ROOT / "analysis"))
from paper_source import find_tex as _find_tex  # noqa: E402
_PAPER_TEX = _find_tex() or (ROOT / "paper" / "latex" / "acl_latex.tex")

sys.path.insert(0, str(ROOT / "analysis"))
from axis_contrast import AXES, Panel, load_challenge  # noqa: E402

TEX = _PAPER_TEX
OUT = ROOT / "results" / "analysis" / "axis_head_contrast.csv"
RUNS = ROOT / "experiments" / "comet_qe" / "results" / "campaign" / "outputs"

HEADS = {"funnel only": ("qe-J-A0-joint", "qe-X-A0-xen"),
         "with extra features": ("qe-J-ref-joint", "qe-X-ref-xen")}

# The ablation build covers five en-to-Indic pairs, so it carries A1 on all five and the other
# three axes on en-hi, en-mr and en-ta. Varying the encoder, the loss and the target scale asks
# whether any design choice escapes the A4 contrast.
ABLATION = {"full features (reference)": "qe-A-cometkiwi-full",
            "min--max target": "qe-A0-cometkiwi-min",
            "CLS and funnel": "qe-E3-upper",
            "COMET-DA encoder": "qe-E1-cometda",
            "XLM-R encoder": "qe-E1-xlmr",
            "CCC loss": "qe-E4-ccc",
            "MSE loss": "qe-E4-mse"}


def load(runs) -> dict[str, float]:
    d = pd.concat([pd.read_csv(RUNS / r / "predictions_test.csv") for r in runs])
    return dict(zip(d.item_id, d.pred.astype(float)))


def main() -> int:
    check = "--check" in sys.argv
    g = load_challenge()
    pan = Panel(g, seed=20260813)
    preds = {k: g.uid.map(load(v)).to_numpy(float) for k, v in HEADS.items()}
    for k, y in preds.items():
        n = int((~np.isnan(y)).sum())
        print(f"{k}: {n} of {len(g)} challenge segments joined on uid")
        if n != len(g):
            print("  REFUSING: the head does not cover the challenge test")
            return 1

    C = pan.cells(preds, list(HEADS))
    C.to_csv(OUT, index=False)
    rows = []
    for k in HEADS:
        y = preds[k]
        overall = float(np.mean([pan.rho(g.index.to_numpy()[(g.language_pair == p).to_numpy()], y)
                                 for p in g.language_pair.unique()]))
        print(f"\n=== {k} === mean within-pair rho {overall:.3f}")
        for a in AXES:
            s = C[(C.axis == a) & (C.system == k)]
            if s.empty:
                continue
            pm = s.groupby("pair").matched.mean()
            pl = s.groupby("pair").plain.mean()
            rows.append({"config": k, "axis": a, "pairs": len(pm), "plain": pl.mean(),
                         "matched": pm.mean(), "neg": int((pm < 0).sum())})
            print(f"  {a}: plain {pl.mean():+.3f}  matched {pm.mean():+.3f}, "
                  f"negative in {int((pm < 0).sum())}/{len(pm)} pairs   "
                  + " ".join(f"{p}:{v:+.2f}" for p, v in pm.items()))
    R = pd.DataFrame(rows)

    ab = {k: g.uid.map(load([v])).to_numpy(float) for k, v in ABLATION.items()}
    A = Panel(g, seed=20260814).cells(ab, list(ABLATION))   # its own draws, order-independent
    print("\n=== ablation build, five pairs === A4 against the DA-matched control")
    means = {}
    for k in ABLATION:
        s = A[(A.system == k) & (A.axis == "A4")].groupby("pair").matched.mean()
        means[k] = float(s.mean())
        print(f"  {k:28s} {means[k]:+.3f}, negative on {int((s < 0).sum())}/{len(s)} pairs")
    print(f"  all {len(means)} negative; from {min(means.values()):+.3f} to "
          f"{max(means.values()):+.3f}, min--max target at {means['min--max target']:+.3f}")

    a4 = R[R.axis == "A4"]
    fails = []
    flat = " ".join(TEX.read_text().split())
    for _, r in a4.iterrows():
        if f"${r.matched:.3f}$" not in flat:
            fails.append(f"the tex does not carry the head's A4 contrast {r.matched:.3f} ({r.config})")
    if not re.search(r"negative on every pair for both", flat) and int(a4.neg.min()) == int(a4.pairs.min()):
        fails.append("the tex does not state that the head's A4 contrast is negative on every pair")
    for f in fails:
        print(f"{'FAIL' if check else 'note'}: {f}")
    return 1 if (check and fails) else 0


if __name__ == "__main__":
    sys.exit(main())
