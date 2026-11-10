#!/usr/bin/env python3
r"""axis_reliability.py — how reliable is the label on each difficulty axis?

A rank correlation is bounded by the reliability of the thing it is correlated against. If the
annotators on a set of segments disagree more with each other than the segments differ from one
another, the mean of their scores carries no stable ordering and no model can be scored against it.
This script reports, per axis, the mean within-segment standard deviation of the released
per-annotator scores, the standard deviation of the segment means, Krippendorff's interval alpha
and ICC(1,k), the reliability of the mean.

It answers a different question from analysis/axis_contrast.py. That script asks whether systems
rank the flagged segments worse than equally-scored segments of the same pair. This one asks
whether the target on those segments is estimable at all.

    python3 analysis/axis_reliability.py
    python3 analysis/axis_reliability.py --check
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
from axis_contrast import AXES, jl, load_challenge  # noqa: E402

TEX = _PAPER_TEX
OUT = ROOT / "results" / "analysis" / "axis_reliability.csv"


def krippendorff_alpha_interval(units: list[list[float]]) -> float:
    """Interval alpha over units of unequal size, by the observed/expected disagreement form."""
    units = [np.asarray(u, float) for u in units if len(u) >= 2]
    if not units:
        return np.nan
    n_total = sum(len(u) for u in units)
    do = sum(float(np.sum((u[:, None] - u[None, :]) ** 2)) / (len(u) - 1) for u in units)
    do /= n_total
    # sum_{i,j} (x_i - x_j)^2 = 2 n sum x^2 - 2 (sum x)^2, so the full matrix is never formed
    allv = np.concatenate(units)
    de = (2 * n_total * float(np.sum(allv ** 2)) - 2 * float(np.sum(allv)) ** 2)
    de /= (n_total - 1) * n_total
    return 1.0 - do / de if de else np.nan


def icc1(units: list[list[float]]) -> float:
    units = [np.asarray(u, float) for u in units if len(u) >= 2]
    n = len(units)
    if n < 2:
        return np.nan
    k = np.array([len(u) for u in units], float)
    means = np.array([u.mean() for u in units])
    grand = np.concatenate(units).mean()
    msb = float(np.sum(k * (means - grand) ** 2) / (n - 1))
    msw = float(np.sum([np.sum((u - u.mean()) ** 2) for u in units]) / (k.sum() - n))
    kbar = (k.sum() - (k ** 2).sum() / k.sum()) / (n - 1)
    return (msb - msw) / (msb + (kbar - 1) * msw) if (msb + (kbar - 1) * msw) else np.nan


def main() -> int:
    check = "--check" in sys.argv
    g = load_challenge()
    g["scores"] = g.da_scores.map(lambda s: [float(x) for x in jl(s)
                                             if str(x).strip() not in ("", "nan", "None")])
    g["within_sd"] = g.scores.map(lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else np.nan)

    rows = []
    for name, sub in [("all", g)] + [(a, g[g[a].fillna(False).astype(bool)]) for a in AXES] + \
                     [("no axis", g[~g[AXES].fillna(False).astype(bool).any(axis=1)])]:
        units = [u for u in sub.scores if len(u) >= 2]
        i1 = icc1(units)
        kbar = float(np.mean([len(u) for u in units])) if units else np.nan
        i1k = (kbar * i1) / (1 + (kbar - 1) * i1) if i1 == i1 and i1 > -1 else np.nan
        rows.append({"set": name, "n": len(sub), "within_sd": sub.within_sd.mean(),
                     "between_sd": sub.da_mean.std(ddof=1),
                     "alpha": krippendorff_alpha_interval(units), "icc1k": i1k})
    R = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT, index=False)
    print(R.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Does attenuation explain the A4 contrast? Correct each correlation by the ceiling
    # sqrt(ICC(1,k)) of the set it was computed on, wherever both sides admit one.
    from axis_contrast import MIN_N, ORDER, Panel, load_preds  # noqa: E402
    P = load_preds()
    pred = {m: g.uid.map(P[m]).to_numpy(float) for m in ORDER}
    pan = Panel(g, seed=4242)
    sc = g.scores.to_numpy(object)

    def ceiling(idx):
        u = [x for x in sc[idx] if len(x) >= 2]
        if len(u) < MIN_N:
            return np.nan
        i1 = icc1(u)
        k = float(np.mean([len(x) for x in u]))
        v = (k * i1) / (1 + (k - 1) * i1) if i1 == i1 else np.nan
        return np.sqrt(v) if v == v and v > 0 else np.nan

    print("\nattenuation check on A4, where the reliability of the mean is estimable on both sides")
    raw, cor, done = [], [], []
    for pair, s in g.groupby("language_pair"):
        if s.A4.isna().all():
            continue
        sel = s.A4.fillna(False).astype(bool).to_numpy()
        on, off = s.index.to_numpy()[sel], s.index.to_numpy()[~sel]
        if len(on) < MIN_N or len(off) < MIN_N:
            continue
        ctls = pan.controls(on, off)
        c_on = ceiling(on)
        c_ct = float(np.nanmean([ceiling(c) for c in ctls]))
        if not (c_on == c_on and c_ct == c_ct):
            print(f"  {pair}: ICC(1,k) not estimable on the flagged set or its control")
            continue
        r, k = [], []
        for m in ORDER:
            o = pan.rho(on, pred[m])
            cs = [x for x in (pan.rho(c, pred[m]) for c in ctls) if x == x]
            if o != o or not cs:
                continue
            c = float(np.mean(cs))
            r.append(o - c)
            k.append(o / c_on - c / c_ct)
        if r:
            print(f"  {pair}: ceiling {c_on:.3f} on the flagged set against {c_ct:.3f} on the "
                  f"control, contrast {np.mean(r):+.3f} raw and {np.mean(k):+.3f} corrected")
            raw.append(np.mean(r))
            cor.append(np.mean(k))
            done.append(pair)
    if raw:
        print(f"  over {len(done)} pairs: {np.mean(raw):+.3f} raw, {np.mean(cor):+.3f} corrected, "
              f"negative on {sum(1 for v in cor if v < 0)}/{len(cor)}")

    body = "\n".join(
        (f"\\texttt{{{r.set}}}" if r.set in AXES else r.set)
        + f" & {int(r.n):,}".replace(",", "{,}")
        + f" & ${r.within_sd:.1f}$ & ${r.between_sd:.1f}$ & "
        + (f"$-{abs(r.alpha):.3f}$" if r.alpha < 0 else f"${r.alpha:.3f}$") + " & "
        + ("---" if pd.isna(r.icc1k) or r.icc1k < 0 else f"${r.icc1k:.3f}$") + r" \\"
        for _, r in R.iterrows()) + "\n"

    txt = TEX.read_text()
    pat = re.compile(r"(% BEGIN GENERATED tab_axisreliab[^\n]*\n).*?(% END GENERATED tab_axisreliab)",
                     re.S)
    if not pat.search(txt):
        print("note: tab_axisreliab sentinels are not in the tex")
        return 0
    want = pat.sub(lambda m: m.group(1) + body + m.group(2), txt)
    if check:
        if want != txt:
            print("FAIL: tab:axisreliab differs from the data")
            return 1
        return 0
    if want != txt:
        TEX.write_text(want)
        print("spliced tab:axisreliab")
    return 0


if __name__ == "__main__":
    sys.exit(main())
