#!/usr/bin/env python3
"""annotator_agreement.py — inter-annotator agreement on the direct assessments.

The release carries per-annotator DA scores in `da_scores` with `n_annotators`, but no annotator
identifiers, so the raters are interchangeable and not crossed across segments. That rules out any
statistic needing rater identity (Cohen's kappa, ICC(2,k), per-rater correlation) and leaves two
that are correct for this design:

  Krippendorff's alpha, interval        built from within-unit disagreement against the overall
                                        spread, so rater identity is never used
  ICC(1), one-way random effects        the model for "each unit rated by k raters drawn from a
                                        population", which is exactly this design

Both are reported per pair and pooled. Also reported: mean within-segment standard deviation and
mean pairwise absolute difference, which are on the DA scale and easier to interpret than a
coefficient.

Guard: `_selftest` checks alpha against three cases with known answers before any real data is
touched, because a silently wrong agreement statistic is worse than none.

Writes results/analysis/annotator_agreement.csv and splices tab:iaa into acl_latex.tex.
"""
from __future__ import annotations

import ast
import glob
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

TEX = _PAPER_TEX
DATA = ROOT / "dataset" / "hf_snapshot" / "data" / "full"
PAIR_ORDER = ["en-hi", "en-mr", "en-ta", "en-te", "en-gu", "en-ml", "et-en", "ne-en", "si-en"]


def as_list(v) -> list[float]:
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return []
    try:
        return [float(x) for x in v if x is not None and np.isfinite(float(x))]
    except (TypeError, ValueError):
        return []


def krippendorff_alpha_interval(units: list[list[float]]) -> float:
    """Interval alpha over units with >=2 ratings each. Rater identity is not used."""
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float("nan")
    do_num = 0.0
    n = 0
    allv: list[float] = []
    for u in units:
        m = len(u)
        a = np.asarray(u, float)
        # sum of squared differences over ordered pairs = 2 * sum over i<j
        sq = m * (a * a).sum() - a.sum() ** 2      # == sum_{i<j}(x_i-x_j)^2
        do_num += (2.0 * sq) / (m - 1)
        n += m
        allv.extend(u)
    if n < 2:
        return float("nan")
    Do = do_num / n
    b = np.asarray(allv, float)
    tot = len(b) * (b * b).sum() - b.sum() ** 2    # == sum_{i<j}(x_i-x_j)^2 over all values
    De = (2.0 * tot) / (n * (n - 1))
    return float("nan") if De == 0 else 1.0 - Do / De


def icc1(units: list[list[float]]) -> float:
    """ICC(1), one-way random effects, for unbalanced groups."""
    units = [u for u in units if len(u) >= 2]
    if len(units) < 2:
        return float("nan")
    k = np.array([len(u) for u in units], float)
    means = np.array([np.mean(u) for u in units])
    N = k.sum()
    grand = sum(sum(u) for u in units) / N
    msb = (k * (means - grand) ** 2).sum() / (len(units) - 1)
    ssw = sum(((np.asarray(u, float) - m) ** 2).sum() for u, m in zip(units, means))
    dfw = N - len(units)
    if dfw <= 0:
        return float("nan")
    msw = ssw / dfw
    k0 = (N - (k ** 2).sum() / N) / (len(units) - 1)
    den = msb + (k0 - 1) * msw
    return float("nan") if den == 0 else float((msb - msw) / den)


def icc1k(icc: float, k: float) -> float:
    """Spearman-Brown step up: reliability of the MEAN of k raters.

    alpha and ICC(1) score a single rater, but the label a model is trained and evaluated against
    is `da_mean`. ICC(1,k) is therefore the ceiling on how well any model can correlate with the
    target, and it is the number a benchmark user needs. Averaging raises reliability, so a pair
    can have poor single-rater agreement and still carry a usable mean.
    """
    if not np.isfinite(icc) or k <= 1:
        return float("nan")
    den = 1 + (k - 1) * icc
    return float("nan") if den == 0 else float(k * icc / den)


def _selftest() -> None:
    """Three cases with answers known a priori."""
    # perfect agreement -> alpha 1, ICC 1
    perfect = [[10.0, 10.0], [50.0, 50.0], [90.0, 90.0], [20.0, 20.0]]
    assert abs(krippendorff_alpha_interval(perfect) - 1.0) < 1e-9, "alpha != 1 on perfect agreement"
    assert abs(icc1(perfect) - 1.0) < 1e-9, "ICC != 1 on perfect agreement"
    # no signal: every unit spans the same range -> alpha ~ 0 or below
    rng = np.random.default_rng(0)
    noise = [list(rng.uniform(0, 100, 3)) for _ in range(4000)]
    a = krippendorff_alpha_interval(noise)
    assert abs(a) < 0.05, f"alpha should be ~0 on pure noise, got {a:.3f}"
    # systematic: raters differ by a constant, units well separated -> high but < 1
    off = [[v, v + 5] for v in np.linspace(0, 100, 200)]
    a2 = krippendorff_alpha_interval(off)
    assert 0.9 < a2 < 1.0, f"alpha should be high-but-not-1 for a constant offset, got {a2:.3f}"
    # Spearman-Brown must raise reliability and stay bounded
    assert abs(icc1k(0.5, 4) - 0.8) < 1e-9, "ICC(1,k) wrong for icc=0.5, k=4"
    assert icc1k(0.2, 3) > 0.2 and icc1k(0.99, 3) < 1.0, "ICC(1,k) out of bounds"
    print("self-test passed: alpha=1 on perfect, ~0 on noise, 0.9-1.0 on constant offset; "
          "ICC(1,k) steps up correctly")


def splice(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(% BEGIN GENERATED {re.escape(name)}[^\n]*\n).*?"
                     rf"(% END GENERATED {re.escape(name)})", re.S)
    if not pat.search(text):
        raise SystemExit(f"sentinels for {name} not found in {TEX.name}")
    return pat.sub(lambda m: m.group(1) + body + m.group(2), text)


def main() -> int:
    _selftest()
    df = pd.concat([pd.read_parquet(p) for p in sorted(DATA.glob("*.parquet"))], ignore_index=True)
    pc = "language_pair" if "language_pair" in df.columns else "pair"
    df["scores"] = df.da_scores.map(as_list)

    rows, all_units = [], []
    for pair in PAIR_ORDER:
        sub = df[df[pc] == pair]
        units = [u for u in sub.scores if len(u) >= 2]
        if not units:
            continue
        all_units.extend(units)
        sds = [float(np.std(u, ddof=1)) for u in units]
        pw = [float(np.mean([abs(a - b) for i, a in enumerate(u) for b in u[i + 1:]]))
              for u in units]
        rows.append(dict(pair=pair, n_segments=len(sub), n_multi=len(units),
                         mean_k=float(np.mean([len(u) for u in units])),
                         alpha=krippendorff_alpha_interval(units), icc1=icc1(units),
                         mean_sd=float(np.mean(sds)), mean_pairwise_absdiff=float(np.mean(pw))))
        rows[-1]["icc1k"] = icc1k(rows[-1]["icc1"], rows[-1]["mean_k"])
        print(f"  {pair}  n={len(units):>6}  k={rows[-1]['mean_k']:.2f}  "
              f"alpha={rows[-1]['alpha']:.3f}  ICC(1)={rows[-1]['icc1']:.3f}  "
              f"ICC(1,k)={rows[-1]['icc1k']:.3f}  sd={rows[-1]['mean_sd']:.1f}")

    out = pd.DataFrame(rows)
    pooled = dict(pair="pooled", n_segments=int(out.n_segments.sum()), n_multi=len(all_units),
                  mean_k=float(np.mean([len(u) for u in all_units])),
                  alpha=krippendorff_alpha_interval(all_units), icc1=icc1(all_units),
                  mean_sd=float(np.mean([np.std(u, ddof=1) for u in all_units])),
                  mean_pairwise_absdiff=float(np.mean(
                      [np.mean([abs(a - b) for i, a in enumerate(u) for b in u[i + 1:]])
                       for u in all_units])))
    pooled["icc1k"] = icc1k(pooled["icc1"], pooled["mean_k"])
    print(f"  POOLED  n={pooled['n_multi']:,}  alpha={pooled['alpha']:.3f}  "
          f"ICC(1)={pooled['icc1']:.3f}  ICC(1,k)={pooled['icc1k']:.3f}")
    out = pd.concat([out, pd.DataFrame([pooled])], ignore_index=True)
    out.to_csv(ROOT / "results/analysis/annotator_agreement.csv", index=False)

    L = []
    for _, r in out.iterrows():
        nm = r"\textbf{pooled}" if r.pair == "pooled" else r.pair
        f = lambda v: f"{int(v):,}".replace(",", "{,}")
        L.append(f"{nm} & ${f(r.n_multi)}$ & ${r.mean_k:.2f}$ & ${r.alpha:.3f}$ & "
                 f"${r.icc1:.3f}$ & ${r.icc1k:.3f}$ & ${r.mean_sd:.1f}$ \\\\\n")
        if r.pair == "en-ml":
            L.append("\\midrule\n")

    tex = TEX.read_text()
    want = splice(tex, "tab_iaa", "".join(L))
    if want == tex:
        print("\nacl_latex.tex is up to date")
        return 0
    if "--check" in sys.argv:
        print("\nOUT OF DATE")
        return 1
    TEX.write_text(want)
    print(f"\nspliced tab:iaa into {TEX.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
