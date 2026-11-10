#!/usr/bin/env python3
"""wmt_metrics.py — WMT-standard meta-evaluation for the saved per-instance predictions.

Implements what the WMT Metrics task now uses, so the QE numbers sit on the current protocol:

  acc_eq   pairwise accuracy with tie calibration (Deutsch et al. 2023, EMNLP). For every pair of
           segments the metric is credited when it orders them as the human does, and credited for
           predicting a tie when the humans tie. The metric tie threshold eps is tuned to maximise
           accuracy, which is what makes metrics that do and do not predict ties comparable.
  boot     bootstrap percentile confidence intervals over segments.
  williams Williams test for the difference between two dependent correlations (same segments),
           which is the test WMT uses when ranking two metrics on one test set.
  mcc      Matthews correlation for word-level OK/BAD, the standard word-level QE statistic.

Usage:
  python3 analysis/wmt_metrics.py --runs qe-A-cometkiwi-full qe-E5-minmax
  python3 analysis/wmt_metrics.py --wordqe          # MCC for the word-tag validation
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr, t as tdist

ROOT = Path(__file__).resolve().parents[1]
CAMP = ROOT / "experiments" / "comet_qe" / "results" / "campaign"


# ---------------------------------------------------------------- pairwise accuracy + tie calib.
def _pair_signs(x: np.ndarray, eps: float) -> np.ndarray:
    d = x[:, None] - x[None, :]
    s = np.zeros_like(d, dtype=np.int8)
    s[d > eps] = 1
    s[d < -eps] = -1
    return s


def acc_eq(gold: np.ndarray, pred: np.ndarray, n_eps: int = 60):
    """Pairwise accuracy with tie calibration. Returns (accuracy, chosen eps).

    Humans tie when gold values are exactly equal. The metric ties when |diff| <= eps. eps is
    swept and the maximising value reported, per Deutsch et al. (2023).
    """
    iu = np.triu_indices(len(gold), k=1)
    gd = (gold[:, None] - gold[None, :])[iu]          # computed once
    pd_ = (pred[:, None] - pred[None, :])[iu]         # computed once
    gv = np.sign(gd).astype(np.int8)
    absd = np.abs(pd_)
    nz = absd[absd > 0]
    cands = np.unique(np.concatenate(
        [[0.0], np.quantile(nz, np.linspace(0, 0.5, n_eps))])) if nz.size else np.array([0.0])
    best, best_eps = -1.0, 0.0
    for e in cands:
        pv = np.sign(pd_) * (absd > e)                 # ties where |diff| <= eps
        a = float((pv == gv).mean())
        if a > best:
            best, best_eps = a, float(e)
    return best, best_eps


def boot_ci(gold, pred, fn, n=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    N = len(gold)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        try:
            vals.append(fn(gold[idx], pred[idx]))
        except Exception:
            pass
    if not vals:
        return (np.nan, np.nan)
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def williams(r12, r13, r23, n):
    """Williams test: is r12 different from r13, given both share variable 1 and r23 relates 2,3."""
    if min(abs(r12), abs(r13)) >= 1 or n < 4:
        return np.nan, np.nan
    R = 1 - r12**2 - r13**2 - r23**2 + 2 * r12 * r13 * r23
    if R <= 0:
        return np.nan, np.nan
    num = (r12 - r13) * np.sqrt((n - 1) * (1 + r23))
    den = np.sqrt(2 * (n - 1) / (n - 3) * R + ((r12 + r13) ** 2 / 4) * (1 - r23) ** 3)
    tval = num / den
    p = 2 * (1 - tdist.cdf(abs(tval), n - 3))
    return float(tval), float(p)


def mcc(gold: list[str], pred: list[str], pos="BAD"):
    g = np.array([1 if x == pos else 0 for x in gold])
    p = np.array([1 if x == pos else 0 for x in pred])
    tp = int(((g == 1) & (p == 1)).sum()); tn = int(((g == 0) & (p == 0)).sum())
    fp = int(((g == 0) & (p == 1)).sum()); fn = int(((g == 1) & (p == 0)).sum())
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


# ---------------------------------------------------------------- drivers
def run_predictions(runs):
    out = []
    for r in runs:
        f = CAMP / "outputs" / r / "predictions_test.csv"
        if not f.exists():
            print(f"  {r}: no predictions_test.csv"); continue
        d = pd.read_csv(f).dropna(subset=["pred", "target"])
        gold = d["target"].to_numpy(float); pred = d["pred"].to_numpy(float)
        # acc_eq is O(n^2) in pairs; subsample for the tie sweep, keep full n for correlations
        if len(gold) > 4000:
            sub = np.random.default_rng(0).choice(len(gold), 4000, replace=False)
            g_s, p_s = gold[sub], pred[sub]
        else:
            g_s, p_s = gold, pred
        sp = spearmanr(gold, pred)[0]
        pe = pearsonr(gold, pred)[0]
        a, eps = acc_eq(g_s, p_s)
        lo, hi = boot_ci(gold, pred, lambda g, p: spearmanr(g, p)[0], n=300)
        print(f"  {r:<28} n={len(d):>5} rho={sp:.3f} [{lo:.3f},{hi:.3f}] r={pe:.3f} "
              f"acc_eq={a:.3f} (eps={eps:.4f})")
        out.append({"run": r, "n": len(d), "spearman": sp, "sp_lo": lo, "sp_hi": hi,
                    "pearson": pe, "acc_eq": a, "eps": eps})
    return pd.DataFrame(out)


def compare(a_run, b_run):
    fa = CAMP / "outputs" / a_run / "predictions_test.csv"
    fb = CAMP / "outputs" / b_run / "predictions_test.csv"
    if not (fa.exists() and fb.exists()):
        print("  missing predictions for comparison"); return
    A = pd.read_csv(fa)[["item_id", "target", "pred"]].rename(columns={"pred": "a"})
    B = pd.read_csv(fb)[["item_id", "pred"]].rename(columns={"pred": "b"})
    m = A.merge(B, on="item_id").dropna()
    r12 = pearsonr(m.target, m.a)[0]; r13 = pearsonr(m.target, m.b)[0]
    r23 = pearsonr(m.a, m.b)[0]
    tv, p = williams(r12, r13, r23, len(m))
    print(f"  {a_run} r={r12:.3f} vs {b_run} r={r13:.3f} (n={len(m)}, r_ab={r23:.3f}) "
          f"Williams t={tv:.2f} p={p:.4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    a = ap.parse_args()
    runs = a.runs or sorted(d.name for d in (CAMP / "outputs").iterdir()
                            if (d / "predictions_test.csv").exists())
    print(f"WMT-standard meta-evaluation over {len(runs)} runs\n")
    df = run_predictions(runs)
    if a.compare:
        print()
        compare(*a.compare)
    o = ROOT / "results" / "analysis" / "wmt_metrics.csv"
    o.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(o, index=False)
    print(f"\nwritten {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
