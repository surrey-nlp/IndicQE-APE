"""Gate-A: is the segment↔system inversion a *calibration* artefact?

Three things, all on the challenge test set, shots=0:

  (1) MECHANISM (non-oracle). Per system, fit a per-pair affine  pred = a_l + b_l * gold
      by OLS, then correlate the intercept a_l against the per-pair mean gold DA across
      the 8 pairs. A *negative* a_l~mean-gold correlation is the direct evidence that the
      metric's per-language offset moves opposite to quality — the cause of the inversion.

  (2) ORACLE RECALIBRATION (labelled oracle / held-out). Split each pair's items 50/50;
      fit the per-pair affine on split A (held-out DA), apply on split B; recompute the
      system-level ranking (per-pair mean pred vs mean gold) on split B, BEFORE vs AFTER.
      Recovery of a negative system Pearson to strongly positive isolates the inversion as
      a calibration failure, not a resolution failure. It is an oracle upper bound (uses
      per-language DA to calibrate) — reported as such.

  (3) ROBUSTNESS. Bootstrap CI over the 8 language pairs for the raw system-level Pearson
      and pairwise accuracy, to show the *direction* of the inversion is stable at n=8.

Reads results/{qe_metrics,qe_closed,qe_open}/*.jsonl exactly like scripts/meta_evaluation.py.
Writes results/analysis/recalibration.csv and prints a summary.

Usage:  python scripts/recalibration_analysis.py [results_dir] [n_boot]

WHY THE RECALIBRATED COLUMN IS NO LONGER REPORTED (2026-08-09)
The oracle recalibration fits gold ~ a + b*pred per pair on a random half and applies it to the
other half. That forces the per-pair mean of the fitted values onto the per-pair mean of gold, so
the resulting correlation is a property of the procedure and not of the system. `--null` measures
exactly this by feeding the same procedure predictors of known quality:

    pure Gaussian noise, no signal      raw +0.14   recalibrated +0.996
    noise, different seed               raw -0.08   recalibrated +0.998
    perfectly ANTI-correlated with gold raw -1.000  recalibrated +1.000
    a good predictor                    raw +1.000  recalibrated +1.000

A random scorer recalibrates to 0.996 and an inverted one to 1.000, so "every system reaches 1.0
after recalibration" says nothing about any system. tab:recal now reports raw r with its bootstrap
interval and the offset-versus-quality correlation, which is the mechanism and is computed directly
from the per-pair fits. Run `python scripts/recalibration_analysis.py --null` to reproduce.
"""
from __future__ import annotations

import csv
import collections
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
NBOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
RNG = np.random.default_rng(0)
#: 8-pair default reproduces the published tab:recal. Set IQE_PAIRS to add enml for the nine-pair
#: refresh, matching scripts/meta_evaluation.py. tab:qe is already nine pairs, so the two tables
#: currently rest on different populations until this is re-run.
# NINE-pair default. The paper's tab:recal and the ABSTRACT quote the nine-pair run
# (offset-vs-quality -0.442, recal-before -0.221, after 0.997). The previous eight-pair default
# produced materially different numbers (-0.646 / -0.649) whose bootstrap interval EXCLUDES zero,
# contradicting the paper's own hedge that cometkiwi-xl's raw coefficient is not identified.
# Anyone running this as documented got the contradicting numbers. Set IQE_PAIRS to reproduce
# the superseded eight-pair control.
PAIRS = os.environ.get(
    "IQE_PAIRS", "enhi,enmr,enta,engu,ente,enml,eten,neen,sien").split(",")
#: trained-metric directory, as in meta_evaluation.py: qe_metrics is the original 8-pair run on the
#: pre-extension challenge set, qe_metrics_ext the 9-pair run on the current extended challenge.
METRICS_DIR = os.environ.get("IQE_METRICS_DIR", "qe_metrics_ext")
#: output name carries the pair count by default, so an 8-pair control and a
#: 9-pair refresh cannot overwrite each other -- they did, and only the log of
#: the control survived.
OUT_CSV = os.environ.get(
    "IQE_RECAL_OUT", f"results/analysis/recalibration_{len(PAIRS)}pair.csv")
TMPL = "gemba-da"
SHOTS = "0"
# focus systems: the inversion victim, the trained baseline, the calibrated LLM, a strong ranker
FOCUS = ["cometkiwi-xl", "cometkiwi-da", "xcomet-xl", "gpt-5.5", "sarvam-m"]


def _load_metric_items():
    """metric-name -> pair -> list[(pred_score, gold)]  (trained metrics)."""
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in sorted(glob.glob(f"{ROOT}/{METRICS_DIR}/*.jsonl")):
        base = os.path.basename(f).replace(".jsonl", "")
        try:
            name, pair = base.rsplit(".", 1)
        except ValueError:
            continue
        if pair not in PAIRS:
            continue
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if "_meta" in r or "_summary" in r:
                continue
            g, p = r.get("gold_da_mean"), r.get("pred_score")
            if g is not None and p is not None:
                data[name][pair].append((float(p), float(g)))
    return data


def _load_judge_items(tmpl=TMPL):
    """judge-name -> pair -> list[(pred, gold)]  (LLM judges, gemba-da, shots=0)."""
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in sorted(glob.glob(f"{ROOT}/qe_open/*.jsonl")) + sorted(glob.glob(f"{ROOT}/qe_closed/*.jsonl")):
        base = os.path.basename(f).replace(".jsonl", "")
        try:
            name, srest = base.split(".s", 1)
            shot, pair = srest.split(".", 1)
        except ValueError:
            continue
        if shot != SHOTS or pair not in PAIRS:
            continue
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if "_meta" in r or "_summary" in r:
                continue
            g = r.get("gold_da_mean")
            pred = r.get("pred")
            p = pred.get(tmpl) if isinstance(pred, dict) else None
            if g is not None and p is not None:
                data[name][pair].append((float(p), float(g)))
    return data


def _norm(name: str) -> str:
    n = name.lower()
    for k in FOCUS:
        if k in n:
            return k
    return n


def pairwise_accuracy(pred_means: list[float], gold_means: list[float]) -> float:
    n = len(pred_means)
    tot = correct = 0
    for i in range(n):
        for j in range(i + 1, n):
            dp, dg = pred_means[i] - pred_means[j], gold_means[i] - gold_means[j]
            if dg == 0:
                continue
            tot += 1
            if np.sign(dp) == np.sign(dg):
                correct += 1
    return correct / tot if tot else float("nan")


def affine_fit(pred: np.ndarray, gold: np.ndarray) -> tuple[float, float]:
    """OLS gold ~ a + b*pred  (map metric score onto the DA scale)."""
    if len(pred) < 3 or np.std(pred) < 1e-9:
        return float(np.mean(gold)) if len(gold) else 0.0, 0.0
    b, a = np.polyfit(pred, gold, 1)  # returns slope, intercept
    return float(a), float(b)


def analyse_system(name: str, by_pair: dict) -> dict | None:
    pairs = [p for p in PAIRS if p in by_pair and len(by_pair[p]) >= 5]
    if len(pairs) < 4:
        return None

    # ---- raw system-level (per-pair means) ----
    raw_pred_mean, gold_mean, intercepts = [], [], []
    for p in pairs:
        arr = np.array(by_pair[p], float)
        pr, gd = arr[:, 0], arr[:, 1]
        raw_pred_mean.append(float(pr.mean()))
        gold_mean.append(float(gd.mean()))
        # per-pair intercept a_l from pred = a_l + b_l * gold  (offset in metric space)
        b, a = np.polyfit(gd, pr, 1) if np.std(gd) > 1e-9 else (0.0, float(pr.mean()))
        intercepts.append(float(a))

    raw_pearson = float(stats.pearsonr(raw_pred_mean, gold_mean).statistic)
    raw_pa = pairwise_accuracy(raw_pred_mean, gold_mean)
    # mechanism: does the per-language offset move against quality?
    a_vs_quality = float(stats.pearsonr(intercepts, gold_mean).statistic)

    # ---- oracle held-out recalibration (50/50 per pair) ----
    recal_pred_mean, base_pred_mean, base_gold_mean = [], [], []
    for p in pairs:
        arr = np.array(by_pair[p], float)
        idx = RNG.permutation(len(arr))
        half = len(idx) // 2
        A, Bset = arr[idx[:half]], arr[idx[half:]]
        if len(A) < 3 or len(Bset) < 3:
            continue
        a, b = affine_fit(A[:, 0], A[:, 1])           # gold ~ a + b*pred on split A
        recal_B = a + b * Bset[:, 0]                  # apply to split B
        recal_pred_mean.append(float(recal_B.mean()))
        base_pred_mean.append(float(Bset[:, 0].mean()))
        base_gold_mean.append(float(Bset[:, 1].mean()))

    before_pearson = float(stats.pearsonr(base_pred_mean, base_gold_mean).statistic)
    after_pearson = float(stats.pearsonr(recal_pred_mean, base_gold_mean).statistic)
    before_pa = pairwise_accuracy(base_pred_mean, base_gold_mean)
    after_pa = pairwise_accuracy(recal_pred_mean, base_gold_mean)

    # ---- bootstrap CI over the 8 pairs on the raw system Pearson & PA ----
    boot_p, boot_pa = [], []
    rpm, gm = np.array(raw_pred_mean), np.array(gold_mean)
    n = len(rpm)
    for _ in range(NBOOT):
        s = RNG.integers(0, n, n)
        if np.std(rpm[s]) < 1e-9 or np.std(gm[s]) < 1e-9:
            continue
        boot_p.append(stats.pearsonr(rpm[s], gm[s]).statistic)
        boot_pa.append(pairwise_accuracy(list(rpm[s]), list(gm[s])))
    lo, hi = (np.nanpercentile(boot_p, [2.5, 97.5]) if boot_p else (float("nan"),) * 2)

    return {
        "system": name, "n_pairs": len(pairs),
        "raw_sys_pearson": round(raw_pearson, 3),
        "raw_sys_PA": round(raw_pa, 3),
        "boot_pearson_lo": round(float(lo), 3), "boot_pearson_hi": round(float(hi), 3),
        "offset_vs_quality_corr": round(a_vs_quality, 3),
        "recal_before_pearson": round(before_pearson, 3),
        "recal_after_pearson": round(after_pearson, 3),
        "recal_before_PA": round(before_pa, 3),
        "recal_after_PA": round(after_pa, 3),
    }



def null_test(reps: int = 20, seed: int = 0) -> None:
    """Feed the recalibration predictors of known quality. See the module docstring."""
    import pandas as pd
    from pathlib import Path
    rng = np.random.default_rng(seed)
    challenge = (Path(__file__).resolve().parents[1] / "dataset" / "hf_snapshot" / "data"
                 / "challenge" / "test.parquet")
    g = pd.read_parquet(challenge)
    g = g[g.has_da]

    def one(pred_fn, label):
        before, after = [], []
        for _ in range(reps):
            bp, bg, rp = [], [], []
            for _pair, sub in g.groupby("language_pair"):
                gold = sub.da_mean.to_numpy()
                pred = pred_fn(sub, rng)
                idx = rng.permutation(len(gold))
                h = len(idx) // 2
                A, B = idx[:h], idx[h:]
                b, a = np.polyfit(pred[A], gold[A], 1)
                rp.append(float((a + b * pred[B]).mean()))
                bp.append(float(pred[B].mean()))
                bg.append(float(gold[B].mean()))
            before.append(stats.pearsonr(bp, bg).statistic)
            after.append(stats.pearsonr(rp, bg).statistic)
        print(f"  {label:36s} raw {np.mean(before):+.3f}   recalibrated {np.mean(after):+.3f}")

    print("null test: the same oracle recalibration, applied to predictors of known quality")
    one(lambda s, r: r.normal(size=len(s)), "pure noise, no signal")
    one(lambda s, r: r.normal(size=len(s)), "pure noise, different draw")
    one(lambda s, r: -s.da_mean.to_numpy() + r.normal(0, 5, len(s)), "anti-correlated with gold")
    one(lambda s, r: s.da_mean.to_numpy() + r.normal(0, 5, len(s)), "a good predictor")
    print("\nA random scorer recalibrates to about 0.996, so the recalibrated column cannot "
          "distinguish\nsystems and is not reported. tab:recal gives raw r and the offset instead.")


def main():
    systems: dict[str, dict[str, list]] = {}
    both = list(_load_metric_items().items()) + list(_load_judge_items().items())
    for name, bp in both:
        key = _norm(name)
        if key not in FOCUS:
            continue
        dst = systems.setdefault(key, {})
        for p, v in bp.items():
            dst.setdefault(p, []).extend(v)

    rows = []
    for name in FOCUS:
        if name not in systems:
            continue
        res = analyse_system(name, systems[name])
        if res:
            rows.append(res)

    if not rows:
        print("No systems found — check results dir:", ROOT)
        return
    cols = list(rows[0].keys())
    outp = OUT_CSV
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # pretty print
    print(f"\nGate-A recalibration analysis  (challenge test, shots=0, gemba-da; n_boot={NBOOT})\n")
    hdr = ["system", "n_pairs", "raw_sys_pearson", "boot_pearson_lo", "boot_pearson_hi",
           "offset_vs_quality_corr", "recal_before_pearson", "recal_after_pearson",
           "recal_before_PA", "recal_after_PA"]
    widths = {h: max(len(h), 8) for h in hdr}
    print("  ".join(h.rjust(widths[h]) for h in hdr))
    for r in rows:
        print("  ".join(str(r[h]).rjust(widths[h]) for h in hdr))
    print(f"\nwrote {outp}")
    print("\nReading guide:")
    print("  raw_sys_pearson < 0 with CI not crossing 0  → inversion is real & robust")
    print("  offset_vs_quality_corr < 0                  → mechanism: per-language offset moves against quality")
    print("  recal_after_pearson >> recal_before         → inversion is a CALIBRATION artefact (oracle upper bound)")


if __name__ == "__main__":
    if "--null" in sys.argv:
        null_test()
        raise SystemExit(0)
    main()
