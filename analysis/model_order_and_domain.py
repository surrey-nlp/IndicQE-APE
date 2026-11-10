#!/usr/bin/env python3
"""model_order_and_domain.py — is anything true of a model independently of the language pair?

Consolidating over models and domains, can any conclusion be drawn about model performance
independently of the language pair? Two things are separable:

  ordering   does the ranking of the nine systems survive changing the pair? Measured as
             Kendall's W over the nine per-pair rankings, plus the mean pairwise Spearman
             between them, on the per-pair Spearman of Table 14's population.

  domain     does a domain make QE harder for everyone? Measured WITHIN pair and then averaged,
             never pooled: pooling segments from different pairs into one domain correlation is
             the aggregation artefact S3 is about, and would manufacture a domain effect out of
             the pairs that happen to carry a domain.

The X->en domain labels are inferred outside the language the classifier was fitted on and S2
conditions no analysis on them, so the domain half runs on en->Indic only.

    python3 analysis/model_order_and_domain.py           # write the CSVs and print the report
    python3 analysis/model_order_and_domain.py --check   # recompute and diff against them

Consumed by Appendix F.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import make_qe_tables as Q  # noqa: E402

SNAP = ROOT / "dataset" / "hf_snapshot" / "data" / "challenge" / "test.parquet"
OUT_ORDER = ROOT / "results" / "analysis" / "model_order_stability.csv"
OUT_DOMAIN = ROOT / "results" / "analysis" / "domain_within_pair.csv"

EN_INDIC = ["enhi", "enmr", "enta", "engu", "ente", "enml"]
#: a domain cell smaller than this is not a measurement, the floor used elsewhere in the paper
MIN_CELL = 30


def main() -> int:
    META = ROOT / "results" / "analysis" / "qe_meta_s0.csv"
    if not (ROOT / "results" / "qe_rerun512").is_dir() or not SNAP.exists() or not META.exists():
        print("SKIP: results/qe_rerun512, the release snapshot or qe_meta_s0.csv is not in this "
              "checkout, so this check cannot run.\n  The per-instance predictions ship in the "
              "evidence bundle (analysis/export_evidence_bundle.py).")
        return 0
    absent = [n for n, _m, dirs in Q.PROMPTED
              if not any((ROOT / "results" / d).is_dir() for d in dirs)]
    if absent:
        print(f"SKIP: per-instance outputs for {', '.join(absent)} are not in this checkout, so a "
              "partial ordering would be compared against the full one.")
        return 0

    loaded = {name: Q.load(name, dirs) for name, _m, dirs in Q.PROMPTED}

    # The ordering statistic must be computed on the numbers the paper prints, so the two sources
    # here are exactly the two that build tab:perpairfull: per-instance Kendall for the prompted
    # models, and the kendall_<pair> columns of qe_meta_s0.csv for the trained metrics.
    tau = {m: Q.per_pair_kendall(c) for m, c in loaded.items()}
    meta = pd.read_csv(META)
    meta = meta[meta.shots == 0].set_index("system")
    for name, _mac in Q.TRAINED:
        tau[name] = {p: float(meta.loc[name, f"kendall_{p}"]) for p in Q.PAIRS}

    # ---- ordering: does the ranking of systems survive changing the pair? ----
    mat = pd.DataFrame(tau).T.reindex(columns=Q.PAIRS)   # systems x pairs
    mat = mat.dropna(how="any")                          # a system must be scored on every pair
    ranks = mat.rank(ascending=False)                    # rank 1 = best system in that pair
    n, k = ranks.shape                                   # n systems, k pairs
    # Kendall's W over the k rankings of n systems
    S = ((ranks.sum(axis=1) - ranks.sum(axis=1).mean()) ** 2).sum()
    W = 12 * S / (k ** 2 * (n ** 3 - n))
    pairwise = [stats.spearmanr(ranks[a], ranks[b]).statistic
                for i, a in enumerate(Q.PAIRS) for b in Q.PAIRS[i + 1:]]
    order = pd.DataFrame({
        "system": mat.index,
        "macro_tau": mat.mean(axis=1).round(3).values,
        "best_rank": ranks.min(axis=1).astype(int).values,
        "worst_rank": ranks.max(axis=1).astype(int).values,
    })

    print(f"Ordering stability of {n} systems across {k} language pairs\n")
    print(order.to_string(index=False))
    print(f"\nKendall's W = {W:.3f} over the {k} per-pair rankings")
    print(f"mean pairwise Spearman between rankings = {np.mean(pairwise):.3f} "
          f"(min {np.min(pairwise):.3f}, max {np.max(pairwise):.3f})")
    moved = order[order.worst_rank - order.best_rank >= 3]
    print(f"systems moving 3 or more places between pairs: "
          f"{', '.join(moved.system) if len(moved) else 'none'}")

    # ---- domain: within pair, then averaged ----
    meta = pd.read_parquet(SNAP, columns=["uid", "language_pair", "domain"])
    rows = []
    for m, cells in loaded.items():
        for pair in EN_INDIC:
            df = cells.get(pair)
            if df is None:
                continue
            d = df.dropna(subset=["gold", "pred"]).merge(meta, left_on="id", right_on="uid")
            for dom, g in d.groupby("domain"):
                if len(g) < MIN_CELL or not g.pred.std() or not g.gold.std():
                    continue
                rows.append({"model": m, "pair": pair, "domain": dom, "n": len(g),
                             "rho": round(float(stats.spearmanr(g.pred, g.gold).statistic), 3)})
    dom = pd.DataFrame(rows).sort_values(["model", "pair", "domain"]).reset_index(drop=True)

    # average within pair first, then over pairs, then over models
    per_model = dom.groupby(["model", "domain"]).rho.mean().unstack()
    summary = pd.DataFrame({
        "pairs": dom.groupby("domain").pair.nunique(),
        "cells": dom.groupby("domain").size(),
        "mean_rho": per_model.mean().round(3),
        "sd_over_models": per_model.std().round(3),
    }).sort_values("mean_rho", ascending=False)
    print(f"\n\nWithin-pair Spearman by domain, en->Indic, cells of at least {MIN_CELL}\n")
    print(summary.to_string())
    span = summary.mean_rho.max() - summary.mean_rho.min()
    print(f"\nspread across all {len(summary)} domains: {span:.3f} "
          f"({summary.mean_rho.idxmax()} {summary.mean_rho.max():.3f} against "
          f"{summary.mean_rho.idxmin()} {summary.mean_rho.min():.3f})")

    # That spread is not a domain effect. News, finance, tech and sports are gold-only en-ml
    # labels (S2), so a domain carried by one pair is that pair wearing a domain's name and its
    # cell cannot be told apart from the pair's own difficulty. Only the domains that appear in
    # more than one pair separate the two.
    multi = summary[summary.pairs > 1]
    mspan = multi.mean_rho.max() - multi.mean_rho.min()
    print(f"restricted to the {len(multi)} domains carried by more than one pair: {mspan:.3f} "
          f"({multi.mean_rho.idxmax()} {multi.mean_rho.max():.3f} against "
          f"{multi.mean_rho.idxmin()} {multi.mean_rho.min():.3f})")
    print(f"  single-pair domains, all en-ml gold labels: "
          f"{', '.join(summary[summary.pairs == 1].index)}")
    # for scale: the same spread across pairs
    pair_span = mat[EN_INDIC].mean().max() - mat[EN_INDIC].mean().min()
    print(f"for comparison, the spread across the six en->Indic pairs is {pair_span:.3f}")

    if "--check" in sys.argv:
        for path, got in ((OUT_ORDER, order), (OUT_DOMAIN, dom)):
            if not path.exists():
                print(f"\nFAIL: {path.relative_to(ROOT)} is missing")
                return 1
            if not pd.read_csv(path).equals(got):
                print(f"\nFAIL: {path.relative_to(ROOT)} is out of date")
                return 1
        print("\n--check: both CSVs are up to date")
        return 0

    OUT_ORDER.parent.mkdir(parents=True, exist_ok=True)
    order.to_csv(OUT_ORDER, index=False)
    dom.to_csv(OUT_DOMAIN, index=False)
    print(f"\nwrote {OUT_ORDER.relative_to(ROOT)} and {OUT_DOMAIN.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
