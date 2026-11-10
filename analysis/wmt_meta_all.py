#!/usr/bin/env python3
"""wmt_meta_all.py — WMT-protocol meta-evaluation over EVERY saved per-instance prediction.

analysis/wmt_metrics.py covers the trained-head ablations only. This covers all three
system families on one protocol so the paper's tables are comparable:

  1. trained metrics   results/qe_metrics{,_ext}/<metric>.<pair>.jsonl     (pred_score)
  2. prompted LLMs     results/qe_{open,closed}/<model>.s<k>.<pair>.jsonl  (pred[template])
  3. trained head      experiments/.../outputs/<run>/predictions_test.csv

Protocol, following Freitag et al. (WMT Metrics findings) and Deutsch et al. 2023:

  * Correlations are computed PER LANGUAGE PAIR and then averaged. Pooling segments across
    pairs manufactures correlation out of per-pair mean differences, which is the very
    pathology section 5 of the paper measures. A pooled figure is also reported, labelled
    as such, so the gap between the two is visible rather than hidden.
  * Pairwise accuracy uses tie calibration (acc_eq), because a metric that never ties is
    otherwise incomparable with one that does.
  * Bootstrap percentile CIs over segments, per pair.
  * Parse rate is carried alongside every prompted cell: a correlation over 22% of the
    segments is not comparable to one over 100%, and reporting it without the rate invites
    exactly that mistake.

Outputs
  results/analysis/wmt_meta_all.csv   one row per (family, system, template, shots, pair)
  results/analysis/wmt_meta_all.md    per-pair and macro-average tables
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from wmt_metrics import acc_eq, boot_ci  # noqa: E402

CAMP = ROOT / "experiments" / "comet_qe" / "results" / "campaign"
OUT_CSV = ROOT / "results" / "analysis" / "wmt_meta_all.csv"
OUT_MD = ROOT / "results" / "analysis" / "wmt_meta_all.md"

MIN_N = 50           # below this a per-pair correlation is noise, not a measurement
#: acc_eq is O(n^2) in segment PAIRS and sweeps the tie threshold over that array, so cost grows
#: as n^2 * n_eps. At n=3000 with 60 candidates that is ~270M operations per cell across ~350
#: cells, which is slow enough to be killed by a session reaper. 1200 segments give ~720k pairs,
#: which is ample for a stable accuracy estimate, and the seed is fixed so it is reproducible.
ACC_CAP = 1200
ACC_EPS_STEPS = 25
NBOOT = 200
#: Written incrementally so a long run is resumable and a kill costs only the current cell.
PART_CSV = ROOT / "results" / "analysis" / "wmt_meta_all.partial.csv"


def _rows(path: Path):
    """Yield data records from a jsonl, skipping _meta/_summary lines."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in r or "_summary" in r:
                continue
            yield r


def score(gold: np.ndarray, pred: np.ndarray, seed: int = 0) -> dict:
    """Correlations, tie-calibrated pairwise accuracy, bootstrap CI. All on the parsed subset."""
    m = np.isfinite(gold) & np.isfinite(pred)
    g, p = gold[m], pred[m]
    if len(g) < MIN_N or np.std(g) == 0 or np.std(p) == 0:
        return {"n": int(len(g)), "spearman": np.nan, "pearson": np.nan,
                "acc_eq": np.nan, "eps": np.nan, "sp_lo": np.nan, "sp_hi": np.nan}
    if len(g) > ACC_CAP:
        idx = np.random.default_rng(seed).choice(len(g), ACC_CAP, replace=False)
        a, eps = acc_eq(g[idx], p[idx], n_eps=ACC_EPS_STEPS)
    else:
        a, eps = acc_eq(g, p, n_eps=ACC_EPS_STEPS)
    lo, hi = boot_ci(g, p, lambda x, y: spearmanr(x, y)[0], n=NBOOT, seed=seed)
    return {"n": int(len(g)), "spearman": float(spearmanr(g, p)[0]),
            "pearson": float(pearsonr(g, p)[0]), "acc_eq": float(a), "eps": float(eps),
            "sp_lo": lo, "sp_hi": hi}


def collect_trained_metrics() -> list[dict]:
    out = []
    for d in ("qe_metrics", "qe_metrics_ext"):
        for f in sorted((ROOT / "results" / d).glob("*.jsonl")):
            metric, pair, _ = f.name.split(".", 2)
            recs = list(_rows(f))
            if not recs:
                continue
            gold = np.array([r.get("gold_da_mean", np.nan) for r in recs], float)
            pred = np.array([r.get("pred_score", np.nan) for r in recs], float)
            n_parsed = int(np.isfinite(pred).sum())
            out.append({"family": "trained-metric", "system": metric, "template": "-",
                        "shots": "-", "pair": pair, "source": d,
                        "n_rows": len(recs), "n_parsed": n_parsed,
                        "parse_rate": n_parsed / max(len(recs), 1),
                        **score(gold, pred)})
    return out


def collect_prompted() -> list[dict]:
    out = []
    pat = re.compile(r"^(?P<model>.+)\.s(?P<shots>\d+)\.(?P<pair>\w+)\.jsonl$")
    for d in ("qe_open", "qe_closed", "qe_open_rerun", "qe_closed_rerun"):
        base = ROOT / "results" / d
        if not base.exists():
            continue
        for f in sorted(base.glob("*.jsonl")):
            m = pat.match(f.name)
            if not m:
                continue
            recs = list(_rows(f))
            if not recs:
                continue
            gold = np.array([r.get("gold_da_mean", np.nan)
                             if r.get("gold_da_mean") is not None else np.nan
                             for r in recs], float)
            templates = sorted({t for r in recs for t in (r.get("pred") or {})})
            for tpl in templates:
                pred = np.array([(r.get("pred") or {}).get(tpl, np.nan)
                                 if (r.get("pred") or {}).get(tpl) is not None else np.nan
                                 for r in recs], float)
                n_parsed = int(np.isfinite(pred).sum())
                out.append({"family": "prompted", "system": m["model"], "template": tpl,
                            "shots": m["shots"], "pair": m["pair"], "source": d,
                            "n_rows": len(recs), "n_parsed": n_parsed,
                            "parse_rate": n_parsed / max(len(recs), 1),
                            **score(gold, pred)})
    return out


def collect_head() -> list[dict]:
    """Trained-head cells, tagged with the build that produced them.

    `qe_indic_enindic.csv` was rebuilt on 2026-07-28; the campaign predictions came from the
    previous build of that same filename. `num_train` distinguishes them (64,416 campaign vs
    64,671 current) and is carried into the output so no downstream table can silently average
    across two populations with different test sets (8,160 vs 8,294 segments).
    """
    out = []
    outputs = CAMP / "outputs"
    if not outputs.exists():
        return out
    builds: set = set()
    for d in sorted(outputs.iterdir()):
        f = d / "predictions_test.csv"
        if not f.exists():
            continue
        sm = d / "run_summary.json"
        n_train = json.loads(sm.read_text()).get("num_train") if sm.exists() else None
        builds.add(n_train)
        t = pd.read_csv(f)
        for pair, g in t.groupby("language_pair"):
            out.append({"family": "trained-head", "system": d.name, "template": "-",
                        "shots": "-", "pair": str(pair).replace("-", ""), "source": "campaign",
                        "num_train": n_train,
                        "n_rows": len(g), "n_parsed": int(g["pred"].notna().sum()),
                        "parse_rate": float(g["pred"].notna().mean()),
                        **score(g["target"].to_numpy(float), g["pred"].to_numpy(float))})
    if len(builds) > 1:
        print(f"  WARNING: campaign outputs span {len(builds)} builds {sorted(builds, key=str)}. "
              f"Head rows are tagged with num_train; do NOT average across them.")
    else:
        print(f"  all campaign runs from one build: num_train={builds.pop() if builds else None}")
    return out


def dedupe_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (family, system, template, shots, pair).

    `qe_metrics` is the original EIGHT-pair run on the pre-extension challenge set (24 files,
    no en-ml). `qe_metrics_ext` is a COMPLETE NINE-pair re-run on the current challenge test
    (27 files); it is not a partial three-pair patch, and every one of its files differs
    numerically from its `qe_metrics` counterpart (e.g. cometkiwi-da.enhi 1,883 lines vs 1,844).
    Both are on disk, so a naive groupby averages eight pairs twice and reports `pairs=9` over
    17 rows. Prefer the extended run throughout, and record what was dropped rather than
    dropping it silently.
    """
    pref = {"qe_metrics_ext": 0, "qe_metrics": 1, "qe_closed_rerun": 0, "qe_open_rerun": 0,
            "qe_closed": 1, "qe_open": 1, "campaign": 0}
    keys = ["family", "system", "template", "shots", "pair"]
    d = df.copy()
    d["_pref"] = d["source"].map(pref).fillna(9)
    d = d.sort_values("_pref")
    dropped = d[d.duplicated(keys, keep="first")]
    if len(dropped):
        print(f"deduped {len(dropped)} superseded cells:")
        for _, r in dropped.iterrows():
            print(f"  dropped {r['system']} {r['pair']} from {r['source']} (n={r['n']})")
    return d.drop_duplicates(keys, keep="first").drop(columns="_pref")


def macro(df: pd.DataFrame) -> pd.DataFrame:
    """WMT aggregation: mean over per-pair values. Never pool segments across pairs."""
    keys = ["family", "system", "template", "shots"]
    g = dedupe_pairs(df[df.spearman.notna()]).groupby(keys)
    agg = g.agg(pairs=("pair", "nunique"), n=("n", "sum"),
                macro_spearman=("spearman", "mean"), macro_pearson=("pearson", "mean"),
                macro_acc_eq=("acc_eq", "mean"), min_parse=("parse_rate", "min"),
                mean_parse=("parse_rate", "mean")).reset_index()
    return agg.sort_values("macro_spearman", ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*",
                    default=["trained-metric", "prompted", "trained-head"])
    a = ap.parse_args()

    rows = []
    for fam, fn in (("trained-metric", collect_trained_metrics),
                    ("prompted", collect_prompted),
                    ("trained-head", collect_head)):
        if fam not in a.families:
            continue
        got = fn()
        rows += got
        print(f"{fam}: {len(got)} cells", flush=True)
        # checkpoint after each family so a kill mid-run does not lose completed work
        PART_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(PART_CSV, index=False)

    if not rows:
        print("no predictions found", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nper-cell rows: {len(df)} -> {OUT_CSV}")

    agg = macro(df)
    L = ["# WMT-protocol meta-evaluation, all systems\n\n",
         "Per-pair correlations averaged (never pooled). Pairwise accuracy is tie-calibrated ",
         "(Deutsch et al. 2023). `min parse` is the worst per-pair output-format compliance for ",
         "that system: a correlation computed over a low parse rate rests on a self-selected ",
         "subset and is not comparable to one at 1.000.\n\n",
         f"Cells with fewer than {MIN_N} scorable segments are excluded.\n\n"]

    for fam, g in agg.groupby("family"):
        L.append(f"## {fam}\n\n")
        L.append("| system | prompt | shots | pairs | segs | macro rho | macro r | macro acc_eq | min parse |\n")
        L.append("|---|---|--:|--:|--:|--:|--:|--:|--:|\n")
        for _, r in g.iterrows():
            L.append(f"| {r['system']} | {r['template']} | {r['shots']} | {r['pairs']} | "
                     f"{int(r['n']):,} | **{r['macro_spearman']:.3f}** | {r['macro_pearson']:.3f} | "
                     f"{r['macro_acc_eq']:.3f} | {r['min_parse']:.3f} |\n")
        L.append("\n")

    L.append("## Pooled vs macro (the aggregation trap)\n\n")
    L.append("Pooling segments across pairs inflates apparent agreement because per-pair mean ")
    L.append("differences add between-pair variance that no metric had to earn. Reported here ")
    L.append("for the trained metrics so the size of the effect is on the record.\n\n")
    L.append("| system | macro rho (per-pair mean) | pooled rho | inflation |\n|---|--:|--:|--:|\n")
    dd = dedupe_pairs(df[df.spearman.notna()])
    for sysname in sorted(dd[dd.family == "trained-metric"].system.unique()):
        sub = dd[(dd.family == "trained-metric") & (dd.system == sysname)]
        # Pool over exactly the cells the macro average used, so the two figures differ only in
        # aggregation and not in population. Reading every file on disk would pool the superseded
        # qe_metrics rows back in and make the comparison meaningless.
        pooled_g, pooled_p = [], []
        for _, cell in sub.iterrows():
            f = ROOT / "results" / cell["source"] / f"{sysname}.{cell['pair']}.jsonl"
            if not f.exists():
                continue
            for r in _rows(f):
                if r.get("gold_da_mean") is not None and r.get("pred_score") is not None:
                    pooled_g.append(r["gold_da_mean"]); pooled_p.append(r["pred_score"])
        if len(pooled_g) < MIN_N:
            continue
        pooled = float(spearmanr(pooled_g, pooled_p)[0])
        mac = float(sub.spearman.mean())
        L.append(f"| {sysname} | {mac:.3f} | {pooled:.3f} | {pooled - mac:+.3f} |\n")

    OUT_MD.write_text("".join(L))
    print("".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
