#!/usr/bin/env python3
"""MetricX-24 QE correlations from the already-scored predictions, using the SAME methodology as
comet_qe.py (per-pair Pearson/Spearman/Kendall of model score vs DA; cross-lingual = Pearson of
per-pair mean score vs per-pair mean DA). MetricX-24 outputs an ERROR score (lower = better), so the
QE "quality score" is qe = -prediction, making it directly comparable to the CometKiwi/XCOMET rows.
Writes runs/metricx_qe/metrics.json. CPU only."""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau

WORK = os.environ.get("IQE_WORK") or sys.exit("set IQE_WORK to the overnight working directory (holds code/, runs/ and the *_output.jsonl files)")
SRC = f"{WORK}/metricx_qe_output.jsonl"
OUT = f"{WORK}/runs/metricx_qe"
os.makedirs(OUT, exist_ok=True)

d = pd.DataFrame(json.loads(l) for l in open(SRC, encoding="utf-8"))
d["qe"] = -pd.to_numeric(d["prediction"])          # error -> quality (higher = better)
d["da"] = pd.to_numeric(d["gold_da"])
d = d.dropna(subset=["qe", "da"])

per, macro = {}, []
for lp, g in d.groupby("language_pair"):
    P, rho, tau = pearsonr(g.qe, g.da)[0], spearmanr(g.qe, g.da)[0], kendalltau(g.qe, g.da)[0]
    per[lp] = [round(P, 3), round(rho, 3), round(tau, 3)]
    macro.append((P, rho, tau))
M = np.array(macro).mean(0)
gm = d.groupby("language_pair").agg(qe=("qe", "mean"), da=("da", "mean"))
xling = pearsonr(gm.qe, gm.da)[0]

res = {"model": "metricx-24-hybrid-xl", "n": int(len(d)), "per_pair_P_rho_tau": per,
       "macro_P_rho_tau": [round(x, 3) for x in M], "cross_lingual_r": round(float(xling), 3)}
json.dump(res, open(f"{OUT}/metrics.json", "w"), indent=2)
print(json.dumps(res, indent=2))
print("\nper-pair means (qe=-MetricX, da):")
print(gm.round(3).to_string())
