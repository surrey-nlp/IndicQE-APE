#!/usr/bin/env python3
"""Full-set QE correlations (6 en->Indic incl en-ml + 3 X->En) for the learned metrics, combining the
overnight en->Indic output.jsonl with the new X->En scores. Same methodology as comet_qe.py: per-pair
Pearson/Spearman/Kendall of model score vs DA; cross-lingual = Pearson of per-pair mean score vs mean
DA. MetricX is an error score -> qe = -prediction. Writes runs/qe_fullset.json. CPU only."""
import json, os, sys, collections
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau

ROOT = os.environ.get("IQE_WORK") or sys.exit("set IQE_WORK to the overnight working directory (holds code/, runs/ and the *_output.jsonl files)")
RUNS = f"{ROOT}/runs"
ENC = {"CometKiwi-DA": "cometkiwi_da", "CometKiwi-XL": "cometkiwi_xl", "XCOMET-XL": "xcomet_xl"}


def load_enc(name):
    rows = []
    for suf in ("", "_xen"):
        p = f"{RUNS}/{name}{suf}/output.jsonl"
        if os.path.exists(p):
            rows += [{"lp": r["language_pair"], "qe": float(r["qe"]), "da": float(r["da"])}
                     for r in map(json.loads, open(p))]
    return rows


def load_metricx():
    rows = []
    for p in (f"{ROOT}/metricx_qe_output.jsonl", f"{ROOT}/metricx_qe_xen_output.jsonl"):
        if os.path.exists(p):
            for r in map(json.loads, open(p)):
                if r.get("prediction") is not None and r.get("gold_da") is not None:
                    rows.append({"lp": r["language_pair"], "qe": -float(r["prediction"]), "da": float(r["gold_da"])})
    return rows


def corr(rows):
    by = collections.defaultdict(lambda: {"qe": [], "da": []})
    for r in rows:
        by[r["lp"]]["qe"].append(r["qe"]); by[r["lp"]]["da"].append(r["da"])
    per, macro = {}, []
    for lp in sorted(by):
        q, a = by[lp]["qe"], by[lp]["da"]
        P, rho, tau = pearsonr(q, a)[0], spearmanr(q, a)[0], kendalltau(q, a)[0]
        per[lp] = [round(P, 3), round(rho, 3), round(tau, 3)]; macro.append((P, rho, tau))
    M = np.array(macro).mean(0)
    lps = sorted(by)
    xling = pearsonr([np.mean(by[l]["qe"]) for l in lps], [np.mean(by[l]["da"]) for l in lps])[0]
    return {"pairs": lps, "per_pair_P_rho_tau": per, "macro_P_rho_tau": [round(x, 3) for x in M],
            "cross_lingual_r": round(float(xling), 3), "n": len(rows)}


out = {}
for disp, name in ENC.items():
    rows = load_enc(name)
    if rows:
        out[disp] = corr(rows)
mx = load_metricx()
if mx:
    out["MetricX-24"] = corr(mx)

json.dump(out, open(f"{RUNS}/qe_fullset.json", "w"), indent=2)
print(f"{'model':14s} {'#pairs':>6s} {'macroP':>7s} {'macro_rho':>9s} {'macro_tau':>9s} {'xling_r':>8s}")
for m, r in out.items():
    P, rho, tau = r["macro_P_rho_tau"]
    print(f"{m:14s} {len(r['pairs']):6d} {P:7.3f} {rho:9.3f} {tau:9.3f} {r['cross_lingual_r']:8.3f}   pairs={r['pairs']}")
