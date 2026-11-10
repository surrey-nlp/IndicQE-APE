#!/usr/bin/env python3
"""GPU1 job: score the QE challenge test (src, mt) with a COMET-family model (CometKiwi / XCOMET in
reference-free QE mode) and correlate with human DA -> a tab:qe row. For XCOMET, also dumps the
error-span severity (for the severity-agreement eval). Single card: set CUDA_VISIBLE_DEVICES=1.

Usage: python comet_qe.py <model_name> <qeda_test.parquet> <out_dir> [gpus]
"""
import sys, os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau
from comet import download_model, load_from_checkpoint

model_name, input_parquet, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
gpus = int(sys.argv[4]) if len(sys.argv) > 4 else 1
os.makedirs(out_dir, exist_ok=True)

d = pd.read_parquet(input_parquet)
d = d[d.has_da == True].reset_index(drop=True)
data = [{"src": str(s), "mt": str(m)} for s, m in zip(d.source, d.mt)]

model = load_from_checkpoint(download_model(model_name))
pred = model.predict(data, batch_size=16, gpus=gpus)
scores = list(pred.scores if hasattr(pred, "scores") else pred["scores"])
d["qe"] = scores
d["da"] = pd.to_numeric(d.da_mean)

# per-instance dump (uid, pair, qe, da) + optional error spans for XCOMET
with open(f"{out_dir}/output.jsonl", "w", encoding="utf-8") as o:
    md = getattr(pred, "metadata", None)
    spans = getattr(md, "error_spans", None) if md is not None else None
    for i, r in d.iterrows():
        rec = {"uid": r.uid, "language_pair": r.language_pair, "qe": float(r.qe), "da": float(r.da)}
        if spans is not None:
            try: rec["error_spans"] = spans[i]
            except Exception: pass
        o.write(json.dumps(rec, ensure_ascii=False) + "\n")

macro = []
per = {}
for p, g in d.groupby("language_pair"):
    P, rho, tau = pearsonr(g.qe, g.da)[0], spearmanr(g.qe, g.da)[0], kendalltau(g.qe, g.da)[0]
    per[p] = [round(P, 3), round(rho, 3), round(tau, 3)]
    macro.append((P, rho, tau))
M = np.array(macro).mean(0)
gm = d.groupby("language_pair").agg(qe=("qe", "mean"), da=("da", "mean"))
xling = pearsonr(gm.qe, gm.da)[0]
res = {"model": model_name, "n": len(d), "per_pair_P_rho_tau": per,
       "macro_P_rho_tau": [round(x, 3) for x in M], "cross_lingual_r": round(xling, 3)}
json.dump(res, open(f"{out_dir}/metrics.json", "w"), indent=2)
print(json.dumps(res, indent=2))
