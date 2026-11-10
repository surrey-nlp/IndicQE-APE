#!/usr/bin/env python3
"""MQM baseline analysis on the en-hi `mqm` config (CPU, no GPU).

Two jobs, both read-only against the live HF release:

 1. DA<->MQM agreement on the 2,327 segments that carry BOTH a human DA score and human MQM
    (the paired subset created by the MQM integration). This is a headline resource result and
    needs no compute: do the two human quality signals agree, and how strongly?

 2. Metric correlation with MQM: given one or more metric-score JSONL files (each row
    {id/uid, pred_score} or {id, gold_da_mean, pred_score}), correlate each metric with the
    segment-level `mqm_score` (and, for the paired rows, with `da_mean`), so a QE metric can be
    scored against MQM as well as DA.

Also reports span-level statistics (error rate, severity mix, spans/segment) for the record.

Usage:
    python3 scripts/mqm_baselines.py                        # DA<->MQM + span stats only
    python3 scripts/mqm_baselines.py --scores a.jsonl b.jsonl   # + metric correlations
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import Counter
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from huggingface_hub import hf_hub_download

REPO = "surrey-nlp/IndicQE-APE"
SPLITS = ("train", "validation", "test")


def load_mqm():
    fr = []
    for sp in SPLITS:
        p = hf_hub_download(REPO, f"data/tasks/mqm/{sp}.parquet", repo_type="dataset")
        d = pd.read_parquet(p); d["_split"] = sp; fr.append(d)
    return pd.concat(fr, ignore_index=True)


def corr_block(x, y):
    x = pd.Series(x).astype(float); y = pd.Series(y).astype(float)
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    if len(x) < 3:
        return None
    return {"n": int(len(x)), "pearson": round(float(pearsonr(x, y)[0]), 4),
            "spearman": round(float(spearmanr(x, y)[0]), 4),
            "kendall": round(float(kendalltau(x, y)[0]), 4)}


def load_scores(path):
    """Return {uid_or_id: pred_score}. Accepts jsonl with keys id|uid + pred_score|pred."""
    m = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if "_meta" in o and len(o) == 1:
            continue
        key = o.get("uid") or o.get("id")
        pv = o.get("pred_score")
        if pv is None:
            pv = o.get("pred")
            if isinstance(pv, dict):
                pv = pv.get("gemba-da")
        try:
            m[key] = float(pv)
        except (TypeError, ValueError):
            pass
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="*", default=[], help="metric-score jsonl files to correlate")
    ap.add_argument("--out", default="results/analysis/mqm_baselines.md")
    a = ap.parse_args()

    print("loading en-hi mqm config from HF ...")
    m = load_mqm()
    print(f"  {len(m)} rows; da_mean present {int(m.da_mean.notna().sum())}; "
          f"mqm_score present {int(m.mqm_score.notna().sum())}")

    lines = ["# MQM baselines — en-hi `mqm` config\n"]

    # 1. DA <-> MQM agreement on the paired subset
    paired = m[m.da_mean.notna() & m.mqm_score.notna()]
    c = corr_block(paired.da_mean, paired.mqm_score)
    print("\n=== 1. DA <-> MQM agreement (paired human labels) ===")
    print(f"  paired n={len(paired)}  -> {c}")
    lines.append("## 1. DA ↔ MQM agreement (paired human labels)\n")
    lines.append(f"On the **{len(paired)}** en-hi segments carrying both a human DA score and human "
                 f"MQM (both 0–100, higher = better):\n")
    lines.append(f"| n | Pearson | Spearman | Kendall |\n|---:|---:|---:|---:|\n"
                 f"| {c['n']} | {c['pearson']} | {c['spearman']} | {c['kendall']} |\n")

    # 2. span-level stats
    print("\n=== span-level statistics ===")
    n_err = 0; sev = Counter(); spans_per = []
    for s in m["mqm_spans"]:
        k = 0 if s is None else len(s)
        spans_per.append(k); n_err += (k > 0)
        if s is not None:
            for e in s:
                sev[e.get("severity")] += 1
    lines.append("\n## 2. Span-level statistics\n")
    lines.append(f"- segments with ≥1 error: {n_err} / {len(m)} ({100*n_err/len(m):.1f}%)\n")
    lines.append(f"- severity mix: {dict(sev)}\n")
    lines.append(f"- mean spans/segment: {sum(spans_per)/len(spans_per):.2f}\n")
    print(f"  error-bearing {n_err}/{len(m)}; severity {dict(sev)}; mean spans/seg "
          f"{sum(spans_per)/len(spans_per):.2f}")

    # 3. metric correlations (optional)
    if a.scores:
        print("\n=== 3. metric correlation with MQM (and DA on paired) ===")
        lines.append("\n## 3. Metric correlation with `mqm_score` (segment-level)\n")
        lines.append("| metric | n(MQM) | Pearson(MQM) | Spearman(MQM) | n(DA) | Spearman(DA) |\n"
                     "|---|---:|---:|---:|---:|---:|\n")
        for f in a.scores:
            sc = load_scores(f)
            m["_pred"] = m["uid"].map(sc)
            cM = corr_block(m["_pred"], m["mqm_score"])
            pd_ = m[m.da_mean.notna()]
            cD = corr_block(pd_["_pred"], pd_["da_mean"])
            name = f.split("/")[-1]
            print(f"  {name}: MQM {cM}  DA {cD}")
            lines.append(f"| {name} | {cM['n'] if cM else 0} | {cM['pearson'] if cM else '-'} | "
                         f"{cM['spearman'] if cM else '-'} | {cD['n'] if cD else 0} | "
                         f"{cD['spearman'] if cD else '-'} |\n")

    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("".join(lines))
    print(f"\nwritten: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
