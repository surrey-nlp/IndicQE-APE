#!/usr/bin/env python3
"""make_cometqe_table.py — assemble §6 comet-QE results + the zero-shot Z row from the campaign.

Reads results/campaign/outputs/<cfg>/run_summary.json (hybrid ablations) and
results/campaign/zeroshot/zeroshot-summary-qe-da-test.json (Strategy-Z baselines), and writes
results/analysis/cometqe_summary.md plus prints the numbers for tab:cometqe and tab:qe's Z row.
All numbers trace to the campaign JSON — nothing hand-typed.
"""
from __future__ import annotations
import json, glob, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMP = ROOT / "experiments" / "comet_qe" / "results" / "campaign"
OUT = CAMP / "outputs"
ZS = CAMP / "zeroshot" / "zeroshot-summary-qe-da-test.json"
MD = ROOT / "results" / "analysis" / "cometqe_summary.md"

# reading order + human label for the ablation
ORDER = [
    ("qe-E1-xlmr", "E1 encoder: XLM-R-large (no QE pretrain, floor)"),
    ("qe-E1-infoxlm", "E1 encoder: InfoXLM-large (CometKiwi arch, no QE FT)"),
    ("qe-E1-cometda", "E1 encoder: COMET-DA (ref-based QE pretrain)"),
    ("qe-A0-cometkiwi-min", "E1/E2-A0: CometKiwi, CLS+funnel only"),
    ("qe-E2-A1", "E2-A1: +TSA"),
    ("qe-E2-A2", "E2-A2: +TSA+TSE"),
    ("qe-E2-A3", "E2-A3: +TSA+TSE+len"),
    ("qe-A-cometkiwi-full", "A-full (reference): CometKiwi, funnel+TSA+TSE+len"),
    ("qe-E3-none", "E3: no funnelling (last layer)"),
    ("qe-E3-upper", "E3: funnel [16,20,-1]"),
    ("qe-E3-buggy", "E3: funnel [8,10,-1] (mis-pick)"),
    ("qe-E4-mse", "E4 loss: MSE"),
    ("qe-E4-huber05", "E4 loss: Huber d=0.5"),
    ("qe-E4-ccc", "E4 loss: CCC"),
    ("qe-E5-raw", "E5 target: raw da_mean 0-100"),
    ("qe-E5-minmax", "E5 target: minmax da_mean/100"),
]


def load(cfg):
    p = OUT / cfg / "run_summary.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return {k: d.get(k) for k in ("test_pearson", "test_spearman", "test_rmse",
                                  "dev_pearson", "dev_spearman", "num_train")}


def f(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


def main():
    lines = ["# comet-QE campaign — §6 numbers (from the campaign run_summary.json files)\n\n",
             "## Hybrid ablation (test split, en→Indic joint)\n\n",
             "| config | test ρ (Spearman) | test r (Pearson) | test RMSE | dev ρ | n_train |\n",
             "|---|---:|---:|---:|---:|---:|\n"]
    got = {}
    for cfg, label in ORDER:
        r = load(cfg)
        if r is None:
            lines.append(f"| {label} | (missing) | | | | |\n"); continue
        got[cfg] = r
        lines.append(f"| {label} | {f(r['test_spearman'])} | {f(r['test_pearson'])} | "
                     f"{f(r['test_rmse'])} | {f(r['dev_spearman'])} | {r['num_train']} |\n")

    # zero-shot Z
    lines.append("\n## Strategy-Z zero-shot COMET (qe-da test)\n\n")
    lines.append("| model | overall ρ | overall r | per-pair ρ |\n|---|---:|---:|---|\n")
    if ZS.exists():
        z = json.load(open(ZS))
        for row in z:
            if "error" in row:
                lines.append(f"| {row['model']} | FAILED | | {row['error'][:40]} |\n"); continue
            ov = row.get("overall") or {}
            pp = row.get("per_pair") or {}
            pps = " ".join(f"{k.split('-')[-1]}:{(v or {}).get('spearman','—')}" for k, v in pp.items())
            lines.append(f"| {row['model']} | {ov.get('spearman','—')} | {ov.get('pearson','—')} | {pps} |\n")
    else:
        lines.append("| (zeroshot summary missing) | | | |\n")

    # quick reads for the narrative
    lines.append("\n## Reads for §6\n")
    if "qe-A-cometkiwi-full" in got and "qe-E1-xlmr" in got:
        a = got["qe-A-cometkiwi-full"]["test_spearman"]; x = got["qe-E1-xlmr"]["test_spearman"]
        cd = got.get("qe-E1-cometda", {}).get("test_spearman")
        lines.append(f"- Encoder: CometKiwi reference ρ={a:.3f} > COMET-DA ρ={cd:.3f} > XLM-R floor "
                     f"ρ={x:.3f} — QE pretraining, ref-free best.\n")
    MD.parent.mkdir(parents=True, exist_ok=True)
    MD.write_text("".join(lines))
    print("".join(lines))
    print(f"written: {MD}")


if __name__ == "__main__":
    main()
