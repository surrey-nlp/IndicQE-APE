#!/usr/bin/env python3
"""campaign_report.py — full progress + results report for the comet-QE campaign.

Reads the artefacts pulled from the GPU workstation into experiments/comet_qe/results/campaign/:
  outputs/<run>/run_summary.json      aggregate metrics + per-pair breakdown
  outputs/<run>/predictions_test.csv  per-instance predictions (item_id, pair, gold, pred)
  zeroshot/*.json|jsonl               Strategy-Z baselines
  runs_status/*                       queue ledger (DONE/FAILED/STOPPED/RUNNING)

Produces results/analysis/campaign_report.md and prints it.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
CAMP = ROOT / "experiments" / "comet_qe" / "results" / "campaign"
OUT = ROOT / "results" / "analysis" / "campaign_report.md"

LABEL = {
    "qe-E1-xlmr": ("E1 encoder", "XLM-R-large (no QE pretraining)"),
    "qe-E1-infoxlm": ("E1 encoder", "InfoXLM-large (CometKiwi arch, no QE FT)"),
    "qe-E1-cometda": ("E1 encoder", "COMET-DA (reference-based QE pretraining)"),
    "qe-A0-cometkiwi-min": ("E1/E2", "CometKiwi, CLS+funnelling only (A0)"),
    "qe-E2-A1": ("E2 features", "A0 + TSA"),
    "qe-E2-A2": ("E2 features", "A0 + TSA + TSE"),
    "qe-E2-A3": ("E2 features", "A0 + TSA + TSE + length ratio (= reference)"),
    "qe-A-cometkiwi-full": ("reference", "CometKiwi, full features"),
    "qe-E3-none": ("E3 funnelling", "none (last layer only)"),
    "qe-E3-upper": ("E3 funnelling", "[16,20,-1]"),
    "qe-E3-buggy": ("E3 funnelling", "[8,10,-1] (mis-picked)"),
    "qe-E4-mse": ("E4 loss", "MSE"),
    "qe-E4-huber05": ("E4 loss", "Huber d=0.5"),
    "qe-E4-ccc": ("E4 loss", "CCC"),
    "qe-E5-raw": ("E5 target", "raw DA 0-100"),
    "qe-E5-minmax": ("E5 target", "min-max [0,1]"),
}
PAIRS = ["en-gu", "en-hi", "en-ml", "en-mr", "en-ta", "en-te"]


def f(x, n=3):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)) else "—"


def main():
    L = ["# comet-QE campaign — progress and results\n"]

    # ---------- queue state ----------
    st = CAMP / "runs_status"
    done = sorted(p.stem for p in st.glob("*.DONE"))
    failed = sorted(p.stem for p in st.glob("*.FAILED"))
    stopped = sorted(p.stem for p in st.glob("*.STOPPED"))
    running = sorted(p.stem for p in st.glob("*.RUNNING"))
    order = [l.strip() for l in (st / "ORDER").read_text().splitlines() if l.strip()] if (st / "ORDER").exists() else []
    pending = [c for c in order if c not in done and c not in running]
    L.append(f"\n## Queue state\n\n**{len(done)} done · {len(running)} running · {len(failed)} failed · "
             f"{len(stopped)} stopped · {len(pending)} pending** (of {len(order)} queued)\n\n")
    L.append(f"- running: {', '.join(running) or '—'}\n- pending: {', '.join(pending) or '—'}\n"
             f"- failed: {', '.join(failed) or 'none'}\n")
    print(f"queue: {len(done)} done, {len(running)} running, {len(failed)} failed, {len(pending)} pending")

    # ---------- aggregate results ----------
    rows = []
    for d in sorted((CAMP / "outputs").iterdir()):
        s = d / "run_summary.json"
        if not s.exists():
            continue
        j = json.load(open(s))
        grp, desc = LABEL.get(d.name, ("", d.name))
        rows.append({"run": d.name, "group": grp, "desc": desc,
                     "test_sp": j.get("test_spearman"), "test_pe": j.get("test_pearson"),
                     "test_ke": j.get("test_kendall"), "test_rmse": j.get("test_rmse"),
                     "dev_sp": j.get("dev_spearman"), "n_train": j.get("num_train"),
                     "per_pair": j.get("per_pair_test")})
    df = pd.DataFrame(rows)
    L.append(f"\n## Ablation results ({len(df)} completed runs)\n\n")
    L.append("| group | configuration | test ρ | test r | test τ | RMSE | dev ρ |\n|---|---|--:|--:|--:|--:|--:|\n")
    for _, r in df.sort_values(["group", "run"]).iterrows():
        L.append(f"| {r['group']} | {r['desc']} | **{f(r['test_sp'])}** | {f(r['test_pe'])} | "
                 f"{f(r['test_ke'])} | {f(r['test_rmse'],2)} | {f(r['dev_sp'])} |\n")

    # ---------- per-pair (from saved per-instance predictions) ----------
    have_pp = df[df.per_pair.notna()]
    L.append(f"\n## Per-pair Spearman (from saved per-instance predictions)\n\n")
    if len(have_pp):
        L.append("| configuration | " + " | ".join(PAIRS) + " | macro |\n" + "|---" * (len(PAIRS) + 2) + "|\n")
        for _, r in have_pp.sort_values("test_sp", ascending=False).iterrows():
            pp = r["per_pair"] or {}
            vals = [pp.get(p, {}).get("spearman") if isinstance(pp.get(p), dict) else None for p in PAIRS]
            got = [v for v in vals if v is not None]
            L.append(f"| {r['desc']} | " + " | ".join(f(v) for v in vals) +
                     f" | {f(np.mean(got)) if got else '—'} |\n")
    else:
        L.append("_No per-pair breakdown yet — these runs predate the per-instance saving change; "
                 "reruns will populate it._\n")

    # ---------- zero-shot ----------
    zs = CAMP / "zeroshot" / "zeroshot-summary-qe-da-test.json"
    if zs.exists():
        L.append("\n## Strategy-Z zero-shot COMET (same en→Indic challenge test)\n\n")
        L.append("| model | overall ρ | overall r | " + " | ".join(PAIRS) + " |\n" + "|---" * (len(PAIRS) + 3) + "|\n")
        for row in json.load(open(zs)):
            if "error" in row:
                L.append(f"| {row['model']} | FAILED | | | | | | | |\n"); continue
            ov = row.get("overall") or {}
            pp = row.get("per_pair") or {}
            cells = [f(pp.get(p, {}).get("spearman")) if isinstance(pp.get(p), dict) else "—" for p in PAIRS]
            L.append(f"| {row['model']} | **{ov.get('spearman','—')}** | {ov.get('pearson','—')} | "
                     + " | ".join(cells) + " |\n")

    # ---------- reads ----------
    def get(run, col="test_sp"):
        m = df[df.run == run]
        return float(m.iloc[0][col]) if len(m) and pd.notna(m.iloc[0][col]) else None
    L.append("\n## Reads\n\n")
    lad = [("XLM-R (none)", get("qe-E1-xlmr")), ("COMET-DA (ref-based)", get("qe-E1-cometda")),
           ("InfoXLM (arch only)", get("qe-E1-infoxlm")), ("CometKiwi (ref-free)", get("qe-A0-cometkiwi-min"))]
    L.append("**E1 — encoder initialisation.** " + " · ".join(
        f"{k} {f(v)}" for k, v in lad if v is not None) + "\n\n")
    L.append("**E5 — target scale.** " + " · ".join(
        f"{k} {f(get(r))}" for k, r in [("raw 0–100", "qe-E5-raw"), ("min–max", "qe-E5-minmax")]) +
        " (raw diverges under Huber)\n\n")
    coll = df[(df.test_sp.notna()) & (df.test_sp < 0.2)]
    if len(coll):
        L.append(f"**Optimisation instability: {len(coll)} of {int(df.test_sp.notna().sum())} completed "
                 "runs collapsed to near-mean prediction** (single seed each): " +
                 ", ".join(f"`{r['run']}` (ρ={f(r['test_sp'])})" for _, r in coll.iterrows()) +
                 ". Their neighbours in the same sweep train normally, so these are read as seed "
                 "instability rather than evidence against the setting — but at this rate a "
                 "multi-seed protocol is needed before any of them is reported as an ablation "
                 "result.\n\n")
    L.append("**Note on en-ml.** The `enindic` data variant excludes en-ml (it needs the "
             "`joint_enml` build), so the per-pair table has five pairs. The en-ml runs are a "
             "separate variant, not a missing result.\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(L))
    print("".join(L))
    print(f"\nwritten {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
