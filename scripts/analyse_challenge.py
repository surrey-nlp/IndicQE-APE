#!/usr/bin/env python3
"""Challenge diagnostics over the per-item run outputs.

Reads the QE/APE run JSONLs (which carry `route` + `is_curated`) and produces:
  - per-system QE correlation, overall and per `route` (the difficulty axes);
  - the easy_control -> hard drop per system (the headline diagnostic);
  - curated (En->Indic) vs uncurated (X->En) correlation;
  - joint QE-APE: do QE-hard and APE-hard items coincide.

  python scripts/analyse_challenge.py --root $IQE_OUT --out $IQE_OUT/analysis
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import pandas as pd
from scipy.stats import spearmanr

HARD = ["A1_disagreement", "A2_edit_effort", "A3_word_error", "A4_contradiction"]


def _read(glob_pat):
    rows = []
    for f in glob.glob(glob_pat):
        sysname = os.path.basename(f).replace(".jsonl", "")
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in r or "_summary" in r:
                continue
            r["system"] = sysname
            rows.append(r)
    return pd.DataFrame(rows)


def _rho(g, a, b):
    g = g.dropna(subset=[a, b])
    return round(spearmanr(g[a], g[b]).correlation, 4) if len(g) > 10 else None


def _explode_qe(qe: pd.DataFrame) -> pd.DataFrame:
    """Long-form QE rows with one `pred_score` per (item, template).

    Records carry `pred` = {template: score}; expand to a `template` column so
    diagnostics are computed per prompt variant (the prompt-sensitivity surface).
    Legacy records with a flat `pred_score` are tagged template="(legacy)".
    """
    out = []
    for r in qe.to_dict("records"):
        preds = r.get("pred")
        if isinstance(preds, dict) and preds:
            for tpl, score in preds.items():
                row = dict(r); row["template"] = tpl; row["pred_score"] = score
                out.append(row)
        else:
            row = dict(r); row.setdefault("template", "(legacy)")
            row["pred_score"] = r.get("pred_score")
            out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    qe = _read(f"{a.root}/qe_*/*.jsonl")
    if len(qe):
        qe = _explode_qe(qe).dropna(subset=["pred_score"])
        # overall + per-route + curated/uncurated per (system, template)
        recs = []
        for (sysname, tpl), g in qe.groupby(["system", "template"]):
            base = {"system": sysname, "template": tpl, "n": len(g),
                    "overall": _rho(g, "gold_da_mean", "pred_score"),
                    "curated": _rho(g[g.is_curated == True], "gold_da_mean", "pred_score"),
                    "xen": _rho(g[g.is_curated == False], "gold_da_mean", "pred_score")}
            for rt, gg in g.groupby("route"):
                base[rt] = _rho(gg, "gold_da_mean", "pred_score")
            recs.append(base)
        diag = pd.DataFrame(recs)
        # easy -> hard drop
        if "easy_control" in diag:
            diag["hard_mean"] = diag[[c for c in HARD if c in diag]].mean(axis=1)
            diag["drop"] = diag["easy_control"] - diag["hard_mean"]
        diag.sort_values(["system", "overall"], ascending=[True, False]).to_csv(
            f"{a.out}/qe_diagnostics.csv", index=False)
        print(f"wrote {a.out}/qe_diagnostics.csv ({len(diag)} system x template rows)")

        # prompt-sensitivity: per system, gemba-da vs gemba-sqm overall rho gap
        piv = diag.pivot_table(index="system", columns="template", values="overall")
        if {"gemba-da", "gemba-sqm"}.issubset(piv.columns):
            piv["da_minus_sqm"] = piv["gemba-da"] - piv["gemba-sqm"]
            piv.sort_values("gemba-da", ascending=False).to_csv(f"{a.out}/qe_prompt_sensitivity.csv")
            print(f"wrote {a.out}/qe_prompt_sensitivity.csv (gemba-da vs gemba-sqm per system)")

    # joint QE-APE on shared items (use the primary prompt variant for QE abs-err)
    ape = _read(f"{a.root}/ape_*/*.jsonl")
    if len(qe) and len(ape):
        import sacrebleu
        prim = "gemba-da" if (qe["template"] == "gemba-da").any() else qe["template"].iloc[0]
        qe = qe[qe["template"] == prim].copy()
        qe["abserr"] = (qe["gold_da_mean"] - qe["pred_score"]).abs()
        ape = ape.dropna(subset=["hyp_pe", "ref_pe"])
        ape["chrf"] = ape.apply(lambda r: sacrebleu.sentence_chrf(r.hyp_pe, [r.ref_pe]).score, axis=1)
        j = (qe.groupby("id").abserr.mean().reset_index()
             .merge(ape.groupby("id").chrf.mean().reset_index(), on="id"))
        j = j.dropna(subset=["abserr", "chrf"])            # drop items where QE never parsed or APE empty
        if len(j) > 20 and j["abserr"].std() > 0 and j["chrf"].std() > 0:
            rho = spearmanr(j.abserr, j.chrf, nan_policy="omit").correlation
            j.to_csv(f"{a.out}/joint_qe_ape.csv", index=False)
            rho_s = f"{rho:+.3f}" if rho == rho else "n/a"    # rho==rho is False for NaN
            print(f"joint QE-APE: {len(j)} shared items, Spearman(QE abs-err, APE chrF) = {rho_s}")
            print("  (negative => items hard for QE are also hard for APE)")
        else:
            print(f"joint QE-APE: insufficient non-constant overlap ({len(j)} items) — skipped")


if __name__ == "__main__":
    main()
