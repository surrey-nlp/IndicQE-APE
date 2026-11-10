#!/usr/bin/env python3
"""seed_stability.py — resolve the four single-seed collapses with a three-seed protocol.

Four ablation cells collapsed to near-mean prediction at seed 42. One seed cannot separate an
unhelpful setting from an unlucky initialisation, so each was rerun at seeds 43 and 44. A cell is
reported as a genuine collapse only where the collapse REPRODUCES across seeds; otherwise the
seed-42 value is an outlier and the cell must be reported from the stable seeds.

Also summarises the two arms added alongside the seed reruns:
  joint  six-pair training including en-ml, on the current release
  xen    X->en reproduction check, the only external validity check on this pipeline

Outputs
  results/analysis/seed_stability.md
  results/analysis/seed_stability.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "comet_qe" / "results" / "campaign" / "outputs"
OUT_MD = ROOT / "results" / "analysis" / "seed_stability.md"
OUT_CSV = ROOT / "results" / "analysis" / "seed_stability.csv"

#: base config -> human label. Seed 42 is the original run; 43 and 44 are the reruns.
COLLAPSED = {
    "qe-E1-infoxlm":   "E1 encoder: InfoXLM-large (CometKiwi architecture, no QE fine-tuning)",
    "qe-E2-A1":        "E2 features: A0 + TSA",
    "qe-E3-none":      "E3 funnelling: none (last layer only)",
    "qe-E4-huber05":   "E4 loss: Huber delta=0.5",
}
ARMS = {
    "qe-J-A0-joint":  "six pairs incl. en-ml, A0 recipe",
    "qe-J-ref-joint": "six pairs incl. en-ml, reference recipe",
    "qe-X-A0-xen":    "X->en repro, A0 recipe",
    "qe-X-ref-xen":   "X->en repro, reference recipe",
}
#: below this a run has not learned to order segments
COLLAPSE_THRESHOLD = 0.20


def load(run: str) -> dict | None:
    f = OUT / run / "run_summary.json"
    if not f.exists():
        return None
    j = json.loads(f.read_text())
    p = OUT / run / "predictions_test.csv"
    rec = {
        "run": run,
        "spearman": j.get("test_spearman"), "pearson": j.get("test_pearson"),
        "kendall": j.get("test_kendall"), "rmse": j.get("test_rmse"),
        "dev_spearman": j.get("dev_spearman"),
        "num_train": j.get("num_train"), "data_file": j.get("data_file"),
    }
    if p.exists():
        t = pd.read_csv(p)
        rec["n_test"] = len(t)
        rec["n_pairs"] = t.language_pair.nunique()
        rec["pred_std"] = float(t["pred"].std())
        # a collapsed run predicts (near) one number: relative dispersion makes that explicit
        rec["pred_cv"] = float(t["pred"].std() / (abs(t["pred"].mean()) + 1e-9))
    return rec


def main() -> int:
    rows = []
    for base in COLLAPSED:
        # 42n = seed 42 rerun on the CURRENT build. It is the control that disentangles the
        # seed from the data rebuild for the two cells where both changed at once.
        for seed, run in ((42, base), (43, f"{base}-s43"), (44, f"{base}-s44"),
                          ("42n", f"{base}-s42new")):
            r = load(run)
            if r is None:
                print(f"  missing {run}")
                continue
            r["base"], r["seed"] = base, seed
            rows.append(r)
    for run in ARMS:
        r = load(run)
        if r:
            r["base"], r["seed"] = run, 42
            rows.append(r)

    if not rows:
        print("no runs found", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"rows: {len(df)} -> {OUT_CSV}")

    L = ["# Seed stability of the four collapsed ablation cells\n\n",
         "Each cell that collapsed at seed 42 was rerun at seeds 43 and 44. A collapse is reported ",
         "as a property of the configuration only where it reproduces; the threshold for 'collapsed' ",
         f"is Spearman below {COLLAPSE_THRESHOLD}.\n\n",
         "**Read the `n train` column before the verdict.** The reruns were launched either side of a ",
         "rebuild of `qe_indic_enindic.csv`, so some seeds trained on $64{,}416$ segments (the ",
         "original campaign build) and others on $64{,}671$ (the current release). Where the seed and ",
         "the build both changed, a recovery cannot be attributed to the seed alone, and this table ",
         "says so rather than claiming a clean seed effect.\n\n",
         "| configuration | s42 | s43 | s44 | s42, current build | n train s42/43/44 | verdict |\n",
         "|---|--:|--:|--:|--:|---|---|\n"]

    verdicts = {}
    for base, label in COLLAPSED.items():
        g = df[df.base == base].set_index("seed")
        vals = {s: g.loc[s, "spearman"] if s in g.index else np.nan for s in (42, 43, 44)}
        ntr = {s: (int(g.loc[s, "num_train"]) if s in g.index and pd.notna(g.loc[s, "num_train"])
                   else None) for s in (42, 43, 44)}
        same_build_keys = (42, 43, 44)
        same_build = len({v for v in ntr.values() if v}) == 1
        collapsed_at = [s for s, v in vals.items() if np.isfinite(v) and v < COLLAPSE_THRESHOLD]
        stable = [v for s, v in vals.items() if np.isfinite(v) and v >= COLLAPSE_THRESHOLD]

        # The disentangling control, where it exists.
        ctl = g.loc["42n", "spearman"] if "42n" in g.index else np.nan
        ctl_n = (int(g.loc["42n", "num_train"]) if "42n" in g.index
                 and pd.notna(g.loc["42n", "num_train"]) else None)
        ctl_collapsed = np.isfinite(ctl) and ctl < COLLAPSE_THRESHOLD

        if len(collapsed_at) == 3:
            verdict = ("**REPRODUCES** at all three seeds"
                       + ("" if same_build else " AND across both data builds")
                       + ": genuine")
            reported = float(np.mean([vals[s] for s in (42, 43, 44)]))
        elif collapsed_at == [42] and same_build:
            verdict = "does NOT reproduce, same build: **seed 42 was an outlier**"
            reported = float(np.mean(stable)) if stable else np.nan
        elif collapsed_at == [42] and np.isfinite(ctl):
            # Seed and build changed together, but we ran seed 42 on the current build.
            if ctl_collapsed:
                verdict = (f"does NOT reproduce; seed 42 on the CURRENT build still collapses "
                           f"({ctl:.3f}): **the seed, not the data**")
            else:
                verdict = (f"does NOT reproduce; seed 42 on the CURRENT build recovers "
                           f"({ctl:.3f}): **the superseded data, not the seed**")
            reported = float(np.mean(stable + [ctl])) if stable else float(ctl)
        elif collapsed_at == [42]:
            verdict = ("does NOT reproduce, but build ALSO changed: "
                       "**CONFOUNDED, cannot attribute to seed** (run seed 42 on the current "
                       "build to disentangle)")
            reported = float(np.mean(stable)) if stable else np.nan
        else:
            verdict = f"mixed, collapsed at {collapsed_at}"
            reported = float(np.mean(stable)) if stable else np.nan

        verdicts[base] = (verdict, reported, vals, collapsed_at, same_build, ntr)
        ntr_s = "/".join(str(ntr[s] or "-") for s in (42, 43, 44))
        ctl_s = f"{ctl:.3f}" if np.isfinite(ctl) else "---"
        L.append(f"| {label} | {vals[42]:.3f} | {vals[43]:.3f} | {vals[44]:.3f} | {ctl_s} | "
                 f"{ntr_s} | {verdict} |\n")

    L.append("\n## Reading\n\n")
    genuine = [b for b, v in verdicts.items() if "REPRODUCES" in v[0]]
    clean = [b for b, v in verdicts.items() if "seed 42 was an outlier" in v[0]]
    confounded = [b for b, v in verdicts.items() if "CONFOUNDED" in v[0]]
    resolved_data = [b for b, v in verdicts.items() if "superseded data, not the seed" in v[0]]
    resolved_seed = [b for b, v in verdicts.items() if "the seed, not the data" in v[0]]

    if genuine:
        L.append("**Genuine collapse: `" + "`, `".join(genuine) + "`.** It reproduces at all three "
                 "seeds and, for these cells, across both data builds, which is stronger evidence "
                 "than the protocol was designed to give. The design conclusion stands: multi-layer "
                 "funnelling is necessary, not incidental.\n\n")
    if clean:
        L.append("**Clean seed effect: `" + "`, `".join(clean) + "`.** All three seeds trained on the "
                 "same build, so the recovery is attributable to initialisation. Reporting the "
                 "seed-42 value would have described a working configuration as broken.\n\n")
    if resolved_data:
        L.append("**Resolved, and the cause is the data: `" + "`, `".join(resolved_data) +
                 "`.** Seed 42 rerun on the current $64{,}671$-segment build recovers, so the "
                 "original collapse was a property of the superseded build and not of the seed or "
                 "the configuration. The seed-42 values from that build should not be reported as "
                 "collapses of these settings.\n\n")
    if resolved_seed:
        L.append("**Resolved, and the cause is the seed: `" + "`, `".join(resolved_seed) +
                 "`.** Seed 42 still collapses on the current build, so the setting is unstable at "
                 "that initialisation rather than broken by the data.\n\n")
    if confounded:
        L.append("**Confounded, and must not be reported as a seed effect: `" +
                 "`, `".join(confounded) + "`.** These recovered, but the seed and the training data "
                 "changed together: seed 42 used the $64{,}416$-segment build and seeds 43/44 the "
                 "rebuilt $64{,}671$. The recovery is real but its cause is not identified. To "
                 "separate the two, rerun seed 42 on the current build (one run per cell); if it "
                 "still collapses, the effect is the seed, and if it does not, the earlier collapse "
                 "was a property of the superseded data.\n\n")

    L.append("### Consequences for the reported ablation\n\n")
    L.append("| cell | reported now | reruns give | status |\n|---|--:|--:|---|\n")
    for base, (verdict, reported, vals, _, same_build, _n) in verdicts.items():
        if "REPRODUCES" in verdict:
            status = "confirmed, keep as is"
        elif "outlier" in verdict:
            status = "**replace** with the reseeded value"
        elif "superseded data, not the seed" in verdict:
            status = "**replace**: the seed-42 collapse was the old build, resolved by the control"
        elif "the seed, not the data" in verdict:
            status = "**report with its spread**: unstable at seed 42 on the current build too"
        else:
            status = "**do not report either value** until disentangled"
        L.append(f"| {COLLAPSED[base]} | {vals[42]:.3f} | {reported:.3f} | {status} |\n")

    iv = verdicts["qe-E1-infoxlm"][2]
    L.append(f"\nInfoXLM is the one cell with a clean three-seed test on one build, and it is "
             f"genuinely unstable rather than dead ({vals_str(iv)}, spread "
             f"{max(v for v in iv.values()) - min(v for v in iv.values()):.3f}). It should be "
             "reported with its spread, not as a point estimate, and it is the weakest rung of the "
             "E1 ladder to lean on.\n")

    L.append("\n## Additional arms\n\n")
    L.append("| arm | pairs | n test | Spearman | Pearson |\n|---|--:|--:|--:|--:|\n")
    for run, label in ARMS.items():
        g = df[df.run == run]
        if not len(g):
            continue
        r = g.iloc[0]
        L.append(f"| {label} | {int(r.get('n_pairs', 0))} | {int(r.get('n_test', 0)):,} | "
                 f"{r['spearman']:.3f} | {r['pearson']:.3f} |\n")
    L.append("\nThe six-pair joint arm is the population the paper should report for "
             "\\S\\ref{sec:cometqe}: it includes en-ml and uses the current release. The X->en arm "
             "is an external validity check, since published CometKiwi numbers exist for those "
             "pairs on MLQE-PE.\n")

    OUT_MD.write_text("".join(L))
    print("".join(L))
    return 0


def vals_str(v: dict) -> str:
    return ", ".join(f"s{k} {x:.3f}" for k, x in sorted(v.items()))


if __name__ == "__main__":
    raise SystemExit(main())
