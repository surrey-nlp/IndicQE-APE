#!/usr/bin/env python3
r"""verify_head_numbers.py — the COMET-encoder head's numbers, checked against the run summaries.

This exists because the paper got these wrong once and nothing caught it. S5 described the
configuration WITH the attention features and the length ratio, and reported 0.596; 0.596 is the
run WITHOUT them. The two differ by 0.015, both are plausible, and no generator touched either
figure, so the mismatch survived several validation passes.

Every number below is read from experiments/comet_qe/results/campaign/outputs/<run>/run_summary.json
and asserted present in the tex. If a run is re-run, this fails until the paper is updated.

    python3 analysis/verify_head_numbers.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The paper source lives in its own repository and is optional here; see
# analysis/paper_source.py for the search order and the env vars that override it.
sys.path.insert(0, str(ROOT / "analysis"))
from paper_source import find_tex as _find_tex  # noqa: E402
_PAPER_TEX = _find_tex() or (ROOT / "paper" / "latex" / "acl_latex.tex")

TEX = _PAPER_TEX
CAMP = ROOT / "experiments" / "comet_qe" / "results" / "campaign" / "outputs"

#: run -> (what the paper calls it, which fields must appear in the tex)
RUNS = {
    "qe-J-A0-joint":   "six-pair funnel only, the S5 headline",
    "qe-J-ref-joint":  "six-pair with attention features and length ratio",
    "qe-X-A0-xen":     "X-to-en funnel only",
    "qe-X-ref-xen":    "X-to-en with the extra features",
    "qe-A-cometkiwi-full": "five-pair ablation reference",
    "qe-A0-cometkiwi-min": "five-pair ablation A0",
}
#: the populations the paper states for each build
POPULATIONS = {"qe-J-A0-joint": 9730, "qe-J-ref-joint": 9730,
               "qe-X-A0-xen": 3000, "qe-X-ref-xen": 3000,
               "qe-A-cometkiwi-full": 8160, "qe-A0-cometkiwi-min": 8160}


def main() -> int:
    flat = " ".join(TEX.read_text().split())
    fails: list[str] = []
    rho: dict[str, float] = {}

    for run, what in RUNS.items():
        f = CAMP / run / "run_summary.json"
        if not f.exists():
            fails.append(f"{run}: no run_summary.json")
            continue
        d = json.loads(f.read_text())
        rho[run] = float(d["test_spearman"])
        if int(d["num_test"]) != POPULATIONS[run]:
            fails.append(f"{run}: num_test={d['num_test']}, paper says {POPULATIONS[run]}")
        s = f"${rho[run]:.3f}$"
        if s not in flat:
            fails.append(f"{run} ({what}): rho {s} not in the tex")
        print(f"  {run:22s} rho={rho[run]:.4f} n={d['num_test']:5d}  {what}")

    # The claims that depend on a DIFFERENCE, which is where the paper went wrong. Note the two
    # builds disagree on the SIGN of the feature effect: -0.015 on six pairs, +0.005 on five. An
    # earlier draft of this fix said "the ablation says the same", which it does not. S5 now states
    # both signs, so both are checked.
    for a, b, lane in [("qe-J-ref-joint", "qe-J-A0-joint", "six-pair"),
                       ("qe-A-cometkiwi-full", "qe-A0-cometkiwi-min", "five-pair ablation")]:
        if {a, b} <= rho.keys():
            eff = rho[a] - rho[b]
            if f"${eff:+.3f}$" not in flat:
                fails.append(f"feature effect on the {lane} lane is {eff:+.3f}, "
                             f"not stated as ${eff:+.3f}$")
            print(f"  feature effect, {lane:19s} {eff:+.3f}")
    for a, b, label in [("qe-X-A0-xen", "qe-J-A0-joint", "A0"),
                        ("qe-X-ref-xen", "qe-J-ref-joint", "reference")]:
        if {a, b} <= rho.keys():
            gap = rho[a] - rho[b]
            if f"${gap:.3f}$" not in flat:
                fails.append(f"X-to-en minus en-to-Indic under {label} is {gap:+.3f}, "
                             f"not stated as ${gap:.3f}$")
            print(f"  X-to-en gap under {label:9s} {gap:+.3f}")

    # the probe's oracle premium: the paper must report the fixed-layer figure, not only the oracle
    for s in ("$0.527$", "an oracle over the six layers"):
        if s not in flat:
            fails.append(f"probe: {s!r} missing, the layer oracle is undisclosed again")

    for f_ in fails:
        print(f"FAIL: {f_}")
    if fails:
        return 1
    print("\nhead and probe numbers OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
