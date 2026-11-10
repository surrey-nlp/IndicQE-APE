#!/usr/bin/env python3
"""probe_vs_prompt.py — read the same frozen model two ways and compare.

S4 trains a regression head on the hidden states of a frozen tiny-aya-fire and reports macro
Spearman 0.527 over the six en->Indic pairs. S3 prompts the same checkpoint for a GEMBA-DA score
and reports 0.290, but that macro is over nine pairs, so the two figures are not comparable as
printed. This recomputes the prompted macro on the six pairs the probe covers, which is the only
comparison that holds the model, the population and the metric fixed.

The prompted per-pair cells are loaded through make_qe_tables.load, so they take the same strict
parser and the same release filter as tab:qe. Anything the parser rejects is dropped rather than
scored, exactly as in S3.

    python3 analysis/probe_vs_prompt.py           # write results/analysis/probe_vs_prompt.csv
    python3 analysis/probe_vs_prompt.py --check   # recompute and diff against the CSV

Consumed by S4. The probe side is read from evidence/alope/, not recomputed here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import make_qe_tables as M  # noqa: E402

#: the six en->Indic pairs the probe is trained and evaluated on
EN_INDIC = ["enhi", "enmr", "enta", "engu", "ente", "enml"]
#: (csv name, source dirs) -- the probed model, and the strongest prompted model as a ceiling
MODELS = [("tiny-aya-fire", ["qe_rerun512"]), ("gpt-5.5", ["qe_closed_rerun", "qe_closed"])]
#: probe per-pair Spearman at the fixed layer -20, from evidence/alope/
PROBE = ROOT / "evidence" / "alope" / "evaluation_challenge_summary_indic.csv"
OUT = ROOT / "results" / "analysis" / "probe_vs_prompt.csv"


def prompted(model: str, dirs: list[str]) -> dict[str, float]:
    cells = M.load(model, dirs)
    out = {}
    for p in EN_INDIC:
        c = cells[p].dropna(subset=["gold", "pred"])
        out[p] = float(spearmanr(c.gold, c.pred).statistic)
    return out


def probe() -> dict[str, float]:
    d = pd.read_csv(PROBE)
    d = d[(d.route == "ALL") & (d.layer == -20)]
    return {p.replace("-", ""): float(v) for p, v in zip(d.lang_pair, d.spearman)}


def main() -> int:
    # The raw prompted-run JSONL is gitignored (megabytes), so it is absent from a bare clone.
    # A missing input is not drift: skip, as the paper-source checks do.
    missing = [d for _, dirs in MODELS for d in dirs if not (ROOT / "results" / d).is_dir()]
    if missing or not PROBE.exists():
        what = "results/" + ", results/".join(sorted(set(missing))) if missing else PROBE.name
        print(f"SKIP: {what} is not in this checkout, so this check cannot run.\n"
              "  The per-instance predictions ship in the evidence bundle "
              "(analysis/export_evidence_bundle.py).")
        return 0
    cols = {"probe": probe()}
    for m, dirs in MODELS:
        cols[m] = prompted(m, dirs)

    df = pd.DataFrame([{"pair": p, **{k: round(v[p], 3) for k, v in cols.items()}}
                       for p in EN_INDIC])
    df.loc[len(df)] = {"pair": "macro",
                       **{k: round(float(np.mean([v[p] for p in EN_INDIC])), 3)
                          for k, v in cols.items()}}

    print("Spearman on the six en->Indic pairs of the challenge test\n")
    print(df.to_string(index=False))
    mac = df[df.pair == "macro"].iloc[0]
    print(f"\nprobe ahead of prompting the same weights on every pair: "
          f"{bool((df[df.pair != 'macro'].probe > df[df.pair != 'macro']['tiny-aya-fire']).all())}")
    print(f"gap, probe minus prompted (same model)  : "
          f"{mac['probe'] - mac['tiny-aya-fire']:+.3f}")
    print(f"gap, probe minus the strongest prompted : {mac['probe'] - mac['gpt-5.5']:+.3f}")

    if "--check" in sys.argv:
        if not OUT.exists():
            print(f"\nFAIL: {OUT.relative_to(ROOT)} is missing")
            return 1
        old = pd.read_csv(OUT)
        if not old.equals(df):
            print(f"\nFAIL: {OUT.relative_to(ROOT)} is out of date")
            return 1
        print(f"\n--check: {OUT.relative_to(ROOT)} is up to date")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
