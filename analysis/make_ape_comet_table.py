#!/usr/bin/env python3
"""make_ape_comet_table.py — collate the CometKiwi pass over the regenerated post-edits.

The ape_regen sbatch runs with --no-comet, because a second large model beside sarvam-m does not
fit in the array job. The neural pass is therefore a separate stage; it was run on a GPU workstation
by cluster/ape_comet_local.sh and its per-cell JSONs live in results/ape_metrics_regen/.

WHY IT HAD TO BE RERUN
The CometKiwi figures previously quoted in S7 for tiny-aya (0.781 / 0.701 / 0.770) were computed
over the OLD post-edits at n = 1,490 / 2,024 / 1,375. Those hypotheses were replaced when the
ape_regen job put tab:ape on the current challenge test, so the scored text no longer existed. The
rerun reproduces them closely on the new population (0.782 / 0.697 / 0.771), and extends the check
from one system to all three.

WHAT THE JOINED TABLE SHOWS
char-TER and chrF++ measure distance to the human post-edit, which is anchored to the MT
(S3, "human-targeted"). CometKiwi scores the output standalone against the source. sarvam-translate
retranslates rather than post-edits, so it departs from the post-edit by construction and the
surface metrics penalise it however good the translation is. That is why the two views agree on
en-hi and invert exactly on en-mr and en-ta.

Writes results/analysis/ape_comet_table.csv.
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMET_DIRS = [ROOT / "results" / "ape_metrics_regen", ROOT / "results" / "ape_metrics_enml"]
SURFACE = ROOT / "results" / "analysis" / "ape_metrics_table.csv"
OUT = ROOT / "results" / "analysis" / "ape_comet_table.csv"

NAME = {"sarvam-m.s0": "sarvam-m", "sarvam-translate.mt": "sarvam-t", "tiny-aya-fire.s0": "tiny-aya"}
PAIRS = ["enhi", "enmr", "enta", "enml"]


def main() -> int:
    files = sorted(f for d in COMET_DIRS for f in glob.glob(str(d / "*.json")))
    if not files:
        raise SystemExit(f"no CometKiwi JSONs in {COMET_DIRS}")
    surface = {r["system"]: r for r in csv.DictReader(open(SURFACE))}

    rows = []
    for f in files:
        d = json.load(open(f))
        key = d["file"].replace(".jsonl", "")
        system, pair = key.rsplit(".", 1)
        s = surface.get(key)
        if s is None:
            raise SystemExit(f"{key} has a CometKiwi score but no surface row in {SURFACE.name}")
        if int(s["n"]) != int(d["n"]):
            raise SystemExit(
                f"POPULATION MISMATCH for {key}: CometKiwi n={d['n']}, surface n={s['n']}. "
                f"The two passes must score the same hypotheses.")
        rows.append({"system": NAME[system], "pair": pair, "n": d["n"],
                     "cometkiwi": round(d["cometkiwi_mean"], 4),
                     "charTER": float(s["charTER"]), "chrF++": float(s["chrF++"])})

    rows.sort(key=lambda r: (PAIRS.index(r["pair"]), -r["cometkiwi"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["system", "pair", "n", "cometkiwi", "charTER", "chrF++"])
        w.writeheader()
        w.writerows(rows)

    print(f"{'pair':<7}{'system':<11}{'n':>6}{'CometKiwi':>11}{'char-TER':>10}{'chrF++':>9}")
    for pair in PAIRS:
        grp = [r for r in rows if r["pair"] == pair]
        for r in grp:
            print(f"{r['pair']:<7}{r['system']:<11}{r['n']:>6}{r['cometkiwi']:>11.4f}"
                  f"{r['charTER']:>10.1f}{r['chrF++']:>9.1f}")
        ck = [r["system"] for r in grp]
        cf = [r["system"] for r in sorted(grp, key=lambda r: -r["chrF++"])]
        print(f"       CometKiwi {' > '.join(ck)}")
        print(f"       chrF++    {' > '.join(cf)}   "
              f"{'agree' if ck == cf else 'INVERTED'}\n")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(rows)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
