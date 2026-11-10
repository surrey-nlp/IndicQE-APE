#!/usr/bin/env python3
"""Recompute APE metrics from the stored post-edits (hyp_pe/ref_pe). One population per pair.

chrF++ and char-TER are the Indic-correct pair; word BLEU/TER are kept for reference only, since
the 13a tokeniser under-credits Indic morphology.

WHY THIS HAS A SOURCE PRECEDENCE
Until the ape_regen cluster job landed (2026-08-08) the en-hi/mr/ta cells could only be scored
from results/ape_open, which holds the OLD challenge test (1,490 / 2,024 / 1,375 rows), while
en-ml was already on the current one (1,436). That put two populations in one table, which is
exactly the defect this table has been cleaned of twice. ape_regen now supplies en-hi/mr/ta on the
current test (1,536 / 2,098 / 1,448), so every cell of tab:ape sits on the same population.

    SOURCES        first directory holding a usable cell wins
      ape_regen    en-hi, en-mr, en-ta   current challenge test, cluster job ape_regen
      ape_open     en-ml                 already current; also the old en-hi/mr/ta, now superseded

Cells whose row count does not match the challenge test are a hard error rather than a warning:
a silently stale cell is the failure mode this script exists to prevent. Pass --allow-stale to
score anyway (it will label the row), which is only useful for a deliberate old-vs-new comparison.

Usage
-----
    python scripts/ape_metrics.py                 # score and write the CSV
    python scripts/ape_metrics.py --compare       # print old (ape_open) against new (ape_regen)
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

from indicqe_ape.metrics import ape_string_metrics

R = "results"
SOURCES = ["ape_regen", "ape_open"]          # precedence order
OUT_CSV = f"{R}/analysis/ape_metrics_table.csv"

#: rows carrying a post-edit in the current challenge test, per pair
EXPECTED = {"enhi": 1536, "enmr": 2098, "enta": 1448, "enml": 1436}


def read_cell(path: str) -> tuple[list[str], list[str]]:
    H, Rf = [], []
    for ln in open(path, encoding="utf-8"):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if "_meta" in r or "_summary" in r:
            continue
        if r.get("hyp_pe") and r.get("ref_pe"):
            H.append(r["hyp_pe"])
            Rf.append(r["ref_pe"])
    return H, Rf


def collect() -> dict[str, tuple[str, list[str], list[str]]]:
    """key -> (source dir, hypotheses, references), first source in SOURCES wins."""
    cells: dict[str, tuple[str, list[str], list[str]]] = {}
    for d in SOURCES:
        for f in sorted(glob.glob(f"{R}/{d}/*.jsonl")):
            key = os.path.basename(f).replace(".jsonl", "")
            if key in cells:
                continue
            H, Rf = read_cell(f)
            if not H:                        # 300-byte _meta-only stubs (gemma-4-31B)
                continue
            cells[key] = (d, H, Rf)
    return cells


def main() -> int:
    compare = "--compare" in sys.argv
    allow_stale = "--allow-stale" in sys.argv

    if compare:
        print(f"{'cell':<30}{'source':<12}{'n':>6}{'chrF++':>9}{'charTER':>9}")
        for d in SOURCES:
            for f in sorted(glob.glob(f"{R}/{d}/*.jsonl")):
                H, Rf = read_cell(f)
                if not H:
                    continue
                m = ape_string_metrics(H, Rf)
                key = os.path.basename(f).replace(".jsonl", "")
                print(f"{key:<30}{d:<12}{len(H):>6}{m['chrf']:>9.1f}{m['chrter']:>9.1f}")
        return 0

    cells = collect()
    stale = []
    for key, (d, H, _) in sorted(cells.items()):
        pair = key.rsplit(".", 1)[-1]
        exp = EXPECTED.get(pair)
        if exp is not None and len(H) != exp:
            stale.append(f"{key} from {d}: {len(H)} rows, challenge test has {exp}")
    if stale and not allow_stale:
        print("REFUSING TO WRITE: cells are not on the current challenge test\n")
        for s in stale:
            print("  " + s)
        print("\nEvery cell of tab:ape must sit on one population. Fix the source or pass "
              "--allow-stale to score anyway.")
        return 2

    rows = []
    for key, (d, H, Rf) in sorted(cells.items()):
        m = ape_string_metrics(H, Rf)
        rows.append((key, m["chrf"], m["chrter"], m["ter"], m["bleu"], len(H), d))

    os.makedirs(f"{R}/analysis", exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "chrF++", "charTER", "wordTER", "wordBLEU", "n", "source"])
        for r in rows:
            w.writerow([r[0]] + [f"{x:.2f}" for x in r[1:5]] + [r[5], r[6]])

    print(f"{'system':<30}{'chrF++':>8}{'charTER':>9}{'wTER':>8}{'wBLEU':>8}{'n':>7}  source")
    for r in rows:
        print(f"{r[0]:<30}{r[1]:>8.1f}{r[2]:>9.1f}{r[3]:>8.1f}{r[4]:>8.1f}{r[5]:>7}  {r[6]}")
    print(f"\nwrote {OUT_CSV}  ({len(rows)} cells, chrF++/charTER are the Indic-correct pair)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
