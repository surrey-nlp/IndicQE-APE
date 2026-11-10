#!/usr/bin/env python3
r"""ape_do_nothing.py — the baseline every APE evaluation needs and this one was missing.

In automatic post-editing the mandatory reference point is leaving the MT untouched. Without it a
reader cannot tell whether an APE system improves anything, and the surface metrics in tab:ape
cannot be interpreted at all: char-TER against a human post-edit measures distance from a string
that was produced BY EDITING THE MT, so the MT itself is the closest hypothesis there is by
construction. Quantifying that is the point.

The three metrics are computed exactly as for the systems, on exactly the same rows:

  char-TER, chrF++   indicqe_ape.metrics.ape_string_metrics(MT, post-edit), the same function the
                     system rows use, so the numbers are comparable cell for cell. The same call
                     returns word-level sacreBLEU and word-level TER, which the APE shared tasks
                     report; those go into the CSV too and are tabulated in the appendix. They are
                     not the primary pair here because word tokenisation charges a whole-word edit
                     for a one-character inflection, which inflates TER and deflates BLEU on
                     agglutinative targets. Reporting them anyway lets this table be compared with
                     the shared-task numbers.
  CometKiwi          reference-free, against the source. The APE pass scored the systems with
                     Unbabel/wmt22-cometkiwi-da, and the zero-shot QE run scored the MT with the
                     same checkpoint on the same challenge rows, so the do-nothing CometKiwi is
                     read from results/qe_metrics_ext rather than recomputed. The script asserts
                     the join covers every row.

Writes results/analysis/ape_do_nothing.csv, which make_ape_table.py splices as the first block of
tab:ape.
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from indicqe_ape.metrics import ape_string_metrics  # noqa: E402

CHALLENGE = ROOT / "dataset" / "hf_snapshot" / "data" / "challenge" / "test.parquet"
OUT = ROOT / "results" / "analysis" / "ape_do_nothing.csv"
PAIRS = {"en-hi": "enhi", "en-mr": "enmr", "en-ta": "enta", "en-ml": "enml"}
#: the APE test population, asserted so a silent population change cannot pass
EXPECTED = {"enhi": 1536, "enmr": 2098, "enta": 1448, "enml": 1436}


def cometkiwi_by_uid() -> dict[str, float]:
    out: dict[str, float] = {}
    for f in sorted(glob.glob(str(ROOT / "results" / "qe_metrics_ext" / "cometkiwi-da.*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if "_meta" in r or "_summary" in r or r.get("id") is None:
                continue
            if r.get("pred_score") is not None:
                out[r["id"]] = float(r["pred_score"])
    return out


def main() -> int:
    df = pd.read_parquet(CHALLENGE)
    ck = cometkiwi_by_uid()
    rows = []
    for pair, code in PAIRS.items():
        sub = df[(df.language_pair == pair) & df.has_pe].dropna(subset=["mt", "post_edit"])
        if len(sub) != EXPECTED[code]:
            print(f"FAIL: {code} has {len(sub)} rows, expected {EXPECTED[code]}")
            return 1
        m = ape_string_metrics(sub.mt.tolist(), sub.post_edit.tolist())
        # CometKiwi is read from the zero-shot QE run, which scored DA-BEARING rows only. Some
        # APE rows carry a post-edit but no DA (they arrive from the APE releases as triples), so
        # the reference-free column cannot be filled on the same population as the surface metrics.
        # Rather than mix populations inside one row, the table leaves it blank and the covered
        # subset is reported here and in the caption.
        have = sub.uid.map(ck)
        cov = float(have.notna().mean())
        identical = float((sub.mt == sub.post_edit).mean())
        rows.append(dict(system="do-nothing", pair=code, n=len(sub),
                         cometkiwi=(round(float(have.mean()), 4) if cov == 1.0 else ""),
                         cometkiwi_subset=round(float(have.mean()), 4),
                         cometkiwi_cov=round(cov, 3),
                         charTER=round(m["chrter"], 2), chrfpp=round(m["chrf"], 2),
                         wordTER=round(m["ter"], 2), wordBLEU=round(m["bleu"], 2),
                         zero_edit=round(identical, 3)))
        print(f"  {code}  n={len(sub):5d}  char-TER {m['chrter']:6.2f}  chrF++ {m['chrf']:6.2f}  "
              f"TER {m['ter']:6.2f}  BLEU {m['bleu']:6.2f}  "
              f"CometKiwi {have.mean():.4f} on {cov:.1%} of rows  zero-edit {identical:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
