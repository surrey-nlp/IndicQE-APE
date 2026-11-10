#!/usr/bin/env python3
"""Build the reference-free scoring input for the two pairs missing from the full-population run.

WHY THESE TWO AND NO OTHERS
---------------------------
`results/qe_metrics_full/` is built on the HF `ape` config: 73,829 rows over 7 pairs. en-gu and
en-te are absent from it because they carry no post-edits, so they are not in `ape` at all. They do
carry direct assessments, and they live in the `qe-da` config. Scoring them extends the
full-population DA table from 4 en->Indic pairs to 6.

REFERENCE-FREE ONLY. Both pairs have has_pe == 0, so XCOMET-XL ref-based and MetricX-24 ref-based
cannot be run on them; only the QE / reference-free modes apply. The builder asserts this rather
than trusting it.

    python3 cluster/r5_build_qeda_gap.py            # writes results/severity/r5_qeda_gap_input.jsonl
"""
from __future__ import annotations

import json
import os

import pandas as pd
from huggingface_hub import hf_hub_download

PAIRS = ["en-gu", "en-te"]
OUT = os.environ.get("R5_OUT", "results/qe_metrics_full/_r5_qeda_gap_input.jsonl")
#: expected release row counts, verified against HF at build time
EXPECTED = {"en-gu": 9109, "en-te": 13157}


def main():
    frames = []
    for split in ("train", "validation", "test"):
        try:
            frames.append(pd.read_parquet(hf_hub_download(
                "surrey-nlp/IndicQE-APE", f"data/tasks/qe-da/{split}.parquet", repo_type="dataset")))
        except Exception as e:                                          # split may not exist
            print(f"[skip] qe-da/{split}: {type(e).__name__}")
    d = pd.concat(frames, ignore_index=True)
    d = d[d.language_pair.isin(PAIRS)]

    got = d.language_pair.value_counts().to_dict()
    bad = {k: (EXPECTED[k], got.get(k, 0)) for k in EXPECTED if got.get(k, 0) != EXPECTED[k]}
    if bad:
        raise SystemExit(f"ABORT: row counts differ from the expected release counts. expected/got: {bad}")

    if "post_edit" in d.columns and d.post_edit.notna().any():
        n = int(d.post_edit.notna().sum())
        raise SystemExit(f"ABORT: {n} post-edits present; these pairs were assumed reference-free. "
                         "Re-check before running any ref-based mode.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in d.itertuples(index=False):
            if not str(r.source).strip() or not str(r.mt).strip():
                continue
            fh.write(json.dumps({
                "id": str(r.uid), "pair": str(r.language_pair).replace("-", ""),
                "source": str(r.source), "mt": str(r.mt),
                "gold_da_mean": None if pd.isna(r.da_mean) else float(r.da_mean),
            }, ensure_ascii=False) + "\n")
            n += 1
    da = int(d.da_mean.notna().sum())
    print(f"[write] {OUT}")
    print(f"  rows written : {n:,}   (en-gu {got['en-gu']:,} + en-te {got['en-te']:,})")
    print(f"  with gold DA : {da:,} ({100*da/len(d):.1f}%)")
    print(f"  post-edits   : 0 -- reference-free modes only")


if __name__ == "__main__":
    main()
