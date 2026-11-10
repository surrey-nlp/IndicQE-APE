#!/usr/bin/env python3
"""Verify every cell of tab:composition against the release.

The appendix claims this script "re-derives every per-pair and per-label count of
Table 1 from the live repository and fails on drift". Until 2026-08-08 it did not:

  * it loaded the `all` configuration (100,366 rows, en->Indic only), so the three
    X->en pairs read 0 across every label column and the Total row printed 100,366
    against the table's 126,754. The nine-pair table cannot be checked from a
    six-pair configuration.
  * its W-tag column counted a non-empty `wtag` field, which predates the tag
    derivation, so it printed 12,604 against the table's 79,301.
  * it had no MQM column at all, and its inline expected-value table was two
    revisions stale (en-ml Expl. 9,637, X->en APE 0).

Column definitions, which are the part that matters:

  Total   rows in `full` for the pair (train + validation + test)
  DA      da_mean is not null
  PE      post_edit is non-empty
  W-tag   wtag_provenance is set, i.e. the row carries a tag from any provenance
          class (upstream, mlqe-pe-upstream, or derived here)
  MQM     mqm_score is not null within `full` (the DA-paired subset; the separate
          `mqm` configuration adds 2,163 MQM-only rows that are not in `full`)
  Expl.   membership of the explainable-qe configuration, not the count of
          non-empty description strings
  Domain  domain_provenance == "annotated", i.e. a gold label rather than one
          inferred by the TF-IDF classifier
  QE      challenge-test rows carrying a DA score
  APE     challenge-test rows carrying a post-edit

Usage
-----
    python3 scripts/verify_tab_composition.py              # from the Hugging Face Hub
    python3 scripts/verify_tab_composition.py --local DIR  # from a local snapshot

Exit status is 0 when every cell matches and 2 otherwise, so it can gate a commit.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

REPO = "surrey-nlp/IndicQE-APE"
PAIRS = ["en-hi", "en-mr", "en-gu", "en-ta", "en-te", "en-ml", "et-en", "ne-en", "si-en"]
COLS = ["Total", "DA", "PE", "W-tag", "MQM", "Expl.", "Domain", "QE", "APE"]

#: What tab:composition prints, in COLS order.
PAPER: dict[str, tuple[int, ...]] = {
    #          Total     DA      PE   W-tag   MQM    Expl.  Domain    QE   APE
    "en-hi": (21092, 15136,  9983,  9982, 2327,  1999, 19301, 1881, 1536),
    "en-mr": (30746, 29182, 22189, 22190,    0,  5000, 30314, 2165, 2098),
    "en-gu": ( 9109,  9109,     0,     0,    0,     0,  8996, 1090,    0),
    "en-ta": (16262,  9002, 10838, 10838,    0,  2645, 14886, 2038, 1448),
    "en-te": (13157, 13157,     0,     0,    0,     0,  8995, 1120,    0),
    "en-ml": (10000, 10000, 10000, 10000,    0, 10000, 10000, 1436, 1436),
    "et-en": ( 9000,  9000,  9000,  9000,    0,     0,     0, 1000, 1000),
    "ne-en": ( 8649,  8649,  8332,  8612,    0,     0,     0, 1000,  962),
    "si-en": ( 8739,  8739,  3487,  8679,    0,     0,     0, 1000,  380),
}


def nonempty(s: pd.Series) -> pd.Series:
    return s.notna() & (s.astype(str).str.strip() != "")


def loader(local: str | None):
    if local:
        root = Path(local)

        def get(config: str, splits=("train", "validation", "test")) -> pd.DataFrame:
            frames = []
            for sp in splits:
                for f in glob.glob(str(root / "data" / config / f"{sp}.parquet")):
                    frames.append(pd.read_parquet(f))
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        from huggingface_hub import hf_hub_download

        def get(config: str, splits=("train", "validation", "test")) -> pd.DataFrame:
            frames = []
            for sp in splits:
                try:
                    p = hf_hub_download(REPO, f"data/{config}/{sp}.parquet",
                                        repo_type="dataset")
                except Exception:
                    continue
                frames.append(pd.read_parquet(p))
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return get


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", help="directory holding data/<config>/<split>.parquet")
    args = ap.parse_args()
    if not args.local:
        default = Path(__file__).resolve().parents[1] / "dataset" / "hf_snapshot"
        if (default / "data" / "full").is_dir():
            args.local = str(default)
            print(f"using local snapshot {default}")
    get = loader(args.local)

    full = get("full")
    if full.empty:
        print("FAIL: could not load the `full` configuration")
        return 2
    xq = get("tasks/explainable-qe")
    chal = get("challenge", ("test",))
    members = set(xq["uid"]) if len(xq) else set()

    rows = {}
    for pair in PAIRS:
        a = full[full.language_pair == pair]
        c = chal[chal.language_pair == pair]
        rows[pair] = (
            len(a),
            int(a["da_mean"].notna().sum()),
            int(nonempty(a["post_edit"]).sum()),
            int(nonempty(a["wtag_provenance"]).sum()),
            int(a["mqm_score"].notna().sum()) if "mqm_score" in a else 0,
            int(a["uid"].isin(members).sum()),
            int((a["domain_provenance"] == "annotated").sum()),
            int(c["da_mean"].notna().sum()),
            int(nonempty(c["post_edit"]).sum()),
        )

    n = len(COLS)
    hdr = f"{'pair':7s}" + "".join(f"{c:>9s}" for c in COLS)
    print("\nMEASURED FROM THE RELEASE")
    print(hdr)
    for p in PAIRS:
        print(f"{p:7s}" + "".join(f"{v:>9,d}" for v in rows[p]))
    tot = [sum(rows[p][i] for p in PAIRS) for i in range(n)]
    print(f"{'TOTAL':7s}" + "".join(f"{v:>9,d}" for v in tot))

    bad = []
    for p in PAIRS:
        d = [rows[p][i] - PAPER[p][i] for i in range(n)]
        if any(d):
            bad.append((p, d))
    if bad:
        print("\nDRIFT AGAINST tab:composition (measured - printed)")
        print(hdr)
        for p, d in bad:
            print(f"{p:7s}" + "".join((f"{x:>+9,d}" if x else f"{'.':>9s}") for x in d))
        print(f"\nFAIL: {len(bad)} of {len(PAIRS)} pairs differ.")
        print("\nLaTeX rows for tab:composition (measured):")
        for p in PAIRS:
            cells = " & ".join(f"{v:,}".replace(",", "{,}") for v in rows[p])
            print(f"  {p} & {cells} \\\\")
        cells = " & ".join(f"\\textbf{{{v:,}}}".replace(",", "{,}") for v in tot)
        print(f"  \\textbf{{Total}} & {cells} \\\\")
        return 2

    print(f"\nPASS: all {len(PAIRS) * n} cells match tab:composition.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
