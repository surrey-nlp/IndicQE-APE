#!/usr/bin/env python3
r"""build_axes.py — compute the four difficulty-axis flags for every released row.

The axes are defined once, in analysis/axis_contrast.py, and this script applies that definition
to the whole release so the flags can ship as columns. It adds columns and moves no rows.

    axis_a1 .. axis_a4   boolean per row, null where the pair has no signal for that axis
    axis_a4_arm          which arm of A4 placed the row, empty if it is not on A4
    n_axes               how many of the four the row is on
    sig_da_std, sig_human_ter, sig_bad_density, sig_hard_error
                         the four signals, released so every flag can be re-derived by hand

Thresholds are quantiles and z scores taken within language pair, over the whole release rather
than over a split, so a segment's flag does not depend on which split it landed in. An axis a pair
carries no signal for is null and not False: en-gu and en-te ship from WMT 2023 and WMT 2024 as
direct assessments alone, so they carry A1 and are null on the other three, and recording that as
"not flagged" would let a reader average it in.

The axes overlap and are deliberately shipped as four booleans rather than one categorical. Of the
7,616 flagged challenge rows only 3,265 sit on exactly one axis, and A2 and A3 share half their
union, so any single label would have to bucket the rest and would hide that A4 is the distinct one.

Four guards refuse to write on any change to the row count, the uid set, split membership or the
per-pair counts, because every prediction under results/ is keyed to the current test set and a
moved row would silently invalidate all of them.

    python3 dataset/build_axes.py            # write dataset/axes/axes.parquet
    python3 dataset/build_axes.py --report   # report only, write nothing
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from axis_contrast import AXES, PAIRS, SNAP, axis_flags, signals  # noqa: E402

OUT = ROOT / "dataset" / "axes"


def load(cfg: str) -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(str(SNAP / cfg / "*.parquet"))):
        d = pd.read_parquet(f)
        d["__split"] = Path(f).stem
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    report_only = "--report" in sys.argv
    full = load("full")
    ch = load("challenge")
    before = {"rows": len(full), "uids": set(full.uid),
              "splits": full.groupby("__split").size().to_dict(),
              "pairs": full.language_pair.value_counts().to_dict()}

    f = axis_flags(signals(full))
    f["n_axes"] = f[AXES].fillna(False).astype(bool).sum(axis=1)

    problems = []
    if len(f) != before["rows"]:
        problems.append(f"row count changed: {before['rows']} -> {len(f)}")
    if set(f.uid) != before["uids"]:
        problems.append("the uid set changed")
    if f.groupby("__split").size().to_dict() != before["splits"]:
        problems.append("split membership changed")
    if f.language_pair.value_counts().to_dict() != before["pairs"]:
        problems.append("per-pair counts changed")
    for p in problems:
        print(f"REFUSING: {p}")
    if problems:
        return 1

    keep = set(ch[ch.has_da.astype("boolean").fillna(False)].uid)
    g = f[f.uid.isin(keep)]
    print(f"release: {len(f)} rows; DA-bearing challenge test: {len(g)}\n")
    print(f"{'pair':7s} {'n':>6s} " + "".join(f"{a:>8s}" for a in AXES))
    for p in PAIRS:
        s = g[g.language_pair == p]
        cells = "".join(f"{'--':>8s}" if s[a].isna().all()
                        else f"{int(s[a].fillna(False).sum()):>8d}" for a in AXES)
        print(f"  {p:5s} {len(s):6d} " + cells)
    print(f"\naxes per row: {g.n_axes.value_counts().sort_index().to_dict()}")
    a4 = g[g.A4.fillna(False)]
    print(f"A4 on {len(a4)} rows over {a4.language_pair.nunique()} pairs, by arm: "
          f"{a4.A4_arm.value_counts().to_dict()}")

    if report_only:
        print("\n--report: nothing written")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    cols = (["uid", "language_pair", "__split", "n_axes", "A4_arm",
             "da_std", "human_ter", "bad_density", "hard_error"] + AXES)
    out = f[cols].rename(columns={a: f"axis_{a.lower()}" for a in AXES} |
                         {"A4_arm": "axis_a4_arm", "da_std": "sig_da_std",
                          "human_ter": "sig_human_ter", "bad_density": "sig_bad_density",
                          "hard_error": "sig_hard_error"})
    for a in AXES:
        out[f"axis_{a.lower()}"] = out[f"axis_{a.lower()}"].astype("boolean")
    out.to_parquet(OUT / "axes.parquet", index=False)
    print(f"\nwrote {(OUT / 'axes.parquet').relative_to(ROOT)} "
          f"({len(out)} rows, {len(out.columns)} columns); merge into the release on uid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
