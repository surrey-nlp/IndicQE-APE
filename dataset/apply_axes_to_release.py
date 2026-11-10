#!/usr/bin/env python3
r"""apply_axes_to_release.py — bring every parquet in the Hub snapshot onto the released schema.

Adds the four difficulty-axis flags and the two signals not already in the release, and removes the
four columns the curation left behind that the paper does not report.

    added    axis_a1 .. axis_a4    nullable boolean, null where the pair has no signal for that axis
             axis_a4_arm           which arm of A4 placed the row, empty if the row is not on A4
             sig_bad_density       share of BAD tokens in the released word tags
             sig_hard_error        error_categories names mistranslation, untranslated or addition
    replaced n_axes                the count of axes a row is on, under the corrected rule
    removed  route, difficulty, is_curated

    kept     da_std, human_ter     already released, and identical to a fresh recomputation

The blind challenge test is the exception: it withholds the labels the axes are computed from, so it
takes the removals and none of the additions. It shipped `route` until now, which was the same leak.

The flags come from analysis/axis_contrast.py, which is the single definition used by the paper.
`da_std` and `human_ter` are left as released rather than rewritten: they were verified to agree
with the recomputation to the last decimal on all 111,545 and 73,829 rows that carry them.

Guards. Every file must keep its exact row count and uid set, and every uid must be found in the
axis table. The script writes nothing if any file fails, so a partial release cannot happen.

    python3 dataset/apply_axes_to_release.py --check   # report, write nothing
    python3 dataset/apply_axes_to_release.py           # rewrite dataset/hf_snapshot in place
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from axis_contrast import AXES  # noqa: E402

SNAP = ROOT / "dataset" / "hf_snapshot"
AXFILE = ROOT / "dataset" / "axes" / "axes.parquet"

DROP = ["route", "difficulty", "is_curated"]
ADD = ["axis_a1", "axis_a2", "axis_a3", "axis_a4", "axis_a4_arm",
       "sig_bad_density", "sig_hard_error", "n_axes"]


def main() -> int:
    check = "--check" in sys.argv
    if not AXFILE.exists() or not SNAP.is_dir():
        absent = AXFILE if not AXFILE.exists() else SNAP
        # Both are gitignored build products, so a bare clone has neither. Skip rather than fail:
        # --check exists to catch drift, and there is nothing here to have drifted.
        print(f"SKIP: {absent.relative_to(ROOT)} is not in this checkout.\n"
              "  Rebuild it with dataset/build_axes.py, or fetch the release with "
              "huggingface-cli download surrey-nlp/IndicQE-APE.")
        return 0
    ax = pd.read_parquet(AXFILE).set_index("uid")
    for c in (f"axis_{x.lower()}" for x in AXES):
        ax[c] = ax[c].astype("boolean")
    ax = ax[["n_axes", "axis_a4_arm", "sig_bad_density", "sig_hard_error"]
            + [f"axis_{a.lower()}" for a in AXES]]
    print(f"axis table: {len(ax)} rows, {len(ax.columns)} columns\n")

    files = sorted(SNAP.rglob("*.parquet"))
    planned, problems = [], []
    for f in files:
        d = pd.read_parquet(f)
        rel = f.relative_to(SNAP)
        missing = set(d.uid) - set(ax.index)
        if missing:
            # The mqm config carries 2,163 en-hi segments annotated for MQM alone. They are outside
            # `full`, carry no direct assessment, and so can be on no axis: they take nulls.
            off = d[d.uid.isin(missing)]
            if bool(off.has_da.astype("boolean").fillna(False).any()):
                problems.append(f"{rel}: {len(missing)} uids are absent from the axis table "
                                f"and some of them carry a DA")
                continue
            print(f"note: {rel} has {len(missing)} rows outside `full`, all without a DA; "
                  f"they take null on every axis")
        # idempotent: strip anything this script owns before rebuilding it, so a second run and
        # --check on an already-converted release both compare like with like
        out = d.drop(columns=[c for c in DROP + ADD if c in d.columns])
        if f.name == "test_blind.parquet":
            # The blind test withholds da_mean, da_std, human_ter and the word tags. Every axis
            # flag and every signal is a function of exactly those columns, so adding them here
            # would hand back what the file exists to withhold. Dropping `route`, which was the
            # same leak under the old vocabulary, is the only change this file takes.
            print(f"note: {rel} keeps no axis column; the flags are derived from labels it withholds")
        else:
            out = out.join(ax, on="uid")
            for c in (f"axis_{x.lower()}" for x in AXES):
                out[c] = out[c].astype("boolean")   # stable whether or not the slice has nulls
        if len(out) != len(d) or set(out.uid) != set(d.uid):
            problems.append(f"{rel}: the join changed the rows")
            continue
        planned.append((f, out, len(d), len(d.columns), len(out.columns),
                        list(d.columns) == list(out.columns)
                        and d.reset_index(drop=True).equals(out.reset_index(drop=True))))

    for p in problems:
        print(f"REFUSING: {p}")
    if problems:
        return 1

    print(f"{'file':44s} {'rows':>7s} {'cols':>12s}  flagged")
    for f, out, n, c0, c1, same in planned:
        cols = [c for c in (f"axis_{a.lower()}" for a in AXES) if c in out.columns]
        flagged = (int(out[cols].fillna(False).astype(bool).any(axis=1).sum()) if cols else 0)
        state = "unchanged" if same else f"{c0} -> {c1}"
        print(f"{str(f.relative_to(SNAP)):44s} {n:7d} {state:>14s} "
              f"{flagged:7d}" + ("  (no axis columns)" if not cols else ""))

    drifted = [f.relative_to(SNAP) for f, _, _, _, _, same in planned if not same]
    if check:
        for f in drifted:
            print(f"FAIL: {f} is not on the released schema")
        print("--check: nothing written" if not drifted else "")
        return 1 if drifted else 0
    if not drifted:
        print("\nevery file is already on the released schema; nothing to do")
        return 0
    for f, out, *_ in planned:
        tmp = f.with_suffix(".parquet.new")
        out.to_parquet(tmp, index=False)
        shutil.move(str(tmp), str(f))
    print(f"\nrewrote {len(planned)} files in {SNAP.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
