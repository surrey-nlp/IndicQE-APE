#!/usr/bin/env python3
r"""release_population.py — the one definition of "a segment we are allowed to score".

WHY THIS EXISTS
Run files accumulate. A model scored on an earlier build of the challenge test leaves a file whose
rows include segments that are no longer in the release, and a loader that reads the file rather
than the release will happily report a correlation over them.

That had happened. gpt-5.5's zero-shot files were produced before the challenge test was extended:

    pair    rows in file   still in the release
    en-gu       1084              550
    en-te       1113              569
    en-hi       2045             1870
    en-mr       2132             1887
    en-ta       2027             1881
    et/ne/si    1000            1000 each

So 1,634 of the 12,537 segments behind gpt-5.5's published zero-shot macro are not in the benchmark,
and half of its en-gu and en-te cells are. Every other prompted model was rerun on the current test
and is unaffected, as are the COMET metrics.

Every loader in analysis/ now filters through `released()`, so a stale file can only ever produce a
smaller cell, never a wrong one. `coverage_report()` prints what each file lost, so a thin cell is
visible rather than silent.
"""
from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHALLENGE = ROOT / "dataset" / "hf_snapshot" / "data" / "challenge" / "test.parquet"


@functools.lru_cache(maxsize=2)
def released(da_only: bool = True) -> frozenset[str]:
    """uids of the released challenge test; DA-bearing only by default."""
    df = pd.read_parquet(CHALLENGE, columns=["uid", "has_da"])
    if da_only:
        df = df[df.has_da]
    return frozenset(df.uid)


def filter_cell(df: pd.DataFrame, uid_col: str = "uid") -> pd.DataFrame:
    """Drop rows that are not in the released challenge test."""
    keep = released()
    return df[df[uid_col].isin(keep)]


def coverage(uids) -> tuple[int, int]:
    """(kept, dropped) for a sequence of uids."""
    keep = released()
    u = list(uids)
    k = sum(1 for x in u if x in keep)
    return k, len(u) - k
