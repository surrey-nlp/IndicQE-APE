#!/usr/bin/env python3
"""paper_source.py -- locate the LaTeX source of the paper, which is optional.

The paper lives in its own repository, github.com/dipteshkanojia/IndicQEAPE, not in this one.
Sixteen of the checks in analysis/run_all_checks.sh compare a generated table or a prose figure
against that source, so they need it on disk; the rest read only the CSVs under results/ and
evidence/ and run on a bare clone of this repository.

Resolution order, first hit wins:

  1. $INDICQE_PAPER_TEX          an explicit path to acl_latex.tex
  2. $INDICQE_PAPER_DIR          a clone of the paper repo; latex/acl_latex.tex beneath it
  3. ./paper/latex/acl_latex.tex a clone placed inside this repository (gitignored)
  4. ../IndicQEAPE/latex/acl_latex.tex   a sibling clone next to this repository

Returns None when the source is absent, so callers can skip rather than crash:

    from paper_source import find_tex
    tex = find_tex()
    if tex is None:
        print("SKIP: paper source not found; see analysis/paper_source.py")
        raise SystemExit(0)

    python3 analysis/paper_source.py     # report what is resolvable, for debugging
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: how to get the paper source, quoted in the skip messages so the fix is one copy-paste away
CLONE_HINT = (
    "git clone https://github.com/dipteshkanojia/IndicQEAPE.git paper\n"
    "  or set INDICQE_PAPER_TEX=/path/to/acl_latex.tex"
)


def candidates() -> list[tuple[str, Path]]:
    """Every place the source might be, in resolution order, labelled for reporting."""
    out: list[tuple[str, Path]] = []
    if (v := os.environ.get("INDICQE_PAPER_TEX")):
        out.append(("$INDICQE_PAPER_TEX", Path(v)))
    if (v := os.environ.get("INDICQE_PAPER_DIR")):
        out.append(("$INDICQE_PAPER_DIR", Path(v) / "latex" / "acl_latex.tex"))
    out.append(("./paper", ROOT / "paper" / "latex" / "acl_latex.tex"))
    out.append(("../IndicQEAPE", ROOT.parent / "IndicQEAPE" / "latex" / "acl_latex.tex"))
    return out


def find_tex() -> Path | None:
    """The paper's acl_latex.tex, or None if no candidate exists."""
    for _, p in candidates():
        if p.is_file():
            return p
    return None


def require_tex() -> Path:
    """As find_tex, but exit 0 with an explanation rather than raising FileNotFoundError.

    Exit 0 is deliberate: a missing optional dependency is not a failed check.
    """
    tex = find_tex()
    if tex is None:
        print("SKIP: the paper source is not present, so this check cannot run.\n"
              f"  {CLONE_HINT}")
        raise SystemExit(0)
    return tex


if __name__ == "__main__":
    found = find_tex()
    print(f"repository root: {ROOT}\n")
    for label, p in candidates():
        mark = "FOUND " if p.is_file() else "  -   "
        print(f"  {mark} {label:22s} {p}")
    print(f"\nresolved: {found or 'none -- tex-dependent checks will skip'}")
