#!/usr/bin/env python3
"""Structural check on tab:qe and tab:perpairfull, plus a values check via the generator.

This script existed before 2026-08-08 but never ran: line 68 held a backslash inside an f-string
expression, which is a SyntaxError on Python below 3.12, so every invocation died at import time
and the table it guards drifted unnoticed. Two defects accumulated:

  * a sync regex written for tab:recal matched across the enclosing table environment and
    overwrote three rows of tab:qe with tab:recal's columns, so \\mSarvam, \\mCKDA and \\mXC
    printed (raw r, bootstrap CI, r recalibrated, offset<->quality) in the (P, rho, tau,
    cross-lingual r) slots. A bootstrap interval was being printed in a Spearman column.
  * \\mAyaB appeared twice, once in the prompted block and again at the foot of the COMET block.

Neither is a rounding question, so the old cell-by-cell tolerance check was the wrong instrument.
The guards below are shaped to the failures actually seen:

  C1  each table has exactly the expected number of data rows
  C2  no model macro appears more than once in a table          <- the duplicate-row defect
  C3  every row has the arity its tabular declares
  C4  no cell contains a bracketed interval, which is tab:recal's signature and cannot
      legitimately appear in either of these tables             <- the cross-table overwrite
  C5  the printed values match analysis/make_qe_tables.py       (delegated to its --check)

Usage
-----
    python3 scripts/verify_tab_qe.py [tex]
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The paper source lives in its own repository and is optional here; see
# analysis/paper_source.py for the search order and the env vars that override it.
sys.path.insert(0, str(ROOT / "analysis"))
from paper_source import find_tex as _find_tex  # noqa: E402
_PAPER_TEX = _find_tex() or (ROOT / "paper" / "latex" / "acl_latex.tex")

TEX = Path(sys.argv[1]) if len(sys.argv) > 1 else _PAPER_TEX

MACROS = [r"\mBharatB", r"\mLlamaB", r"\mTiny", r"\mAyaB", r"\mSarvam", r"\mGPT",
          r"\mCKDA", r"\mCKXL", r"\mXC"]
#: label -> (expected data rows, cells per row including the leading macro)
SPEC = {"tab:qe": (9, 5), "tab:perpairfull": (9, 11)}
INTERVAL = re.compile(r"\[\s*[+-]?\d*\.?\d+\s*,")


def block_for(tex: str, label: str) -> str:
    """The table* or table environment that carries `label`."""
    pat = re.compile(r"\\begin\{table\*?\}(?:(?!\\begin\{table).)*?\\label\{"
                     + re.escape(label) + r"\}", re.S)
    m = pat.search(tex)
    if not m:
        sys.exit(f"FAIL: could not locate the {label} table environment")
    return m.group(0)


def main() -> int:
    tex = TEX.read_text(encoding="utf-8")
    failures: list[str] = []

    for label, (want_rows, want_cells) in SPEC.items():
        block = block_for(tex, label)
        seen: list[str] = []
        for line in block.splitlines():
            m = re.match(r"\s*(\\m\w+)\s*&(.*?)\\\\\s*$", line)
            if not m or m.group(1) not in MACROS:
                continue
            macro, rest = m.group(1), m.group(2)
            seen.append(macro)
            n = len(rest.split("&")) + 1
            if n != want_cells:                                                    # C3
                failures.append(f"{label}: {macro} has {n} cells, expected {want_cells}")
            if INTERVAL.search(rest):                                              # C4
                failures.append(
                    f"{label}: {macro} contains a bracketed interval, which belongs to "
                    f"tab:recal -- this table has been overwritten across environments")
        if len(seen) != want_rows:                                                 # C1
            failures.append(f"{label}: {len(seen)} data rows, expected {want_rows}")
        dupes = {m for m in seen if seen.count(m) > 1}                             # C2
        if dupes:
            failures.append(f"{label}: duplicated rows for {', '.join(sorted(dupes))}")
        missing = [m for m in MACROS if m not in seen]
        if missing:
            failures.append(f"{label}: missing rows for {', '.join(missing)}")
        print(f"{label:18s} rows={len(seen)}  unique={len(set(seen))}  "
              f"{'OK' if not dupes and len(seen) == want_rows else 'PROBLEM'}")

    gen = ROOT / "analysis" / "make_qe_tables.py"
    if gen.exists():                                                               # C5
        print(f"\nrunning {gen.relative_to(ROOT)} --check ...")
        r = subprocess.run([sys.executable, str(gen), "--check"],
                           cwd=ROOT, capture_output=True, text=True)
        tail = (r.stdout or r.stderr).strip().splitlines()
        print("  " + "\n  ".join(tail[-3:]))
        if r.returncode != 0:
            failures.append("printed values do not match make_qe_tables.py")
    else:
        failures.append(f"generator missing: {gen}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nPASS: tab:qe and tab:perpairfull are well-formed and match the generator.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
