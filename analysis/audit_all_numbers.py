#!/usr/bin/env python3
r"""audit_all_numbers.py — every number in the paper, against every number on disk.

WHY
run_all_checks.sh verifies the generated tables and about twenty named prose figures. It says
nothing about the other ~1,200 numeric literals in the prose, captions and hand-written tables,
and that is exactly where this paper's errors have been: 0.117 for 0.119, 0.981 for a model rate
that is 0.986, "33,119 segments" described as "every released tag" when the pool is 38,616, a
head score attached to the wrong configuration. Each of those sat in prose no generator touched.

WHAT IT DOES
Two passes, neither of which trusts the paper.

  HARVEST. Walk every result artefact in the repository and collect the numbers in it: the
  aggregate CSVs, the run summaries, the per-instance predictions, the _summary and _meta records
  inside every scoring JSONL, and a recomputation of the dataset's own composition from the
  released parquet. Each observed value is recorded with every file that produced it. Values are
  indexed at several roundings, because the paper prints 0.596 for 0.5956.

  MATCH. Pull every numeric literal out of the tex, skipping comments and the eleven
  BEGIN/END GENERATED regions that the generators already own, and look each one up. What comes
  back is the set of figures in the paper that appear nowhere in the evidence.

WHAT A HIT MEANS
Not every unmatched number is wrong. Differences, percentages, sums, counts of systems, layer
indices, years, section numbers and page budgets are all legitimately absent from the result
files. The output is therefore a triage list, printed with its surrounding sentence so each one
can be judged, and grouped so the ones worth reading come first.

    python3 analysis/audit_all_numbers.py            # full report
    python3 analysis/audit_all_numbers.py --strict   # exit 1 if a QUANTITY-shaped number is unseen
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The paper source lives in its own repository and is optional here; see
# analysis/paper_source.py for the search order and the env vars that override it.
sys.path.insert(0, str(ROOT / "analysis"))
from paper_source import find_tex as _find_tex  # noqa: E402
_PAPER_TEX = _find_tex() or (ROOT / "paper" / "latex" / "acl_latex.tex")

TEX = _PAPER_TEX

#: where results live. Anything under a checkpoint directory is model weights, not a result.
ROOTS = ["results", "results_srv", "evidence",
         "experiments/comet_qe/results", "results/analysis"]
SKIP = re.compile(r"(checkpoint-|/\.git/|__pycache__|\.safetensors|\.bin$|/logs?/)")

NUM = re.compile(r"(?<![\w.])(-?\d[\d,]*(?:\.\d+)?)")
#: "2020--2024" yields a spurious -2024, "[12,16,20,-1]" a spurious 12,16,20, and a dated model
#: snapshot a spurious 04. None is a quantity the paper asserts.
ARTEFACT = re.compile(r"^(-20\d\d|0,1|12,16,20,|04|8,10,|16,20,)$")


def canon(x: float) -> list[str]:
    """Every rounding the paper might print this value at."""
    out = []
    for d in (0, 1, 2, 3, 4):
        out.append(f"{x:.{d}f}")
    if abs(x) < 1e12 and float(x).is_integer():
        out.append(str(int(x)))
    return out


def note(idx: dict[str, set[str]], value, where: str) -> None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return
    if x != x:                     # NaN
        return
    for k in canon(x):
        idx[k].add(where)
    # percentages: the paper prints 86.8% for a stored 0.868, and 0.5% for 0.005
    if -1.0 <= x <= 1.0:
        for k in canon(x * 100):
            idx[k].add(where + " (as %)")


def harvest_text(path: Path, idx: dict[str, set[str]], rel: str) -> None:
    try:
        txt = path.read_text(errors="ignore")
    except Exception:
        return
    # In a CSV the comma is a delimiter, but NUM treats it as a thousands separator, so
    # "general,4477,35.2" matches as the single token "4477,35.2" and neither value is indexed.
    # Split delimited files into fields first; elsewhere the thousands reading is the right one.
    if path.suffix in (".csv", ".tsv"):
        sep = "\t" if path.suffix == ".tsv" else ","
        for line in txt.splitlines():
            for field in line.split(sep):
                for m in NUM.finditer(field):
                    note(idx, m.group(1).replace(",", ""), rel)
        return
    for m in NUM.finditer(txt):
        note(idx, m.group(1).replace(",", ""), rel)


def harvest_jsonl(path: Path, idx: dict[str, set[str]], rel: str) -> None:
    """Scoring files are huge; only the _meta/_summary records carry reportable aggregates.

    The per-segment rows are re-aggregated by the dedicated generators, not here.
    """
    try:
        with path.open(encoding="utf-8", errors="ignore") as fh:
            for i, line in enumerate(fh):
                if i > 3 and '"_summary"' not in line:
                    if i > 400:
                        break
                    continue
                if '"_meta"' in line or '"_summary"' in line:
                    for m in NUM.finditer(line):
                        note(idx, m.group(1).replace(",", ""), rel)
    except Exception:
        return


def harvest_dataset(idx: dict[str, set[str]]) -> dict[str, int]:
    """Recompute the release's own composition rather than reading a cached count."""
    import pandas as pd
    facts: dict[str, int] = {}
    base = ROOT / "dataset" / "hf_snapshot" / "data"
    if not base.exists():
        return facts
    for cfg in sorted(p.name for p in base.iterdir() if p.is_dir()):
        fs = sorted(glob.glob(str(base / cfg / "*.parquet")))
        if not fs:
            continue
        parts = {Path(f).stem: pd.read_parquet(f) for f in fs}
        total = sum(len(d) for d in parts.values())
        note(idx, total, f"dataset/{cfg} total")
        facts[f"{cfg}:total"] = total
        for split, d in parts.items():
            note(idx, len(d), f"dataset/{cfg}/{split}")
            facts[f"{cfg}:{split}"] = len(d)
        df = pd.concat(parts.values(), ignore_index=True)
        if "language_pair" in df:
            for pair, n in df.language_pair.value_counts().items():
                note(idx, n, f"dataset/{cfg} {pair}")
            for flag in [c for c in df.columns if c.startswith("has_")]:
                note(idx, int(df[flag].sum()), f"dataset/{cfg} {flag}")
                for pair, sub in df.groupby("language_pair"):
                    note(idx, int(sub[flag].sum()), f"dataset/{cfg} {pair} {flag}")
        for col in ("domain", "domain_provenance", "wtag_provenance", "route", "sources"):
            if col in df:
                for v, n in df[col].astype(str).value_counts().items():
                    note(idx, n, f"dataset/{cfg} {col}={v}")
                if "language_pair" in df:
                    for (pair, v), n in df.groupby(["language_pair", df[col].astype(str)]).size().items():
                        note(idx, n, f"dataset/{cfg} {pair} {col}={v}")
        if {"mt", "post_edit"} <= set(df.columns):
            sub = df.dropna(subset=["mt", "post_edit"])
            for pair, s in sub.groupby("language_pair"):
                note(idx, float((s.mt == s.post_edit).mean()), f"dataset/{cfg} {pair} zero-edit")
        if "da_mean" in df:
            for pair, s in df.dropna(subset=["da_mean"]).groupby("language_pair"):
                note(idx, float(s.da_mean.mean()), f"dataset/{cfg} {pair} DA mean")
                note(idx, float(s.da_mean.std(ddof=1)), f"dataset/{cfg} {pair} DA sd")
                q = s.da_mean.quantile([.25, .75])
                note(idx, float(q.iloc[1] - q.iloc[0]), f"dataset/{cfg} {pair} DA IQR")
    return facts


def build_index() -> tuple[dict[str, set[str]], int]:
    idx: dict[str, set[str]] = defaultdict(set)
    n = 0
    for r in ROOTS:
        base = ROOT / r
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or SKIP.search(str(p)):
                continue
            rel = str(p.relative_to(ROOT))
            if p.suffix in (".csv", ".json", ".md", ".txt", ".tsv", ".out"):
                harvest_text(p, idx, rel); n += 1
            elif p.suffix == ".jsonl":
                harvest_jsonl(p, idx, rel); n += 1
    harvest_dataset(idx)
    return idx, n


# --------------------------------------------------------------------------------------------
# Numbers in the tex that are not quantities and never will be in a result file.
# --------------------------------------------------------------------------------------------
STRUCTURAL = re.compile(
    r"\\(?:S|ref|label|cite\w*|section|paragraph|multirow|multicolumn|cmidrule|resizebox|"
    r"includegraphics|columnwidth|textwidth|hskip|vspace|arraystretch|tiny|footnotesize)"
)
#: Numbers that are correct but computed rather than stored, so no result file contains them.
#: Each was verified against the released parquet or the scoring files on 12 Aug, and each is named
#: with the check that now owns it. A value leaving this map should be re-derived, not re-added.
VERIFIED_ELSEWHERE = {
    # tab:composition W-tag column: wtag_provenance set. scripts/verify_tab_composition.py
    "9982": "verify_tab_composition", "22190": "verify_tab_composition",
    "8612": "verify_tab_composition", "8679": "verify_tab_composition",
    # tab:routedist, recomputed from the challenge parquet (route and domain value counts)
    "2731": "challenge parquet", "2728": "challenge parquet",
    "1568": "challenge parquet", "1533": "challenge parquet",
    # S2 prose, recomputed from the full parquet
    "26388": "full parquet, X-to-en inferred domains",
    "89399": "full parquet, annotated and deduplicated",
    "5956": "full parquet, en-hi rows with no DA",
    "234446": "full parquet, sources per row summed",
    "77505": "full parquet, rows with more than one source",
    "5497": "38,616 real tags minus the 33,119 scored; the split is 5,473 + 24",
    # Appendix E / tab:direction / tab:fullda. analysis/verify_fullda.py
    "81315": "verify_fullda", "20625": "verify_fullda",
    # Appendix L, recomputed from the mqm task config
    "1116": "mqm parquet, WMT24 rows with a span",
    "2474": "mqm parquet, EAMT batch size",
}

KNOWN_NON_DATA = {
    # scales, definitions, settings and identifiers, all stated as design not measurement
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "16", "20", "24",
    "100", "95", "42", "43", "44", "256", "512", "300", "2020", "2022", "2023", "2024", "2025",
    "2026", "-1", "-5", "-7", "-11", "-16", "-20", "-24", "1.0", "0.0", "0.5", "4.0", "2.2",
    "2.4.2", "2.2.2", "1.3", "3.0", "3.2", "3.4", "0.8", "48", "80", "32", "22",
}


def main() -> int:
    strict = "--strict" in sys.argv
    idx, nfiles = build_index()
    print(f"harvested {len(idx):,} distinct roundings from {nfiles:,} result files "
          f"plus a recomputation of the released parquet\n")

    raw = TEX.read_text()
    stripped = re.sub(r"% BEGIN GENERATED.*?% END GENERATED \S+", " ", raw, flags=re.S)
    lines = re.sub(r"(?m)^%.*$", "", stripped).split("\n")

    unseen: list[tuple[str, str]] = []
    seen = 0
    for ln, line in enumerate(lines, 1):
        for m in NUM.finditer(line.replace("{,}", ",")):
            v = m.group(1)
            key = v.replace(",", "")
            ctx = line[max(0, m.start() - 60):m.end() + 60].strip()
            if STRUCTURAL.search(line[max(0, m.start() - 22):m.start()]):
                continue
            if ARTEFACT.match(v):
                continue
            if key in KNOWN_NON_DATA:
                continue
            if key in VERIFIED_ELSEWHERE:
                seen += 1
                continue
            if key in idx:
                seen += 1
            else:
                unseen.append((v, " ".join(ctx.split())))

    print(f"checkable numeric literals outside the generated tables: {seen + len(unseen):,}")
    print(f"  found in a result file : {seen:,}")
    print(f"  NOT found anywhere     : {len(unseen):,}\n")

    # a derived difference or sum is expected to be absent; a raw-looking score is not
    def shape(v: str) -> str:
        if "." in v and len(v.split(".")[1]) >= 3:
            return "3dp score"
        if "." in v:
            return "1-2dp"
        return "count"
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for v, c in unseen:
        buckets[shape(v)].append((v, c))
    for b in ("3dp score", "count", "1-2dp"):
        rows = buckets.get(b, [])
        if not rows:
            continue
        print(f"--- {b}: {len(rows)} not found in any result file " + "-" * 30)
        for v, c in rows:
            print(f"  {v:>10s}   {c[:112]}")
        print()

    if strict and buckets.get("3dp score"):
        print("STRICT: a three-decimal figure appears in the paper and in no result file")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
