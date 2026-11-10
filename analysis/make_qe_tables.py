#!/usr/bin/env python3
"""make_qe_tables.py — generate tab:qe and tab:perpairfull from the run outputs.

Both tables were hand-maintained and had drifted, in two separate ways.

1. tab:qe was CORRUPTED. A sync regex written for tab:recal matched across the enclosing table
   environment and overwrote three rows of tab:qe with tab:recal's columns, so \\mSarvam, \\mCKDA
   and \\mXC carried (raw r, bootstrap CI, r recalibrated, offset<->quality) in the
   (P, rho, tau, cross-lingual r) slots -- a bootstrap interval was printed in a Spearman column.
   \\mAyaB was also duplicated, appearing once in the prompted block and again at the foot of the
   COMET block.

2. Both tables were on the LEGACY vintage (results/analysis/qe_meta_s0.csv: permissive parser),
   while tab:prompting and tab:parse4 report the strict parser on results/qe_rerun512. The paper
   argues in S4 that the parsing rule is part of the measurement, so the strict vintage is the
   reported one and these two tables have to move onto it. The differences are small but real
   (tiny-aya rho 0.276 legacy against 0.290 strict), and leaving them would have tab:qe and
   tab:prompting print different zero-shot macro Spearman for the same model.

Conventions, matching make_prompting_tables.py so the three tables cannot disagree:

  parser      parse_qe_score_strict on the saved raw reply
  source      results/qe_rerun512 for the open models; gpt-5.5 from results/qe_closed_rerun if it
              exists, else results/qe_closed
  population  the 12,730 DA-bearing rows of the 13,032-segment challenge test. A prompted model
              contributes the subset it returns a usable score for; the COMET metrics score all of
              them. Compliance is reported in tab:prompting, not folded into these correlations.
  aggregation macro over the nine pairs for P/rho/tau. The cross-lingual column is a single Pearson
              correlation across pairs between per-pair model means and per-pair human means, so it
              rests on nine points and is printed at 2 dp to avoid implying precision it lacks.
  COMET rows  read from results/analysis/qe_meta_s0.csv. These metrics never see a prompt and have
              no parse step, so the two vintages are identical for them.

Splices between the sentinels; --check exits 1 if the file is out of date.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
# The paper source lives in its own repository and is optional here; see
# analysis/paper_source.py for the search order and the env vars that override it.
sys.path.insert(0, str(ROOT / "analysis"))
from paper_source import find_tex as _find_tex  # noqa: E402
_PAPER_TEX = _find_tex() or (ROOT / "paper" / "latex" / "acl_latex.tex")

sys.path.insert(0, str(ROOT / "src"))
from indicqe_ape.prompts import parse_qe_score_strict  # noqa: E402
sys.path.insert(0, str(ROOT / "analysis"))
from release_population import filter_cell  # noqa: E402

TEX = _PAPER_TEX
TPL = "gemba-da"
PAIRS = ["enhi", "enmr", "enta", "engu", "ente", "enml", "eten", "neen", "sien"]

#: (csv name, LaTeX macro, source dirs) -- prompted models, weakest first
PROMPTED = [
    ("BharatGPT-3B-Indic", r"\mBharatB", ["qe_rerun512"]),
    ("Llama-3.2-3B-Instruct", r"\mLlamaB", ["qe_rerun512"]),
    ("tiny-aya-fire", r"\mTiny", ["qe_rerun512"]),
    ("aya-expanse-32b", r"\mAyaB", ["qe_rerun512"]),
    ("sarvam-m", r"\mSarvam", ["qe_rerun512"]),
    ("gpt-5.5", r"\mGPT", ["qe_closed_rerun", "qe_closed"]),
]
#: (csv name, LaTeX macro) -- trained metrics, read from qe_meta_s0.csv
TRAINED = [("cometkiwi-da", r"\mCKDA"), ("cometkiwi-xl", r"\mCKXL"), ("xcomet-xl", r"\mXC")]

SUMMARY_REQUIRED = {"qe_rerun512", "qe_closed_rerun"}


def load(model: str, dirs: list[str]) -> dict[str, pd.DataFrame]:
    """pair -> DataFrame(gold, pred) at zero shots, strict parser."""
    out: dict[str, pd.DataFrame] = {}
    for d in dirs:
        for f in glob.glob(str(ROOT / "results" / d / f"{model}.s0.*.jsonl")):
            pair = Path(f).stem.split(".")[-1]
            if pair in out:
                continue                     # first dir wins, so a rerun overrides the original
            rows, complete = [], False
            for line in open(f, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "_summary" in r:
                    complete = True
                    continue
                if "_meta" in r:
                    continue
                raw = (r.get("raw") or {}).get(TPL)
                rows.append((r.get("gold_da_mean"),
                             parse_qe_score_strict(raw) if raw is not None
                             else (r.get("pred") or {}).get(TPL),
                             r.get("id")))
            if rows and not complete and d in SUMMARY_REQUIRED:
                raise SystemExit(
                    f"PARTIAL CELL: {Path(f).name} has {len(rows)} rows and no _summary. The run "
                    f"was interrupted; do not report a macro computed over it.")
            if rows:
                # A run file can predate the current challenge test. Scoring a segment that is no
                # longer in the release would report a correlation over data the benchmark does not
                # contain, so the cell is filtered to the release and the loss is printed.
                cell = pd.DataFrame(rows, columns=["gold", "pred", "id"])
                kept = filter_cell(cell, "id")
                if len(kept) != len(cell):
                    print(f"    {model}.s0.{pair}: {len(cell) - len(kept)} of {len(cell)} rows are "
                          f"not in the released challenge test and are dropped")
                out[pair] = kept
    return out


def stats_for(cells: dict[str, pd.DataFrame]) -> tuple[float, float, float, float, int]:
    """macro Pearson / Spearman / Kendall, cross-lingual Pearson, and the scored n."""
    P, R, K, mm, hh, n = [], [], [], [], [], 0
    for _, df in sorted(cells.items()):
        d = df.dropna(subset=["gold", "pred"])
        if len(d) < 11 or d.pred.std() == 0 or d.gold.std() == 0:
            continue
        P.append(stats.pearsonr(d.pred, d.gold).statistic)
        R.append(stats.spearmanr(d.pred, d.gold).statistic)
        K.append(stats.kendalltau(d.pred, d.gold).statistic)
        mm.append(d.pred.mean())
        hh.append(d.gold.mean())
        n += len(d)
    x = stats.pearsonr(mm, hh).statistic if len(mm) > 2 else np.nan
    return float(np.mean(P)), float(np.mean(R)), float(np.mean(K)), float(x), n


def per_pair_kendall(cells: dict[str, pd.DataFrame]) -> dict[str, float]:
    out = {}
    for p, df in cells.items():
        d = df.dropna(subset=["gold", "pred"])
        if len(d) < 11 or d.pred.std() == 0 or d.gold.std() == 0:
            continue
        out[p] = float(stats.kendalltau(d.pred, d.gold).statistic)
    return out


def splice(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(% BEGIN GENERATED {re.escape(name)}[^\n]*\n).*?"
                     rf"(% END GENERATED {re.escape(name)})", re.S)
    if not pat.search(text):
        raise SystemExit(f"sentinels for {name} not found in {TEX.name}")
    return pat.sub(lambda m: m.group(1) + body + m.group(2), text)


def main() -> int:
    meta = pd.read_csv(ROOT / "results/analysis/qe_meta_s0.csv")
    meta = meta[meta.shots == 0].set_index("system")

    rows, kend, scored = [], {}, {}
    for name, mac, dirs in PROMPTED:
        cells = load(name, dirs)
        if len(cells) != 9:
            raise SystemExit(f"{name}: {len(cells)} pairs, expected 9")
        P, R, K, X, n = stats_for(cells)
        rows.append((mac, P, R, K, X, False))
        kend[mac] = per_pair_kendall(cells)
        scored[mac] = n
        print(f"  {name:<24} P {P:.3f}  rho {R:.3f}  tau {K:.3f}  xling {X:+.3f}  n={n:,}")

    for name, mac in TRAINED:
        L = meta.loc[name]
        rows.append((mac, L.macro_pearson, L.macro_spearman, L.macro_kendall, L.syslvl_pearson, True))
        kend[mac] = {p: L[f"kendall_{p}"] for p in PAIRS}
        scored[mac] = 12730
        print(f"  {name:<24} P {L.macro_pearson:.3f}  rho {L.macro_spearman:.3f}  "
              f"tau {L.macro_kendall:.3f}  xling {L.syslvl_pearson:+.3f}  n=12,730")

    # ---- tab:qe -------------------------------------------------------------------------------
    best = {c: max(r[i] for r in rows) for i, c in ((1, "P"), (2, "R"), (3, "K"))}
    L = []
    for mac, P, R, K, X, trained in rows:
        if mac == r"\mCKDA":
            L.append("\\midrule\n")
        def f(v, key):
            s = f"{v:.3f}"
            return f"\\best{{${s}$}}" if abs(v - best[key]) < 1e-9 else f"${s}$"
        L.append(f"{mac} & {f(P,'P')} & {f(R,'R')} & {f(K,'K')} & ${X:.2f}$ \\\\\n")

    # ---- tab:perpairfull ----------------------------------------------------------------------
    colbest = {p: max(kend[m].get(p, -9) for m in kend) for p in PAIRS}
    macbest = max(np.mean(list(kend[m].values())) for m in kend)
    Q = []
    for mac, *_ in rows:
        if mac == r"\mCKDA":
            Q.append("\\midrule\n")
        cells = []
        for p in PAIRS:
            v = kend[mac].get(p)
            if v is None:
                cells.append("---")
            else:
                s = f"{v:.3f}" if v >= 0 else f"$-{abs(v):.3f}$"
                cells.append(f"\\best{{{s}}}" if abs(v - colbest[p]) < 1e-9 else s)
        m = float(np.mean(list(kend[mac].values())))
        ms = f"{m:.3f}" if m >= 0 else f"$-{abs(m):.3f}$"
        Q.append(f"{mac} & " + " & ".join(cells) + " & "
                 + (f"\\best{{{ms}}}" if abs(m - macbest) < 1e-9 else ms) + " \\\\\n")
    Q.append("\\midrule\n")
    means = [float(np.mean([kend[m][p] for m in kend if p in kend[m]])) for p in PAIRS]
    lo = min(means)
    Q.append("mean & " + " & ".join(
        (f"\\best{{{v:.3f}}}" if abs(v - lo) < 1e-9 else f"{v:.3f}") for v in means) + " & \\\\\n")

    print(f"\n  scored populations: {min(scored.values()):,} to {max(scored.values()):,}")

    tex = TEX.read_text()
    want = splice(splice(tex, "tab_qe_body", "".join(L)), "tab_perpair_body", "".join(Q))
    if want == tex:
        print("acl_latex.tex is up to date")
        return 0
    if "--check" in sys.argv:
        print("OUT OF DATE: run without --check to update.")
        return 1
    TEX.write_text(want)
    print(f"spliced tab:qe and tab:perpairfull into {TEX.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
