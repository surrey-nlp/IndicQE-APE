#!/usr/bin/env python3
r"""template_compare.py — every GEMBA-DA vs GEMBA-SQM number in S3, recomputed from the runs.

S3 makes four claims about the prompt template. All four were written when gpt-5.5's four-shot run
covered eight pairs at 400 segments each, so the cell counts and the averages behind them are stale
now that the full nine-pair rerun has landed:

  1. "Averaged over pairs, GEMBA-DA and GEMBA-SQM differ by X in Spearman"
  2. "SQM is the better of the two on N of M system-by-pair cells"
  3. "BharatGPT-3B at zero shots scores A under DA against B under SQM, whereas for GPT-5.5 the two
     templates agree to within C"
  4. "SQM is the weaker of the two at zero shots for two models: aya-expanse-32B parses P (ne-en Q)
     and tiny-aya R (si-en S), against T and U under DA on the same population"

Every run file carries both templates in `raw`, so all of this is recoverable without rerunning
anything. Same strict parser and same directories as tab:prompting, so the populations match.

    python3 analysis/template_compare.py            # print the numbers and the sentences
    python3 analysis/template_compare.py --check    # fail if the tex disagrees with the runs
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
from make_prompting_tables import MIN_COVERAGE  # noqa: E402

TEX = _PAPER_TEX
OUT = ROOT / "results" / "analysis" / "template_compare.csv"
PAIRS = ["enhi", "enmr", "enta", "engu", "ente", "enml", "eten", "neen", "sien"]
TEMPLATES = ["gemba-da", "gemba-sqm"]
MODELS = [("BharatGPT-3B-Indic", ["qe_rerun512"]), ("Llama-3.2-3B-Instruct", ["qe_rerun512"]),
          ("tiny-aya-fire", ["qe_rerun512"]), ("aya-expanse-32b", ["qe_rerun512"]),
          ("sarvam-m", ["qe_rerun512"]), ("gpt-5.5", ["qe_closed_rerun", "qe_closed"])]


def load(model: str, shots: str, dirs: list[str]) -> pd.DataFrame:
    rows, seen = [], set()
    for d in dirs:
        for f in sorted(glob.glob(str(ROOT / "results" / d / f"{model}.s{shots}.*.jsonl"))):
            pair = Path(f).stem.split(".")[-1]
            if pair in seen:
                continue
            seen.add(pair)
            for line in open(f, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "_meta" in r or "_summary" in r:
                    continue
                raw, pred = r.get("raw") or {}, r.get("pred") or {}
                row = {"pair": pair, "id": r.get("id"), "gold": r.get("gold_da_mean")}
                for t in TEMPLATES:
                    row[t] = parse_qe_score_strict(raw[t]) if raw.get(t) is not None else pred.get(t)
                rows.append(row)
    return filter_cell(pd.DataFrame(rows), "id")


def rho(d: pd.DataFrame, col: str) -> float:
    """Spearman for one model-pair-template cell, or NaN if the cell is not a measurement.

    MIN_COVERAGE is imported from make_prompting_tables rather than restated, because this script
    and that one report the same quantity in adjacent sentences of S3 and a divergent floor would
    put two different conventions a paragraph apart. See the rationale there: a cell the model
    answered on fewer than half its segments describes which segments it chose to answer.
    """
    x = d.dropna(subset=["gold", col])
    if len(d) and len(x) / len(d) < MIN_COVERAGE:
        return float("nan")
    if len(x) < 11 or x[col].std() == 0 or x.gold.std() == 0:
        return float("nan")
    return float(stats.spearmanr(x[col], x.gold).statistic)


def main() -> int:
    check = "--check" in sys.argv
    cells = []
    for model, dirs in MODELS:
        for shots in ("0", "4"):
            df = load(model, shots, dirs)
            if df.empty:
                continue
            for pair, d in df.groupby("pair"):
                cells.append(dict(model=model, shots=int(shots), pair=pair, n=len(d),
                                  da=rho(d, "gemba-da"), sqm=rho(d, "gemba-sqm"),
                                  parse_da=d["gemba-da"].notna().mean(),
                                  parse_sqm=d["gemba-sqm"].notna().mean()))
    c = pd.DataFrame(cells)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c.to_csv(OUT, index=False)

    ok = c.dropna(subset=["da", "sqm"])
    mean_abs = float((ok.da - ok.sqm).abs().mean())
    sqm_better = int((ok.sqm > ok.da).sum())
    total = len(ok)
    print(f"cells with both templates scored: {total} "
          f"({c.model.nunique()} models x {c.pair.nunique()} pairs x 2 shot settings, "
          f"minus {c[['da', 'sqm']].isna().any(axis=1).sum()} unscored)")
    print(f"mean |rho_DA - rho_SQM| over cells: {mean_abs:.3f}")
    print(f"SQM higher than DA in {sqm_better} of {total} cells")

    z = c[c.shots == 0].groupby("model")[["da", "sqm"]].mean()
    z["gap"] = (z.da - z.sqm).abs()
    print("\nzero-shot macro by template:")
    print(z.round(3).to_string())

    p = c[c.shots == 0].set_index(["model", "pair"])
    print("\nzero-shot parse rate by template (macro, and the worst pair under SQM):")
    for m in c.model.unique():
        s = c[(c.shots == 0) & (c.model == m)]
        worst = s.loc[s.parse_sqm.idxmin()]
        print(f"  {m:24s} DA {s.parse_da.mean():.3f}  SQM {s.parse_sqm.mean():.3f}  "
              f"worst SQM pair {worst.pair} {worst.parse_sqm:.2f}")

    weak = [m for m in c.model.unique()
            if c[(c.shots == 0) & (c.model == m)].parse_sqm.mean() < 0.98]
    print(f"\nmodels whose zero-shot SQM parse rate is below 0.98: {weak}")

    # Like-for-like against the shot-count effect: both as a per-MODEL macro gap. The per-cell
    # figure above is a different quantity and is not comparable to S3's shot-count deltas.
    tmpl_gap = float(z.gap.mean())
    shots = c.pivot_table(index=["model", "pair"], columns="shots", values="da")
    shot_gap = float(shots.groupby("model").mean().pipe(lambda d: (d[4] - d[0]).abs()).mean())
    print(f"\nmacro gap per model: template {tmpl_gap:.3f}, shot count {shot_gap:.3f} "
          f"({'template matters less' if tmpl_gap < shot_gap else 'TEMPLATE MATTERS MORE'})")

    print("\n--- sentence, ready to paste ---")
    bg = z.loc["BharatGPT-3B-Indic"]
    gp = z.loc["gpt-5.5"]
    print(f"Averaged over pairs, GEMBA-DA and GEMBA-SQM differ by ${mean_abs:.3f}$ in Spearman, and "
          f"SQM is the better of the two on ${sqm_better}$ of ${total}$ system-by-pair cells. "
          f"... \\mBharatB{{}} at zero shots scores ${bg.da:.3f}$ under DA against ${bg.sqm:.3f}$ "
          f"under SQM, whereas for \\mGPT{{}} the two templates agree to within ${gp.gap:.3f}$.")

    if check:
        flat = " ".join(TEX.read_text().split())
        fails = []
        for want, what in [(f"${tmpl_gap:.3f}$ in macro Spearman", "mean template gap"),
                           # not "... system-by-pair cells": the paper now spells out what the
                           # cells are (6 models x 9 pairs x 2 shot counts, less the unscored), so
                           # pin the two figures and their adjacency, not the noun after them.
                           (f"${sqm_better}$ of the ${total}$ cells", "cell count"),
                           (f"${bg.da:.3f}$ under DA against ${bg.sqm:.3f}$", "BharatGPT gap")]:
            if want not in flat:
                fails.append(f"{what}: tex does not contain {want!r}")
        for f in fails:
            print(f"FAIL: {f}")
        return 1 if fails else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
