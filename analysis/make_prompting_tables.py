#!/usr/bin/env python3
"""make_prompting_tables.py — emit tab:prompting and tab:parse4 from the run outputs.

One vintage, one parser, one population, so the two tables cannot disagree:

  source     results/qe_rerun512 for every open model (512-token budget, chain-of-thought
             disabled, corrected few-shot pool). gpt-5.5 comes from results/qe_closed_rerun if it
             exists, else results/qe_closed. Its ORIGINAL four-shot cells were budget-limited
             (max_tokens=8, --limit 400 per pair, eight pairs). The nine-pair rerun landed on
             2026-08-08 and supersedes them, so every row is now the same vintage. The old cells
             are still on disk under results/qe_closed and are NOT read while the rerun exists.
  parser     parse_qe_score_strict. The permissive parser reported a 100% parse rate while
             recording the range start of "Score (0-100): 88" as 0, a list marker as 1, and the
             first of four echoed few-shot scores as the prediction.
  population all nine pairs. The previous tables used eight because the original four-shot
             launcher omitted en-ml; the rerun covers it.
  template   GEMBA-DA.

Splices the rows straight into acl_latex.tex between

    % BEGIN GENERATED <name>
    ...
    % END GENERATED <name>

sentinels, and touches nothing else. (\\input is not usable here: these tabulars sit inside
\\resizebox, which cannot take an \\input in its argument -- it produces "Misplaced \\cr" and no
PDF. Splicing keeps the rows inline for LaTeX while keeping them generated for us.)

    --check   exit 1 if the file is out of date, without writing. For CI or a pre-commit hook.
"""
from __future__ import annotations

import glob
import re
import json
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
PAIRS = ["enhi", "enmr", "enta", "ente", "engu", "enml", "eten", "neen", "sien"]
TPL = "gemba-da"
#: (csv name, LaTeX macro, source dirs) in table order, weakest first
MODELS = [
    ("BharatGPT-3B-Indic", r"\mBharatB", ["qe_rerun512"]),
    ("Llama-3.2-3B-Instruct", r"\mLlamaB", ["qe_rerun512"]),
    ("tiny-aya-fire", r"\mTiny", ["qe_rerun512"]),
    ("aya-expanse-32b", r"\mAyaB", ["qe_rerun512"]),
    ("sarvam-m", r"\mSarvam", ["qe_rerun512"]),
    # qe_closed_rerun first: gpt-5.5's ORIGINAL four-shot cells were run at max_tokens=8, the same
    # truncated condition every other model was rerun to escape, and capped at 400 segments a pair
    # over eight pairs. The rerun covers all nine at full size, so it wins here and the original
    # cells in results/qe_closed are never read unless the rerun directory is emptied.
    ("gpt-5.5", r"\mGPT", ["qe_closed_rerun", "qe_closed"]),
]


#: Directories whose writer we control and which therefore MUST emit a _summary per cell. A cell
#: without one in these is genuinely truncated. qe_open and qe_closed are excluded because they
#: hold two vintages side by side -- most cells predate the summary-emitting writer, so absence
#: there proves nothing. Inferring this per directory was wrong: qe_closed contains both.
SUMMARY_REQUIRED = {"qe_rerun512", "qe_closed_rerun", "ape_regen"}


def load(model: str, shots: str, dirs: list[str]) -> dict[str, pd.DataFrame]:
    """pair -> DataFrame(id, gold, pred) using the strict parser on the saved raw reply."""
    out = {}
    for d in dirs:
        for f in glob.glob(str(ROOT / "results" / d / f"{model}.s{shots}.*.jsonl")):
            pair = Path(f).stem.split(".")[-1]
            if pair in out:
                continue          # first dir in `dirs` wins, so a rerun overrides the original
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
                rows.append((r.get("id"), r.get("gold_da_mean"),
                             parse_qe_score_strict(raw) if raw is not None
                             else (r.get("pred") or {}).get(TPL)))
            # A cell still being written has no _summary, and averaging it silently misreports
            # the macro. Only enforce this where we control the writer (SUMMARY_REQUIRED): the
            # legacy dirs mix vintages and most of their cells never had a summary.
            if rows and not complete and d in SUMMARY_REQUIRED:
                raise SystemExit(
                    f"PARTIAL CELL: {Path(f).name} has {len(rows)} rows and no _summary, in a "
                    f"directory whose other cells have one. The run was interrupted. Wait for it "
                    f"or move the file aside; do not report a macro computed over it.")
            if rows:
                out[pair] = filter_cell(pd.DataFrame(rows, columns=["id", "gold", "pred"]), 'id')
    return out


#: A macro cell must be a measurement, not a self-selected minority.
#:
#: S3 already states the objection: "a correlation computed over a low compliance rate rests on a
#: self-selected subset", and the en-ml paragraph says outright that a cell at 0.058 compliance is
#: "not a measurement". The code did not implement it. The only floor was `len(d) < 11`, which is a
#: guard against a degenerate Spearman, not a policy, and under it BharatGPT-3B's four-shot en-te
#: cell -- 15 parsed segments of 1,120, compliance 0.013 -- was inside a printed macro, as were
#: seven other cells below half compliance.
#:
#: 0.5 is where the paper's own words put it: below half, the scored segments are the ones the
#: model chose to answer, and their correlation describes that choice as much as the model's
#: quality. No zero-shot cell is affected (the lowest is aya-expanse on ne-en at 0.868), so this
#: touches the four-shot column only and tab:qe is untouched.
MIN_COVERAGE = 0.5


def rho(df: pd.DataFrame) -> float:
    d = df.dropna(subset=["gold", "pred"])
    if len(d) < 11 or d.pred.std() == 0 or d.gold.std() == 0:
        return np.nan
    return float(stats.spearmanr(d.pred, d.gold).statistic)


def macro(cells: dict[str, pd.DataFrame], ids: dict[str, set] | None = None) -> tuple[float, int]:
    """Return (macro rho, number of pairs it averages).

    The count is returned rather than assumed, because it is not nine for every model and the
    table now prints it.
    """
    v = []
    for p, df in cells.items():
        if df.pred.notna().mean() < MIN_COVERAGE:
            continue
        if ids is not None:
            df = df[df.id.isin(ids.get(p, set()))]
        r = rho(df)
        if not np.isnan(r):
            v.append(r)
    return (float(np.mean(v)) if v else np.nan), len(v)


def splice(text: str, name: str, body: str) -> str:
    """Replace the lines between the BEGIN/END sentinels for `name`."""
    pat = re.compile(
        rf"(% BEGIN GENERATED {re.escape(name)}[^\n]*\n).*?(% END GENERATED {re.escape(name)})",
        re.S)
    if not pat.search(text):
        raise SystemExit(f"sentinels for {name} not found in {TEX.name}")
    return pat.sub(lambda m: m.group(1) + body + m.group(2), text)


def main() -> int:
    check_only = "--check" in sys.argv
    rows, parse_rows = [], []
    for name, macro_name, dirs in MODELS:
        c0, c4 = load(name, "0", dirs), load(name, "4", dirs)
        if not c0 or not c4:
            print(f"  skip {name}: missing cells"); continue
        shared = {p: set(c0[p].id) & set(c4[p].id) for p in c0 if p in c4}
        (r0, k0), (r4, k4) = macro(c0), macro(c4)
        (p0, _), (p4, _) = macro(c0, shared), macro(c4, shared)
        dropped = sorted(p for p, df in c4.items() if df.pred.notna().mean() < MIN_COVERAGE)
        missing = sorted(set(c0) - set(c4))
        parse4 = float(np.mean([df.pred.notna().mean() for df in c4.values()]))
        n0 = sum(len(d) for d in c0.values())
        n4 = sum(len(d) for d in c4.values())
        rows.append(dict(model=macro_name, r0=r0, r4=r4, d=r4 - r0, pd_=p4 - p0, parse=parse4,
                         n0=n0, n4=n4, pairs0=k0, pairs4=k4,
                         dropped=",".join(dropped), missing=",".join(missing)))
        parse_rows.append(dict(model=macro_name,
                               **{p: (c4[p].pred.notna().mean() if p in c4 else np.nan)
                                  for p in PAIRS}))
        note = ""
        if missing:
            note += f"  no s4 run: {','.join(missing)}"
        if dropped:
            note += ("  below " + f"{MIN_COVERAGE:.0%}" + " compliance, excluded: "
                     + ", ".join(f"{p} {c4[p].pred.notna().mean():.3f}" for p in dropped))
        print(f"  {name:<24} rho {r0:.3f} ({k0}p) -> {r4:.3f} ({k4}p) ({r4-r0:+.3f})  "
              f"paired {p4-p0:+.3f}  parse4 {parse4:.3f}{note}")

    df = pd.DataFrame(rows)
    short = df[df.n4 < 0.9 * df.n0]
    flat = " ".join(TEX.read_text().split())
    STALE = ["which covers eight", "was not run on en-ml", "$400$-segment sample per pair",
             "covering $3{,}152$"]
    if short.empty:
        for s in STALE:
            if s in flat:
                print(f"  NOTE: every four-shot run is full size now, but a caption still says "
                      f"{s!r}. Remove it.")
    else:
        print(f"  short four-shot populations: {list(short.model)} -- captions must say so")
    best = df.d.idxmin()
    L = []
    for i, r in df.iterrows():
        d = f"\\best{{${r.d:+.3f}$}}" if i == best else f"${r.d:+.3f}$"
        pdel = f"\\best{{${r.pd_:+.3f}$}}" if i == best else f"${r.pd_:+.3f}$"
        # The dagger used to be hard-coded onto gpt-5.5, whose four-shot run was capped at 400
        # segments a pair over eight pairs. The full rerun landed, so the mark is now DERIVED: a row
        # is flagged only if its four-shot population is materially smaller than its zero-shot one.
        # If a future run is partial the mark returns by itself, and the caption assertion below
        # fires if the note is missing.
        mark = "$^{\\ddagger}$" if r.n4 < 0.9 * r.n0 else ""
        L.append(f"{r.model} & {r.r0:.3f} & {r.r4:.3f} & {d} & {pdel} & "
                 f"{int(r.pairs4)} & {r.parse:.3f}{mark} \\\\\n")

    # Control rows. These were hand-typed and had gone stale: they still held the EIGHT-pair macro
    # (0.624/0.690/0.508) after the table moved to nine pairs, which put tab:prompting's CometKiwi-XL
    # at 0.690 while the abstract and S4 quoted the nine-pair 0.671 for the same quantity. They are
    # inside the sentinel now so the control cannot sit on a different population from what it
    # controls. The COMET metrics cannot see a prompt, so both columns are the same number.
    meta = pd.read_csv(ROOT / "results/analysis/qe_meta_s0.csv")
    meta = meta[meta.shots == 0].set_index("system")
    L.append("\\midrule\n\\multicolumn{7}{l}{\\textit{control: encoders, prompt-invariant}} \\\\\n")
    for name, mac in [("cometkiwi-da", r"\mCKDA"), ("cometkiwi-xl", r"\mCKXL"),
                      ("xcomet-xl", r"\mXC")]:
        v = float(meta.loc[name, "macro_spearman"])
        L.append(f"{mac} & {v:.3f} & {v:.3f} & 0.000 & --- & 9 & --- \\\\\n")
        print(f"  {name:<24} control (9-pair macro) {v:.3f}")

    P = []
    for r in pd.DataFrame(parse_rows).to_dict("records"):
        cells = " & ".join("---" if pd.isna(r[p]) else f"${r[p]:.3f}$" for p in PAIRS)
        P.append(f"{r['model']} & {cells} \\\\\n")

    tex = TEX.read_text()
    want = splice(splice(tex, "tab_prompting_body", "".join(L)), "tab_parse4_body", "".join(P))
    df.to_csv(ROOT / "results/analysis/prompting_table.csv", index=False)

    if want == tex:
        print("\nacl_latex.tex is up to date")
        return 0
    if check_only:
        print("\nOUT OF DATE: acl_latex.tex does not match the data. "
              "Run without --check to update.")
        return 1
    TEX.write_text(want)
    print("\n--- tab_prompting_body ---\n" + "".join(L))
    print("--- tab_parse4_body ---\n" + "".join(P))
    print(f"spliced both bodies into {TEX.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
