#!/usr/bin/env python3
"""make_split_table.py — train/validation/test sizes per released configuration.

The paper asserted split hygiene across train, validation and test but never sized the splits,
which is the table a benchmark user consults first. Read straight from the released parquet files
so the paper cannot disagree with the artefact.

Splices into acl_latex.tex between the tab_splits sentinels. --check exits 1 if out of date.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# The paper source lives in its own repository and is optional here; see
# analysis/paper_source.py for the search order and the env vars that override it.
sys.path.insert(0, str(ROOT / "analysis"))
from paper_source import find_tex as _find_tex  # noqa: E402
_PAPER_TEX = _find_tex() or (ROOT / "paper" / "latex" / "acl_latex.tex")

TEX = _PAPER_TEX
#: the task configs live under data/tasks/, the whole-corpus views one level up under data/
DATA = ROOT / "dataset" / "hf_snapshot" / "data"

#: config -> LaTeX label. Order matches how S3 introduces them.
CONFIGS = [
    ("full", r"\texttt{full}", "nine pairs, every instance"),
    ("all", r"\texttt{all}", r"en$\to$Indic only"),
    ("qe-da", r"\texttt{qe-da}", r"en$\to$Indic, DA-bearing"),
    ("ape", r"\texttt{ape}", "seven post-edited pairs"),
    ("word-qe", r"\texttt{word-qe}", "word-level OK/BAD tags"),
    ("explainable-qe", r"\texttt{explainable-qe}", "error explanations"),
    ("mqm", r"\texttt{mqm}", "en-hi MQM layer"),
    ("challenge", r"\texttt{challenge}", "curated test only"),
]


def sizes(cfg: str) -> dict[str, int]:
    for cand in (DATA / "tasks" / cfg, DATA / cfg):
        if list(cand.glob("*.parquet")):
            d = cand
            break
    else:
        raise SystemExit(f"no parquet files for config {cfg!r} under {DATA}")
    out = {}
    for sp in ("train", "validation", "test"):
        f = d / f"{sp}.parquet"
        out[sp] = len(pd.read_parquet(f)) if f.exists() else 0
    out["total"] = sum(out.values())
    return out


def splice(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(% BEGIN GENERATED {re.escape(name)}[^\n]*\n).*?"
                     rf"(% END GENERATED {re.escape(name)})", re.S)
    if not pat.search(text):
        raise SystemExit(f"sentinels for {name} not found")
    return pat.sub(lambda m: m.group(1) + body + m.group(2), text)


def main() -> int:
    rows, L = [], []
    for cfg, mac, note in CONFIGS:
        s = sizes(cfg)
        rows.append({"config": cfg, **s})
        f = lambda n: f"{n:,}".replace(",", "{,}")
        tr = "---" if s["train"] == 0 else f"${f(s['train'])}$"
        va = "---" if s["validation"] == 0 else f"${f(s['validation'])}$"
        L.append(f"{mac} & {tr} & {va} & ${f(s['test'])}$ & ${f(s['total'])}$ & {note} \\\\\n")
        print(f"  {cfg:<16} train {s['train']:>7}  val {s['validation']:>5}  "
              f"test {s['test']:>6}  total {s['total']:>7}")

    df = pd.DataFrame(rows)
    assert int(df[df.config == "full"].total.iloc[0]) == 126754, "full config is not 126,754"
    df.to_csv(ROOT / "results/analysis/split_sizes.csv", index=False)

    tex = TEX.read_text()
    want = splice(tex, "tab_splits", "".join(L))
    if want == tex:
        print("\nacl_latex.tex is up to date")
        return 0
    if "--check" in sys.argv:
        print("\nOUT OF DATE")
        return 1
    TEX.write_text(want)
    print("\n" + "".join(L) + f"spliced into {TEX.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
