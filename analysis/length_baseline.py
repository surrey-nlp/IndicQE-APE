#!/usr/bin/env python3
r"""length_baseline.py — how much of the layer probe is a length detector?

S4 reports that the probe's prediction correlates with input length at -0.26 to -0.78 and then
length-controls only the route drops, never the headline per-pair correlation. A reader cannot tell
whether reading DA off a frozen model's hidden state beats counting tokens. This computes the
missing control: the correlation between human DA and length alone, per pair, on the same
challenge split the probe is evaluated on.

Three length features, all trivially available at inference time:

  source tokens   whitespace tokens of the English source
  MT tokens       shipped mt_tokens where present, else whitespace tokens of the MT
  length ratio    MT tokens over source tokens, the feature the COMET head also uses

Reported against the probe's per-pair rho from tab:alope. If length alone approaches the probe, the
probe is not established as a quality signal; if it does not, the probe survives its own caveat.

    python3 analysis/length_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
# The paper source lives in its own repository and is optional here; see
# analysis/paper_source.py for the search order and the env vars that override it.
sys.path.insert(0, str(ROOT / "analysis"))
from paper_source import find_tex as _find_tex  # noqa: E402
_PAPER_TEX = _find_tex() or (ROOT / "paper" / "latex" / "acl_latex.tex")

CHALLENGE = ROOT / "dataset" / "hf_snapshot" / "data" / "challenge" / "test.parquet"
OUT = ROOT / "results" / "analysis" / "length_baseline.csv"

#: tab:alope, best layer per pair (single-layer probe over frozen tiny-aya)
PROBE = {"en-gu": 0.662, "en-hi": 0.449, "en-ml": 0.515, "en-mr": 0.666, "en-ta": 0.662,
         "en-te": 0.282}


def main() -> int:
    df = pd.read_parquet(CHALLENGE)
    df = df[df.has_da & df.language_pair.isin(PROBE)].copy()
    df["src_len"] = df.source.str.split().str.len()
    df["mt_len"] = [len(t) if isinstance(t, (list, np.ndarray)) and len(t) else
                    (len(str(m).split()) if isinstance(m, str) else np.nan)
                    for t, m in zip(df.mt_tokens, df.mt)]
    df["ratio"] = df.mt_len / df.src_len

    rows = []
    for pair, d in df.groupby("language_pair"):
        d = d.dropna(subset=["src_len", "mt_len", "ratio", "da_mean"])
        r = {"pair": pair, "n": len(d)}
        for f in ("src_len", "mt_len", "ratio"):
            r[f] = round(float(spearmanr(d[f], d.da_mean).statistic), 3)
        r["best_len"] = round(max(abs(r["src_len"]), abs(r["mt_len"]), abs(r["ratio"])), 3)
        r["probe"] = PROBE[pair]
        r["margin"] = round(r["probe"] - r["best_len"], 3)
        rows.append(r)
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"\nmacro: best length feature {out.best_len.mean():.3f}, probe {out.probe.mean():.3f}, "
          f"margin {out.margin.mean():+.3f}")
    worst = out.loc[out.margin.idxmin()]
    print(f"smallest margin: {worst.pair} probe {worst.probe} against length {worst.best_len} "
          f"({worst.margin:+.3f})")
    print(f"pairs where length alone beats the probe: "
          f"{out[out.margin < 0].pair.tolist() or 'none'}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    if "--check" in sys.argv:
        flat = " ".join((_PAPER_TEX).read_text().split())
        fails = [f"tex does not contain ${v:.3f}$" for v in (out.best_len.mean(), out.probe.mean())
                 if f"${v:.3f}$" not in flat]
        if (out.margin < 0).any() and "beats the probe on no pair" in flat:
            fails.append("the tex says length beats the probe on no pair, but it does")
        for f in fails:
            print(f"FAIL: {f}")
        return 1 if fails else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
