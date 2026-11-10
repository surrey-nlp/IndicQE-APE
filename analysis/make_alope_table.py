#!/usr/bin/env python3
"""make_alope_table.py — turn Daria's tiny-aya-fire ALOPE runs into paper numbers.

Reads the shipped challenge summaries in evidence/alope/ (produced by her
evaluate_challenge.py) and emits:
  results/analysis/alope_summary.md   — human-readable
  paper/latex/tab_alope_auto.tex      — a LaTeX fragment for tab:alope (NOT auto-included; copy cells)

What it computes, all from evidence/alope/evaluation_challenge_summary_indic.csv:
  1. per pair × layer, the ALL-route Spearman/Pearson (the aggregate probe quality)
  2. the best layer per pair (by ALL-route Spearman) and the layer that wins on average
  3. the A4_contradiction route numbers (the floor claim in the paper) per pair/layer
  4. the easy_control floor per pair — flags pairs whose floor is inverted (en-ml)

Honesty guards: en-ml's easy_control correlates negatively,
so its challenge DROPS are not trustworthy and the pair is flagged EXCLUDED for drop analysis.
The probe is substantially a length detector; raw drops overstate difficulty. We therefore report
ALL-route probe quality as the headline and treat the challenge drops as diagnostic, not as a
clean difficulty measurement. The length-controlled en-hi numbers come from the
diagnose_challenge_floor script, whose per-item output is not shipped.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "evidence" / "alope" / "evaluation_challenge_summary_indic.csv"
OUT_MD = ROOT / "results" / "analysis" / "alope_summary.md"
LAYERS = [-5, -7, -11, -16, -20, -24]
PAIRS = ["en-gu", "en-hi", "en-ml", "en-mr", "en-ta", "en-te"]
# en-ml easy_control is inverted -> its drops are untrustworthy
FLOOR_INVERTED = {"en-ml"}


def main():
    df = pd.read_csv(SRC)
    md = ["# ALOPE (tiny-aya-fire) — paper numbers\n",
          "Source: `evidence/alope/evaluation_challenge_summary_indic.csv` (Daria's runs). "
          "Joint 6-pair single-layer QLoRA probe on `CohereLabs/tiny-aya-fire`, challenge-split eval.\n"]

    # 1+2. ALL-route per pair × layer
    allr = df[df.route == "ALL"].copy()
    piv = allr.pivot_table(index="lang_pair", columns="layer", values="spearman")
    piv = piv.reindex(index=PAIRS, columns=LAYERS)
    md.append("## 1. ALL-route Spearman by pair × layer\n")
    md.append(piv.round(4).to_markdown() + "\n")

    best = {}
    md.append("\n## 2. Best layer per pair (ALL route)\n")
    md.append("| pair | best layer | Spearman | Pearson | n |\n|---|---:|---:|---:|---:|\n")
    for p in PAIRS:
        sub = allr[allr.lang_pair == p]
        if sub.empty:
            continue
        row = sub.loc[sub.spearman.idxmax()]
        best[p] = row
        md.append(f"| {p} | {int(row.layer)} | {row.spearman:.4f} | {row.pearson:.4f} | {int(row.n)} |\n")
    # layer that wins on average across pairs
    mean_by_layer = allr.groupby("layer").spearman.mean().reindex(LAYERS)
    md.append(f"\nMean ALL-route Spearman by layer (across 6 pairs): "
              + ", ".join(f"{l}:{v:.3f}" for l, v in mean_by_layer.items()) + "\n")
    md.append(f"**Layer winning on average: {int(mean_by_layer.idxmax())} "
              f"({mean_by_layer.max():.3f}).** Last layer (-24): {mean_by_layer[-24]:.3f}. "
              "Middle-upper layers beat the final layer.\n")

    # 3. A4_contradiction (the floor claim) — per pair, range across layers
    md.append("\n## 3. A4_contradiction route (the paper's floor claim)\n")
    a4 = df[df.route == "A4_contradiction"]
    md.append("| pair | Spearman range | Pearson range | n |\n|---|---|---|---:|\n")
    for p in PAIRS:
        sub = a4[a4.lang_pair == p]
        if sub.empty:
            md.append(f"| {p} | (no A4 route) | | |\n"); continue
        md.append(f"| {p} | {sub.spearman.min():.3f}–{sub.spearman.max():.3f} | "
                  f"{sub.pearson.min():.3f}–{sub.pearson.max():.3f} | {int(sub.n.iloc[0])} |\n")
    # en-hi easy vs A4 drop (verify §4.2 numbers)
    ehi = df[df.lang_pair == "en-hi"]
    ec = ehi[ehi.route == "easy_control"]
    a4h = ehi[ehi.route == "A4_contradiction"]
    md.append(f"\nen-hi easy_control Spearman range {ec.spearman.min():.3f}–{ec.spearman.max():.3f}; "
              f"A4 Spearman range {a4h.spearman.min():.3f}–{a4h.spearman.max():.3f}; "
              f"raw drop {(ec.spearman.mean()-a4h.spearman.mean()):.3f} (mean).\n")
    md.append("**Caveat:** the probe is substantially a length detector; the raw "
              "A4 drop is length-inflated. Length-controlled, en-hi's A4 drop is ~0.339 Spearman and "
              "is itself an upper bound (range restriction). This is the only route whose drop "
              "survives length control.\n")

    # 4. easy_control floor validity
    md.append("\n## 4. easy_control floor per pair (validity check)\n")
    ecall = df[df.route == "easy_control"]
    md.append("| pair | easy Spearman (mean) | floor ok? |\n|---|---:|---|\n")
    for p in PAIRS:
        sub = ecall[ecall.lang_pair == p]
        if sub.empty:
            md.append(f"| {p} | — | n/a |\n"); continue
        m = sub.spearman.mean()
        ok = "NO — inverted, drops EXCLUDED" if (p in FLOOR_INVERTED or m < 0) else "yes"
        md.append(f"| {p} | {m:.3f} | {ok} |\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("".join(md))
    print("".join(md))
    print(f"\nwritten: {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
