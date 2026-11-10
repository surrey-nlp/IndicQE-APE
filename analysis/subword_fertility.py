#!/usr/bin/env python3
"""subword_fertility.py — how many subwords a word costs, per language, in the shared vocabulary.

What does byte-pair/sentencepiece segmentation do to the languages here? Appendix B answers the *word* half of that question (which tokeniser the released `mt_tokens`
index). This answers the subword half, and it is answerable exactly for the models that matter:
CometKiwi-DA, CometKiwi-XL, XCOMET-XL and the COMET-encoder head of S5 all read the same
XLM-R SentencePiece vocabulary, so one measurement covers every trained metric in the paper.

Fertility is subwords per whitespace token, measured on the target side of the challenge test:

    fertility(pair) = sum over segments of |sentencepiece(mt)| / |mt.split()|

The prompted models are deliberately not measured. They carry six different vocabularies, four of
them behind a gated repository, and a fertility computed from a tokeniser we cannot pin to the
served checkpoint would not be evidence. What can be said about them is said from their model
cards instead (analysis/model_language_coverage.py).

    python3 analysis/subword_fertility.py           # write the CSV and print the report
    python3 analysis/subword_fertility.py --check   # recompute and diff against the CSV

Consumed by Appendix B.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "dataset" / "hf_snapshot" / "data" / "challenge" / "test.parquet"
OUT = ROOT / "results" / "analysis" / "subword_fertility.csv"
#: per-segment confusion counts behind the word-tag validation of Appendix B, so the fertility
#: order can be checked against the derived-tag agreement order without trusting a pasted number
MCC = ROOT / "results" / "analysis" / "wordqe_mcc_perinstance.csv"

#: the encoder every trained metric in this paper is built on
TOKENISER = "FacebookAI/xlm-roberta-large"

#: paper order
PAIRS = ["en-hi", "en-mr", "en-ta", "en-gu", "en-te", "en-ml", "et-en", "ne-en", "si-en"]


def main() -> int:
    if not SNAP.exists():
        print(f"SKIP: {SNAP.relative_to(ROOT)} is not in this checkout, so this check cannot "
              "run.\n  Rebuild it with dataset/make_hf_repo.py, or pull the release from the Hub.")
        return 0
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("SKIP: transformers is not installed, so this check cannot run.\n"
              "  pip install transformers sentencepiece")
        return 0

    tok = AutoTokenizer.from_pretrained(TOKENISER)
    df = pd.read_parquet(SNAP, columns=["language_pair", "source", "mt"])

    rows = []
    for pair in PAIRS:
        sub = df[df.language_pair == pair]
        # target side: the string a word-level tag layer would have to index
        words = sum(len(str(s).split()) for s in sub.mt)
        pieces = sum(len(tok.tokenize(str(s))) for s in sub.mt)
        # source side, for the direction contrast
        swords = sum(len(str(s).split()) for s in sub.source)
        spieces = sum(len(tok.tokenize(str(s))) for s in sub.source)
        rows.append({
            "pair": pair,
            "n": len(sub),
            "tgt_words": words,
            "tgt_fertility": round(pieces / words, 2),
            "src_fertility": round(spieces / swords, 2),
        })
    out = pd.DataFrame(rows)

    print(f"Subword fertility under {TOKENISER}, challenge test\n")
    print("subwords per whitespace token; the target side is what a word-tag layer must index\n")
    print(out.to_string(index=False))
    hi = out.loc[out.tgt_fertility.idxmax()]
    lo = out.loc[out.tgt_fertility.idxmin()]
    print(f"\nhighest: {hi.pair} at {hi.tgt_fertility}   lowest: {lo.pair} at {lo.tgt_fertility}"
          f"   ratio {hi.tgt_fertility / lo.tgt_fertility:.1f}x")
    eni = out[out.pair.str.startswith("en-")]
    xen = out[out.pair.str.endswith("-en")]
    print(f"en->Indic targets mean {eni.tgt_fertility.mean():.2f}, "
          f"X->en (English) targets mean {xen.tgt_fertility.mean():.2f}")

    # Appendix B reads the derived-tag agreement order as morphological complexity. Fertility is
    # an independent measure of the same thing, so the two orders should broadly agree.
    if MCC.exists():
        import numpy as np
        from scipy.stats import spearmanr
        # float64, not the summed int64: the four-way product in the denominator reaches 3.5e19
        # on et-en, overflows a signed 64-bit integer, goes negative, and silently yields NaN.
        c = (pd.read_csv(MCC).groupby("language_pair")[["tp", "tn", "fp", "fn"]]
             .sum().astype("float64"))
        num = c.tp * c.tn - c.fp * c.fn
        den = np.sqrt((c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn))
        c["mcc"] = (num / den).round(3)
        j = out.set_index("pair").join(c[["mcc"]], how="inner").dropna(subset=["mcc"])
        r = spearmanr(j.tgt_fertility, j.mcc)
        print(f"\nfertility against derived-tag MCC over the {len(j)} pairs with real tags: "
              f"Spearman {r.statistic:+.3f} (p={r.pvalue:.3f})")
        print(j[["tgt_fertility", "mcc"]].sort_values("tgt_fertility").to_string())

    if "--check" in sys.argv:
        if not OUT.exists():
            print(f"\nFAIL: {OUT.relative_to(ROOT)} is missing")
            return 1
        if not pd.read_csv(OUT).equals(out):
            print(f"\nFAIL: {OUT.relative_to(ROOT)} is out of date")
            return 1
        print(f"\n--check: {OUT.relative_to(ROOT)} is up to date")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
