#!/usr/bin/env python3
"""model_language_coverage.py — does a model's declared language list predict how it behaves?

Were the prompted models trained on all nine of the benchmark's languages, and does that explain
the zero-shot against four-shot gap? The question is answerable from the model cards, so this
answers it rather than speculating.

DECLARED holds the `language:` metadata field of each model card, verbatim, with the revision date
we read it on. A language pair counts as *declared* for a model when BOTH its languages are in that
list; English is in every list, so for en->X it is the target that decides and for X->en the source.

Two things are then tested against the declared/undeclared split:

  compliance   the share of segments returning a single parsable GEMBA-DA score, four shots
  correlation  per-pair Spearman against human DA, zero shots

The finding is narrow and is reported as such: the split lands exactly on tiny-aya-fire's two
four-shot compliance failures and predicts nothing for the other five models.

    python3 analysis/model_language_coverage.py           # write the CSV and print the report
    python3 analysis/model_language_coverage.py --check   # recompute and diff against the CSV

Consumed by Appendix M (tab:models and the coverage paragraph).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import make_prompting_tables as P  # noqa: E402
import make_qe_tables as Q  # noqa: E402

OUT = ROOT / "results" / "analysis" / "model_language_coverage.csv"

#: ISO 639-1 codes of the two sides of each directional pair, in the paper's pair order
PAIR_LANGS = {
    "enhi": ("en", "hi"), "enmr": ("en", "mr"), "enta": ("en", "ta"),
    "engu": ("en", "gu"), "ente": ("en", "te"), "enml": ("en", "ml"),
    "eten": ("et", "en"), "neen": ("ne", "en"), "sien": ("si", "en"),
}

#: The `language:` field of each model card, verbatim. Read 2026-09-04. A model absent from this
#: map has no published language list; gpt-5.5 is the only such model in the panel.
#:
#: Sources, all the model card's own metadata block:
#:   CoRover/BharatGPT-3B-Indic          hf.co/CoRover/BharatGPT-3B-Indic
#:   meta-llama/Llama-3.2-3B-Instruct    hf.co/meta-llama/Llama-3.2-3B-Instruct
#:   CohereLabs/tiny-aya-fire            hf.co/CohereLabs/tiny-aya-fire  (70 languages, and the
#:                                       Cohere docs page docs.cohere.com/docs/tiny-aya agrees)
#:   CohereLabs/aya-expanse-32b          hf.co/CohereLabs/aya-expanse-32b
#:   sarvamai/sarvam-m                   hf.co/sarvamai/sarvam-m
#:   sarvamai/sarvam-translate           hf.co/sarvamai/sarvam-translate
DECLARED = {
    "BharatGPT-3B-Indic": set("hi pa gu kn mr te ml or ta ur bn en".split()),
    "Llama-3.2-3B-Instruct": set("en de fr it pt hi es th".split()),
    "tiny-aya-fire": set(
        "en nl fr it pt ro es cs pl uk ru el de da sv no ca gl cy ga eu hr lv lt sk sl et fi hu "
        "sr bg ar fa ur tr mt he hi mr bn gu pa ta te ne tl ms id vi jv km th lo zh my ja ko am "
        "ha ig mg sn sw wo xh yo zu".split()),
    "aya-expanse-32b": set(
        "en fr de es it pt ja ko zh ar el fa pl id cs he hi nl ro ru tr uk vi".split()),
    "sarvam-m": set("en bn hi kn gu mr ml or pa ta te".split()),
    "sarvam-translate": set(
        "as bn brx doi gom gu en hi kn ks mai ml mni mr ne or pa sa sat sd ta te ur".split()),
}

#: the six prompted QE models, in the paper's table order
PANEL = [(name, dirs) for name, _macro, dirs in P.MODELS]


def declared(model: str, pair: str) -> bool | None:
    """True/False if the model publishes a language list, None if it does not."""
    langs = DECLARED.get(model)
    if langs is None:
        return None
    return all(l in langs for l in PAIR_LANGS[pair])


def main() -> int:
    missing = [d for _, dirs in PANEL for d in dirs if not (ROOT / "results" / d).is_dir()]
    # results/qe_closed_rerun is preferred but optional; only complain if no source dir survives
    missing = [d for d in missing if not any(
        (ROOT / "results" / alt).is_dir() for _, dirs in PANEL for alt in dirs if alt != d)]
    if not (ROOT / "results" / "qe_rerun512").is_dir():
        print("SKIP: results/qe_rerun512 is not in this checkout, so this check cannot run.\n"
              "  The per-instance predictions ship in the evidence bundle "
              "(analysis/export_evidence_bundle.py).")
        return 0
    absent = [m for m, dirs in PANEL if not any((ROOT / "results" / d).is_dir() for d in dirs)]
    if absent:
        print(f"SKIP: per-instance outputs for {', '.join(absent)} are not in this checkout, so a "
              "partial table would be compared against the full one.")
        return 0

    rows = []
    for model, dirs in PANEL:
        four = P.load(model, "4", dirs)
        zero = Q.load(model, dirs)
        for pair in Q.PAIRS:
            d4, d0 = four.get(pair), zero.get(pair)
            rows.append({
                "model": model,
                "pair": pair,
                "declared": declared(model, pair),
                "parse4": (round(float(d4.pred.notna().mean()), 3)
                           if d4 is not None and len(d4) else np.nan),
                "rho0": (round(P.rho(d0), 3) if d0 is not None and len(d0) else np.nan),
            })
    df = pd.DataFrame(rows)

    print("Declared language coverage against measured behaviour\n")
    print("A pair is 'declared' when both its languages appear in the model card's language "
          "field.\nparse4 is four-shot output-format compliance; rho0 is zero-shot Spearman.\n")
    for model, _ in PANEL:
        sub = df[df.model == model]
        if sub.declared.isna().all():
            print(f"{model:24s} no published language list")
            continue
        und = sorted(sub[~sub.declared.astype(bool)].pair)
        dec = sub[sub.declared.astype(bool)]
        undf = sub[~sub.declared.astype(bool)]
        print(f"{model:24s} undeclared pairs: {', '.join(und) if und else 'none'}")
        for col, label in (("parse4", "4-shot compliance"), ("rho0", "0-shot Spearman  ")):
            dm, um = dec[col].mean(), undf[col].mean()
            print(f"{'':24s}   {label}  declared {dm:.3f}   undeclared "
                  f"{um:.3f}" if not np.isnan(um) else
                  f"{'':24s}   {label}  declared {dm:.3f}   undeclared  ---")
        # Does the undeclared set coincide with the worst cells? Only a question worth asking
        # when both sides of the split have scored cells: for a model that declares one of our
        # languages, "the k worst are the undeclared ones" is true by arithmetic and says nothing.
        ranked = sub.dropna(subset=["parse4"]).sort_values("parse4")
        k = len(und)
        n_dec = int(dec.parse4.notna().sum())
        if not k or len(ranked) < k:
            pass
        elif n_dec == 0:
            print(f"{'':24s}   no scored declared pair, so the split is vacuous here")
        else:
            worst = set(ranked.head(k).pair)
            print(f"{'':24s}   the {k} worst compliance cells are "
                  f"{'exactly' if worst == set(und) else 'NOT'} the undeclared ones "
                  f"({', '.join(sorted(worst))})")

    if "--check" in sys.argv:
        if not OUT.exists():
            print(f"\nFAIL: {OUT.relative_to(ROOT)} is missing")
            return 1
        old = pd.read_csv(OUT)
        if not old.equals(df):
            print(f"\nFAIL: {OUT.relative_to(ROOT)} is out of date")
            return 1
        print(f"\n--check: {OUT.relative_to(ROOT)} is up to date")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
