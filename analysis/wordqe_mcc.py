#!/usr/bin/env python3
"""wordqe_mcc.py — word-tag validation with MCC, the WMT word-level metric.

The G1 gate in scripts/derive_word_tags.py reports token accuracy and BAD-F1.
WMT's word-level QE task scores with Matthews correlation coefficient, because
accuracy is inflated by the OK majority class and F1 ignores true negatives.
This recomputes the same validation with MCC added, and saves PER-INSTANCE rows
so any downstream table can be rebuilt without re-running.

Reads local parquet (no network). Two validation populations:
  A. en->Indic gold      — tags shipped by the shared tasks
  B. MLQE-PE real labels — tags imported from the X->en upstream

Outputs
  results/analysis/wordqe_mcc.md                summary table
  results/analysis/wordqe_mcc_perinstance.csv   one row per segment
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from derive_word_tags import align_tags, as_list, is_placeholder, tok  # noqa: E402

BUILD = ROOT / "dataset" / "nullfill_build"
OUT_MD = ROOT / "results" / "analysis" / "wordqe_mcc.md"
OUT_CSV = ROOT / "results" / "analysis" / "wordqe_mcc_perinstance.csv"


def mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """Matthews correlation coefficient from a 2x2 count table."""
    num = (tp * tn) - (fp * fn)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float(num / den) if den > 0 else 0.0


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def load_full() -> pd.DataFrame:
    return pd.concat(
        [pd.read_parquet(BUILD / f"full_{s}.parquet") for s in ("train", "validation", "test")],
        ignore_index=True,
    )


def evaluate(d: pd.DataFrame) -> pd.DataFrame:
    """Re-derive tags for every row that already carries a real tag; score per segment.

    Returns one row per evaluated segment with its own confusion counts, so the
    corpus figure is a sum over saved per-instance numbers rather than an
    unverifiable aggregate.
    """
    w = d[d.wtag.notna() & d.post_edit.notna()].copy()
    w = w[~w.post_edit.map(is_placeholder)]
    print(f"candidate rows with a real tag and a usable post-edit: {len(w)}")

    recs = []
    skipped = {"no_mt_tokens": 0, "length_mismatch": 0}
    for _, r in w.iterrows():
        gold = r["wtag"].split() if isinstance(r["wtag"], str) else list(r["wtag"])
        mtt = as_list(r["mt_tokens"])
        if not mtt:
            skipped["no_mt_tokens"] += 1
            continue
        if str(mtt[-1]) == "<EOS>":
            mtt, gold = mtt[:-1], gold[:-1]
        if len(gold) != len(mtt):
            skipped["length_mismatch"] += 1
            continue

        pred = align_tags([str(t) for t in mtt], tok(r["post_edit"]))
        gold = [str(g).upper() for g in gold]

        tp = sum(1 for g, p in zip(gold, pred) if p == "BAD" and g == "BAD")
        fp = sum(1 for g, p in zip(gold, pred) if p == "BAD" and g == "OK")
        fn = sum(1 for g, p in zip(gold, pred) if p == "OK" and g == "BAD")
        tn = sum(1 for g, p in zip(gold, pred) if p == "OK" and g == "OK")

        recs.append({
            "uid": r.get("uid"),
            "language_pair": r["language_pair"],
            "wtag_provenance": r.get("wtag_provenance"),
            "n_tokens": len(gold),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "n_correct": tp + tn,
            "gold_bad": tp + fn,
            "pred_bad": tp + fp,
            "exact_sequence": int(pred == gold),
        })

    print(f"evaluated {len(recs)} | skipped {skipped}")
    return pd.DataFrame(recs)


def score_block(g: pd.DataFrame) -> dict:
    tp, tn, fp, fn = int(g.tp.sum()), int(g.tn.sum()), int(g.fp.sum()), int(g.fn.sum())
    tot = tp + tn + fp + fn
    p, r, f = prf(tp, fp, fn)
    return {
        "n_segments": len(g), "n_tokens": tot,
        "token_acc": (tp + tn) / tot if tot else 0.0,
        "bad_p": p, "bad_r": r, "bad_f1": f,
        "mcc": mcc(tp, tn, fp, fn),
        "exact_seq": g.exact_sequence.mean() if len(g) else 0.0,
        "gold_bad_rate": (tp + fn) / tot if tot else 0.0,
    }


def main() -> int:
    d = load_full()
    per = evaluate(d)
    if per.empty:
        print("no rows evaluated", file=sys.stderr)
        return 1

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    per.to_csv(OUT_CSV, index=False)
    print(f"per-instance rows written: {len(per)} -> {OUT_CSV}")

    L = ["# Word-tag validation with MCC\n\n",
         "Derived tags are scored against every **real** released tag. MCC is the WMT ",
         "word-level metric; accuracy is inflated by the OK majority class and BAD-F1 ",
         "ignores true negatives. Per-instance confusion counts in ",
         "`wordqe_mcc_perinstance.csv`; every figure below is a sum over those rows.\n\n"]

    blocks = [("ALL real tags", per)]
    if per.wtag_provenance.notna().any():
        for prov, g in per.groupby("wtag_provenance"):
            blocks.append((f"provenance: {prov}", g))

    L.append("| scope | segs | tokens | gold BAD rate | token acc | BAD P | BAD R | BAD F1 | **MCC** | exact seq |\n")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n")
    for name, g in blocks:
        s = score_block(g)
        L.append(f"| {name} | {s['n_segments']:,} | {s['n_tokens']:,} | {s['gold_bad_rate']:.3f} | "
                 f"{s['token_acc']:.4f} | {s['bad_p']:.3f} | {s['bad_r']:.3f} | {s['bad_f1']:.3f} | "
                 f"**{s['mcc']:.3f}** | {s['exact_seq']:.3f} |\n")

    L.append("\n## Per language pair\n\n")
    L.append("| pair | segs | tokens | gold BAD rate | token acc | BAD F1 | **MCC** |\n|---|--:|--:|--:|--:|--:|--:|\n")
    for pair, g in sorted(per.groupby("language_pair")):
        s = score_block(g)
        L.append(f"| {pair} | {s['n_segments']:,} | {s['n_tokens']:,} | {s['gold_bad_rate']:.3f} | "
                 f"{s['token_acc']:.4f} | {s['bad_f1']:.3f} | **{s['mcc']:.3f}** |\n")

    L.append("\n## Reading\n\n")
    a = score_block(per)
    L.append(f"Corpus MCC is {a['mcc']:.3f} over {a['n_tokens']:,} tokens on {a['n_segments']:,} "
             f"segments. Token accuracy of {a['token_acc']:.4f} sits against a gold BAD rate of "
             f"{a['gold_bad_rate']:.3f}, so a majority-class baseline that tagged everything OK "
             f"would already score {1 - a['gold_bad_rate']:.4f} accuracy and MCC 0. MCC is the "
             f"figure to report.\n")

    OUT_MD.write_text("".join(L))
    print("".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
