#!/usr/bin/env python3
"""wordqe_derive_poc.py — feasibility test for deriving word-level OK/BAD tags from MT+PE.

Premise: WMT word-level QE tags are not direct human annotation; they are produced by aligning MT
to its post-edit (tercom) and marking substituted/deleted MT tokens BAD. If that is how the tags in
IndicQE-APE were made, we can reproduce them — and then derive tags for the 61k rows that have a
post-edit but no tags.

This script measures whether we can reproduce the EXISTING tags. It uses the dataset's own
`mt_tokens` (authoritative, drops the trailing <EOS>) as the MT token sequence, tokenises the
post-edit with the same rule set, aligns, and compares to the gold `wtag`.

Reported: token-level accuracy, BAD-class precision/recall/F1, and exact-sequence match. A high
number licenses derivation on the remaining rows; a low number means the upstream pipeline differs
(different tokenisation/truecasing/tercom shifts) and we must use the real tercom jar.
"""
from __future__ import annotations
import json, re, sys
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download

PUNCT = r"[\.,;:!\?\(\)\[\]\{\}\"'«»„“”‘’…।॥\-–—/]"


def tok(s: str) -> list[str]:
    """Whitespace split with punctuation separated (matches the shape of the shipped mt_tokens)."""
    s = str(s).strip()
    s = re.sub(f"({PUNCT})", r" \1 ", s)
    return [t for t in s.split() if t]


def as_list(x):
    if x is None: return None
    if isinstance(x, str):
        try: return json.loads(x)
        except Exception: return x.split()
    return list(x)


def align_tags(mt: list[str], pe: list[str]) -> list[str]:
    """Levenshtein alignment MT->PE; MT token is BAD if substituted or deleted, OK if matched.
    (Insertions in PE are gap errors and carry no MT word tag.)"""
    n, m = len(mt), len(pe)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1); D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if mt[i - 1] == pe[j - 1] else 1
            D[i, j] = min(D[i - 1, j - 1] + cost, D[i - 1, j] + 1, D[i, j - 1] + 1)
    tags = ["OK"] * n
    i, j = n, m
    while i > 0 and j > 0:
        cost = 0 if mt[i - 1] == pe[j - 1] else 1
        if D[i, j] == D[i - 1, j - 1] + cost:
            tags[i - 1] = "OK" if cost == 0 else "BAD"; i -= 1; j -= 1
        elif D[i, j] == D[i - 1, j] + 1:
            tags[i - 1] = "BAD"; i -= 1              # deletion
        else:
            j -= 1                                   # insertion -> gap, no MT tag
    while i > 0:
        tags[i - 1] = "BAD"; i -= 1
    return tags


def main(limit=1500):
    p = hf_hub_download("surrey-nlp/IndicQE-APE", "data/full/train.parquet", repo_type="dataset")
    d = pd.read_parquet(p)
    w = d[d.wtag.notna() & d.post_edit.notna() & d.post_edit.astype(str).str.strip().ne("")].copy()
    print(f"rows with gold tags AND post-edit: {len(w)}")
    # stratify across pairs so every pair is represented
    per = max(limit // max(w.language_pair.nunique(), 1), 1)
    w = w.groupby("language_pair", group_keys=False).head(per)
    print(f"stratified sample: {len(w)} rows ({w.language_pair.value_counts().to_dict()})")
    limit = len(w)

    agg = {"tok_total": 0, "tok_correct": 0, "tp": 0, "fp": 0, "fn": 0,
           "rows": 0, "exact": 0, "skipped_len": 0}
    per_pair = {}
    eos_tags = []
    for _, r in w.head(limit).iterrows():
        gold = r["wtag"].split() if isinstance(r["wtag"], str) else list(r["wtag"])
        mtt = as_list(r["mt_tokens"])
        if not mtt:
            continue
        if str(mtt[-1]) == "<EOS>":
            eos_tags.append(gold[-1] if len(gold) == len(mtt) else None)
            mtt_use, gold_use = mtt[:-1], gold[:-1]
        else:
            mtt_use, gold_use = mtt, gold
        if len(gold_use) != len(mtt_use):
            agg["skipped_len"] += 1
            continue
        pred = align_tags([str(t) for t in mtt_use], tok(r["post_edit"]))
        agg["rows"] += 1
        pair = r["language_pair"]
        pp = per_pair.setdefault(pair, {"tok": 0, "cor": 0, "tp": 0, "fp": 0, "fn": 0, "rows": 0})
        pp["rows"] += 1
        if pred == gold_use:
            agg["exact"] += 1
        for g, q in zip(gold_use, pred):
            g = str(g).upper(); agg["tok_total"] += 1; pp["tok"] += 1
            if g == q: agg["tok_correct"] += 1; pp["cor"] += 1
            if q == "BAD" and g == "BAD": agg["tp"] += 1; pp["tp"] += 1
            elif q == "BAD" and g == "OK": agg["fp"] += 1; pp["fp"] += 1
            elif q == "OK" and g == "BAD": agg["fn"] += 1; pp["fn"] += 1

    def prf(tp, fp, fn):
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f_ = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        return p_, r_, f_

    P, R, F = prf(agg["tp"], agg["fp"], agg["fn"])
    print(f"\nevaluated rows: {agg['rows']} (skipped for token-length mismatch: {agg['skipped_len']})")
    print(f"token accuracy      : {agg['tok_correct']/max(agg['tok_total'],1):.4f}")
    print(f"BAD precision/recall/F1: {P:.4f} / {R:.4f} / {F:.4f}")
    print(f"exact sequence match: {agg['exact']/max(agg['rows'],1):.4f}")
    from collections import Counter
    print(f"gold tag at <EOS> position: {Counter([e for e in eos_tags if e])}")
    print("\nper pair:")
    for pair, s in sorted(per_pair.items()):
        p_, r_, f_ = prf(s["tp"], s["fp"], s["fn"])
        print(f"  {pair}: n={s['rows']:>4} acc={s['cor']/max(s['tok'],1):.3f} BAD-F1={f_:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500))
