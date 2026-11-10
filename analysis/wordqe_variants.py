#!/usr/bin/env python3
"""wordqe_variants.py — diagnose the reproduction gap and compare alignment/normalisation variants
for Indic MT->PE word tagging. Answers: is TERCOM's exact-match word comparison the right primitive
for agglutinative Indic targets, and does Unicode/Indic normalisation or a character-similarity
substitution cost do better?

Variants (all word-level tags, WMT-compatible):
  V0 exact          — baseline: exact string match (what tercom does after tokenisation)
  V1 nfc            — V0 + Unicode NFC normalisation of both sides
  V2 nfc+case       — V1 + casefold (matters only for X->en targets)
  V3 charsim        — V1 + substitution treated as OK when grapheme-level similarity >= threshold
                       (a word whose suffix changed is a partial match, not a wholesale error)
Also dumps a diagnostic of the disagreements so the dominant error mode is visible.
"""
from __future__ import annotations
import json, re, sys, unicodedata
from collections import Counter
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download

PUNCT = r"[\.,;:!\?\(\)\[\]\{\}\"'«»„“”‘’…।॥\-–—/]"


def tok(s):
    s = re.sub(f"({PUNCT})", r" \1 ", str(s).strip())
    return [t for t in s.split() if t]


def as_list(x):
    if x is None: return None
    if isinstance(x, str):
        try: return json.loads(x)
        except Exception: return x.split()
    return list(x)


def norm(t, nfc=False, case=False):
    if nfc: t = unicodedata.normalize("NFC", t)
    if case: t = t.casefold()
    return t


def graphemes(s):
    """Approximate grapheme clusters: a base char plus following combining marks."""
    out = []
    for ch in s:
        if out and unicodedata.combining(ch):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def charsim(a, b):
    """Normalised grapheme-level similarity (1 - edit_distance/max_len)."""
    A, B = graphemes(a), graphemes(b)
    if not A and not B: return 1.0
    n, m = len(A), len(B)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j - 1] + (0 if A[i-1] == B[j-1] else 1), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return 1.0 - prev[m] / max(n, m)


def align(mt, pe, nfc=False, case=False, sim_thr=None, collect=None):
    """Levenshtein alignment; returns word tags for mt. If sim_thr is set, a substitution whose
    grapheme similarity >= sim_thr is recorded as OK (morphological variant, not a new word)."""
    M = [norm(t, nfc, case) for t in mt]
    P = [norm(t, nfc, case) for t in pe]
    n, m = len(M), len(P)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1); D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if M[i-1] == P[j-1] else 1
            D[i, j] = min(D[i-1, j-1] + c, D[i-1, j] + 1, D[i, j-1] + 1)
    tags = ["OK"] * n
    i, j = n, m
    while i > 0 and j > 0:
        c = 0 if M[i-1] == P[j-1] else 1
        if D[i, j] == D[i-1, j-1] + c:
            if c == 0:
                tags[i-1] = "OK"
            else:
                s = charsim(M[i-1], P[j-1])
                if collect is not None: collect.append(("sub", M[i-1], P[j-1], round(s, 2)))
                tags[i-1] = "OK" if (sim_thr is not None and s >= sim_thr) else "BAD"
            i -= 1; j -= 1
        elif D[i, j] == D[i-1, j] + 1:
            if collect is not None: collect.append(("del", M[i-1], "", 0.0))
            tags[i-1] = "BAD"; i -= 1
        else:
            j -= 1
    while i > 0:
        tags[i-1] = "BAD"; i -= 1
    return tags


def prf(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    return p, r, (2*p*r/(p+r) if p+r else 0.0)


def main(per_pair_n=250):
    p = hf_hub_download("surrey-nlp/IndicQE-APE", "data/full/train.parquet", repo_type="dataset")
    d = pd.read_parquet(p)
    w = d[d.wtag.notna() & d.post_edit.notna() & d.post_edit.astype(str).str.strip().ne("")]
    w = w.groupby("language_pair", group_keys=False).head(per_pair_n)
    print(f"sample: {len(w)} rows {w.language_pair.value_counts().to_dict()}\n")

    VARIANTS = {
        "V0 exact":      dict(nfc=False, case=False, sim_thr=None),
        "V1 nfc":        dict(nfc=True,  case=False, sim_thr=None),
        "V2 nfc+case":   dict(nfc=True,  case=True,  sim_thr=None),
        "V3 charsim.60": dict(nfc=True,  case=False, sim_thr=0.60),
        "V3 charsim.75": dict(nfc=True,  case=False, sim_thr=0.75),
        "V3 charsim.85": dict(nfc=True,  case=False, sim_thr=0.85),
    }
    rows = {k: {"tot":0,"cor":0,"tp":0,"fp":0,"fn":0} for k in VARIANTS}
    per_pair = {k: {} for k in VARIANTS}
    fn_diag, fp_diag = [], []

    for _, r in w.iterrows():
        gold = r["wtag"].split() if isinstance(r["wtag"], str) else list(r["wtag"])
        mtt = as_list(r["mt_tokens"])
        if not mtt: continue
        if str(mtt[-1]) == "<EOS>": mtt, gold = mtt[:-1], gold[:-1]
        if len(gold) != len(mtt): continue
        mtt = [str(t) for t in mtt]; pet = tok(r["post_edit"])
        for name, kw in VARIANTS.items():
            coll = [] if name == "V1 nfc" else None
            pred = align(mtt, pet, collect=coll, **kw)
            s = rows[name]
            pp = per_pair[name].setdefault(r["language_pair"], {"tot":0,"cor":0,"tp":0,"fp":0,"fn":0})
            for g, q, mtok in zip(gold, pred, mtt):
                g = str(g).upper()
                s["tot"] += 1; pp["tot"] += 1
                if g == q: s["cor"] += 1; pp["cor"] += 1
                if q=="BAD" and g=="BAD": s["tp"]+=1; pp["tp"]+=1
                elif q=="BAD" and g=="OK":
                    s["fp"]+=1; pp["fp"]+=1
                    if name=="V1 nfc" and len(fp_diag)<12: fp_diag.append((r["language_pair"], mtok))
                elif q=="OK" and g=="BAD":
                    s["fn"]+=1; pp["fn"]+=1
                    if name=="V1 nfc" and len(fn_diag)<12: fn_diag.append((r["language_pair"], mtok))

    print(f"{'variant':<16}{'tok acc':>9}{'BAD P':>8}{'BAD R':>8}{'BAD F1':>8}")
    for name, s in rows.items():
        P,R,F = prf(s["tp"],s["fp"],s["fn"])
        print(f"{name:<16}{s['cor']/max(s['tot'],1):>9.4f}{P:>8.3f}{R:>8.3f}{F:>8.3f}")

    print("\nper-pair BAD-F1 by variant:")
    pairs = sorted({p for v in per_pair.values() for p in v})
    print(f"{'variant':<16}" + "".join(f"{p:>9}" for p in pairs))
    for name in VARIANTS:
        cells = []
        for pr in pairs:
            s = per_pair[name].get(pr)
            cells.append(f"{prf(s['tp'],s['fp'],s['fn'])[2]:>9.3f}" if s else f"{'-':>9}")
        print(f"{name:<16}" + "".join(cells))

    print("\ndisagreement sample (V1): gold BAD but we predict OK (unchanged tokens):")
    for lp, t in fn_diag[:8]: print(f"  {lp}: {t}")
    print("gold OK but we predict BAD:")
    for lp, t in fp_diag[:8]: print(f"  {lp}: {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 250))
