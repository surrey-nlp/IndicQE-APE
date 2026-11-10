#!/usr/bin/env python3
"""wordqe_petok_test.py — does tokenising the post-edit with the SAME tokeniser that produced
`mt_tokens` (IndicNLP trivial_tokenize, identified by tokeniser_bakeoff.py) improve tag
reproduction over an ad-hoc regex? MT side is fixed (the shipped mt_tokens); only the PE
tokeniser varies, so this isolates the tokenisation effect from the alignment algorithm.
"""
from __future__ import annotations
import json, re, sys, unicodedata
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download
from indicnlp.tokenize import indic_tokenize

PUNCT = r"[\.,;:!\?\(\)\[\]\{\}\"'«»„“”‘’…।॥\-–—/]"


def tok_regex(s):
    s = re.sub(f"({PUNCT})", r" \1 ", str(s).strip())
    return [t for t in s.split() if t]


def tok_indic(s):
    return [t for t in indic_tokenize.trivial_tokenize(str(s)) if t.strip()]


def as_list(x):
    if x is None: return None
    if isinstance(x, str):
        try: return json.loads(x)
        except Exception: return x.split()
    return list(x)


def align(mt, pe):
    M = [unicodedata.normalize("NFC", t) for t in mt]
    P = [unicodedata.normalize("NFC", t) for t in pe]
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
            tags[i-1] = "OK" if c == 0 else "BAD"; i -= 1; j -= 1
        elif D[i, j] == D[i-1, j] + 1:
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


def main(n=250):
    p = hf_hub_download("surrey-nlp/IndicQE-APE", "data/full/train.parquet", repo_type="dataset")
    d = pd.read_parquet(p)
    w = d[d.wtag.notna() & d.post_edit.notna() & d.post_edit.astype(str).str.strip().ne("")]
    w = w.groupby("language_pair", group_keys=False).head(n)
    print(f"sample {len(w)} rows {w.language_pair.value_counts().to_dict()}\n")
    TOKS = {"PE=regex": tok_regex, "PE=indicnlp": tok_indic}
    agg = {k: {"tot":0,"cor":0,"tp":0,"fp":0,"fn":0,"exact":0,"rows":0} for k in TOKS}
    pp = {k: {} for k in TOKS}
    for _, r in w.iterrows():
        gold = r["wtag"].split() if isinstance(r["wtag"], str) else list(r["wtag"])
        mtt = as_list(r["mt_tokens"])
        if not mtt: continue
        if str(mtt[-1]) == "<EOS>": mtt, gold = mtt[:-1], gold[:-1]
        if len(gold) != len(mtt): continue
        mtt = [str(t) for t in mtt]
        for name, fn in TOKS.items():
            pred = align(mtt, fn(r["post_edit"]))
            s = agg[name]; s["rows"] += 1
            q = pp[name].setdefault(r["language_pair"], {"tot":0,"cor":0,"tp":0,"fp":0,"fn":0})
            if pred == [str(g).upper() for g in gold]: s["exact"] += 1
            for g, pr in zip(gold, pred):
                g = str(g).upper(); s["tot"] += 1; q["tot"] += 1
                if g == pr: s["cor"] += 1; q["cor"] += 1
                if pr=="BAD" and g=="BAD": s["tp"]+=1; q["tp"]+=1
                elif pr=="BAD" and g=="OK": s["fp"]+=1; q["fp"]+=1
                elif pr=="OK" and g=="BAD": s["fn"]+=1; q["fn"]+=1
    print(f"{'PE tokeniser':<14}{'tok acc':>9}{'BAD P':>8}{'BAD R':>8}{'BAD F1':>8}{'exact':>8}")
    for name, s in agg.items():
        P,R,F = prf(s["tp"],s["fp"],s["fn"])
        print(f"{name:<14}{s['cor']/max(s['tot'],1):>9.4f}{P:>8.3f}{R:>8.3f}{F:>8.3f}"
              f"{s['exact']/max(s['rows'],1):>8.3f}")
    pairs = sorted({x for v in pp.values() for x in v})
    print(f"\nBAD-F1 per pair:\n{'PE tokeniser':<14}" + "".join(f"{x:>9}" for x in pairs))
    for name in TOKS:
        cells = []
        for x in pairs:
            s = pp[name].get(x)
            cells.append(f"{prf(s['tp'],s['fp'],s['fn'])[2]:>9.3f}" if s else f"{'-':>9}")
        print(f"{name:<14}" + "".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 250))
