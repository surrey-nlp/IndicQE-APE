#!/usr/bin/env python3
"""derive_word_tags.py — derive word-level OK/BAD QE tags from MT + post-edit.

Word-level QE tags in the WMT lineage are not direct human annotation: they are produced by aligning
MT to its post-edit and marking substituted/deleted MT tokens BAD (cf. Unbabel/word-level-qe-corpus-
builder). They are therefore a deterministic function of (MT, post-edit), so any row carrying a
post-edit can be tagged with the same rule and the same provenance class.

Design decisions, all measured:
  * Tokeniser = IndicNLP `trivial_tokenize` on BOTH sides. It reproduces the shipped `mt_tokens` at
    0.960 exact-sequence (1.000 on en-ml) where Moses reaches only 0.660 (0.313 on en-ml).
  * Comparison = exact string match after NFC. Softening it with grapheme similarity was measured
    WORSE (BAD-F1 0.942 -> 0.889): the gold labels do mark morphological variants BAD.
  * Alignment = Levenshtein (substitution/deletion -> BAD, PE insertion -> gap, no MT tag).
    TER *shift* accounting (real tercom) remains an open refinement for the recall gap.

Modes:
  --validate   reproduce the EXISTING gold tags and report per-pair agreement (the G1 gate).
               MT side uses the shipped `mt_tokens`, so only the PE tokenisation/alignment is tested.
  --derive     emit derived tags for rows that have a post-edit but no tags, to a parquet of
               (uid, mt_tokens_derived, wtag_derived, wtag_provenance, n_tokens, bad_rate).

Safety: rows whose post-edit is a placeholder (XXX/xxx, empty, or pure punctuation) are EXCLUDED,
never tagged — they would otherwise produce meaningless all-BAD sequences.
"""
from __future__ import annotations
import argparse, json, re, sys, unicodedata
from pathlib import Path
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download
from indicnlp.tokenize import indic_tokenize

REPO = "surrey-nlp/IndicQE-APE"
SPLITS = ("train", "validation", "test")
PLACEHOLDER = re.compile(r"^(x{2,}|X{2,}|-+|\.+|\s*)$")


def tok(s) -> list[str]:
    return [t for t in indic_tokenize.trivial_tokenize(str(s)) if t.strip()]


def as_list(x):
    if x is None: return None
    if isinstance(x, str):
        try: return json.loads(x)
        except Exception: return x.split()
    return list(x)


def is_placeholder(pe) -> bool:
    if pe is None: return True
    s = str(pe).strip()
    if not s or PLACEHOLDER.match(s): return True
    # trailing sentinel seen in the APE data ("...xxx", "...XXX")
    return bool(re.fullmatch(r"[xX]{3,}", s))


def align_tags(mt: list[str], pe: list[str]) -> list[str]:
    """Levenshtein alignment MT->PE. MT token substituted or deleted -> BAD, matched -> OK.
    Insertions in PE are gap errors and carry no MT word tag."""
    M = [unicodedata.normalize("NFC", t) for t in mt]
    P = [unicodedata.normalize("NFC", t) for t in pe]
    n, m = len(M), len(P)
    if n == 0: return []
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1); D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        Mi = M[i - 1]
        for j in range(1, m + 1):
            c = 0 if Mi == P[j - 1] else 1
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


def load_full() -> pd.DataFrame:
    fr = []
    for sp in SPLITS:
        p = hf_hub_download(REPO, f"data/full/{sp}.parquet", repo_type="dataset")
        d = pd.read_parquet(p); d["_split"] = sp; fr.append(d)
    return pd.concat(fr, ignore_index=True)


def prf(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    return p, r, (2*p*r/(p+r) if p+r else 0.0)


def validate(d: pd.DataFrame, out: Path | None):
    """G1: reproduce the existing gold tags. MT tokens are taken as shipped."""
    w = d[d.wtag.notna() & d.post_edit.notna()].copy()
    w = w[~w.post_edit.map(is_placeholder)]
    print(f"G1 validation over {len(w)} gold rows with a usable post-edit")
    agg = {"tot":0,"cor":0,"tp":0,"fp":0,"fn":0,"rows":0,"exact":0,"skip":0}
    pp = {}
    for _, r in w.iterrows():
        gold = r["wtag"].split() if isinstance(r["wtag"], str) else list(r["wtag"])
        mtt = as_list(r["mt_tokens"])
        if not mtt: agg["skip"] += 1; continue
        if str(mtt[-1]) == "<EOS>": mtt, gold = mtt[:-1], gold[:-1]
        if len(gold) != len(mtt): agg["skip"] += 1; continue
        pred = align_tags([str(t) for t in mtt], tok(r["post_edit"]))
        agg["rows"] += 1
        q = pp.setdefault(r["language_pair"], {"tot":0,"cor":0,"tp":0,"fp":0,"fn":0,"rows":0})
        q["rows"] += 1
        gold = [str(g).upper() for g in gold]
        if pred == gold: agg["exact"] += 1
        for g, pr_ in zip(gold, pred):
            agg["tot"] += 1; q["tot"] += 1
            if g == pr_: agg["cor"] += 1; q["cor"] += 1
            if pr_=="BAD" and g=="BAD": agg["tp"]+=1; q["tp"]+=1
            elif pr_=="BAD" and g=="OK": agg["fp"]+=1; q["fp"]+=1
            elif pr_=="OK" and g=="BAD": agg["fn"]+=1; q["fn"]+=1
    P,R,F = prf(agg["tp"],agg["fp"],agg["fn"])
    acc = agg["cor"]/max(agg["tot"],1)
    lines = ["# G1 — reproduction of existing word tags from MT + post-edit\n",
             f"rows evaluated {agg['rows']} (skipped {agg['skip']}), tokens {agg['tot']}\n",
             f"\n| scope | token acc | BAD P | BAD R | BAD F1 | exact seq |\n|---|--:|--:|--:|--:|--:|\n",
             f"| ALL | {acc:.4f} | {P:.3f} | {R:.3f} | {F:.3f} | {agg['exact']/max(agg['rows'],1):.3f} |\n"]
    print(f"\nALL: acc {acc:.4f} | BAD P/R/F1 {P:.3f}/{R:.3f}/{F:.3f} | exact {agg['exact']/max(agg['rows'],1):.3f}")
    for pair, q in sorted(pp.items()):
        p_,r_,f_ = prf(q["tp"],q["fp"],q["fn"])
        a_ = q["cor"]/max(q["tot"],1)
        print(f"  {pair}: n={q['rows']:>5} acc={a_:.4f} BAD-F1={f_:.3f}")
        lines.append(f"| {pair} | {a_:.4f} | {p_:.3f} | {r_:.3f} | {f_:.3f} | — |\n")
    gate = acc >= 0.97 and F >= 0.94
    print(f"\nGATE (acc>=0.97 and BAD-F1>=0.94): {'PASS' if gate else 'FAIL'}")
    lines.append(f"\n**Gate (acc >= 0.97, BAD-F1 >= 0.94): {'PASS' if gate else 'FAIL'}**\n")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text("".join(lines))
        print(f"written {out}")
    return gate


def derive(d: pd.DataFrame, out: Path):
    """Emit derived tags for rows with a post-edit but no tags."""
    has_wt = d.wtag.notna() & d.wtag.astype(str).str.strip().ne("")
    cand = d[~has_wt & d.post_edit.notna()].copy()
    n0 = len(cand)
    cand = cand[~cand.post_edit.map(is_placeholder)]
    cand = cand[cand.mt.notna() & cand.mt.astype(str).str.strip().ne("")]
    print(f"candidates: {n0} PE-and-no-tag rows -> {len(cand)} after placeholder/empty filter "
          f"(excluded {n0-len(cand)})")
    recs = []
    for _, r in cand.iterrows():
        mt_t = tok(r["mt"]); pe_t = tok(r["post_edit"])
        if not mt_t: continue
        tags = align_tags(mt_t, pe_t)
        assert len(tags) == len(mt_t)                      # G2 self-consistency
        recs.append({"uid": r["uid"], "language_pair": r["language_pair"],
                     "mt_tokens_derived": json.dumps(mt_t + ["<EOS>"], ensure_ascii=False),
                     "wtag_derived": " ".join(tags + ["OK"]),
                     "wtag_provenance": "derived-ter-indicnlp",
                     "n_tokens": len(mt_t),
                     "bad_rate": float(np.mean([t == "BAD" for t in tags]))})
    out_df = pd.DataFrame(recs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    print(f"\nderived {len(out_df)} rows -> {out}")
    print("per pair:", out_df.groupby('language_pair').size().to_dict())
    print(f"BAD rate: mean {out_df.bad_rate.mean():.3f} median {out_df.bad_rate.median():.3f} "
          f"all-OK rows {(out_df.bad_rate==0).mean():.1%}")
    return out_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--derive", action="store_true")
    ap.add_argument("--report", default="results/analysis/wordqe_g1_validation.md")
    ap.add_argument("--out", default="dataset/derived_word_tags.parquet")
    a = ap.parse_args()
    d = load_full()
    print(f"loaded full: {len(d)} rows")
    if a.validate:
        validate(d, Path(a.report))
    if a.derive:
        derive(d, Path(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
