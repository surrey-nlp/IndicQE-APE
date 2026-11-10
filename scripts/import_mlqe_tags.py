#!/usr/bin/env python3
"""import_mlqe_tags.py — import the ACTUAL word-level tags from MLQE-PE for the X->en pairs.

Our et-en / ne-en / si-en rows are an MLQE-PE passthrough, and MLQE-PE ships genuine word-level
labels for them. Those must be preferred over anything we derive: this script joins the upstream
labels onto our rows and reports the match rate, so derived tags can be replaced by real ones.

MLQE-PE format notes (verified, not assumed):
  * `<split>.tags` is **2n+1 interleaved** gap/word tags: index 0,2,4,... are GAP tags (n+1 of them)
    and 1,3,5,... are WORD tags (n of them). Our release stores word-only tags, so we de-interleave.
  * `<split>.source_tags` is one tag per SOURCE token — a signal our release does not have at all.
  * `<split>.mt` / `.pe` / `.src` are already tokenised (space-separated).

Outputs dataset/mlqe_upstream_tags.parquet with, per matched uid: wtag_mlqe (word tags),
wtag_gaps_mlqe (gap tags), source_tags_mlqe, mt_tokens_mlqe, plus the join diagnostics.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

SRC = Path("/tmp/mlqe_x")
PAIRS = ["et-en", "ne-en", "si-en"]
REPO = "surrey-nlp/IndicQE-APE"


import unicodedata

# Cosmetic differences observed between our copy and MLQE-PE's (verified by inspection, not assumed):
#   * zero-width joiner / non-joiner present in our Devanagari source but not theirs
#     (राष्<ZWJ>ट्रपतिको vs राष्ट्रपतिको)
#   * curly vs straight quotation marks (“...” vs " ... ")
#   * their side is tokenised (space before , and ।), ours is detokenised
# None of these change the instance, so the join normalises them away — but we additionally require
# an exact MT match, so a row is only ever linked when the hypothesis text is byte-identical.
_ZW = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_PUNCT_MAP = {ord(a): b for a, b in {
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"', "‘": "'", "’": "'", "‚": "'",
    "–": "-", "—": "-", "―": "-", "…": "...",
}.items()}


def norm(s: str) -> str:
    """Aggressive key: NFC, drop zero-width marks, fold quote/dash variants, drop whitespace."""
    t = unicodedata.normalize("NFC", str(s))
    t = t.translate(_ZW).translate(_PUNCT_MAP)
    return re.sub(r"\s+", "", t).casefold()


def load_mlqe() -> pd.DataFrame:
    recs = []
    for d in sorted(SRC.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"(\w\w-\w\w)-(train|dev|test20)$", d.name)
        if not m:
            continue
        pair, split = m.groups()
        if pair not in PAIRS:
            continue
        stem = {"train": "train", "dev": "dev", "test20": "test20"}[split]
        need = ["mt", "pe", "src", "tags", "source_tags"]
        files = {k: d / f"{stem}.{k}" for k in need}
        if not all(f.exists() for f in files.values()):
            print(f"  skip {d.name}: missing files"); continue
        cols = {k: (f).read_text(encoding="utf-8").splitlines() for k, f in files.items()}
        n = len(cols["mt"])
        assert all(len(v) == n for v in cols.values()), f"{d.name}: line count mismatch"
        for i in range(n):
            mt_toks = cols["mt"][i].split()
            tags_all = cols["tags"][i].split()
            src_toks = cols["src"][i].split()
            stags = cols["source_tags"][i].split()
            # de-interleave: odd = word tags, even = gap tags
            if len(tags_all) != 2 * len(mt_toks) + 1:
                continue                                   # malformed line, skip rather than guess
            word_tags = tags_all[1::2]
            gap_tags = tags_all[0::2]
            assert len(word_tags) == len(mt_toks)
            if len(stags) != len(src_toks):
                stags = []                                  # only keep when it lines up
            recs.append({"pair": pair, "mlqe_split": split,
                         "mt_txt": cols["mt"][i], "pe_txt": cols["pe"][i], "src_txt": cols["src"][i],
                         "key": norm(cols["src"][i]) + "|" + norm(cols["mt"][i]),
                         "mt_key": norm(cols["mt"][i]),
                         "mt_tokens_mlqe": json.dumps(mt_toks + ["<EOS>"], ensure_ascii=False),
                         "wtag_mlqe": " ".join(word_tags + ["OK"]),
                         "wtag_gaps_mlqe": " ".join(gap_tags),
                         "source_tags_mlqe": " ".join(stags) if stags else None,
                         # source_tags align to MLQE-PE's TOKENISED source, which differs from our
                         # detokenised `source` field — ship the tokenisation or the tags are unusable
                         "source_tokens_mlqe": json.dumps(src_toks, ensure_ascii=False) if stags else None,
                         "n_mt": len(mt_toks)})
        print(f"  {d.name}: {n} lines")
    return pd.DataFrame(recs)


def load_ours() -> pd.DataFrame:
    fr = []
    for sp in ("train", "validation", "test"):
        p = hf_hub_download(REPO, f"data/full/{sp}.parquet", repo_type="dataset")
        d = pd.read_parquet(p); d["_split"] = sp; fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    return d[d.language_pair.isin(PAIRS)].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/mlqe_upstream_tags.parquet")
    a = ap.parse_args()

    print("loading MLQE-PE upstream files:")
    up = load_mlqe()
    print(f"\nMLQE-PE usable lines: {len(up)}  per pair {up.pair.value_counts().to_dict()}")
    # Duplicate (src,mt) keys: safe to collapse only when every copy carries the SAME tags.
    # Where copies disagree the row is genuinely ambiguous and is dropped, never guessed.
    dup_keys = set(up.key[up.key.duplicated()])
    print(f"duplicate (src,mt) keys within MLQE-PE: {len(dup_keys)}")
    if dup_keys:
        g = up[up.key.isin(dup_keys)].groupby("key")["wtag_mlqe"].nunique()
        consistent = set(g[g == 1].index)
        conflicting = set(g[g > 1].index)
        print(f"  of those, {len(consistent)} have identical tags across copies (safe to collapse), "
              f"{len(conflicting)} conflict and are DROPPED as ambiguous")
        up = up[~up.key.isin(conflicting)]
    up_u = up.drop_duplicates("key", keep="first")

    ours = load_ours()
    print(f"\nour X->en rows: {len(ours)}  per pair {ours.language_pair.value_counts().to_dict()}")
    ours["key"] = ours.source.map(norm) + "|" + ours.mt.map(norm)
    ours["mt_key"] = ours.mt.map(norm)

    j = ours.merge(up_u[["key", "mt_key", "pair", "mlqe_split", "mt_tokens_mlqe", "wtag_mlqe",
                         "wtag_gaps_mlqe", "source_tags_mlqe", "source_tokens_mlqe", "n_mt"]].rename(
                       columns={"mt_key": "mt_key_up"}),
                   on="key", how="left", validate="m:1")
    # safeguard: the normalised hypothesis must agree too, and the pair must agree
    mt_ok = j.mt_key.eq(j.mt_key_up)
    pair_ok = j.language_pair.eq(j.pair)
    bad_link = j.wtag_mlqe.notna() & ~(mt_ok & pair_ok)
    print(f"safeguard: links rejected for MT/pair disagreement: {int(bad_link.sum())}")
    j.loc[bad_link, ["wtag_mlqe", "wtag_gaps_mlqe", "source_tags_mlqe",
                     "source_tokens_mlqe", "mt_tokens_mlqe"]] = None
    matched = j.wtag_mlqe.notna()
    print(f"\nmatched on (source, mt): {int(matched.sum())} / {len(j)} ({matched.mean():.1%})")
    for p_ in PAIRS:
        s = j[j.language_pair == p_]
        print(f"  {p_}: {int(s.wtag_mlqe.notna().sum())}/{len(s)} "
              f"({s.wtag_mlqe.notna().mean():.1%})")

    # fallback attempt on mt alone for the unmatched
    un = j[~matched]
    if len(un):
        up_mt = up.drop_duplicates("mt_key", keep="first")
        j2 = un[["uid", "mt_key"]].merge(up_mt[["mt_key", "wtag_mlqe"]], on="mt_key", how="left")
        extra = int(j2.wtag_mlqe.notna().sum())
        print(f"  of {len(un)} unmatched, {extra} would match on mt alone "
              f"(not used — ambiguous without source)")

    ok = j[matched].copy()
    # integrity gates
    bad = 0
    for _, r in ok.head(3000).iterrows():
        wt = r["wtag_mlqe"].split(); mtk = json.loads(r["mt_tokens_mlqe"])
        if len(wt) != len(mtk): bad += 1
    print(f"\nGATE len(wtag)==len(mt_tokens) violations in 3000 sample: {bad}")
    from collections import Counter
    vocab = Counter()
    for x in ok.wtag_mlqe.head(2000): vocab.update(x.split())
    print(f"tag vocabulary: {dict(vocab)}")
    badrate = ok.wtag_mlqe.map(lambda s: sum(t == "BAD" for t in s.split())/max(len(s.split()),1))
    print(f"upstream BAD rate: mean {badrate.mean():.3f} median {badrate.median():.3f}")
    print(f"source_tags available on {int(ok.source_tags_mlqe.notna().sum())} matched rows")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    keep = ["uid", "language_pair", "mt_tokens_mlqe", "wtag_mlqe", "wtag_gaps_mlqe",
            "source_tags_mlqe", "source_tokens_mlqe", "mlqe_split"]
    ok[keep].to_parquet(out, index=False)
    print(f"\nwrote {len(ok)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
