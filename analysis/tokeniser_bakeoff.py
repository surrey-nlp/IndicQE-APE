#!/usr/bin/env python3
"""tokeniser_bakeoff.py — identify the tokeniser behind the shipped `mt_tokens`.

Every tagged row gives a supervised example: raw `mt` in, `mt_tokens` out (minus the trailing
<EOS>). So we do not have to guess which tokeniser the upstream pipeline used: we can test
candidates against ground truth and pick the one that reproduces it. Whatever wins is the tokeniser
we must use on the post-edit side, because word tags are only meaningful if MT and PE are tokenised
the same way.

Metric: exact sequence match rate, plus mean token-level F1 for the near misses.
"""
from __future__ import annotations
import json, re, sys
import pandas as pd
from huggingface_hub import hf_hub_download

PUNCT_ALL = r"[\.,;:!\?\(\)\[\]\{\}\"'«»„“”‘’…।॥\-–—/]"
# number/date/url-aware: do not split punctuation that sits between digits
PUNCT_SAFE = re.compile(r"(?<!\d)([\.,;:!\?\(\)\[\]\{\}\"'«»„“”‘’…।॥–—])(?!\d)|(?<=\s)([\-/])(?=\s)")


def t_naive(s):
    return str(s).split()


def t_regex_all(s):
    s = re.sub(f"({PUNCT_ALL})", r" \1 ", str(s).strip())
    return [t for t in s.split() if t]


def t_regex_safe(s):
    s = str(s).strip()
    out = []
    for w in s.split():
        # peel leading/trailing punctuation but keep internal (3/4, 29.9.2011, 8%)
        m = re.match(r"^([\"'«„“‘\(\[\{]*)(.*?)([\.\,;:!\?\"'»”’\)\]\}…।॥]*)$", w)
        if not m:
            out.append(w); continue
        lead, core, trail = m.groups()
        out += list(lead) + ([core] if core else []) + list(trail)
    return [t for t in out if t]


def make_moses(lang):
    from sacremoses import MosesTokenizer
    mt = MosesTokenizer(lang=lang)
    return lambda s: mt.tokenize(str(s), escape=False)


def t_indic(s):
    from indicnlp.tokenize import indic_tokenize
    return list(indic_tokenize.trivial_tokenize(str(s)))


def as_list(x):
    if x is None: return None
    if isinstance(x, str):
        try: return json.loads(x)
        except Exception: return x.split()
    return list(x)


def f1(a, b):
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    ov = sum((ca & cb).values())
    if not ov: return 0.0
    p, r = ov / len(a), ov / len(b)
    return 2 * p * r / (p + r)


def main(n_per_pair=150):
    p = hf_hub_download("surrey-nlp/IndicQE-APE", "data/full/train.parquet", repo_type="dataset")
    d = pd.read_parquet(p)
    w = d[d.wtag.notna() & d.mt_tokens.notna()]
    w = w.groupby("language_pair", group_keys=False).head(n_per_pair)
    print(f"sample {len(w)} rows {w.language_pair.value_counts().to_dict()}\n")

    LANG = {"en-hi": "hi", "en-ml": "ml", "en-mr": "mr", "en-ta": "ta",
            "et-en": "en", "ne-en": "en", "si-en": "en"}
    cands = {"naive-split": t_naive, "regex-all": t_regex_all,
             "regex-safe": t_regex_safe, "indicnlp-trivial": t_indic}
    try:
        cands["moses-tgtlang"] = "MOSES"       # built per-row on the target language
    except Exception:
        pass

    res = {k: {"exact": 0, "n": 0, "f1": 0.0} for k in cands}
    per_pair = {k: {} for k in cands}
    moses_cache = {}
    for _, r in w.iterrows():
        gold = [str(t) for t in as_list(r["mt_tokens"])]
        if gold and gold[-1] == "<EOS>": gold = gold[:-1]
        if not gold: continue
        lp = r["language_pair"]
        for name, fn in cands.items():
            if fn == "MOSES":
                lang = LANG.get(lp, "en")
                if lang not in moses_cache:
                    try: moses_cache[lang] = make_moses(lang)
                    except Exception: moses_cache[lang] = make_moses("en")
                got = moses_cache[lang](r["mt"])
            else:
                got = fn(r["mt"])
            s = res[name]; s["n"] += 1
            pp = per_pair[name].setdefault(lp, {"exact": 0, "n": 0})
            pp["n"] += 1
            if got == gold:
                s["exact"] += 1; pp["exact"] += 1
            s["f1"] += f1(got, gold)

    print(f"{'tokeniser':<20}{'exact seq':>11}{'token F1':>10}")
    for name, s in sorted(res.items(), key=lambda kv: -kv[1]["exact"]):
        print(f"{name:<20}{s['exact']/max(s['n'],1):>11.3f}{s['f1']/max(s['n'],1):>10.3f}")

    pairs = sorted({p for v in per_pair.values() for p in v})
    print(f"\nexact-match rate per pair:\n{'tokeniser':<20}" + "".join(f"{p:>9}" for p in pairs))
    for name in res:
        cells = []
        for pr in pairs:
            s = per_pair[name].get(pr)
            cells.append(f"{s['exact']/max(s['n'],1):>9.3f}" if s else f"{'-':>9}")
        print(f"{name:<20}" + "".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 150))
