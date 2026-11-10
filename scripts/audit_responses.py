#!/usr/bin/env python3
"""Audit LLM-judge QE responses across every model.

For each model/template it answers: did the model actually EMIT a usable score, is the parser
extracting the RIGHT number, and how often does it fail / hallucinate a number out of prose
(e.g. a translation). Reads the stored `raw` text (no model re-run, no API spend).

Usage:  python scripts/audit_responses.py [results_dir]   (default: results)
"""
import json, glob, os, re, sys, collections

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
NUM = re.compile(r"-?\d+(?:\.\d+)?")

# ---- current (shipped) parser: first number in the string, clamped 0..100
def cur_parse(resp):
    if not resp:
        return None
    m = NUM.search(resp)
    if not m:
        return None
    try:
        return max(0.0, min(100.0, float(m.group(0))))
    except ValueError:
        return None

# ---- hardened parser candidate (see notes below)
_SCORE_CUE = re.compile(r"(?:score|rating|quality|qe|da)\D{0,12}(-?\d+(?:\.\d+)?)", re.I)
_FRAC = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*(\d+)")
_TRANSLATION_CUE = re.compile(r"here'?s the translation|translation:|^(?:the )?translat", re.I)
def hard_parse(resp):
    if not resp:
        return None
    s = resp.strip()
    # reject obvious non-answers (model translated instead of scoring) UNLESS a clear score cue exists
    if _TRANSLATION_CUE.search(s) and not _SCORE_CUE.search(s):
        return None
    # x/100 or x/5 style -> normalise to 0..100
    fr = _FRAC.search(s)
    if fr:
        num, den = float(fr.group(1)), float(fr.group(2))
        if den in (5.0, 10.0, 100.0) and num <= den:
            return max(0.0, min(100.0, num / den * 100.0))
    # explicit "score: N"
    cue = _SCORE_CUE.search(s)
    if cue:
        try:
            return max(0.0, min(100.0, float(cue.group(1))))
        except ValueError:
            pass
    # else first number, but ONLY if the response is short / number-led (avoid prose digits)
    if len(s) <= 25 or re.match(r"^\s*-?\d", s):
        m = NUM.search(s)
        if m:
            try:
                return max(0.0, min(100.0, float(m.group(0))))
            except ValueError:
                return None
    return None

def classify(resp):
    if resp is None:
        return "missing"
    s = resp.strip()
    if not s:
        return "empty"
    nums = NUM.findall(s)
    if _TRANSLATION_CUE.search(s):
        return "translation"
    if not nums:
        return "no_number"
    if re.fullmatch(r"-?\d+(?:\.\d+)?\s*[.%]?", s):
        return "number_only"
    if len(s) <= 25:
        return "short_w_number"
    return "number_in_prose"

def model_of(base):
    return base.split(".s")[0]

files = sorted(glob.glob(os.path.join(ROOT, "qe_open", "*.jsonl"))) + \
        sorted(glob.glob(os.path.join(ROOT, "qe_closed", "*.jsonl")))

agg = collections.defaultdict(lambda: collections.Counter())
print(f"# Auditing {len(files)} QE response files under {ROOT}\n")
for f in files:
    base = os.path.basename(f).replace(".jsonl", "")
    recs = []
    for ln in open(f):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if "_meta" in r or "_summary" in r:
            continue
        recs.append(r)
    if not recs:
        continue
    tmpls = set()
    for r in recs:
        raw = r.get("raw")
        if isinstance(raw, dict):
            tmpls.update(raw.keys())
    if not tmpls:
        tmpls = {"_single"}
    for t in sorted(tmpls):
        cats = collections.Counter()
        n_cur = n_hard = n_disagree = 0
        samples = []
        for r in recs:
            raw = r.get("raw")
            rv = raw.get(t) if isinstance(raw, dict) else raw
            cats[classify(rv)] += 1
            pc, ph = cur_parse(rv), hard_parse(rv)
            n_cur += pc is not None
            n_hard += ph is not None
            if (pc is None) != (ph is None) or (pc is not None and ph is not None and abs(pc - ph) > 1e-6):
                n_disagree += 1
                if len(samples) < 5:
                    samples.append((classify(rv), pc, ph, (rv or "")[:90]))
        m = model_of(base)
        agg[(m, t)]["n"] += len(recs)
        agg[(m, t)]["cur"] += n_cur
        agg[(m, t)]["hard"] += n_hard
        agg[(m, t)]["disagree"] += n_disagree
        for c, v in cats.items():
            agg[(m, t)][f"cat:{c}"] += v
        if samples and t == "gemba-da":
            print(f"--- {base} [{t}] cur->hard disagreements ---")
            for c, pc, ph, s in samples:
                print(f"    {c:16} cur={pc} hard={ph}  raw={s!r}")

print("\n================ PER-MODEL SUMMARY (gemba-da) ================")
print(f"{'model':<26}{'n':>7}{'cur%':>7}{'hard%':>7}{'CHANGED':>9}{'transl':>8}{'noNum':>7}{'prose':>7}")
for (m, t), c in sorted(agg.items()):
    if t != "gemba-da":
        continue
    n = c["n"] or 1
    print(f"{m:<26}{c['n']:>7}{100*c['cur']/n:>6.0f}%{100*c['hard']/n:>6.0f}%"
          f"{c['disagree']:>6}({100*c['disagree']/n:>3.0f}%){c['cat:translation']:>7}{c['cat:no_number']:>7}{c['cat:number_in_prose']:>7}")
print("\n(cur% = shipped parser hit-rate; hard% = hardened parser; transl/noNum/prose/numOnly = raw-response categories)")
