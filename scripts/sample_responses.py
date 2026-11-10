#!/usr/bin/env python3
"""Eyeball QE quality: for a sample of a model's records, show source + MT + gold DA next to the
model's predicted score and its raw text. Joins the qe_open/qe_closed jsonl (which stores id=uid,
gold, pred, raw) back to the challenge source/MT by uid.

Usage:
  python scripts/sample_responses.py results/qe_open/sarvam-m.s0.enhi.jsonl 12
  python scripts/sample_responses.py $IQE_OUT/qe_open/sarvam-m.s0.enhi.jsonl 12   (on the server)
"""
import json, os, re, sys, random

f = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

recs = []
for ln in open(f):
    if not ln.strip() or '"_meta"' in ln or '"_summary"' in ln:
        continue
    try:
        r = json.loads(ln)
    except Exception:
        continue
    if isinstance(r.get("raw"), dict):
        recs.append(r)
if not recs:
    print("no scorable records yet"); sys.exit(0)

# join to source/MT by uid, if the dataset is reachable
src_mt = {}
m = re.search(r"\.(en[a-z][a-z]|[a-z][a-z]en)\.jsonl$", os.path.basename(f))
pair = m.group(1) if m else None
try:
    from indicqe_ape.data import DataLoader
    df = DataLoader().load("challenge", pair=pair)
    for _, row in df.iterrows():
        src_mt[str(row["uid"])] = (str(row["source"]), str(row["mt"]))
except Exception as e:
    print(f"(source/MT join unavailable: {e})\n")

random.seed(1)
samp = random.sample(recs, min(n, len(recs)))
for r in samp:
    p, raw = r.get("pred") or {}, r.get("raw") or {}
    s, mt = src_mt.get(str(r.get("id")), ("", ""))
    print("=" * 100)
    print(f"route={r.get('route','')}  gold_DA={r.get('gold_da_mean')}   pred: DA={p.get('gemba-da')}  SQM={p.get('gemba-sqm')}")
    if s:
        print(f"  SRC: {s[:140]}")
        print(f"  MT : {mt[:140]}")
    print(f"  raw[DA] : {str(raw.get('gemba-da'))[:120]!r}")
