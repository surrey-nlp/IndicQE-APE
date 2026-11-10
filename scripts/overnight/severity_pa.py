#!/usr/bin/env python3
"""Char-matched pairwise accuracy (Eq. pacm) for char-TER / MATRA / MATRA++ / MetricX-24, all on the
identical injected severity set. Reads metricx_sev_output.jsonl (MetricX preserves the carried fields
mqm/base_id/char_ter/matra/matrapp and adds `prediction`, its error score). For every same-base pair of
edits with different MQM and |Δchar-TER| <= 3, count how often the higher-MQM edit gets the higher
(more-severe) metric value. 0.5 = chance. MetricX prediction is an error (higher = worse = more severe),
so it uses the same direction as the edit metrics."""
import json, os, sys, collections

WORK = os.environ.get("IQE_WORK") or sys.exit("set IQE_WORK to the overnight working directory (holds code/, runs/ and the *_output.jsonl files)")
SRC = f"{WORK}/metricx_sev_output.jsonl"
rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
METRICS = collections.OrderedDict([
    ("char-TER", "char_ter"), ("MATRA", "matra"), ("MATRA++", "matrapp"),
    ("MetricX-24 (ref)", "prediction"),
])


def pa(rows, key):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["base_id"]].append(r)
    corr = tot = 0
    for rs in by.values():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                if a["mqm"] == b["mqm"]:
                    continue
                if a.get(key) is None or b.get(key) is None:
                    continue
                if abs(a["char_ter"] - b["char_ter"]) > 3:
                    continue
                hi, lo = (a, b) if a["mqm"] > b["mqm"] else (b, a)
                if hi[key] == lo[key]:
                    continue
                tot += 1
                corr += int(hi[key] - lo[key] > 0)
    return (corr / tot if tot else float("nan")), tot


print(f"char-matched pairwise accuracy on {len(rows)} injected pairs (0.5 = chance):")
for disp, key in METRICS.items():
    p, n = pa(rows, key)
    print(f"  {disp:18s} PA = {p:.3f}  (n={n})")
