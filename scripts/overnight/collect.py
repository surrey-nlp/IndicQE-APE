#!/usr/bin/env python3
"""Roll up all overnight run results into one table. Run any time: python collect.py [runs_dir]"""
import json, glob, os, sys
runs = sys.argv[1] if len(sys.argv) > 1 else "runs"
print(f"{'job':16s} {'status':9s} summary")
print("-" * 70)
for d in sorted(glob.glob(f"{runs}/*/")):
    jid = os.path.basename(d.rstrip("/"))
    stf = os.path.join(d, "status")
    st = open(stf).read().split()[0] if os.path.exists(stf) else "-"
    mf = os.path.join(d, "metrics.json")
    summary = ""
    if os.path.exists(mf):
        try:
            j = json.load(open(mf))
            if "macro_P_rho_tau" in j:
                summary = f"macro P/rho/tau={j['macro_P_rho_tau']} xling={j.get('cross_lingual_r')}"
            else:  # matra_variants-style per-pair means
                summary = " ".join(f"{k}:{v.get('matra')}/{v.get('matra-norm')}"
                                   for k, v in list(j.items())[:3] if isinstance(v, dict))
                summary = "matra/matra-norm per pair -> " + summary + " ..."
        except Exception as e:
            summary = f"(metrics parse err: {e})"
    print(f"{jid:16s} {st:9s} {summary}")
print(f"\nqueue log tail:  (runs/queue.log)")
if os.path.exists(f"{runs}/queue.log"):
    print("".join(open(f"{runs}/queue.log").readlines()[-6:]))
