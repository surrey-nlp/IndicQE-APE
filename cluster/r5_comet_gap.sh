#!/bin/bash
# R5 (workstation) — COMET-family reference-free QE on the two pairs missing from the full population.
#
#   en-gu 9,109 + en-te 13,157 = 22,266 rows, all DA-bearing, no post-edits.
#   Reference-free ONLY: these pairs have no reference, so XCOMET-XL ref-based is not applicable.
#
# Env: point PY at a python with unbabel-comet 2.2.7, torchmetrics 0.10.3, torch 2.4.1+cu121.
# NOT unbabel-comet 1.1.2, which fails on a torchmetrics import.
#
# Written for a two-GPU workstation; it uses GPU 1 by default. Set CUDA_VISIBLE_DEVICES as needed.
# Checkpoints are fetched into the HF cache on first use.
#
#   IQE_REPO=/path/to/repo PY=/path/to/env/bin/python bash cluster/r5_comet_gap.sh
set -u
cd "${IQE_REPO:?set IQE_REPO to the repository root}"
PY="${PY:-python3}"
IN="$PWD/r5_qeda_gap_input.jsonl"
OUT="$PWD/results_r5"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"

[ -f "$IN" ] || { echo "MISSING INPUT: $IN -- build it with cluster/r5_build_qeda_gap.py and copy it here first"; exit 1; }
echo "=== R5 COMET gap · $(date) · $(wc -l < "$IN") rows ==="

$PY - "$IN" "$OUT" <<'PY'
import json, sys, os
from comet import download_model, load_from_checkpoint

IN, OUT = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(IN, encoding="utf-8")]
print(f"[data] {len(rows):,} rows", flush=True)

# reference-free models only; XCOMET-XL is reference-based but is also run reference-free here,
# exactly as in the existing full-population files, so the columns stay comparable.
JOBS = [("cometkiwi-xl-reffree", "Unbabel/wmt23-cometkiwi-da-xl", 8),
        ("xcomet-xl-reffree",    "Unbabel/XCOMET-XL",             8)]

for tag, repo, bs in JOBS:
    done = {p: os.path.join(OUT, f"{tag}.{p}.jsonl") for p in {r["pair"] for r in rows}}
    if all(os.path.exists(p) for p in done.values()):
        print(f"[skip] {tag} already complete", flush=True); continue
    print(f"[load] {tag} <- {repo}", flush=True)
    model = load_from_checkpoint(download_model(repo))
    for pair in sorted({r["pair"] for r in rows}):
        outp = done[pair]
        if os.path.exists(outp):
            print(f"  [skip] {tag}.{pair}", flush=True); continue
        sub = [r for r in rows if r["pair"] == pair]
        data = [{"src": r["source"], "mt": r["mt"]} for r in sub]
        pred = model.predict(data, batch_size=bs, gpus=1, progress_bar=False)
        with open(outp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"_meta": {"metric": tag, "model": repo, "pair": pair,
                                           "mode": "reference-free", "n": len(sub)}}) + "\n")
            for r, s in zip(sub, pred.scores):
                fh.write(json.dumps({"id": r["id"], "pair": r["pair"],
                                     "gold_da_mean": r["gold_da_mean"],
                                     "pred_score": float(s)}, ensure_ascii=False) + "\n")
        print(f"  [done] {tag}.{pair}  n={len(sub):,}", flush=True)
    del model
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
print("[ALL DONE]", flush=True)
PY
echo "=== exit=$? $(date) ==="
