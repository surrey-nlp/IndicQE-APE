#!/bin/bash
#SBATCH --job-name=r0-sev
##SBATCH --partition=<your-partition>   # pass -p at submit time (48 GB GPU)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/r0-sev-%j.out

# R0 GPU pass — pure scoring service. The cluster does trained-metric inference and nothing else:
# no MATRA, no set construction, no analysis. Those run locally, where the metric code is.
#
# IN : results/severity/r0_gpu_input.jsonl    {id, src, mt, ref}                    (scp up)
# OUT: results/severity/r0_gpu_scores.jsonl   {id, <one field per metric/mode>}     (scp down)
#
# Joined back on `id` locally. The GPU side never sees the MATRA columns or the MQM prior, so it
# cannot accidentally contaminate the comparison, and the file that travels is 5 MB not 7.4 MB.
#
#   XCOMET-XL      reference-free AND reference-based (clean base as reference)
#   CometKiwi-XL   reference-free only — it is a QE model with no reference input
#
# CometKiwi-DA deliberately excluded: correlates 0.931 with CometKiwi-XL, adds no coverage.

set -euo pipefail
cd "${IQE_REPO:?set IQE_REPO to the repository root}"
mkdir -p logs results/severity

source cluster/env.sh
activate "$CONDA_IQE"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

IN=results/severity/r0_gpu_input.jsonl
[ -s "$IN" ] || { echo "FATAL: $IN missing/empty — scp it up first"; exit 1; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "input: $IN ($(wc -l < "$IN") records)"

python - <<'PY'
import json, os, torch
from comet import download_model, load_from_checkpoint

IN, OUT = "results/severity/r0_gpu_input.jsonl", "results/severity/r0_gpu_scores.jsonl"
rows = [json.loads(l) for l in open(IN, encoding="utf-8")]
print(f"[data] {len(rows)} records", flush=True)
assert all(r.get("id") for r in rows), "every record needs an id — the join depends on it"
gpus = 1 if torch.cuda.is_available() else 0

# resume: keep anything already scored in a previous run of this job
out = {}
if os.path.exists(OUT):
    for l in open(OUT, encoding="utf-8"):
        r = json.loads(l)
        out[r["id"]] = r
for r in rows:
    out.setdefault(r["id"], {"id": r["id"]})

JOBS = [
    ("xcomet_xl_reffree",  "Unbabel/XCOMET-XL",             False),
    ("xcomet_xl_refbased", "Unbabel/XCOMET-XL",             True),
    ("cometkiwi_xl",       "Unbabel/wmt23-cometkiwi-da-xl", False),
]

for col, repo, use_ref in JOBS:
    if all(col in out[r["id"]] for r in rows):
        print(f"[skip] {col} (already complete)", flush=True)
        continue
    print(f"[load] {repo} ref={use_ref} -> {col}", flush=True)
    model = load_from_checkpoint(download_model(repo))
    data = [({"src": r["src"], "mt": r["mt"], "ref": r["ref"]} if use_ref
             else {"src": r["src"], "mt": r["mt"]}) for r in rows]
    pred = model.predict(data, batch_size=16, gpus=gpus, progress_bar=False)
    for r, s in zip(rows, pred.scores):
        out[r["id"]][col] = float(s)
    print(f"[done] {col} mean={sum(pred.scores)/len(pred.scores):.4f}", flush=True)
    del model
    if gpus:
        torch.cuda.empty_cache()
    # write after each metric so a later OOM does not lose earlier work
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(out[r["id"]], ensure_ascii=False) + "\n")

print(f"[write] {OUT} ({len(rows)} records)", flush=True)
PY

echo
echo "DONE. Bring this back:  results/severity/r0_gpu_scores.jsonl"
echo "Scratch may be purged on a rolling window — copy it off now."
