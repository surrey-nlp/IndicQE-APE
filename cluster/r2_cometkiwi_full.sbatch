#!/bin/bash
#SBATCH --job-name=r2-ckiwi
##SBATCH --partition=<your-partition>   # pass -p at submit time (48 GB GPU)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/r2-ckiwi-%j.out

# R2 — CometKiwi-XL over the FULL ape config (73,829 rows), reference-free.
#
# CometKiwi-XL is a QE model: it takes source + hypothesis and has no reference input, so unlike
# XCOMET-XL there is only ONE mode to run. It is the strongest DA baseline measured
# (rho 0.510 on en-hi vs XCOMET-XL's 0.094), which is exactly why it needs to be on the same
# full population as everything else rather than the old partial sample.
#
# CometKiwi-DA is NOT run: it correlates 0.931 with this model, so reporting both would inflate
# apparent baseline coverage without adding information.
#
# Can run concurrently with R1 (XCOMET-XL) — separate job, separate GPU.

set -euo pipefail
cd "${IQE_REPO:?set IQE_REPO to the repository root}"
mkdir -p logs results/qe_metrics

source cluster/env.sh
activate "$CONDA_IQE"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export IQE_PARQUET="$SCRATCH/indicqeape/ape_parquet"
[ -d "$IQE_PARQUET" ] || { echo "FATAL: $IQE_PARQUET missing"; exit 1; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python - <<'PY'
import json, os, torch
import pandas as pd
from comet import download_model, load_from_checkpoint

OUT, SRC = "results/qe_metrics", os.environ["IQE_PARQUET"]
os.makedirs(OUT, exist_ok=True)

df = pd.concat([pd.read_parquet(f"{SRC}/{sp}.parquet")
                for sp in ("train", "validation", "test")], ignore_index=True)
df = df[df.mt.astype(str).str.strip().ne("") & df.post_edit.astype(str).str.strip().ne("")]
print(f"[data] {len(df)} rows over {sorted(df.language_pair.unique())}", flush=True)
assert len(df) == 73829, f"expected 73829 rows, got {len(df)} -- HF release changed, re-reconcile"

model = load_from_checkpoint(download_model("Unbabel/wmt23-cometkiwi-da-xl"))
gpus = 1 if torch.cuda.is_available() else 0

for pair, sub in df.groupby("language_pair"):
    dst = f"{OUT}/cometkiwi-xl-reffree.{pair.replace('-', '')}.jsonl"
    if os.path.exists(dst):
        print(f"[skip] {dst}", flush=True)
        continue
    data = [{"src": str(s), "mt": str(m)} for s, m in zip(sub.source, sub.mt)]
    print(f"[run ] {pair} n={len(data)}", flush=True)
    out = model.predict(data, batch_size=16, gpus=gpus, progress_bar=False)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_meta": {"metric": "cometkiwi-xl", "mode": "reffree",
                                       "model": "Unbabel/wmt23-cometkiwi-da-xl",
                                       "pair": pair, "reference": None,
                                       "n": len(data)}}, ensure_ascii=False) + "\n")
        for uid, oid, sc, gold in zip(sub.uid, sub.original_id, out.scores, sub.da_mean):
            fh.write(json.dumps({"id": str(uid), "original_id": str(oid), "pair": pair,
                                 "gold_da_mean": None if pd.isna(gold) else float(gold),
                                 "pred_score": float(sc)}, ensure_ascii=False) + "\n")
    print(f"[done] {dst} mean={sum(out.scores)/len(out.scores):.4f}", flush=True)
PY

echo "DONE. Bring back: results/qe_metrics/cometkiwi-xl-reffree.*.jsonl"
echo "Scratch may be purged on a rolling window — copy off now."
