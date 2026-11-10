#!/bin/bash
#SBATCH --job-name=xcomet-both
##SBATCH --partition=<your-partition>   # pass -p at submit time (48 GB GPU)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/xcomet-both-%j.out

# XCOMET-XL over the FULL ape config (73,829 rows), in BOTH modes:
#   (a) reference-free  : src + mt          -- how it was run before
#   (b) reference-based : src + mt + ref=post_edit  -- the mode it was TRAINED for
#
# (b) is the point of this job. XCOMET-XL is a reference-based model that has only ever been run
# reference-free here, which is the most likely explanation for its outlier behaviour
# (rho 0.094 against DA on en-hi, versus 0.510 for CometKiwi-XL on the same rows).
#
# Post-edit serves as the reference throughout -- this release has no independent references.

set -euo pipefail
cd "${IQE_REPO:?set IQE_REPO to the repository root}"
mkdir -p logs results/qe_metrics

source cluster/env.sh
activate "$CONDA_IQE"                 # `indicqe`: comet 2.2.7, torch 2.4.1+cu121

# XCOMET-XL is pre-cached under $HF_HOME/hub, so the MODEL resolves offline. The dataset
# parquet is NOT cached, so it must be staged on the login node first (see the companion
# prefetch step) -- compute nodes have no outbound network.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export IQE_PARQUET="$SCRATCH/indicqeape/ape_parquet"   # stage the HF `ape` config {train,validation,test}.parquet here
[ -d "$IQE_PARQUET" ] || { echo "FATAL: $IQE_PARQUET missing -- run the prefetch on the login node first"; exit 1; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python - <<'PY'
import json, os, torch
import pandas as pd
from comet import download_model, load_from_checkpoint

OUT = "results/qe_metrics"
os.makedirs(OUT, exist_ok=True)
SRC = os.environ["IQE_PARQUET"]

# ---- full population, staged from HF on the login node: train + validation + test ------------
df = pd.concat([pd.read_parquet(f"{SRC}/{sp}.parquet")
                for sp in ("train", "validation", "test")], ignore_index=True)
df = df[df.mt.astype(str).str.strip().ne("") & df.post_edit.astype(str).str.strip().ne("")]
print(f"[data] {len(df)} rows over {sorted(df.language_pair.unique())}", flush=True)
assert len(df) == 73829, f"expected 73829 rows, got {len(df)} -- HF release changed, re-reconcile"

model = load_from_checkpoint(download_model("Unbabel/XCOMET-XL"))
gpus = 1 if torch.cuda.is_available() else 0

for mode in ("reffree", "refbased"):
    for pair, sub in df.groupby("language_pair"):
        dst = f"{OUT}/xcomet-xl-{mode}.{pair.replace('-', '')}.jsonl"
        if os.path.exists(dst):
            print(f"[skip] {dst}", flush=True)
            continue
        if mode == "reffree":
            data = [{"src": str(s), "mt": str(m)}
                    for s, m in zip(sub.source, sub.mt)]
        else:
            data = [{"src": str(s), "mt": str(m), "ref": str(r)}
                    for s, m, r in zip(sub.source, sub.mt, sub.post_edit)]
        print(f"[run ] {mode} {pair} n={len(data)}", flush=True)
        out = model.predict(data, batch_size=16, gpus=gpus, progress_bar=False)
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"_meta": {"metric": "xcomet-xl", "mode": mode,
                                           "model": "Unbabel/XCOMET-XL", "pair": pair,
                                           "reference": "post_edit" if mode == "refbased" else None,
                                           "n": len(data)}}, ensure_ascii=False) + "\n")
            for uid, oid, sc, gold in zip(sub.uid, sub.original_id, out.scores, sub.da_mean):
                fh.write(json.dumps({"id": str(uid), "original_id": str(oid), "pair": pair,
                                     "gold_da_mean": None if pd.isna(gold) else float(gold),
                                     "pred_score": float(sc)}, ensure_ascii=False) + "\n")
        print(f"[done] {dst}  mean={sum(out.scores)/len(out.scores):.4f}", flush=True)
PY

# Scratch may be purged on a rolling window -- copy results off immediately.
echo "REMEMBER: rsync results/qe_metrics off scratch before the purge window."
