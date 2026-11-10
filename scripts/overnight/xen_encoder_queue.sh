#!/bin/bash
# X->En QE scoring for the encoders on GPU1 (single card) — feeds the full-set cross-lingual recompute.
set -e
WORK="${IQE_WORK:?set IQE_WORK to the overnight working directory (holds code/ = this repo, runs/, xen_qeda.parquet)}"
cd "$WORK/code"
# Activate an env with unbabel-comet 2.2.7 if CONDA_ENV is set; otherwise use the python on PATH.
if [ -n "${CONDA_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
IN="$WORK/xen_qeda.parquet"
for spec in "cometkiwi_da:Unbabel/wmt22-cometkiwi-da" \
            "cometkiwi_xl:Unbabel/wmt23-cometkiwi-da-xl" \
            "xcomet_xl:Unbabel/XCOMET-XL"; do
  name=${spec%%:*}; model=${spec##*:}
  echo "===== START $name ($model) $(date) gpu1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1) ====="
  PYTHONPATH=$PWD python scripts/overnight/comet_qe.py "$model" "$IN" "$WORK/runs/${name}_xen" 1 2>&1 | grep -vE "Predicting DataLoader|it/s\]" | tail -20
  echo "===== DONE $name $(date) ====="
done
echo "ALL_XEN_ENCODERS_DONE $(date)"
