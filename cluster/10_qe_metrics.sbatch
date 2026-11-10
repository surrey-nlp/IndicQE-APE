#!/usr/bin/env bash
#SBATCH --job-name=iqe-metrics
##SBATCH --partition=<your-partition>   # pass -p at submit time (80 GB GPU)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Trained-metric reference-free QE over the challenge test, via the COMET library (card-verified):
#   cometkiwi-da (wmt22-cometkiwi-da), cometkiwi-xl (wmt23-cometkiwi-da-xl), xcomet-xl (XCOMET-XL).
# NO Tower. All gated — accept licences + login + prefetch on the login node first (see 01_prefetch).
# Resumable (skips done metric/pair). Output -> $IQE_OUT/qe_metrics (picked up by analyse_challenge).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
activate "$CONDA_IQE"                                  # unbabel-comet lives in the indicqe env

python scripts/run_comet_metrics.py --out "$IQE_OUT/qe_metrics" --gpus 1 --batch-size 16

# 2nd pass: CometKiwi over any generated post-edits (APE quality, reference-free)
if compgen -G "$IQE_OUT/ape_open/*.jsonl" >/dev/null; then
  python scripts/run_comet_metrics.py --ape "$IQE_OUT/ape_open" --out "$IQE_OUT/ape_metrics" --gpus 1
fi
echo "metrics -> $IQE_OUT/qe_metrics (+ ape_metrics)"
