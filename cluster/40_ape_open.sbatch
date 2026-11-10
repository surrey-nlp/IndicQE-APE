#!/usr/bin/env bash
#SBATCH --job-name=iqe-ape-open
##SBATCH --partition=<your-partition>   # pass -p at submit time (48 GB GPU)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/%x-%A_%a.out
# Open-weight APE on the challenge test: 4 models x 2 shots x 3 PE-pairs (en-hi/mr/ta) = 24.
# Rows without a human post-edit are skipped automatically. 27B bf16 -> use an 80 GB partition.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
activate "$CONDA_IQE"

MODELS=( meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3-14B google/gemma-3-27b-it )
SHOTS=( 0 4 )
PAIRS=( enhi enmr enta )
M=$(( SLURM_ARRAY_TASK_ID / 6 )); S=$(( (SLURM_ARRAY_TASK_ID % 6) / 3 )); P=$(( SLURM_ARRAY_TASK_ID % 3 ))
MODEL="${MODELS[$M]}"; SH="${SHOTS[$S]}"; PAIR="${PAIRS[$P]}"

python -m indicqe_ape.cli ape --provider hf --model "$MODEL" \
  --pair "$PAIR" --subset challenge --split test --shots "$SH" --no-comet \
  --out "$IQE_OUT/ape_open/${MODEL##*/}.s${SH}.${PAIR}.jsonl" \
  --cache-dir "$IQE_CACHE/${MODEL##*/}.s${SH}.${PAIR}"   # per-task cache (parallel array tasks must not share)
# COMET-Kiwi over generated post-edits is added by 10_qe_metrics (2nd pass).
