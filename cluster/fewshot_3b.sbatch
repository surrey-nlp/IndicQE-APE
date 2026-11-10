#!/usr/bin/env bash
#SBATCH --job-name=iqe-fs3b
##SBATCH --partition=<your-partition>   # pass -p at submit time (24 GB GPU)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
# Completes the 4-shot QE matrix for the 3B judges (Llama, BharatGPT) that the main orchestrator
# (DO_4SHOT only covered tiny-aya + sarvam-m) left out. hf provider, fits a 24GB GPU.
# Resumable: skips any (model,pair) that already has a _summary. Both GEMBA prompts run per item.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh; activate "$CONDA_IQE"
PAIRS=(enhi enmr enta ente engu enml eten neen sien)
MODELS=(meta-llama/Llama-3.2-3B-Instruct CoRover/BharatGPT-3B-Indic)
for m in "${MODELS[@]}"; do
  for p in "${PAIRS[@]}"; do
    out="$IQE_OUT/qe_open/${m##*/}.s4.${p}.jsonl"
    if grep -q '"_summary"' "$out" 2>/dev/null; then echo "skip $(basename "$out")"; continue; fi
    echo ">>> $(date +%H:%M) 4-shot QE ${m##*/} $p"
    python -m indicqe_ape.cli qe --provider hf --model "$m" --pair "$p" --subset challenge \
      --shots 4 --out "$out" --cache-dir "$IQE_CACHE/${m##*/}.s4.${p}" || echo "  !! FAIL ${m##*/} $p"
  done
done
echo "######## fewshot-3b DONE $(date) ########"
