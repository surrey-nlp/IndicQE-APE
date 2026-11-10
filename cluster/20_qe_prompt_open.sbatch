#!/usr/bin/env bash
#SBATCH --job-name=iqe-qe-open
##SBATCH --partition=<your-partition>   # pass -p at submit time (48 GB GPU)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-44
#SBATCH --output=logs/%x-%A_%a.out
# Broad ZERO-SHOT open-weight QE sweep over cluster/models_open.txt (5 active models x 8 pairs
# = 40; set --array to N_models*8-1 if you edit the list). Both GEMBA prompts run per item.
# Row = model_index*8 + pair_index. Submit each tier with the matching provider/partition:
#   3B Indic     (rows 0-15):  IQE_PROVIDER=hf   ... -p "$PART_ITER" --array=0-15
#   sarvam-m 24B (rows 16-23): IQE_PROVIDER=vllm ... -p "$PART_BIG" --gres=gpu:1 --array=16-23
#   general 27-31B(rows 24-39): IQE_PROVIDER=vllm ... -p "$PART_BIG" --gres=gpu:1 --array=24-39
# (most a100 cards are 80GB -> TP=1 fits each 24-32B model; the orchestrator run_all_a100.sbatch
#  runs all of this sequentially on one a100.)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh

mapfile -t MODELS < <(grep -vE '^\s*#|^\s*$' cluster/models_open.txt | awk '{print $1}')
PAIRS=( enhi enmr enta ente engu enml eten neen sien )
N=${#PAIRS[@]}
M=$(( SLURM_ARRAY_TASK_ID / N )); P=$(( SLURM_ARRAY_TASK_ID % N ))
MODEL="${MODELS[$M]}"; PAIR="${PAIRS[$P]}"
PROVIDER="${IQE_PROVIDER:-hf}"     # hf = 3B text models in the `indicqe` env (default, no vLLM).
SHOTS="${IQE_SHOTS:-0}"            # zero-shot sweep by default. Both GEMBA prompts run per item.

# Pick the env that matches the provider: vLLM (27-31B/multimodal/quantized) lives in its OWN
# cu12 env (CONDA_VLLM), separate from the indicqe env so it can't clobber torch 2.4.1+cu121.
if [[ "$PROVIDER" == "vllm" ]]; then ENV="$CONDA_VLLM"; else ENV="$CONDA_IQE"; fi
activate "$ENV"

echo ">> $MODEL  $PAIR  (provider=$PROVIDER env=$ENV shots=$SHOTS TP=${VLLM_TP:-1})"
python -m indicqe_ape.cli qe --provider "$PROVIDER" --model "$MODEL" \
  --pair "$PAIR" --subset challenge --split test --shots "$SHOTS" \
  --out "$IQE_OUT/qe_open/${MODEL##*/}.s${SHOTS}.${PAIR}.jsonl" \
  --cache-dir "$IQE_CACHE/${MODEL##*/}.s${SHOTS}.${PAIR}"   # per-task cache: parallel array tasks must NOT share a cache file (corrupts it)
