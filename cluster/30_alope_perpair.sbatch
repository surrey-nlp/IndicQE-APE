#!/usr/bin/env bash
#SBATCH --job-name=iqe-alope-pp
##SBATCH --partition=<your-partition>   # pass -p at submit time (48 GB GPU)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x-%A_%a.out
# Per-pair ALOPE: 4 strategies x 5 curated En->Indic pairs = 20. LLaMA-3.2-3B 4-bit QLoRA,
# TL-7. Trains on each pair's curated train; evals on the challenge test. Fits 24GB/48GB.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh

DATADIR="$IQE_OUT/alope_in"; OUT="$IQE_OUT/alope_pp"; mkdir -p "$DATADIR" "$OUT"
STRATS=( single-layer dynamic-weighting multi-head ft-no-lora )
PAIRS=( enhi enmr enta ente engu )
S=$(( SLURM_ARRAY_TASK_ID / 5 )); P=$(( SLURM_ARRAY_TASK_ID % 5 ))
STRAT="${STRATS[$S]}"; PAIR="${PAIRS[$P]}"

activate "$CONDA_IQE"
for SP in train validation test; do
  python scripts/export_jsonl.py --config "$PAIR" --split "$SP" --require-da --out "$DATADIR/$PAIR.$SP.jsonl"
done
python scripts/export_jsonl.py --config challenge --pair "$PAIR" --require-da --out "$DATADIR/$PAIR.challenge.jsonl"

activate "$CONDA_ALOPE"; cd "$ALOPE_REPO"
# backbone: default LLaMA-3.2-3B; CoRover/BharatGPT-3B-Indic (llama arch) is an Indic-specialist
# alternative — pick whichever the zero-shot sweep selected.
export ALOPE_BACKBONE="${ALOPE_BACKBONE:-meta-llama/Llama-3.2-3B-Instruct}"
export ALOPE_PAIR="$PAIR" ALOPE_LAYER="-7"
export ALOPE_TRAIN_FILE="$DATADIR/$PAIR.train.jsonl"
export ALOPE_DEV_FILE="$DATADIR/$PAIR.validation.jsonl"
export ALOPE_TEST_FILE="$DATADIR/$PAIR.challenge.jsonl"
export ALOPE_OUT="$OUT/$STRAT.$PAIR"
# VERIFY: ALOPE scripts read source/mt/da_mean via these env vars (or adapt).
case "$STRAT" in
  single-layer)      python single-layer-regression_train.py && python single-layer-regression_inference.py ;;
  dynamic-weighting) python Dynamic_weighting-train.py && python Dynamic_weighting-inference.py ;;
  multi-head)        python Multi-head_regression_train.py && python Multi-head_regression_inference.py ;;
  ft-no-lora)        python FT-without_LORA.PY ;;
esac
echo "ALOPE $STRAT/$PAIR -> $ALOPE_OUT"
