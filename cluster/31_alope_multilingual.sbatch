#!/usr/bin/env bash
#SBATCH --job-name=iqe-alope-ml
##SBATCH --partition=<your-partition>   # pass -p at submit time (48 GB GPU)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
# Multilingual ALOPE: 1 model per strategy on the POOLED curated train (config `all`),
# eval per pair on the challenge test (incl. zero-shot transfer to X->En).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
DATADIR="$IQE_OUT/alope_in"; OUT="$IQE_OUT/alope_ml"; mkdir -p "$DATADIR" "$OUT"
STRATS=( single-layer dynamic-weighting multi-head ft-no-lora )
STRAT="${STRATS[$SLURM_ARRAY_TASK_ID]}"

activate "$CONDA_IQE"
python scripts/export_jsonl.py --config all --split train --require-da --out "$DATADIR/all.train.jsonl"
python scripts/export_jsonl.py --config all --split validation --require-da --out "$DATADIR/all.dev.jsonl"
for PAIR in enhi enmr enta ente engu enml eten neen sien; do
  python scripts/export_jsonl.py --config challenge --pair "$PAIR" --require-da --out "$DATADIR/$PAIR.challenge.jsonl"
done

activate "$CONDA_ALOPE"; cd "$ALOPE_REPO"
export ALOPE_BACKBONE="${ALOPE_BACKBONE:-meta-llama/Llama-3.2-3B-Instruct}"   # or CoRover/BharatGPT-3B-Indic
export ALOPE_PAIR="multilingual" ALOPE_LAYER="-7"
export ALOPE_TRAIN_FILE="$DATADIR/all.train.jsonl" ALOPE_DEV_FILE="$DATADIR/all.dev.jsonl"
export ALOPE_OUT="$OUT/$STRAT.multilingual"
case "$STRAT" in
  single-layer)      python single-layer-regression_train.py ;;
  dynamic-weighting) python Dynamic_weighting-train.py ;;
  multi-head)        python Multi-head_regression_train.py ;;
  ft-no-lora)        python FT-without_LORA.PY ;;
esac
for PAIR in enhi enmr enta ente engu enml eten neen sien; do
  export ALOPE_PAIR="$PAIR" ALOPE_TEST_FILE="$DATADIR/$PAIR.challenge.jsonl" ALOPE_OUT="$OUT/$STRAT.multilingual.$PAIR"
  case "$STRAT" in
    single-layer)      python single-layer-regression_inference.py ;;
    dynamic-weighting) python Dynamic_weighting-inference.py ;;
    multi-head)        python Multi-head_regression_inference.py ;;
    ft-no-lora)        python FT-without_LORA.PY ;;
  esac
done
echo "multilingual ALOPE $STRAT -> $OUT"
