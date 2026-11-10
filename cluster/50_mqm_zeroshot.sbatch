#!/usr/bin/env bash
#SBATCH --job-name=iqe-mqm-zs
##SBATCH --partition=<your-partition>   # pass -p at submit time (48 GB GPU)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#
# Score the en-hi MQM layer with the three pretrained COMET metrics. Nothing has ever been
# scored against `mqm_score`: the mqm config was released but was absent from data.py's
# _CONFIG_DIR, so it could not be loaded. This is the first MQM-gold evaluation in the project.
#
# Population: data/tasks/mqm, test split = 2,639 segments, of which 2,016 are the WMT24
# general-MT batch that carries MQM and NO direct assessment. Those 2,016 have never been seen
# by any system here. The remaining 623 are EAMT segments. Label is `mqm_score` (0-100, higher
# is better), selected automatically by run_zeroshot_comet.py when --config mqm.
#
# PARTITION. Any 48 GB GPU (e.g. L40S, RTX A6000) runs the 3.5B XCOMET-XL predict comfortably
# in fp16; a 24 GB GPU is fine for predict (not for full fine-tune). Pass it with -p.
#
# Resumable: run_zeroshot_comet.py skips a (model, config, split) whose output JSONL already
# exists, so re-submitting after a pre-emption continues rather than repeating.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh
activate "$CONDA_IQE"

export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset IQE_QE_MAXTOK || true          # irrelevant to COMET, unset for consistency

OUTDIR="$IQE_OUT/zeroshot_mqm"
mkdir -p "$OUTDIR" logs

echo ">> MQM zero-shot | node=$(hostname) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python experiments/comet_qe/scripts/run_zeroshot_comet.py \
  --config mqm --split test \
  --models Unbabel/wmt22-cometkiwi-da Unbabel/wmt23-cometkiwi-da-xl Unbabel/XCOMET-XL \
  --gpus 1 --batch-size 16 --out-dir "$OUTDIR"

echo ">> scoring metrics against MQM gold"
python scripts/mqm_baselines.py \
  --scores "$OUTDIR"/mqm-test-*.jsonl \
  --out results/analysis/mqm_baselines_with_metrics.md

echo ">> done. Copy off scratch before it is purged, e.g. to a shared results directory:"
echo "   cp -r $OUTDIR ${IQE_SHARED:-<shared-results-dir>}/"
