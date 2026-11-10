#!/usr/bin/env bash
#SBATCH --job-name=iqe-qe-rerun
##SBATCH --partition=<your-partition>   # pass -p at submit time (48 GB GPU)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-89
#SBATCH --output=logs/%x-%A_%a.out
#
# Full rerun of the open-weight prompted QE sweep, fixing the two configuration faults that
# produced the parse-rate collapse and the missing pair:
#
#   1. max_tokens. The original runs set IQE_QE_MAXTOK=8. Models that begin their reply with any
#      text are cut off before the score. Measured: BharatGPT-3B en-ta four-shot parsed 0.22, and
#      its unparsed outputs are 3-8 characters of Tamil; tiny-aya si-en parsed 0.31 with outputs cut
#      mid-sentence. The one run that used 512 had zero unparsed outputs. This script leaves
#      IQE_QE_MAXTOK unset so the code default of 512 applies.
#   2. en-ml. The PAIRS array in 20_qe_prompt_open.sbatch has eight pairs and omits enml, which is
#      why no four-shot en-ml run exists. PAIRS below has all nine.
#
# Grid: N_MODELS x 9 pairs x 2 shot settings.
#   row = model_index*18 + pair_index*2 + shot_index
# Both GEMBA templates run per item in one call, so one task is one pass over the pair.
#
# ARRAY SIZE. models_open.txt currently has FIVE uncommented ids, so the grid is 5*18 = 90 tasks
# (rows 0-89). The previous header said 6 models / 0-107; rows 90-107 index MODELS[5], which is
# unbound, and under `set -euo pipefail` those tasks abort immediately. The script now asserts
# the array id is in range rather than failing obscurely. Re-check after editing models_open.txt:
#   printf '%s\n' $(( $(grep -vcE '^\s*#|^\s*$' cluster/models_open.txt) * 18 - 1 ))
#
# Resumable: the CLI skips ids already present in --out. Re-submitting the array after a
# pre-emption continues rather than restarting. Output goes to a NEW directory so the original
# run is preserved for comparison; nothing is overwritten.
#
# Submit by tier. With the current five models, rows 0-35 are the two 3B models (hf provider)
# and rows 36-89 are the three 24-32B models (vllm, 80 GB GPU):
#   sbatch -p "$PART_ITER" --array=0-35  --export=ALL,IQE_PROVIDER=hf   cluster/22_qe_rerun_open.sbatch
#   sbatch -p "$PART_BIG"  --array=36-89 --export=ALL,IQE_PROVIDER=vllm cluster/22_qe_rerun_open.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh

mapfile -t MODELS < <(grep -vE '^\s*#|^\s*$' cluster/models_open.txt | awk '{print $1}')
PAIRS=( enhi enmr enta ente engu enml eten neen sien )   # nine pairs; enml restored
SHOTS_LIST=( 0 4 )
NP=${#PAIRS[@]}; NS=${#SHOTS_LIST[@]}; NM=${#MODELS[@]}
PER_MODEL=$(( NP * NS ))
NTASKS=$(( NM * PER_MODEL ))

# Fail loudly and early on an out-of-range array id. Without this the task dies on an unbound
# MODELS[$M] with no explanation, which is how rows 90-107 of the previous 0-107 array behaved.
if (( SLURM_ARRAY_TASK_ID >= NTASKS )); then
  echo "ERROR: array id $SLURM_ARRAY_TASK_ID >= $NTASKS ($NM models x $NP pairs x $NS shots)." >&2
  echo "       models_open.txt has $NM uncommented ids; valid range is 0-$(( NTASKS - 1 ))." >&2
  exit 2
fi

M=$(( SLURM_ARRAY_TASK_ID / PER_MODEL ))
REM=$(( SLURM_ARRAY_TASK_ID % PER_MODEL ))
P=$(( REM / NS )); S=$(( REM % NS ))
MODEL="${MODELS[$M]}"; PAIR="${PAIRS[$P]}"; SHOTS="${SHOTS_LIST[$S]}"
PROVIDER="${IQE_PROVIDER:-hf}"

# Do NOT export IQE_QE_MAXTOK. The default of 512 is the fix.
unset IQE_QE_MAXTOK || true

if [[ "$PROVIDER" == "vllm" ]]; then ENV="$CONDA_VLLM"; else ENV="$CONDA_IQE"; fi
activate "$ENV"

OUTDIR="$IQE_OUT/qe_open_rerun"
mkdir -p "$OUTDIR"
echo ">> RERUN $MODEL $PAIR shots=$SHOTS provider=$PROVIDER maxtok=default(512)"
python -m indicqe_ape.cli qe --provider "$PROVIDER" --model "$MODEL" \
  --pair "$PAIR" --subset challenge --split test --shots "$SHOTS" \
  --out "$OUTDIR/${MODEL##*/}.s${SHOTS}.${PAIR}.jsonl" \
  --cache-dir "$IQE_CACHE/rerun.${MODEL##*/}.s${SHOTS}.${PAIR}"
