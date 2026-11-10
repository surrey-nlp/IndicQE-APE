#!/usr/bin/env bash
#SBATCH --job-name=iqe-rerun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%A_%a.out
#
# A4: prompted-QE rerun at the DEFAULT 512-token decoding budget, all nine pairs, tier-filtered.
#
# Supersedes 22_qe_rerun_open.sbatch, which reads models_open.txt. That list is the v1 sweep and is
# wrong for this purpose: it names Qwen3.6-27B and gemma-4-31B (both failed to load, reported
# nowhere) and omits Llama-3.2-3B and aya-expanse-32b (both in the paper). This reads
# models_rerun.txt and filters by tier, so the array size is derived from the tier rather than
# hardcoded, which is what caused the 0-107-for-5-models fault previously.
#
# WHY TIERS. VRAM is the binding constraint, so the tier picks the partition:
#   IQE_TIER=small   3 models, ~7 GB each   -> any GPU with >= 11 GB. 54 tasks.
#   IQE_TIER=large   1 model, ~47 GB        -> a 48 GB GPU (e.g. L40S, RTX A6000). 18 tasks.
#   IQE_TIER=xl      1 model, ~64 GB bf16   -> an 80 GB GPU (A100), or 48 GB with 4-bit. 18 tasks.
#
# Task index within a tier: row = model_index*18 + pair_index*2 + shot_index
# Both GEMBA templates run per item in one call, so one task is one pass over one pair.
#
# ALWAYS check the array size before submitting; the script prints it:
#   IQE_TIER=small bash cluster/24_qe_rerun_tiered.sbatch --dry-run
#
# Submit (partition and array both come from the dry-run output):
#   IQE_TIER=small sbatch -p <small-gpu-partition> --array=0-53 --export=ALL,IQE_TIER=small,IQE_PROVIDER=hf   cluster/24_qe_rerun_tiered.sbatch
#   IQE_TIER=large sbatch -p <48GB-partition>      --array=0-17 --export=ALL,IQE_TIER=large,IQE_PROVIDER=vllm cluster/24_qe_rerun_tiered.sbatch
#
# Resumable: the CLI skips ids already present in --out, so a pre-empted task continues. Output
# goes to a NEW directory; the original 8-token run is preserved for the before/after comparison.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source cluster/env.sh

TIER="${IQE_TIER:?set IQE_TIER to small, large or xl}"
LIST="${IQE_MODEL_LIST:-cluster/models_rerun.txt}"
PAIRS=( enhi enmr enta ente engu enml eten neen sien )   # all nine
SHOTS_LIST=( 0 4 )

mapfile -t MODELS < <(awk -v t="$TIER" '$0 !~ /^[[:space:]]*#/ && NF>=2 && $2==t {print $1}' "$LIST")
NM=${#MODELS[@]}
NP=${#PAIRS[@]}; NS=${#SHOTS_LIST[@]}
PER_MODEL=$(( NP * NS ))
NTASKS=$(( NM * PER_MODEL ))

if (( NM == 0 )); then
  echo "ERROR: no models with tier='$TIER' in $LIST" >&2; exit 2
fi

if [[ "${1:-}" == "--dry-run" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "tier=$TIER  models=$NM  pairs=$NP  shots=$NS"
  printf '  %s\n' "${MODELS[@]}"
  echo "TASKS=$NTASKS   ->  submit with --array=0-$(( NTASKS - 1 ))"
  [[ "${1:-}" == "--dry-run" ]] && exit 0
  echo "ERROR: no SLURM_ARRAY_TASK_ID; this must be run under sbatch --array" >&2; exit 2
fi

if (( SLURM_ARRAY_TASK_ID >= NTASKS )); then
  echo "ERROR: array id $SLURM_ARRAY_TASK_ID >= $NTASKS for tier=$TIER." >&2
  echo "       Valid range is 0-$(( NTASKS - 1 )). Re-run with --dry-run to get it." >&2
  exit 2
fi

M=$(( SLURM_ARRAY_TASK_ID / PER_MODEL ))
REM=$(( SLURM_ARRAY_TASK_ID % PER_MODEL ))
P=$(( REM / NS )); S=$(( REM % NS ))
MODEL="${MODELS[$M]}"; PAIR="${PAIRS[$P]}"; SHOTS="${SHOTS_LIST[$S]}"
PROVIDER="${IQE_PROVIDER:-hf}"

# The whole point of the rerun: leave this unset so the code default of 512 applies. The original
# runs exported 8, which truncated any model that emitted a word before the score.
unset IQE_QE_MAXTOK || true

# Hybrid reasoning models (sarvam-m) emit a chain of thought before the score unless told not to,
# which costs ~28s/item instead of ~1s and produces a DIFFERENT score, not just a slower one. A
# run without this took 11h26m for eight of eighteen cells before it was killed. Default it on for
# the vLLM path; set IQE_THINK=1 to opt back in, and expect the cost.
if [[ "$PROVIDER" == "vllm" && "${IQE_THINK:-0}" != "1" ]]; then
  export IQE_NO_THINK=1
fi

if [[ "$PROVIDER" == "vllm" ]]; then ENV="$CONDA_VLLM"; else ENV="$CONDA_IQE"; fi
activate "$ENV"

OUTDIR="$IQE_OUT/qe_rerun512"
mkdir -p "$OUTDIR" logs
OUT="$OUTDIR/${MODEL##*/}.s${SHOTS}.${PAIR}.jsonl"

echo ">> tier=$TIER task=$SLURM_ARRAY_TASK_ID/$NTASKS model=$MODEL pair=$PAIR shots=$SHOTS"
echo "   provider=$PROVIDER maxtok=default(512) node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Fail fast if this task did not actually get a GPU. On shared/preemptible partitions SLURM has
# handed tasks to nodes whose GPUs were already taken, and the HF provider then falls back to
# `cpu dtype=float32` and runs for days instead of minutes. Four tasks did exactly that. A CPU
# run is never what we want here, so refuse rather than waste the allocation.
python - <<'PYCHK' || { echo "ERROR: no usable CUDA device; refusing to run on CPU." >&2; exit 3; }
import sys
try:
    import torch
    ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if ok:
        free, total = torch.cuda.mem_get_info()
        print(f"   CUDA ok: {torch.cuda.get_device_name(0)} "
              f"{free/2**30:.1f} GiB free of {total/2**30:.1f} GiB")
    sys.exit(0 if ok else 1)
except Exception as e:
    print(f"   CUDA probe failed: {e}")
    sys.exit(1)
PYCHK

python -u -m indicqe_ape.cli qe --provider "$PROVIDER" --model "$MODEL" \
  --pair "$PAIR" --subset challenge --split test --shots "$SHOTS" \
  --out "$OUT" \
  --cache-dir "$IQE_CACHE/rerun512.${MODEL##*/}.s${SHOTS}.${PAIR}"

echo ">> done: $OUT"
