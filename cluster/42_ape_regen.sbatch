#!/usr/bin/env bash
#SBATCH --job-name=iqe-ape-regen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
#
# Regenerate the APE cells for en-hi, en-mr and en-ta, and optionally add the X->en pairs.
#
# WHY THIS EXISTS
#   1. results/ape_open/sarvam-m.*.{enhi,enmr,enta} and sarvam-translate.mt.* are 300-byte stubs
#      holding only _meta. The runs happened -- the aggregates survive in
#      results/analysis/improve/APE_MATRA.md with n = 1490/2024/1375 -- but the per-instance rows
#      are gone, so those numbers cannot be re-scored, verified, or joined to the QE analysis.
#   2. tiny-aya-fire DOES have per-instance data, but on the OLD challenge test: 1490/2024/1375
#      against the current 1536/2098/1448. Regenerating only the sarvam cells would leave
#      tab:ape with two populations in one table, which is the mistake we spent this week undoing.
#   So all three systems are rerun on the current test, giving one population.
#
#   en-ml is NOT in here. It is already complete for all six systems on the current 1,436-row
#   population (results/enml-runs). en-te and en-gu carry no post-edits and cannot be run at all.
#
# TIERS
#   core  3 systems x {enhi, enmr, enta}  = 9 cells   the regeneration, array 0-8
#   xen   3 systems x {eten, neen, sien}  = 9 cells   new coverage, takes APE from 4 pairs to 7
#
# USAGE
#   sbatch -p "$PART_BIG" --array=0-8 --export=ALL,IQE_APE_TIER=core cluster/42_ape_regen.sbatch
#   sbatch -p "$PART_BIG" --array=0-8 --export=ALL,IQE_APE_TIER=xen  cluster/42_ape_regen.sbatch
#   IQE_APE_TIER=core cluster/42_ape_regen.sbatch --dry-run          # print the grid, run nothing
#
# An 80 GB GPU because sarvam-m is 24B; a 48 GB GPU (e.g. L40S) also fits.
#
# VALIDATE-C4-EXEMPT: engu, ente, enml -- en-gu and en-te carry zero post-edits in the challenge
# test so APE on them is impossible, not skipped; en-ml is already complete for all six systems on
# the current 1,436-row population (results/enml-runs) and must not be regenerated.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh

TIER="${IQE_APE_TIER:-core}"
case "$TIER" in
  core) PAIRS=(enhi enmr enta) ;;
  xen)  PAIRS=(eten neen sien) ;;
  *)    echo "ERROR: IQE_APE_TIER must be core or xen, got '$TIER'." >&2; exit 2 ;;
esac

# (model, mode, out-infix). retranslate is a dedicated MT system translating from scratch; the CLI
# writes hyp_pe either way so both are scored identically downstream.
MODELS=( CohereLabs/tiny-aya-fire sarvamai/sarvam-m sarvamai/sarvam-translate )
MODES=(   postedit                postedit          retranslate               )
INFIX=(   s0                      s0                mt                        )
PROVIDERS=( hf                    vllm              vllm                      )

NP=${#PAIRS[@]}; NM=${#MODELS[@]}; NTASKS=$(( NM * NP ))

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "tier=$TIER -> ${NTASKS} tasks, use --array=0-$(( NTASKS - 1 ))"
  for ((i=0;i<NTASKS;i++)); do
    m=$(( i / NP )); p=$(( i % NP ))
    printf "  %2d  %-28s %-11s %s\n" "$i" "${MODELS[$m]}" "${MODES[$m]}" "${PAIRS[$p]}"
  done
  exit 0
fi

: "${SLURM_ARRAY_TASK_ID:?must run as an array job; use --dry-run to see the grid}"
if (( SLURM_ARRAY_TASK_ID >= NTASKS )); then
  echo "ERROR: array id $SLURM_ARRAY_TASK_ID >= $NTASKS for tier=$TIER." >&2
  echo "       Valid range is 0-$(( NTASKS - 1 )). Re-run with --dry-run." >&2
  exit 2
fi

M=$(( SLURM_ARRAY_TASK_ID / NP )); P=$(( SLURM_ARRAY_TASK_ID % NP ))
MODEL="${MODELS[$M]}"; MODE="${MODES[$M]}"; TAG="${INFIX[$M]}"; PROVIDER="${PROVIDERS[$M]}"
PAIR="${PAIRS[$P]}"

# sarvam-m is a hybrid reasoning model: without this it emits a chain of thought before the
# post-edit, at ~28s/item against ~1s, and returns different text. Set IQE_THINK=1 to opt back in.
if [[ "$PROVIDER" == "vllm" && "${IQE_THINK:-0}" != "1" ]]; then
  export IQE_NO_THINK=1
fi
# APE generates a full sentence, so the budget must not be clipped. Leave unset for the code
# default of 512; an inherited 8 from the shell would truncate every post-edit.
unset IQE_QE_MAXTOK || true

if [[ "$PROVIDER" == "vllm" ]]; then ENV="$CONDA_VLLM"; else ENV="$CONDA_IQE"; fi
activate "$ENV"

OUTDIR="$IQE_OUT/ape_regen"
mkdir -p "$OUTDIR" logs
OUT="$OUTDIR/${MODEL##*/}.${TAG}.${PAIR}.jsonl"

echo ">> tier=$TIER task=$SLURM_ARRAY_TASK_ID/$NTASKS model=$MODEL mode=$MODE pair=$PAIR"
echo "   provider=$PROVIDER no_think=${IQE_NO_THINK:-0} out=$OUT node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Refuse to run on CPU. On shared partitions SLURM has handed tasks to nodes whose GPUs were
# already taken; the hf provider then falls back to float32 on CPU and runs for days.
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

# --no-comet: CometKiwi over the generated post-edits is a separate second pass (10_qe_metrics),
# and running it inside the array would load a second large model beside sarvam-m.
python -m indicqe_ape.cli ape \
  --provider "$PROVIDER" --model "$MODEL" --mode "$MODE" \
  --pair "$PAIR" --subset challenge --split test --shots 0 --no-comet \
  --out "$OUT" --cache-dir "$IQE_CACHE/ape_regen.${MODEL##*/}.${TAG}.${PAIR}"

echo "<< done $(date +%H:%M:%S)  rows=$(grep -vc '"_meta"\|"_summary"' "$OUT" || true)"
