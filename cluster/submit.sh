#!/usr/bin/env bash
# Pre-flight + submit wrapper for a Slurm cluster.
# Runs the cheap checks that catch the common failures, then submits (adding
# --exclude="$FLAKY_NODE" when that is set in env.sh). The partition is site-specific, so
# pass it with -p (the scripts ship with their #SBATCH --partition line disabled). Usage:
#   cluster/submit.sh cluster/20_qe_prompt_open.sbatch -p "$PART_ITER"
#   cluster/submit.sh cluster/20_qe_prompt_open.sbatch -p "$PART_BIG" --gres=gpu:1
#   cluster/submit.sh cluster/10_qe_metrics.sbatch -p "$PART_BIG" --dry-run
set -euo pipefail
cd "$(dirname "$0")/.."; source cluster/env.sh

[[ $# -ge 1 ]] || { echo "usage: $0 <script.sbatch> [--dry-run] [sbatch args...]"; exit 2; }
SCRIPT="$1"; shift
DRY=0; EXTRA=()
while [[ $# -gt 0 ]]; do case "$1" in --dry-run) DRY=1; shift;; *) EXTRA+=("$1"); shift;; esac; done

[[ -f "$SCRIPT" ]] || { echo "FAIL: $SCRIPT not found"; exit 1; }

# resolve partition (CLI -p override else the script's #SBATCH line)
PART=""
for ((i=0;i<${#EXTRA[@]};i++)); do case "${EXTRA[$i]}" in
  -p) PART="${EXTRA[$((i+1))]:-}";; --partition=*) PART="${EXTRA[$i]#--partition=}";; esac; done
[[ -z "$PART" ]] && PART=$(awk -F= '/^#SBATCH[[:space:]]+--partition=/{print $2;exit}' "$SCRIPT" | awk '{print $1}')
[[ -z "$PART" ]] && { echo "FAIL: no partition; pass -p <partition> (see PART_* in cluster/env.sh)"; exit 1; }
command -v sinfo >/dev/null && { sinfo -h -p "$PART" -o '%P' 2>/dev/null | grep -q . \
  || { echo "FAIL: partition '$PART' not on cluster (sinfo)"; exit 1; }; }

# optional node exclusion (FLAKY_NODE in env.sh; empty = none)
EXCL=(); [[ -n "${FLAKY_NODE:-}" ]] && EXCL=(--exclude="$FLAKY_NODE")

# conda env spelled right (lowercase)
command -v conda >/dev/null && { conda env list | awk '{print $1}' | grep -qx "$CONDA_IQE" \
  || echo "warn: conda env '$CONDA_IQE' not in 'conda env list' — fix CONDA_IQE in env.sh"; }

# data snapshot present
[[ -d "$INDICQEAPE_DATA/data" ]] || echo "warn: $INDICQEAPE_DATA/data missing — run 00_setup.sh"

# dry-run the submission (catches bad gres/mem/time/partition)
command -v sbatch >/dev/null && { sbatch --test-only ${EXCL[@]+"${EXCL[@]}"} "${EXTRA[@]}" "$SCRIPT" \
  || { echo "FAIL: sbatch --test-only rejected"; exit 1; }; }

[[ "$DRY" == "1" ]] && { echo "dry-run OK"; exit 0; }
sbatch ${EXCL[@]+"${EXCL[@]}"} "${EXTRA[@]}" "$SCRIPT"
