#!/usr/bin/env bash
# run_queue.sh <gpu_id> — one worker lane pinned to a GPU.
#
# Both lanes pull from the SAME cheapest-first list (runs_status/ORDER) using an atomic
# mkdir claim, so neither GPU idles and no run is done twice. Idempotent + resumable:
#   runs_status/<name>.DONE     -> skip (already finished; survives restarts)
#   runs_status/<name>.RUNNING  -> in flight (host/gpu/pid/time)
#   runs_status/<name>.FAILED   -> non-zero exit + last 20 log lines
#   runs_status/<name>.STOPPED  -> lane was killed mid-run — EXPLICIT PICK-UP SIGNAL
# On SIGTERM/SIGINT the in-flight run is marked .STOPPED (not lost). Re-running the lane
# (or launch_campaign.sh) retries everything that is not .DONE.
set -u
cd "${IQE_REPO:?set IQE_REPO to the repository root}/experiments/comet_qe" || exit 1
# env python directly (not `conda run`): live unbuffered logs + a killable single-process tree.
PY="${PY:-python3}"
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)
export CUDA_VISIBLE_DEVICES="$1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation on the 24 GB A5000
S=runs_status; L=logs; mkdir -p "$S" "$L"

current=""
on_stop() {
  if [ -n "$current" ]; then
    { echo "STOPPED gpu=$CUDA_VISIBLE_DEVICES pid=$$ $(date -Is)";
      echo "reason: lane received signal — this run did not finish, pick it up";
      tail -20 "$L/$current.log" 2>/dev/null; } > "$S/$current.STOPPED"
    rm -f "$S/$current.RUNNING"
  fi
  echo "lane gpu=$CUDA_VISIBLE_DEVICES stopped $(date -Is)" > "$S/lane_$CUDA_VISIBLE_DEVICES.STOPPED"
  exit 143
}
trap on_stop TERM INT

while IFS= read -r name; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  [ -f "$S/$name.DONE" ] && continue
  if ! mkdir "$S/$name.claim" 2>/dev/null; then continue; fi   # another lane owns it
  current="$name"
  rm -f "$S/$name.STOPPED" "$S/$name.FAILED"
  echo "gpu=$CUDA_VISIBLE_DEVICES pid=$$ start=$(date -Is)" > "$S/$name.RUNNING"
  if $PY -u src/train.py --config "configs/qe/$name.yaml" > "$L/$name.log" 2>&1; then
    { echo "DONE gpu=$CUDA_VISIBLE_DEVICES end=$(date -Is)";
      grep -Eo '"(dev|test)_(pearson|spearman|rmse)": [-0-9.]+' "outputs/$name/run_summary.json" 2>/dev/null; \
    } > "$S/$name.DONE"
  else
    rc=$?
    { echo "FAILED gpu=$CUDA_VISIBLE_DEVICES rc=$rc end=$(date -Is)"; tail -20 "$L/$name.log"; } > "$S/$name.FAILED"
  fi
  rm -f "$S/$name.RUNNING"; current=""
done < "$S/ORDER"

echo "lane gpu=$CUDA_VISIBLE_DEVICES drained $(date -Is)" > "$S/lane_$CUDA_VISIBLE_DEVICES.DRAINED"
