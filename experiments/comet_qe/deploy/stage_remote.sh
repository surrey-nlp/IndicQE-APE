#!/usr/bin/env bash
# stage_remote.sh — put everything the pending GPU work needs onto a remote GPU workstation,
# WITHOUT starting it.
#
# Both GPUs are currently busy. This script only syncs code, data and the queue definition, then
# reports what is staged and what is running. Nothing is launched. Start the lanes yourself with
# deploy/launch_campaign.sh once a GPU frees up; the lanes claim work atomically from
# runs_status/ORDER, so they can be started while other work is still finishing.
#
# Run this from your local checkout, in the repo root:
#     NODE=user@host bash experiments/comet_qe/deploy/stage_remote.sh
# REMOTE (the experiments/comet_qe dir on the node) and REMOTE_PY (its interpreter) are optional.
#
# Idempotent. Safe to re-run: rsync only moves changed files and the ORDER file is rewritten
# from scratch each time, while runs_status/*.DONE markers are never touched.
set -euo pipefail

NODE="${NODE:?set NODE=user@host}"
REMOTE="${REMOTE:-\$HOME/indicqe-ape/experiments/comet_qe}"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "0. preflight — what is the node doing right now"
ssh "$NODE" 'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader; echo; echo "python processes:"; \
  ps -eo pid,etime,pcpu,rss,cmd --sort=-rss | grep -E "train\.py|run_zeroshot" | grep -v grep || echo "  none"'

say "1. code sync (no model blobs, no results, no checkpoints)"
rsync -az --stats \
  --exclude 'models/' --exclude 'results/' --exclude 'outputs/' --exclude 'logs/' \
  --exclude '__pycache__/' --exclude '.git/' --exclude 'data/*.csv' \
  "$LOCAL/" "$NODE:$REMOTE/"

say "2. data variants"
# enindic is REBUILT from the current HF snapshot: the frame the campaign used came from the
# stale dataset/release_build/hf_repo (five pairs, 8,160-row test). The previous file is kept
# on the node as _STALE_campaign so the published numbers remain reproducible.
rsync -az --stats \
  "$LOCAL/data/qe_indic_enindic.csv" \
  "$LOCAL/data/qe_indic_enindic.scaler.json" \
  "$LOCAL/data/qe_indic_enindic_STALE_campaign.csv" \
  "$LOCAL/data/qe_indic_enindic_STALE_campaign.scaler.json" \
  "$LOCAL/data/qe_indic_joint_enml.csv" \
  "$LOCAL/data/qe_indic_joint_enml.scaler.json" \
  "$LOCAL/data/qe_indic_xen.csv" \
  "$LOCAL/data/qe_indic_xen.scaler.json" \
  "$NODE:$REMOTE/data/"

say "3. verify the data landed intact (row counts must match the local copy)"
for f in enindic joint_enml xen; do
  local_n=$(( $(wc -l < "$LOCAL/data/qe_indic_$f.csv") - 1 ))
  remote_n=$(ssh "$NODE" "cd $REMOTE && expr \$(wc -l < data/qe_indic_$f.csv) - 1")
  if [ "$local_n" = "$remote_n" ]; then
    echo "  OK   qe_indic_$f.csv  $local_n rows"
  else
    echo "  FAIL qe_indic_$f.csv  local=$local_n node=$remote_n"; exit 1
  fi
done

say "4. APPEND new work to the live queue — DOES NOT START ANYTHING, DOES NOT DISTURB RUNNING LANES"
# The lanes are already draining runs_status/ORDER (the seed reruns are in flight). Overwriting
# ORDER would yank the list out from under a running `while read` loop, so new names are APPENDED
# and only if absent. A lane that has already passed the end of the file will not see them; the
# launch block below covers that by restarting a lane once the current one drains.
ssh "$NODE" "cd $REMOTE && mkdir -p runs_status logs && touch runs_status/ORDER && \
for n in qe-E1-infoxlm-s43 qe-E1-infoxlm-s44 qe-E2-A1-s43 qe-E2-A1-s44 \
         qe-E3-none-s43 qe-E3-none-s44 qe-E4-huber05-s43 qe-E4-huber05-s44 \
         qe-X-A0-xen qe-X-ref-xen qe-J-A0-joint qe-J-ref-joint; do
  grep -qxF \"\$n\" runs_status/ORDER || { echo \"\$n\" >> runs_status/ORDER; echo \"  queued \$n\"; }
done
echo; echo 'ORDER now has' \$(grep -vc '^#' runs_status/ORDER) 'entries;' \
     \$(ls runs_status/*.DONE 2>/dev/null | wc -l) 'DONE,' \
     \$(ls runs_status/*.RUNNING 2>/dev/null | wc -l) 'RUNNING'"

say "5. confirm every queued config exists on the node and parses"
ssh "$NODE" "cd $REMOTE && ${REMOTE_PY:-python3} - <<'PY'
import sys, yaml, pathlib
order = [l.strip() for l in open('runs_status/ORDER')
         if l.strip() and not l.startswith('#')]
bad = []
for name in order:
    p = pathlib.Path('configs/qe') / f'{name}.yaml'
    if not p.exists():
        bad.append((name, 'missing config')); continue
    c = yaml.safe_load(p.read_text())
    d = pathlib.Path(c['data_path'])
    if not d.exists():
        bad.append((name, f'missing data {d}')); continue
    print(f'  OK   {name:<22} seed={c.get(\"seed\")} data={c[\"data_path\"]}')
for name, why in bad:
    print(f'  FAIL {name}: {why}')
sys.exit(1 if bad else 0)
PY"

say "6. staged. NOTHING STARTED. Running lanes were not disturbed."
cat <<'EOF'
Both GPUs are currently busy with the seed reruns, which is the front of this same queue, so
there is nothing to launch right now. The appended runs are picked up automatically only if a
lane has not yet read past the end of ORDER. When the current lanes drain (watch for
runs_status/lane_*.DRAINED), restart them to sweep the remainder:

    ssh "$NODE"
    export IQE_REPO=/path/to/repo PY=/path/to/python
    cd "$IQE_REPO/experiments/comet_qe"
    nohup bash deploy/run_queue.sh 0 > logs/lane0.out 2>&1 &
    nohup bash deploy/run_queue.sh 1 > logs/lane1.out 2>&1 &

Restarting is always safe: .DONE runs are skipped and claims are atomic.

Zero-shot COMET against the en-hi MQM gold (~1 GPU-hour, the single highest-value run):

    export HF_TOKEN=$(cat ~/.cache/huggingface/token)
    CUDA_VISIBLE_DEVICES=0 $PY \
      scripts/run_zeroshot_comet.py --config mqm --split test \
      --models Unbabel/wmt22-cometkiwi-da Unbabel/wmt23-cometkiwi-da-xl Unbabel/XCOMET-XL \
      --gpus 1 --batch-size 16 --out-dir results/zeroshot

Progress at any time:

    ls runs_status/*.DONE | wc -l ; ls runs_status/*.RUNNING runs_status/*.FAILED 2>/dev/null
EOF
