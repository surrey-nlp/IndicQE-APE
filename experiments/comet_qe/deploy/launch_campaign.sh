#!/usr/bin/env bash
# launch_campaign.sh — start the IndicQE-APE comet-QE campaign on a two-GPU workstation
# (e.g. 2x 24 GB A5000). Needs IQE_REPO (repository root) and, optionally, PY (interpreter with
# unbabel-comet and the campaign deps; default python3).
#
# GPU 1: training lane, starts immediately, pulls cheapest-first from runs_status/ORDER.
# GPU 0: runs the zero-shot COMET baselines first (fast, gives the Z table), then joins the
#       training lane. Both lanes share ORDER via atomic claim so neither idles.
#
# Resumable: re-running this clears stale RUNNING/STOPPED/claim markers but KEEPS .DONE,
# so finished runs are skipped and only unfinished work is retried.
set -u
export IQE_REPO="${IQE_REPO:?set IQE_REPO to the repository root}"
export PY="${PY:-python3}"
cd "$IQE_REPO/experiments/comet_qe" || exit 1
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)
S=runs_status; L=logs; mkdir -p "$S" "$L"

# Cheapest-first ORDER: distinct CORE hybrid runs (aliases collapsed). Minimal-feature and
# no-funnelling variants first; encoder swaps and full-feature reference later.
cat > "$S/ORDER" <<'EOF'
qe-A0-cometkiwi-min
qe-E3-none
qe-E4-mse
qe-E4-huber05
qe-E4-ccc
qe-E5-raw
qe-E5-minmax
qe-E2-A1
qe-E2-A2
qe-E2-A3
qe-E3-upper
qe-E3-buggy
qe-E1-xlmr
qe-E1-infoxlm
qe-E1-cometda
qe-A-cometkiwi-full
EOF

# clear stale in-flight markers (keep DONE) so this is a clean resume
find "$S" -name '*.RUNNING' -delete 2>/dev/null
find "$S" -name '*.STOPPED' -delete 2>/dev/null
find "$S" -name '*.claim' -type d -exec rmdir {} + 2>/dev/null
find "$S" -name 'lane_*' -delete 2>/dev/null

echo "campaign starting $(date -Is); status dir: $S"

# GPU 1 — training lane now
nohup bash deploy/run_queue.sh 1 > "$L/lane1.out" 2>&1 &
echo "  lane GPU 1 pid $!"

# GPU 0 — zero-shot first, then join training lane.
# CUDA_VISIBLE_DEVICES=0 pins every child (incl. the pytorch-lightning predict workers, which
# otherwise ignore --gpus and can squat on GPU 1 and starve the training lane) to physical GPU 0.
nohup bash -c '
  cd "$IQE_REPO/experiments/comet_qe"
  PY="${PY:-python3}"
  export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)
  export CUDA_VISIBLE_DEVICES=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  S=runs_status
  echo "gpu=0 zeroshot start=$(date -Is)" > "$S/zeroshot.RUNNING"
  if $PY -u scripts/run_zeroshot_comet.py --config qe-da --split test \
        --models Unbabel/wmt22-cometkiwi-da Unbabel/wmt23-cometkiwi-da-xl Unbabel/XCOMET-XL \
        --gpus 1 --batch-size 8 --out-dir results/zeroshot > logs/zeroshot.log 2>&1; then
     echo "DONE $(date -Is)" > "$S/zeroshot.DONE"
  else
     { echo "FAILED rc=$? $(date -Is)"; tail -20 logs/zeroshot.log; } > "$S/zeroshot.FAILED"
  fi
  rm -f "$S/zeroshot.RUNNING"
  bash deploy/run_queue.sh 0
' > "$L/lane0.out" 2>&1 &
echo "  lane GPU 0 (zeroshot -> training) pid $!"

echo "watch:  ls -t $PWD/runs_status/"
