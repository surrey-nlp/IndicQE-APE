#!/bin/bash
# Overnight experiment queue for a two-GPU workstation: one job at a time on GPU 1, leaving GPU 0
# free for other work (set CUDA_VISIBLE_DEVICES in manifest.txt as needed).
# Resumable (DONE jobs skipped), fail-safe (a failure is logged and the queue continues),
# survives disconnect (launch with: setsid bash run_queue.sh &). Everything is logged.
BASE="${IQE_WORK:?set IQE_WORK to the overnight working directory (holds code/ = this repo, data/, runs/)}"
cd "$BASE" || exit 1
RUNS="$BASE/runs"; MAN="$RUNS/manifest.txt"; QLOG="$RUNS/queue.log"
mkdir -p "$RUNS"
echo "======== queue start $(date) ========" >> "$QLOG"
while IFS='|' read -r id cmd; do
  [ -z "$id" ] && continue
  case "$id" in \#*) continue;; esac
  st="$RUNS/$id/status"
  if [ -f "$st" ] && grep -q DONE "$st"; then echo "skip(DONE) $id" >> "$QLOG"; continue; fi
  mkdir -p "$RUNS/$id"
  # wait until GPU1 is free (<2GB) in case a previous job is still releasing
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ')
    [ -n "$used" ] && [ "$used" -lt 2000 ] && break; sleep 10
  done
  echo "START $id $(date) (gpu1_used=${used}MiB)" >> "$QLOG"; echo "STARTED $(date)" > "$st"
  if bash -lc "$cmd" > "$RUNS/$id/log.txt" 2>&1; then echo "DONE $(date)" > "$st"; else echo "FAILED $(date)" > "$st"; fi
  echo "$(head -1 "$st") -> $id $(date)" >> "$QLOG"
done < "$MAN"
echo "======== queue end $(date) ========" >> "$QLOG"
