#!/usr/bin/env bash
# 60_a3_enml_fewshot.sh — A3: the missing four-shot en-ml cells, on a single-GPU workstation.
#
# HISTORICAL (fixed): every four-shot prompted run used to cover eight pairs and omit en-ml,
# because the v1
# launchers' PAIRS array did. This fills that gap for the models that FIT ON A 24 GB GPU, which
# is the 3B tier only. The 24-32B tier (sarvam-m 23.6B, aya-expanse-32b) needs 48 GB and belongs on
# the Slurm cluster; gpt-5.5 is covered by the API stage (23_qe_rerun_closed.sh STAGE=enml).
#
# Runs at the DEFAULT decoding budget of 512 tokens. The eight-pair four-shot cells on disk were
# produced at 8 tokens, which truncated any model that emitted a word before the score. So these
# en-ml cells are NOT directly comparable to the existing four-shot cells until those are rerun
# (A4). Output goes to a separate directory to keep that distinction on disk.
#
# Waits for GPU 0 to be free rather than fighting a running job. Resumable: the CLI skips ids
# already in --out.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python3}"
GPU="${GPU:-0}"
OUTDIR="${OUTDIR:-results/qe_enml_fewshot}"
LIMIT_ARG="${LIMIT:+--limit $LIMIT}"
mkdir -p "$OUTDIR" logs cache

export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset IQE_QE_MAXTOK || true          # default 512 is the whole point

# 3B tier only: these fit in 24 GB. Do not add the 24-32B models here.
MODELS=(
  meta-llama/Llama-3.2-3B-Instruct
  CohereLabs/tiny-aya-fire
  CoRover/BharatGPT-3B-Indic
)

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "0. wait for GPU $GPU to be free"
for i in $(seq 1 240); do
  used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$used" -lt 2000 ]; then echo "  GPU $GPU free (${used} MiB)"; break; fi
  [ "$i" = 1 ] && echo "  GPU $GPU busy (${used} MiB), waiting..."
  sleep 30
done

say "1. pre-download weights (compute and generation both need them present)"
for m in "${MODELS[@]}"; do
  if $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$m', allow_patterns=['*.json','*.safetensors','*.model','*.txt'])
print('  cached $m')
" 2>&1 | tail -1; then :; else echo "  FAILED to fetch $m"; fi
done

say "2. run en-ml four-shot, both GEMBA templates per item"
for m in "${MODELS[@]}"; do
  base="${m##*/}"
  out="$OUTDIR/${base}.s4.enml.jsonl"
  echo ">> $base  en-ml  shots=4  maxtok=default(512)"
  start=$(date +%s)
  $PY -u -m indicqe_ape.cli qe --provider hf --model "$m" \
      --pair enml --subset challenge --split test --shots 4 $LIMIT_ARG \
      --out "$out" --cache-dir "cache/a3.${base}.s4.enml" \
      2>&1 | tail -3
  echo "   elapsed $(( $(date +%s) - start ))s"
done

say "3. parse rates for what just ran"
$PY - <<'PY'
import json, glob, os
for f in sorted(glob.glob("results/qe_enml_fewshot/*.jsonl")):
    tot, parsed = 0, {}
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        if "_meta" in r or "_summary" in r:
            continue
        tot += 1
        for t, v in (r.get("pred") or {}).items():
            parsed[t] = parsed.get(t, 0) + (v is not None)
    rates = ", ".join(f"{t} {c/max(tot,1):.3f}" for t, c in sorted(parsed.items()))
    print(f"  {os.path.basename(f):<44} n={tot:<5} {rates}")
PY

cat <<'EOF'

A3 done. NOT comparable to the existing eight-pair four-shot cells until A4 reruns those at 512.
Copy results off, then run analysis/parse_rate_audit.py locally.
EOF
