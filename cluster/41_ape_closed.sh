#!/usr/bin/env bash
# Closed-weight APE (API; login node). 2 providers x 2 shots x 3 PE-pairs.
set -euo pipefail
cd "$(dirname "$0")/.."; source cluster/env.sh
export HF_HUB_OFFLINE=1
activate "$CONDA_IQE"
GPT_MODEL="${GPT_MODEL:-gpt-4.1}"; GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-pro}"
for spec in "google:$GEMINI_MODEL" "openai:$GPT_MODEL"; do
  PROV="${spec%%:*}"; MODEL="${spec#*:}"
  for SH in 0 4; do
    for PAIR in enhi enmr enta; do
      python -m indicqe_ape.cli ape --provider "$PROV" --model "$MODEL" \
        --pair "$PAIR" --subset challenge --split test --shots "$SH" --no-comet \
        --out "$IQE_OUT/ape_closed/${MODEL//\//-}.s${SH}.${PAIR}.jsonl" --cache-dir "$IQE_CACHE" || true
    done
  done
done
