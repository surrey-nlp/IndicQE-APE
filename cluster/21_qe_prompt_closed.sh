#!/usr/bin/env bash
# Closed-weight QE prompting (API; run on the login node — no GPU, no SLURM).
# 2 providers x 2 shots x 8 pairs, stratified subsample per pair for cost.
# Needs GOOGLE_API_KEY / OPENAI_API_KEY. Set GPT_MODEL to the current flagship.
set -euo pipefail
cd "$(dirname "$0")/.."; source cluster/env.sh
export HF_HUB_OFFLINE=1            # data is local; no model download for API providers
activate "$CONDA_IQE"

GPT_MODEL="${GPT_MODEL:-gpt-4.1}"; GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-pro}"
LIMIT="${LIMIT:-500}"
for spec in "google:$GEMINI_MODEL" "openai:$GPT_MODEL"; do
  PROV="${spec%%:*}"; MODEL="${spec#*:}"
  for SH in 0 4; do
    for PAIR in enhi enmr enta ente engu enml eten neen sien; do
      python -m indicqe_ape.cli qe --provider "$PROV" --model "$MODEL" \
        --pair "$PAIR" --subset challenge --split test --shots "$SH" --limit "$LIMIT" \
        --out "$IQE_OUT/qe_closed/${MODEL//\//-}.s${SH}.${PAIR}.jsonl" --cache-dir "$IQE_CACHE" || true
    done
  done
done
