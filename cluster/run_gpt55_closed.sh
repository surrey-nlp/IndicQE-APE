#!/usr/bin/env bash
# Comprehensive GPT-5.5 closed-source eval on the IndicQE-APE challenge set.
# Runs (all reasoning_effort=none unless noted; resumable; one failed cell won't stop the rest):
#   1) QE  0-shot DA+SQM, full challenge (8 pairs)
#   2) APE 0-shot, full challenge (PE pairs: en-hi/mr/ta), string metrics (COMET later on GPU)
#   3) QE  4-shot DA+SQM, subset 400/pair
#   4) APE 4-shot, subset 400/pair
#   5) Reasoning-effort study (DA only, subset): low + medium  (none = the cell-1 run)
#
# Launch detached:
#   source ~/.openai_key
#   tmux new -s gpt55 'cd $PWD && source cluster/env.sh && activate "$CONDA_IQE" && source ~/.openai_key && bash cluster/run_gpt55_closed.sh; bash'
set -uo pipefail                      # NOT -e: keep going if a cell errors (all runs resume)
cd "$(dirname "$0")/.."
source cluster/env.sh
: "${OPENAI_API_KEY:?source ~/.openai_key first}"
export HF_HUB_OFFLINE=1                # data is local; no model download for the API provider

MODEL="${GPT_MODEL:-gpt-5.5-2026-04-23}"
QE="$IQE_OUT/qe_closed"; APE="$IQE_OUT/ape_closed"; mkdir -p "$QE" "$APE"
PAIRS=(enhi enmr enta ente engu enml eten neen sien)
APE_PAIRS=(enhi enmr enta)            # only PE-bearing pairs
RPAIRS=(enhi enta)                    # reasoning-study subset: 1 Indo-Aryan + 1 Dravidian
run(){ echo ">>> $(date +%H:%M) $*"; "$@"; }

echo "######## 1/5  QE 0-shot (GEMBA-DA + SQM), full challenge ########"
for P in "${PAIRS[@]}"; do
  run python -m indicqe_ape.cli qe --provider openai --model "$MODEL" \
    --pair "$P" --subset challenge --shots 0 \
    --out "$QE/gpt-5.5.s0.$P.jsonl" --cache-dir "$IQE_CACHE"
done

echo "######## 2/5  APE 0-shot, full challenge (PE pairs) ########"
for P in "${APE_PAIRS[@]}"; do
  run python -m indicqe_ape.cli ape --provider openai --model "$MODEL" \
    --pair "$P" --subset challenge --shots 0 --no-comet \
    --out "$APE/gpt-5.5.s0.$P.jsonl" --cache-dir "$IQE_CACHE"
done

echo "######## 3/5  QE 4-shot (DA + SQM), subset 400/pair ########"
for P in "${PAIRS[@]}"; do
  run python -m indicqe_ape.cli qe --provider openai --model "$MODEL" \
    --pair "$P" --subset challenge --shots 4 --limit 400 \
    --out "$QE/gpt-5.5.s4.$P.jsonl" --cache-dir "$IQE_CACHE"
done

echo "######## 4/5  APE 4-shot, subset 400/pair (PE pairs) ########"
for P in "${APE_PAIRS[@]}"; do
  run python -m indicqe_ape.cli ape --provider openai --model "$MODEL" \
    --pair "$P" --subset challenge --shots 4 --limit 400 --no-comet \
    --out "$APE/gpt-5.5.s4.$P.jsonl" --cache-dir "$IQE_CACHE"
done

echo "######## 5/5  Reasoning-effort study (DA only, 400/pair x ${RPAIRS[*]}) ########"
echo "    none baseline = cell 1; here we add low + medium (separate caches so no stale hits)"
for EFF in low medium; do
  for P in "${RPAIRS[@]}"; do
    run env IQE_REASONING_EFFORT="$EFF" python -m indicqe_ape.cli qe --provider openai --model "$MODEL" \
      --pair "$P" --subset challenge --shots 0 --limit 400 --templates gemba-da \
      --out "$QE/gpt-5.5.eff-$EFF.s0.$P.jsonl" --cache-dir "$IQE_CACHE/eff-$EFF"
  done
  echo "    >>> ${EFF} effort done — glance at platform.openai.com/usage; medium bills reasoning tokens"
done

echo "######## ALL DONE ($(date)) ########"
for f in "$QE"/gpt-5.5.*.jsonl "$APE"/gpt-5.5.*.jsonl; do echo "== $(basename "$f") =="; tail -1 "$f"; done
