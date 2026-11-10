#!/usr/bin/env bash
# Closed-weight (API) QE prompting rerun. Login node, no GPU, no SLURM.
#
# COMPLETED 2026-08-08: all nine pairs, zero parse failures, outputs copied into
# results/qe_closed_rerun/. tab:prompting and tab:parse4 now read them.
#
# Why rerun: the existing gpt-5.5 four-shot run used --limit 400 per pair, so it covers 3,200 of
# 12,730 DA-bearing challenge segments, and it omits en-ml because the original PAIRS array had
# eight pairs. The zero-shot run is complete. This script covers all nine pairs with no per-pair
# limit, at the default max_tokens of 512.
#
# COST CONTROL. This spends money, so it is staged and every stage is resumable. The CLI skips ids
# already written to --out, so re-running a stage continues rather than repeating. Set STAGE to
# choose how much to do:
#
#   STAGE=probe   40 items per pair, 9 pairs, one shot setting. ~360 calls. Confirms the parse rate
#                 recovers before spending. RUN THIS FIRST.
#   STAGE=enml    en-ml only, all items, both shot settings. ~2,872 calls. Closes the missing pair.
#   STAGE=full4   all nine pairs, four-shot, no limit. ~12,730 calls. Replaces the 400/pair sample.
#   STAGE=full0   all nine pairs, zero-shot, no limit. Only if you want zero-shot regenerated at
#                 512 for consistency; the existing zero-shot already parses at 1.000.
#
# Workload estimate, measured from the challenge set (avg source 115 chars, mt 112 chars,
# ~3.6 chars/token for mixed English and Indic script):
#   zero-shot ~183 input tokens/call  -> 12,730 calls ~ 2.33M input tokens
#   four-shot ~436 input tokens/call  -> 12,730 calls ~ 5.55M input tokens
#   output is short: parse rate is 1.000 for this model, so replies are a bare number, ~5 tokens.
# Apply your own per-token rate to those figures. Check spend at platform.openai.com/usage.
set -euo pipefail
cd "$(dirname "$0")/.."; source cluster/env.sh
export HF_HUB_OFFLINE=1
activate "$CONDA_IQE"

: "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
GPT_MODEL="${GPT_MODEL:-gpt-5.5-2026-04-23}"
STAGE="${STAGE:-probe}"
OUTDIR="$IQE_OUT/qe_closed_rerun"; mkdir -p "$OUTDIR"
ALL_PAIRS=( enhi enmr enta ente engu enml eten neen sien )

unset IQE_QE_MAXTOK || true          # default 512

# The output filename is gpt-5.5.s{shots}.{pair}.jsonl, NOT the dated model string. The analysis
# scripts glob for `gpt-5.5.s*.jsonl`, so a dated name would produce files nothing picks up.
# The dated model id is recorded in each file's _meta, which is where provenance belongs.
run_one () {  # pair, shots, limit_args...
  local PAIR="$1" SH="$2"; shift 2
  echo ">> $GPT_MODEL $PAIR shots=$SH $*"
  python -m indicqe_ape.cli qe --provider openai --model "$GPT_MODEL" \
    --pair "$PAIR" --subset challenge --split test --shots "$SH" "$@" \
    --out "$OUTDIR/gpt-5.5.s${SH}.${PAIR}.jsonl" \
    --cache-dir "$IQE_CACHE/rerun.closed.${PAIR}.s${SH}"
}

case "$STAGE" in
  probe)
    for PAIR in "${ALL_PAIRS[@]}"; do run_one "$PAIR" 4 --limit 40; done
    echo
    echo "Probe done. Check the parse rate before spending further:"
    echo "  python3 analysis/parse_rate_audit.py"
    echo "If four-shot parse rate is at or near 1.000, proceed to STAGE=enml then STAGE=full4."
    ;;
  enml)
    for SH in 0 4; do run_one enml "$SH"; done
    ;;
  full4)
    for PAIR in "${ALL_PAIRS[@]}"; do run_one "$PAIR" 4; done
    ;;
  full0)
    for PAIR in "${ALL_PAIRS[@]}"; do run_one "$PAIR" 0; done
    ;;
  *)
    echo "unknown STAGE: $STAGE (probe|enml|full4|full0)" >&2; exit 2 ;;
esac
echo "STAGE=$STAGE complete. Outputs in $OUTDIR"
