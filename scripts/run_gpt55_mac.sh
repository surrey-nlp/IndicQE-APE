#!/usr/bin/env bash
# Staged GPT-5.5 closed-source eval — run ONE setting at a time for spend control.
# Each stage: prints a cost estimate, waits 5s (Ctrl-C to abort), runs pair-by-pair, resumable.
#
#   bash scripts/run_gpt55_mac.sh qe0            # zero-shot QE (DA+SQM), all pairs, full challenge
#   bash scripts/run_gpt55_mac.sh qe4            # 4-shot QE (DA+SQM); LIMIT caps items/pair (default 400)
#   bash scripts/run_gpt55_mac.sh ape  [pair..]  # APE 0-shot (string metrics; COMET later on GPU)
#   bash scripts/run_gpt55_mac.sh ape4 [pair..]  # APE 4-shot
#
# Pairs are optional — give one or more to run just those (e.g. `ape enhi`, then `ape enmr`).
# Env: LIMIT (items/pair), IQE_OUT, IQE_CACHE, GPT_MODEL, IQE_REASONING_EFFORT (default none).
# Tip: wrap the long qe0 run in `caffeinate -i` so the Mac doesn't sleep.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
: "${OPENAI_API_KEY:?source ~/.openai_key first}"
export INDICQEAPE_DATA="${INDICQEAPE_DATA:-$REPO/dataset/release_build/hf_repo}"
export IQE_OUT="${IQE_OUT:-$REPO/results}"
export IQE_CACHE="${IQE_CACHE:-$REPO/results/llmcache}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
[ -f "$INDICQEAPE_DATA/data/challenge/test.parquet" ] || { echo "FAIL: no data at $INDICQEAPE_DATA/data — set INDICQEAPE_DATA"; exit 1; }

MODEL="${GPT_MODEL:-gpt-5.5-2026-04-23}"
QE="$IQE_OUT/qe_closed"; APE="$IQE_OUT/ape_closed"; mkdir -p "$QE" "$APE"
ALL=(enhi enmr enta ente engu eten neen sien); PE=(enhi enmr enta)

STAGE="${1:-}"; shift 2>/dev/null || true; PAIRS_IN=("$@")
defp(){ if [ "${#PAIRS_IN[@]}" -gt 0 ]; then P=("${PAIRS_IN[@]}"); else P=("$@"); fi; }  # sets global P (bash 3.2-safe)

estimate(){ # task shots pair...
  python - "$@" <<'PY'
import sys, os
from indicqe_ape.data import DataLoader
task, shots = sys.argv[1], int(sys.argv[2]); pairs = sys.argv[3:]
lim = int(os.environ.get("LIMIT", "0")) or None
M = {"enhi":"en-hi","enmr":"en-mr","enta":"en-ta","ente":"en-te","engu":"en-gu","eten":"et-en","neen":"ne-en","sien":"si-en"}
ch = DataLoader().load("challenge")
per = 2 if task == "qe" else 1                      # GEMBA-DA + SQM for QE
in_tok = 220 + (shots * 180 if shots else 0)
out_tok = 12 if task == "qe" else 120
cpc = in_tok * 5 / 1e6 + out_tok * 30 / 1e6
calls = 0
for p in pairs:
    s = ch[ch.language_pair == M[p]]
    if task == "ape": s = s[s["has_pe"] == True]
    n = len(s) if lim is None else min(len(s), lim)
    calls += n * per
print(f"  estimate: ~{calls:,} API calls, ~${calls*cpc:,.2f}  (rough; reasoning_effort=none)")
PY
}
go(){ echo "  starting in 5s — Ctrl-C to abort"; sleep 5; }
# Re-invoke a (resumable) cli call until it exits 0 — survives internet drops, laptop sleep,
# rate limits. Each retry resumes from the JSONL + cache, so nothing is lost or re-charged.
runretry(){
  local max="${IQE_MAX_RETRIES:-90}" n=0      # 90 x 120s ≈ 3h of patience; raise for longer
  until "$@"; do
    n=$((n+1)); [ "$n" -ge "$max" ] && { echo "  !! gave up after $n attempts: $*"; return 1; }
    echo "  …call failed (internet down / asleep / rate-limited?) — resume $n/$max in 120s ($(date +%H:%M))"
    sleep 120
  done
}

case "$STAGE" in
  qe0)
    defp "${ALL[@]}"
    echo "STAGE qe0 — zero-shot QE (GEMBA-DA + SQM), full challenge: ${P[*]}"
    estimate qe 0 "${P[@]}"; go
    for x in "${P[@]}"; do echo ">>> $(date +%H:%M) qe0 $x"
      runretry python -m indicqe_ape.cli qe --provider openai --model "$MODEL" \
        --pair "$x" --subset challenge --shots 0 \
        --out "$QE/gpt-5.5.s0.$x.jsonl" --cache-dir "$IQE_CACHE"
    done ;;
  qe4)
    defp "${ALL[@]}"; L="${LIMIT:-400}"
    echo "STAGE qe4 — 4-shot QE (GEMBA-DA + SQM), LIMIT=$L/pair: ${P[*]}"
    LIMIT="$L" estimate qe 4 "${P[@]}"; go
    for x in "${P[@]}"; do echo ">>> $(date +%H:%M) qe4 $x"
      runretry python -m indicqe_ape.cli qe --provider openai --model "$MODEL" \
        --pair "$x" --subset challenge --shots 4 --limit "$L" \
        --out "$QE/gpt-5.5.s4.$x.jsonl" --cache-dir "$IQE_CACHE"
    done ;;
  ape)
    defp "${PE[@]}"
    echo "STAGE ape — 0-shot APE (string metrics; COMET later on GPU): ${P[*]}"
    estimate ape 0 "${P[@]}"; go
    for x in "${P[@]}"; do echo ">>> $(date +%H:%M) ape $x"
      runretry python -m indicqe_ape.cli ape --provider openai --model "$MODEL" \
        --pair "$x" --subset challenge --shots 0 --no-comet \
        --out "$APE/gpt-5.5.s0.$x.jsonl" --cache-dir "$IQE_CACHE"
    done ;;
  ape4)
    defp "${PE[@]}"; L="${LIMIT:-400}"
    echo "STAGE ape4 — 4-shot APE, LIMIT=$L/pair: ${P[*]}"
    LIMIT="$L" estimate ape 4 "${P[@]}"; go
    for x in "${P[@]}"; do echo ">>> $(date +%H:%M) ape4 $x"
      runretry python -m indicqe_ape.cli ape --provider openai --model "$MODEL" \
        --pair "$x" --subset challenge --shots 4 --limit "$L" --no-comet \
        --out "$APE/gpt-5.5.s4.$x.jsonl" --cache-dir "$IQE_CACHE"
    done ;;
  reason)
    defp enhi enta; L="${LIMIT:-400}"
    echo "STAGE reason — reasoning-effort study (GPT-5.5, DA only), LIMIT=$L/pair: ${P[*]}"
    echo "  none baseline = your qe0 run; this adds low + medium with separate caches (medium bills reasoning tokens)"
    LIMIT="$L" estimate qe 0 "${P[@]}"; echo "  (x2 for low+medium; medium costs more)"; go
    for EFF in low medium; do
      for x in "${P[@]}"; do echo ">>> $(date +%H:%M) reason-$EFF $x"
        runretry env IQE_REASONING_EFFORT="$EFF" python -m indicqe_ape.cli qe --provider openai --model "$MODEL" \
          --pair "$x" --subset challenge --shots 0 --limit "$L" --templates gemba-da \
          --out "$QE/gpt-5.5.eff-$EFF.s0.$x.jsonl" --cache-dir "$IQE_CACHE/eff-$EFF"
      done
      echo "    >>> $EFF effort done — glance at platform.openai.com/usage"
    done ;;
  *)
    echo "usage: bash scripts/run_gpt55_mac.sh {qe0|qe4|ape|ape4|reason} [pair ...]"
    echo "  qe0    zero-shot QE, all pairs           (~\$33)"
    echo "  qe4    4-shot QE, LIMIT=400/pair          (~\$32; set LIMIT= to change)"
    echo "  ape    APE 0-shot, PE pairs               (~\$23 all; or one pair: ape enhi)"
    echo "  ape4   APE 4-shot, LIMIT=400/pair         (~\$10)"
    echo "  reason reasoning none/low/medium study    (~\$25; en-hi+en-ta, DA only)"
    exit 2 ;;
esac
echo "STAGE $STAGE done ($(date +%H:%M)). Outputs in $IQE_OUT/{qe_closed,ape_closed}. Check spend: platform.openai.com/usage"
